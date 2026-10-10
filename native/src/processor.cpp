#include "wemm/processor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include "wemm/common.h"
#include "wemm/media.h"

namespace fs = std::filesystem;

namespace wemm {

namespace {

// qwen-vl-utils constants
constexpr double kMaxRatio = 200;
constexpr int64_t kImageMinTokens = 4;
constexpr int64_t kImageMaxTokens = 16384;
constexpr int64_t kVideoMinTokens = 128;
constexpr int64_t kVideoMaxTokens = 768;
constexpr double kFps = 2.0;
constexpr int64_t kFrameFactor = 2;
constexpr int64_t kFpsMinFrames = 4;
constexpr int64_t kFpsMaxFrames = 768;

const char* kImagePad = "<|image_pad|>";
const char* kVideoPad = "<|video_pad|>";
const char* kVisionStart = "<|vision_start|>";
const char* kVisionEnd = "<|vision_end|>";
const char* kPlaceholder = "<|placeholder|>";

// Python's round() (half to even) on a double.
inline double py_round(double x) { return std::nearbyint(x); }

int64_t round_by_factor(double number, int64_t factor) { return int64_t(py_round(number / factor)) * factor; }
int64_t ceil_by_factor(double number, int64_t factor) { return int64_t(std::ceil(number / factor)) * factor; }
int64_t floor_by_factor(double number, int64_t factor) { return int64_t(std::floor(number / factor)) * factor; }

double model_seq_len() {
  if (const char* env = std::getenv("MODEL_SEQ_LEN")) return double(int64_t(std::atof(env)));
  return 128000;
}

// Python str.strip() whitespace set, applied on UTF-8.
bool is_py_space(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}

std::string py_strip(const std::string& s) {
  // Decode to code points with byte offsets.
  std::vector<std::pair<uint32_t, size_t>> cps;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    uint32_t cp = n == 1 ? c : c & (0x7F >> n);
    for (size_t k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    cps.emplace_back(cp, i);
    i += n;
  }
  size_t b = 0, e = cps.size();
  while (b < e && is_py_space(cps[b].first)) ++b;
  while (e > b && is_py_space(cps[e - 1].first)) --e;
  if (b == e) return "";
  size_t start = cps[b].second;
  size_t end = e < cps.size() ? cps[e].second : s.size();
  return s.substr(start, end - start);
}

bool replace_first(std::string& s, const std::string& from, const std::string& to) {
  auto pos = s.find(from);
  if (pos == std::string::npos) return false;
  s.replace(pos, from.size(), to);
  return true;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (true) {
    auto pos = s.find(from, i);
    if (pos == std::string::npos) break;
    out.append(s, i, pos - i);
    out += to;
    i = pos + from.size();
  }
  out.append(s, i, std::string::npos);
  s.swap(out);
}

std::string repeat(const char* s, int64_t n) {
  std::string out;
  size_t len = std::strlen(s);
  out.reserve(len * size_t(n));
  for (int64_t i = 0; i < n; ++i) out.append(s, len);
  return out;
}

std::string render_content(const std::vector<ContentItem>& content) {
  std::string out;
  for (const auto& item : content) {
    switch (item.type) {
      case ContentItem::Type::Image:
        out += "<|vision_start|><|image_pad|><|vision_end|>";
        break;
      case ContentItem::Type::Video:
        out += "<|vision_start|><|video_pad|><|vision_end|>";
        break;
      case ContentItem::Type::Text:
        out += item.text;
        break;
    }
  }
  return out;
}

}  // namespace

// ------------------------------------------------------------ resizing ------
std::pair<int64_t, int64_t> qwen_smart_resize(int64_t height, int64_t width, int64_t factor, double min_pixels,
                                              double max_pixels) {
  WEMM_CHECK(max_pixels >= min_pixels, "max_pixels must be >= min_pixels");
  WEMM_CHECK(double(std::max(height, width)) / double(std::min(height, width)) <= kMaxRatio,
             "absolute aspect ratio must be smaller than ", kMaxRatio);
  int64_t h_bar = std::max(factor, round_by_factor(double(height), factor));
  int64_t w_bar = std::max(factor, round_by_factor(double(width), factor));
  if (double(h_bar * w_bar) > max_pixels) {
    double beta = std::sqrt(double(height * width) / max_pixels);
    h_bar = floor_by_factor(height / beta, factor);
    w_bar = floor_by_factor(width / beta, factor);
  } else if (double(h_bar * w_bar) < min_pixels) {
    double beta = std::sqrt(min_pixels / double(height * width));
    h_bar = ceil_by_factor(height * beta, factor);
    w_bar = ceil_by_factor(width * beta, factor);
  }
  return {h_bar, w_bar};
}

std::pair<int64_t, int64_t> hf_image_smart_resize(int64_t height, int64_t width, int64_t factor,
                                                  int64_t min_pixels, int64_t max_pixels) {
  WEMM_CHECK(double(std::max(height, width)) / double(std::min(height, width)) <= 200,
             "absolute aspect ratio must be smaller than 200");
  int64_t h_bar = int64_t(py_round(double(height) / factor)) * factor;
  int64_t w_bar = int64_t(py_round(double(width) / factor)) * factor;
  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt(double(height * width) / double(max_pixels));
    h_bar = std::max(factor, int64_t(std::floor(height / beta / factor)) * factor);
    w_bar = std::max(factor, int64_t(std::floor(width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(double(min_pixels) / double(height * width));
    h_bar = int64_t(std::ceil(height * beta / factor)) * factor;
    w_bar = int64_t(std::ceil(width * beta / factor)) * factor;
  }
  return {h_bar, w_bar};
}

std::pair<int64_t, int64_t> hf_video_smart_resize(int64_t num_frames, int64_t height, int64_t width,
                                                  int64_t temporal_factor, int64_t factor, int64_t min_pixels,
                                                  int64_t max_pixels) {
  WEMM_CHECK(height >= factor && width >= factor, "video frame smaller than factor");
  int64_t h_bar = int64_t(py_round(double(height) / factor)) * factor;
  int64_t w_bar = int64_t(py_round(double(width) / factor)) * factor;
  int64_t t_bar = int64_t(std::ceil(double(num_frames) / temporal_factor)) * temporal_factor;
  if (t_bar * h_bar * w_bar > max_pixels) {
    double beta = std::sqrt(double(num_frames * height * width) / double(max_pixels));
    h_bar = std::max(factor, int64_t(std::floor(height / beta / factor)) * factor);
    w_bar = std::max(factor, int64_t(std::floor(width / beta / factor)) * factor);
  } else if (t_bar * h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(double(min_pixels) / double(num_frames * height * width));
    h_bar = int64_t(std::ceil(height * beta / factor)) * factor;
    w_bar = int64_t(std::ceil(width * beta / factor)) * factor;
  }
  return {h_bar, w_bar};
}

// ------------------------------------------------------------ processor -----
Processor::Processor(const std::string& model_dir)
    : mcfg_(ModelConfig::load(model_dir)), pcfg_(ProcessorConfig::load(model_dir)) {
  tok_ = std::make_unique<Tokenizer>((fs::path(model_dir) / "tokenizer.json").string());
}

std::string Processor::render_chat(const Conversation& conv) {
  WEMM_CHECK(!conv.empty(), "No messages provided.");
  std::string out;
  if (conv[0].role == "system") {
    out += "<|im_start|>system\n" + py_strip(render_content(conv[0].content)) + "<|im_end|>\n";
  }
  // Index of the last real user query.
  int64_t last_query = -1;
  for (int64_t i = int64_t(conv.size()) - 1; i >= 0; --i) {
    if (conv[i].role != "user") continue;
    std::string c = py_strip(render_content(conv[i].content));
    bool tool_resp = c.rfind("<tool_response>", 0) == 0 && c.size() >= 16 &&
                     c.compare(c.size() - 16, 16, "</tool_response>") == 0;
    if (!tool_resp) {
      last_query = i;
      break;
    }
  }
  WEMM_CHECK(last_query >= 0, "No user query found in messages.");
  for (size_t i = 0; i < conv.size(); ++i) {
    const auto& m = conv[i];
    std::string content = py_strip(render_content(m.content));
    if (m.role == "system") {
      WEMM_CHECK(i == 0, "System message must be at the beginning.");
    } else if (m.role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      std::string reasoning;
      auto think_end = content.find("</think>");
      if (think_end != std::string::npos) {
        std::string before = content.substr(0, think_end);
        while (!before.empty() && before.back() == '\n') before.pop_back();
        auto ts = before.rfind("<think>");
        reasoning = ts == std::string::npos ? before : before.substr(ts + 7);
        size_t k = 0;
        while (k < reasoning.size() && reasoning[k] == '\n') ++k;
        reasoning = reasoning.substr(k);
        std::string after = content.substr(content.rfind("</think>") + 8);
        k = 0;
        while (k < after.size() && after[k] == '\n') ++k;
        content = after.substr(k);
      }
      reasoning = py_strip(reasoning);
      if (int64_t(i) > last_query)
        out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
      else
        out += "<|im_start|>assistant\n" + content;
      out += "<|im_end|>\n";
    } else if (m.role == "tool") {
      if (i > 0 && conv[i - 1].role != "tool") out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == conv.size() || conv[i + 1].role != "tool") out += "<|im_end|>\n";
    } else {
      throw Error("Unexpected message role: " + m.role);
    }
  }
  return out;
}

Visual Processor::load_image(const ContentItem& item) const {
  at::Tensor img = item.image.defined() ? item.image : wemm::load_image(item.source);
  WEMM_CHECK(img.dim() == 3 && img.size(2) == 3 && img.scalar_type() == at::kByte, "image must be [H,W,3] uint8");
  const int64_t factor = int64_t(pcfg_.patch_size) * pcfg_.merge_size;
  const auto& o = item.image_options;

  // qwen-vl-utils fetch_image: smart_resize + PIL bicubic resize.
  std::pair<int64_t, int64_t> hw;
  if (o.resized_height && o.resized_width) {
    hw = qwen_smart_resize(*o.resized_height, *o.resized_width, factor, double(kImageMinTokens * factor * factor),
                           double(kImageMaxTokens * factor * factor));
  } else {
    double min_p = double(o.min_pixels.value_or(kImageMinTokens * factor * factor));
    double max_p = double(o.max_pixels.value_or(kImageMaxTokens * factor * factor));
    hw = qwen_smart_resize(img.size(0), img.size(1), factor, min_p, max_p);
  }
  img = pil_resize_bicubic(img, hw.first, hw.second);

  // Qwen2VLImageProcessorFast: second smart_resize with the processor's limits,
  // torchvision uint8 bicubic (antialias) when the size changes.
  auto hw2 = hf_image_smart_resize(img.size(0), img.size(1), factor, pcfg_.image_min_pixels, pcfg_.image_max_pixels);
  img = torch_resize_uint8(img, hw2.first, hw2.second);

  Visual v;
  v.frames = img.unsqueeze(0);
  v.grid_t = 1;
  v.grid_h = hw2.first / pcfg_.patch_size;
  v.grid_w = hw2.second / pcfg_.patch_size;
  return v;
}

Visual Processor::load_video(const ContentItem& item) const {
  const auto& o = item.video_options;
  VideoInfo info = probe_video(item.source);

  // calculate_video_frame_range
  int64_t start_frame = 0, end_frame = info.total_frames - 1, total = info.total_frames;
  if (o.video_start || o.video_end) {
    double max_dur = double(info.total_frames) / info.fps;
    if (o.video_start) start_frame = int64_t(std::ceil(std::max(0.0, std::min(*o.video_start, max_dur)) * info.fps));
    if (o.video_end) {
      end_frame = int64_t(std::floor(std::max(0.0, std::min(*o.video_end, max_dur)) * info.fps));
      end_frame = std::min(end_frame, info.total_frames - 1);
    }
    WEMM_CHECK(start_frame < end_frame, "invalid video time range");
    total = end_frame - start_frame + 1;
  }

  // smart_nframes
  int64_t nframes;
  if (o.nframes) {
    WEMM_CHECK(!o.fps, "Only accept either `fps` or `nframes`");
    nframes = round_by_factor(double(*o.nframes), kFrameFactor);
  } else {
    double fps = o.fps.value_or(kFps);
    double min_frames = double(ceil_by_factor(double(o.min_frames.value_or(kFpsMinFrames)), kFrameFactor));
    double max_frames =
        double(floor_by_factor(double(o.max_frames.value_or(std::min(kFpsMaxFrames, total))), kFrameFactor));
    double n = double(total) / info.fps * fps;
    n = std::min(std::min(std::max(n, min_frames), max_frames), double(total));
    nframes = floor_by_factor(n, kFrameFactor);
  }
  WEMM_CHECK(kFrameFactor <= nframes && nframes <= total, "nframes should be in [", kFrameFactor, ", ", total,
             "], got ", nframes);

  // idx = torch.linspace(start, end, nframes).round()
  std::vector<int64_t> idx(static_cast<size_t>(nframes));
  for (int64_t i = 0; i < nframes; ++i) {
    double v = nframes == 1 ? double(start_frame)
                            : double(start_frame) + double(end_frame - start_frame) * double(i) / double(nframes - 1);
    idx[size_t(i)] = int64_t(py_round(v));
  }
  DecodedVideo dec = decode_video_frames(item.source, idx);

  // fetch_video resizing
  const int64_t factor = int64_t(pcfg_.patch_size) * pcfg_.merge_size;
  const int64_t H = dec.frames.size(1), W = dec.frames.size(2);
  double min_pixels = double(o.min_pixels.value_or(kVideoMinTokens * factor * factor));
  double total_pixels = o.total_pixels.value_or(model_seq_len() * double(factor * factor) * 0.9);
  double max_pixels = std::max(std::min(double(kVideoMaxTokens * factor * factor),
                                        total_pixels / double(nframes) * double(kFrameFactor)),
                               double(int64_t(min_pixels * 1.05)));
  if (o.max_pixels) max_pixels = std::min(double(*o.max_pixels), max_pixels);
  std::pair<int64_t, int64_t> hw;
  if (o.resized_height && o.resized_width)
    hw = qwen_smart_resize(*o.resized_height, *o.resized_width, factor, double(kImageMinTokens * factor * factor),
                           double(kImageMaxTokens * factor * factor));
  else
    hw = qwen_smart_resize(H, W, factor, min_pixels, max_pixels);
  at::Tensor frames = torch_resize_uint8_via_float(dec.frames, hw.first, hw.second);

  // Qwen3VLVideoProcessor: second smart_resize, float bicubic if the size changes.
  auto hw2 = hf_video_smart_resize(nframes, hw.first, hw.second, pcfg_.temporal_patch_size, factor,
                                   pcfg_.video_min_pixels, pcfg_.video_max_pixels);
  if (hw2 != hw) {
    at::Tensor x = frames.permute({0, 3, 1, 2}).to(at::kFloat);
    x = at::_upsample_bicubic2d_aa(x, {hw2.first, hw2.second}, false);
    frames = x.permute({0, 2, 3, 1}).contiguous();
  }

  Visual v;
  v.is_video = true;
  v.frames = frames;
  v.grid_t = (nframes + pcfg_.temporal_patch_size - 1) / pcfg_.temporal_patch_size;
  v.grid_h = hw2.first / pcfg_.patch_size;
  v.grid_w = hw2.second / pcfg_.patch_size;
  v.fps = info.fps;
  v.frame_indices = idx;
  return v;
}

ProcessedSample Processor::process(const Conversation& conv) const {
  ProcessedSample s;
  std::string text = render_chat(conv);

  for (const auto& m : conv)
    for (const auto& item : m.content) {
      if (item.type == ContentItem::Type::Image) s.visuals.push_back(load_image(item));
      if (item.type == ContentItem::Type::Video) s.visuals.push_back(load_video(item));
    }

  const int64_t merge_len = int64_t(pcfg_.merge_size) * pcfg_.merge_size;
  // Images (Qwen3VLProcessor: all images first, then all videos).
  for (const auto& v : s.visuals) {
    if (v.is_video) continue;
    int64_t n = v.grid_t * v.grid_h * v.grid_w / merge_len;
    WEMM_CHECK(replace_first(text, kImagePad, repeat(kPlaceholder, n)), "more images than <|image_pad|> tokens");
  }
  replace_all(text, kPlaceholder, kImagePad);
  for (const auto& v : s.visuals) {
    if (!v.is_video) continue;
    std::vector<int64_t> ind = v.frame_indices;
    const int64_t tp = pcfg_.temporal_patch_size;
    if (ind.size() % size_t(tp) != 0) ind.resize(ind.size() + size_t(tp) - ind.size() % size_t(tp), ind.back());
    std::string placeholder;
    const int64_t frame_seqlen = v.grid_h * v.grid_w / merge_len;
    const std::string frame_tokens = std::string(kVisionStart) + repeat(kPlaceholder, frame_seqlen) + kVisionEnd;
    for (int64_t t = 0; t < v.grid_t; ++t) {
      double a = double(ind[size_t(t * tp)]) / v.fps;
      double b = double(ind[size_t(t * tp + tp - 1)]) / v.fps;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "<%.1f seconds>", (a + b) / 2);
      placeholder += buf;
      placeholder += frame_tokens;
    }
    const std::string wrapped = std::string(kVisionStart) + kVideoPad + kVisionEnd;
    if (!replace_first(text, wrapped, placeholder))
      WEMM_CHECK(replace_first(text, kVideoPad, placeholder), "more videos than <|video_pad|> tokens");
  }
  replace_all(text, kPlaceholder, kVideoPad);

  s.input_ids = tok_->encode(text, /*add_special_tokens=*/true);
  s.prompt = std::move(text);
  s.position_ids = mrope_position_ids(s.input_ids, s.visuals, mcfg_);
  return s;
}

std::vector<ProcessedSample> Processor::process_batch(const std::vector<Conversation>& convs, int threads) const {
  std::vector<ProcessedSample> out(convs.size());
  if (threads <= 0) threads = int(std::min<size_t>(convs.size(), std::max(1u, std::thread::hardware_concurrency() / 2)));
  threads = std::max(1, std::min<int>(threads, int(convs.size())));
  if (threads == 1) {
    for (size_t i = 0; i < convs.size(); ++i) out[i] = process(convs[i]);
    return out;
  }
  std::atomic<size_t> next{0};
  std::vector<std::exception_ptr> errors(convs.size());
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (size_t i; (i = next++) < convs.size();) {
        try {
          out[i] = process(convs[i]);
        } catch (...) {
          errors[i] = std::current_exception();
        }
      }
    });
  for (auto& th : pool) th.join();
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
  return out;
}

at::Tensor mrope_position_ids(const std::vector<int32_t>& ids, const std::vector<Visual>& visuals,
                              const ModelConfig& cfg) {
  const int64_t L = int64_t(ids.size());
  const int64_t m = cfg.vision.spatial_merge_size;
  at::Tensor pos = at::empty({3, L}, at::kLong);
  int64_t* pt = pos.data_ptr<int64_t>();
  int64_t* ph = pt + L;
  int64_t* pw = ph + L;

  // Per-frame grids (videos are split into frames: Qwen3.5 uses timestamps).
  std::vector<std::pair<int64_t, int64_t>> image_grids, video_grids;
  for (const auto& v : visuals) {
    auto& dst = v.is_video ? video_grids : image_grids;
    for (int64_t t = 0; t < (v.is_video ? v.grid_t : 1); ++t) dst.emplace_back(v.grid_h / m, v.grid_w / m);
    WEMM_CHECK(v.is_video || v.grid_t == 1, "image with grid_t != 1");
  }
  size_t ii = 0, vi = 0;
  int64_t p = 0;
  for (int64_t i = 0; i < L;) {
    const int32_t tok = ids[size_t(i)];
    if (tok == cfg.image_token_id || tok == cfg.video_token_id) {
      bool img = tok == cfg.image_token_id;
      auto& grids = img ? image_grids : video_grids;
      size_t& gi = img ? ii : vi;
      WEMM_CHECK(gi < grids.size(), "more vision placeholders than visual inputs");
      auto [gh, gw] = grids[gi++];
      WEMM_CHECK(i + gh * gw <= L, "truncated vision placeholder run");
      for (int64_t r = 0; r < gh; ++r)
        for (int64_t c = 0; c < gw; ++c) {
          int64_t k = i + r * gw + c;
          WEMM_CHECK(ids[size_t(k)] == tok, "vision placeholder run shorter than its grid");
          pt[k] = p;
          ph[k] = p + r;
          pw[k] = p + c;
        }
      p += std::max(gh, gw);
      i += gh * gw;
    } else {
      pt[i] = ph[i] = pw[i] = p++;
      ++i;
    }
  }
  WEMM_CHECK(ii == image_grids.size() && vi == video_grids.size(), "unused visual inputs");
  return pos;
}

}  // namespace wemm
