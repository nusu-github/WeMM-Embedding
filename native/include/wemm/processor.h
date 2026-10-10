#pragma once

#include <ATen/ATen.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "wemm/config.h"
#include "wemm/tokenizer.h"

namespace wemm {

// Per-item overrides with qwen-vl-utils semantics (all optional).
struct ImageOptions {
  std::optional<int64_t> min_pixels, max_pixels;
  std::optional<int64_t> resized_height, resized_width;
};

struct VideoOptions {
  std::optional<double> fps;         // sampling fps (default 2.0)
  std::optional<int64_t> nframes;    // fixed frame count (mutually exclusive with fps)
  std::optional<int64_t> min_frames, max_frames;
  std::optional<double> video_start, video_end;  // seconds
  std::optional<int64_t> min_pixels, max_pixels;
  std::optional<double> total_pixels;
  std::optional<int64_t> resized_height, resized_width;
};

struct ContentItem {
  enum class Type { Text, Image, Video };
  Type type = Type::Text;
  std::string text;    // Type::Text
  std::string source;  // Type::Image / Type::Video: file path
  at::Tensor image;    // Type::Image alternative to `source`: decoded RGB [H,W,3] uint8
  ImageOptions image_options;
  VideoOptions video_options;

  static ContentItem Text(std::string t) {
    ContentItem c;
    c.type = Type::Text;
    c.text = std::move(t);
    return c;
  }
  static ContentItem Image(std::string path, ImageOptions o = {}) {
    ContentItem c;
    c.type = Type::Image;
    c.source = std::move(path);
    c.image_options = o;
    return c;
  }
  static ContentItem ImageTensor(at::Tensor rgb, ImageOptions o = {}) {
    ContentItem c;
    c.type = Type::Image;
    c.image = std::move(rgb);
    c.image_options = o;
    return c;
  }
  static ContentItem Video(std::string path, VideoOptions o = {}) {
    ContentItem c;
    c.type = Type::Video;
    c.source = std::move(path);
    c.video_options = o;
    return c;
  }
};

struct Message {
  std::string role = "user";
  std::vector<ContentItem> content;
};
using Conversation = std::vector<Message>;

// Final-resolution pixels of one image or video, before normalization.
struct Visual {
  at::Tensor frames;  // [T, H, W, 3], uint8 (or float32 in rare re-resize cases), CPU
  int64_t grid_t = 1, grid_h = 0, grid_w = 0;  // in patches (H/patch, W/patch)
  bool is_video = false;
  // Video metadata (for the prompt timestamps).
  double fps = 0;
  std::vector<int64_t> frame_indices;
};

struct ProcessedSample {
  std::string prompt;  // fully expanded prompt text
  std::vector<int32_t> input_ids;
  std::vector<Visual> visuals;  // in prompt order (images and videos interleaved)
  at::Tensor position_ids;      // [3, L] int64, mrope positions
};

class Processor {
 public:
  explicit Processor(const std::string& model_dir);

  // Default chat template (chat_template.jinja) with add_generation_prompt=False.
  static std::string render_chat(const Conversation& conv);

  // Loads/decodes/resizes media and tokenizes one conversation.
  ProcessedSample process(const Conversation& conv) const;
  // Processes conversations in parallel on a thread pool.
  std::vector<ProcessedSample> process_batch(const std::vector<Conversation>& convs, int threads = 0) const;

  Visual load_image(const ContentItem& item) const;
  Visual load_video(const ContentItem& item) const;

  const ModelConfig& model_config() const { return mcfg_; }
  const ProcessorConfig& config() const { return pcfg_; }
  const Tokenizer& tokenizer() const { return *tok_; }

 private:
  ModelConfig mcfg_;
  ProcessorConfig pcfg_;
  std::unique_ptr<Tokenizer> tok_;
};

// Helpers exposed for testing.
std::pair<int64_t, int64_t> qwen_smart_resize(int64_t h, int64_t w, int64_t factor, double min_pixels,
                                              double max_pixels);
std::pair<int64_t, int64_t> hf_image_smart_resize(int64_t h, int64_t w, int64_t factor, int64_t min_pixels,
                                                  int64_t max_pixels);
std::pair<int64_t, int64_t> hf_video_smart_resize(int64_t t, int64_t h, int64_t w, int64_t temporal_factor,
                                                  int64_t factor, int64_t min_pixels, int64_t max_pixels);
at::Tensor mrope_position_ids(const std::vector<int32_t>& ids, const std::vector<Visual>& visuals,
                              const ModelConfig& cfg);

}  // namespace wemm
