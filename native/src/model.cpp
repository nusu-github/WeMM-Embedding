// Weight loading, per-batch preludes and dispatch to the fused / reference
// implementations (model_fused.cpp, model_reference.cpp).

#include "wemm/model.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdlib>

#include "model_internal.h"
#include "wemm/common.h"
#include "wemm/pixels.h"
#include "wemm/safetensors.h"

namespace wemm {

using namespace detail;

namespace {

// Fails early on configurations the kernels do not implement.
void validate_supported(const ModelConfig& cfg) {
  const auto& t = cfg.text;
  const auto& v = cfg.vision;
  WEMM_CHECK(t.linear_key_head_dim == 128 && t.linear_value_head_dim == 128,
             "gated delta rule head dims must be 128");
  WEMM_CHECK(t.linear_num_value_heads % t.linear_num_key_heads == 0, "value heads must be a multiple of key heads");
  WEMM_CHECK(t.linear_conv_kernel_dim == 4, "only conv kernel size 4 is supported");
  WEMM_CHECK(t.head_dim == 128 || t.head_dim == 256, "attention head_dim must be 128 or 256");
  WEMM_CHECK(t.num_attention_heads % t.num_key_value_heads == 0, "invalid GQA configuration");
  WEMM_CHECK(t.attn_output_gate, "attention without output gate is not supported");
  WEMM_CHECK(t.hidden_size % 8 == 0 && v.hidden_size % 8 == 0, "hidden sizes must be multiples of 8");
  WEMM_CHECK(v.hidden_size % v.num_heads == 0 && (v.hidden_size / v.num_heads) % 2 == 0, "invalid ViT heads");
  for (const auto& lt : t.layer_types)
    WEMM_CHECK(lt == "linear_attention" || lt == "full_attention", "unknown layer type ", lt);
}

}  // namespace

// ================================================================ loading ====
WeMMModel::WeMMModel(const std::string& model_dir, ModelOptions opts)
    : cfg_(ModelConfig::load(model_dir)),
      pcfg_(ProcessorConfig::load(model_dir)),
      opts_(opts),
      device_(at::kCUDA, int8_t(opts.device_index)) {
  validate_supported(cfg_);
  WEMM_CHECK(opts.device_index >= 0 && opts.device_index < int(at::cuda::device_count()), "invalid CUDA device ",
             opts.device_index);
  if (opts.expandable_segments && !std::getenv("PYTORCH_CUDA_ALLOC_CONF"))
    c10::CachingAllocator::setAllocatorSettings("expandable_segments:True");

  c10::InferenceMode guard;
  c10::cuda::CUDAGuard dev_guard(device_);
  const auto* prop = at::cuda::getDeviceProperties(opts.device_index);
  const bool ampere = prop->major >= 8;
  WEMM_CHECK(prop->major * 10 + prop->minor >= 75, "requires a Turing (sm_75) or newer GPU");
  act_ = opts.compute_dtype.value_or(at::kHalf);
  WEMM_CHECK(act_ == at::kBFloat16 || act_ == at::kHalf, "compute_dtype must be bf16 or fp16");
  WEMM_CHECK(ampere || act_ == at::kHalf, "bf16 needs sm_80+; use fp16 on this GPU");
  resid_ = opts.fp32_residual ? at::kFloat : act_;
  attn_ = opts.attention == AttentionBackend::Auto ? (ampere ? AttentionBackend::Flash : AttentionBackend::MemEfficient)
                                                   : opts.attention;
  WEMM_CHECK(attn_ != AttentionBackend::Flash || ampere, "FlashAttention needs sm_80+");
  WEMM_CHECK(attn_ != AttentionBackend::MemEfficient || ampere || act_ == at::kHalf,
             "memory-efficient attention on sm_75 needs fp16");

  load_weights(model_dir);
  // Loading leaves converted/concatenated temporaries in the caching allocator.
  c10::cuda::CUDACachingAllocator::emptyCache();
}

void WeMMModel::load_weights(const std::string& model_dir) {
  auto st = SafeTensors::open(model_dir);
  // Fused weights are assembled on the host and uploaded once (no transient
  // device copies); dtype conversion happens on the device.
  auto up = [&](const at::Tensor& cpu, at::ScalarType dt) {
    at::Tensor t = cpu.contiguous().to(device_).to(dt).contiguous();
    // fp16 has a narrower range than the bf16 checkpoint.
    if (dt == at::kHalf) WEMM_CHECK(!t.isinf().any().item<bool>(), "a weight overflows fp16; use compute_dtype=bf16");
    return t;
  };
  auto act = [&](const std::string& n) { return up(st.get(n), act_); };
  auto f32 = [&](const std::string& n) { return up(st.get(n), at::kFloat); };
  // GEMM weight matrices (optionally compressed)
  auto mat_host = [&](const at::Tensor& host) { return Weight::load(host, act_, device_, opts_.compress_weights); };
  auto mat = [&](const std::string& n) { return mat_host(st.get(n)); };
  auto mat_cat = [&](std::vector<at::Tensor> parts, int64_t pad_rows_to = 1) {
    at::Tensor t = at::cat(parts, 0);
    const int64_t rows = t.size(0), padded = (rows + pad_rows_to - 1) / pad_rows_to * pad_rows_to;
    if (padded != rows) t = at::cat({t, at::zeros({padded - rows, t.size(1)}, t.options())}, 0);
    return mat_host(t);
  };

  // ---------------------------------------------------------------- vision
  const auto& vc = cfg_.vision;
  const std::string vp = "model.visual.";
  patch_w = mat_host(st.get(vp + "patch_embed.proj.weight").reshape({vc.hidden_size, -1}));
  patch_b = f32(vp + "patch_embed.proj.bias");
  pos_embed = act(vp + "pos_embed.weight");
  for (int i = 0; i < vc.depth; ++i) {
    const std::string p = vp + "blocks." + std::to_string(i) + ".";
    VisionBlock b;
    b.norm1_w = f32(p + "norm1.weight");
    b.norm1_b = f32(p + "norm1.bias");
    b.norm2_w = f32(p + "norm2.weight");
    b.norm2_b = f32(p + "norm2.bias");
    b.qkv_w = mat(p + "attn.qkv.weight");
    b.qkv_b = act(p + "attn.qkv.bias");
    b.proj_w = mat(p + "attn.proj.weight");
    b.proj_b = up(st.get(p + "attn.proj.bias"), resid_);
    b.fc1_w = mat(p + "mlp.linear_fc1.weight");
    b.fc1_b = act(p + "mlp.linear_fc1.bias");
    b.fc2_w = mat(p + "mlp.linear_fc2.weight");
    b.fc2_b = up(st.get(p + "mlp.linear_fc2.bias"), resid_);
    vblocks_.push_back(std::move(b));
  }
  merger_norm_w = f32(vp + "merger.norm.weight");
  merger_norm_b = f32(vp + "merger.norm.bias");
  merger_fc1_w = mat(vp + "merger.linear_fc1.weight");
  merger_fc1_b = act(vp + "merger.linear_fc1.bias");
  merger_fc2_w = mat(vp + "merger.linear_fc2.weight");
  merger_fc2_b = act(vp + "merger.linear_fc2.bias");
  {
    const int64_t dim = vc.hidden_size / vc.num_heads / 2;  // Qwen3_5VisionRotaryEmbedding(head_dim // 2)
    vision_inv_freq = (1.0 / at::pow(10000.0, at::arange(0, dim, 2, at::kFloat) / double(dim))).to(device_);
  }

  // ------------------------------------------------------------------ text
  const auto& tc = cfg_.text;
  const std::string tp = "model.language_model.";
  // The table keeps its mapping alive through the tensor's deleter.
  embed_tokens = opts_.embeddings_on_gpu ? act(tp + "embed_tokens.weight") : st.get(tp + "embed_tokens.weight");
  for (int i = 0; i < tc.num_hidden_layers; ++i) {
    const std::string p = tp + "layers." + std::to_string(i) + ".";
    TextLayer l;
    l.linear = tc.layer_types[size_t(i)] == "linear_attention";
    l.in_norm = 1.0 + f32(p + "input_layernorm.weight");
    l.post_norm = 1.0 + f32(p + "post_attention_layernorm.weight");
    if (l.linear) {
      const std::string a = p + "linear_attn.";
      // One GEMM for [qkv | z | b | a], padded to 128 outputs (faster tiles).
      l.in_proj = mat_cat({st.get(a + "in_proj_qkv.weight"), st.get(a + "in_proj_z.weight"),
                           st.get(a + "in_proj_b.weight"), st.get(a + "in_proj_a.weight")},
                          128);
      l.conv_w = f32(a + "conv1d.weight").squeeze(1).contiguous();
      l.neg_A = -f32(a + "A_log").exp();
      l.dt_bias = f32(a + "dt_bias");
      l.gnorm_w = f32(a + "norm.weight");
      l.out_proj = mat(a + "out_proj.weight");
    } else {
      const std::string a = p + "self_attn.";
      l.qkv_proj = mat_cat({st.get(a + "q_proj.weight"), st.get(a + "k_proj.weight"), st.get(a + "v_proj.weight")});
      l.q_norm = 1.0 + f32(a + "q_norm.weight");
      l.k_norm = 1.0 + f32(a + "k_norm.weight");
      l.o_proj = mat(a + "o_proj.weight");
    }
    l.gate_up = mat_cat({st.get(p + "mlp.gate_proj.weight"), st.get(p + "mlp.up_proj.weight")});
    l.down = mat(p + "mlp.down_proj.weight");
    layers_.push_back(std::move(l));
  }
  final_norm = 1.0 + f32(tp + "norm.weight");
  {
    const int64_t dim = int64_t(tc.head_dim * tc.partial_rotary_factor);
    text_inv_freq = (1.0 / at::pow(tc.rope_theta, at::arange(0, dim, 2, at::kLong).to(at::kFloat) / double(dim)))
                        .to(device_);
  }
}

// ============================================================ attention ======
at::Tensor WeMMModel::attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu_q,
                                const at::Tensor& cu_k, int64_t max_q, int64_t max_k, bool causal) const {
  // q: [Tq, H, D], k/v: [Tk, Hkv, D] packed sequences.
  const double scale = 1.0 / std::sqrt(double(q.size(2)));
  if (attn_ == AttentionBackend::Flash)
    return std::get<0>(at::_flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, false, scale));
  // Memory-efficient attention: no native GQA, so repeat the KV heads.
  at::Tensor kk = k, vv = v;
  if (k.size(1) != q.size(1)) {
    const int64_t rep = q.size(1) / k.size(1);
    kk = k.repeat_interleave(rep, 1);
    vv = v.repeat_interleave(rep, 1);
  }
  auto res = at::_efficient_attention_forward(q.unsqueeze(0), kk.contiguous().unsqueeze(0),
                                              vv.contiguous().unsqueeze(0), std::nullopt, cu_q, cu_k, max_q, max_k,
                                              0.0, /*custom_mask_type=*/causal ? 1 : 0, false, scale);
  return std::get<0>(res).squeeze(0);
}

// =============================================================== preludes ====
WeMMModel::VisionPrelude WeMMModel::make_vision_prelude(const std::vector<std::array<int64_t, 3>>& grids) const {
  const auto& vc = cfg_.vision;
  const int64_t m = vc.spatial_merge_size, D = vc.hidden_size;
  const int64_t side = int64_t(std::lround(std::sqrt(double(vc.num_position_embeddings))));
  int64_t P = 0;
  for (const auto& g : grids) P += g[0] * g[1] * g[2];

  // Position-embedding interpolation (fast_pos_embed_interpolate) and 2D rotary
  // indices, in the merge-window patch order.
  std::vector<int64_t> idx[4];
  std::vector<float> wts[4];
  std::vector<float> rows, cols;
  for (int i = 0; i < 4; ++i) {
    idx[i].reserve(size_t(P));
    wts[i].reserve(size_t(P));
  }
  rows.reserve(size_t(P));
  cols.reserve(size_t(P));
  std::vector<int32_t> cu{0};
  VisionPrelude vp;
  for (const auto& g : grids) {
    const int64_t t = g[0], h = g[1], w = g[2];
    at::Tensor hl = at::linspace(0, double(side - 1), h, at::kFloat);
    at::Tensor wl = at::linspace(0, double(side - 1), w, at::kFloat);
    const float* hp = hl.data_ptr<float>();
    const float* wp = wl.data_ptr<float>();
    for (int64_t ti = 0; ti < t; ++ti)
      for (int64_t br = 0; br < h / m; ++br)
        for (int64_t bc = 0; bc < w / m; ++bc)
          for (int64_t ir = 0; ir < m; ++ir)
            for (int64_t ic = 0; ic < m; ++ic) {
              const int64_t r = br * m + ir, c = bc * m + ic;
              const int64_t hf = int64_t(hp[r]), wf = int64_t(wp[c]);
              const int64_t hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
              const float dh = hp[r] - float(hf), dw = wp[c] - float(wf);
              idx[0].push_back(hf * side + wf);
              idx[1].push_back(hf * side + wc);
              idx[2].push_back(hc * side + wf);
              idx[3].push_back(hc * side + wc);
              wts[0].push_back((1 - dh) * (1 - dw));
              wts[1].push_back((1 - dh) * dw);
              wts[2].push_back(dh * (1 - dw));
              wts[3].push_back(dh * dw);
              rows.push_back(float(r));
              cols.push_back(float(c));
            }
    for (int64_t ti = 0; ti < t; ++ti) cu.push_back(cu.back() + int32_t(h * w));
    vp.max_len = std::max(vp.max_len, h * w);
  }

  // Interpolated position embeddings, summed in fp32.
  Upload up;
  for (int i = 0; i < 4; ++i) up.add(idx[i]).add(wts[i]);
  const auto dev = up.add(rows).add(cols).add(cu).to(device_);
  vp.pos_embeds = at::zeros({P, D}, pos_embed.options().dtype(at::kFloat));
  for (int i = 0; i < 4; ++i)
    vp.pos_embeds += pos_embed.index_select(0, dev[size_t(2 * i)]).to(at::kFloat) * dev[size_t(2 * i + 1)].unsqueeze(1);
  const at::Tensor &rf = dev[8], &cf = dev[9];
  at::Tensor emb = at::cat({rf.unsqueeze(1) * vision_inv_freq, cf.unsqueeze(1) * vision_inv_freq}, 1);
  emb = at::cat({emb, emb}, 1);  // [P, head_dim]
  vp.cos = emb.cos();
  vp.sin = emb.sin();
  vp.cu_seqlens = dev[10];
  return vp;
}

WeMMModel::Packing WeMMModel::make_packing(const std::vector<int64_t>& seq_lens, const at::Tensor& position_ids) const {
  const auto& tc = cfg_.text;
  Packing pk;
  pk.seq_lens = seq_lens;
  std::vector<int32_t> cu{0}, seq_start, padded_row;
  std::vector<int64_t> last;
  pk.chunk_off_h.push_back(0);
  for (int64_t len : seq_lens) {
    WEMM_CHECK(len > 0, "empty sequence in batch");
    const int32_t base_tok = cu.back();
    const int32_t base_row = int32_t(pk.chunk_start_h.size()) * int32_t(kChunk);
    for (int64_t j = 0; j < len; ++j) {
      seq_start.push_back(base_tok);
      padded_row.push_back(base_row + int32_t(j));
    }
    for (int64_t c = 0; c * kChunk < len; ++c) {
      pk.chunk_start_h.push_back(base_tok + int32_t(c * kChunk));
      pk.chunk_valid_h.push_back(int32_t(std::min<int64_t>(kChunk, len - c * kChunk)));
    }
    pk.chunk_off_h.push_back(int32_t(pk.chunk_start_h.size()));
    cu.push_back(base_tok + int32_t(len));
    last.push_back(cu.back() - 1);
    pk.max_len = std::max(pk.max_len, len);
  }
  const int64_t L = cu.back();
  const int64_t half = text_inv_freq.size(0);
  std::vector<int64_t> sel(size_t(half), 0);
  for (int dim = 1; dim <= 2; ++dim)
    for (int64_t j = dim; j < tc.mrope_section[size_t(dim)] * 3 && j < half; j += 3) sel[size_t(j)] = dim;
  const at::Tensor pos_h = position_ids.to(at::kCPU, at::kLong).contiguous();
  WEMM_CHECK(pos_h.numel() == 3 * L, "position_ids must be [3, L]");
  const auto dev = Upload()
                       .add(cu)
                       .add(seq_start)
                       .add(last)
                       .add(pk.chunk_start_h)
                       .add(pk.chunk_valid_h)
                       .add(pk.chunk_off_h)
                       .add(padded_row)
                       .add(sel)
                       .add(pos_h.data_ptr<int64_t>(), pos_h.numel())
                       .to(device_);
  pk.cu_seqlens = dev[0];
  pk.seq_start = dev[1];
  pk.last_rows = dev[2];
  pk.chunk_start = dev[3];
  pk.chunk_valid = dev[4];
  pk.chunk_off = dev[5];
  pk.padded_row = dev[6];

  // Interleaved M-RoPE (T/H/W sections), fp32.
  at::Tensor pos = dev[8].view({3, L}).to(at::kFloat);   // [3, L]
  at::Tensor freqs = pos.unsqueeze(-1) * text_inv_freq;  // [3, L, R/2]
  at::Tensor sel_t = dev[7].view({1, 1, half}).expand({1, L, half});
  freqs = freqs.gather(0, sel_t).squeeze(0);  // [L, R/2]
  at::Tensor emb = at::cat({freqs, freqs}, -1);
  pk.cos = emb.cos();
  pk.sin = emb.sin();
  return pk;
}

at::Tensor WeMMModel::lookup_embeddings(const std::vector<int64_t>& ids) const {
  if (embed_tokens.is_cuda()) return embed_tokens.index_select(0, Upload().add(ids).to(device_)[0]);
  at::Tensor ids_cpu = at::from_blob(const_cast<int64_t*>(ids.data()), {int64_t(ids.size())}, at::kLong);
  // Gather the rows on the host into pinned memory, then one async copy.
  at::Tensor rows = at::empty({int64_t(ids.size()), embed_tokens.size(1)},
                              at::TensorOptions().dtype(embed_tokens.scalar_type()).pinned_memory(true));
  at::index_select_out(rows, embed_tokens, 0, ids_cpu);
  return rows.to(device_, /*non_blocking=*/true).to(act_);
}

// ================================================================ forward ====
at::Tensor WeMMModel::vision_forward(const at::Tensor& pixel_values, const std::vector<std::array<int64_t, 3>>& grids) {
  NvtxRange range("vision");
  VisionPrelude vp = make_vision_prelude(grids);
  WEMM_CHECK(vp.pos_embeds.size(0) == pixel_values.size(0), "pixel_values rows do not match grids");
  return opts_.fused_kernels ? vision_fused(pixel_values, vp) : vision_reference(pixel_values, vp);
}

at::Tensor WeMMModel::text_forward(at::Tensor h, const at::Tensor& position_ids, const std::vector<int64_t>& seq_lens) {
  NvtxRange range("text");
  Packing pk = make_packing(seq_lens, position_ids);
  WEMM_CHECK(pk.cos.size(0) == h.size(0), "sequence lengths do not add up");
  at::Tensor out = opts_.fused_kernels ? text_fused(std::move(h), pk) : text_reference(std::move(h), pk);
  WEMM_CHECK(out.size(0) == int64_t(seq_lens.size()), "unexpected output rows");
  return out;
}

at::Tensor WeMMModel::embed(const std::vector<ProcessedSample>& batch) {
  c10::InferenceMode guard;
  c10::cuda::CUDAGuard dev_guard(device_);
  WEMM_CHECK(!batch.empty(), "empty batch");

  std::vector<int64_t> ids, seq_lens;
  std::vector<at::Tensor> pos, pixels;
  std::vector<std::array<int64_t, 3>> grids;
  for (const auto& s : batch) {
    ids.insert(ids.end(), s.input_ids.begin(), s.input_ids.end());
    seq_lens.push_back(int64_t(s.input_ids.size()));
    pos.push_back(s.position_ids);
    for (const auto& v : s.visuals) {
      pixels.push_back(make_pixel_values(v, pcfg_, device_, act_));
      grids.push_back({v.grid_t, v.grid_h, v.grid_w});
    }
  }
  at::Tensor h = lookup_embeddings(ids);
  if (!pixels.empty()) {
    at::Tensor feats = vision_forward(at::cat(pixels, 0), grids);
    pixels.clear();
    std::vector<int64_t> where;  // rows of the vision placeholder tokens
    for (size_t i = 0; i < ids.size(); ++i)
      if (ids[i] == cfg_.image_token_id || ids[i] == cfg_.video_token_id) where.push_back(int64_t(i));
    WEMM_CHECK(int64_t(where.size()) == feats.size(0), "vision tokens (", where.size(), ") != features (",
               feats.size(0), ")");
    h.index_copy_(0, Upload().add(where).to(device_)[0], feats.to(h.scalar_type()));
  }
  at::Tensor out = text_forward(std::move(h), at::cat(pos, 1), seq_lens);
  WEMM_CHECK(at::isfinite(out).all().item<bool>(), "non-finite embedding",
             act_ == at::kHalf ? " (fp16 overflow; use compute_dtype=bf16 on sm_80+)" : "");
  return out / out.norm(2, -1, true).clamp_min(1e-12);
}

}  // namespace wemm
