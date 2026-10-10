#include "wemm/model.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>

#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <numeric>

#include "wemm/common.h"
#include "wemm/gemm.h"
#include "wemm/kernels.h"
#include "wemm/pixels.h"
#include "wemm/safetensors.h"

namespace wemm {

namespace {

constexpr int64_t kChunk = 64;

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w1p, double eps, at::ScalarType out = at::kBFloat16) {
  at::Tensor xf = x.to(at::kFloat);
  return (xf * at::rsqrt(xf.pow(2).mean(-1, true) + eps) * w1p).to(out);
}

at::Tensor rotate_half(const at::Tensor& x) {
  const int64_t h = x.size(-1) / 2;
  return at::cat({-x.narrow(-1, h, h), x.narrow(-1, 0, h)}, -1);
}

// Flash attention over packed sequences ([total, heads, dim] bf16).
at::Tensor flash_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu_q,
                        const at::Tensor& cu_k, int64_t max_q, int64_t max_k, bool causal, double scale) {
  auto res = at::_flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, false, scale);
  return std::get<0>(res);
}

at::Tensor to_device_i32(const std::vector<int32_t>& v, const at::Device& d) {
  return at::tensor(at::ArrayRef<int32_t>(v), at::kInt).to(d, /*non_blocking=*/true);
}
at::Tensor to_device_i64(const std::vector<int64_t>& v, const at::Device& d) {
  return at::tensor(at::ArrayRef<int64_t>(v), at::kLong).to(d, /*non_blocking=*/true);
}

}  // namespace

// ================================================================ loading ====
WeMMModel::WeMMModel(const std::string& model_dir, ModelOptions opts)
    : cfg_(ModelConfig::load(model_dir)), pcfg_(ProcessorConfig::load(model_dir)), opts_(opts), device_(at::kCUDA, int8_t(opts.device_index)) {
  c10::InferenceMode guard;
  c10::cuda::CUDAGuard dev_guard(device_);
  // fp32 math must stay fp32 (no TF32) to keep the recurrence exact.
  at::globalContext().setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL, at::Float32Precision::IEEE);
  at::globalContext().setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::CONV, at::Float32Precision::IEEE);

  auto st = SafeTensors::open(model_dir);
  auto bf16 = [&](const std::string& n) { return st.get(n).to(device_, at::kBFloat16).contiguous(); };
  auto f32 = [&](const std::string& n) { return st.get(n).to(at::kFloat).to(device_).contiguous(); };

  const auto& vc = cfg_.vision;
  const std::string vp = "model.visual.";
  patch_w = bf16(vp + "patch_embed.proj.weight").reshape({vc.hidden_size, -1}).contiguous();
  patch_b = bf16(vp + "patch_embed.proj.bias");
  patch_b32 = patch_b.to(at::kFloat);
  pos_embed = bf16(vp + "pos_embed.weight");
  for (int i = 0; i < vc.depth; ++i) {
    const std::string p = vp + "blocks." + std::to_string(i) + ".";
    VisionBlock b;
    b.norm1_w = bf16(p + "norm1.weight");
    b.norm1_b = bf16(p + "norm1.bias");
    b.norm2_w = bf16(p + "norm2.weight");
    b.norm2_b = bf16(p + "norm2.bias");
    b.qkv_w = bf16(p + "attn.qkv.weight");
    b.qkv_b = bf16(p + "attn.qkv.bias");
    b.proj_w = bf16(p + "attn.proj.weight");
    b.proj_b = bf16(p + "attn.proj.bias");
    b.fc1_w = bf16(p + "mlp.linear_fc1.weight");
    b.fc1_b = bf16(p + "mlp.linear_fc1.bias");
    b.fc2_w = bf16(p + "mlp.linear_fc2.weight");
    b.fc2_b = bf16(p + "mlp.linear_fc2.bias");
    b.proj_b32 = b.proj_b.to(at::kFloat);
    b.fc2_b32 = b.fc2_b.to(at::kFloat);
    b.n1w = b.norm1_w.to(at::kFloat);
    b.n1b = b.norm1_b.to(at::kFloat);
    b.n2w = b.norm2_w.to(at::kFloat);
    b.n2b = b.norm2_b.to(at::kFloat);
    vblocks_.push_back(std::move(b));
  }
  merger_norm_w = bf16(vp + "merger.norm.weight");
  merger_norm_b = bf16(vp + "merger.norm.bias");
  merger_nw = merger_norm_w.to(at::kFloat);
  merger_nb = merger_norm_b.to(at::kFloat);
  merger_fc1_w = bf16(vp + "merger.linear_fc1.weight");
  merger_fc1_b = bf16(vp + "merger.linear_fc1.bias");
  merger_fc2_w = bf16(vp + "merger.linear_fc2.weight");
  merger_fc2_b = bf16(vp + "merger.linear_fc2.bias");
  {
    const int64_t dim = vc.hidden_size / vc.num_heads / 2;  // Qwen3_5VisionRotaryEmbedding(head_dim // 2)
    vision_inv_freq =
        (1.0 / at::pow(10000.0, at::arange(0, dim, 2, at::kFloat) / double(dim))).to(device_);
  }

  const auto& tc = cfg_.text;
  const std::string tp = "model.language_model.";
  embed_tokens = bf16(tp + "embed_tokens.weight");
  for (int i = 0; i < tc.num_hidden_layers; ++i) {
    const std::string p = tp + "layers." + std::to_string(i) + ".";
    TextLayer l;
    l.linear = tc.layer_types[size_t(i)] == "linear_attention";
    l.in_norm = 1.0 + f32(p + "input_layernorm.weight");
    l.post_norm = 1.0 + f32(p + "post_attention_layernorm.weight");
    if (l.linear) {
      const std::string a = p + "linear_attn.";
      l.in_proj = at::cat({bf16(a + "in_proj_qkv.weight"), bf16(a + "in_proj_z.weight"), bf16(a + "in_proj_b.weight"),
                           bf16(a + "in_proj_a.weight")},
                          0);
      // Pad the fused projection to a multiple of 128 outputs (faster GEMM tiles).
      const int64_t rows = l.in_proj.size(0), padded = (rows + 127) / 128 * 128;
      l.in_proj = at::cat({l.in_proj, at::zeros({padded - rows, l.in_proj.size(1)}, l.in_proj.options())}, 0)
                      .contiguous();
      l.conv_w = f32(a + "conv1d.weight").squeeze(1).contiguous();
      l.neg_A = -f32(a + "A_log").exp();
      l.dt_bias = f32(a + "dt_bias");
      l.gnorm_w = f32(a + "norm.weight");
      l.out_proj = bf16(a + "out_proj.weight");
    } else {
      const std::string a = p + "self_attn.";
      WEMM_CHECK(tc.attn_output_gate, "attention without output gate is not supported");
      l.qkv_proj = at::cat({bf16(a + "q_proj.weight"), bf16(a + "k_proj.weight"), bf16(a + "v_proj.weight")}, 0)
                       .contiguous();
      l.q_norm = 1.0 + f32(a + "q_norm.weight");
      l.k_norm = 1.0 + f32(a + "k_norm.weight");
      l.o_proj = bf16(a + "o_proj.weight");
    }
    l.gate_up = at::cat({bf16(p + "mlp.gate_proj.weight"), bf16(p + "mlp.up_proj.weight")}, 0).contiguous();
    l.down = bf16(p + "mlp.down_proj.weight");
    layers_.push_back(std::move(l));
  }
  final_norm = 1.0 + f32(tp + "norm.weight");
  {
    const int64_t dim = int64_t(tc.head_dim * tc.partial_rotary_factor);
    text_inv_freq = (1.0 / at::pow(tc.rope_theta, at::arange(0, dim, 2, at::kLong).to(at::kFloat) / double(dim)))
                        .to(device_);
  }
}

// ================================================================ vision =====
at::Tensor WeMMModel::vision_forward(const at::Tensor& pixel_values, const std::vector<std::array<int64_t, 3>>& grids) {
  NvtxRange range("vision");
  const auto& vc = cfg_.vision;
  const int64_t m = vc.spatial_merge_size, D = vc.hidden_size, nh = vc.num_heads, hd = D / nh;
  const int64_t side = int64_t(std::lround(std::sqrt(double(vc.num_position_embeddings))));

  // Position-embedding interpolation (fast_pos_embed_interpolate) and 2D rotary
  // indices, in the merge-window patch order.
  std::vector<int64_t> idx[4];
  std::vector<float> wts[4];
  std::vector<float> rows, cols;
  std::vector<int32_t> cu{0};
  int64_t max_len = 0;
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
    max_len = std::max(max_len, h * w);
  }
  const int64_t P = int64_t(rows.size());
  WEMM_CHECK(P == pixel_values.size(0), "pixel_values rows do not match grids");

  at::Tensor pe = at::zeros({P, D}, pos_embed.options().dtype(at::kFloat));
  for (int i = 0; i < 4; ++i) {
    at::Tensor ii = to_device_i64(idx[i], device_);
    at::Tensor ww = at::tensor(at::ArrayRef<float>(wts[i])).to(device_);
    pe += pos_embed.index_select(0, ii).to(at::kFloat) * ww.unsqueeze(1);
  }

  at::Tensor rf = at::tensor(at::ArrayRef<float>(rows)).to(device_);
  at::Tensor cf = at::tensor(at::ArrayRef<float>(cols)).to(device_);
  at::Tensor emb = at::cat({rf.unsqueeze(1) * vision_inv_freq, cf.unsqueeze(1) * vision_inv_freq}, 1);
  emb = at::cat({emb, emb}, 1);  // [P, hd]
  at::Tensor cu_t = to_device_i32(cu, device_);
  if (opts_.fused_kernels) return vision_forward_fused(pixel_values, pe, emb.cos(), emb.sin(), cu_t, max_len);

  at::Tensor x = at::linear(pixel_values.to(at::kBFloat16), patch_w, patch_b);  // Conv3d == linear
  x = (x.to(at::kFloat) + pe).to(at::kBFloat16);
  at::Tensor cos = emb.cos().unsqueeze(1), sin = emb.sin().unsqueeze(1);
  const double scale = 1.0 / std::sqrt(double(hd));

  for (const auto& b : vblocks_) {
    at::Tensor y = at::layer_norm(x, {D}, b.norm1_w, b.norm1_b, 1e-6);
    at::Tensor qkv = at::linear(y, b.qkv_w, b.qkv_b).view({P, 3, nh, hd});
    at::Tensor q = qkv.select(1, 0).to(at::kFloat), k = qkv.select(1, 1).to(at::kFloat);
    q = (q * cos + rotate_half(q) * sin).to(at::kBFloat16);
    k = (k * cos + rotate_half(k) * sin).to(at::kBFloat16);
    at::Tensor v = qkv.select(1, 2).contiguous();
    at::Tensor o = flash_varlen(q, k, v, cu_t, cu_t, max_len, max_len, false, scale);
    x = x + at::linear(o.reshape({P, D}), b.proj_w, b.proj_b);
    y = at::layer_norm(x, {D}, b.norm2_w, b.norm2_b, 1e-6);
    x = x + at::linear(at::gelu(at::linear(y, b.fc1_w, b.fc1_b), "tanh"), b.fc2_w, b.fc2_b);
  }
  at::Tensor y = at::layer_norm(x, {D}, merger_norm_w, merger_norm_b, 1e-6).view({P / (m * m), D * m * m});
  return at::linear(at::gelu(at::linear(y, merger_fc1_w, merger_fc1_b)), merger_fc2_w, merger_fc2_b);
}

at::Tensor WeMMModel::vision_forward_fused(const at::Tensor& pixel_values, const at::Tensor& pe, const at::Tensor& cos,
                                           const at::Tensor& sin, const at::Tensor& cu, int64_t max_len) {
  const auto& vc = cfg_.vision;
  const int64_t m = vc.spatial_merge_size, D = vc.hidden_size, nh = vc.num_heads, hd = D / nh;
  const int64_t P = pixel_values.size(0);
  const double scale = 1.0 / std::sqrt(double(hd));
  const auto rdt = opts_.fp32_residual ? at::kFloat : at::kBFloat16;
  // Conv3d == linear; the position embedding is added in fp32.
  at::Tensor x = (gemm(pixel_values.to(at::kBFloat16).contiguous(), patch_w, patch_b32, std::nullopt, Epilogue::None,
                       at::kFloat) + pe).to(rdt);
  const bool f32 = opts_.fp32_residual;
  for (const auto& b : vblocks_) {
    at::Tensor y = kernels::layernorm(x, b.n1w, b.n1b, 1e-6);
    at::Tensor qkv = gemm(y, b.qkv_w, b.qkv_b);
    auto [q, k] = kernels::vision_qk_rope(qkv, cos, sin, nh);
    at::Tensor v = qkv.view({P, 3, nh, hd}).select(1, 2);
    at::Tensor o = flash_varlen(q, k, v, cu, cu, max_len, max_len, false, scale);
    x = gemm(o.view({P, D}), b.proj_w, f32 ? b.proj_b32 : b.proj_b, x);
    y = kernels::layernorm(x, b.n2w, b.n2b, 1e-6);
    x = gemm(gemm(y, b.fc1_w, b.fc1_b, std::nullopt, Epilogue::Gelu), b.fc2_w, f32 ? b.fc2_b32 : b.fc2_b, x);
  }
  at::Tensor y = kernels::layernorm(x, merger_nw, merger_nb, 1e-6).view({P / (m * m), D * m * m});
  return gemm(at::gelu(gemm(y, merger_fc1_w, merger_fc1_b)), merger_fc2_w, merger_fc2_b);
}

// ================================================================ text =======
WeMMModel::Packing WeMMModel::make_packing(const std::vector<int64_t>& seq_lens) const {
  Packing pk;
  pk.seq_lens = seq_lens;
  std::vector<int32_t> cu{0}, seq_start, chunk_start, chunk_valid, chunk_off{0}, padded_row;
  std::vector<int64_t> last;
  for (int64_t len : seq_lens) {
    const int32_t base_tok = cu.back();
    const int32_t base_row = int32_t(chunk_start.size()) * int32_t(kChunk);
    for (int64_t j = 0; j < len; ++j) {
      seq_start.push_back(base_tok);
      padded_row.push_back(base_row + int32_t(j));
    }
    const int64_t nch = (len + kChunk - 1) / kChunk;
    for (int64_t c = 0; c < nch; ++c) {
      chunk_start.push_back(base_tok + int32_t(c * kChunk));
      chunk_valid.push_back(int32_t(std::min<int64_t>(kChunk, len - c * kChunk)));
    }
    chunk_off.push_back(chunk_off.back() + int32_t(nch));
    cu.push_back(base_tok + int32_t(len));
    last.push_back(cu.back() - 1);
    pk.max_len = std::max(pk.max_len, len);
  }
  pk.cu_seqlens = to_device_i32(cu, device_);
  pk.seq_start = to_device_i32(seq_start, device_);
  pk.last_rows = to_device_i64(last, device_);
  pk.chunk_start = to_device_i32(chunk_start, device_);
  pk.chunk_valid = to_device_i32(chunk_valid, device_);
  pk.chunk_off = to_device_i32(chunk_off, device_);
  pk.padded_row = to_device_i32(padded_row, device_);
  return pk;
}

at::Tensor WeMMModel::linear_attention_fused(const TextLayer& l, const at::Tensor& x, const at::Tensor& h,
                                             const Packing& pk, bool last) {
  NvtxRange range("linear_attn");
  const auto& tc = cfg_.text;
  const int64_t Hk = tc.linear_num_key_heads, Hv = tc.linear_num_value_heads;
  const int64_t dk = tc.linear_key_head_dim, dv = tc.linear_value_head_dim;
  const int64_t key_dim = Hk * dk, value_dim = Hv * dv, conv_dim = 2 * key_dim + value_dim;
  const int64_t N = int64_t(pk.seq_lens.size());

  at::Tensor proj = gemm(x, l.in_proj);
  at::Tensor mixed = kernels::causal_conv1d_silu(proj.narrow(1, 0, conv_dim), l.conv_w, pk.seq_start);
  at::Tensor z = proj.narrow(1, conv_dim, value_dim);
  at::Tensor b = proj.narrow(1, conv_dim + value_dim, Hv);
  at::Tensor a = proj.narrow(1, conv_dim + value_dim + Hv, Hv);
  at::Tensor o;
  {
    NvtxRange r2("gdn");
    auto ch = kernels::gdn_chunk_prepare(mixed, b, a, l.neg_A, l.dt_bias, pk.chunk_start, pk.chunk_valid, Hk, Hv,
                                         dk, dv);
    o = kernels::gdn_chunk_recurrence(ch.qg, ch.kd, ch.u, ch.w, ch.aqk, ch.dl, pk.chunk_off, N);
  }
  const int64_t C = o.size(1);
  if (last) {
    at::Tensor rows = pk.padded_row.index_select(0, pk.last_rows);
    at::Tensor on = kernels::gdn_gated_rmsnorm(o.view({Hv, C * kChunk, dv}), rows, z.index_select(0, pk.last_rows),
                                               l.gnorm_w, tc.rms_norm_eps);
    return gemm(on, l.out_proj, std::nullopt, h.index_select(0, pk.last_rows));
  }
  at::Tensor on = kernels::gdn_gated_rmsnorm(o.view({Hv, C * kChunk, dv}), pk.padded_row, z, l.gnorm_w, tc.rms_norm_eps);
  return gemm(on, l.out_proj, std::nullopt, h);
}

at::Tensor WeMMModel::full_attention_fused(const TextLayer& l, const at::Tensor& x, const at::Tensor& h,
                                           const at::Tensor& cos, const at::Tensor& sin, const Packing& pk, bool last) {
  NvtxRange range("full_attn");
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0), H = tc.num_attention_heads, Hkv = tc.num_key_value_heads, hd = tc.head_dim;
  at::Tensor proj = gemm(x, l.qkv_proj);
  at::Tensor qg = proj.narrow(1, 0, H * hd * 2).view({L, H, 2 * hd});
  at::Tensor qv = qg.narrow(2, 0, hd), gate = qg.narrow(2, hd, hd);
  at::Tensor kv = proj.narrow(1, H * hd * 2, Hkv * hd).view({L, Hkv, hd});
  at::Tensor v = proj.narrow(1, H * hd * 2 + Hkv * hd, Hkv * hd).view({L, Hkv, hd});
  at::Tensor rows = last ? pk.last_rows : at::Tensor();
  at::Tensor q = kernels::head_rmsnorm_rope(qv, l.q_norm, cos, sin, rows, tc.rms_norm_eps);
  at::Tensor k = kernels::head_rmsnorm_rope(kv, l.k_norm, cos, sin, at::Tensor(), tc.rms_norm_eps);
  const double scale = 1.0 / std::sqrt(double(hd));
  at::Tensor o;
  if (last) {
    const int64_t N = q.size(0);
    at::Tensor cu_q = at::arange(0, N + 1, at::TensorOptions().dtype(at::kInt).device(device_));
    o = flash_varlen(q, k, v, cu_q, pk.cu_seqlens, 1, pk.max_len, /*causal=*/false, scale);
  } else {
    o = flash_varlen(q, k, v, pk.cu_seqlens, pk.cu_seqlens, pk.max_len, pk.max_len, /*causal=*/true, scale);
  }
  at::Tensor og = kernels::sigmoid_gate_mul(o, gate, rows);
  return gemm(og, l.o_proj, std::nullopt, last ? h.index_select(0, pk.last_rows) : h);
}

at::Tensor WeMMModel::gated_delta_rule(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const at::Tensor& g, const at::Tensor& beta,
                                       const std::vector<int64_t>& seq_lens) {
  NvtxRange range("gdn");
  const int64_t L = q.size(0), H = q.size(1), dk = q.size(2), dv = v.size(2);
  // Chunk-padded layout: every sequence starts on a chunk boundary.
  std::vector<int64_t> gather, scatter(static_cast<size_t>(L));
  std::vector<int32_t> chunk_off{0};
  int64_t tok = 0;
  for (int64_t len : seq_lens) {
    const int64_t nch = (len + kChunk - 1) / kChunk;
    const int64_t base = int64_t(gather.size());
    for (int64_t j = 0; j < nch * kChunk; ++j) gather.push_back(j < len ? tok + j : L);
    for (int64_t j = 0; j < len; ++j) scatter[size_t(tok + j)] = base + j;
    chunk_off.push_back(chunk_off.back() + int32_t(nch));
    tok += len;
  }
  const int64_t C = chunk_off.back();
  at::Tensor gidx = to_device_i64(gather, device_);
  at::Tensor sidx = to_device_i64(scatter, device_);

  auto l2norm = [](const at::Tensor& x) {
    at::Tensor xf = x.to(at::kFloat);
    return xf * at::rsqrt((xf * xf).sum(-1, true) + 1e-6);
  };
  auto pad_rows = [&](const at::Tensor& x) {  // x: [L, ...] -> append a zero row
    return at::cat({x, at::zeros_like(x.narrow(0, 0, 1))}, 0);
  };
  auto to_chunks = [&](const at::Tensor& x) {  // [L, H, d] -> [H, C, 64, d]
    return pad_rows(x).index_select(0, gidx).view({C, kChunk, H, -1}).permute({2, 0, 1, 3}).contiguous();
  };
  auto to_chunks2 = [&](const at::Tensor& x) {  // [L, H] -> [H, C, 64]
    return pad_rows(x).index_select(0, gidx).view({C, kChunk, H}).permute({2, 0, 1}).contiguous();
  };

  at::Tensor Q = to_chunks(l2norm(q) * (1.0 / std::sqrt(double(dk))));
  at::Tensor K = to_chunks(l2norm(k));
  at::Tensor V = to_chunks(v.to(at::kFloat));
  at::Tensor G = to_chunks2(g);
  at::Tensor B = to_chunks2(beta);

  at::Tensor gc = G.cumsum(-1);
  at::Tensor eg = gc.exp();
  at::Tensor kb = K * B.unsqueeze(-1);
  at::Tensor vb = V * B.unsqueeze(-1);
  at::Tensor lower = at::ones({kChunk, kChunk}, at::TensorOptions().dtype(at::kBool).device(device_)).tril();
  at::Tensor decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(lower.logical_not(), 0).exp() * lower;
  at::Tensor A = at::matmul(kb, K.transpose(-1, -2)) * decay;
  A.diagonal(0, -2, -1).fill_(0);
  A.diagonal(0, -2, -1).add_(1);  // I + strictly-lower(A)
  at::Tensor rhs = at::cat({vb, kb * eg.unsqueeze(-1)}, -1);
  at::Tensor sol = at::linalg_solve_triangular(A, rhs, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
  at::Tensor u = sol.narrow(-1, 0, dv).contiguous();
  at::Tensor w = sol.narrow(-1, dv, dk).contiguous();
  at::Tensor aqk = (at::matmul(Q, K.transpose(-1, -2)) * decay).contiguous();
  at::Tensor glast = gc.select(-1, kChunk - 1);
  at::Tensor qg = (Q * eg.unsqueeze(-1)).contiguous();
  at::Tensor kd = (K * (glast.unsqueeze(-1) - gc).exp().unsqueeze(-1)).contiguous();
  at::Tensor dl = glast.exp().contiguous();

  at::Tensor o;
  if (opts_.fused_kernels) {
    o = kernels::gdn_chunk_recurrence(qg, kd, u, w, aqk, dl, to_device_i32(chunk_off, device_),
                                      int64_t(seq_lens.size()));
  } else {
    o = at::empty({H, C, kChunk, dv}, Q.options());
    for (size_t s = 0; s < seq_lens.size(); ++s) {
      at::Tensor S = at::zeros({H, dk, dv}, Q.options());
      for (int64_t c = chunk_off[s]; c < chunk_off[s + 1]; ++c) {
        at::Tensor vnew = u.select(1, c) - at::matmul(w.select(1, c), S);
        o.select(1, c).copy_(at::matmul(qg.select(1, c), S) + at::matmul(aqk.select(1, c), vnew));
        S = S * dl.select(1, c).view({H, 1, 1}) + at::matmul(kd.select(1, c).transpose(-1, -2), vnew);
      }
    }
  }
  return o.permute({1, 2, 0, 3}).reshape({C * kChunk, H, dv}).index_select(0, sidx);  // [L, H, dv] fp32
}

at::Tensor WeMMModel::linear_attention(const TextLayer& l, const at::Tensor& x, const at::Tensor& seq_start,
                                       const std::vector<int64_t>& seq_lens) {
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0);
  const int64_t Hk = tc.linear_num_key_heads, Hv = tc.linear_num_value_heads;
  const int64_t dk = tc.linear_key_head_dim, dv = tc.linear_value_head_dim;
  const int64_t key_dim = Hk * dk, value_dim = Hv * dv, conv_dim = 2 * key_dim + value_dim;

  at::Tensor proj = at::linear(x, l.in_proj);
  at::Tensor mixed;
  if (opts_.fused_kernels) {
    mixed = kernels::causal_conv1d_silu(proj.narrow(1, 0, conv_dim), l.conv_w, seq_start);
  } else {
    // Reference: per-sequence depthwise conv in fp32, single rounding.
    at::Tensor xs = proj.narrow(1, 0, conv_dim).to(at::kFloat);
    std::vector<at::Tensor> outs;
    int64_t off = 0;
    for (int64_t len : seq_lens) {
      at::Tensor xi = xs.narrow(0, off, len).t().unsqueeze(0);  // [1, C, len]
      at::Tensor yi = at::conv1d(xi, l.conv_w.unsqueeze(1), {}, 1, l.conv_w.size(1) - 1, 1, conv_dim);
      outs.push_back(at::silu(yi.narrow(2, 0, len)).squeeze(0).t());
      off += len;
    }
    mixed = at::cat(outs, 0).to(at::kBFloat16);
  }
  at::Tensor z = proj.narrow(1, conv_dim, value_dim);
  at::Tensor b = proj.narrow(1, conv_dim + value_dim, Hv);
  at::Tensor a = proj.narrow(1, conv_dim + value_dim + Hv, Hv);

  at::Tensor q = mixed.narrow(1, 0, key_dim).view({L, Hk, dk});
  at::Tensor k = mixed.narrow(1, key_dim, key_dim).view({L, Hk, dk});
  at::Tensor v = mixed.narrow(1, 2 * key_dim, value_dim).view({L, Hv, dv});
  at::Tensor beta = at::sigmoid(b.to(at::kFloat));
  at::Tensor g = l.neg_A * at::softplus(a.to(at::kFloat) + l.dt_bias);
  if (Hv / Hk > 1) {
    q = q.repeat_interleave(Hv / Hk, 1);
    k = k.repeat_interleave(Hv / Hk, 1);
  }
  at::Tensor o = gated_delta_rule(q, k, v, g, beta, seq_lens);  // [L, Hv, dv] fp32
  // Qwen3_5RMSNormGated: norm(o) * w * silu(z)
  at::Tensor zf = z.reshape({L, Hv, dv}).to(at::kFloat);
  o = o * at::rsqrt(o.pow(2).mean(-1, true) + tc.rms_norm_eps) * l.gnorm_w * at::silu(zf);
  return at::linear(o.reshape({L, value_dim}).to(at::kBFloat16), l.out_proj);
}

at::Tensor WeMMModel::full_attention(const TextLayer& l, const at::Tensor& x, const at::Tensor& cos,
                                     const at::Tensor& sin, const at::Tensor& cu_seqlens, int64_t max_len,
                                     const at::Tensor* q_rows) {
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0), H = tc.num_attention_heads, Hkv = tc.num_key_value_heads, hd = tc.head_dim;
  const int64_t rot = cos.size(-1);
  at::Tensor proj = at::linear(x, l.qkv_proj);
  at::Tensor qg = proj.narrow(1, 0, H * hd * 2).view({L, H, 2 * hd});
  at::Tensor q = qg.narrow(2, 0, hd), gate = qg.narrow(2, hd, hd);
  at::Tensor k = proj.narrow(1, H * hd * 2, Hkv * hd).view({L, Hkv, hd});
  at::Tensor v = proj.narrow(1, H * hd * 2 + Hkv * hd, Hkv * hd).view({L, Hkv, hd}).contiguous();
  at::Tensor qcos = cos, qsin = sin;
  if (q_rows) {
    q = q.index_select(0, *q_rows);
    gate = gate.index_select(0, *q_rows);
    qcos = cos.index_select(0, *q_rows);
    qsin = sin.index_select(0, *q_rows);
  }
  auto norm_rope = [&](const at::Tensor& t, const at::Tensor& w, const at::Tensor& c, const at::Tensor& s) {
    at::Tensor f = rmsnorm(t, w, tc.rms_norm_eps, at::kFloat);
    at::Tensor r = f.narrow(-1, 0, rot);
    r = r * c.unsqueeze(1) + rotate_half(r) * s.unsqueeze(1);
    return at::cat({r, f.narrow(-1, rot, hd - rot)}, -1).to(at::kBFloat16);
  };
  q = norm_rope(q, l.q_norm, qcos, qsin);
  k = norm_rope(k, l.k_norm, cos, sin);
  const double scale = 1.0 / std::sqrt(double(hd));
  at::Tensor o;
  if (q_rows) {
    const int64_t N = q.size(0);
    at::Tensor cu_q = at::arange(0, N + 1, at::TensorOptions().dtype(at::kInt).device(device_));
    o = flash_varlen(q, k, v, cu_q, cu_seqlens, 1, max_len, /*causal=*/false, scale);
  } else {
    o = flash_varlen(q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, /*causal=*/true, scale);
  }
  o = (o.to(at::kFloat) * at::sigmoid(gate.to(at::kFloat))).to(at::kBFloat16);
  return at::linear(o.reshape({o.size(0), H * hd}), l.o_proj);
}

at::Tensor WeMMModel::mlp(const TextLayer& l, const at::Tensor& x) {
  NvtxRange range("mlp");
  at::Tensor gu = at::linear(x, l.gate_up);
  const int64_t I = gu.size(1) / 2;
  at::Tensor act = (at::silu(gu.narrow(1, 0, I).to(at::kFloat)) * gu.narrow(1, I, I).to(at::kFloat)).to(at::kBFloat16);
  return at::linear(act, l.down);
}

at::Tensor WeMMModel::text_forward(at::Tensor h, const at::Tensor& position_ids, const std::vector<int64_t>& seq_lens) {
  NvtxRange range("text");
  const auto& tc = cfg_.text;
  const int64_t L = h.size(0), N = int64_t(seq_lens.size());
  std::vector<int32_t> cu{0}, seq_start;
  std::vector<int64_t> last;
  seq_start.reserve(size_t(L));
  int64_t max_len = 0;
  for (int64_t len : seq_lens) {
    for (int64_t j = 0; j < len; ++j) seq_start.push_back(cu.back());
    cu.push_back(cu.back() + int32_t(len));
    last.push_back(cu.back() - 1);
    max_len = std::max(max_len, len);
  }
  WEMM_CHECK(cu.back() == L, "sequence lengths do not add up");
  at::Tensor cu_t = to_device_i32(cu, device_);
  at::Tensor seq_start_t = to_device_i32(seq_start, device_);
  at::Tensor last_t = to_device_i64(last, device_);

  // Interleaved M-RoPE (T/H/W sections), fp32.
  at::Tensor pos = position_ids.to(device_).to(at::kFloat);              // [3, L]
  at::Tensor freqs = pos.unsqueeze(-1) * text_inv_freq;                  // [3, L, R/2]
  const int64_t half = text_inv_freq.size(0);
  std::vector<int64_t> sel(size_t(half), 0);
  for (int dim = 1; dim <= 2; ++dim)
    for (int64_t j = dim; j < tc.mrope_section[size_t(dim)] * 3 && j < half; j += 3) sel[size_t(j)] = dim;
  at::Tensor sel_t = to_device_i64(sel, device_).view({1, 1, half}).expand({1, L, half});
  freqs = freqs.gather(0, sel_t).squeeze(0);                              // [L, R/2]
  at::Tensor emb = at::cat({freqs, freqs}, -1);
  at::Tensor cos = emb.cos(), sin = emb.sin();

  if (opts_.fused_kernels) {
    Packing pk = make_packing(seq_lens);
    if (opts_.fp32_residual) h = h.to(at::kFloat);
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto& l = layers_[i];
      const bool is_last = i + 1 == layers_.size();
      at::Tensor x = kernels::rmsnorm(h, l.in_norm, tc.rms_norm_eps);
      h = l.linear ? linear_attention_fused(l, x, h, pk, is_last) : full_attention_fused(l, x, h, cos, sin, pk, is_last);
      NvtxRange r("mlp");
      at::Tensor gu = gemm(kernels::rmsnorm(h, l.post_norm, tc.rms_norm_eps), l.gate_up);
      h = gemm(kernels::silu_mul(gu), l.down, std::nullopt, h);
    }
    WEMM_CHECK(h.size(0) == N, "unexpected output rows");
    return kernels::rmsnorm(h, final_norm, tc.rms_norm_eps, at::kFloat);
  }
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto& l = layers_[i];
    const bool is_last = i + 1 == layers_.size();
    at::Tensor x = rmsnorm(h, l.in_norm, tc.rms_norm_eps);
    at::Tensor mix;
    NvtxRange lr(l.linear ? "linear_attn" : "full_attn");
    if (l.linear) {
      mix = linear_attention(l, x, seq_start_t, seq_lens);
      if (is_last) mix = mix.index_select(0, last_t);
    } else {
      mix = full_attention(l, x, cos, sin, cu_t, max_len, is_last ? &last_t : nullptr);
    }
    if (is_last) h = h.index_select(0, last_t);  // only the <embedding> rows are needed from here on
    h = h + mix;
    h = h + mlp(l, rmsnorm(h, l.post_norm, tc.rms_norm_eps));
  }
  WEMM_CHECK(h.size(0) == N, "unexpected output rows");
  return rmsnorm(h, final_norm, tc.rms_norm_eps, at::kFloat);
}

// ================================================================ embed ======
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
      pixels.push_back(make_pixel_values(v, pcfg_, device_, at::kBFloat16));
      grids.push_back({v.grid_t, v.grid_h, v.grid_w});
    }
  }
  at::Tensor ids_t = to_device_i64(ids, device_);
  at::Tensor h = embed_tokens.index_select(0, ids_t);
  if (!pixels.empty()) {
    at::Tensor feats = vision_forward(at::cat(pixels, 0), grids);
    at::Tensor is_vis = (ids_t == cfg_.image_token_id).logical_or_(ids_t == cfg_.video_token_id);
    at::Tensor where = is_vis.nonzero().squeeze(1);
    WEMM_CHECK(where.size(0) == feats.size(0), "vision tokens (", where.size(0), ") != features (", feats.size(0), ")");
    h.index_copy_(0, where, feats.to(h.scalar_type()));
  }
  at::Tensor out = text_forward(h, at::cat(pos, 1), seq_lens);
  return out / out.norm(2, -1, true).clamp_min(1e-12);
}

}  // namespace wemm
