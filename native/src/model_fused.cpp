// Fused-kernel implementation (default): cuBLASLt GEMMs with fused epilogues,
// hand-written kernels, optional fp32 residual streams, bounded workspaces.

#include "model_internal.h"
#include "wemm/common.h"
#include "wemm/gemm.h"
#include "wemm/kernels.h"
#include "wemm/model.h"

namespace wemm {

using namespace detail;

// ==================================================================== ViT ====
at::Tensor WeMMModel::vision_fused(const at::Tensor& pixel_values, const VisionPrelude& vp) {
  const auto& vc = cfg_.vision;
  const int64_t m = vc.spatial_merge_size, D = vc.hidden_size, nh = vc.num_heads, hd = D / nh;
  const int64_t P = pixel_values.size(0);
  // Conv3d == linear; the position embedding is added in fp32.
  at::Tensor x = gemm(pixel_values.to(act_).contiguous(), patch_w, patch_b, std::nullopt, Epilogue::None, at::kFloat);
  x.add_(vp.pos_embeds);
  x = x.to(resid_);
  for (const auto& b : vblocks_) {
    {
      at::Tensor qkv = gemm(kernels::layernorm(x, b.norm1_w, b.norm1_b, 1e-6, act_), b.qkv_w, b.qkv_b);
      auto [q, k] = kernels::vision_qk_rope(qkv, vp.cos, vp.sin, nh);
      at::Tensor v = qkv.view({P, 3, nh, hd}).select(1, 2);
      at::Tensor o = attention(q, k, v, vp.cu_seqlens, vp.cu_seqlens, vp.max_len, vp.max_len, false);
      gemm_accumulate(o.view({P, D}), b.proj_w, b.proj_b, x);
    }
    at::Tensor y = gemm(kernels::layernorm(x, b.norm2_w, b.norm2_b, 1e-6, act_), b.fc1_w, b.fc1_b, std::nullopt,
                        Epilogue::Gelu);
    gemm_accumulate(y, b.fc2_w, b.fc2_b, x);
  }
  at::Tensor y = kernels::layernorm(x, merger_norm_w, merger_norm_b, 1e-6, act_).view({P / (m * m), D * m * m});
  return gemm(at::gelu(gemm(y, merger_fc1_w, merger_fc1_b)), merger_fc2_w, merger_fc2_b);
}

// ===================================================================== LM ====
at::Tensor WeMMModel::text_fused(at::Tensor h, const Packing& pk) {
  const auto& tc = cfg_.text;
  h = h.to(resid_).contiguous();
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto& l = layers_[i];
    const bool last = i + 1 == layers_.size();
    at::Tensor x = kernels::rmsnorm(h, l.in_norm, tc.rms_norm_eps, act_);
    // From the last layer's mixer on, only the <embedding> rows are kept.
    h = l.linear ? linear_attention_fused(l, x, std::move(h), pk, last)
                 : full_attention_fused(l, x, std::move(h), pk, last);
    mlp_fused(l, h);
  }
  return kernels::rmsnorm(h, final_norm, tc.rms_norm_eps, at::kFloat);
}

void WeMMModel::mlp_fused(const TextLayer& l, at::Tensor& h) {
  NvtxRange range("mlp");
  const auto& tc = cfg_.text;
  const int64_t L = h.size(0), step = std::max<int64_t>(1, opts_.mlp_chunk_tokens);
  for (int64_t r = 0; r < L; r += step) {
    at::Tensor hr = h.narrow(0, r, std::min(step, L - r));  // contiguous row block
    at::Tensor gu = gemm(kernels::rmsnorm(hr, l.post_norm, tc.rms_norm_eps, act_), l.gate_up);
    gemm_accumulate(kernels::silu_mul(gu), l.down, std::nullopt, hr);
  }
}

at::Tensor WeMMModel::linear_attention_fused(const TextLayer& l, const at::Tensor& x, at::Tensor h, const Packing& pk,
                                             bool last) {
  NvtxRange range("linear_attn");
  const auto& tc = cfg_.text;
  const int64_t Hk = tc.linear_num_key_heads, Hv = tc.linear_num_value_heads;
  const int64_t dk = tc.linear_key_head_dim, dv = tc.linear_value_head_dim;
  const int64_t key_dim = Hk * dk, value_dim = Hv * dv, conv_dim = 2 * key_dim + value_dim;
  const int64_t L = x.size(0), N = int64_t(pk.seq_lens.size()), C = int64_t(pk.chunk_start_h.size());

  at::Tensor proj = gemm(x, l.in_proj);
  at::Tensor mixed = kernels::causal_conv1d_silu(proj.narrow(1, 0, conv_dim), l.conv_w, pk.seq_start);
  at::Tensor z = proj.narrow(1, conv_dim, value_dim);
  at::Tensor b = proj.narrow(1, conv_dim + value_dim, Hv);
  at::Tensor a = proj.narrow(1, conv_dim + value_dim + Hv, Hv);

  // Gated delta rule over windows of chunks: the fp32 workspace is bounded by
  // gdn_window_bytes; sequences crossing a window edge carry their state.
  at::Tensor normed = at::empty({L, value_dim}, x.options());
  {
    NvtxRange r2("gdn");
    const int64_t per_chunk = kernels::gdn_bytes_per_head_chunk(dk, dv) * Hv;
    const int64_t W = std::max<int64_t>(1, opts_.gdn_window_bytes / per_chunk);
    at::Tensor carry;
    if (W < C) carry = at::empty({Hv, N, dk, dv}, x.options().dtype(at::kFloat));
    for (int64_t c0 = 0; c0 < C; c0 += W) {
      const int64_t c1 = std::min(C, c0 + W);
      auto ch = kernels::gdn_chunk_prepare(mixed, b, a, l.neg_A, l.dt_bias, pk.chunk_start, pk.chunk_valid, Hk, Hv,
                                           dk, dv, c0, c1);
      at::Tensor o = kernels::gdn_chunk_recurrence(ch, pk.chunk_off, N, c0, carry);
      ch = {};
      const int64_t t0 = pk.chunk_start_h[size_t(c0)];
      const int64_t t1 = pk.chunk_start_h[size_t(c1 - 1)] + pk.chunk_valid_h[size_t(c1 - 1)];
      at::Tensor out_rows = normed.narrow(0, t0, t1 - t0);
      kernels::gdn_gated_rmsnorm(o.view({Hv, (c1 - c0) * kChunk, dv}), pk.padded_row.narrow(0, t0, t1 - t0),
                                 z.narrow(0, t0, t1 - t0), l.gnorm_w, tc.rms_norm_eps, c0 * kChunk, out_rows);
    }
  }
  if (last) {
    h = h.index_select(0, pk.last_rows);
    normed = normed.index_select(0, pk.last_rows);
  }
  gemm_accumulate(normed, l.out_proj, std::nullopt, h);
  return h;
}

at::Tensor WeMMModel::full_attention_fused(const TextLayer& l, const at::Tensor& x, at::Tensor h, const Packing& pk,
                                           bool last) {
  NvtxRange range("full_attn");
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0), H = tc.num_attention_heads, Hkv = tc.num_key_value_heads, hd = tc.head_dim;
  at::Tensor proj = gemm(x, l.qkv_proj);
  at::Tensor qg = proj.narrow(1, 0, H * hd * 2).view({L, H, 2 * hd});  // per head: [query | gate]
  at::Tensor kv = proj.narrow(1, H * hd * 2, Hkv * hd).view({L, Hkv, hd});
  at::Tensor v = proj.narrow(1, H * hd * 2 + Hkv * hd, Hkv * hd).view({L, Hkv, hd});
  at::Tensor rows = last ? pk.last_rows : at::Tensor();  // last layer: queries of the <embedding> rows only
  at::Tensor q = kernels::head_rmsnorm_rope(qg.narrow(2, 0, hd), l.q_norm, pk.cos, pk.sin, rows, tc.rms_norm_eps);
  at::Tensor k = kernels::head_rmsnorm_rope(kv, l.k_norm, pk.cos, pk.sin, at::Tensor(), tc.rms_norm_eps);
  at::Tensor o;
  if (last) {
    const int64_t N = q.size(0);
    at::Tensor cu_q = at::arange(0, N + 1, at::TensorOptions().dtype(at::kInt).device(device_));
    o = attention(q, k, v, cu_q, pk.cu_seqlens, 1, pk.max_len, /*causal=*/false);  // 1 query sees its whole sequence
    h = h.index_select(0, pk.last_rows);
  } else {
    o = attention(q, k, v, pk.cu_seqlens, pk.cu_seqlens, pk.max_len, pk.max_len, /*causal=*/true);
  }
  gemm_accumulate(kernels::sigmoid_gate_mul(o, qg.narrow(2, hd, hd), rows), l.o_proj, std::nullopt, h);
  return h;
}

}  // namespace wemm
