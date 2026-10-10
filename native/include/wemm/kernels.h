#pragma once

#include <ATen/ATen.h>

namespace wemm::kernels {

// Depthwise causal conv1d + SiLU over packed variable-length sequences.
// x: [L, C] bf16 (row stride may exceed C), w: [C, K] fp32,
// seq_start: [L] int32 = index of the first token of each token's sequence.
// Returns [L, C] bf16 (fp32 accumulation, single rounding).
at::Tensor causal_conv1d_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& seq_start);

// Inter-chunk recurrence of the chunked gated delta rule (fp32).
// All per-chunk tensors are [H, C, 64, d] contiguous fp32; dl is [H, C];
// chunk_offsets is [N+1] int32 (chunk ranges of each sequence) on device.
// For each sequence, with S = 0 initially and for its chunks c in order:
//   vnew = u_c - w_c @ S
//   o_c  = qg_c @ S + aqk_c @ vnew
//   S    = S * dl_c + kd_c^T @ vnew
// Returns o: [H, C, 64, dv] fp32.
at::Tensor gdn_chunk_recurrence(const at::Tensor& qg, const at::Tensor& kd, const at::Tensor& u,
                                const at::Tensor& w, const at::Tensor& aqk, const at::Tensor& dl,
                                const at::Tensor& chunk_offsets, int64_t num_seqs);

}  // namespace wemm::kernels

namespace wemm::kernels {

// Qwen3.5 RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w  (w already holds 1 + weight).
// x: [N, D] bf16 or fp32 (row stride may exceed D). Output bf16 or fp32 [N, D].
at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, at::ScalarType out_dtype = at::kBFloat16);

// LayerNorm over the last dim with fp32 math: x [N, D] fp32 or bf16, gamma/beta
// fp32. Returns bf16 [N, D].
at::Tensor layernorm(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double eps);

// Per-head RMSNorm followed by partial rotary embedding on the first `rot`
// dims (rotate_half convention). x: [L, H, hd] bf16 view (unit inner stride),
// cos/sin: [L_all, rot] fp32 indexed by `rows` (or by row number when
// undefined). Returns contiguous bf16 [rows, H, hd].
at::Tensor head_rmsnorm_rope(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cos, const at::Tensor& sin,
                             const at::Tensor& rows, double eps);

// out = o * sigmoid(gate). o: [N, H, hd] bf16 contiguous; gate: [L, H, hd] bf16
// view selected by `rows` (or identity). Returns [N, H*hd] bf16.
at::Tensor sigmoid_gate_mul(const at::Tensor& o, const at::Tensor& gate, const at::Tensor& rows);

// out = silu(gu[:, :I]) * gu[:, I:]  -> [N, I] bf16
at::Tensor silu_mul(const at::Tensor& gu);

// Gated RMSNorm of the delta-rule output: norm(o) * w * silu(z).
// o: [H, C*64, dv] fp32 chunk layout, padded_row: [L] int32 mapping tokens to
// chunk rows, z: [L, H*dv] bf16 view. Returns [L, H*dv] bf16.
at::Tensor gdn_gated_rmsnorm(const at::Tensor& o, const at::Tensor& padded_row, const at::Tensor& z,
                             const at::Tensor& w, double eps);

// 2D vision rotary on q and k of a fused qkv projection [P, 3*nh*hd] bf16.
// Returns {q, k} bf16 [P, nh, hd].
std::pair<at::Tensor, at::Tensor> vision_qk_rope(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                                                 int64_t num_heads);

// Fused chunk preparation + intra-chunk part of the gated delta rule (fp32):
// gathers q/k/v of each 64-token chunk from the conv output, applies the
// L2 norms and the query scale, computes g = neg_A*softplus(a+dt_bias),
// beta = sigmoid(b), the in-chunk cumulative decay, the WY representation
// (I + tril(beta K K^T * decay, -1))^{-1}, and emits the operands of
// `gdn_chunk_recurrence`.
struct GdnChunks {
  at::Tensor qg, kd, u, w, aqk, dl;
};
GdnChunks gdn_chunk_prepare(const at::Tensor& mixed, const at::Tensor& b, const at::Tensor& a,
                            const at::Tensor& neg_A, const at::Tensor& dt_bias, const at::Tensor& chunk_start,
                            const at::Tensor& chunk_valid, int64_t num_k_heads, int64_t num_v_heads, int64_t dk,
                            int64_t dv);

}  // namespace wemm::kernels
