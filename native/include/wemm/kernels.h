#pragma once

// Hand-written CUDA kernels. "act" tensors are bf16 (sm_80+) or fp16 (sm_75);
// all reductions and elementwise math run in fp32 and round once.

#include <ATen/ATen.h>

#include <array>
#include <utility>

namespace wemm::kernels {

// ------------------------------------------------------------------- norms
// Qwen3.5 RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w  (w already holds 1 + weight).
// x: [N, D] fp32 or act (row stride may exceed D). Output act or fp32 [N, D].
at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, at::ScalarType out_dtype);

// LayerNorm over the last dim: x [N, D] fp32 or act, gamma/beta fp32.
at::Tensor layernorm(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double eps,
                     at::ScalarType out_dtype);

// ---------------------------------------------------------- full attention
// Per-head RMSNorm followed by partial rotary embedding on the first `rot`
// dims (rotate_half convention). x: [L, H, hd] act view (unit inner stride),
// cos/sin: [L_all, rot] fp32 indexed by `rows` (int64, or row number when
// undefined). Returns contiguous act [rows, H, hd].
at::Tensor head_rmsnorm_rope(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cos, const at::Tensor& sin,
                             const at::Tensor& rows, double eps);

// out = o * sigmoid(gate). o: [N, H, hd] act contiguous; gate: [L, H, hd] act
// view selected by `rows` (or identity). Returns [N, H*hd].
at::Tensor sigmoid_gate_mul(const at::Tensor& o, const at::Tensor& gate, const at::Tensor& rows);

// --------------------------------------------------------------------- mlp
// out = silu(gu[:, :I]) * gu[:, I:]  -> [N, I]
at::Tensor silu_mul(const at::Tensor& gu);

// --------------------------------------------------------------------- vit
// 2D rotary on q and k of a fused qkv projection [P, 3*nh*hd].
// Returns {q, k} [P, nh, hd].
std::pair<at::Tensor, at::Tensor> vision_qk_rope(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                                                 int64_t num_heads);

// ------------------------------------------------------- gated delta rule
// Depthwise causal conv1d + SiLU over packed variable-length sequences.
// x: [L, C] act (row stride may exceed C), w: [C, K] fp32, seq_start: [L]
// int32 = first token of each token's sequence. Returns [L, C] act.
at::Tensor causal_conv1d_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& seq_start);

// Operands of the chunked gated delta rule for a window of 64-token chunks
// (all fp32, [H, C, 64, d] / [H, C]).
struct GdnChunks {
  at::Tensor qg, kd, u, w, aqk, dl;
};

// Fused chunk preparation + intra-chunk part: gathers q/k/v of each chunk
// from the conv output, applies the L2 norms and the query scale, computes
// g = neg_A*softplus(a+dt_bias), beta = sigmoid(b), the in-chunk cumulative
// decay and the WY representation (I + tril(beta K K^T * decay, -1))^{-1}.
// Only chunks [chunk_begin, chunk_end) are prepared (chunk_end < 0: all).
GdnChunks gdn_chunk_prepare(const at::Tensor& mixed, const at::Tensor& b, const at::Tensor& a,
                            const at::Tensor& neg_A, const at::Tensor& dt_bias, const at::Tensor& chunk_start,
                            const at::Tensor& chunk_valid, int64_t num_k_heads, int64_t num_v_heads, int64_t dk,
                            int64_t dv, int64_t chunk_begin = 0, int64_t chunk_end = -1);

// Inter-chunk recurrence (fp32). For each sequence, with S = 0 initially and
// for its chunks c in order:
//   S_c = S;  vnew_c = u_c - w_c @ S;  S = S * dl_c + kd_c^T @ vnew_c
// The per-chunk tensors may cover only chunks [chunk_begin, chunk_begin + C)
// of the packed batch (chunk_offsets [N+1] int32 is global); sequences that
// continue past the window store/restore S in `carry` [H, N, dk, dv].
// The state kernel double-buffers its tiles with cp.async when the GPU allows
// (sm_80+, >= 96 KB smem); allow_double_buffer=false forces the 64 KB
// single-buffer variant (sm_75).
struct GdnRecurrence {
  at::Tensor states;  // [H, C, dk, dv] S_c; undefined when every chunk starts a sequence
  at::Tensor vnew;    // [H, C, 64, dv]
};
GdnRecurrence gdn_chunk_recurrence(const GdnChunks& ch, const at::Tensor& chunk_offsets, int64_t num_seqs,
                                   int64_t chunk_begin = 0, const at::Tensor& carry = at::Tensor(),
                                   bool allow_double_buffer = true);

// Chunk outputs o_c = qg_c @ S_c + aqk_c @ vnew_c written to token rows
// t = chunk_start[c] + i (i < chunk_valid[c]) of out [L, H*dv]. With z
// defined, the gated RMSNorm norm(o) * w * silu(z) is applied and out has
// z's activation dtype (z: [L, H*dv] view); otherwise out is the raw fp32 o.
// seq_start: [L] int32 first token of each token's sequence.
void gdn_chunk_output(const GdnChunks& ch, const GdnRecurrence& rec, const at::Tensor& chunk_start,
                      const at::Tensor& chunk_valid, const at::Tensor& seq_start, int64_t chunk_begin,
                      const at::Tensor& z, const at::Tensor& w, double eps, at::Tensor& out);

// ------------------------------------------------------ weight compression
// Exponent histogram of a bf16 CUDA tensor (synchronizes).
std::array<int64_t, 256> weight_exponent_histogram(const at::Tensor& x);

// Compressed form of a bf16 CUDA tensor (numel % 64 == 0), see wemm/weight.h:
// exponents in [base+1, base+7] become codes 1..7, the rest are outliers.
struct CompressedWeight {
  at::Tensor planes, sm, exps, offs;
};
CompressedWeight compress_weight(const at::Tensor& x, int base);

// Decodes a compressed bf16 weight (wemm/weight.h) into out (bf16 or fp16,
// contiguous, numel = sm.numel()).
void decompress_weight(const at::Tensor& planes, const at::Tensor& sm, const at::Tensor& exps,
                       const at::Tensor& offs, int base, at::Tensor& out);

// Bytes of fp32 workspace one (head, chunk) pair needs in the GDN path; used
// to size chunk windows.
int64_t gdn_bytes_per_head_chunk(int64_t dk, int64_t dv);

}  // namespace wemm::kernels
