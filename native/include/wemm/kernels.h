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
