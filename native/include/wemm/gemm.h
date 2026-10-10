#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace wemm {

enum class Epilogue { None, Gelu /* tanh approximation */ };

// y = x @ w^T (+ bias) (+ residual), via cuBLASLt with fused epilogues.
// x: [M, K] bf16 (row stride may exceed K), w: [N, K] bf16 contiguous,
// bias: [N] bf16, residual: [M, N] bf16 or fp32 contiguous. Accumulation and
// epilogue math are fp32; the result has the residual's dtype (else
// `out_dtype`) and is rounded once.
at::Tensor gemm(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias = std::nullopt,
                const std::optional<at::Tensor>& residual = std::nullopt, Epilogue epilogue = Epilogue::None,
                at::ScalarType out_dtype = at::kBFloat16);

// Strided-batched fp32 GEMM with true fp32 math (never TF32, regardless of
// TORCH_ALLOW_TF32_CUBLAS_OVERRIDE): out[b] = a[b] @ b[b] (+ out[b] if accumulate).
// a: [B, M, K], b: [B, K, N], out: [B, M, N], all contiguous fp32.
void gemm_f32_batched(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, bool accumulate);

}  // namespace wemm
