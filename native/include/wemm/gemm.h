#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace wemm {

enum class Epilogue { None, Gelu /* tanh approximation */ };

// y = x @ w^T (+ bias) (+ residual), via cuBLASLt with fused epilogues.
// x: [M, K] bf16/fp16 (row stride may exceed K), w: [N, K] same dtype,
// contiguous. The result has the residual's dtype if given, else `out_dtype`
// (default: x's dtype); bias must have the result's dtype. Accumulation and
// epilogue math are fp32 and the result is rounded once.
// Thread-safe; algorithms are autotuned per shape bucket (disable with
// WEMM_GEMM_AUTOTUNE=0).
at::Tensor gemm(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias = std::nullopt,
                const std::optional<at::Tensor>& residual = std::nullopt, Epilogue epilogue = Epilogue::None,
                std::optional<at::ScalarType> out_dtype = std::nullopt);

// In place: acc += x @ w^T (+ bias). acc: [M, N] fp32 or x's dtype,
// contiguous; bias must have acc's dtype. Saves the extra residual buffer.
void gemm_accumulate(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                     at::Tensor& acc);

// Strided-batched fp32 GEMM with true fp32 math (never TF32, regardless of
// TORCH_ALLOW_TF32_CUBLAS_OVERRIDE): out[b] = a[b] @ b[b] (+ out[b] if accumulate).
// a: [B, M, K], b: [B, K, N], out: [B, M, N], all contiguous fp32.
void gemm_f32_batched(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, bool accumulate);

}  // namespace wemm
