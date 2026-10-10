#pragma once

#include <ATen/ATen.h>

#include <optional>

#include "wemm/weight.h"

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

// Same with a Weight (decompressed into a temporary when compressed).
at::Tensor gemm(const at::Tensor& x, const Weight& w, const std::optional<at::Tensor>& bias = std::nullopt,
                const std::optional<at::Tensor>& residual = std::nullopt, Epilogue epilogue = Epilogue::None,
                std::optional<at::ScalarType> out_dtype = std::nullopt);
void gemm_accumulate(const at::Tensor& x, const Weight& w, const std::optional<at::Tensor>& bias, at::Tensor& acc);

}  // namespace wemm
