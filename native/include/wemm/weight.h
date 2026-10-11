#pragma once

// GEMM weight matrices, optionally stored with lossless compression.
//
// Compressed format (bf16 checkpoints; inspired by ZipServ's fixed-length
// exponent coding, arXiv:2603.17435): the exponents of LLM weights cluster in
// a narrow range, so each element keeps its sign + 7 mantissa bits as one
// byte and its 8-bit exponent becomes a 3-bit code: 1..7 = base + code,
// 0 = outlier whose exponent is stored separately. Every group of 64
// consecutive elements is one 96-byte record (12 bits per weight, 25%
// smaller than bf16/fp16):
//   [0, 64)   sign/mantissa bytes
//   [64, 88)  three 64-bit code bit-planes
//   [88, 92)  int32: outliers spilled by the previous groups
//   [92, 96)  exponents of the group's first 4 outliers
// Further outliers (rare) go to a separate spill array. Records are read as
// one contiguous stream and decode with a few bit operations per element,
// so short-input GEMMs decode straight into registers; decompression
// reproduces the dense weight bit for bit.

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <optional>

namespace wemm {

class Weight {
 public:
  Weight() = default;
  // Dense weight already in its final dtype and device.
  explicit Weight(at::Tensor dense) : dense_(std::move(dense)) {}
  // From a host bf16 matrix [N, K]: compressed when `compress` and the data
  // allows it (bf16, numel % 64 == 0), dense otherwise. `dtype` is the dtype
  // the weight is used in (bf16 or fp16); fp16 overflow is rejected.
  static Weight load(const at::Tensor& host, at::ScalarType dtype, const at::Device& device, bool compress);

  bool defined() const { return dense_.defined() || groups_.defined(); }
  bool compressed() const { return groups_.defined(); }
  int64_t size(int64_t d) const { return compressed() ? shape_[size_t(d)] : dense_.size(d); }
  at::ScalarType dtype() const { return compressed() ? dtype_ : dense_.scalar_type(); }
  // Device bytes held by this weight.
  int64_t nbytes() const;
  // The dense [N, K] matrix: the stored tensor, or a freshly decompressed one.
  at::Tensor dense() const;
  // Same as dense(), but decompresses into `out` ([N, K], dtype()) when compressed.
  const at::Tensor& dense_into(at::Tensor& out) const;
  // Compressed weights: out (+)= x @ W^T (+ bias) with the few-row kernel
  // that decodes the weight on the fly (no dense temporary). False when it
  // does not apply (dense weight, M > 64, sm < 80, shapes, or disabled with
  // WEMM_SKINNY_GEMM=0) and nothing was done.
  bool small_gemm(const at::Tensor& x, const std::optional<at::Tensor>& bias, at::Tensor& out,
                  bool accumulate) const;

 private:
  at::Tensor dense_;
  // compressed representation (all on the device)
  at::Tensor groups_;  // uint8 [N*K/64, 96] records
  at::Tensor spill_;   // uint8 exponents of each group's 5th and later outliers
  int base_ = 0;
  at::ScalarType dtype_ = at::kHalf;
  std::array<int64_t, 2> shape_{};
};

}  // namespace wemm
