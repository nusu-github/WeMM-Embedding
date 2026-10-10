#pragma once

// GEMM weight matrices, optionally stored with lossless compression.
//
// Compressed format (bf16 checkpoints; inspired by ZipServ's fixed-length
// exponent coding, arXiv:2603.17435): the exponents of LLM weights cluster in
// a narrow range, so each element keeps its sign + 7 mantissa bits as one
// byte and its 8-bit exponent becomes a 3-bit code: 1..7 = base + code,
// 0 = outlier whose exponent is stored separately. Codes are stored as three
// 64-bit bit-planes per group of 64 elements, so decoding is branch-light and
// fully parallel. ~11.7 bits per weight (~27% smaller), and decompression
// reproduces the dense weight bit for bit.

#include <ATen/ATen.h>

#include <array>
#include <cstdint>

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

  bool defined() const { return dense_.defined() || planes_.defined(); }
  bool compressed() const { return planes_.defined(); }
  int64_t size(int64_t d) const { return compressed() ? shape_[size_t(d)] : dense_.size(d); }
  at::ScalarType dtype() const { return compressed() ? dtype_ : dense_.scalar_type(); }
  // Device bytes held by this weight.
  int64_t nbytes() const;
  // The dense [N, K] matrix: the stored tensor, or a freshly decompressed one.
  at::Tensor dense() const;
  // Same as dense(), but decompresses into `out` ([N, K], dtype()) when compressed.
  const at::Tensor& dense_into(at::Tensor& out) const;

 private:
  at::Tensor dense_;
  // compressed representation (all on the device)
  at::Tensor planes_;  // int64 [G, 3] code bit-planes per 64-element group
  at::Tensor sm_;      // uint8 [N*K] sign << 7 | mantissa
  at::Tensor exps_;    // uint8 [outliers] exponents of code-0 elements, in order
  at::Tensor offs_;    // int32 [G] outliers before each group
  int base_ = 0;
  at::ScalarType dtype_ = at::kHalf;
  std::array<int64_t, 2> shape_{};
};

}  // namespace wemm
