// Weight storage: dense, or losslessly compressed bf16 (see wemm/weight.h).

#include "wemm/weight.h"

#include <cstdlib>
#include <string>

#include "wemm/common.h"
#include "wemm/kernels.h"

namespace wemm {

namespace {
constexpr int kFp16OverflowExp = 143;  // bf16 biased exponent of 2^16: rounds to inf in fp16
}  // namespace

Weight Weight::load(const at::Tensor& host, at::ScalarType dtype, const at::Device& device, bool compress) {
  WEMM_CHECK(host.dim() == 2, "weight must be a matrix");
  at::Tensor src = host.contiguous().to(device);
  if (!compress || src.scalar_type() != at::kBFloat16 || src.numel() == 0 || src.numel() % 64 != 0) {
    at::Tensor t = src.to(dtype).contiguous();
    // fp16 has a narrower range than the bf16 checkpoint.
    if (dtype == at::kHalf) WEMM_CHECK(!t.isinf().any().item<bool>(), "a weight overflows fp16; use compute_dtype=bf16");
    return Weight(t);
  }
  // The 7 codes cover the densest window of exponents (compressed on the GPU).
  const auto hist = kernels::weight_exponent_histogram(src);
  if (dtype == at::kHalf)
    for (int e = kFp16OverflowExp; e < 256; ++e)
      WEMM_CHECK(hist[size_t(e)] == 0, "a weight overflows fp16 (or is inf/NaN); use compute_dtype=bf16");
  int base = 0;
  int64_t best = -1;
  for (int b = 0; b + 7 < 256; ++b) {
    int64_t cover = 0;
    for (int c = 1; c <= 7; ++c) cover += hist[size_t(b + c)];
    if (cover > best) best = cover, base = b;
  }
  auto c = kernels::compress_weight(src, base);
  Weight w;
  w.groups_ = std::move(c.groups);
  w.spill_ = std::move(c.spill);
  w.base_ = base;
  w.dtype_ = dtype;
  w.shape_ = {src.size(0), src.size(1)};
  return w;
}

bool Weight::small_gemm(const at::Tensor& x, const std::optional<at::Tensor>& bias, at::Tensor& out,
                        bool accumulate) const {
  static const bool enabled = [] {
    const char* e = std::getenv("WEMM_SKINNY_GEMM");
    return !(e && std::string(e) == "0");
  }();
  // Dense weights stay with cuBLASLt, which streams them at least as fast.
  if (!enabled || !compressed() || x.dim() != 2 || x.scalar_type() != dtype() || x.size(1) != size(1) ||
      !kernels::skinny_gemm_supported(x.size(0), size(0), size(1), x))
    return false;
  // otherwise fall back to the dense path (which reports bad arguments)
  auto ok_dtype = [&](const at::Tensor& t) {
    return t.scalar_type() == at::kFloat || t.scalar_type() == x.scalar_type();
  };
  if (!out.is_contiguous() || out.dim() != 2 || out.size(0) != x.size(0) || out.size(1) != size(0) ||
      out.device() != x.device() || !ok_dtype(out) ||
      (bias && (!bias->is_contiguous() || bias->numel() != size(0) || !ok_dtype(*bias))))
    return false;
  kernels::SkinnyWeight w;
  w.groups = groups_;
  w.spill = spill_;
  w.base = base_;
  kernels::skinny_gemm(x, w, bias ? *bias : at::Tensor(), out, accumulate);
  return true;
}

int64_t Weight::nbytes() const {
  if (!compressed()) return dense_.defined() ? int64_t(dense_.nbytes()) : 0;
  return int64_t(groups_.nbytes() + spill_.nbytes());
}

at::Tensor Weight::dense() const {
  if (!compressed()) return dense_;
  at::Tensor out = at::empty({shape_[0], shape_[1]}, groups_.options().dtype(dtype_));
  return dense_into(out);
}

const at::Tensor& Weight::dense_into(at::Tensor& out) const {
  if (!compressed()) return dense_;
  WEMM_CHECK(out.is_contiguous() && out.scalar_type() == dtype_ && out.dim() == 2 && out.size(0) == shape_[0] &&
                 out.size(1) == shape_[1] && out.device() == groups_.device(),
             "Weight::dense_into: bad output");
  kernels::decompress_weight(groups_, spill_, base_, out);
  return out;
}

}  // namespace wemm
