#pragma once

// Helpers shared by model.cpp / model_fused.cpp / model_reference.cpp.

#include <ATen/ATen.h>
#include <nvtx3/nvToolsExt.h>

#include <vector>

namespace wemm::detail {

constexpr int64_t kChunk = 64;  // gated-delta-rule chunk length

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

inline at::Tensor to_device_i32(const std::vector<int32_t>& v, const at::Device& d) {
  return at::tensor(at::ArrayRef<int32_t>(v), at::kInt).to(d, /*non_blocking=*/true);
}
inline at::Tensor to_device_i64(const std::vector<int64_t>& v, const at::Device& d) {
  return at::tensor(at::ArrayRef<int64_t>(v), at::kLong).to(d, /*non_blocking=*/true);
}

}  // namespace wemm::detail
