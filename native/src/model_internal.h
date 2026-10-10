#pragma once

// Helpers shared by model.cpp / model_fused.cpp / model_reference.cpp.

#include <ATen/ATen.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace wemm::detail {

constexpr int64_t kChunk = 64;  // gated-delta-rule chunk length

struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// Copies host arrays to the device through one pinned staging buffer and a
// single asynchronous copy. Pageable host-to-device copies synchronize the
// stream first, which would drain the GPU queue in the middle of a forward.
class Upload {
 public:
  template <typename T>
  Upload& add(const T* p, int64_t n) {
    items_.push_back({p, n, c10::CppTypeToScalarType<T>::value, bytes_});
    bytes_ += (n * int64_t(sizeof(T)) + 15) / 16 * 16;
    return *this;
  }
  template <typename T>
  Upload& add(const std::vector<T>& v) {
    return add(v.data(), int64_t(v.size()));
  }
  // Device tensors (1-D, in the order added) once all copies are queued.
  std::vector<at::Tensor> to(const at::Device& d) const {
    at::Tensor host = at::empty({std::max<int64_t>(bytes_, 16)}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    uint8_t* hp = host.data_ptr<uint8_t>();
    for (const auto& it : items_) std::memcpy(hp + it.off, it.p, size_t(it.n) * c10::elementSize(it.t));
    at::Tensor dev = host.to(d, /*non_blocking=*/true);
    std::vector<at::Tensor> out;
    out.reserve(items_.size());
    for (const auto& it : items_)
      out.push_back(dev.narrow(0, it.off, it.n * int64_t(c10::elementSize(it.t))).view(it.t));
    return out;
  }

 private:
  struct Item {
    const void* p;
    int64_t n;
    at::ScalarType t;
    int64_t off;
  };
  std::vector<Item> items_;
  int64_t bytes_ = 0;
};

}  // namespace wemm::detail
