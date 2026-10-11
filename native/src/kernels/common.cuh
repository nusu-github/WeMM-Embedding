#pragma once

// Shared device helpers for the kernels (bf16 / fp16 activations, fp32 math).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <mutex>
#include <set>
#include <utility>

#include "wemm/common.h"

namespace wemm::kernels::detail {

using bf16 = __nv_bfloat16;
using fp16 = __half;

// ----------------------------------------------------------- conversions
__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(fp16 x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T from_f(float x);
template <>
__device__ __forceinline__ float from_f<float>(float x) { return x; }
template <>
__device__ __forceinline__ bf16 from_f<bf16>(float x) { return __float2bfloat16(x); }
template <>
__device__ __forceinline__ fp16 from_f<fp16>(float x) { return __float2half(x); }

// 8 consecutive elements <-> fp32 (16-byte aligned for 16-bit types).
template <typename T>
__device__ __forceinline__ void load8(const T* p, float* f) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const T* h = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) f[i] = to_f(h[i]);
}
__device__ __forceinline__ void load8(const float* p, float* f) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
  f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
template <typename T>
__device__ __forceinline__ void store8(T* p, const float* f) {
  uint4 raw;
  T* h = reinterpret_cast<T*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = from_f<T>(f[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}
__device__ __forceinline__ void store8(float* p, const float* f) {
  *reinterpret_cast<float4*>(p) = make_float4(f[0], f[1], f[2], f[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(f[4], f[5], f[6], f[7]);
}
// 4 consecutive 16-bit elements (8-byte aligned).
template <typename T>
__device__ __forceinline__ void load4(const T* p, float* f) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const T* h = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) f[i] = to_f(h[i]);
}
template <typename T>
__device__ __forceinline__ void store4(T* p, const float* f) {
  uint2 raw;
  T* h = reinterpret_cast<T*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = from_f<T>(f[i]);
  *reinterpret_cast<uint2*>(p) = raw;
}

// ------------------------------------------------------------- reductions
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float red[32];
  __syncthreads();
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  return red[0];
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }

// ---------------------------------------------------------------- cp.async
// 16-byte global -> shared copies; synchronous on pre-Ampere GPUs.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
#else
  *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// ------------------------------------------------------------------- host
inline unsigned blocks_for(int64_t threads_total, int block) {
  return unsigned((threads_total + block - 1) / block);
}

// Runs `f` with T = bf16 or fp16 matching `dtype`.
template <typename F>
void dispatch_act(at::ScalarType dtype, F&& f) {
  switch (dtype) {
    case at::kBFloat16:
      f(bf16{});
      break;
    case at::kHalf:
      f(fp16{});
      break;
    default:
      throw Error(std::string("unsupported activation dtype ") + c10::toString(dtype));
  }
}

template <typename T>
T* ptr(const at::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

inline bool is_act(const at::Tensor& t) {
  return t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf;
}

// Opt-in shared memory limit of the current device.
inline int max_smem_optin() {
  int dev = 0, v = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  return v;
}

inline bool device_has_cp_async() {
  return at::cuda::getCurrentDeviceProperties()->major >= 8;
}

// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) once per (kernel, device).
template <typename K>
void ensure_smem(K kernel, size_t bytes) {
  static std::mutex mu;
  static std::set<std::pair<const void*, int>> done;
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  std::lock_guard<std::mutex> lock(mu);
  if (done.insert({reinterpret_cast<const void*>(kernel), dev}).second) {
    WEMM_CHECK(int(bytes) <= max_smem_optin(), "kernel needs ", bytes, " bytes of shared memory, device allows ",
               max_smem_optin());
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  }
}

}  // namespace wemm::kernels::detail
