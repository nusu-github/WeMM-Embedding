#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "wemm/common.h"
#include "wemm/kernels.h"

namespace wemm::kernels {

namespace {

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// 8 channels per thread.
template <int K>
__global__ void conv1d_silu_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                                   const int* __restrict__ seq_start, __nv_bfloat16* __restrict__ out, int L, int C) {
  const int groups = C / 8;
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= int64_t(L) * groups) return;
  const int t = int(gid / groups);
  const int c = int(gid % groups) * 8;
  const int s0 = seq_start[t];
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int tt = t - (K - 1) + k;
    if (tt < s0) continue;
    const uint4 raw = *reinterpret_cast<const uint4*>(x + int64_t(tt) * ldx + c);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += __bfloat162float(v[j]) * w[(c + j) * K + k];
  }
  __align__(16) __nv_bfloat16 o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(acc[j] / (1.f + expf(-acc[j])));
  *reinterpret_cast<uint4*>(out + int64_t(t) * C + c) = *reinterpret_cast<const uint4*>(o);
}

}  // namespace

at::Tensor causal_conv1d_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& seq_start) {
  WEMM_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "conv1d: x must be bf16 with unit column stride");
  WEMM_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous(), "conv1d: w must be contiguous fp32");
  const int L = int(x.size(0)), C = int(x.size(1)), K = int(w.size(1));
  WEMM_CHECK(C % 8 == 0 && x.stride(0) % 8 == 0, "conv1d: channels must be a multiple of 8");
  at::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({L, C}, x.options());
  if (L == 0) return out;
  const int64_t total = int64_t(L) * (C / 8);
  const int threads = 256;
  const unsigned blocks = unsigned((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  switch (K) {
    case 4:
      conv1d_silu_kernel<4><<<blocks, threads, 0, stream>>>(xp, x.stride(0), w.data_ptr<float>(),
                                                            seq_start.data_ptr<int>(), op, L, C);
      break;
    default:
      throw Error("conv1d: unsupported kernel size " + std::to_string(K));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace wemm::kernels
