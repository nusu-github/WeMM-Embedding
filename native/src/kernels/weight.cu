// Decompression of losslessly compressed bf16 weights (see wemm/weight.h).

#include "common.cuh"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

// 8 consecutive elements per thread (8 threads per 64-element group).
template <typename T>
__global__ void decompress_kernel(const uint64_t* __restrict__ planes, const uint8_t* __restrict__ sm,
                                  const uint8_t* __restrict__ exps, const int* __restrict__ offs, int base,
                                  T* __restrict__ out, int64_t n8) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const int64_t g = i >> 3;
  const int j0 = int(i & 7) * 8;
  const uint64_t p0 = planes[g * 3], p1 = planes[g * 3 + 1], p2 = planes[g * 3 + 2];
  const uint2 raw = *reinterpret_cast<const uint2*>(sm + i * 8);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  // outliers of this group before element j0
  int k = offs[g] + __popcll(~(p0 | p1 | p2) & ((1ull << j0) - 1));
  uint16_t bits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int s = j0 + j;
    const int code = int((p0 >> s) & 1) | int((p1 >> s) & 1) << 1 | int((p2 >> s) & 1) << 2;
    const int e = code ? base + code : exps[k++];
    bits[j] = uint16_t((b[j] & 0x80) << 8 | e << 7 | (b[j] & 0x7f));
  }
  uint4 o;
  uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if constexpr (std::is_same_v<T, bf16>) {
      oh[j] = bits[j];
    } else {  // bf16 -> fp16, round to nearest even (as the dense load path)
      const fp16 hv = __float2half_rn(__uint_as_float(uint32_t(bits[j]) << 16));
      oh[j] = *reinterpret_cast<const uint16_t*>(&hv);
    }
  }
  reinterpret_cast<uint4*>(out)[i] = o;
}

// Exponent histogram (bf16 bits >> 7 & 0xff) of x.
__global__ void exp_hist_kernel(const uint16_t* __restrict__ x, int64_t n, unsigned long long* __restrict__ hist) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    atomicAdd(&h[(x[i] >> 7) & 0xff], 1u);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    if (h[i]) atomicAdd(&hist[i], (unsigned long long)h[i]);
}

__device__ __forceinline__ int code_of(uint16_t v, int base) {
  const int c = int((v >> 7) & 0xff) - base;
  return c >= 1 && c <= 7 ? c : 0;
}

// 8 threads per 64-element group (8 elements each). Pass 0: bit-planes,
// sign/mantissa bytes and outlier counts; pass 1: outlier exponents.
template <int kPass>
__global__ void compress_kernel(const uint16_t* __restrict__ x, int64_t n8, int base, uint64_t* __restrict__ planes,
                                uint8_t* __restrict__ sm, int* __restrict__ counts, const int64_t* __restrict__ offs,
                                uint8_t* __restrict__ exps) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // whole 8-lane groups stay together: n8 is a multiple of 8 and blocks are multiples of 8 threads
  const bool live = i < n8;
  const int64_t g = i >> 3;
  const int j0 = int(i & 7) * 8;
  uint16_t v[8] = {};
  if (live) {
    const uint4 raw = *reinterpret_cast<const uint4*>(x + i * 8);
    const uint16_t* r = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = r[j];
  }
  uint64_t p[3] = {0, 0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int c = code_of(v[j], base);
#pragma unroll
    for (int b = 0; b < 3; ++b) p[b] |= uint64_t((c >> b) & 1) << (j0 + j);
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1)
#pragma unroll
    for (int b = 0; b < 3; ++b) p[b] |= __shfl_xor_sync(0xffffffffu, p[b], o);
  if (!live) return;
  const uint64_t outl = ~(p[0] | p[1] | p[2]);
  if constexpr (kPass == 0) {
    uint2 o;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) ob[j] = uint8_t(((v[j] >> 8) & 0x80) | (v[j] & 0x7f));
    reinterpret_cast<uint2*>(sm)[i] = o;
    if ((i & 7) == 0) {
#pragma unroll
      for (int b = 0; b < 3; ++b) planes[g * 3 + b] = p[b];
      counts[g] = __popcll(outl);
    }
  } else {
    int64_t k = offs[g] + __popcll(outl & ((1ull << j0) - 1));
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (code_of(v[j], base) == 0) exps[k++] = uint8_t((v[j] >> 7) & 0xff);
  }
}

}  // namespace

std::array<int64_t, 256> weight_exponent_histogram(const at::Tensor& x) {
  WEMM_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.is_cuda(), "histogram: bf16 CUDA input");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor h = at::zeros({256}, x.options().dtype(at::kLong));
  const int64_t n = x.numel();
  if (n > 0) {
    exp_hist_kernel<<<unsigned(std::min<int64_t>(1024, (n + 255) / 256)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr()), n,
        reinterpret_cast<unsigned long long*>(h.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  at::Tensor hc = h.cpu();
  std::array<int64_t, 256> out{};
  std::copy_n(hc.data_ptr<int64_t>(), 256, out.begin());
  return out;
}

CompressedWeight compress_weight(const at::Tensor& x, int base) {
  WEMM_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.is_cuda() && x.numel() % 64 == 0,
             "compress_weight: bf16 CUDA input with numel % 64 == 0");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t n = x.numel(), G = n / 64;
  auto bytes = x.options().dtype(at::kByte);
  CompressedWeight r;
  r.planes = at::empty({G, 3}, x.options().dtype(at::kLong));
  r.sm = at::empty({n}, bytes);
  at::Tensor counts = at::empty({G}, x.options().dtype(at::kInt));
  const int block = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  auto* pp = reinterpret_cast<uint64_t*>(r.planes.data_ptr<int64_t>());
  compress_kernel<0><<<blocks_for(n / 8, block), block, 0, stream>>>(xp, n / 8, base, pp, r.sm.data_ptr<uint8_t>(),
                                                                     counts.data_ptr<int>(), nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor incl = counts.cumsum(0, at::kLong);
  const int64_t total = incl[G - 1].item<int64_t>();
  WEMM_CHECK(total <= INT32_MAX, "compress_weight: too many outliers");
  at::Tensor offs = incl - counts;  // exclusive prefix (int64)
  r.exps = at::empty({std::max<int64_t>(total, 1)}, bytes);
  compress_kernel<1><<<blocks_for(n / 8, block), block, 0, stream>>>(xp, n / 8, base, nullptr, nullptr, nullptr,
                                                                     offs.data_ptr<int64_t>(), r.exps.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  r.offs = offs.to(at::kInt);
  return r;
}

void decompress_weight(const at::Tensor& planes, const at::Tensor& sm, const at::Tensor& exps,
                       const at::Tensor& offs, int base, at::Tensor& out) {
  const int64_t n = sm.numel();
  WEMM_CHECK(n % 64 == 0 && planes.numel() == n / 64 * 3 && offs.numel() == n / 64 && out.numel() == n &&
                 out.is_contiguous() && is_act(out),
             "decompress_weight: bad arguments");
  if (n == 0) return;
  c10::cuda::CUDAGuard guard(out.device());
  const int block = 256;
  dispatch_act(out.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    decompress_kernel<T><<<blocks_for(n / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint64_t*>(planes.data_ptr<int64_t>()), sm.data_ptr<uint8_t>(),
        exps.data_ptr<uint8_t>(), offs.data_ptr<int>(), base, ptr<T>(out), n / 8);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace wemm::kernels
