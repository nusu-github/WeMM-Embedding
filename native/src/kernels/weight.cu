// Compression / decompression of bf16 weights (format: wemm/weight.h).

#include "common.cuh"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

constexpr int kGroup = 64, kRecord = 96, kInline = 4;
constexpr int kPlanes = 64, kSpillOff = 88, kInlineOff = 92;  // byte offsets in a record

__device__ __forceinline__ int code_of(uint16_t v, int base) {
  const int c = int((v >> 7) & 0xff) - base;
  return c >= 1 && c <= 7 ? c : 0;
}

// Exponent of the outlier with rank r in its group.
__device__ __forceinline__ uint32_t outlier_exp(int r, uint32_t inl, int spill_off, const uint8_t* __restrict__ spill) {
  return r < kInline ? (inl >> 8 * r) & 0xff : spill[spill_off + r - kInline];
}

// 8 consecutive elements per thread (8 threads per group).
template <typename T>
__global__ void decompress_kernel(const uint8_t* __restrict__ groups, const uint8_t* __restrict__ spill, int base,
                                  T* __restrict__ out, int64_t n8) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const uint8_t* rec = groups + (i >> 3) * kRecord;
  const int j0 = int(i & 7) * 8;
  const uint64_t* pl = reinterpret_cast<const uint64_t*>(rec + kPlanes);
  const uint64_t p0 = pl[0], p1 = pl[1], p2 = pl[2];
  const int spill_off = *reinterpret_cast<const int*>(rec + kSpillOff);
  const uint32_t inl = *reinterpret_cast<const uint32_t*>(rec + kInlineOff);
  const uint2 raw = *reinterpret_cast<const uint2*>(rec + j0);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  int r = __popcll(~(p0 | p1 | p2) & ((1ull << j0) - 1));  // outliers before element j0
  uint4 o;
  uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int s = j0 + j;
    const int code = int((p0 >> s) & 1) | int((p1 >> s) & 1) << 1 | int((p2 >> s) & 1) << 2;
    const uint32_t e = code ? uint32_t(base + code) : outlier_exp(r++, inl, spill_off, spill);
    const uint16_t bits = uint16_t((b[j] & 0x80) << 8 | e << 7 | (b[j] & 0x7f));
    if constexpr (std::is_same_v<T, bf16>) {
      oh[j] = bits;
    } else {  // bf16 -> fp16, round to nearest even (as the dense load path)
      const fp16 hv = __float2half_rn(__uint_as_float(uint32_t(bits) << 16));
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

// 8 threads per group (8 elements each). Pass 0: sign/mantissa bytes,
// bit-planes, inline outliers and spill counts; pass 1: spill offsets and
// spilled outliers.
template <int kPass>
__global__ void compress_kernel(const uint16_t* __restrict__ x, int64_t n8, int base, uint8_t* __restrict__ groups,
                                int* __restrict__ counts, const int64_t* __restrict__ offs,
                                uint8_t* __restrict__ spill) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // whole 8-lane groups stay together: blocks are multiples of 8 threads
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
  uint8_t* rec = groups + g * kRecord;
  const uint64_t outl = ~(p[0] | p[1] | p[2]);
  int r = __popcll(outl & ((1ull << j0) - 1));
  if constexpr (kPass == 0) {
    uint2 o;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ob[j] = uint8_t(((v[j] >> 8) & 0x80) | (v[j] & 0x7f));
      if (code_of(v[j], base) == 0) {
        if (r < kInline) rec[kInlineOff + r] = uint8_t((v[j] >> 7) & 0xff);
        ++r;
      }
    }
    *reinterpret_cast<uint2*>(rec + j0) = o;
    if (j0 == 0) {
      uint64_t* pl = reinterpret_cast<uint64_t*>(rec + kPlanes);
#pragma unroll
      for (int b = 0; b < 3; ++b) pl[b] = p[b];
      counts[g] = max(0, __popcll(outl) - kInline);
    }
  } else {
    const int64_t off = offs[g];
    if (j0 == 0) *reinterpret_cast<int*>(rec + kSpillOff) = int(off);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (code_of(v[j], base) == 0) {
        if (r >= kInline) spill[off + r - kInline] = uint8_t((v[j] >> 7) & 0xff);
        ++r;
      }
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
  WEMM_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.is_cuda() && x.numel() > 0 &&
                 x.numel() % kGroup == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
             "compress_weight: aligned bf16 CUDA input with numel % 64 == 0");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t n = x.numel(), G = n / kGroup;
  auto bytes = x.options().dtype(at::kByte);
  CompressedWeight r;
  r.groups = at::zeros({G, kRecord}, bytes);
  at::Tensor counts = at::empty({G}, x.options().dtype(at::kInt));
  const int block = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  uint8_t* gp = r.groups.data_ptr<uint8_t>();
  compress_kernel<0><<<blocks_for(n / 8, block), block, 0, stream>>>(xp, n / 8, base, gp, counts.data_ptr<int>(),
                                                                     nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor incl = counts.cumsum(0, at::kLong);
  const int64_t total = incl[G - 1].item<int64_t>();
  WEMM_CHECK(total <= INT32_MAX, "compress_weight: too many outliers");
  at::Tensor offs = incl - counts;  // exclusive prefix
  r.spill = at::zeros({std::max<int64_t>(total, 1)}, bytes);
  compress_kernel<1><<<blocks_for(n / 8, block), block, 0, stream>>>(xp, n / 8, base, gp, nullptr,
                                                                     offs.data_ptr<int64_t>(), r.spill.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return r;
}

void decompress_weight(const at::Tensor& groups, const at::Tensor& spill, int base, at::Tensor& out) {
  const int64_t n = groups.numel() / kRecord * kGroup;
  WEMM_CHECK(groups.dim() == 2 && groups.size(1) == kRecord && out.numel() == n && out.is_contiguous() && is_act(out),
             "decompress_weight: bad arguments");
  if (n == 0) return;
  c10::cuda::CUDAGuard guard(out.device());
  const int block = 256;
  dispatch_act(out.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    decompress_kernel<T><<<blocks_for(n / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        groups.data_ptr<uint8_t>(), spill.data_ptr<uint8_t>(), base, ptr<T>(out), n / 8);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace wemm::kernels
