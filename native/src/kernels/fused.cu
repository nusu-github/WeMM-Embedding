#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "wemm/common.h"
#include "wemm/kernels.h"

namespace wemm::kernels {

namespace {

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

namespace wemm::kernels {

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void load8(const bf16* p, float* f) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 v = __bfloat1622float2(h[i]);
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

__device__ __forceinline__ void store8(bf16* p, const float* f) {
  uint4 raw;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ void load8(const float* p, float* f) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
  f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

__device__ float block_sum(float v) {
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

// ------------------------------------------------------------------ rmsnorm
template <typename T, bool OUT_F32>
__global__ void rmsnorm_kernel(const T* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                               void* __restrict__ out, int D, float eps) {
  const int64_t row = blockIdx.x;
  const T* xr = x + row * ldx;
  float ss = 0.f;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    float f[8];
    load8(xr + i, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
  }
  const float r = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    float f[8];
    load8(xr + i, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = f[j] * r * w[i + j];
    if constexpr (OUT_F32) {
      float* o = static_cast<float*>(out) + row * D + i;
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = f[j];
    } else {
      store8(static_cast<bf16*>(out) + row * D + i, f);
    }
  }
}

// ---------------------------------------------------------------- layernorm
template <typename T>
__global__ void layernorm_kernel(const T* __restrict__ x, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, bf16* __restrict__ out, int D, float eps) {
  const int64_t row = blockIdx.x;
  const T* xr = x + row * D;
  float s = 0.f;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    float f[8];
    load8(xr + i, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) s += f[j];
  }
  const float mean = block_sum(s) / D;
  float v = 0.f;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    float f[8];
    load8(xr + i, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) v += (f[j] - mean) * (f[j] - mean);
  }
  const float r = rsqrtf(block_sum(v) / D + eps);
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    float f[8];
    load8(xr + i, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = (f[j] - mean) * r * gamma[i + j] + beta[i + j];
    store8(out + row * D + i, f);
  }
}

// ------------------------------------------------------ head rmsnorm + rope
// One warp per (row, head); VPT = hd / 32 consecutive elements per lane.
template <int VPT>
__global__ void head_norm_rope_kernel(const bf16* __restrict__ x, int64_t s_row, int64_t s_head,
                                      const float* __restrict__ w, const float* __restrict__ cos,
                                      const float* __restrict__ sin, int rot, const int64_t* __restrict__ rows,
                                      bf16* __restrict__ out, int N, int H, float eps) {
  const int64_t gw = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= int64_t(N) * H) return;
  const int n = int(gw / H), h = int(gw % H);
  const int64_t src = rows ? rows[n] : n;
  const int hd = VPT * 32;
  float f[VPT];
  const bf16* xp = x + src * s_row + h * s_head + lane * VPT;
#pragma unroll
  for (int j = 0; j < VPT; j += 8) load8(xp + j, f + j);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) ss += f[j] * f[j];
  const float r = rsqrtf(warp_sum(ss) / hd + eps);
#pragma unroll
  for (int j = 0; j < VPT; ++j) f[j] *= r * w[lane * VPT + j];
  // rotate_half partner lives rot/2 elements away -> (rot/2)/VPT lanes away.
  const int half = rot / 2;
  const int lane_off = half / VPT;
  float g[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) g[j] = __shfl_xor_sync(0xffffffffu, f[j], lane_off);
  const int e0 = lane * VPT;
  if (e0 < rot) {
    const float* c = cos + src * rot;
    const float* s = sin + src * rot;
    const bool first = e0 < half;
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
      const int e = e0 + j;
      f[j] = first ? f[j] * c[e] - g[j] * s[e] : f[j] * c[e] + g[j] * s[e];
    }
  }
  bf16* op = out + (int64_t(n) * H + h) * hd + lane * VPT;
#pragma unroll
  for (int j = 0; j < VPT; j += 8) store8(op + j, f + j);
}

// --------------------------------------------------------- sigmoid gating
__global__ void gate_mul_kernel(const bf16* __restrict__ o, const bf16* __restrict__ gate, int64_t g_row,
                                int64_t g_head, const int64_t* __restrict__ rows, bf16* __restrict__ out, int N,
                                int H, int hd) {
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int per_row = H * hd / 8;
  if (gid >= int64_t(N) * per_row) return;
  const int n = int(gid / per_row), e = int(gid % per_row) * 8;
  const int h = e / hd, d = e % hd;
  const int64_t src = rows ? rows[n] : n;
  float a[8], g[8];
  load8(o + int64_t(n) * H * hd + e, a);
  load8(gate + src * g_row + h * g_head + d, g);
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] *= sigmoidf_(g[j]);
  store8(out + int64_t(n) * H * hd + e, a);
}

// --------------------------------------------------------------- silu*mul
__global__ void silu_mul_kernel(const bf16* __restrict__ gu, bf16* __restrict__ out, int64_t N, int I) {
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int per_row = I / 8;
  if (gid >= N * per_row) return;
  const int64_t n = gid / per_row;
  const int e = int(gid % per_row) * 8;
  float g[8], u[8];
  load8(gu + n * 2 * I + e, g);
  load8(gu + n * 2 * I + I + e, u);
#pragma unroll
  for (int j = 0; j < 8; ++j) g[j] = g[j] / (1.f + expf(-g[j])) * u[j];
  store8(out + n * I + e, g);
}

// ------------------------------------------------------ gated rmsnorm (GDN)
template <int VPT>
__global__ void gdn_gated_norm_kernel(const float* __restrict__ o, int64_t o_head_stride,
                                      const int* __restrict__ padded_row, const bf16* __restrict__ z, int64_t ldz,
                                      const float* __restrict__ w, bf16* __restrict__ out, int L, int H, float eps) {
  const int64_t gw = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= int64_t(L) * H) return;
  const int t = int(gw / H), h = int(gw % H);
  constexpr int dv = VPT * 32;
  const float* op = o + h * o_head_stride + int64_t(padded_row[t]) * dv + lane * VPT;
  float f[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) f[j] = op[j];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) ss += f[j] * f[j];
  const float r = rsqrtf(warp_sum(ss) / dv + eps);
  const bf16* zp = z + int64_t(t) * ldz + h * dv + lane * VPT;
  bf16* outp = out + (int64_t(t) * H + h) * dv + lane * VPT;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const float zz = __bfloat162float(zp[j]);
    outp[j] = __float2bfloat16(f[j] * r * w[lane * VPT + j] * (zz / (1.f + expf(-zz))));
  }
}

// ------------------------------------------------------------ vision rope
template <int VPT>
__global__ void vision_rope_kernel(const bf16* __restrict__ qkv, const float* __restrict__ cos,
                                   const float* __restrict__ sin, bf16* __restrict__ q_out, bf16* __restrict__ k_out,
                                   int P, int nh) {
  const int64_t gw = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= int64_t(P) * nh * 2) return;
  const int which = int(gw % 2);  // 0 = q, 1 = k
  const int64_t ph = gw / 2;
  const int p = int(ph / nh), h = int(ph % nh);
  constexpr int hd = VPT * 32;
  const bf16* src = qkv + int64_t(p) * 3 * nh * hd + int64_t(which) * nh * hd + h * hd + lane * VPT;
  float f[VPT], g[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) f[j] = __bfloat162float(src[j]);
#pragma unroll
  for (int j = 0; j < VPT; ++j) g[j] = __shfl_xor_sync(0xffffffffu, f[j], 16);
  const float* c = cos + int64_t(p) * hd + lane * VPT;
  const float* s = sin + int64_t(p) * hd + lane * VPT;
  bf16* dst = (which ? k_out : q_out) + (int64_t(p) * nh + h) * hd + lane * VPT;
  const bool first = lane < 16;
#pragma unroll
  for (int j = 0; j < VPT; ++j) dst[j] = __float2bfloat16(first ? f[j] * c[j] - g[j] * s[j] : f[j] * c[j] + g[j] * s[j]);
}

inline unsigned blocks_for(int64_t threads_total, int block) { return unsigned((threads_total + block - 1) / block); }

}  // namespace

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, at::ScalarType out_dtype) {
  const bool in_f32 = x.scalar_type() == at::kFloat;
  WEMM_CHECK((in_f32 || x.scalar_type() == at::kBFloat16) && x.dim() == 2 && x.stride(1) == 1, "rmsnorm: bad input");
  const int64_t N = x.size(0), D = x.size(1);
  WEMM_CHECK(D % 8 == 0 && x.stride(0) % 8 == 0 && w.numel() == D, "rmsnorm: D must be a multiple of 8");
  at::Tensor out = at::empty({N, D}, x.options().dtype(out_dtype));
  if (N == 0) return out;
  const int threads = int(std::min<int64_t>(1024, std::max<int64_t>(32, ((D / 8 + 31) / 32) * 32)));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool of = out_dtype == at::kFloat;
#define WEMM_RMS(T, OF)                                                                                    \
  rmsnorm_kernel<T, OF><<<unsigned(N), threads, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()), x.stride(0), \
                                                             w.data_ptr<float>(), out.data_ptr(), int(D), float(eps))
  if (in_f32 && of) WEMM_RMS(float, true);
  else if (in_f32) WEMM_RMS(float, false);
  else if (of) WEMM_RMS(bf16, true);
  else WEMM_RMS(bf16, false);
#undef WEMM_RMS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor layernorm(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double eps) {
  WEMM_CHECK(x.is_contiguous() && x.dim() == 2 && x.size(1) % 8 == 0, "layernorm: bad input");
  const int64_t N = x.size(0), D = x.size(1);
  at::Tensor out = at::empty({N, D}, x.options().dtype(at::kBFloat16));
  if (N == 0) return out;
  const int threads = int(std::min<int64_t>(1024, std::max<int64_t>(32, ((D / 8 + 31) / 32) * 32)));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* op = reinterpret_cast<bf16*>(out.data_ptr());
  if (x.scalar_type() == at::kFloat)
    layernorm_kernel<float><<<unsigned(N), threads, 0, stream>>>(x.data_ptr<float>(), gamma.data_ptr<float>(),
                                                                 beta.data_ptr<float>(), op, int(D), float(eps));
  else
    layernorm_kernel<bf16><<<unsigned(N), threads, 0, stream>>>(reinterpret_cast<const bf16*>(x.data_ptr()),
                                                                gamma.data_ptr<float>(), beta.data_ptr<float>(), op,
                                                                int(D), float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor head_rmsnorm_rope(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cos, const at::Tensor& sin,
                             const at::Tensor& rows, double eps) {
  WEMM_CHECK(x.dim() == 3 && x.stride(2) == 1 && x.scalar_type() == at::kBFloat16, "head_rmsnorm_rope: bad x");
  const int64_t H = x.size(1), hd = x.size(2), rot = cos.size(1);
  const int64_t N = rows.defined() ? rows.numel() : x.size(0);
  WEMM_CHECK(hd == 256 || hd == 128, "head_rmsnorm_rope: head_dim must be 128 or 256");
  const int vpt = int(hd / 32);
  WEMM_CHECK(rot <= hd && (rot / 2) % vpt == 0 && rot / 2 / vpt <= 16, "head_rmsnorm_rope: unsupported rotary dim");
  WEMM_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0, "head_rmsnorm_rope: misaligned input");
  at::Tensor out = at::empty({N, H, hd}, x.options());
  if (N == 0) return out;
  const int block = 256;
  const unsigned grid = blocks_for(N * H * 32, block);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t* rp = rows.defined() ? rows.data_ptr<int64_t>() : nullptr;
  auto* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  auto* op = reinterpret_cast<bf16*>(out.data_ptr());
  if (vpt == 8)
    head_norm_rope_kernel<8><<<grid, block, 0, stream>>>(xp, x.stride(0), x.stride(1), w.data_ptr<float>(),
                                                         cos.data_ptr<float>(), sin.data_ptr<float>(), int(rot), rp,
                                                         op, int(N), int(H), float(eps));
  else
    head_norm_rope_kernel<4><<<grid, block, 0, stream>>>(xp, x.stride(0), x.stride(1), w.data_ptr<float>(),
                                                         cos.data_ptr<float>(), sin.data_ptr<float>(), int(rot), rp,
                                                         op, int(N), int(H), float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor sigmoid_gate_mul(const at::Tensor& o, const at::Tensor& gate, const at::Tensor& rows) {
  WEMM_CHECK(o.is_contiguous() && o.dim() == 3 && gate.dim() == 3 && gate.stride(2) == 1, "gate_mul: bad input");
  const int64_t N = o.size(0), H = o.size(1), hd = o.size(2);
  WEMM_CHECK(hd % 8 == 0 && gate.stride(0) % 8 == 0 && gate.stride(1) % 8 == 0, "gate_mul: misaligned");
  at::Tensor out = at::empty({N, H * hd}, o.options());
  if (N == 0) return out;
  const int block = 256;
  gate_mul_kernel<<<blocks_for(N * H * hd / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(o.data_ptr()), reinterpret_cast<const bf16*>(gate.data_ptr()), gate.stride(0),
      gate.stride(1), rows.defined() ? rows.data_ptr<int64_t>() : nullptr, reinterpret_cast<bf16*>(out.data_ptr()),
      int(N), int(H), int(hd));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor silu_mul(const at::Tensor& gu) {
  WEMM_CHECK(gu.is_contiguous() && gu.dim() == 2 && gu.size(1) % 16 == 0, "silu_mul: bad input");
  const int64_t N = gu.size(0), I = gu.size(1) / 2;
  at::Tensor out = at::empty({N, I}, gu.options());
  if (N == 0) return out;
  const int block = 256;
  silu_mul_kernel<<<blocks_for(N * I / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(gu.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()), N, int(I));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor gdn_gated_rmsnorm(const at::Tensor& o, const at::Tensor& padded_row, const at::Tensor& z,
                             const at::Tensor& w, double eps) {
  WEMM_CHECK(o.dim() == 3 && o.is_contiguous() && o.scalar_type() == at::kFloat, "gated_norm: bad o");
  const int64_t H = o.size(0), dv = o.size(2), L = padded_row.numel();
  WEMM_CHECK(dv == 128, "gated_norm: dv must be 128");
  WEMM_CHECK(z.stride(1) == 1 && z.size(1) == H * dv, "gated_norm: bad z");
  at::Tensor out = at::empty({L, H * dv}, z.options());
  if (L == 0) return out;
  const int block = 256;
  gdn_gated_norm_kernel<4><<<blocks_for(L * H * 32, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
      o.data_ptr<float>(), o.stride(0), padded_row.data_ptr<int>(), reinterpret_cast<const bf16*>(z.data_ptr()),
      z.stride(0), w.data_ptr<float>(), reinterpret_cast<bf16*>(out.data_ptr()), int(L), int(H), float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<at::Tensor, at::Tensor> vision_qk_rope(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                                                 int64_t nh) {
  WEMM_CHECK(qkv.is_contiguous() && qkv.scalar_type() == at::kBFloat16, "vision_rope: bad qkv");
  const int64_t P = qkv.size(0), hd = qkv.size(1) / 3 / nh;
  WEMM_CHECK(hd == 64 && cos.size(1) == hd && cos.is_contiguous() && sin.is_contiguous(),
             "vision_rope: head_dim must be 64");
  at::Tensor q = at::empty({P, nh, hd}, qkv.options()), k = at::empty({P, nh, hd}, qkv.options());
  if (P == 0) return {q, k};
  const int block = 256;
  vision_rope_kernel<2><<<blocks_for(P * nh * 2 * 32, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), cos.data_ptr<float>(), sin.data_ptr<float>(),
      reinterpret_cast<bf16*>(q.data_ptr()), reinterpret_cast<bf16*>(k.data_ptr()), int(P), int(nh));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k};
}

}  // namespace wemm::kernels
