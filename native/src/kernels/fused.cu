// Elementwise / normalization kernels.

#include "common.cuh"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

// ------------------------------------------------------------------ rmsnorm
template <typename Tin, typename Tout>
__global__ void rmsnorm_kernel(const Tin* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                               Tout* __restrict__ out, int D, float eps) {
  const int64_t row = blockIdx.x;
  const Tin* xr = x + row * ldx;
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
    store8(out + row * D + i, f);
  }
}

// ---------------------------------------------------------------- layernorm
template <typename Tin, typename Tout>
__global__ void layernorm_kernel(const Tin* __restrict__ x, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, Tout* __restrict__ out, int D, float eps) {
  const int64_t row = blockIdx.x;
  const Tin* xr = x + row * D;
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
template <typename T, int VPT>
__global__ void head_norm_rope_kernel(const T* __restrict__ x, int64_t s_row, int64_t s_head,
                                      const float* __restrict__ w, const float* __restrict__ cos,
                                      const float* __restrict__ sin, int rot, const int64_t* __restrict__ rows,
                                      T* __restrict__ out, int N, int H, float eps) {
  const int64_t gw = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= int64_t(N) * H) return;
  const int n = int(gw / H), h = int(gw % H);
  const int64_t src = rows ? rows[n] : n;
  constexpr int hd = VPT * 32;
  float f[VPT];
  const T* xp = x + src * s_row + h * s_head + lane * VPT;
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
  float g[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) g[j] = __shfl_xor_sync(0xffffffffu, f[j], half / VPT);
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
  T* op = out + (int64_t(n) * H + h) * hd + lane * VPT;
#pragma unroll
  for (int j = 0; j < VPT; j += 8) store8(op + j, f + j);
}

// --------------------------------------------------------- sigmoid gating
template <typename T>
__global__ void gate_mul_kernel(const T* __restrict__ o, const T* __restrict__ gate, int64_t g_row, int64_t g_head,
                                const int64_t* __restrict__ rows, T* __restrict__ out, int N, int H, int hd) {
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
template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ gu, T* __restrict__ out, int64_t N, int I) {
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int per_row = I / 8;
  if (gid >= N * per_row) return;
  const int64_t n = gid / per_row;
  const int e = int(gid % per_row) * 8;
  float g[8], u[8];
  load8(gu + n * 2 * I + e, g);
  load8(gu + n * 2 * I + I + e, u);
#pragma unroll
  for (int j = 0; j < 8; ++j) g[j] = siluf_(g[j]) * u[j];
  store8(out + n * I + e, g);
}

// ------------------------------------------------------------ vision rope
// Fast path for head_dim 64: one warp per (patch, head, q|k), 2 elements per lane.
template <typename T>
__global__ void vision_rope64_kernel(const T* __restrict__ qkv, const float* __restrict__ cos,
                                     const float* __restrict__ sin, T* __restrict__ q_out, T* __restrict__ k_out, int P,
                                     int nh) {
  constexpr int VPT = 2, hd = 64;
  const int64_t gw = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= int64_t(P) * nh * 2) return;
  const int which = int(gw % 2);  // 0 = q, 1 = k
  const int64_t ph = gw / 2;
  const int p = int(ph / nh), h = int(ph % nh);
  const T* src = qkv + int64_t(p) * 3 * nh * hd + int64_t(which) * nh * hd + h * hd + lane * VPT;
  float f[VPT], g[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) f[j] = to_f(src[j]);
#pragma unroll
  for (int j = 0; j < VPT; ++j) g[j] = __shfl_xor_sync(0xffffffffu, f[j], 16);
  const float* c = cos + int64_t(p) * hd + lane * VPT;
  const float* s = sin + int64_t(p) * hd + lane * VPT;
  T* dst = (which ? k_out : q_out) + (int64_t(p) * nh + h) * hd + lane * VPT;
  const bool first = lane < 16;
#pragma unroll
  for (int j = 0; j < VPT; ++j) dst[j] = from_f<T>(first ? f[j] * c[j] - g[j] * s[j] : f[j] * c[j] + g[j] * s[j]);
}

// Any even head_dim (e.g. 72 for the 9B ViT): one thread per rotate_half pair.
template <typename T>
__global__ void vision_rope_generic_kernel(const T* __restrict__ qkv, const float* __restrict__ cos,
                                           const float* __restrict__ sin, T* __restrict__ q_out, T* __restrict__ k_out,
                                           int P, int nh, int hd) {
  const int half = hd / 2;
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= int64_t(P) * nh * 2 * half) return;
  const int i = int(gid % half);
  int64_t rest = gid / half;
  const int which = int(rest % 2);
  rest /= 2;
  const int h = int(rest % nh), p = int(rest / nh);
  const T* src = qkv + int64_t(p) * 3 * nh * hd + int64_t(which) * nh * hd + int64_t(h) * hd;
  const float x1 = to_f(src[i]), x2 = to_f(src[i + half]);
  const float* c = cos + int64_t(p) * hd;
  const float* s = sin + int64_t(p) * hd;
  T* dst = (which ? k_out : q_out) + (int64_t(p) * nh + h) * hd;
  dst[i] = from_f<T>(x1 * c[i] - x2 * s[i]);
  dst[i + half] = from_f<T>(x2 * c[i + half] + x1 * s[i + half]);
}

// ---------------------------------------------------------- causal conv1d
// Each thread computes 8 channels x kConvRows consecutive tokens, sliding a
// K-row window through registers (every input row is read once plus a K-1
// halo); w: [C, K], so a thread's 8 x K weights are contiguous.
constexpr int kConvRows = 8;
template <typename T, int K>
__global__ void conv1d_silu_kernel(const T* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                                   const int* __restrict__ seq_start, T* __restrict__ out, int L, int C) {
  const int groups = C / 8;
  const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= int64_t((L + kConvRows - 1) / kConvRows) * groups) return;
  const int t0 = int(gid / groups) * kConvRows;
  const int c = int(gid % groups) * 8;
  float wk[8][K];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const float4 v = *reinterpret_cast<const float4*>(w + (c + j) * K + k);
      wk[j][k] = v.x, wk[j][k + 1] = v.y, wk[j][k + 2] = v.z, wk[j][k + 3] = v.w;
    }
  float win[K][8];  // win[K-1] = current row, win[K-1-i] = row t-i
#pragma unroll
  for (int k = 0; k < K - 1; ++k) {
    const int tt = t0 - (K - 1) + k;
    if (tt >= 0) load8(x + int64_t(tt) * ldx + c, win[k + 1]);
  }
  const int t1 = min(L, t0 + kConvRows);
  for (int t = t0; t < t1; ++t) {
#pragma unroll
    for (int k = 0; k < K - 1; ++k)
#pragma unroll
      for (int j = 0; j < 8; ++j) win[k][j] = win[k + 1][j];
    load8(x + int64_t(t) * ldx + c, win[K - 1]);
    const int s0 = seq_start[t];
    float acc[8] = {};
#pragma unroll
    for (int k = 0; k < K; ++k) {
      if (t - (K - 1) + k < s0) continue;  // rows of the previous sequence (or before 0)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += win[k][j] * wk[j][k];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = siluf_(acc[j]);
    store8(out + int64_t(t) * C + c, acc);
  }
}

int norm_threads(int64_t D) { return int(std::min<int64_t>(1024, std::max<int64_t>(32, ((D / 8 + 31) / 32) * 32))); }

}  // namespace

// ============================================================== host side
at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, at::ScalarType out_dtype) {
  WEMM_CHECK(x.dim() == 2 && x.stride(1) == 1 && (is_act(x) || x.scalar_type() == at::kFloat), "rmsnorm: bad input");
  const int64_t N = x.size(0), D = x.size(1);
  WEMM_CHECK(D % 8 == 0 && x.stride(0) % 8 == 0, "rmsnorm: D must be a multiple of 8");
  WEMM_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == D, "rmsnorm: bad weight");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({N, D}, x.options().dtype(out_dtype));
  if (N == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto in_tag, auto out_tag) {
    using Ti = decltype(in_tag);
    using To = decltype(out_tag);
    rmsnorm_kernel<Ti, To><<<unsigned(N), norm_threads(D), 0, stream>>>(ptr<Ti>(x), x.stride(0), w.data_ptr<float>(),
                                                                        ptr<To>(out), int(D), float(eps));
  };
  auto with_out = [&](auto in_tag) {
    if (out_dtype == at::kFloat) launch(in_tag, float{});
    else dispatch_act(out_dtype, [&](auto o) { launch(in_tag, o); });
  };
  if (x.scalar_type() == at::kFloat) with_out(float{});
  else dispatch_act(x.scalar_type(), with_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor layernorm(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double eps,
                     at::ScalarType out_dtype) {
  WEMM_CHECK(x.is_contiguous() && x.dim() == 2 && x.size(1) % 8 == 0 && (is_act(x) || x.scalar_type() == at::kFloat),
             "layernorm: bad input");
  const int64_t N = x.size(0), D = x.size(1);
  WEMM_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.numel() == D &&
                 beta.numel() == D,
             "layernorm: gamma/beta must be fp32 [D]");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({N, D}, x.options().dtype(out_dtype));
  if (N == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_act(out_dtype, [&](auto o) {
    using To = decltype(o);
    auto launch = [&](auto in_tag) {
      using Ti = decltype(in_tag);
      layernorm_kernel<Ti, To><<<unsigned(N), norm_threads(D), 0, stream>>>(
          ptr<Ti>(x), gamma.data_ptr<float>(), beta.data_ptr<float>(), ptr<To>(out), int(D), float(eps));
    };
    if (x.scalar_type() == at::kFloat) launch(float{});
    else dispatch_act(x.scalar_type(), launch);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor head_rmsnorm_rope(const at::Tensor& x, const at::Tensor& w, const at::Tensor& cos, const at::Tensor& sin,
                             const at::Tensor& rows, double eps) {
  WEMM_CHECK(x.dim() == 3 && x.stride(2) == 1 && is_act(x), "head_rmsnorm_rope: bad x");
  const int64_t H = x.size(1), hd = x.size(2), rot = cos.size(1);
  const int64_t N = rows.defined() ? rows.numel() : x.size(0);
  WEMM_CHECK(hd == 256 || hd == 128, "head_rmsnorm_rope: head_dim must be 128 or 256");
  const int vpt = int(hd / 32);
  WEMM_CHECK(rot <= hd && (rot / 2) % vpt == 0 && rot / 2 / vpt <= 16, "head_rmsnorm_rope: unsupported rotary dim");
  WEMM_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0, "head_rmsnorm_rope: misaligned input");
  WEMM_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat && cos.is_contiguous() &&
                 sin.is_contiguous() && cos.size(0) >= x.size(0),
             "head_rmsnorm_rope: cos/sin must be contiguous fp32 [L, rot]");
  WEMM_CHECK(!rows.defined() || rows.scalar_type() == at::kLong, "head_rmsnorm_rope: rows must be int64");
  WEMM_CHECK(w.scalar_type() == at::kFloat && w.numel() == hd, "head_rmsnorm_rope: bad weight");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({N, H, hd}, x.options());
  if (N == 0) return out;
  const int block = 256;
  const unsigned grid = blocks_for(N * H * 32, block);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t* rp = rows.defined() ? rows.data_ptr<int64_t>() : nullptr;
  dispatch_act(x.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    auto args = std::make_tuple(ptr<T>(x), x.stride(0), x.stride(1), w.data_ptr<float>(), cos.data_ptr<float>(),
                                sin.data_ptr<float>(), int(rot), rp, ptr<T>(out), int(N), int(H), float(eps));
    if (vpt == 8)
      std::apply([&](auto... a) { head_norm_rope_kernel<T, 8><<<grid, block, 0, stream>>>(a...); }, args);
    else
      std::apply([&](auto... a) { head_norm_rope_kernel<T, 4><<<grid, block, 0, stream>>>(a...); }, args);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor sigmoid_gate_mul(const at::Tensor& o, const at::Tensor& gate, const at::Tensor& rows) {
  WEMM_CHECK(o.is_contiguous() && o.dim() == 3 && gate.dim() == 3 && gate.stride(2) == 1 && is_act(o) &&
                 gate.scalar_type() == o.scalar_type(),
             "gate_mul: bad input");
  const int64_t N = o.size(0), H = o.size(1), hd = o.size(2);
  WEMM_CHECK(hd % 8 == 0 && gate.stride(0) % 8 == 0 && gate.stride(1) % 8 == 0, "gate_mul: misaligned");
  c10::cuda::CUDAGuard guard(o.device());
  at::Tensor out = at::empty({N, H * hd}, o.options());
  if (N == 0) return out;
  const int block = 256;
  dispatch_act(o.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    gate_mul_kernel<T><<<blocks_for(N * H * hd / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<T>(o), ptr<T>(gate), gate.stride(0), gate.stride(1), rows.defined() ? rows.data_ptr<int64_t>() : nullptr,
        ptr<T>(out), int(N), int(H), int(hd));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor silu_mul(const at::Tensor& gu) {
  WEMM_CHECK(gu.is_contiguous() && gu.dim() == 2 && gu.size(1) % 16 == 0 && is_act(gu), "silu_mul: bad input");
  const int64_t N = gu.size(0), I = gu.size(1) / 2;
  c10::cuda::CUDAGuard guard(gu.device());
  at::Tensor out = at::empty({N, I}, gu.options());
  if (N == 0) return out;
  const int block = 256;
  dispatch_act(gu.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    silu_mul_kernel<T><<<blocks_for(N * I / 8, block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<T>(gu), ptr<T>(out), N, int(I));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<at::Tensor, at::Tensor> vision_qk_rope(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                                                 int64_t nh) {
  WEMM_CHECK(qkv.is_contiguous() && is_act(qkv), "vision_rope: bad qkv");
  const int64_t P = qkv.size(0), hd = qkv.size(1) / 3 / nh;
  WEMM_CHECK(hd % 2 == 0 && cos.size(1) == hd && cos.is_contiguous() && sin.is_contiguous() &&
                 cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
             "vision_rope: cos/sin must be contiguous fp32 [P, head_dim]");
  c10::cuda::CUDAGuard guard(qkv.device());
  at::Tensor q = at::empty({P, nh, hd}, qkv.options()), k = at::empty({P, nh, hd}, qkv.options());
  if (P == 0) return {q, k};
  const int block = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_act(qkv.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    if (hd == 64)
      vision_rope64_kernel<T><<<blocks_for(P * nh * 2 * 32, block), block, 0, stream>>>(
          ptr<T>(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(), ptr<T>(q), ptr<T>(k), int(P), int(nh));
    else
      vision_rope_generic_kernel<T><<<blocks_for(P * nh * hd, block), block, 0, stream>>>(
          ptr<T>(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(), ptr<T>(q), ptr<T>(k), int(P), int(nh), int(hd));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k};
}

at::Tensor causal_conv1d_silu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& seq_start) {
  WEMM_CHECK(is_act(x) && x.stride(1) == 1, "conv1d: x must be bf16/fp16 with unit column stride");
  WEMM_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.size(0) == x.size(1), "conv1d: bad weight");
  WEMM_CHECK(seq_start.scalar_type() == at::kInt && seq_start.numel() == x.size(0), "conv1d: bad seq_start");
  const int L = int(x.size(0)), C = int(x.size(1)), K = int(w.size(1));
  WEMM_CHECK(C % 8 == 0 && x.stride(0) % 8 == 0, "conv1d: channels must be a multiple of 8");
  WEMM_CHECK(K == 4, "conv1d: only kernel size 4 is supported, got ", K);
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({L, C}, x.options());
  if (L == 0) return out;
  const int block = 256;
  dispatch_act(x.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    conv1d_silu_kernel<T, 4><<<blocks_for(int64_t((L + kConvRows - 1) / kConvRows) * (C / 8), block), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        ptr<T>(x), x.stride(0), w.data_ptr<float>(), seq_start.data_ptr<int>(), ptr<T>(out), L, C);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


}  // namespace wemm::kernels
