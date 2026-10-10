#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "wemm/common.h"
#include "wemm/gemm.h"
#include "wemm/kernels.h"

namespace wemm::kernels {

namespace {

constexpr int kChunk = 64;
constexpr int kDK = 128;
constexpr int kBV = 32;       // value columns per block
constexpr int kThreads = 256;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

// State propagation of the chunked gated delta rule. One block owns the
// state columns S[:, v0:v0+32] of one (head, sequence); warp w keeps rows
// k in [16w, 16w+16) of that slice in registers (lane = column). Per chunk:
//   S_out[c] = S                       (state at the chunk start)
//   vnew     = u - w @ S               (cross-warp reduction through smem)
//   S        = S * dl + kd^T @ vnew
// The next chunk's w / kd tiles are prefetched with cp.async while the other
// phase computes.
__global__ void __launch_bounds__(kThreads)
gdn_state_kernel(const float* __restrict__ kd, const float* __restrict__ u, const float* __restrict__ w,
                 const float* __restrict__ dl, const int* __restrict__ chunk_offsets, float* __restrict__ s_out,
                 float* __restrict__ vnew_out, int C, int DV) {
  const int h = blockIdx.x, seq = blockIdx.y, v0 = blockIdx.z * kBV;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  extern __shared__ __align__(16) float smem[];
  float* Ws = smem;                    // [64][128]
  float* KDs = Ws + kChunk * kDK;      // [64][128]
  float* red = KDs + kChunk * kDK;     // [8][16][32]
  float* vn = red + 8 * 16 * 32;       // [64][32]
  float* Us = vn + kChunk * kBV;       // [64][32] u slice of the current chunk

  const int c0 = chunk_offsets[seq], c1 = chunk_offsets[seq + 1];
  if (c0 == c1) return;
  auto load_tile = [&](float* dst, const float* src) {
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads) cp_async16(dst + i * 4, src + i * 4);
    cp_async_commit();
  };
  // group order per chunk: {w_c, u_c}, {kd_c}
  auto load_wu = [&](size_t hc) {
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads) cp_async16(Ws + i * 4, w + hc * kChunk * kDK + i * 4);
    for (int i = tid; i < kChunk * kBV / 4; i += kThreads) {
      const int r = i / (kBV / 4), q = i % (kBV / 4);
      cp_async16(Us + r * kBV + q * 4, u + hc * kChunk * DV + size_t(r) * DV + v0 + q * 4);
    }
    cp_async_commit();
  };
  load_wu(size_t(h) * C + c0);
  load_tile(KDs, kd + (size_t(h) * C + c0) * kChunk * kDK);

  float s[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) s[j] = 0.f;
  const int k0 = warp * 16;

  for (int c = c0; c < c1; ++c) {
    const size_t hc = size_t(h) * C + c;
    // state at the chunk start
    float* so = s_out + hc * kDK * DV + size_t(k0) * DV + v0 + lane;
#pragma unroll
    for (int j = 0; j < 16; ++j) so[size_t(j) * DV] = s[j];

    cp_async_wait1();  // w_c landed (kd_c may still be in flight)
    __syncthreads();
    // ---- vnew = u - w @ S, 16 rows per round
    for (int rb = 0; rb < kChunk; rb += 16) {
#pragma unroll 4
      for (int rr = 0; rr < 16; ++rr) {
        const float* wr = Ws + (rb + rr) * kDK + k0;
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
          const float4 a = *reinterpret_cast<const float4*>(wr + j);
          p += a.x * s[j] + a.y * s[j + 1] + a.z * s[j + 2] + a.w * s[j + 3];
        }
        red[(warp * 16 + rr) * 32 + lane] = p;
      }
      __syncthreads();
      for (int idx = tid; idx < 16 * 32; idx += kThreads) {
        const int rr = idx >> 5, l = idx & 31;
        float acc = 0.f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww) acc += red[(ww * 16 + rr) * 32 + l];
        const int r = rb + rr;
        const float v = Us[r * kBV + l] - acc;
        vn[r * 32 + l] = v;
        vnew_out[hc * kChunk * DV + size_t(r) * DV + v0 + l] = v;
      }
      __syncthreads();
    }
    // w_c consumed: prefetch w_{c+1}
    if (c + 1 < c1) load_wu(hc + 1);
    else cp_async_commit();
    cp_async_wait1();  // kd_c landed
    __syncthreads();
    // ---- S = S * dl + kd^T @ vnew
    {
      const float d = dl[hc];
      float acc[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) acc[j] = s[j] * d;
#pragma unroll 4
      for (int r = 0; r < kChunk; ++r) {
        const float v = vn[r * 32 + lane];
        const float* kr = KDs + r * kDK + k0;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
          const float4 a = *reinterpret_cast<const float4*>(kr + j);
          acc[j] += a.x * v;
          acc[j + 1] += a.y * v;
          acc[j + 2] += a.z * v;
          acc[j + 3] += a.w * v;
        }
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) s[j] = acc[j];
    }
    __syncthreads();  // kd_c and vn consumed
    if (c + 1 < c1) load_tile(KDs, kd + (hc + 1) * kChunk * kDK);
    else cp_async_commit();
  }
}

}  // namespace

at::Tensor gdn_chunk_recurrence(const at::Tensor& qg, const at::Tensor& kd, const at::Tensor& u, const at::Tensor& w,
                                const at::Tensor& aqk, const at::Tensor& dl, const at::Tensor& chunk_offsets,
                                int64_t num_seqs) {
  const int64_t H = qg.size(0), C = qg.size(1), DV = u.size(3);
  WEMM_CHECK(qg.size(2) == kChunk && qg.size(3) == kDK && w.size(3) == kDK && kd.size(3) == kDK,
             "gdn kernel expects chunk 64 and key dim 128");
  WEMM_CHECK(DV % kBV == 0, "value dim must be a multiple of ", kBV);
  for (auto* t : {&qg, &kd, &u, &w, &aqk, &dl})
    WEMM_CHECK(t->is_contiguous() && t->scalar_type() == at::kFloat, "gdn inputs must be contiguous fp32");
  at::cuda::CUDAGuard guard(qg.device());
  at::Tensor states = at::empty({H, C, kDK, DV}, qg.options());
  at::Tensor vnew = at::empty({H, C, kChunk, DV}, qg.options());
  const size_t smem = sizeof(float) * (2 * kChunk * kDK + 8 * 16 * 32 + 2 * kChunk * kBV);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    attr_set = true;
  }
  dim3 grid{unsigned(H), unsigned(num_seqs), unsigned(DV / kBV)};
  gdn_state_kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      kd.data_ptr<float>(), u.data_ptr<float>(), w.data_ptr<float>(), dl.data_ptr<float>(),
      chunk_offsets.data_ptr<int>(), states.data_ptr<float>(), vnew.data_ptr<float>(), int(C), int(DV));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Outputs are independent per chunk: o = qg @ S_c + aqk @ vnew (fp32 batched GEMMs).
  at::Tensor o = at::empty({H * C, kChunk, DV}, qg.options());
  gemm_f32_batched(qg.view({H * C, kChunk, kDK}), states.view({H * C, kDK, DV}), o, false);
  gemm_f32_batched(aqk.view({H * C, kChunk, kChunk}), vnew.view({H * C, kChunk, DV}), o, true);
  return o.view({H, C, kChunk, DV});
}

}  // namespace wemm::kernels

namespace wemm::kernels {

namespace {

constexpr int kLd = 132;  // row stride (floats) of the 64x128 tiles: float4-aligned, spreads banks
constexpr int kLt = 68;   // row stride of the 64x64 tiles

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// One block per (chunk, value head). 256 threads. dk = dv = 128.
__global__ void __launch_bounds__(256)
gdn_prepare_kernel(const __nv_bfloat16* __restrict__ mixed, int64_t ld_mixed, const __nv_bfloat16* __restrict__ bvec,
                   const __nv_bfloat16* __restrict__ avec, int64_t ld_ab, const float* __restrict__ neg_A,
                   const float* __restrict__ dt_bias, const int* __restrict__ chunk_start,
                   const int* __restrict__ chunk_valid, int key_dim, int head_ratio, int C, float scale,
                   float* __restrict__ qg_out, float* __restrict__ kd_out, float* __restrict__ u_out,
                   float* __restrict__ w_out, float* __restrict__ aqk_out, float* __restrict__ dl_out) {
  const int c = blockIdx.x, h = blockIdx.y, kh = h / head_ratio;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t0 = chunk_start[c], nvalid = chunk_valid[c];

  extern __shared__ __align__(16) float sm[];
  float* Ks = sm;                    // [64][kLd]  K, later K*beta*exp(gc)
  float* Qs = Ks + kChunk * kLd;     // [64][kLd]  Q, later T scratch, later V*beta
  float* As = Qs + kChunk * kLd;     // [64][kLt]  A (strict lower), later T
  float* gc = As + kChunk * kLt;     // [64] cumulative log decay
  float* bt = gc + kChunk;           // [64] beta
  float* eg = bt + kChunk;           // [64] exp(gc)
  float* ed = eg + kChunk;           // [64] exp(glast - gc)
  float* Ts = Qs;                    // [64][kLt] while Q is dead

  // ---- gates and in-chunk cumulative decay (warp scans)
  if (tid < kChunk) {
    float g = 0.f, b = 0.f;
    if (tid < nvalid) {
      const int64_t t = t0 + tid;
      const float x = __bfloat162float(avec[t * ld_ab + h]) + dt_bias[h];
      const float sp = x > 20.f ? x : log1pf(expf(x));  // softplus, threshold 20
      g = neg_A[h] * sp;
      b = 1.f / (1.f + expf(-__bfloat162float(bvec[t * ld_ab + h])));
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float n = __shfl_up_sync(0xffffffffu, g, o);
      if (lane >= o) g += n;
    }
    gc[tid] = g;
    bt[tid] = b;
  }
  // ---- q/k rows with L2 norm (each warp: 8 rows, 4 elements per lane)
  for (int r = warp; r < kChunk; r += 8) {
    float q[4] = {0.f, 0.f, 0.f, 0.f}, k[4] = {0.f, 0.f, 0.f, 0.f};
    if (r < nvalid) {
      const __nv_bfloat16* row = mixed + int64_t(t0 + r) * ld_mixed;
      const uint2 qr = *reinterpret_cast<const uint2*>(row + kh * kDK + lane * 4);
      const uint2 kr = *reinterpret_cast<const uint2*>(row + key_dim + kh * kDK + lane * 4);
      const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qr);
      const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kr);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        q[j] = __bfloat162float(qb[j]);
        k[j] = __bfloat162float(kb[j]);
      }
    }
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sq += q[j] * q[j];
      sk += k[j] * k[j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
    const float rq = rsqrtf(sq + 1e-6f) * scale, rk = rsqrtf(sk + 1e-6f);
    *reinterpret_cast<float4*>(Qs + r * kLd + lane * 4) = make_float4(q[0] * rq, q[1] * rq, q[2] * rq, q[3] * rq);
    *reinterpret_cast<float4*>(Ks + r * kLd + lane * 4) = make_float4(k[0] * rk, k[1] * rk, k[2] * rk, k[3] * rk);
  }
  __syncthreads();
  if (tid >= 32 && tid < kChunk) gc[tid] += gc[31];
  __syncthreads();
  if (tid < kChunk) {
    eg[tid] = expf(gc[tid]);
    ed[tid] = expf(gc[kChunk - 1] - gc[tid]);
  }
  __syncthreads();

  const size_t hc = size_t(h) * C + c;
  // ---- A = beta_i K_i.K_j decay (i > j), Aqk = Q_i.K_j decay (i >= j); 4x4 tile per thread
  {
    const int ti = tid >> 4, tj = tid & 15;
    const int i0 = ti * 4, j0 = tj * 4;
    float* aqk = aqk_out + hc * kChunk * kChunk;
    if (tj <= ti) {
      float a[4][4] = {}, q[4][4] = {};
#pragma unroll 2
      for (int d = 0; d < kDK; d += 4) {
        float4 ki[4], qi[4], kj[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          ki[x] = ld4(Ks + (i0 + x) * kLd + d);
          qi[x] = ld4(Qs + (i0 + x) * kLd + d);
          kj[x] = ld4(Ks + (j0 + x) * kLd + d);
        }
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
          for (int y = 0; y < 4; ++y) {
            a[x][y] += ki[x].x * kj[y].x + ki[x].y * kj[y].y + ki[x].z * kj[y].z + ki[x].w * kj[y].w;
            q[x][y] += qi[x].x * kj[y].x + qi[x].y * kj[y].y + qi[x].z * kj[y].z + qi[x].w * kj[y].w;
          }
      }
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const int i = i0 + x;
        float av[4], qv[4];
#pragma unroll
        for (int y = 0; y < 4; ++y) {
          const int j = j0 + y;
          const float dec = j <= i ? expf(gc[i] - gc[j]) : 0.f;
          av[y] = j < i ? bt[i] * a[x][y] * dec : 0.f;
          qv[y] = q[x][y] * dec;
        }
        *reinterpret_cast<float4*>(As + i * kLt + j0) = make_float4(av[0], av[1], av[2], av[3]);
        *reinterpret_cast<float4*>(aqk + i * kChunk + j0) = make_float4(qv[0], qv[1], qv[2], qv[3]);
      }
    } else {
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        *reinterpret_cast<float4*>(As + (i0 + x) * kLt + j0) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(aqk + (i0 + x) * kChunk + j0) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }
  // qg = Q * exp(gc); kd = K * exp(g_last - gc)
  {
    float4* qg = reinterpret_cast<float4*>(qg_out + hc * kChunk * kDK);
    float4* kd = reinterpret_cast<float4*>(kd_out + hc * kChunk * kDK);
    for (int idx = tid; idx < kChunk * kDK / 4; idx += 256) {
      const int r = idx >> 5, d = (idx & 31) * 4;
      const float4 q = ld4(Qs + r * kLd + d), k = ld4(Ks + r * kLd + d);
      const float e1 = eg[r], e2 = ed[r];
      qg[idx] = make_float4(q.x * e1, q.y * e1, q.z * e1, q.w * e1);
      kd[idx] = make_float4(k.x * e2, k.y * e2, k.z * e2, k.w * e2);
    }
    if (tid == 0) dl_out[hc] = eg[kChunk - 1];
  }
  __syncthreads();  // Q is dead: its buffer becomes T scratch

  // ---- T = (I + A)^{-1}, 16x16 blocked. Diagonal blocks by forward substitution.
  if (warp < 4 && lane < 16) {
    const int base = warp * 16;
    const int j = lane;
    for (int r = 0; r < 16; ++r) {
      float v;
      if (j < r) {
        v = -As[(base + r) * kLt + base + j];
        for (int m = j + 1; m < r; ++m) v -= As[(base + r) * kLt + base + m] * Ts[(base + m) * kLt + base + j];
      } else {
        v = j == r ? 1.f : 0.f;
      }
      Ts[(base + r) * kLt + base + j] = v;
      __syncwarp(0x0000ffffu);
    }
  }
  __syncthreads();
  // Off-diagonal blocks by distance: T_ij = -T_ii * sum_{k=j}^{i-1} A_ik T_kj.
  {
    const int r = tid >> 4, cc = tid & 15;
    for (int dist = 1; dist < 4; ++dist) {
      for (int i = dist; i < 4; ++i) {
        const int j = i - dist;
        float mval = 0.f;
        for (int k = j; k < i; ++k)
#pragma unroll
          for (int m = 0; m < 16; ++m)
            mval += As[(i * 16 + r) * kLt + k * 16 + m] * Ts[(k * 16 + m) * kLt + j * 16 + cc];
        As[(j * 16 + r) * kLt + i * 16 + cc] = mval;  // stash in the unused upper block (j, i)
        __syncthreads();
        float tval = 0.f;
#pragma unroll
        for (int m = 0; m < 16; ++m) tval -= Ts[(i * 16 + r) * kLt + i * 16 + m] * As[(j * 16 + m) * kLt + i * 16 + cc];
        Ts[(i * 16 + r) * kLt + j * 16 + cc] = tval;
        __syncthreads();
      }
    }
  }
  // Move T into the A buffer (zeroing its upper part) and free the Q buffer.
  for (int idx = tid; idx < kChunk * kChunk; idx += 256) {
    const int r = idx >> 6, m = idx & 63;
    As[r * kLt + m] = m <= r ? Ts[r * kLt + m] : 0.f;
  }
  __syncthreads();
  // V*beta into the Q buffer, K*beta*exp(gc) in place.
  float* Vs = Qs;
  for (int idx = tid; idx < kChunk * kDK / 4; idx += 256) {
    const int r = idx >> 5, d = (idx & 31) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < nvalid) {
      const uint2 raw = *reinterpret_cast<const uint2*>(mixed + int64_t(t0 + r) * ld_mixed + 2 * key_dim + h * kDK + d);
      const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&raw);
      const float br = bt[r];
      v = make_float4(__bfloat162float(vb[0]) * br, __bfloat162float(vb[1]) * br, __bfloat162float(vb[2]) * br,
                      __bfloat162float(vb[3]) * br);
    }
    *reinterpret_cast<float4*>(Vs + r * kLd + d) = v;
    const float kb = bt[r] * eg[r];
    float4 k = ld4(Ks + r * kLd + d);
    *reinterpret_cast<float4*>(Ks + r * kLd + d) = make_float4(k.x * kb, k.y * kb, k.z * kb, k.w * kb);
  }
  __syncthreads();
  // ---- u = T Vb, w = T Kbg: 4 rows x 8 cols per thread for each output.
  {
    const int tr = tid >> 4, tc = tid & 15;
    const int i0 = tr * 4, d0 = tc * 8;
    float su[4][8] = {}, sw[4][8] = {};
    const int mmax = i0 + 4;
    for (int m = 0; m < mmax; ++m) {
      float t[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) t[x] = As[(i0 + x) * kLt + m];
      const float4 v0 = ld4(Vs + m * kLd + d0), v1 = ld4(Vs + m * kLd + d0 + 4);
      const float4 k0 = ld4(Ks + m * kLd + d0), k1 = ld4(Ks + m * kLd + d0 + 4);
      const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
      const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 8; ++y) {
          su[x][y] += t[x] * vv[y];
          sw[x][y] += t[x] * kk[y];
        }
    }
    float* u = u_out + hc * kChunk * kDK;
    float* w = w_out + hc * kChunk * kDK;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      float4* up = reinterpret_cast<float4*>(u + (i0 + x) * kDK + d0);
      float4* wp = reinterpret_cast<float4*>(w + (i0 + x) * kDK + d0);
      up[0] = make_float4(su[x][0], su[x][1], su[x][2], su[x][3]);
      up[1] = make_float4(su[x][4], su[x][5], su[x][6], su[x][7]);
      wp[0] = make_float4(sw[x][0], sw[x][1], sw[x][2], sw[x][3]);
      wp[1] = make_float4(sw[x][4], sw[x][5], sw[x][6], sw[x][7]);
    }
  }
}

}  // namespace

GdnChunks gdn_chunk_prepare(const at::Tensor& mixed, const at::Tensor& b, const at::Tensor& a,
                            const at::Tensor& neg_A, const at::Tensor& dt_bias, const at::Tensor& chunk_start,
                            const at::Tensor& chunk_valid, int64_t Hk, int64_t Hv, int64_t dk, int64_t dv) {
  WEMM_CHECK(dk == kDK && dv == kDK, "gdn_chunk_prepare: head dims must be 128");
  WEMM_CHECK(mixed.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1 && a.stride(0) == b.stride(0),
             "gdn_chunk_prepare: bad strides");
  const int64_t C = chunk_start.numel();
  auto opts = mixed.options().dtype(at::kFloat);
  GdnChunks r;
  r.qg = at::empty({Hv, C, kChunk, dk}, opts);
  r.kd = at::empty({Hv, C, kChunk, dk}, opts);
  r.u = at::empty({Hv, C, kChunk, dv}, opts);
  r.w = at::empty({Hv, C, kChunk, dk}, opts);
  r.aqk = at::empty({Hv, C, kChunk, kChunk}, opts);
  r.dl = at::empty({Hv, C}, opts);
  if (C == 0) return r;
  const size_t smem = sizeof(float) * (2 * kChunk * kLd + kChunk * kLt + 4 * kChunk);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    attr_set = true;
  }
  dim3 grid{unsigned(C), unsigned(Hv)};
  gdn_prepare_kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(mixed.data_ptr()), mixed.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      b.stride(0), neg_A.data_ptr<float>(), dt_bias.data_ptr<float>(), chunk_start.data_ptr<int>(),
      chunk_valid.data_ptr<int>(), int(Hk * dk), int(Hv / Hk), int(C), float(1.0 / std::sqrt(double(dk))),
      r.qg.data_ptr<float>(), r.kd.data_ptr<float>(), r.u.data_ptr<float>(), r.w.data_ptr<float>(),
      r.aqk.data_ptr<float>(), r.dl.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return r;
}

}  // namespace wemm::kernels
