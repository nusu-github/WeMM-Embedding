// Chunked gated delta rule (Gated DeltaNet), fp32 throughout.
//
//   prepare   (parallel over chunks x heads): q/k/v gather + L2 norm, gates,
//             in-chunk decay, WY representation -> qg, kd, u, w, aqk, dl
//   state     (sequential over chunks): S_c and vnew_c per chunk
//   output    (parallel): o_c = qg_c @ S_c + aqk_c @ vnew_c   (fp32 batched GEMMs)

#include "common.cuh"
#include "wemm/gemm.h"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

constexpr int kChunk = 64;
constexpr int kDK = 128;   // key and value head dim
constexpr int kHalf = 64;  // d-columns per smem tile in the prepare kernel
constexpr int kLt = 68;    // row stride (floats) of the 64x64 smem tiles: float4-aligned, spreads banks
constexpr int kBV = 32;    // value columns per block in the state kernel
constexpr int kThreads = 256;

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float a, float b, float c, float d) {
  *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

// =================================================================== prepare
// One block per (chunk, value head), 256 threads. The 64x128 q/k/v tiles are
// processed as two 64x64 column halves so the block needs ~54 KB of shared
// memory (fits sm_75's 64 KB limit). Phases:
//   1. gates g, beta and the in-chunk cumulative decay gc; row L2 norms of q, k
//   2. A = beta_i K_i.K_j decay (i > j) and Aqk = Q_i.K_j decay (i >= j),
//      accumulated over the two halves; qg = Q e^gc and kd = K e^(g_last-gc)
//      are written per half
//   3. T = (I + A)^-1, 16x16 blocked (diagonal blocks by forward substitution)
//   4. u = T (V beta), w = T (K beta e^gc), per half
template <typename T>
__global__ void __launch_bounds__(kThreads)
gdn_prepare_kernel(const T* __restrict__ mixed, int64_t ld_mixed, const T* __restrict__ bvec,
                   const T* __restrict__ avec, int64_t ld_ab, const float* __restrict__ neg_A,
                   const float* __restrict__ dt_bias, const int* __restrict__ chunk_start,
                   const int* __restrict__ chunk_valid, int key_dim, int head_ratio, int C, int cb, float scale,
                   float* __restrict__ qg_out, float* __restrict__ kd_out, float* __restrict__ u_out,
                   float* __restrict__ w_out, float* __restrict__ aqk_out, float* __restrict__ dl_out) {
  const int c = blockIdx.x, h = blockIdx.y, kh = h / head_ratio;  // c: index within the window
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t0 = chunk_start[cb + c], nvalid = chunk_valid[cb + c];
  const size_t hc = size_t(h) * C + c;

  extern __shared__ __align__(16) float sm[];
  float* Ks = sm;                   // [64][kLt] K half; later K*beta*e^gc half
  float* Qs = Ks + kChunk * kLt;    // [64][kLt] Q half; later T
  float* As = Qs + kChunk * kLt;    // [64][kLt] A (strict lower); later V*beta half
  float* gc = As + kChunk * kLt;    // [64] cumulative log decay
  float* bt = gc + kChunk;          // [64] beta
  float* eg = bt + kChunk;          // [64] e^gc
  float* ed = eg + kChunk;          // [64] e^(g_last - gc)
  float* rq = ed + kChunk;          // [64] query row scale (1/||q|| * scale)
  float* rk = rq + kChunk;          // [64] key row scale (1/||k||)
  float* Ts = Qs;

  auto row = [&](int r) { return mixed + int64_t(t0 + r) * ld_mixed; };

  // ---- 1. gates (warp scan) and row norms
  if (tid < kChunk) {
    float g = 0.f, b = 0.f;
    if (tid < nvalid) {
      const int64_t t = t0 + tid;
      const float x = to_f(avec[t * ld_ab + h]) + dt_bias[h];
      const float sp = x > 20.f ? x : log1pf(expf(x));  // softplus, threshold 20
      g = neg_A[h] * sp;
      b = sigmoidf_(to_f(bvec[t * ld_ab + h]));
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float n = __shfl_up_sync(0xffffffffu, g, o);
      if (lane >= o) g += n;
    }
    gc[tid] = g;
    bt[tid] = b;
  }
  for (int r = warp; r < kChunk; r += 8) {
    float sq = 0.f, sk = 0.f;
    if (r < nvalid) {
      float q[4], k[4];
      load4(row(r) + kh * kDK + lane * 4, q);
      load4(row(r) + key_dim + kh * kDK + lane * 4, k);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sq += q[j] * q[j];
        sk += k[j] * k[j];
      }
    }
    sq = warp_sum(sq);
    sk = warp_sum(sk);
    if (lane == 0) {
      rq[r] = r < nvalid ? rsqrtf(sq + 1e-6f) * scale : 0.f;
      rk[r] = r < nvalid ? rsqrtf(sk + 1e-6f) : 0.f;
    }
  }
  __syncthreads();
  if (tid >= 32 && tid < kChunk) gc[tid] += gc[31];
  __syncthreads();
  if (tid < kChunk) {
    eg[tid] = expf(gc[tid]);
    ed[tid] = expf(gc[kChunk - 1] - gc[tid]);
  }
  if (tid == 0) dl_out[hc] = expf(gc[kChunk - 1]);

  // ---- 2. A / Aqk (4x4 tile per thread), qg / kd
  const int ti = tid >> 4, tj = tid & 15;
  const int i0 = ti * 4, j0 = tj * 4;
  float accA[4][4] = {}, accQ[4][4] = {};
  for (int half = 0; half < 2; ++half) {
    __syncthreads();
    // normalized q, k half tiles: 64 rows x 64 cols, 4 per thread-step
    for (int idx = tid; idx < kChunk * kHalf / 4; idx += kThreads) {
      const int r = idx / (kHalf / 4), d = (idx % (kHalf / 4)) * 4;
      float q[4] = {0.f, 0.f, 0.f, 0.f}, k[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < nvalid) {
        load4(row(r) + kh * kDK + half * kHalf + d, q);
        load4(row(r) + key_dim + kh * kDK + half * kHalf + d, k);
      }
      st4(Qs + r * kLt + d, q[0] * rq[r], q[1] * rq[r], q[2] * rq[r], q[3] * rq[r]);
      st4(Ks + r * kLt + d, k[0] * rk[r], k[1] * rk[r], k[2] * rk[r], k[3] * rk[r]);
    }
    __syncthreads();
    if (tj <= ti) {
#pragma unroll 2
      for (int d = 0; d < kHalf; d += 4) {
        float4 ki[4], qi[4], kj[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          ki[x] = ld4(Ks + (i0 + x) * kLt + d);
          qi[x] = ld4(Qs + (i0 + x) * kLt + d);
          kj[x] = ld4(Ks + (j0 + x) * kLt + d);
        }
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
          for (int y = 0; y < 4; ++y) {
            accA[x][y] += ki[x].x * kj[y].x + ki[x].y * kj[y].y + ki[x].z * kj[y].z + ki[x].w * kj[y].w;
            accQ[x][y] += qi[x].x * kj[y].x + qi[x].y * kj[y].y + qi[x].z * kj[y].z + qi[x].w * kj[y].w;
          }
      }
    }
    float* qg = qg_out + hc * kChunk * kDK + half * kHalf;
    float* kd = kd_out + hc * kChunk * kDK + half * kHalf;
    for (int idx = tid; idx < kChunk * kHalf / 4; idx += kThreads) {
      const int r = idx / (kHalf / 4), d = (idx % (kHalf / 4)) * 4;
      const float4 q = ld4(Qs + r * kLt + d), k = ld4(Ks + r * kLt + d);
      st4(qg + r * kDK + d, q.x * eg[r], q.y * eg[r], q.z * eg[r], q.w * eg[r]);
      st4(kd + r * kDK + d, k.x * ed[r], k.y * ed[r], k.z * ed[r], k.w * ed[r]);
    }
  }
  {
    float* aqk = aqk_out + hc * kChunk * kChunk;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int i = i0 + x;
      float av[4], qv[4];
#pragma unroll
      for (int y = 0; y < 4; ++y) {
        const int j = j0 + y;
        const float dec = (tj <= ti && j <= i) ? expf(gc[i] - gc[j]) : 0.f;
        av[y] = j < i ? bt[i] * accA[x][y] * dec : 0.f;
        qv[y] = accQ[x][y] * dec;
      }
      st4(As + i * kLt + j0, av[0], av[1], av[2], av[3]);
      st4(aqk + i * kChunk + j0, qv[0], qv[1], qv[2], qv[3]);
    }
  }
  __syncthreads();  // Q is dead: its buffer becomes T

  // ---- 3. T = (I + A)^{-1}, unit lower triangular
  if (warp < 4 && lane < 16) {  // diagonal 16x16 blocks
    const int base = warp * 16, j = lane;
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
  {  // off-diagonal blocks by distance: T_ij = -T_ii * sum_{k=j}^{i-1} A_ik T_kj
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
  // zero T's upper triangle (scratch / stale Q there)
  for (int idx = tid; idx < kChunk * kChunk; idx += kThreads) {
    const int r = idx >> 6, m = idx & 63;
    if (m > r) Ts[r * kLt + m] = 0.f;
  }

  // ---- 4. u = T (V*beta), w = T (K*beta*e^gc), 4x4 tile per thread, per half
  float* Vs = As;
  for (int half = 0; half < 2; ++half) {
    __syncthreads();
    for (int idx = tid; idx < kChunk * kHalf / 4; idx += kThreads) {
      const int r = idx / (kHalf / 4), d = (idx % (kHalf / 4)) * 4;
      float v[4] = {0.f, 0.f, 0.f, 0.f}, k[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < nvalid) {
        load4(row(r) + 2 * key_dim + h * kDK + half * kHalf + d, v);
        load4(row(r) + key_dim + kh * kDK + half * kHalf + d, k);
      }
      const float br = bt[r], kr = rk[r], kb = bt[r] * eg[r];
      st4(Vs + r * kLt + d, v[0] * br, v[1] * br, v[2] * br, v[3] * br);
      st4(Ks + r * kLt + d, (k[0] * kr) * kb, (k[1] * kr) * kb, (k[2] * kr) * kb, (k[3] * kr) * kb);
    }
    __syncthreads();
    const int tr = tid >> 4, tc = tid & 15;
    const int r0 = tr * 4, d0 = tc * 4;
    float su[4][4] = {}, sw[4][4] = {};
    for (int m = 0; m < r0 + 4; ++m) {
      float t[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) t[x] = Ts[(r0 + x) * kLt + m];
      const float4 v = ld4(Vs + m * kLt + d0), k = ld4(Ks + m * kLt + d0);
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        su[x][0] += t[x] * v.x; su[x][1] += t[x] * v.y; su[x][2] += t[x] * v.z; su[x][3] += t[x] * v.w;
        sw[x][0] += t[x] * k.x; sw[x][1] += t[x] * k.y; sw[x][2] += t[x] * k.z; sw[x][3] += t[x] * k.w;
      }
    }
    float* u = u_out + hc * kChunk * kDK + half * kHalf;
    float* w = w_out + hc * kChunk * kDK + half * kHalf;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      st4(u + (r0 + x) * kDK + d0, su[x][0], su[x][1], su[x][2], su[x][3]);
      st4(w + (r0 + x) * kDK + d0, sw[x][0], sw[x][1], sw[x][2], sw[x][3]);
    }
  }
}

constexpr size_t kPrepareSmem = sizeof(float) * (3 * kChunk * kLt + 6 * kChunk);

// ===================================================================== state
// One block owns the state columns S[:, v0:v0+32] of one (head, sequence);
// warp w keeps rows k in [16w, 16w+16) of that slice in registers (lane =
// column). Per chunk:
//   S_out[c] = S;  vnew = u - w @ S;  S = S * dl + kd^T @ vnew
// kDouble: w/u and kd tiles live in separate buffers and the next chunk's
// tiles are prefetched with cp.async (96 KB smem, sm_80+). Otherwise a single
// tile buffer is reused (64 KB, fits sm_75).
template <bool kDouble>
__global__ void __launch_bounds__(kThreads)
gdn_state_kernel(const float* __restrict__ kd, const float* __restrict__ u, const float* __restrict__ w,
                 const float* __restrict__ dl, const int* __restrict__ chunk_offsets, float* __restrict__ s_out,
                 float* __restrict__ vnew_out, int C, int DV, int cb, float* __restrict__ carry, int num_seqs) {
  const int h = blockIdx.x, seq = blockIdx.y, v0 = blockIdx.z * kBV;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  extern __shared__ __align__(16) float smem[];
  float* Ws = smem;                                  // [64][128]
  float* KDs = kDouble ? Ws + kChunk * kDK : Ws;     // [64][128]
  float* red = KDs + kChunk * kDK;                   // [8][16][32]
  float* vn = red + 8 * 16 * 32;                     // [64][32]
  float* Us = vn + kChunk * kBV;                     // [64][32]

  // Window [cb, cb + C) of the global chunk range; per-window tensors are
  // indexed by c - cb. Sequences crossing the window edges carry S.
  const int s0 = chunk_offsets[seq], s1 = chunk_offsets[seq + 1];
  const int c0 = max(s0, cb), c1 = min(s1, cb + C);
  if (c0 >= c1) return;
  float* carry_p = carry ? carry + ((size_t(h) * num_seqs + seq) * kDK + warp * 16) * DV + v0 + lane : nullptr;

  auto load_wu = [&](size_t hc) {
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads) cp_async16(Ws + i * 4, w + hc * kChunk * kDK + i * 4);
    for (int i = tid; i < kChunk * kBV / 4; i += kThreads) {
      const int r = i / (kBV / 4), q = i % (kBV / 4);
      cp_async16(Us + r * kBV + q * 4, u + hc * kChunk * DV + size_t(r) * DV + v0 + q * 4);
    }
    cp_async_commit();
  };
  auto load_kd = [&](size_t hc) {
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads) cp_async16(KDs + i * 4, kd + hc * kChunk * kDK + i * 4);
    cp_async_commit();
  };

  const size_t hbase = size_t(h) * C - cb;  // hc = hbase + c
  load_wu(hbase + c0);
  if (kDouble) load_kd(hbase + c0);

  float s[16];
  const bool resume = carry_p && s0 < cb;
#pragma unroll
  for (int j = 0; j < 16; ++j) s[j] = resume ? carry_p[size_t(j) * DV] : 0.f;
  const int k0 = warp * 16;

  for (int c = c0; c < c1; ++c) {
    const size_t hc = hbase + c;
    float* so = s_out + hc * kDK * DV + size_t(k0) * DV + v0 + lane;  // state at the chunk start
#pragma unroll
    for (int j = 0; j < 16; ++j) so[size_t(j) * DV] = s[j];

    if (kDouble) cp_async_wait<1>();  // w_c, u_c landed (kd_c may still be in flight)
    else cp_async_wait<0>();
    __syncthreads();
    // ---- vnew = u - w @ S, 16 rows per round (cross-warp reduction in smem)
    for (int rb = 0; rb < kChunk; rb += 16) {
#pragma unroll 4
      for (int rr = 0; rr < 16; ++rr) {
        const float* wr = Ws + (rb + rr) * kDK + k0;
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
          const float4 a = ld4(wr + j);
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
    if (kDouble) {
      if (c + 1 < c1) load_wu(hc + 1);  // w_c consumed: prefetch w_{c+1}
      else cp_async_commit();
      cp_async_wait<1>();  // kd_c landed
    } else {
      load_kd(hc);  // reuses the w buffer
      cp_async_wait<0>();
    }
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
          const float4 a = ld4(kr + j);
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
    if (kDouble) {
      if (c + 1 < c1) load_kd(hc + 1);
      else cp_async_commit();
    } else if (c + 1 < c1) {
      load_wu(hc + 1);
    }
  }
  if (carry_p && s1 > cb + C) {
#pragma unroll
    for (int j = 0; j < 16; ++j) carry_p[size_t(j) * DV] = s[j];
  }
}

constexpr size_t state_smem(bool dbl) {
  return sizeof(float) * ((dbl ? 2 : 1) * kChunk * kDK + 8 * 16 * 32 + 2 * kChunk * kBV);
}
static_assert(state_smem(false) <= 65536, "single-buffer state kernel must fit sm_75's 64 KB");
static_assert(kPrepareSmem <= 65536, "prepare kernel must fit sm_75's 64 KB");

}  // namespace

// ================================================================ host side
int64_t gdn_bytes_per_head_chunk(int64_t dk, int64_t dv) {
  // qg, kd, w [64, dk]; u, vnew, o [64, dv]; aqk [64, 64]; states [dk, dv]; dl
  return int64_t(sizeof(float)) * (kChunk * (3 * dk + 3 * dv + kChunk) + dk * dv + 1);
}

GdnChunks gdn_chunk_prepare(const at::Tensor& mixed, const at::Tensor& b, const at::Tensor& a,
                            const at::Tensor& neg_A, const at::Tensor& dt_bias, const at::Tensor& chunk_start,
                            const at::Tensor& chunk_valid, int64_t Hk, int64_t Hv, int64_t dk, int64_t dv,
                            int64_t chunk_begin, int64_t chunk_end) {
  WEMM_CHECK(dk == kDK && dv == kDK, "gdn_chunk_prepare: head dims must be 128");
  WEMM_CHECK(is_act(mixed) && b.scalar_type() == mixed.scalar_type() && a.scalar_type() == mixed.scalar_type(),
             "gdn_chunk_prepare: mixed/a/b must share the activation dtype");
  WEMM_CHECK(mixed.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1 && a.stride(0) == b.stride(0) &&
                 mixed.stride(0) % 4 == 0,
             "gdn_chunk_prepare: bad strides");
  WEMM_CHECK(Hv % Hk == 0, "gdn_chunk_prepare: value heads must be a multiple of key heads");
  if (chunk_end < 0) chunk_end = chunk_start.numel();
  WEMM_CHECK(0 <= chunk_begin && chunk_begin <= chunk_end && chunk_end <= chunk_start.numel(), "gdn: bad window");
  c10::cuda::CUDAGuard guard(mixed.device());
  const int64_t C = chunk_end - chunk_begin;
  auto opts = mixed.options().dtype(at::kFloat);
  GdnChunks r;
  r.qg = at::empty({Hv, C, kChunk, dk}, opts);
  r.kd = at::empty({Hv, C, kChunk, dk}, opts);
  r.u = at::empty({Hv, C, kChunk, dv}, opts);
  r.w = at::empty({Hv, C, kChunk, dk}, opts);
  r.aqk = at::empty({Hv, C, kChunk, kChunk}, opts);
  r.dl = at::empty({Hv, C}, opts);
  if (C == 0) return r;
  dispatch_act(mixed.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    ensure_smem(gdn_prepare_kernel<T>, kPrepareSmem);
    dim3 grid{unsigned(C), unsigned(Hv)};
    gdn_prepare_kernel<T><<<grid, kThreads, kPrepareSmem, at::cuda::getCurrentCUDAStream()>>>(
        ptr<T>(mixed), mixed.stride(0), ptr<T>(b), ptr<T>(a), b.stride(0), neg_A.data_ptr<float>(),
        dt_bias.data_ptr<float>(), chunk_start.data_ptr<int>(), chunk_valid.data_ptr<int>(), int(Hk * dk),
        int(Hv / Hk), int(C), int(chunk_begin), float(1.0 / std::sqrt(double(dk))), r.qg.data_ptr<float>(),
        r.kd.data_ptr<float>(), r.u.data_ptr<float>(), r.w.data_ptr<float>(), r.aqk.data_ptr<float>(),
        r.dl.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return r;
}

at::Tensor gdn_chunk_recurrence(const GdnChunks& ch, const at::Tensor& chunk_offsets, int64_t num_seqs,
                                int64_t chunk_begin, const at::Tensor& carry, bool allow_double_buffer) {
  const int64_t H = ch.qg.size(0), C = ch.qg.size(1), DV = ch.u.size(3);
  WEMM_CHECK(ch.qg.size(2) == kChunk && ch.qg.size(3) == kDK && ch.w.size(3) == kDK && ch.kd.size(3) == kDK,
             "gdn kernel expects chunk 64 and key dim 128");
  WEMM_CHECK(DV % kBV == 0, "value dim must be a multiple of ", kBV);
  for (const at::Tensor* t : {&ch.qg, &ch.kd, &ch.u, &ch.w, &ch.aqk, &ch.dl})
    WEMM_CHECK(t->is_contiguous() && t->scalar_type() == at::kFloat, "gdn inputs must be contiguous fp32");
  WEMM_CHECK(num_seqs <= 65535, "too many sequences in one batch (", num_seqs, ")");
  if (carry.defined())
    WEMM_CHECK(carry.is_contiguous() && carry.numel() == H * num_seqs * kDK * DV, "gdn: bad carry buffer");
  else
    WEMM_CHECK(chunk_begin == 0, "gdn: a window after the first needs the carry buffer");
  c10::cuda::CUDAGuard guard(ch.qg.device());
  at::Tensor states = at::empty({H, C, kDK, DV}, ch.qg.options());
  at::Tensor vnew = at::empty({H, C, kChunk, DV}, ch.qg.options());
  if (C > 0) {
    const bool dbl = allow_double_buffer && device_has_cp_async() && max_smem_optin() >= int(state_smem(true));
    dim3 grid{unsigned(H), unsigned(num_seqs), unsigned(DV / kBV)};
    auto launch = [&](auto kernel, size_t smem) {
      ensure_smem(kernel, smem);
      kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
          ch.kd.data_ptr<float>(), ch.u.data_ptr<float>(), ch.w.data_ptr<float>(), ch.dl.data_ptr<float>(),
          chunk_offsets.data_ptr<int>(), states.data_ptr<float>(), vnew.data_ptr<float>(), int(C), int(DV),
          int(chunk_begin), carry.defined() ? carry.data_ptr<float>() : nullptr, int(num_seqs));
    };
    if (dbl) launch(gdn_state_kernel<true>, state_smem(true));
    else launch(gdn_state_kernel<false>, state_smem(false));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // Outputs are independent per chunk: o = qg @ S_c + aqk @ vnew (fp32 batched GEMMs).
  at::Tensor o = at::empty({H * C, kChunk, DV}, ch.qg.options());
  gemm_f32_batched(ch.qg.view({H * C, kChunk, kDK}), states.view({H * C, kDK, DV}), o, false);
  gemm_f32_batched(ch.aqk.view({H * C, kChunk, kChunk}), vnew.view({H * C, kChunk, DV}), o, true);
  return o.view({H, C, kChunk, DV});
}

}  // namespace wemm::kernels
