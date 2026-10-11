// Chunked gated delta rule (Gated DeltaNet), fp32 throughout.
//
//   prepare   (parallel over chunks x heads): q/k/v gather + L2 norm, gates,
//             in-chunk decay, WY representation -> qg, kd, u, w, aqk, dl
//   state     (sequential over chunks): S_c and vnew_c per chunk
//   output    (parallel): o_c = qg_c @ S_c + aqk_c @ vnew_c, fused with the
//             gated RMSNorm and the scatter back to token rows

#include "common.cuh"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

constexpr int kChunk = 64;
constexpr int kDK = 128;   // key and value head dim
constexpr int kPart = 32;  // d-columns per q/k/v smem tile in the prepare kernel
constexpr int kLt = 68;    // row stride (floats) of the 64x64 smem tiles: float4-aligned, spreads banks
constexpr int kLp = 36;    // row stride of the 64 x kPart tiles
constexpr int kBV = 32;    // value columns per block in the state kernel
constexpr int kThreads = 256;

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float a, float b, float c, float d) {
  *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

// =================================================================== prepare
// One block per (chunk, value head), 256 threads. The 64x128 q/k/v tiles are
// processed in 64x32 column parts so the block needs ~45 KB of shared memory
// (two blocks per SM on sm_86/89, fits sm_75's 64 KB). Phases:
//   1. gates g, beta and the in-chunk cumulative decay gc; row L2 norms of q, k
//   2. A = beta_i K_i.K_j decay (i > j) and Aqk = Q_i.K_j decay (i >= j),
//      accumulated over the two halves; qg = Q e^gc and kd = K e^(g_last-gc)
//      are written per part
//   3. T = (I + A)^-1, 16x16 blocked (diagonal blocks by forward substitution)
//   4. u = T (V beta), w = T (K beta e^gc), per part
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
  float* Ks = sm;                   // [64][kLp] K part; later K*beta*e^gc part
  float* Qs = Ks + kChunk * kLp;    // [64][kLp] Q part; later T [64][kLt]
  float* As = Qs + kChunk * kLt;    // [64][kLt] A (strict lower); later V*beta part [64][kLp]
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
  for (int part = 0; part < kDK / kPart; ++part) {
    __syncthreads();
    // normalized q, k part tiles: 64 rows x 32 cols, 4 per thread-step
    for (int idx = tid; idx < kChunk * kPart / 4; idx += kThreads) {
      const int r = idx / (kPart / 4), d = (idx % (kPart / 4)) * 4;
      float q[4] = {0.f, 0.f, 0.f, 0.f}, k[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < nvalid) {
        load4(row(r) + kh * kDK + part * kPart + d, q);
        load4(row(r) + key_dim + kh * kDK + part * kPart + d, k);
      }
      st4(Qs + r * kLp + d, q[0] * rq[r], q[1] * rq[r], q[2] * rq[r], q[3] * rq[r]);
      st4(Ks + r * kLp + d, k[0] * rk[r], k[1] * rk[r], k[2] * rk[r], k[3] * rk[r]);
    }
    __syncthreads();
    if (tj <= ti) {
#pragma unroll 2
      for (int d = 0; d < kPart; d += 4) {
        float4 ki[4], qi[4], kj[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          ki[x] = ld4(Ks + (i0 + x) * kLp + d);
          qi[x] = ld4(Qs + (i0 + x) * kLp + d);
          kj[x] = ld4(Ks + (j0 + x) * kLp + d);
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
    float* qg = qg_out + hc * kChunk * kDK + part * kPart;
    float* kd = kd_out + hc * kChunk * kDK + part * kPart;
    for (int idx = tid; idx < kChunk * kPart / 4; idx += kThreads) {
      const int r = idx / (kPart / 4), d = (idx % (kPart / 4)) * 4;
      const float4 q = ld4(Qs + r * kLp + d), k = ld4(Ks + r * kLp + d);
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

  // ---- 4. u = T (V*beta), w = T (K*beta*e^gc), 2x4 tile per thread, per part
  float* Vs = As;
  for (int part = 0; part < kDK / kPart; ++part) {
    __syncthreads();
    for (int idx = tid; idx < kChunk * kPart / 4; idx += kThreads) {
      const int r = idx / (kPart / 4), d = (idx % (kPart / 4)) * 4;
      float v[4] = {0.f, 0.f, 0.f, 0.f}, k[4] = {0.f, 0.f, 0.f, 0.f};
      if (r < nvalid) {
        load4(row(r) + 2 * key_dim + h * kDK + part * kPart + d, v);
        load4(row(r) + key_dim + kh * kDK + part * kPart + d, k);
      }
      const float br = bt[r], kr = rk[r], kb = bt[r] * eg[r];
      st4(Vs + r * kLp + d, v[0] * br, v[1] * br, v[2] * br, v[3] * br);
      st4(Ks + r * kLp + d, (k[0] * kr) * kb, (k[1] * kr) * kb, (k[2] * kr) * kb, (k[3] * kr) * kb);
    }
    __syncthreads();
    const int tr = tid >> 3, tc = tid & 7;  // rows 2tr, 2tr+1; columns 4tc..4tc+3
    const int r0 = tr * 2, d0 = tc * 4;
    float su[2][4] = {}, sw[2][4] = {};
    for (int m = 0; m < r0 + 2; ++m) {
      const float t[2] = {Ts[r0 * kLt + m], Ts[(r0 + 1) * kLt + m]};
      const float4 v = ld4(Vs + m * kLp + d0), k = ld4(Ks + m * kLp + d0);
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        su[x][0] += t[x] * v.x; su[x][1] += t[x] * v.y; su[x][2] += t[x] * v.z; su[x][3] += t[x] * v.w;
        sw[x][0] += t[x] * k.x; sw[x][1] += t[x] * k.y; sw[x][2] += t[x] * k.z; sw[x][3] += t[x] * k.w;
      }
    }
    float* u = u_out + hc * kChunk * kDK + part * kPart;
    float* w = w_out + hc * kChunk * kDK + part * kPart;
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      st4(u + (r0 + x) * kDK + d0, su[x][0], su[x][1], su[x][2], su[x][3]);
      st4(w + (r0 + x) * kDK + d0, sw[x][0], sw[x][1], sw[x][2], sw[x][3]);
    }
  }
}

constexpr size_t kPrepareSmem = sizeof(float) * (2 * kChunk * kLt + kChunk * kLp + 6 * kChunk);

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

// ==================================================================== output
// One block per (chunk, head): o = qg @ S + aqk @ vnew as a 64x128 tile with
// K = 128 + 64 (S = 0, and the first term vanishes, on a sequence's first
// chunk), 4 rows x 8 columns per thread, K staged through smem 16 at a time.
// kNorm: out[t, h*128 + :] = rmsnorm(o_row) * w * silu(z) in the activation
// dtype; otherwise the raw fp32 o row.
constexpr int kOutK = 16;
template <typename T, bool kNorm>
__global__ void __launch_bounds__(kThreads)
gdn_output_kernel(const float* __restrict__ qg, const float* __restrict__ states, const float* __restrict__ aqk,
                  const float* __restrict__ vnew, const int* __restrict__ chunk_start,
                  const int* __restrict__ chunk_valid, const int* __restrict__ seq_start, int C, int cb,
                  const T* __restrict__ z, int64_t ldz, const float* __restrict__ gw, float eps,
                  void* __restrict__ out, int64_t ldo) {
  const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const size_t hc = size_t(h) * C + c;
  const int t0 = chunk_start[cb + c], nvalid = chunk_valid[cb + c];
  const bool with_state = states != nullptr && seq_start[t0] != t0;

  __shared__ __align__(16) float At[kOutK][kChunk + 4];  // A^T tile (k-major), padded
  __shared__ __align__(16) float Bs[kOutK][kDK];
  const int tr = tid >> 4, tc = tid & 15;  // rows 4tr.., columns 4tc.. and 64+4tc..
  float acc[4][8] = {};

  // K steps: [qg | aqk] x [S ; vnew]
  const int k_begin = with_state ? 0 : kDK, k_end = kDK + kChunk;
  const float* qg_t = qg + hc * kChunk * kDK;
  const float* aqk_t = aqk + hc * kChunk * kChunk;
  const float* s_t = with_state ? states + hc * kDK * kDK : nullptr;
  const float* vn_t = vnew + hc * kChunk * kDK;
  // global -> register staging: A 64x16 (one float4 per thread), B 16x128 (two)
  const int ar = tid >> 2, ak = (tid & 3) * 4;
  float4 ra, rb0, rb1;
  auto fetch = [&](int k0) {
    ra = k0 < kDK ? ld4(qg_t + ar * kDK + k0 + ak) : ld4(aqk_t + ar * kChunk + (k0 - kDK) + ak);
    const float* bsrc = k0 < kDK ? s_t + size_t(k0) * kDK : vn_t + size_t(k0 - kDK) * kDK;
    rb0 = ld4(bsrc + tid * 4);
    rb1 = ld4(bsrc + (tid + kThreads) * 4);
  };
  fetch(k_begin);
  for (int k0 = k_begin; k0 < k_end; k0 += kOutK) {
    __syncthreads();
    At[ak][ar] = ra.x, At[ak + 1][ar] = ra.y, At[ak + 2][ar] = ra.z, At[ak + 3][ar] = ra.w;
    reinterpret_cast<float4*>(&Bs[0][0])[tid] = rb0;
    reinterpret_cast<float4*>(&Bs[0][0])[tid + kThreads] = rb1;
    __syncthreads();
    if (k0 + kOutK < k_end) fetch(k0 + kOutK);
#pragma unroll
    for (int k = 0; k < kOutK; ++k) {
      const float4 a = *reinterpret_cast<const float4*>(&At[k][tr * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tc * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][64 + tc * 4]);
      const float av[4] = {a.x, a.y, a.z, a.w}, bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] += av[i] * bv[j];
    }
  }

  // epilogue: the 16 threads of a row group are lanes [0,16) or [16,32) of a warp
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = tr * 4 + i;
    const int64_t t = t0 + r;
    if constexpr (kNorm) {
      float sq = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) sq += acc[i][j] * acc[i][j];
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
      if (r >= nvalid) continue;
      const float rs = rsqrtf(sq / kDK + eps);
      T* op = static_cast<T*>(out) + t * ldo + h * kDK;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int col = half * 64 + tc * 4;
        float zz[4], y[4];
        load4(z + t * ldz + h * kDK + col, zz);
#pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = acc[i][half * 4 + j] * rs * gw[col + j] * siluf_(zz[j]);
        store4(op + col, y);
      }
    } else {
      if (r >= nvalid) continue;
      float* op = static_cast<float*>(out) + t * ldo + h * kDK;
      st4(op + tc * 4, acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
      st4(op + 64 + tc * 4, acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
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
  // qg, kd, w [64, dk]; u, vnew [64, dv]; aqk [64, 64]; states [dk, dv]; dl
  return int64_t(sizeof(float)) * (kChunk * (3 * dk + 2 * dv + kChunk) + dk * dv + 1);
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

GdnRecurrence gdn_chunk_recurrence(const GdnChunks& ch, const at::Tensor& chunk_offsets, int64_t num_seqs,
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
  // Every sequence has at least one chunk (sequences are non-empty), so
  // C == num_seqs in a single window means one chunk each: S stays 0 and vnew == u, no recurrence to run.
  if (chunk_begin == 0 && C == num_seqs && !carry.defined()) return {at::Tensor(), ch.u};
  c10::cuda::CUDAGuard guard(ch.qg.device());
  GdnRecurrence r{at::empty({H, C, kDK, DV}, ch.qg.options()), at::empty({H, C, kChunk, DV}, ch.qg.options())};
  if (C > 0) {
    const bool dbl = allow_double_buffer && device_has_cp_async() && max_smem_optin() >= int(state_smem(true));
    dim3 grid{unsigned(H), unsigned(num_seqs), unsigned(DV / kBV)};
    auto launch = [&](auto kernel, size_t smem) {
      ensure_smem(kernel, smem);
      kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
          ch.kd.data_ptr<float>(), ch.u.data_ptr<float>(), ch.w.data_ptr<float>(), ch.dl.data_ptr<float>(),
          chunk_offsets.data_ptr<int>(), r.states.data_ptr<float>(), r.vnew.data_ptr<float>(), int(C), int(DV),
          int(chunk_begin), carry.defined() ? carry.data_ptr<float>() : nullptr, int(num_seqs));
    };
    if (dbl) launch(gdn_state_kernel<true>, state_smem(true));
    else launch(gdn_state_kernel<false>, state_smem(false));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return r;
}

void gdn_chunk_output(const GdnChunks& ch, const GdnRecurrence& rec, const at::Tensor& chunk_start,
                      const at::Tensor& chunk_valid, const at::Tensor& seq_start, int64_t chunk_begin,
                      const at::Tensor& z, const at::Tensor& w, double eps, at::Tensor& out) {
  const int64_t H = ch.qg.size(0), C = ch.qg.size(1);
  WEMM_CHECK(ch.u.size(3) == kDK, "gdn_chunk_output: value dim must be 128");
  WEMM_CHECK(rec.vnew.is_contiguous() && rec.vnew.sizes() == ch.u.sizes(), "gdn_chunk_output: bad vnew");
  WEMM_CHECK(!rec.states.defined() || (rec.states.is_contiguous() && rec.states.numel() == H * C * kDK * kDK),
             "gdn_chunk_output: bad states");
  WEMM_CHECK(out.dim() == 2 && out.stride(1) == 1 && out.size(1) == H * kDK && out.stride(0) % 4 == 0 &&
                 out.size(0) == seq_start.numel() && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
             "gdn_chunk_output: out must be [L, H*128] with unit column stride");
  WEMM_CHECK(chunk_begin + C <= chunk_start.numel(), "gdn_chunk_output: bad window");
  if (C == 0) return;
  c10::cuda::CUDAGuard guard(out.device());
  const dim3 grid{unsigned(C), unsigned(H)};
  const float* states = rec.states.defined() ? rec.states.data_ptr<float>() : nullptr;
  auto launch = [&](auto kernel, const auto* zp, int64_t ldz, const float* wp) {
    kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        ch.qg.data_ptr<float>(), states, ch.aqk.data_ptr<float>(), rec.vnew.data_ptr<float>(),
        chunk_start.data_ptr<int>(), chunk_valid.data_ptr<int>(), seq_start.data_ptr<int>(), int(C),
        int(chunk_begin), zp, ldz, wp, float(eps), out.data_ptr(), out.stride(0));
  };
  if (!z.defined()) {
    WEMM_CHECK(out.scalar_type() == at::kFloat, "gdn_chunk_output: raw output must be fp32");
    launch(gdn_output_kernel<float, false>, static_cast<const float*>(nullptr), 0, nullptr);
  } else {
    WEMM_CHECK(is_act(z) && z.stride(1) == 1 && z.stride(0) % 4 == 0 && z.size(1) == H * kDK &&
                   reinterpret_cast<uintptr_t>(z.data_ptr()) % 8 == 0 &&
                   z.size(0) == out.size(0) && out.scalar_type() == z.scalar_type(),
               "gdn_chunk_output: bad z/out");
    WEMM_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == kDK, "gdn_chunk_output: bad w");
    dispatch_act(z.scalar_type(), [&](auto tag) {
      using T = decltype(tag);
      launch(gdn_output_kernel<T, true>, ptr<T>(z), z.stride(0), w.data_ptr<float>());
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace wemm::kernels
