// GEMM for few rows (M <= 64) on compressed weights (wemm/weight.h):
// y (+)= x @ W^T, streaming the 96-byte group records once and decoding them
// straight into tensor-core fragments (no dense temporary). Short inputs are
// bound by the weight bandwidth.
//
// Each warp owns 16 W rows and a range of 64-wide k steps, keeps its W loads
// kDepth steps ahead in registers, and reads x (small, cache resident)
// directly into A fragments: no shared memory and no block barriers in the
// main loop. Within a step, lane quad position q handles k = 16q..16q+15:
// the same permutation of k on both mma operands leaves the product
// unchanged and makes every lane's loads contiguous. K is split across
// blocks to fill the GPU; the last block of each row tile sums the fp32
// partials in a fixed order (deterministic) and applies the epilogue.
// sm_80+ (mma.m16n8k16).

#include "common.cuh"
#include "wemm/kernels.h"

namespace wemm::kernels {

using namespace detail;

namespace {

constexpr int kWarps = 4, kThreads = kWarps * 32;
constexpr int kRows = kWarps * 16;  // W rows per block
constexpr int kStep = 64;           // k per step (one compression group per row)
[[maybe_unused]] constexpr int kDepth = 3;  // W steps in flight per warp

struct Params {
  const void* x;
  int64_t ldx;
  int M, N, K;
  const uint8_t* groups;  // records [N*K/64, 96]
  const uint8_t* spill;
  int base;
  const void* bias;
  bool bias_f32;
  void* out;  // [M, N] contiguous; also the addend when accumulating
  bool out_f32, accumulate;
  float* partial;  // [ksplit, M, N] when ksplit > 1
  int* counters;   // [N / kRows], zeroed
  int ksplit, steps_per_split;
};

template <typename T>
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
  if constexpr (std::is_same_v<T, bf16>)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One lane's raw W data for one row and step (elements k = 16q..16q+15):
// a = its 16 sign/mantissa bytes, b = record header half q (q < 2):
// planes p0 p1 | p2, spill offset, inline outlier exponents.
struct Raw {
  uint4 a, b;
};

[[maybe_unused]] __device__ __forceinline__ void load_raw(const Params& p, int64_t row, int step, int q, Raw& r) {
  const uint4* rec = reinterpret_cast<const uint4*>(p.groups + (row * (p.K / kStep) + step) * 96);
  r.a = __ldg(rec + q);
  if (q < 2) r.b = __ldg(rec + 4 + q);
}

// Per-row decode state of a compressed step: the lane's 16-bit slices of the
// three code planes, the rank of its first outlier within the group, and the
// group's spill offset and inline outlier exponents.
struct Hdr {
  uint32_t c01, c2;  // c0 | c1 << 16, c2
  int rank, spill_off;
  uint32_t inl;
};

[[maybe_unused]] __device__ __forceinline__ void prepare(const Raw& r, int lane, Hdr& h) {
  const int qb = lane & ~3, sh = 16 * (lane & 3);
  const uint32_t w0 = __shfl_sync(0xffffffffu, r.b.x, qb), w1 = __shfl_sync(0xffffffffu, r.b.y, qb),
                 w2 = __shfl_sync(0xffffffffu, r.b.z, qb), w3 = __shfl_sync(0xffffffffu, r.b.w, qb);
  const uint32_t w4 = __shfl_sync(0xffffffffu, r.b.x, qb + 1), w5 = __shfl_sync(0xffffffffu, r.b.y, qb + 1);
  h.spill_off = int(__shfl_sync(0xffffffffu, r.b.z, qb + 1));
  h.inl = __shfl_sync(0xffffffffu, r.b.w, qb + 1);
  const uint64_t p0 = uint64_t(w0) | uint64_t(w1) << 32, p1 = uint64_t(w2) | uint64_t(w3) << 32,
                 p2 = uint64_t(w4) | uint64_t(w5) << 32;
  h.c01 = (uint32_t(p0 >> sh) & 0xffffu) | uint32_t(p1 >> sh) << 16;
  h.c2 = uint32_t(p2 >> sh) & 0xffffu;
  h.rank = __popcll(~(p0 | p1 | p2) & ((1ull << sh) - 1));
}

// bit i of a nibble -> byte i (0/1)
[[maybe_unused]] __device__ __forceinline__ uint32_t nibble_bytes(uint32_t n) { return (n * 0x00204081u) & 0x01010101u; }

// B fragments for the 4 k16 sub-steps: b[s] = elements 16q + 4s + {0,1} and {2,3}.
template <typename T>
__device__ __forceinline__ void decode(const Params& p, const Raw& r, const Hdr& h, uint32_t (*b)[2]) {
  {
    const uint32_t c0 = h.c01 & 0xffffu, c1 = h.c01 >> 16, c2 = h.c2;
    const uint32_t om = ~(c0 | c1 | c2) & 0xffffu;  // outliers (code 0)
    int rank = h.rank;
    const uint32_t base4 = uint32_t(p.base) * 0x01010101u;
    const uint32_t sm[4] = {r.a.x, r.a.y, r.a.z, r.a.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // elements 4i .. 4i+3
      // exponent bytes: base + code, outliers patched below
      uint32_t ex = base4 + (nibble_bytes((c0 >> 4 * i) & 0xf) | nibble_bytes((c1 >> 4 * i) & 0xf) << 1 |
                             nibble_bytes((c2 >> 4 * i) & 0xf) << 2);
      for (uint32_t o = (om >> 4 * i) & 0xf; o; o &= o - 1) {
        const int byte = __ffs(o) - 1;
        const uint32_t e = rank < 4 ? (h.inl >> 8 * rank) & 0xff : p.spill[h.spill_off + rank - 4];
        ++rank;
        ex = (ex & ~(0xffu << 8 * byte)) | e << 8 * byte;
      }
      // bf16 pairs: x = [sm_j, ex_j, sm_j+1, ex_j+1] -> sign << 15 | ex << 7 | mant per half
#pragma unroll
      for (int pr = 0; pr < 2; ++pr) {
        const uint32_t x = __byte_perm(sm[i], ex, pr ? 0x7362 : 0x5140);
        uint32_t v = ((x >> 1) & 0x7f807f80u) | (x & 0x007f007fu) | ((x & 0x00800080u) << 8);
        if constexpr (!std::is_same_v<T, bf16>) {  // exact bf16 -> fp16 (round to nearest even)
          const __half2 hv = __floats2half2_rn(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
          v = *reinterpret_cast<const uint32_t*>(&hv);
        }
        b[i][pr] = v;
      }
    }
  }
}

template <typename T>
__device__ __forceinline__ float load_out(const Params& p, int64_t idx) {
  return p.out_f32 ? static_cast<const float*>(p.out)[idx] : to_f(static_cast<const T*>(p.out)[idx]);
}

template <typename T, int MT>
__global__ void __launch_bounds__(kThreads) skinny_kernel(Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __shared__ int last_flag;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3;
  const int tile = blockIdx.x, split = blockIdx.y;
  const int n0 = tile * kRows + warp * 16;
  const int s0 = split * p.steps_per_split, s1 = min(p.K / kStep, s0 + p.steps_per_split);

  float acc[MT][2][4] = {};
  Raw raw[kDepth][2];
  Hdr hdr[2];
#pragma unroll
  for (int d = 0; d < kDepth; ++d)
    if (s0 + d < s1)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) load_raw(p, n0 + nt * 8 + g, s0 + d, q, raw[d][nt]);
  if (s0 < s1)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) prepare(raw[0][nt], lane, hdr[nt]);

  for (int sb = s0; sb < s1; sb += kDepth) {
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      const int s = sb + d;
      if (s >= s1) break;
      uint32_t b[2][4][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) decode<T>(p, raw[d][nt], hdr[nt], b[nt]);
      if (s + 1 < s1)  // the next step's records were requested kDepth - 1 steps ago
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) prepare(raw[(d + 1) % kDepth][nt], lane, hdr[nt]);
      if (s + kDepth < s1)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) load_raw(p, n0 + nt * 8 + g, s + kDepth, q, raw[d][nt]);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        // A rows mt*16 + g and + 8, elements 16q..16q+15 of this step
        uint4 xa[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = mt * 16 + g + 8 * h;
          if (m < p.M) {
            const uint4* xp = reinterpret_cast<const uint4*>(static_cast<const T*>(p.x) + m * p.ldx +
                                                             int64_t(s) * kStep + 16 * q);
            xa[h][0] = __ldg(xp);
            xa[h][1] = __ldg(xp + 1);
          } else {
            xa[h][0] = xa[h][1] = make_uint4(0, 0, 0, 0);
          }
        }
        const uint32_t* x0 = reinterpret_cast<const uint32_t*>(xa[0]);
        const uint32_t* x1 = reinterpret_cast<const uint32_t*>(xa[1]);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const uint32_t a[4] = {x0[2 * ks], x1[2 * ks], x0[2 * ks + 1], x1[2 * ks + 1]};
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) mma<T>(acc[mt][nt], a, b[nt][ks]);
        }
      }
    }
  }

  // Epilogue. Accumulator (mt, nt): rows mt*16 + g (+8), columns n0 + nt*8 + 2q (+1).
  auto finish = [&](int m, int n, float v) {
    const int64_t idx = int64_t(m) * p.N + n;
    if (p.bias) v += p.bias_f32 ? static_cast<const float*>(p.bias)[n] : to_f(static_cast<const T*>(p.bias)[n]);
    if (p.accumulate) v += load_out<T>(p, idx);
    if (p.out_f32) static_cast<float*>(p.out)[idx] = v;
    else static_cast<T*>(p.out)[idx] = from_f<T>(v);
  };
  if (p.ksplit == 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = mt * 16 + g + (e >> 1) * 8, n = n0 + nt * 8 + 2 * q + (e & 1);
          if (m < p.M) finish(m, n, acc[mt][nt][e]);
        }
    return;
  }
  const int64_t MN = int64_t(p.M) * p.N;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + g + (e >> 1) * 8, n = n0 + nt * 8 + 2 * q + (e & 1);
        if (m < p.M) p.partial[split * MN + int64_t(m) * p.N + n] = acc[mt][nt][e];
      }
  __threadfence();
  __syncthreads();
  if (tid == 0) last_flag = atomicAdd(p.counters + tile, 1) == p.ksplit - 1;
  __syncthreads();
  if (!last_flag) return;
  __threadfence();
  // last block of this row tile: sum the partials in split order
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + g + (e >> 1) * 8, n = n0 + nt * 8 + 2 * q + (e & 1);
        if (m >= p.M) continue;
        float v = 0.f;
        for (int sp = 0; sp < p.ksplit; ++sp) v += __ldcg(p.partial + sp * MN + int64_t(m) * p.N + n);
        finish(m, n, v);
      }
#endif
}

}  // namespace

bool skinny_gemm_supported(int64_t M, int64_t N, int64_t K, const at::Tensor& x) {
  return M >= 1 && M <= 64 && N % kRows == 0 && K % kStep == 0 && x.is_cuda() && is_act(x) && x.stride(1) == 1 &&
         x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
         at::cuda::getDeviceProperties(x.get_device())->major >= 8;
}

void skinny_gemm(const at::Tensor& x, const SkinnyWeight& w, const at::Tensor& bias, at::Tensor& out,
                 bool accumulate) {
  const int64_t M = x.size(0), K = x.size(1), N = out.size(1);
  WEMM_CHECK(skinny_gemm_supported(M, N, K, x) && out.is_contiguous() && out.size(0) == M &&
                 (out.scalar_type() == at::kFloat || out.scalar_type() == x.scalar_type()),
             "skinny_gemm: unsupported arguments");
  WEMM_CHECK(w.groups.numel() == N * K / 64 * 96, "skinny_gemm: bad weight");
  WEMM_CHECK(!bias.defined() || (bias.is_contiguous() && bias.numel() == N &&
                                 (bias.scalar_type() == at::kFloat || bias.scalar_type() == x.scalar_type())),
             "skinny_gemm: bad bias");
  c10::cuda::CUDAGuard guard(x.device());
  const int tiles = int(N / kRows), steps = int(K / kStep);
  const int sms = at::cuda::getDeviceProperties(x.get_device())->multiProcessorCount;
  // Split K only when there are fewer row tiles than SMs (then aim for ~2
  // blocks per SM, >= 2 steps each): the partial sums cost more than they
  // gain otherwise.
  int ksplit = tiles >= sms ? 1 : std::max(1, std::min((2 * sms + tiles - 1) / tiles, steps / 2));
  const int per = (steps + ksplit - 1) / ksplit;
  ksplit = (steps + per - 1) / per;

  Params p{};
  p.x = x.data_ptr();
  p.ldx = x.stride(0);
  p.M = int(M), p.N = int(N), p.K = int(K);
  p.groups = w.groups.data_ptr<uint8_t>();
  p.spill = w.spill.data_ptr<uint8_t>();
  p.base = w.base;
  p.bias = bias.defined() ? bias.data_ptr() : nullptr;
  p.bias_f32 = bias.defined() && bias.scalar_type() == at::kFloat;
  p.out = out.data_ptr();
  p.out_f32 = out.scalar_type() == at::kFloat;
  p.accumulate = accumulate;
  p.ksplit = ksplit;
  p.steps_per_split = per;
  at::Tensor partial, counters;
  if (ksplit > 1) {
    partial = at::empty({ksplit, M, N}, x.options().dtype(at::kFloat));
    counters = at::zeros({tiles}, x.options().dtype(at::kInt));
    p.partial = partial.data_ptr<float>();
    p.counters = counters.data_ptr<int>();
  }
  const dim3 grid{unsigned(tiles), unsigned(ksplit)};
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_act(x.scalar_type(), [&](auto tag) {
    using T = decltype(tag);
    auto launch = [&](auto mt_tag) {
      constexpr int MT = decltype(mt_tag)::value;
      skinny_kernel<T, MT><<<grid, kThreads, 0, stream>>>(p);
    };
    if (M <= 16) launch(std::integral_constant<int, 1>{});
    else if (M <= 32) launch(std::integral_constant<int, 2>{});
    else if (M <= 48) launch(std::integral_constant<int, 3>{});
    else launch(std::integral_constant<int, 4>{});
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace wemm::kernels
