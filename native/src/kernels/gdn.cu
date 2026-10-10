#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "wemm/common.h"
#include "wemm/kernels.h"

namespace wemm::kernels {

namespace {

constexpr int kChunk = 64;
constexpr int kDK = 128;
constexpr int kBV = 32;       // value columns per block
constexpr int kThreads = 256;

// One block owns S[:, v0:v0+kBV] of one (head, sequence) and walks its chunks.
__global__ void __launch_bounds__(kThreads)
gdn_recurrence_kernel(const float* __restrict__ qg, const float* __restrict__ kd, const float* __restrict__ u,
                      const float* __restrict__ w, const float* __restrict__ aqk, const float* __restrict__ dl,
                      const int* __restrict__ chunk_offsets, float* __restrict__ out, int C, int DV) {
  const int h = blockIdx.x;
  const int seq = blockIdx.y;
  const int v0 = blockIdx.z * kBV;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float* S = smem;                          // [kDK][kBV]
  float* buf = S + kDK * kBV;               // [kChunk][kDK]  (w, qg, then kd)
  float* vn = buf + kChunk * kDK;           // [kChunk][kBV]
  float* A = vn + kChunk * kBV;             // [kChunk][kChunk]

  for (int i = tid; i < kDK * kBV; i += kThreads) S[i] = 0.f;

  const int c0 = chunk_offsets[seq], c1 = chunk_offsets[seq + 1];
  // thread mapping for [64 x 32] outputs: row r, 8 consecutive columns
  const int r = tid >> 2;
  const int cc = (tid & 3) * 8;
  // thread mapping for S update [128 x 32]: row k, 16 consecutive columns
  const int sk = tid >> 1;
  const int sc = (tid & 1) * 16;

  for (int c = c0; c < c1; ++c) {
    const size_t hc = size_t(h) * C + c;
    const float* w_c = w + hc * kChunk * kDK;
    const float* qg_c = qg + hc * kChunk * kDK;
    const float* kd_c = kd + hc * kChunk * kDK;
    const float* u_c = u + hc * kChunk * DV;
    const float* a_c = aqk + hc * kChunk * kChunk;
    float* o_c = out + hc * kChunk * DV;

    __syncthreads();  // S ready / previous iteration done with buffers
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads)
      reinterpret_cast<float4*>(buf)[i] = reinterpret_cast<const float4*>(w_c)[i];
    for (int i = tid; i < kChunk * kChunk / 4; i += kThreads)
      reinterpret_cast<float4*>(A)[i] = reinterpret_cast<const float4*>(a_c)[i];
    __syncthreads();

    // vnew = u - w @ S
    {
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.f;
      const float* wr = buf + r * kDK;
#pragma unroll 4
      for (int k = 0; k < kDK; ++k) {
        const float a = wr[k];
        const float4 s0 = *reinterpret_cast<const float4*>(S + k * kBV + cc);
        const float4 s1 = *reinterpret_cast<const float4*>(S + k * kBV + cc + 4);
        acc[0] += a * s0.x; acc[1] += a * s0.y; acc[2] += a * s0.z; acc[3] += a * s0.w;
        acc[4] += a * s1.x; acc[5] += a * s1.y; acc[6] += a * s1.z; acc[7] += a * s1.w;
      }
      const float* ur = u_c + size_t(r) * DV + v0 + cc;
#pragma unroll
      for (int j = 0; j < 8; ++j) vn[r * kBV + cc + j] = ur[j] - acc[j];
    }
    __syncthreads();
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads)
      reinterpret_cast<float4*>(buf)[i] = reinterpret_cast<const float4*>(qg_c)[i];
    __syncthreads();

    // o = qg @ S + A @ vnew
    {
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.f;
      const float* qr = buf + r * kDK;
#pragma unroll 4
      for (int k = 0; k < kDK; ++k) {
        const float a = qr[k];
        const float4 s0 = *reinterpret_cast<const float4*>(S + k * kBV + cc);
        const float4 s1 = *reinterpret_cast<const float4*>(S + k * kBV + cc + 4);
        acc[0] += a * s0.x; acc[1] += a * s0.y; acc[2] += a * s0.z; acc[3] += a * s0.w;
        acc[4] += a * s1.x; acc[5] += a * s1.y; acc[6] += a * s1.z; acc[7] += a * s1.w;
      }
      const float* ar = A + r * kChunk;
      for (int j2 = 0; j2 <= r; ++j2) {  // A is lower triangular
        const float a = ar[j2];
        const float4 s0 = *reinterpret_cast<const float4*>(vn + j2 * kBV + cc);
        const float4 s1 = *reinterpret_cast<const float4*>(vn + j2 * kBV + cc + 4);
        acc[0] += a * s0.x; acc[1] += a * s0.y; acc[2] += a * s0.z; acc[3] += a * s0.w;
        acc[4] += a * s1.x; acc[5] += a * s1.y; acc[6] += a * s1.z; acc[7] += a * s1.w;
      }
      float* orow = o_c + size_t(r) * DV + v0 + cc;
#pragma unroll
      for (int j = 0; j < 8; ++j) orow[j] = acc[j];
    }
    __syncthreads();
    for (int i = tid; i < kChunk * kDK / 4; i += kThreads)
      reinterpret_cast<float4*>(buf)[i] = reinterpret_cast<const float4*>(kd_c)[i];
    __syncthreads();

    // S = S * dl + kd^T @ vnew
    {
      const float d = dl[hc];
      float acc[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) acc[j] = 0.f;
#pragma unroll 4
      for (int rr = 0; rr < kChunk; ++rr) {
        const float a = buf[rr * kDK + sk];
        const float* vr = vn + rr * kBV + sc;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
          const float4 s = *reinterpret_cast<const float4*>(vr + j);
          acc[j] += a * s.x; acc[j + 1] += a * s.y; acc[j + 2] += a * s.z; acc[j + 3] += a * s.w;
        }
      }
      float* srow = S + sk * kBV + sc;
#pragma unroll
      for (int j = 0; j < 16; ++j) srow[j] = srow[j] * d + acc[j];
    }
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
  at::Tensor out = at::empty({H, C, kChunk, DV}, qg.options());
  const size_t smem = sizeof(float) * (kDK * kBV + kChunk * kDK + kChunk * kBV + kChunk * kChunk);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_recurrence_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    attr_set = true;
  }
  dim3 grid(unsigned(H), unsigned(num_seqs), unsigned(DV / kBV));
  gdn_recurrence_kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      qg.data_ptr<float>(), kd.data_ptr<float>(), u.data_ptr<float>(), w.data_ptr<float>(), aqk.data_ptr<float>(),
      dl.data_ptr<float>(), chunk_offsets.data_ptr<int>(), out.data_ptr<float>(), int(C), int(DV));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace wemm::kernels
