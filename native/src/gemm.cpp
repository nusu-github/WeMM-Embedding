#include "wemm/gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublasLt.h>

#include <cstdlib>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "wemm/common.h"

namespace wemm {

namespace {

#define LT_CHECK(expr)                                                                   \
  do {                                                                                   \
    cublasStatus_t st_ = (expr);                                                         \
    WEMM_CHECK(st_ == CUBLAS_STATUS_SUCCESS, #expr " failed with status ", int(st_));   \
  } while (0)

constexpr size_t kWorkspace = 32u << 20;

struct Key {
  int64_t m, n, k, lda, ldb;
  int epi;
  bool beta, f32;
  int device;
  bool operator==(const Key& o) const {
    return std::tie(m, n, k, lda, ldb, epi, beta, f32, device) ==
           std::tie(o.m, o.n, o.k, o.lda, o.ldb, o.epi, o.beta, o.f32, o.device);
  }
};
struct KeyHash {
  size_t operator()(const Key& k) const {
    size_t h = std::hash<int64_t>()(k.m);
    for (int64_t v : {k.n, k.k, k.lda, k.ldb, int64_t(k.epi), int64_t(k.beta), int64_t(k.f32), int64_t(k.device)})
      h = h * 1000003u ^ std::hash<int64_t>()(v);
    return h;
  }
};

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

std::mutex g_mu;
std::unordered_map<Key, Plan, KeyHash> g_plans;
// Autotuned algorithms, keyed by the plan key with M rounded to a bucket so
// that varying token counts reuse one tuning result.
std::unordered_map<Key, cublasLtMatmulAlgo_t, KeyHash> g_tuned;

int64_t m_bucket(int64_t m) {
  if (m <= 256) {
    int64_t b = 1;
    while (b < m) b <<= 1;
    return b;
  }
  return (m + 511) / 512 * 512;
}

bool autotune_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("WEMM_GEMM_AUTOTUNE");
    return !(e && std::string(e) == "0");
  }();
  return on;
}

}  // namespace

at::Tensor gemm(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                const std::optional<at::Tensor>& residual, Epilogue epilogue, at::ScalarType out_dtype) {
  WEMM_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "gemm: shape mismatch");
  WEMM_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "gemm: bf16 only");
  WEMM_CHECK(x.stride(1) == 1 && w.is_contiguous(), "gemm: x rows must be unit-stride, w contiguous");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  if (residual) out_dtype = residual->scalar_type();
  WEMM_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kFloat, "gemm: output must be bf16 or fp32");
  const bool f32 = out_dtype == at::kFloat;
  at::Tensor y = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0) return y;
  if (residual) WEMM_CHECK(residual->is_contiguous() && residual->size(0) == M && residual->size(1) == N,
                           "gemm: bad residual");
  if (bias) WEMM_CHECK(bias->is_contiguous() && bias->numel() == N &&
                           bias->scalar_type() == (f32 ? at::kFloat : at::kBFloat16),
                       "gemm: bias must be contiguous and match the output dtype");

  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
  if (bias && epilogue == Epilogue::Gelu) epi = CUBLASLT_EPILOGUE_GELU_BIAS;
  else if (bias) epi = CUBLASLT_EPILOGUE_BIAS;
  else if (epilogue == Epilogue::Gelu) epi = CUBLASLT_EPILOGUE_GELU;

  const int device = x.get_device();
  // Column-major view: Y^T[N,M] = W[N,K] (A, transposed) * X^T[K,M] (B).
  Key key{M, N, K, w.stride(0), x.stride(0), int(epi), residual.has_value(), f32, device};
  Plan* plan;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
      Plan p;
      LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
      if (bias) {
        cudaDataType_t bt = f32 ? CUDA_R_32F : CUDA_R_16BF;
        LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
      }
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, w.stride(0)));
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, x.stride(0)));
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N));
      cublasLtMatmulPreference_t pref;
      LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
      size_t ws = kWorkspace;
      LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      // All pointers come from the caching allocator or are row slices of
      // 16-byte aligned rows; require 16-byte alignment.
      uint32_t align = 16;
      for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, attr, &align, sizeof(align)));
      constexpr int kMaxAlgos = 8;
      cublasLtMatmulHeuristicResult_t res[kMaxAlgos]{};
      int n_res = 0;
      LT_CHECK(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.a, p.b, p.c, p.c, pref,
                                              autotune_enabled() ? kMaxAlgos : 1, res, &n_res));
      cublasLtMatmulPreferenceDestroy(pref);
      WEMM_CHECK(n_res > 0, "gemm: no cuBLASLt algorithm for M=", M, " N=", N, " K=", K);
      p.algo = res[0].algo;
      if (n_res > 1) {
        Key bkey = key;
        bkey.m = m_bucket(M);
        auto tuned = g_tuned.find(bkey);
        bool reused = false;
        if (tuned != g_tuned.end()) {
          cublasLtMatmulHeuristicResult_t chk{};
          if (cublasLtMatmulAlgoCheck(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.a, p.b, p.c, p.c,
                                      &tuned->second, &chk) == CUBLAS_STATUS_SUCCESS &&
              chk.workspaceSize <= kWorkspace) {
            p.algo = tuned->second;
            reused = true;
          }
        }
        if (!reused) {
          // Time each candidate on the real operands (outputs go to scratch).
          auto stream = at::cuda::getCurrentCUDAStream();
          at::Tensor scratch = at::empty({M, N}, x.options().dtype(out_dtype));
          at::Tensor ws = at::empty({int64_t(kWorkspace)}, x.options().dtype(at::kByte));
          if (bias) {
            const void* bp = bias->data_ptr();
            LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
          }
          const float one = 1.f, zero = 0.f;
          cudaEvent_t e0, e1;
          cudaEventCreate(&e0);
          cudaEventCreate(&e1);
          float best = 1e30f;
          for (int i = 0; i < n_res; ++i) {
            if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > kWorkspace) continue;
            auto run = [&] {
              return cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), p.op, &one, w.data_ptr(), p.a,
                                    x.data_ptr(), p.b, residual ? &one : &zero,
                                    residual ? residual->data_ptr() : scratch.data_ptr(), p.c, scratch.data_ptr(), p.c,
                                    &res[i].algo, ws.data_ptr(), kWorkspace, stream);
            };
            if (run() != CUBLAS_STATUS_SUCCESS) continue;  // warm-up
            cudaEventRecord(e0, stream);
            for (int r = 0; r < 3; ++r) run();
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) {
              best = ms;
              p.algo = res[i].algo;
            }
          }
          cudaEventDestroy(e0);
          cudaEventDestroy(e1);
          g_tuned[bkey] = p.algo;
        }
      }
      it = g_plans.emplace(key, p).first;
    }
    plan = &it->second;
  }

  if (bias) {
    const void* bp = bias->data_ptr();
    LT_CHECK(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
  }
  WEMM_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.stride(0) * 2) % 16 == 0,
             "gemm: x must be 16-byte aligned");
  const float alpha = 1.f, beta = residual ? 1.f : 0.f;
  at::Tensor ws = at::empty({int64_t(kWorkspace)}, x.options().dtype(at::kByte));
  LT_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->op, &alpha, w.data_ptr(), plan->a,
                          x.data_ptr(), plan->b, &beta, residual ? residual->data_ptr() : y.data_ptr(), plan->c,
                          y.data_ptr(), plan->c, &plan->algo, ws.data_ptr(), kWorkspace,
                          at::cuda::getCurrentCUDAStream()));
  return y;
}

void gemm_f32_batched(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, bool accumulate) {
  WEMM_CHECK(a.dim() == 3 && b.dim() == 3 && out.dim() == 3, "gemm_f32_batched: expects 3-D tensors");
  WEMM_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "gemm_f32_batched: contiguous only");
  WEMM_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
             "gemm_f32_batched: fp32 only");
  const int64_t B = a.size(0), M = a.size(1), K = a.size(2), N = b.size(2);
  WEMM_CHECK(b.size(0) == B && b.size(1) == K && out.size(0) == B && out.size(1) == M && out.size(2) == N,
             "gemm_f32_batched: shape mismatch");
  if (B == 0) return;
  // Column-major: out^T[N,M] = b^T[N,K] * a^T[K,M]; both operands untransposed.
  cublasLtMatmulDesc_t op;
  LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F));
  cublasLtMatrixLayout_t la, lb, lc;
  LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, N, K, N));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
  const int32_t bc = int32_t(B);
  const int64_t sa = K * N, sb = M * K, sc = M * N;
  for (auto [l, stride] : {std::pair{la, sa}, std::pair{lb, sb}, std::pair{lc, sc}}) {
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
  }
  const float alpha = 1.f, beta = accumulate ? 1.f : 0.f;
  at::Tensor ws = at::empty({int64_t(kWorkspace)}, a.options().dtype(at::kByte));
  LT_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), op, &alpha, b.data_ptr(), la, a.data_ptr(), lb,
                          &beta, out.data_ptr(), lc, out.data_ptr(), lc, nullptr, ws.data_ptr(), kWorkspace,
                          at::cuda::getCurrentCUDAStream()));
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
}

}  // namespace wemm
