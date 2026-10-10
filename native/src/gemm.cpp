#include "wemm/gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

#include "wemm/common.h"

namespace wemm {

namespace {

#define LT_CHECK(expr)                                                                 \
  do {                                                                                 \
    cublasStatus_t st_ = (expr);                                                       \
    WEMM_CHECK(st_ == CUBLAS_STATUS_SUCCESS, #expr " failed with status ", int(st_)); \
  } while (0)

constexpr size_t kWorkspace = 32u << 20;
constexpr int kMaxAlgos = 8;

// Our own cuBLASLt handle per device: LibTorch distributions bundle a
// privately named cuBLASLt, so a handle obtained from ATen must not be mixed
// with the library this file links against.
cublasLtHandle_t lt_handle() {
  static std::mutex mu;
  static std::map<int, cublasLtHandle_t> handles;
  thread_local int cached_dev = -1;
  thread_local cublasLtHandle_t cached = nullptr;
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev == cached_dev) return cached;
  std::lock_guard<std::mutex> lock(mu);
  auto it = handles.find(dev);
  if (it == handles.end()) {
    cublasLtHandle_t h;
    LT_CHECK(cublasLtCreate(&h));
    it = handles.emplace(dev, h).first;
  }
  cached_dev = dev;
  cached = it->second;
  return cached;
}

bool autotune_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("WEMM_GEMM_AUTOTUNE");
    return !(e && std::string(e) == "0");
  }();
  return on;
}

// ---- RAII wrappers for the per-call descriptors
struct OpDesc {
  cublasLtMatmulDesc_t d = nullptr;
  OpDesc(cublasComputeType_t ct, cudaDataType_t st) { LT_CHECK(cublasLtMatmulDescCreate(&d, ct, st)); }
  ~OpDesc() { cublasLtMatmulDescDestroy(d); }
  template <typename V>
  void set(cublasLtMatmulDescAttributes_t a, const V& v) {
    LT_CHECK(cublasLtMatmulDescSetAttribute(d, a, &v, sizeof(v)));
  }
};
struct Layout {
  cublasLtMatrixLayout_t l = nullptr;
  Layout(cudaDataType_t t, uint64_t rows, uint64_t cols, int64_t ld) {
    LT_CHECK(cublasLtMatrixLayoutCreate(&l, t, rows, cols, ld));
  }
  ~Layout() { cublasLtMatrixLayoutDestroy(l); }
  void batch(int32_t count, int64_t stride) {
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
  }
};
struct Pref {
  cublasLtMatmulPreference_t p = nullptr;
  Pref() {
    LT_CHECK(cublasLtMatmulPreferenceCreate(&p));
    size_t ws = kWorkspace;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    // The heuristic otherwise assumes 256-byte aligned pointers; row-block
    // views are only guaranteed 16-byte aligned (checked in gemm_impl).
    uint32_t align = 16;
    for (auto a : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                   CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      LT_CHECK(cublasLtMatmulPreferenceSetAttribute(p, a, &align, sizeof(align)));
  }
  ~Pref() { cublasLtMatmulPreferenceDestroy(p); }
};
struct Events {
  cudaEvent_t a = nullptr, b = nullptr;
  Events() {
    C10_CUDA_CHECK(cudaEventCreate(&a));
    C10_CUDA_CHECK(cudaEventCreate(&b));
  }
  ~Events() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
};

// Algorithm cache: shape with M rounded to a bucket, so varying token counts
// share one tuning result and the cache stays bounded.
using AlgoKey = std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int>;
std::mutex g_mu;
std::map<AlgoKey, cublasLtMatmulAlgo_t> g_algos;

int64_t m_bucket(int64_t m) {
  if (m <= 256) {
    int64_t b = 1;
    while (b < m) b <<= 1;
    return b;
  }
  return (m + 511) / 512 * 512;
}

cudaDataType_t cuda_type(at::ScalarType t) {
  switch (t) {
    case at::kBFloat16: return CUDA_R_16BF;
    case at::kHalf: return CUDA_R_16F;
    case at::kFloat: return CUDA_R_32F;
    default: throw Error(std::string("gemm: unsupported dtype ") + c10::toString(t));
  }
}

struct Problem {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  const void *A, *B, *C;
  void* D;
  float beta;
};

cublasStatus_t run(const Problem& p, const cublasLtMatmulAlgo_t* algo, void* ws, cudaStream_t stream) {
  const float alpha = 1.f;
  return cublasLtMatmul(lt_handle(), p.op, &alpha, p.A, p.a, p.B, p.b, &p.beta, p.C, p.c, p.D, p.c, algo, ws,
                        kWorkspace, stream);
}

// Cached algorithm for `key`, if it is valid for this problem.
std::optional<cublasLtMatmulAlgo_t> lookup_algo(const AlgoKey& key, const Problem& p) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_algos.find(key);
  if (it == g_algos.end()) return std::nullopt;
  cublasLtMatmulHeuristicResult_t chk{};
  if (cublasLtMatmulAlgoCheck(lt_handle(), p.op, p.a, p.b, p.c, p.c, &it->second, &chk) != CUBLAS_STATUS_SUCCESS ||
      chk.workspaceSize > kWorkspace)
    return std::nullopt;
  return it->second;
}

// Picks an algorithm (the heuristic's first choice, or the fastest of its top
// candidates when autotuning) and caches it. Timing runs write p.D, so the
// caller points D at scratch memory when C aliases the real output.
cublasLtMatmulAlgo_t tune_algo(const AlgoKey& key, const Problem& p, void* ws, cudaStream_t stream, bool autotune) {
  Pref pref;
  cublasLtMatmulHeuristicResult_t res[kMaxAlgos]{};
  int n = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle(), p.op, p.a, p.b, p.c, p.c, pref.p, autotune ? kMaxAlgos : 1,
                                          res, &n));
  WEMM_CHECK(n > 0, "gemm: no cuBLASLt algorithm for this problem");
  cublasLtMatmulAlgo_t best = res[0].algo;
  if (n > 1) {
    Events ev;
    float best_ms = 1e30f;
    for (int i = 0; i < n; ++i) {
      if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > kWorkspace) continue;
      if (run(p, &res[i].algo, ws, stream) != CUBLAS_STATUS_SUCCESS) continue;  // warm-up
      C10_CUDA_CHECK(cudaEventRecord(ev.a, stream));
      for (int r = 0; r < 3; ++r) run(p, &res[i].algo, ws, stream);
      C10_CUDA_CHECK(cudaEventRecord(ev.b, stream));
      C10_CUDA_CHECK(cudaEventSynchronize(ev.b));
      float ms = 0;
      C10_CUDA_CHECK(cudaEventElapsedTime(&ms, ev.a, ev.b));
      if (ms < best_ms) {
        best_ms = ms;
        best = res[i].algo;
      }
    }
  }
  std::lock_guard<std::mutex> lock(g_mu);
  g_algos[key] = best;
  return best;
}

// Cached or freshly tuned algorithm. When C aliases D (in-place
// accumulation) tuning writes to a scratch output, never to the accumulator.
cublasLtMatmulAlgo_t algo_for(const AlgoKey& key, const Problem& p, const at::Tensor& out, void* ws,
                              cudaStream_t stream, bool autotune) {
  if (auto a = lookup_algo(key, p)) return *a;
  Problem tp = p;
  at::Tensor scratch;
  if (p.C == p.D) {
    scratch = at::empty_like(out);
    tp.C = tp.D = scratch.data_ptr();
    if (p.beta != 0.f) scratch.zero_();
  }
  return tune_algo(key, tp, ws, stream, autotune);
}

}  // namespace

namespace {

// D = x @ w^T (+ bias) (+ C). C may alias D (in-place accumulation).
void gemm_impl(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias, const at::Tensor* c,
               at::Tensor& d, Epilogue epilogue) {
  WEMM_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "gemm: shape mismatch");
  WEMM_CHECK((x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && w.scalar_type() == x.scalar_type(),
             "gemm: x and w must both be bf16 or fp16");
  WEMM_CHECK(x.stride(1) == 1 && w.is_contiguous(), "gemm: x rows must be unit-stride, w contiguous");
  WEMM_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.stride(0) * 2) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
             "gemm: operands must be 16-byte aligned");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  const at::ScalarType odt = d.scalar_type();
  WEMM_CHECK(odt == x.scalar_type() || odt == at::kFloat, "gemm: output must be fp32 or the input dtype");
  WEMM_CHECK(d.is_contiguous() && d.size(0) == M && d.size(1) == N, "gemm: bad output");
  if (c)
    WEMM_CHECK(c->is_contiguous() && c->scalar_type() == odt && c->size(0) == M && c->size(1) == N,
               "gemm: bad residual");
  if (bias)
    WEMM_CHECK(bias->is_contiguous() && bias->numel() == N && bias->scalar_type() == odt,
               "gemm: bias must be contiguous and match the output dtype");
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&d, c, bias ? &*bias : nullptr})
    WEMM_CHECK(!t || reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "gemm: output/residual/bias misaligned");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(x.device());

  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
  if (bias && epilogue == Epilogue::Gelu) epi = CUBLASLT_EPILOGUE_GELU_BIAS;
  else if (bias) epi = CUBLASLT_EPILOGUE_BIAS;
  else if (epilogue == Epilogue::Gelu) epi = CUBLASLT_EPILOGUE_GELU;

  // Column-major view: Y^T[N,M] = W[N,K] (A, transposed) * X^T[K,M] (B).
  const cudaDataType_t in_t = cuda_type(x.scalar_type()), out_t = cuda_type(odt);
  OpDesc op(CUBLAS_COMPUTE_32F, CUDA_R_32F);
  op.set(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
  op.set(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
  op.set(CUBLASLT_MATMUL_DESC_EPILOGUE, epi);
  if (bias) {
    op.set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_t);
    op.set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias->data_ptr());
  }
  Layout la(in_t, K, N, w.stride(0)), lb(in_t, K, M, x.stride(0)), lc(out_t, N, M, N);
  Problem p{op.d, la.l, lb.l, lc.l, w.data_ptr(), x.data_ptr(), c ? c->data_ptr() : d.data_ptr(), d.data_ptr(),
            c ? 1.f : 0.f};

  auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor ws = at::empty({int64_t(kWorkspace)}, x.options().dtype(at::kByte));
  const AlgoKey key{x.get_device(), m_bucket(M), N, K, w.stride(0), x.stride(0), int(epi), int(c != nullptr),
                    int(in_t), int(out_t), 0};
  const cublasLtMatmulAlgo_t algo = algo_for(key, p, d, ws.data_ptr(), stream, autotune_enabled());
  LT_CHECK(run(p, &algo, ws.data_ptr(), stream));
}

}  // namespace

at::Tensor gemm(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                const std::optional<at::Tensor>& residual, Epilogue epilogue,
                std::optional<at::ScalarType> out_dtype) {
  const at::ScalarType odt = residual ? residual->scalar_type() : out_dtype.value_or(x.scalar_type());
  at::Tensor y = at::empty({x.size(0), w.size(0)}, x.options().dtype(odt));
  gemm_impl(x, w, bias, residual ? &*residual : nullptr, y, epilogue);
  return y;
}

void gemm_accumulate(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                     at::Tensor& acc) {
  gemm_impl(x, w, bias, &acc, acc, Epilogue::None);
}

void gemm_f32_batched(const at::Tensor& a, const at::Tensor& b, at::Tensor& out, bool accumulate) {
  WEMM_CHECK(a.dim() == 3 && b.dim() == 3 && out.dim() == 3, "gemm_f32_batched: expects 3-D tensors");
  WEMM_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "gemm_f32_batched: contiguous only");
  WEMM_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
             "gemm_f32_batched: fp32 only");
  const int64_t B = a.size(0), M = a.size(1), K = a.size(2), N = b.size(2);
  WEMM_CHECK(b.size(0) == B && b.size(1) == K && out.size(0) == B && out.size(1) == M && out.size(2) == N,
             "gemm_f32_batched: shape mismatch");
  WEMM_CHECK(B <= INT32_MAX, "gemm_f32_batched: batch too large");
  if (B == 0) return;
  c10::cuda::CUDAGuard guard(a.device());
  // Column-major: out^T[N,M] = b^T[N,K] * a^T[K,M]; both operands untransposed.
  OpDesc op(CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F);
  Layout la(CUDA_R_32F, N, K, N), lb(CUDA_R_32F, K, M, K), lc(CUDA_R_32F, N, M, N);
  la.batch(int32_t(B), K * N);
  lb.batch(int32_t(B), M * K);
  lc.batch(int32_t(B), M * N);
  Problem p{op.d, la.l, lb.l, lc.l, b.data_ptr(), a.data_ptr(), out.data_ptr(), out.data_ptr(), accumulate ? 1.f : 0.f};
  auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor ws = at::empty({int64_t(kWorkspace)}, a.options().dtype(at::kByte));
  const AlgoKey key{a.get_device(), m_bucket(B), M, N, K, 0, -1, int(accumulate), int(CUDA_R_32F), int(CUDA_R_32F), 1};
  // The heuristic's pick is fine for these small batched tiles (no autotune).
  const cublasLtMatmulAlgo_t algo = algo_for(key, p, out, ws.data_ptr(), stream, /*autotune=*/false);
  LT_CHECK(run(p, &algo, ws.data_ptr(), stream));
}

}  // namespace wemm
