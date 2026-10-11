// Micro-benchmark: wemm::gemm (cuBLASLt + epilogues) vs ATen for model shapes.
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>

#include <chrono>
#include <cstdio>

#include "wemm/gemm.h"
#include "wemm/weight.h"

template <typename F>
double time_ms(F f, int iters = 20) {
  for (int i = 0; i < 3; ++i) f();
  at::cuda::getCurrentCUDAStream().synchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) f();
  at::cuda::getCurrentCUDAStream().synchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main() {
  c10::InferenceMode guard;
  auto o = at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  struct S { int64_t M, N, K; bool bias, gelu, resid32; const char* name; };
  S shapes[] = {
      {10560, 3072, 1024, true, false, false, "vit qkv+bias"},
      {10560, 1024, 1024, true, false, true, "vit proj+bias+res32"},
      {10560, 4096, 1024, true, true, false, "vit fc1+bias+gelu"},
      {10560, 1024, 4096, true, false, true, "vit fc2+bias+res32"},
      {2743, 8320, 2048, false, false, false, "lm in_proj"},
      {2743, 2048, 2048, false, false, true, "lm out_proj+res32"},
      {2743, 12288, 2048, false, false, false, "lm gate_up"},
      {2743, 2048, 6144, false, false, true, "lm down+res32"},
      {13, 12288, 2048, false, false, false, "lm gate_up M=13"},
      {13, 8224, 2048, false, false, false, "lm in_proj M=13"},
      {13, 2048, 2048, false, false, true, "lm out_proj M=13"},
      {13, 2048, 6144, false, false, true, "lm down M=13"},
      {64, 12288, 2048, false, false, false, "lm gate_up M=64"},
      {64, 2048, 6144, false, false, true, "lm down M=64"},
      {200, 12288, 2048, false, false, false, "lm gate_up M=200"},
  };
  for (auto& s : shapes) {
    at::Tensor x = at::randn({s.M, s.K}, o), w = at::randn({s.N, s.K}, o) * 0.02;
    at::Tensor b = at::randn({s.N}, o), r = at::randn({s.M, s.N}, o.dtype(at::kFloat));
    std::optional<at::Tensor> bias;
    if (s.bias) bias = s.resid32 ? b.to(at::kFloat) : b;
    std::optional<at::Tensor> res;
    if (s.resid32) res = r;
    double t_lt = time_ms([&] { wemm::gemm(x, w, bias, res, s.gelu ? wemm::Epilogue::Gelu : wemm::Epilogue::None); });
    double t_aten = time_ms([&] {
      at::Tensor y = s.bias ? at::linear(x, w, b) : at::linear(x, w);
      if (s.gelu) y = at::gelu(y, "tanh");
      if (s.resid32) y = r + y;
    });
    double t_plain = time_ms([&] { at::mm(x, w.t()); });
    std::printf("%-22s M=%6lld N=%6lld K=%5lld  lt=%.3f ms  aten=%.3f ms  plain_mm=%.3f ms (%.1f TFLOPS)\n", s.name,
                (long long)s.M, (long long)s.N, (long long)s.K, t_lt, t_aten, t_plain,
                2.0 * s.M * s.N * s.K / t_plain / 1e9);
    std::printf("%22s weight stream: lt %.0f GB/s\n", "", 2.0 * s.N * s.K / t_lt / 1e6);
  }

  // ---- few-row GEMM on Weight (dense and compressed) vs cuBLASLt
  std::printf("\nWeight GEMM (fp16): cuBLASLt dense | skinny compressed | decompress+cuBLASLt\n");
  struct T { int64_t M, N, K; bool acc; const char* name; };
  T ts[] = {{1, 12288, 2048, false, "gate_up"},  {13, 12288, 2048, false, "gate_up"}, {21, 8320, 2048, false, "in_proj"},
            {21, 2048, 2048, true, "out_proj+=f32"}, {21, 2048, 6144, true, "down+=f32"}, {40, 12288, 2048, false, "gate_up"},
            {64, 12288, 2048, false, "gate_up"},  {64, 2048, 6144, true, "down+=f32"}};
  for (auto& t : ts) {
    at::Tensor wh = (at::randn({t.N, t.K}) * 0.02).to(at::kBFloat16);
    // a few large-exponent outliers
    wh.view(-1).index_put_({at::randint(t.N * t.K, {64}, at::kLong)}, at::tensor(3.5).to(at::kBFloat16));
    const auto dev = at::Device(at::kCUDA);
    wemm::Weight wd = wemm::Weight::load(wh, at::kHalf, dev, false), wc = wemm::Weight::load(wh, at::kHalf, dev, true);
    at::Tensor x = at::randn({t.M, t.K}, o);
    at::Tensor acc0 = at::randn({t.M, t.N}, o.dtype(at::kFloat));
    auto run = [&](const wemm::Weight& w, bool via_dense) {
      if (t.acc) {
        at::Tensor a = acc0.clone();
        if (via_dense) wemm::gemm_accumulate(x, w.dense(), std::nullopt, a);
        else wemm::gemm_accumulate(x, w, std::nullopt, a);
        return a;
      }
      return via_dense ? wemm::gemm(x, w.dense()) : wemm::gemm(x, w);
    };
    at::Tensor ref = run(wd, true).to(at::kFloat), yc = run(wc, false).to(at::kFloat);
    const double ec = ((yc - ref).abs().max() / ref.abs().max()).item<double>();
    at::Tensor acc = acc0.clone();
    auto timed = [&](const wemm::Weight& w, bool via_dense) {
      return time_ms([&] {
        if (t.acc) {
          if (via_dense) wemm::gemm_accumulate(x, w.dense(), std::nullopt, acc);
          else wemm::gemm_accumulate(x, w, std::nullopt, acc);
        } else {
          via_dense ? wemm::gemm(x, w.dense()) : wemm::gemm(x, w);
        }
      }, 50);
    };
    const double t_lt = timed(wd, true), t_sc = timed(wc, false), t_dc = timed(wc, true);
    std::printf("%-14s M=%3lld N=%5lld K=%5lld  %.1f | %.1f | %.1f us   (%.0f | %.0f GB/s)  rel_err=%.1e\n", t.name,
                (long long)t.M, (long long)t.N, (long long)t.K, t_lt * 1e3, t_sc * 1e3, t_dc * 1e3,
                2.0 * t.N * t.K / t_lt / 1e6, wc.nbytes() / t_sc / 1e6, ec);
  }
}
