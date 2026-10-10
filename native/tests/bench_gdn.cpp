// Micro-benchmark + correctness check of the gated-delta-rule kernels against
// an fp32 ATen reference (chunked algorithm, as in the unfused model path).
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "wemm/kernels.h"

template <typename F>
double time_ms(F f, int iters = 20) {
  for (int i = 0; i < 3; ++i) f();
  at::cuda::getCurrentCUDAStream().synchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) f();
  at::cuda::getCurrentCUDAStream().synchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main(int argc, char** argv) {
  c10::InferenceMode guard;
  std::vector<int64_t> lens = {2743};
  if (argc > 1) {
    lens.clear();
    for (int i = 1; i < argc; ++i) lens.push_back(std::atoll(argv[i]));
  }
  const int64_t H = 16, dk = 128, dv = 128, key_dim = H * dk, conv_dim = 2 * key_dim + H * dv;
  int64_t L = 0;
  for (auto l : lens) L += l;
  auto dev = at::TensorOptions().device(at::kCUDA);
  at::manual_seed(0);
  at::Tensor mixed = (at::randn({L, conv_dim}, dev) * 0.5).to(at::kBFloat16);
  at::Tensor ab = at::randn({L, 2 * H}, dev).to(at::kBFloat16);
  at::Tensor b = ab.narrow(1, 0, H), a = ab.narrow(1, H, H);
  at::Tensor neg_A = -at::rand({H}, dev) * 8 - 0.1;
  at::Tensor dt_bias = at::randn({H}, dev);

  std::vector<int32_t> cs, cv, co{0}, pr;
  int64_t tok = 0;
  for (auto len : lens) {
    for (int64_t j = 0; j < len; ++j) pr.push_back(int32_t(cs.size() * 64 + j));
    for (int64_t c = 0; c * 64 < len; ++c) {
      cs.push_back(int32_t(tok + c * 64));
      cv.push_back(int32_t(std::min<int64_t>(64, len - c * 64)));
    }
    co.push_back(int32_t(cs.size()));
    tok += len;
  }
  auto i32 = [&](std::vector<int32_t>& v) { return at::tensor(v, at::kInt).to(at::kCUDA); };
  at::Tensor chunk_start = i32(cs), chunk_valid = i32(cv), chunk_off = i32(co);
  const int64_t C = int64_t(cs.size());

  auto ch = wemm::kernels::gdn_chunk_prepare(mixed, b, a, neg_A, dt_bias, chunk_start, chunk_valid, H, H, dk, dv);
  at::Tensor o = wemm::kernels::gdn_chunk_recurrence(ch, chunk_off, int64_t(lens.size()));
  at::Tensor o_single = wemm::kernels::gdn_chunk_recurrence(ch, chunk_off, int64_t(lens.size()), 0, at::Tensor(), false);
  std::printf("single-buffer vs double-buffer max diff: %.3e\n", (o - o_single).abs().max().item<double>());
  // windowed (3 chunks per window, state carried across windows)
  {
    const int64_t C = chunk_start.numel(), W = 3;
    at::Tensor carry = at::empty({H, int64_t(lens.size()), dk, dv}, mixed.options().dtype(at::kFloat));
    double md = 0;
    at::Tensor oc = o.view({H, C, 64, dv});
    for (int64_t c0 = 0; c0 < C; c0 += W) {
      const int64_t c1 = std::min(C, c0 + W);
      auto cw = wemm::kernels::gdn_chunk_prepare(mixed, b, a, neg_A, dt_bias, chunk_start, chunk_valid, H, H, dk, dv,
                                                 c0, c1);
      at::Tensor ow = wemm::kernels::gdn_chunk_recurrence(cw, chunk_off, int64_t(lens.size()), c0, carry);
      md = std::max(md, (ow - oc.narrow(1, c0, c1 - c0)).abs().max().item<double>());
    }
    std::printf("windowed vs full max diff: %.3e\n", md);
  }

  // ---- reference: token-by-token recurrence in fp64 (exact semantics).
  {
    at::Tensor q = mixed.narrow(1, 0, key_dim).view({L, H, dk}).to(at::kDouble);
    at::Tensor k = mixed.narrow(1, key_dim, key_dim).view({L, H, dk}).to(at::kDouble);
    at::Tensor v = mixed.narrow(1, 2 * key_dim, H * dv).view({L, H, dv}).to(at::kDouble);
    q = q * at::rsqrt((q * q).sum(-1, true) + 1e-6) / std::sqrt(double(dk));
    k = k * at::rsqrt((k * k).sum(-1, true) + 1e-6);
    at::Tensor g = neg_A.to(at::kDouble) * at::softplus(a.to(at::kDouble) + dt_bias.to(at::kDouble));
    at::Tensor beta = at::sigmoid(b.to(at::kDouble));
    at::Tensor ref = at::empty({L, H, dv}, q.options());
    int64_t off = 0;
    for (auto len : lens) {
      at::Tensor S = at::zeros({H, dk, dv}, q.options());
      for (int64_t t = off; t < off + len; ++t) {
        S = S * g[t].exp().view({H, 1, 1});
        at::Tensor kv = (S * k[t].unsqueeze(-1)).sum(1);
        at::Tensor delta = (v[t] - kv) * beta[t].unsqueeze(-1);
        S = S + k[t].unsqueeze(-1) * delta.unsqueeze(1);
        ref[t] = (S * q[t].unsqueeze(-1)).sum(1);
      }
      off += len;
    }
    at::Tensor mine = o.view({H, C * 64, dv}).index_select(1, i32(pr).to(at::kLong)).permute({1, 0, 2}).to(at::kDouble);
    double err = (mine - ref).abs().max().item<double>();
    double rel = ((mine - ref).norm() / ref.norm()).item<double>();
    std::printf("L=%lld chunks=%lld  max_abs_err=%.3e  rel_err=%.3e\n", (long long)L, (long long)C, err, rel);
  }

  double tp = time_ms([&] {
    wemm::kernels::gdn_chunk_prepare(mixed, b, a, neg_A, dt_bias, chunk_start, chunk_valid, H, H, dk, dv);
  });
  double tr = time_ms([&] { wemm::kernels::gdn_chunk_recurrence(ch, chunk_off, int64_t(lens.size())); });
  double ts = time_ms(
      [&] { wemm::kernels::gdn_chunk_recurrence(ch, chunk_off, int64_t(lens.size()), 0, at::Tensor(), false); });
  std::printf("prepare %.3f ms   recurrence %.3f ms (single-buffer %.3f ms)\n", tp, tr, ts);
}
