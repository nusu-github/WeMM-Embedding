// Throughput/latency benchmark of the native model on fixed inputs.
//
//   wemm_bench --model DIR [--iters N] [--batch B] (--text T | --image P | --video P)...

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <chrono>
#include <cstdio>
#include <iostream>

#include "wemm/embedder.h"

using namespace wemm;

namespace {
double gib(int64_t b) { return double(b) / (1 << 30); }
// [allocated now, allocated peak, reserved now] on device 0
void mem_line(const char* tag) {
  auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto agg = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  std::printf("%s: allocated %.2f GiB, peak %.2f GiB, reserved %.2f GiB\n", tag,
              gib(st.allocated_bytes[agg].current), gib(st.allocated_bytes[agg].peak),
              gib(st.reserved_bytes[agg].current));
}
}  // namespace

int main(int argc, char** argv) {
  std::string model;
  int iters = 20, batch = 1;
  std::vector<Conversation> inputs;
  EmbedderOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") model = argv[++i];
    else if (a == "--iters") iters = std::stoi(argv[++i]);
    else if (a == "--batch") batch = std::stoi(argv[++i]);
    else if (a == "--unfused") opts.model.fused_kernels = false;
    else if (a == "--bf16-residual") opts.model.fp32_residual = false;
    else if (a == "--fp16") opts.model.compute_dtype = at::kHalf;
    else if (a == "--bf16") opts.model.compute_dtype = at::kBFloat16;
    else if (a == "--mem-efficient") opts.model.attention = AttentionBackend::MemEfficient;
    else if (a == "--emb-gpu") opts.model.embeddings_on_gpu = true;
    else if (a == "--compress") opts.model.compress_weights = true;
    else if (a == "--gdn-window") opts.model.gdn_window_bytes = std::stoll(argv[++i]);
    else if (a == "--mlp-chunk") opts.model.mlp_chunk_tokens = std::stoll(argv[++i]);
    else if (a == "--batch-tokens") opts.max_batch_tokens = std::stoll(argv[++i]);
    else if (a == "--text") inputs.push_back(text_conversation(argv[++i]));
    else if (a == "--image") inputs.push_back(image_conversation(argv[++i]));
    else if (a == "--video") inputs.push_back(video_conversation(argv[++i]));
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  c10::InferenceMode guard;
  Embedder emb(model, opts);
  mem_line("after load");
  for (const auto& conv : inputs) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    std::vector<Conversation> convs(size_t(batch), conv);
    auto t0 = std::chrono::steady_clock::now();
    auto samples = emb.processor().process_batch(convs);
    auto t1 = std::chrono::steady_clock::now();
    for (int w = 0; w < 3; ++w) emb.model().embed(samples);
    at::cuda::getCurrentCUDAStream().synchronize();
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) emb.model().embed(samples);
    at::cuda::getCurrentCUDAStream().synchronize();
    auto t3 = std::chrono::steady_clock::now();
    double fwd = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;
    std::printf("tokens/sample=%zu batch=%d  preprocess=%.1f ms  forward=%.2f ms  (%.1f samples/s)\n",
                samples[0].input_ids.size(), batch, std::chrono::duration<double, std::milli>(t1 - t0).count(), fwd,
                1000.0 * batch / fwd);
    mem_line("  memory");
  }
  return 0;
}
