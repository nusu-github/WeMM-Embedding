// Throughput/latency benchmark of the native model on fixed inputs.
//
//   wemm_bench --model DIR [--iters N] [--batch B] (--text T | --image P | --video P)...

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>

#include <chrono>
#include <cstdio>
#include <iostream>

#include "wemm/embedder.h"

using namespace wemm;

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
  for (const auto& conv : inputs) {
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
  }
  return 0;
}
