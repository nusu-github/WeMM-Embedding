// Compares the native pipeline against reference dumps produced by
// tools/make_reference.py (prompt, token ids, pixel tensors, embeddings).
//
//   wemm_test_reference <model_dir> <ref_dir> [--unfused]

#include <c10/core/InferenceMode.h>

#include <cstdio>
#include <fstream>
#include <iostream>

#include "wemm/embedder.h"
#include "wemm/json_input.h"
#include "wemm/pixels.h"
#include "wemm/safetensors.h"

using namespace wemm;

static double cos_sim(const at::Tensor& a, const at::Tensor& b) {
  auto x = a.to(at::kDouble).flatten(), y = b.to(at::kDouble).flatten();
  return (x.dot(y) / (x.norm() * y.norm())).item<double>();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <model_dir> <ref_dir> [--unfused]\n";
    return 2;
  }
  const std::string model_dir = argv[1], ref_dir = argv[2];
  EmbedderOptions opts;
  std::string only;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--unfused") opts.model.fused_kernels = false;
    if (std::string(argv[i]) == "--bf16-residual") opts.model.fp32_residual = false;
    if (std::string(argv[i]) == "--only" && i + 1 < argc) only = "," + std::string(argv[++i]) + ",";
  }

  c10::InferenceMode guard;
  std::ifstream in(ref_dir + "/manifest.json");
  auto manifest = nlohmann::json::parse(in);
  Embedder emb(model_dir, opts);
  Processor& proc = emb.processor();

  int failures = 0;
  std::vector<std::string> names;
  std::vector<ProcessedSample> samples;
  std::vector<at::Tensor> ref_bf16, ref_fp32;
  for (auto& [name, entry] : manifest.items()) {
    if (!only.empty() && only.find("," + name + ",") == std::string::npos) continue;
    const auto& content = entry.at("content");
    nlohmann::json msg = !content.empty() && content[0].contains("role")
                             ? content
                             : nlohmann::json{{{"role", "user"}, {"content", content}}};
    Conversation conv = conversation_from_json(msg);
    ProcessedSample s = proc.process(conv);
    auto ref = SafeTensors::open(ref_dir + "/" + name + ".safetensors");

    bool prompt_ok = Processor::render_chat(conv) == entry.at("prompt").get<std::string>();
    at::Tensor ids_ref = ref.get("input_ids").flatten().to(at::kLong);
    at::Tensor ids = at::tensor(at::ArrayRef<int32_t>(s.input_ids)).to(at::kLong);
    bool ids_ok = ids.sizes() == ids_ref.sizes() && ids.equal(ids_ref);

    double pix_diff = 0;
    bool grid_ok = true;
    for (const char* key : {"pixel_values", "pixel_values_videos"}) {
      if (!ref.contains(key)) continue;
      bool video = std::string(key) == "pixel_values_videos";
      std::vector<at::Tensor> mine;
      std::vector<int64_t> grids;
      for (const auto& v : s.visuals)
        if (v.is_video == video) {
          mine.push_back(make_pixel_values(v, proc.config(), at::kCPU));
          grids.insert(grids.end(), {v.grid_t, v.grid_h, v.grid_w});
        }
      at::Tensor m = at::cat(mine, 0), r = ref.get(key).to(at::kFloat);
      at::Tensor gref = ref.get(video ? "video_grid_thw" : "image_grid_thw").flatten().to(at::kLong);
      grid_ok &= gref.equal(at::tensor(grids));
      if (m.sizes() != r.sizes()) {
        pix_diff = INFINITY;
      } else {
        pix_diff = std::max(pix_diff, (m - r).abs().max().item<double>());
      }
    }
    std::printf("%-12s prompt=%s ids=%s(%lld) grid=%s pixel_maxdiff=%.3g\n", name.c_str(), prompt_ok ? "ok" : "DIFF",
                ids_ok ? "ok" : "DIFF", (long long)ids.numel(), grid_ok ? "ok" : "DIFF", pix_diff);
    if (!prompt_ok || !ids_ok || !grid_ok || pix_diff > 1e-5) {
      ++failures;
      if (!ids_ok) {
        std::printf("   ref ids n=%lld mine n=%lld\n", (long long)ids_ref.numel(), (long long)ids.numel());
        std::printf("   expanded prompt (first 300 bytes): %s\n", s.prompt.substr(0, 300).c_str());
      }
    }
    names.push_back(name);
    samples.push_back(std::move(s));
    ref_bf16.push_back(ref.get("embedding_bfloat16").to(at::kFloat).flatten());
    ref_fp32.push_back(ref.contains("embedding_float32") ? ref.get("embedding_float32").to(at::kFloat).flatten()
                                                         : at::Tensor());
  }

  // One sample per forward, then everything packed into a single batch.
  std::printf("\n%-12s %14s %14s %14s %14s %14s\n", "case", "cos(nat,hf16)", "cos(nat,hf32)", "cos(hf16,hf32)",
              "cos(batch,single)", "cos(batch,hf32)");
  at::Tensor batched = emb.model().embed(samples).cpu();
  for (size_t i = 0; i < samples.size(); ++i) {
    at::Tensor single = emb.model().embed({samples[i]}).cpu().flatten();
    double a = cos_sim(single, ref_bf16[i]);
    double b = ref_fp32[i].defined() ? cos_sim(single, ref_fp32[i]) : NAN;
    double c = ref_fp32[i].defined() ? cos_sim(ref_bf16[i], ref_fp32[i]) : NAN;
    double d = cos_sim(single, batched[int64_t(i)]);
    double e = ref_fp32[i].defined() ? cos_sim(batched[int64_t(i)], ref_fp32[i]) : NAN;
    std::printf("%-12s %14.6f %14.6f %14.6f %14.6f %14.6f\n", names[i].c_str(), a, b, c, d, e);
    // Pass if the native embedding is as close to the fp32 model as the
    // Transformers bf16 run (or within 1e-3 of it); without an fp32 reference,
    // require agreement with the bf16 run.
    const bool ok = ref_fp32[i].defined() ? (b >= 0.999 || b >= c) : a >= 0.99;
    if (!ok) {
      std::printf("   ^ FAIL\n");
      ++failures;
    }
  }
  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
