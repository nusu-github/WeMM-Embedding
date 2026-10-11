// Checks that compressed weights decompress bit-exactly to the dense
// conversion (bf16 -> compute dtype) for every matrix of a checkpoint.
//
//   wemm_test_weight <model_dir> [--bf16]
#include <c10/core/InferenceMode.h>

#include <cstdio>
#include <string>

#include "wemm/safetensors.h"
#include "wemm/weight.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: wemm_test_weight <model_dir> [--bf16]\n");
    return 2;
  }
  c10::InferenceMode guard;
  const at::ScalarType dt = argc > 2 && std::string(argv[2]) == "--bf16" ? at::kBFloat16 : at::kHalf;
  const at::Device dev(at::kCUDA, 0);
  auto st = wemm::SafeTensors::open(argv[1]);
  int64_t n = 0, dense_bytes = 0, comp_bytes = 0, failures = 0;
  for (const auto& name : st.names()) {
    at::Tensor h = st.get(name);
    if (h.dim() != 2 || h.scalar_type() != at::kBFloat16 || h.numel() % 64 != 0 || name.find("embed_tokens") != std::string::npos)
      continue;
    wemm::Weight w = wemm::Weight::load(h, dt, dev, /*compress=*/true);
    at::Tensor ref = h.to(dev).to(dt);
    const bool eq = w.compressed() && at::equal(w.dense().view(at::kShort), ref.view(at::kShort));
    if (!eq) {
      std::printf("MISMATCH %s\n", name.c_str());
      ++failures;
    }
    ++n;
    dense_bytes += int64_t(ref.nbytes());
    comp_bytes += w.nbytes();
  }
  std::printf("%lld matrices, %.1f MiB -> %.1f MiB (%.1f%%), %lld mismatches\n", (long long)n, dense_bytes / 1048576.0,
              comp_bytes / 1048576.0, 100.0 * comp_bytes / dense_bytes, (long long)failures);
  return failures ? 1 : 0;
}
