#include "wemm/c_api.h"

#include <c10/core/InferenceMode.h>

#include <cstring>
#include <string>

#include "wemm/embedder.h"
#include "wemm/json_input.h"

struct wemm_embedder {
  std::unique_ptr<wemm::Embedder> impl;
};

namespace {
thread_local std::string g_error;

template <typename F>
int guarded(F&& f) {
  try {
    g_error.clear();
    f();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
  } catch (...) {
    g_error = "unknown error";
  }
  return -1;
}
}  // namespace

extern "C" {

wemm_embedder* wemm_create(const char* model_dir, int device, const char* options_json) {
  wemm_embedder* out = nullptr;
  guarded([&] {
    wemm::EmbedderOptions opts;
    opts.model.device_index = device < 0 ? 0 : device;  // validated by the model
    if (options_json && *options_json) {
      auto j = nlohmann::json::parse(options_json);
      opts.max_batch_tokens = j.value("max_batch_tokens", opts.max_batch_tokens);
      opts.max_batch_size = j.value("max_batch_size", opts.max_batch_size);
      opts.preprocess_threads = j.value("preprocess_threads", opts.preprocess_threads);
      opts.intra_op_threads = j.value("intra_op_threads", opts.intra_op_threads);
      opts.model.fp32_residual = j.value("fp32_residual", opts.model.fp32_residual);
      opts.model.embeddings_on_gpu = j.value("embeddings_on_gpu", opts.model.embeddings_on_gpu);
      opts.model.gdn_window_bytes = j.value("gdn_window_bytes", opts.model.gdn_window_bytes);
      opts.model.mlp_chunk_tokens = j.value("mlp_chunk_tokens", opts.model.mlp_chunk_tokens);
      if (j.contains("compute_dtype")) {
        const std::string dt = j.at("compute_dtype").get<std::string>();
        WEMM_CHECK(dt == "fp16" || dt == "bf16", "compute_dtype must be \"fp16\" or \"bf16\"");
        opts.model.compute_dtype = dt == "fp16" ? at::kHalf : at::kBFloat16;
      }
    }
    auto e = std::make_unique<wemm_embedder>();
    e->impl = std::make_unique<wemm::Embedder>(model_dir, opts);
    out = e.release();
  });
  return out;
}

void wemm_destroy(wemm_embedder* e) { delete e; }

int wemm_dim(const wemm_embedder* e, int dimension) {
  if (!e) return 0;
  const int hidden = e->impl->model().config().text.hidden_size;
  if (dimension <= 0) return hidden;
  return dimension <= hidden ? dimension : 0;  // 0: invalid dimension
}

int wemm_embed_json(wemm_embedder* e, const char* const* inputs_json, size_t n, int dimension, float* out) {
  return guarded([&] {
    WEMM_CHECK(e && out, "null argument");
    c10::InferenceMode guard;
    std::vector<wemm::Conversation> convs;
    convs.reserve(n);
    for (size_t i = 0; i < n; ++i) convs.push_back(wemm::conversation_from_json(nlohmann::json::parse(inputs_json[i])));
    std::optional<int64_t> dim;
    if (dimension > 0) dim = dimension;
    at::Tensor emb = e->impl->embed(convs, dim).contiguous();
    std::memcpy(out, emb.data_ptr<float>(), size_t(emb.numel()) * sizeof(float));
  });
}

const char* wemm_last_error(void) { return g_error.c_str(); }

}  // extern "C"
