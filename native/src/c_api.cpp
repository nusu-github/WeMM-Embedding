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
    opts.model.device_index = device < 0 ? 0 : device;
    if (options_json && *options_json) {
      auto j = nlohmann::json::parse(options_json);
      opts.max_batch_tokens = j.value("max_batch_tokens", opts.max_batch_tokens);
      opts.max_batch_size = j.value("max_batch_size", opts.max_batch_size);
      opts.preprocess_threads = j.value("preprocess_threads", opts.preprocess_threads);
      opts.intra_op_threads = j.value("intra_op_threads", opts.intra_op_threads);
      opts.model.fp32_residual = j.value("fp32_residual", opts.model.fp32_residual);
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
  return dimension > 0 ? dimension : e->impl->model().config().text.hidden_size;
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
