#include "wemm/embedder.h"

#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <future>
#include <numeric>

#include "wemm/common.h"

namespace wemm {

Embedder::Embedder(const std::string& model_dir, EmbedderOptions opts) : opts_(opts) {
  // The ATen CPU resize kernels scale poorly past ~16 threads on small images.
  if (opts.intra_op_threads > 0) at::set_num_threads(opts.intra_op_threads);
  processor_ = std::make_unique<Processor>(model_dir);
  model_ = std::make_unique<WeMMModel>(model_dir, opts.model);
}

at::Tensor Embedder::embed(const std::vector<Conversation>& inputs, std::optional<int64_t> dimension) {
  c10::InferenceMode guard;
  const auto& dims = model_->config().matryoshka_dimensions;
  if (dimension && !dims.empty())
    WEMM_CHECK(std::find(dims.begin(), dims.end(), int(*dimension)) != dims.end(), "unsupported dimension ",
               *dimension);
  const int64_t hidden = model_->config().text.hidden_size;
  at::Tensor out = at::empty({int64_t(inputs.size()), hidden}, at::kFloat);
  if (inputs.empty()) return out;

  // Preprocess in groups; each group is split into token-bounded GPU batches.
  // Group i+1 is preprocessed while group i runs on the GPU.
  const size_t group = size_t(std::max<int64_t>(1, opts_.max_batch_size));
  auto prep = [this, &inputs](size_t b, size_t e) {
    c10::InferenceMode guard;  // thread-local
    std::vector<Conversation> part(inputs.begin() + b, inputs.begin() + e);
    return processor_->process_batch(part, opts_.preprocess_threads);
  };
  std::future<std::vector<ProcessedSample>> next = std::async(std::launch::async, prep, 0, std::min(group, inputs.size()));
  for (size_t b = 0; b < inputs.size(); b += group) {
    const size_t e = std::min(b + group, inputs.size());
    std::vector<ProcessedSample> samples = next.get();
    if (e < inputs.size()) next = std::async(std::launch::async, prep, e, std::min(e + group, inputs.size()));

    // Length-sorted packing keeps batches dense.
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t x, size_t y) { return samples[x].input_ids.size() < samples[y].input_ids.size(); });
    size_t i = 0;
    while (i < order.size()) {
      std::vector<ProcessedSample> batch;
      std::vector<size_t> rows;
      int64_t tokens = 0;
      while (i < order.size()) {
        const int64_t n = int64_t(samples[order[i]].input_ids.size());
        if (!batch.empty() && tokens + n > opts_.max_batch_tokens) break;
        tokens += n;
        rows.push_back(b + order[i]);
        batch.push_back(std::move(samples[order[i]]));
        ++i;
      }
      at::Tensor emb;
      try {
        emb = model_->embed(batch).cpu();
      } catch (...) {
        if (next.valid()) next.wait();  // do not leave the prefetch running against `inputs`
        throw;
      }
      std::vector<int64_t> rows64(rows.begin(), rows.end());
      out.index_copy_(0, at::tensor(rows64, at::kLong), emb);
    }
  }
  if (dimension) {
    out = out.narrow(1, 0, *dimension);
    out = out / out.norm(2, -1, true).clamp_min(1e-12);
  }
  return out;
}

Conversation text_conversation(const std::string& text) {
  return {Message{"user", {ContentItem::Text(text)}}};
}

Conversation image_conversation(const std::string& path, const std::string& text) {
  Message m{"user", {ContentItem::Image(path)}};
  if (!text.empty()) m.content.push_back(ContentItem::Text(text));
  return {m};
}

Conversation video_conversation(const std::string& path, const std::string& text) {
  Message m{"user", {}};
  if (!text.empty()) m.content.push_back(ContentItem::Text(text));
  m.content.push_back(ContentItem::Video(path));
  return {m};
}

at::Tensor Embedder::embed_texts(const std::vector<std::string>& texts, std::optional<int64_t> dimension) {
  std::vector<Conversation> c;
  for (const auto& t : texts) c.push_back(text_conversation(t));
  return embed(c, dimension);
}

at::Tensor Embedder::embed_images(const std::vector<std::string>& paths, std::optional<int64_t> dimension) {
  std::vector<Conversation> c;
  for (const auto& p : paths) c.push_back(image_conversation(p));
  return embed(c, dimension);
}

at::Tensor Embedder::embed_videos(const std::vector<std::string>& paths, std::optional<int64_t> dimension) {
  std::vector<Conversation> c;
  for (const auto& p : paths) c.push_back(video_conversation(p));
  return embed(c, dimension);
}

}  // namespace wemm
