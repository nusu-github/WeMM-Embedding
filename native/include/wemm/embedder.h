#pragma once

#include <ATen/ATen.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "wemm/model.h"
#include "wemm/processor.h"

namespace wemm {

struct EmbedderOptions {
  ModelOptions model;
  int preprocess_threads = 0;        // 0 = auto
  int intra_op_threads = 16;         // ATen CPU threads (resizing); <= 0 keeps the default
  int64_t max_batch_tokens = 32768;  // packed tokens per forward pass
  int64_t max_batch_size = 64;
};

// End-to-end embedding: chat template + media + tokenizer on CPU threads,
// model on the GPU, with preprocessing of the next batch overlapped with the
// forward pass of the current one.
class Embedder {
 public:
  Embedder(const std::string& model_dir, EmbedderOptions opts = {});

  // Returns [N, dim] float32 CPU embeddings, L2-normalized. When `dimension`
  // is set, the Matryoshka prefix is taken and re-normalized.
  at::Tensor embed(const std::vector<Conversation>& inputs, std::optional<int64_t> dimension = std::nullopt);

  at::Tensor embed_texts(const std::vector<std::string>& texts, std::optional<int64_t> dimension = std::nullopt);
  at::Tensor embed_images(const std::vector<std::string>& paths, std::optional<int64_t> dimension = std::nullopt);
  at::Tensor embed_videos(const std::vector<std::string>& paths, std::optional<int64_t> dimension = std::nullopt);

  Processor& processor() { return *processor_; }
  WeMMModel& model() { return *model_; }

 private:
  EmbedderOptions opts_;
  std::unique_ptr<Processor> processor_;
  std::unique_ptr<WeMMModel> model_;
};

// Single-message conversation helpers.
Conversation text_conversation(const std::string& text);
Conversation image_conversation(const std::string& path, const std::string& text = "");
Conversation video_conversation(const std::string& path, const std::string& text = "");

}  // namespace wemm
