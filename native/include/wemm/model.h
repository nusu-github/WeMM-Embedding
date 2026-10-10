#pragma once

#include <ATen/ATen.h>

#include <array>
#include <string>
#include <vector>

#include "wemm/config.h"
#include "wemm/processor.h"

namespace wemm {

struct ModelOptions {
  int device_index = 0;
  // Use the hand-written fused CUDA kernels (otherwise plain ATen ops; useful
  // for validating the kernels).
  bool fused_kernels = true;
};

// Native Qwen3.5 (Gated DeltaNet / gated attention) + ViT embedding model.
// Weights are loaded straight from the Hugging Face safetensors checkpoint.
class WeMMModel {
 public:
  WeMMModel(const std::string& model_dir, ModelOptions opts = {});

  // Packs the samples into one variable-length batch (no padding) and returns
  // L2-normalized embeddings [N, hidden] (float32, on the model device).
  at::Tensor embed(const std::vector<ProcessedSample>& batch);

  const ModelConfig& config() const { return cfg_; }
  at::Device device() const { return device_; }

  // Exposed for testing/benchmarking.
  at::Tensor vision_forward(const at::Tensor& pixel_values, const std::vector<std::array<int64_t, 3>>& grids);
  at::Tensor text_forward(at::Tensor hidden, const at::Tensor& position_ids, const std::vector<int64_t>& seq_lens);

 private:
  struct VisionBlock {
    at::Tensor norm1_w, norm1_b, norm2_w, norm2_b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b;
  };
  struct TextLayer {
    bool linear = true;
    at::Tensor in_norm, post_norm;  // fp32 (1 + w)
    // linear attention
    at::Tensor in_proj;             // [qkv | z | b | a] x hidden, bf16
    at::Tensor conv_w;              // [conv_dim, K] fp32
    at::Tensor neg_A;               // -exp(A_log) fp32 [Hv]
    at::Tensor dt_bias;             // fp32 [Hv]
    at::Tensor gnorm_w;             // fp32 [dv]
    at::Tensor out_proj;
    // full attention
    at::Tensor qkv_proj;            // [q(+gate) | k | v] x hidden, bf16
    at::Tensor q_norm, k_norm;      // fp32 (1 + w)
    at::Tensor o_proj;
    // mlp
    at::Tensor gate_up, down;
  };

  at::Tensor linear_attention(const TextLayer& l, const at::Tensor& x, const at::Tensor& seq_start,
                              const std::vector<int64_t>& seq_lens);
  at::Tensor full_attention(const TextLayer& l, const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin,
                            const at::Tensor& cu_seqlens, int64_t max_len, const at::Tensor* q_rows);
  at::Tensor gated_delta_rule(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                              const at::Tensor& beta, const std::vector<int64_t>& seq_lens);
  at::Tensor mlp(const TextLayer& l, const at::Tensor& x);

  ModelConfig cfg_;
  ProcessorConfig pcfg_;
  ModelOptions opts_;
  at::Device device_;

  // vision
  at::Tensor patch_w, patch_b, pos_embed;
  std::vector<VisionBlock> vblocks_;
  at::Tensor merger_norm_w, merger_norm_b, merger_fc1_w, merger_fc1_b, merger_fc2_w, merger_fc2_b;
  at::Tensor vision_inv_freq;  // fp32 [head_dim/4]
  // text
  at::Tensor embed_tokens;
  std::vector<TextLayer> layers_;
  at::Tensor final_norm;
  at::Tensor text_inv_freq;  // fp32 [rotary_dim/2]
};

}  // namespace wemm
