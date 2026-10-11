#pragma once

#include <ATen/ATen.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "wemm/config.h"
#include "wemm/processor.h"
#include "wemm/weight.h"

namespace wemm {

enum class AttentionBackend {
  Auto,          // FlashAttention on sm_80+, memory-efficient attention otherwise
  Flash,         // FlashAttention-2 (sm_80+)
  MemEfficient,  // CUTLASS memory-efficient attention (sm_75+, fp16 on sm_75)
};

struct ModelOptions {
  int device_index = 0;
  // Hand-written fused CUDA kernels. false selects a plain-ATen reference
  // implementation, kept to validate the kernels (slower, more memory).
  bool fused_kernels = true;
  // Keep the residual streams (ViT and LM) in fp32: GEMM epilogues accumulate
  // into fp32 and norms read fp32. Closer to the fp32 model than 16-bit
  // residuals at a few percent cost. Only used with fused_kernels.
  bool fp32_residual = true;
  // Activation/weight dtype: fp16 (default) or bf16 (sm_80+). The bf16
  // checkpoint converts to fp16 exactly, and fp16's 10-bit mantissa keeps the
  // embeddings ~10x closer to the fp32 model than bf16 at the same speed.
  // Intermediate activations stay far below the fp16 range (< 2% of it on the
  // reference suite; the residual streams are fp32); a non-finite result
  // raises an error suggesting bf16.
  std::optional<at::ScalarType> compute_dtype;
  AttentionBackend attention = AttentionBackend::Auto;
  // Token embedding table in VRAM. Off by default: the table (1-2 GB) stays
  // in memory-mapped host memory and only the looked-up rows are copied.
  bool embeddings_on_gpu = false;
  // Caps the fp32 workspace of one gated-delta-rule layer; long inputs are
  // processed in windows of chunks with the state carried across windows.
  int64_t gdn_window_bytes = int64_t(512) << 20;
  // Token rows per MLP pass (bounds the [rows, 2*intermediate] activation).
  int64_t mlp_chunk_tokens = 8192;
  // Use the caching allocator's expandable segments (less fragmentation)
  // unless PYTORCH_CUDA_ALLOC_CONF is set. Process-wide; applied before the
  // first CUDA allocation only.
  bool expandable_segments = true;
  // Store the GEMM weight matrices with lossless compression (~27% less
  // weight memory, bit-identical results). Each matrix is decompressed into
  // a temporary right before its GEMM: a few percent slower on long inputs,
  // noticeably slower on very short ones.
  bool compress_weights = false;
};

// Native Qwen3.5 (Gated DeltaNet / gated attention) + ViT embedding model.
// Weights are loaded straight from the Hugging Face safetensors checkpoint.
// Not thread-safe: use one instance per thread (or serialize calls).
class WeMMModel {
 public:
  WeMMModel(const std::string& model_dir, ModelOptions opts = {});

  // Packs the samples into one variable-length batch (no padding) and returns
  // L2-normalized embeddings [N, hidden] (float32, on the model device).
  at::Tensor embed(const std::vector<ProcessedSample>& batch);

  const ModelConfig& config() const { return cfg_; }
  at::Device device() const { return device_; }
  at::ScalarType compute_dtype() const { return act_; }
  AttentionBackend attention_backend() const { return attn_; }

  // Exposed for testing/benchmarking.
  at::Tensor vision_forward(const at::Tensor& pixel_values, const std::vector<std::array<int64_t, 3>>& grids);
  at::Tensor text_forward(at::Tensor hidden, const at::Tensor& position_ids, const std::vector<int64_t>& seq_lens);

 private:
  struct VisionBlock {
    at::Tensor norm1_w, norm1_b, norm2_w, norm2_b;  // fp32
    Weight qkv_w, proj_w, fc1_w, fc2_w;
    at::Tensor qkv_b, fc1_b;                        // act
    at::Tensor proj_b, fc2_b;                       // residual dtype (fp32 or act)
  };
  struct TextLayer {
    bool linear = true;
    at::Tensor in_norm, post_norm;  // fp32 (1 + w)
    // linear attention
    Weight in_proj;                 // [qkv | z | b | a | pad] x hidden
    at::Tensor conv_w;              // [conv_dim, K] fp32
    at::Tensor neg_A;               // -exp(A_log) fp32 [Hv]
    at::Tensor dt_bias;             // fp32 [Hv]
    at::Tensor gnorm_w;             // fp32 [dv]
    Weight out_proj;
    // full attention
    Weight qkv_proj;                // [q(+gate) | k | v] x hidden
    at::Tensor q_norm, k_norm;      // fp32 (1 + w)
    Weight o_proj;
    // mlp
    Weight gate_up, down;
  };

  // Per-batch inputs shared by all text layers (both implementations).
  struct Packing {
    std::vector<int64_t> seq_lens;
    int64_t max_len = 0;
    at::Tensor cu_seqlens;   // [N+1] int32
    at::Tensor seq_start;    // [L] int32 first token of each token's sequence
    at::Tensor last_rows;    // [N] int64 row of each <embedding> token
    at::Tensor cos, sin;     // [L, rotary_dim] fp32 M-RoPE tables
    // 64-token chunk layout of the gated delta rule
    std::vector<int32_t> chunk_start_h, chunk_valid_h, chunk_off_h;  // host copies
    at::Tensor chunk_start;  // [C] int32 first token of each chunk
    at::Tensor chunk_valid;  // [C] int32 valid rows in the chunk
    at::Tensor chunk_off;    // [N+1] int32 chunk range of each sequence
    at::Tensor padded_row;   // [L] int32 token -> chunk row
  };
  // Per-batch inputs of the ViT.
  struct VisionPrelude {
    at::Tensor pos_embeds;   // [P, D] fp32 interpolated position embeddings
    at::Tensor cos, sin;     // [P, head_dim] fp32 2D rotary tables
    at::Tensor cu_seqlens;   // [frames+1] int32 (attention is per frame)
    int64_t max_len = 0;
  };

  void load_weights(const std::string& model_dir);
  Packing make_packing(const std::vector<int64_t>& seq_lens, const at::Tensor& position_ids) const;
  VisionPrelude make_vision_prelude(const std::vector<std::array<int64_t, 3>>& grids) const;
  at::Tensor lookup_embeddings(const std::vector<int64_t>& ids) const;
  at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu_q,
                       const at::Tensor& cu_k, int64_t max_q, int64_t max_k, bool causal) const;

  // model_fused.cpp
  at::Tensor vision_fused(const at::Tensor& pixel_values, const VisionPrelude& vp);
  at::Tensor text_fused(at::Tensor h, const Packing& pk);
  at::Tensor linear_attention_fused(const TextLayer& l, const at::Tensor& x, at::Tensor h, const Packing& pk,
                                    bool last);
  at::Tensor full_attention_fused(const TextLayer& l, const at::Tensor& x, at::Tensor h, const Packing& pk,
                                  bool last);
  void mlp_fused(const TextLayer& l, at::Tensor& h);

  // model_reference.cpp
  at::Tensor vision_reference(const at::Tensor& pixel_values, const VisionPrelude& vp);
  at::Tensor text_reference(at::Tensor h, const Packing& pk);
  at::Tensor linear_attention_reference(const TextLayer& l, const at::Tensor& x, const Packing& pk);
  at::Tensor full_attention_reference(const TextLayer& l, const at::Tensor& x, const Packing& pk, bool last);
  at::Tensor gated_delta_rule_reference(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& g, const at::Tensor& beta, const Packing& pk);

  ModelConfig cfg_;
  ProcessorConfig pcfg_;
  ModelOptions opts_;
  at::Device device_;
  at::ScalarType act_ = at::kBFloat16;
  at::ScalarType resid_ = at::kFloat;  // residual dtype of the fused path
  AttentionBackend attn_ = AttentionBackend::Flash;

  // vision
  Weight patch_w;
  at::Tensor patch_b /* fp32 */, pos_embed;
  std::vector<VisionBlock> vblocks_;
  at::Tensor merger_norm_w, merger_norm_b /* fp32 */, merger_fc1_b, merger_fc2_b;
  Weight merger_fc1_w, merger_fc2_w;
  at::Tensor vision_inv_freq;  // fp32 [head_dim/4]
  // text
  at::Tensor embed_tokens;  // [V, D]: device (act) or memory-mapped host (bf16)
  std::vector<TextLayer> layers_;
  at::Tensor final_norm;
  at::Tensor text_inv_freq;  // fp32 [rotary_dim/2]
};

}  // namespace wemm
