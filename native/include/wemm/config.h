#pragma once

#include <array>
#include <string>
#include <vector>

namespace wemm {

struct VisionConfig {
  int depth = 24;
  int hidden_size = 1024;
  int num_heads = 16;
  int intermediate_size = 4096;
  int out_hidden_size = 2048;
  int patch_size = 16;
  int spatial_merge_size = 2;
  int temporal_patch_size = 2;
  int num_position_embeddings = 2304;
  int in_channels = 3;
  std::string hidden_act = "gelu_pytorch_tanh";
  std::vector<int> deepstack_visual_indexes;
};

struct TextConfig {
  int hidden_size = 2048;
  int num_hidden_layers = 24;
  int num_attention_heads = 8;
  int num_key_value_heads = 2;
  int head_dim = 256;
  int intermediate_size = 6144;
  double rms_norm_eps = 1e-6;
  bool attn_output_gate = true;
  std::vector<std::string> layer_types;
  int linear_conv_kernel_dim = 4;
  int linear_key_head_dim = 128;
  int linear_value_head_dim = 128;
  int linear_num_key_heads = 16;
  int linear_num_value_heads = 16;
  double rope_theta = 1e7;
  double partial_rotary_factor = 0.25;
  std::array<int, 3> mrope_section{11, 11, 10};
  bool mrope_interleaved = true;
  int vocab_size = 248078;
};

struct ModelConfig {
  VisionConfig vision;
  TextConfig text;
  int image_token_id = 248056;
  int video_token_id = 248057;
  int vision_start_token_id = 248053;
  int vision_end_token_id = 248054;
  std::vector<int> matryoshka_dimensions;

  static ModelConfig load(const std::string& model_dir);
};

// Values from processor_config.json (the Transformers-side image/video processors).
struct ProcessorConfig {
  // image processor
  int64_t image_min_pixels = 65536;
  int64_t image_max_pixels = 16777216;
  // video processor
  int64_t video_min_pixels = 4096;
  int64_t video_max_pixels = 234881024;
  double video_fps = 2.0;
  int video_min_frames = 4;
  int video_max_frames = 768;
  std::array<float, 3> image_mean{0.5f, 0.5f, 0.5f};
  std::array<float, 3> image_std{0.5f, 0.5f, 0.5f};
  double rescale_factor = 1.0 / 255.0;
  int patch_size = 16;
  int merge_size = 2;
  int temporal_patch_size = 2;

  static ProcessorConfig load(const std::string& model_dir);
};

}  // namespace wemm
