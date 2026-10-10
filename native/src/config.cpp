#include "wemm/config.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "wemm/common.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace wemm {

namespace {
json read_json(const fs::path& p) {
  std::ifstream in(p);
  WEMM_CHECK(in.good(), "cannot read ", p.string());
  return json::parse(in);
}
template <typename T>
void maybe(const json& j, const char* key, T& out) {
  if (j.contains(key) && !j.at(key).is_null()) out = j.at(key).get<T>();
}
}  // namespace

ModelConfig ModelConfig::load(const std::string& model_dir) {
  auto j = read_json(fs::path(model_dir) / "config.json");
  ModelConfig c;
  maybe(j, "image_token_id", c.image_token_id);
  maybe(j, "video_token_id", c.video_token_id);
  maybe(j, "vision_start_token_id", c.vision_start_token_id);
  maybe(j, "vision_end_token_id", c.vision_end_token_id);
  maybe(j, "matryoshka_dimensions", c.matryoshka_dimensions);

  const auto& v = j.at("vision_config");
  auto& vc = c.vision;
  maybe(v, "depth", vc.depth);
  maybe(v, "hidden_size", vc.hidden_size);
  maybe(v, "num_heads", vc.num_heads);
  maybe(v, "intermediate_size", vc.intermediate_size);
  maybe(v, "out_hidden_size", vc.out_hidden_size);
  maybe(v, "patch_size", vc.patch_size);
  maybe(v, "spatial_merge_size", vc.spatial_merge_size);
  maybe(v, "temporal_patch_size", vc.temporal_patch_size);
  maybe(v, "num_position_embeddings", vc.num_position_embeddings);
  maybe(v, "in_channels", vc.in_channels);
  maybe(v, "hidden_act", vc.hidden_act);
  maybe(v, "deepstack_visual_indexes", vc.deepstack_visual_indexes);
  WEMM_CHECK(vc.deepstack_visual_indexes.empty(), "deepstack visual features are not supported");

  const auto& t = j.at("text_config");
  auto& tc = c.text;
  maybe(t, "hidden_size", tc.hidden_size);
  maybe(t, "num_hidden_layers", tc.num_hidden_layers);
  maybe(t, "num_attention_heads", tc.num_attention_heads);
  maybe(t, "num_key_value_heads", tc.num_key_value_heads);
  maybe(t, "head_dim", tc.head_dim);
  maybe(t, "intermediate_size", tc.intermediate_size);
  maybe(t, "rms_norm_eps", tc.rms_norm_eps);
  maybe(t, "attn_output_gate", tc.attn_output_gate);
  maybe(t, "layer_types", tc.layer_types);
  maybe(t, "linear_conv_kernel_dim", tc.linear_conv_kernel_dim);
  maybe(t, "linear_key_head_dim", tc.linear_key_head_dim);
  maybe(t, "linear_value_head_dim", tc.linear_value_head_dim);
  maybe(t, "linear_num_key_heads", tc.linear_num_key_heads);
  maybe(t, "linear_num_value_heads", tc.linear_num_value_heads);
  maybe(t, "vocab_size", tc.vocab_size);
  if (t.contains("rope_parameters")) {
    const auto& r = t.at("rope_parameters");
    maybe(r, "rope_theta", tc.rope_theta);
    maybe(r, "partial_rotary_factor", tc.partial_rotary_factor);
    maybe(r, "mrope_interleaved", tc.mrope_interleaved);
    if (r.contains("mrope_section")) {
      auto s = r.at("mrope_section").get<std::vector<int>>();
      WEMM_CHECK(s.size() == 3, "mrope_section must have 3 entries");
      tc.mrope_section = {s[0], s[1], s[2]};
    }
    if (r.contains("rope_type")) {
      WEMM_CHECK(r.at("rope_type").get<std::string>() == "default", "only default rope is supported");
    }
  }
  WEMM_CHECK(static_cast<int>(tc.layer_types.size()) == tc.num_hidden_layers, "layer_types size mismatch");
  WEMM_CHECK(tc.mrope_interleaved, "only interleaved mrope is supported");
  return c;
}

ProcessorConfig ProcessorConfig::load(const std::string& model_dir) {
  ProcessorConfig c;
  auto path = fs::path(model_dir) / "processor_config.json";
  if (!fs::exists(path)) return c;
  auto j = read_json(path);
  if (j.contains("image_processor")) {
    const auto& ip = j.at("image_processor");
    if (ip.contains("size")) {
      maybe(ip.at("size"), "shortest_edge", c.image_min_pixels);
      maybe(ip.at("size"), "longest_edge", c.image_max_pixels);
    }
    maybe(ip, "min_pixels", c.image_min_pixels);
    maybe(ip, "max_pixels", c.image_max_pixels);
    maybe(ip, "image_mean", c.image_mean);
    maybe(ip, "image_std", c.image_std);
    maybe(ip, "rescale_factor", c.rescale_factor);
    maybe(ip, "patch_size", c.patch_size);
    maybe(ip, "merge_size", c.merge_size);
    maybe(ip, "temporal_patch_size", c.temporal_patch_size);
  }
  if (j.contains("video_processor")) {
    const auto& vp = j.at("video_processor");
    if (vp.contains("size")) {
      maybe(vp.at("size"), "shortest_edge", c.video_min_pixels);
      maybe(vp.at("size"), "longest_edge", c.video_max_pixels);
    }
    maybe(vp, "fps", c.video_fps);
    maybe(vp, "min_frames", c.video_min_frames);
    maybe(vp, "max_frames", c.video_max_frames);
  }
  return c;
}

}  // namespace wemm
