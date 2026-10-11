#pragma once

// Parsing of Hugging Face style message JSON into wemm::Conversation.
// Header-only; requires nlohmann/json.

#include <nlohmann/json.hpp>

#include "wemm/common.h"
#include "wemm/processor.h"

namespace wemm {

inline std::vector<ContentItem> content_from_json(const nlohmann::json& content) {
  std::vector<ContentItem> items;
  if (content.is_string()) {
    items.push_back(ContentItem::Text(content.get<std::string>()));
    return items;
  }
  WEMM_CHECK(content.is_array(), "message content must be a string or a list");
  for (const auto& it : content) {
    std::string type = it.value("type", "");
    auto opt_i = [&](const char* k, std::optional<int64_t>& dst) {
      if (it.contains(k)) dst = it.at(k).get<int64_t>();
    };
    auto opt_d = [&](const char* k, std::optional<double>& dst) {
      if (it.contains(k)) dst = it.at(k).get<double>();
    };
    if (it.contains("image") || it.contains("image_url") || type == "image") {
      ImageOptions o;
      opt_i("min_pixels", o.min_pixels);
      opt_i("max_pixels", o.max_pixels);
      opt_i("resized_height", o.resized_height);
      opt_i("resized_width", o.resized_width);
      std::string src = it.contains("image") ? it.at("image").get<std::string>() : it.at("image_url").get<std::string>();
      items.push_back(ContentItem::Image(src, o));
    } else if (it.contains("video") || type == "video") {
      VideoOptions o;
      opt_d("fps", o.fps);
      opt_i("nframes", o.nframes);
      opt_i("min_frames", o.min_frames);
      opt_i("max_frames", o.max_frames);
      opt_d("video_start", o.video_start);
      opt_d("video_end", o.video_end);
      opt_i("min_pixels", o.min_pixels);
      opt_i("max_pixels", o.max_pixels);
      opt_d("total_pixels", o.total_pixels);
      opt_i("resized_height", o.resized_height);
      opt_i("resized_width", o.resized_width);
      items.push_back(ContentItem::Video(it.at("video").get<std::string>(), o));
    } else if (it.contains("text")) {
      items.push_back(ContentItem::Text(it.at("text").get<std::string>()));
    } else {
      throw Error("Unexpected item type in content.");
    }
  }
  return items;
}

// Accepts either a list of messages ([{"role":..., "content":...}, ...]), a
// single message object, or a shorthand object {"text"|"image"|"video": ...}
// (several keys combine into one user message in image, video, text order).
inline Conversation conversation_from_json(const nlohmann::json& j) {
  Conversation conv;
  if (j.is_array()) {
    for (const auto& m : j) conv.push_back(Message{m.value("role", "user"), content_from_json(m.at("content"))});
  } else if (j.is_object() && j.contains("content")) {
    conv.push_back(Message{j.value("role", "user"), content_from_json(j.at("content"))});
  } else if (j.is_object()) {
    nlohmann::json content = nlohmann::json::array();
    if (j.contains("image")) content.push_back({{"type", "image"}, {"image", j.at("image")}});
    if (j.contains("video")) content.push_back({{"type", "video"}, {"video", j.at("video")}});
    if (j.contains("text")) content.push_back({{"type", "text"}, {"text", j.at("text")}});
    conv.push_back(Message{"user", content_from_json(content)});
  } else if (j.is_string()) {
    conv.push_back(Message{"user", {ContentItem::Text(j.get<std::string>())}});
  } else {
    throw Error("cannot interpret input JSON");
  }
  return conv;
}

}  // namespace wemm
