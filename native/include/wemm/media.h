#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wemm {

// "file:///a/b.jpg" -> "/a/b.jpg"; other strings unchanged.
std::string strip_file_scheme(const std::string& path);

// ---------------------------------------------------------------- images ----
// Decodes an image file (or encoded bytes) with OpenCV into an RGB uint8 tensor
// [H, W, 3]. Mirrors qwen-vl-utils `to_rgb`: RGBA inputs are composited onto a
// white background with Pillow's paste arithmetic, EXIF orientation is ignored
// (as `PIL.Image.open` does).
at::Tensor load_image(const std::string& path);
at::Tensor decode_image(const uint8_t* data, size_t size);

// Bit-exact port of Pillow's `Image.resize(..., BICUBIC)` for 8-bit RGB
// (libImaging/Resample.c, two-pass fixed-point convolution). [H,W,3] uint8.
at::Tensor pil_resize_bicubic(const at::Tensor& hwc, int64_t out_h, int64_t out_w);

// torchvision v2 `resize(uint8, BICUBIC, antialias=True)` on CPU: the native
// uint8 antialiased kernel. Accepts [..., H, W, 3] uint8, returns the same layout.
at::Tensor torch_resize_uint8(const at::Tensor& hwc, int64_t out_h, int64_t out_w);

// torchvision v1 `functional.resize` on a uint8 tensor: float32 antialiased
// bicubic, clamp to [0,255], round half-to-even. Returns uint8 (lossless, the
// values are integral). Accepts [T, H, W, 3] uint8.
at::Tensor torch_resize_uint8_via_float(const at::Tensor& thwc, int64_t out_h, int64_t out_w);

// ---------------------------------------------------------------- videos ----
struct VideoInfo {
  int64_t total_frames = 0;  // as reported by the container (decord `len(vr)`)
  double fps = 0.0;          // average frame rate (decord `get_avg_fps()`)
  int width = 0, height = 0;
};

struct DecodedVideo {
  at::Tensor frames;                  // [T, H, W, 3] uint8
  std::vector<int64_t> frame_indices; // index of each returned frame in the stream
  VideoInfo info;
};

// Probes container metadata without decoding frames.
VideoInfo probe_video(const std::string& path);

// Decodes the given (sorted, possibly repeated) frame indices with FFmpeg and
// converts them to RGB24 with swscale (SWS_BICUBIC, as decord's filter graph).
// `threads` <= 0 uses 8 decoder threads. `info` skips re-probing the file.
DecodedVideo decode_video_frames(const std::string& path, const std::vector<int64_t>& indices, int threads = 0,
                                 const VideoInfo* info = nullptr);

}  // namespace wemm
