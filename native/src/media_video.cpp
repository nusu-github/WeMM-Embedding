#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "wemm/common.h"
#include "wemm/media.h"

namespace wemm {

namespace {

struct FormatCtx {
  AVFormatContext* ctx = nullptr;
  ~FormatCtx() {
    if (ctx) avformat_close_input(&ctx);
  }
};
struct CodecCtx {
  AVCodecContext* ctx = nullptr;
  ~CodecCtx() {
    if (ctx) avcodec_free_context(&ctx);
  }
};
struct Packet {
  AVPacket* p = av_packet_alloc();
  ~Packet() { av_packet_free(&p); }
};
struct Frame {
  AVFrame* f = av_frame_alloc();
  ~Frame() { av_frame_free(&f); }
};
struct Sws {
  SwsContext* ctx = nullptr;
  ~Sws() {
    if (ctx) sws_freeContext(ctx);
  }
};

std::string av_err(int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(code, buf, sizeof(buf));
  return buf;
}

int open_stream(FormatCtx& fmt, const std::string& path) {
  int rc = avformat_open_input(&fmt.ctx, strip_file_scheme(path).c_str(), nullptr, nullptr);
  WEMM_CHECK(rc >= 0, "cannot open video ", path, ": ", av_err(rc));
  rc = avformat_find_stream_info(fmt.ctx, nullptr);
  WEMM_CHECK(rc >= 0, "cannot read stream info of ", path, ": ", av_err(rc));
  int idx = av_find_best_stream(fmt.ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  WEMM_CHECK(idx >= 0, "no video stream in ", path);
  return idx;
}

double stream_fps(const AVStream* st) {
  if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0) return av_q2d(st->avg_frame_rate);
  if (st->r_frame_rate.num > 0 && st->r_frame_rate.den > 0) return av_q2d(st->r_frame_rate);
  return 0.0;
}

// swscale flags for YUV->RGB24: libavfilter's `scale` default (bicubic), which
// is what decord's filter graph uses. Verified bit-exact against decord.
constexpr int kSwsFlags = SWS_BICUBIC;

}  // namespace

VideoInfo probe_video(const std::string& path) {
  FormatCtx fmt;
  int idx = open_stream(fmt, path);
  const AVStream* st = fmt.ctx->streams[idx];
  VideoInfo info;
  info.fps = stream_fps(st);
  info.width = st->codecpar->width;
  info.height = st->codecpar->height;
  info.total_frames = st->nb_frames;
  if (info.total_frames <= 0) {
    // Container without a frame count: estimate from the duration like decord does.
    double dur = st->duration != AV_NOPTS_VALUE ? st->duration * av_q2d(st->time_base)
                                                : fmt.ctx->duration / double(AV_TIME_BASE);
    info.total_frames = static_cast<int64_t>(dur * info.fps + 0.5);
  }
  WEMM_CHECK(info.total_frames > 0 && info.fps > 0, "cannot determine frame count/fps of ", path);
  return info;
}

DecodedVideo decode_video_frames(const std::string& path, const std::vector<int64_t>& indices, int threads,
                                 const VideoInfo* info) {
  WEMM_CHECK(!indices.empty(), "no frames requested");
  WEMM_CHECK(std::is_sorted(indices.begin(), indices.end()), "frame indices must be sorted");
  DecodedVideo out;
  out.info = info ? *info : probe_video(path);
  out.frame_indices = indices;

  FormatCtx fmt;
  int sidx = open_stream(fmt, path);
  AVStream* st = fmt.ctx->streams[sidx];
  for (unsigned i = 0; i < fmt.ctx->nb_streams; ++i)
    if (int(i) != sidx) fmt.ctx->streams[i]->discard = AVDISCARD_ALL;

  const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
  WEMM_CHECK(codec, "no decoder for ", path);
  CodecCtx dec;
  dec.ctx = avcodec_alloc_context3(codec);
  int rc = avcodec_parameters_to_context(dec.ctx, st->codecpar);
  WEMM_CHECK(rc >= 0, "cannot configure decoder: ", av_err(rc));
  // FFmpeg's auto mode spawns one thread per core; beyond ~8 frame threads
  // decoding gets slower again.
  if (threads <= 0) threads = 8;
  dec.ctx->thread_count = threads;
  dec.ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
  rc = avcodec_open2(dec.ctx, codec, nullptr);
  WEMM_CHECK(rc >= 0, "cannot open decoder: ", av_err(rc));

  const int W = st->codecpar->width, H = st->codecpar->height;
  out.frames = at::empty({int64_t(indices.size()), H, W, 3}, at::kByte);
  uint8_t* dst_base = out.frames.data_ptr<uint8_t>();
  const int64_t frame_bytes = int64_t(H) * W * 3;

  Sws sws;
  // swscale's SIMD paths may write past the end of a row, so convert into a
  // padded, aligned scratch image and copy the rows out.
  const int scratch_stride = FFALIGN(W * 3, 64) + 64;
  std::unique_ptr<uint8_t, decltype(&av_free)> scratch(
      static_cast<uint8_t*>(av_malloc(size_t(scratch_stride) * (H + 1))), &av_free);
  WEMM_CHECK(scratch, "out of memory");
  size_t want = 0;       // next position in `indices`
  int64_t frame_no = 0;  // decoded frame counter
  Packet pkt;
  Frame frame;

  auto consume = [&](AVFrame* f) {
    while (want < indices.size() && indices[want] == frame_no) {
      WEMM_CHECK(f->width == W && f->height == H, "variable frame size is not supported");
      sws.ctx = sws_getCachedContext(sws.ctx, f->width, f->height, AVPixelFormat(f->format), W, H,
                                     AV_PIX_FMT_RGB24, kSwsFlags, nullptr, nullptr, nullptr);
      WEMM_CHECK(sws.ctx, "cannot create swscale context");
      uint8_t* dst = dst_base + int64_t(want) * frame_bytes;
      if (want > 0 && indices[want - 1] == frame_no) {
        std::memcpy(dst, dst - frame_bytes, size_t(frame_bytes));  // repeated index
      } else {
        uint8_t* planes[4] = {scratch.get(), nullptr, nullptr, nullptr};
        int strides[4] = {scratch_stride, 0, 0, 0};
        sws_scale(sws.ctx, f->data, f->linesize, 0, f->height, planes, strides);
        for (int y = 0; y < H; ++y)
          std::memcpy(dst + int64_t(y) * W * 3, scratch.get() + int64_t(y) * scratch_stride, size_t(W) * 3);
      }
      ++want;
    }
    ++frame_no;
  };
  auto drain = [&]() {
    while (true) {
      int r = avcodec_receive_frame(dec.ctx, frame.f);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return;
      WEMM_CHECK(r >= 0, "decode error: ", av_err(r));
      consume(frame.f);
      av_frame_unref(frame.f);
      if (want == indices.size()) return;
    }
  };

  while (want < indices.size() && av_read_frame(fmt.ctx, pkt.p) >= 0) {
    if (pkt.p->stream_index == sidx) {
      rc = avcodec_send_packet(dec.ctx, pkt.p);
      WEMM_CHECK(rc >= 0 || rc == AVERROR(EAGAIN), "send_packet failed: ", av_err(rc));
      drain();
    }
    av_packet_unref(pkt.p);
  }
  if (want < indices.size()) {
    avcodec_send_packet(dec.ctx, nullptr);
    drain();
  }
  WEMM_CHECK(want == indices.size(), "video ", path, " ended after ", frame_no, " frames; frame ",
             indices[want], " was requested");
  return out;
}

}  // namespace wemm
