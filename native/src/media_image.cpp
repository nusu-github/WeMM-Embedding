#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "wemm/common.h"
#include "wemm/media.h"

namespace wemm {

// ------------------------------------------------------------------ decode --
at::Tensor decode_image(const uint8_t* data, size_t size) {
  cv::Mat buf(1, static_cast<int>(size), CV_8U, const_cast<uint8_t*>(data));
  // IMREAD_UNCHANGED keeps alpha and does not apply EXIF orientation (PIL.Image.open
  // does not either).
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
  WEMM_CHECK(!img.empty(), "failed to decode image");
  if (img.depth() == CV_16U) {
    // Pillow: 16-bit RGB(A) keeps the high byte; 16-bit grayscale ("I;16")
    // converts to RGB by clipping to 255.
    cv::Mat out(img.rows, img.cols, CV_MAKETYPE(CV_8U, img.channels()));
    const bool gray = img.channels() == 1;
    for (int r = 0; r < img.rows; ++r) {
      const uint16_t* s = img.ptr<uint16_t>(r);
      uint8_t* d = out.ptr<uint8_t>(r);
      for (int i = 0; i < img.cols * img.channels(); ++i)
        d[i] = gray ? uint8_t(std::min<uint16_t>(s[i], 255)) : uint8_t(s[i] >> 8);
    }
    img = out;
  }
  WEMM_CHECK(img.depth() == CV_8U, "unsupported image bit depth");

  cv::Mat rgb;
  switch (img.channels()) {
    case 1:
      cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
      break;
    case 3:
      cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
      break;
    case 4: {
      // qwen-vl-utils: Image.new("RGB", size, white).paste(img, mask=alpha).
      // Pillow blends with DIV255(in1 * (255 - a) + in2 * a).
      rgb.create(img.rows, img.cols, CV_8UC3);
      at::parallel_for(0, img.rows, 16, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const uint8_t* s = img.ptr<uint8_t>(static_cast<int>(r));
          uint8_t* d = rgb.ptr<uint8_t>(static_cast<int>(r));
          for (int c = 0; c < img.cols; ++c, s += 4, d += 3) {
            const uint32_t a = s[3];
            for (int ch = 0; ch < 3; ++ch) {
              const uint32_t v = 255u * (255u - a) + uint32_t(s[2 - ch]) * a + 128u;
              d[ch] = static_cast<uint8_t>(((v >> 8) + v) >> 8);
            }
          }
        }
      });
      break;
    }
    default:
      throw Error("unsupported channel count " + std::to_string(img.channels()));
  }
  WEMM_CHECK(rgb.isContinuous(), "unexpected non-continuous image");
  auto t = at::empty({rgb.rows, rgb.cols, 3}, at::kByte);
  std::memcpy(t.data_ptr<uint8_t>(), rgb.ptr<uint8_t>(0), size_t(rgb.total()) * 3);
  return t;
}

std::string strip_file_scheme(const std::string& path) {
  return path.rfind("file://", 0) == 0 ? path.substr(7) : path;
}

at::Tensor load_image(const std::string& path) {
  const std::string p = strip_file_scheme(path);
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  WEMM_CHECK(in.good(), "cannot open image ", p);
  std::vector<uint8_t> bytes(size_t(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
  WEMM_CHECK(in.good() && !bytes.empty(), "cannot read image ", p);
  return decode_image(bytes.data(), bytes.size());
}

// --------------------------------------------------------- Pillow resample --
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

inline double bicubic_filter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

inline uint8_t clip8(int32_t in) {
  const int32_t v = in >> kPrecisionBits;  // arithmetic shift, as in Pillow
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

struct Coeffs {
  int ksize;
  std::vector<int> bounds;   // (xmin, xcount) per output
  std::vector<int32_t> kk;   // fixed-point coefficients
};

Coeffs precompute_coeffs(int in_size, float in0, float in1, int out_size) {
  constexpr double support_base = 2.0;
  double scale = static_cast<double>(in1 - in0) / out_size;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  double support = support_base * filterscale;
  int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  std::vector<double> kk(size_t(out_size) * ksize, 0.0);
  Coeffs c;
  c.ksize = ksize;
  c.bounds.resize(size_t(out_size) * 2);
  for (int xx = 0; xx < out_size; ++xx) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &kk[size_t(xx) * ksize];
    for (int x = 0; x < xmax; ++x) {
      double w = bicubic_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x)
      if (ww != 0.0) k[x] /= ww;
    c.bounds[xx * 2 + 0] = xmin;
    c.bounds[xx * 2 + 1] = xmax;
  }
  c.kk.resize(kk.size());
  for (size_t i = 0; i < kk.size(); ++i) {
    c.kk[i] = kk[i] < 0 ? static_cast<int32_t>(-0.5 + kk[i] * (1 << kPrecisionBits))
                        : static_cast<int32_t>(0.5 + kk[i] * (1 << kPrecisionBits));
  }
  return c;
}

}  // namespace

at::Tensor pil_resize_bicubic(const at::Tensor& src_in, int64_t out_h, int64_t out_w) {
  WEMM_CHECK(src_in.dim() == 3 && src_in.size(2) == 3 && src_in.scalar_type() == at::kByte,
             "pil_resize_bicubic expects [H,W,3] uint8");
  at::Tensor src = src_in.contiguous();
  const int in_h = static_cast<int>(src.size(0)), in_w = static_cast<int>(src.size(1));
  const int oh = static_cast<int>(out_h), ow = static_cast<int>(out_w);
  if (oh == in_h && ow == in_w) return src.clone();  // Image.resize returns a copy

  const bool need_h = ow != in_w;
  const bool need_v = oh != in_h;
  Coeffs ch = precompute_coeffs(in_w, 0.f, float(in_w), ow);
  Coeffs cv_ = precompute_coeffs(in_h, 0.f, float(in_h), oh);
  int ybox_first = cv_.bounds[0];
  int ybox_last = cv_.bounds[oh * 2 - 2] + cv_.bounds[oh * 2 - 1];

  at::Tensor cur = src;
  if (need_h) {
    for (int i = 0; i < oh; ++i) cv_.bounds[i * 2] -= ybox_first;
    const int rows = ybox_last - ybox_first;
    at::Tensor tmp = at::empty({rows, ow, 3}, at::kByte);
    const uint8_t* in = src.data_ptr<uint8_t>();
    uint8_t* out = tmp.data_ptr<uint8_t>();
    at::parallel_for(0, rows, 8, [&](int64_t y0, int64_t y1) {
      for (int64_t yy = y0; yy < y1; ++yy) {
        const uint8_t* row = in + (yy + ybox_first) * int64_t(in_w) * 3;
        uint8_t* orow = out + yy * int64_t(ow) * 3;
        for (int xx = 0; xx < ow; ++xx) {
          const int xmin = ch.bounds[xx * 2], xmax = ch.bounds[xx * 2 + 1];
          const int32_t* k = &ch.kk[size_t(xx) * ch.ksize];
          int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
          for (int x = 0; x < xmax; ++x) {
            const uint8_t* p = row + (x + xmin) * 3;
            s0 += p[0] * k[x];
            s1 += p[1] * k[x];
            s2 += p[2] * k[x];
          }
          orow[xx * 3 + 0] = clip8(s0);
          orow[xx * 3 + 1] = clip8(s1);
          orow[xx * 3 + 2] = clip8(s2);
        }
      }
    });
    cur = tmp;
  }
  if (need_v) {
    const int w = static_cast<int>(cur.size(1));
    at::Tensor dst = at::empty({oh, w, 3}, at::kByte);
    const uint8_t* in = cur.data_ptr<uint8_t>();
    uint8_t* out = dst.data_ptr<uint8_t>();
    at::parallel_for(0, oh, 8, [&](int64_t y0, int64_t y1) {
      for (int64_t yy = y0; yy < y1; ++yy) {
        const int ymin = cv_.bounds[yy * 2], ymax = cv_.bounds[yy * 2 + 1];
        const int32_t* k = &cv_.kk[size_t(yy) * cv_.ksize];
        uint8_t* orow = out + yy * int64_t(w) * 3;
        for (int xx = 0; xx < w * 3; xx += 3) {
          int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
          for (int y = 0; y < ymax; ++y) {
            const uint8_t* p = in + (int64_t(y + ymin) * w) * 3 + xx;
            s0 += p[0] * k[y];
            s1 += p[1] * k[y];
            s2 += p[2] * k[y];
          }
          orow[xx + 0] = clip8(s0);
          orow[xx + 1] = clip8(s1);
          orow[xx + 2] = clip8(s2);
        }
      }
    });
    cur = dst;
  }
  return cur;
}

// ------------------------------------------------------ torch AA resizes ----
at::Tensor torch_resize_uint8(const at::Tensor& hwc, int64_t out_h, int64_t out_w) {
  WEMM_CHECK(hwc.scalar_type() == at::kByte && hwc.size(-1) == 3, "expects [...,H,W,3] uint8");
  const int64_t H = hwc.size(-3), W = hwc.size(-2);
  if (H == out_h && W == out_w) return hwc;
  auto lead = hwc.sizes().slice(0, hwc.dim() - 3).vec();
  // NCHW view over NHWC memory (channels_last), exactly like torchvision's
  // `from_numpy(hwc).permute(2, 0, 1)` input.
  at::Tensor x = hwc.reshape({-1, H, W, 3}).permute({0, 3, 1, 2});
  at::Tensor y = at::_upsample_bicubic2d_aa(x, {out_h, out_w}, /*align_corners=*/false);
  y = y.permute({0, 2, 3, 1}).contiguous();
  lead.insert(lead.end(), {out_h, out_w, 3});
  return y.reshape(lead);
}

at::Tensor torch_resize_uint8_via_float(const at::Tensor& thwc, int64_t out_h, int64_t out_w) {
  WEMM_CHECK(thwc.scalar_type() == at::kByte && thwc.dim() == 4, "expects [T,H,W,3] uint8");
  // qwen-vl-utils: decord frames -> torch.tensor(...).permute(0, 3, 1, 2) -> v1 resize.
  // v1 resize always interpolates (no same-size shortcut).
  at::Tensor x = thwc.permute({0, 3, 1, 2}).to(at::kFloat);
  at::Tensor y = at::_upsample_bicubic2d_aa(x, {out_h, out_w}, /*align_corners=*/false);
  y = y.clamp_(0, 255).round_().to(at::kByte);
  return y.permute({0, 2, 3, 1}).contiguous();
}

}  // namespace wemm
