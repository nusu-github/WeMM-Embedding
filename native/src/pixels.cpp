#include "wemm/pixels.h"

#include "model_internal.h"

#include "wemm/common.h"

namespace wemm {

at::Tensor make_pixel_values(const Visual& v, const ProcessorConfig& cfg, const at::Device& device,
                             at::ScalarType dtype) {
  const int64_t p = cfg.patch_size, m = cfg.merge_size, tp = cfg.temporal_patch_size;
  at::Tensor x = v.frames.to(device, /*non_blocking=*/true).to(at::kFloat);  // [T,H,W,3]
  const int64_t T = x.size(0), H = x.size(1), W = x.size(2), C = x.size(3);
  WEMM_CHECK(H % (p * m) == 0 && W % (p * m) == 0, "frame size not divisible by patch*merge");

  float mean[3], std_[3];
  for (int c = 0; c < 3; ++c) {
    // BaseImageProcessorFast fuses rescale into normalize: mean/rescale, std/rescale.
    mean[c] = float(double(cfg.image_mean[c]) * (1.0 / cfg.rescale_factor));
    std_[c] = float(double(cfg.image_std[c]) * (1.0 / cfg.rescale_factor));
  }
  const auto ms = detail::Upload().add(mean, 3).add(std_, 3).to(device);
  x = (x - ms[0]) / ms[1];

  if (int64_t pad = (tp - T % tp) % tp) x = at::cat({x, x.slice(0, T - 1, T).expand({pad, H, W, C})}, 0);
  const int64_t gt = x.size(0) / tp, gh = H / p, gw = W / p;
  WEMM_CHECK(gt == v.grid_t && gh == v.grid_h && gw == v.grid_w, "visual grid mismatch");
  // [gt, tp, gh/m, m, p, gw/m, m, p, C] -> [gt, gh/m, gw/m, m, m, C, tp, p, p]
  x = x.view({gt, tp, gh / m, m, p, gw / m, m, p, C}).permute({0, 2, 5, 3, 6, 8, 1, 4, 7});
  return x.reshape({gt * gh * gw, C * tp * p * p}).to(dtype);
}

}  // namespace wemm
