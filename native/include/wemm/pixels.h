#pragma once

#include <ATen/ATen.h>

#include "wemm/config.h"
#include "wemm/processor.h"

namespace wemm {

// Normalizes and flattens one visual into vision-encoder patches, exactly like
// the Transformers processors: (x - mean/rescale) / (std/rescale) in float32,
// temporal padding by repeating the last frame, and the
// (t, h/m, w/m, m, m, C, tp, p, p) patch order. Runs on `device`; returns
// [grid_t*grid_h*grid_w, C*tp*p*p] in `dtype`.
at::Tensor make_pixel_values(const Visual& v, const ProcessorConfig& cfg, const at::Device& device,
                             at::ScalarType dtype = at::kFloat);

}  // namespace wemm
