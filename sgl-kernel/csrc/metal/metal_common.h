// Shared helpers for the Metal AOT kernels compiled into the single
// `sgl_kernel._metal` extension. Header-only so both kernel translation
// units get one inline definition with no shared-state linkage concerns.
#pragma once

#include <stdexcept>

#include "mlx/array.h"

namespace sgl_metal {

// Maps an MLX float dtype to the suffix used in Metal kernel host-names
// (e.g. "rms_norm_f16"). Throws for any unsupported dtype.
inline const char* dtype_suffix(mlx::core::Dtype dt) {
  switch (dt) {
    case mlx::core::float16:
      return "f16";
    case mlx::core::bfloat16:
      return "bf16";
    case mlx::core::float32:
      return "f32";
    default:
      throw std::runtime_error("sgl_metal: unsupported dtype");
  }
}

}  // namespace sgl_metal
