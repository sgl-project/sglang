#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

namespace sglang::metal_common {

extern MTL::Library* g_library;

const char* dtype_suffix(mlx::core::Dtype dtype);
void register_library(const std::string& path);
nanobind::object wrap_array(mlx::core::array&& value);

}  // namespace sglang::metal_common

void register_rope_pool_fused(nanobind::module_& module);
void register_radix_attention(nanobind::module_& module);
