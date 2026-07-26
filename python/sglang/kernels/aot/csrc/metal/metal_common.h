#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace nb = nanobind;
using namespace mlx::core;

namespace sglang::metal_common {

extern MTL::Library* g_library;

const char* dtype_suffix(Dtype dt);
void register_library(const std::string& path);
MTL::Size pick_tg(uint32_t gx, uint32_t gy, uint32_t gz);
metal::CommandEncoder& command_encoder(Stream stream);
nb::object wrap_array(array&& value);

}  // namespace sglang::metal_common

void register_rope_pool_fused(nb::module_& m);
void register_paged_attention(nb::module_& m);
