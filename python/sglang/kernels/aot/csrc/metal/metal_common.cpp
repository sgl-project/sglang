#include "metal_common.h"

#include <algorithm>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"

namespace sglang::metal_common {

constexpr const char* kLibraryName = "sgl_metal_kernels";

MTL::Library* g_library = nullptr;

const char* dtype_suffix(Dtype dt) {
  switch (dt) {
    case float16:
      return "f16";
    case bfloat16:
      return "bf16";
    case float32:
      return "f32";
    default:
      throw std::runtime_error("sgl_metal: unsupported dtype");
  }
}

void register_library(const std::string& path) {
  if (path.empty()) {
    throw std::runtime_error("register_library requires a non-empty path");
  }
  auto& d = metal::device(Device::gpu);
  g_library = d.get_library(kLibraryName, path);
  if (g_library == nullptr) {
    throw std::runtime_error("failed to load .metallib from: " + path);
  }
}

MTL::Size pick_tg(uint32_t gx, uint32_t gy, uint32_t gz) {
  constexpr uint32_t kMaxThreads = 256;
  uint32_t tx = std::min<uint32_t>(gx, 32u);
  uint32_t ty = std::min<uint32_t>(gy, kMaxThreads / std::max<uint32_t>(tx, 1u));
  uint32_t tz = std::min<uint32_t>(gz, kMaxThreads / std::max<uint32_t>(tx * ty, 1u));
  while (ty > 1 && (gy % ty) != 0)
    --ty;
  while (tz > 1 && (gz % tz) != 0)
    --tz;
  return MTL::Size::Make(tx, std::max<uint32_t>(ty, 1u), std::max<uint32_t>(tz, 1u));
}

metal::CommandEncoder& command_encoder(Stream stream) {
  return metal::get_command_encoder(stream);
}

nb::object wrap_array(array&& value) {
  nb::module_ mx_core = nb::module_::import_("mlx.core");
  nb::object py_array_type = mx_core.attr("array");
  nb::object py_obj = py_array_type(0);
  auto* dst = nb::inst_ptr<array>(py_obj);
  new (dst) array(std::move(value));
  nb::inst_mark_ready(py_obj);
  return py_obj;
}

}  // namespace sglang::metal_common
