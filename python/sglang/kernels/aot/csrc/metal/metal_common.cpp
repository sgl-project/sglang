#include "metal_common.h"

#include <stdexcept>
#include <utility>

namespace nb = nanobind;
using namespace mlx::core;

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
      throw std::runtime_error("rope_pool_fused: unsupported dtype");
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

nb::object wrap_array(array&& value) {
  nb::object output = nb::module_::import_("mlx.core").attr("array")(0);
  *nb::inst_ptr<array>(output) = std::move(value);
  return output;
}

}  // namespace sglang::metal_common
