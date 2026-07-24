#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "metal_common.h"
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

namespace nb = nanobind;
using namespace mlx::core;

namespace {
constexpr const char* kLibraryName = "sgl_metal_kernels";

class RMSNorm : public Primitive {
 public:
  RMSNorm(Stream stream, float eps) : Primitive(stream), eps_(eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("rms_norm: CPU eval not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& out = outputs[0];

    out.set_data(allocator::malloc(out.nbytes()));

    auto& d = metal::device(stream().device);
    auto* lib = d.get_library(kLibraryName);
    if (lib == nullptr) {
      throw std::runtime_error("rms_norm: metallib not loaded; call register_library() first");
    }

    const uint32_t B = x.shape(0);
    const uint32_t H = x.shape(1);

    uint32_t TG = ((H + 31u) / 32u) * 32u;
    TG = std::min<uint32_t>(std::max<uint32_t>(TG, 32u), 256u);

    auto consts = metal::MTLFCList{
        {&H, MTL::DataType::DataTypeUInt, 0},
        {&eps_, MTL::DataType::DataTypeFloat, 1},
    };

    const std::string kname = std::string("rms_norm_") + sgl_metal::dtype_suffix(x.dtype());
    // Encode eps by its raw bits: the hash doubles as the Metal
    // function name, which must be a valid identifier (no '.').
    uint32_t eps_bits;
    std::memcpy(&eps_bits, &eps_, sizeof(eps_bits));
    const std::string hash = kname + "_H" + std::to_string(H) + "_eps" + std::to_string(eps_bits);

    auto* pipe = d.get_kernel(kname, lib, hash, consts);
    if (!pipe) {
      throw std::runtime_error("rms_norm: failed to resolve kernel");
    }
    auto& enc = metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(pipe);
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out, 2);
    enc.dispatch_threads(MTL::Size::Make(TG, B, 1), MTL::Size::Make(TG, 1, 1));
  }

  const char* name() const override {
    return "RMSNorm";
  }

  bool is_equivalent(const Primitive& other) const override {
    auto* o = dynamic_cast<const RMSNorm*>(&other);
    return o != nullptr && o->eps_ == eps_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  float eps_;
};

// Python entry: returns 1 output array.
nb::object rms_norm_py(nb::handle x_h, nb::handle w_h, float eps) {
  auto& x = *nb::inst_ptr<array>(x_h);
  auto& w = *nb::inst_ptr<array>(w_h);
  if (x.ndim() != 2 || w.ndim() != 1) {
    throw std::runtime_error("rms_norm: x must be 2D and w must be 1D");
  }
  if (x.shape(1) != w.shape(0)) {
    throw std::runtime_error("rms_norm: x and w must be (B, H) and (H,) in shape. H mismtach");
  }
  if (x.dtype() != w.dtype()) {
    throw std::runtime_error("rms_norm: x and w must be of same type");
  }

  auto stream = default_stream(Device::gpu);
  auto primitive = std::make_shared<RMSNorm>(stream, eps);
  auto outs = array::make_arrays({x.shape()}, {x.dtype()}, primitive, {x, w});

  // Cross-module nb cast doesn't work cleanly - explicitly construct.
  nb::module_ mx_core = nb::module_::import_("mlx.core");
  nb::object py_array_type = mx_core.attr("array");

  nb::object py_obj = py_array_type(0);
  auto* dst = nb::inst_ptr<array>(py_obj);
  *dst = std::move(outs[0]);
  nb::inst_mark_ready(py_obj);

  return py_obj;
}
}  // namespace

// External linkage so rope_pool_fused.cpp's NB_MODULE can register this kernel
// into the single shared `_metal` module. rms_norm_py lives in the anonymous
// namespace above but is visible here in the same translation unit.
void register_rms_norm(nb::module_& m) {
  m.def("rms_norm", &rms_norm_py, nb::arg("x"), nb::arg("w"), nb::arg("eps"));
}
