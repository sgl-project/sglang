#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

#include "metal_common.h"
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

namespace nb = nanobind;
using namespace mlx::core;
using namespace sglang::metal_common;

namespace {

struct Plan {
  int dim, heads, kv_heads, splits, group;
  float scale;
  bool tail;

  bool operator==(const Plan& other) const {
    return std::tie(dim, heads, kv_heads, splits, group, scale, tail) ==
           std::tie(other.dim, other.heads, other.kv_heads, other.splits, other.group, other.scale, other.tail);
  }
};

class RadixAttention : public Primitive {
 public:
  RadixAttention(Stream stream, Plan plan, bool reduce) : Primitive(stream), plan_(plan), reduce_(reduce) {}

  const char* name() const override {
    return reduce_ ? "AotRadixReduce" : "AotRadixAttention";
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const RadixAttention*>(&other);
    return rhs != nullptr && plan_ == rhs->plan_ && reduce_ == rhs->reduce_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const int batch = inputs[0].shape(0);
    if (!reduce_ && plan_.splits > 1) return {{batch, plan_.heads, plan_.splits, plan_.dim + 2}};
    return {{batch, plan_.heads, plan_.dim}};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("AOT radix attention requires a Metal GPU stream");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    if (g_library == nullptr) throw std::runtime_error("radix_attention: register_library() not called yet");
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    auto& device = metal::device(stream().device);
    const auto& p = plan_;
    const uint32_t heads = p.heads, kv_heads = p.kv_heads, splits = p.splits;
    const uint32_t rows = reduce_ ? 0 : inputs[5].shape(0);
    const uint32_t width = reduce_ ? 0 : inputs[5].shape(1);
    const uint32_t slots = reduce_ ? 0 : inputs[3].shape(0);
    metal::MTLFCList constants = {
        {&heads, MTL::DataType::DataTypeUInt, 0},
        {&splits, MTL::DataType::DataTypeUInt, 2},
    };
    std::string kernel = reduce_ ? "radix_reduce_" : "radix_";
    kernel += dtype_suffix(reduce_ ? out.dtype() : inputs[0].dtype());
    kernel += "_d" + std::to_string(p.dim);
    if (!reduce_) {
      kernel += "_w4_g" + std::to_string(p.group) + "_p" + std::to_string(p.splits > 1);
      constants.emplace_back(&kv_heads, MTL::DataType::DataTypeUInt, 1);
      constants.emplace_back(&p.scale, MTL::DataType::DataTypeFloat, 3);
      constants.emplace_back(&p.tail, MTL::DataType::DataTypeBool, 4);
      constants.emplace_back(&rows, MTL::DataType::DataTypeUInt, 5);
      constants.emplace_back(&width, MTL::DataType::DataTypeUInt, 6);
      constants.emplace_back(&slots, MTL::DataType::DataTypeUInt, 7);
    }
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &p.scale, sizeof(scale_bits));
    const std::string key = kernel + "_h" + std::to_string(heads) + "_k" + std::to_string(kv_heads) + "_s" +
                            std::to_string(splits) + "_scale" + std::to_string(scale_bits) + "_tail" +
                            std::to_string(p.tail) + "_rows" + std::to_string(rows) + "_width" + std::to_string(width) +
                            "_slots" + std::to_string(slots);
    auto* pipeline = device.get_kernel(kernel, g_library, key, constants);
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(pipeline);
    if (reduce_) {
      encoder.set_input_array(inputs[0], 0);
      encoder.set_output_array(out, 1);
    } else {
      for (int i = 0; i < 8; ++i)
        encoder.set_input_array(inputs[i], i);
      encoder.set_input_array(inputs[p.tail ? 8 : 1], 8);
      encoder.set_input_array(inputs[p.tail ? 9 : 2], 9);
      encoder.set_output_array(out, 10);
    }
    encoder.dispatch_threadgroups(
        MTL::Size::Make(reduce_ ? 1 : splits, heads / (reduce_ ? 1 : p.group), out.shape(0)),
        MTL::Size::Make(reduce_ ? 32 : 128, 1, 1));
  }

 private:
  Plan plan_;
  bool reduce_;
};

void validate(const std::vector<array>& a, float scale, bool tail) {
  const auto& q = a[0];
  const auto& k = a[1];
  const auto& pool = a[3];
  const auto& table = a[5];
  if (q.ndim() != 3 || k.ndim() != 3 || pool.ndim() != 3 || table.ndim() != 2)
    throw std::invalid_argument("Compiled radix requires 3D Q/K/V/pools and a 2D table");
  const int batch = q.shape(0), heads = q.shape(1), dim = q.shape(2), kv_heads = k.shape(1);
  if (batch < 1 || heads < 1 || (dim != 64 && dim != 128 && dim != 256) || kv_heads < 1 || heads % kv_heads ||
      k.shape() != Shape{batch, kv_heads, dim} || a[2].shape() != k.shape() || pool.shape(0) < 1 ||
      pool.shape(1) != kv_heads || pool.shape(2) != dim || a[4].shape() != pool.shape() || table.shape(0) < 1 ||
      table.shape(1) < 1 || a[6].shape() != Shape{batch} || a[7].shape() != Shape{batch} || !std::isfinite(scale) ||
      scale <= 0)
    throw std::invalid_argument("Unsupported compiled radix attention geometry");
  if (q.dtype() != float16 && q.dtype() != bfloat16 && q.dtype() != float32)
    throw std::invalid_argument("Unsupported compiled radix attention dtype");
  for (int i = 1; i < 5; ++i)
    if (a[i].dtype() != q.dtype()) throw std::invalid_argument("Unsupported compiled radix attention dtype");
  if (table.dtype() != int32 || (a[6].dtype() != int32 && a[6].dtype() != int64) ||
      (a[7].dtype() != int32 && a[7].dtype() != int64))
    throw std::invalid_argument("Unsupported compiled radix attention dtype");
  if (tail && (a[8].shape() != k.shape() || a[9].shape() != k.shape() || a[8].dtype() != q.dtype() ||
               a[9].dtype() != q.dtype()))
    throw std::invalid_argument("Pending K/V must match the current-token K/V");
}

nb::object radix_py(
    nb::handle q,
    nb::handle k,
    nb::handle v,
    nb::handle kp,
    nb::handle vp,
    nb::handle table,
    nb::handle requests,
    nb::handle lengths,
    float scale,
    nb::handle tail_k,
    nb::handle tail_v) {
  auto array_type = nb::module_::import_("mlx.core").attr("array");
  auto unwrap = [&](nb::handle object) -> array {
    if (!nb::isinstance(object, array_type)) throw nb::type_error("AOT radix inputs must be MLX arrays");
    return *nb::inst_ptr<array>(object);
  };
  std::vector<array> inputs = {
      unwrap(q), unwrap(k), unwrap(v), unwrap(kp), unwrap(vp), unwrap(table), unwrap(requests), unwrap(lengths)};
  const bool tail = !tail_k.is_none();
  if (tail != !tail_v.is_none()) throw std::invalid_argument("Pending K/V must be supplied together");
  if (tail) {
    inputs.push_back(unwrap(tail_k));
    inputs.push_back(unwrap(tail_v));
  }
  validate(inputs, scale, tail);
  const int batch = inputs[0].shape(0), heads = inputs[0].shape(1), dim = inputs[0].shape(2);
  const int kv_heads = inputs[1].shape(1);
  const int group = (heads / kv_heads) % 2 == 0 ? 2 : 1;
  const int64_t groups = int64_t(batch) * heads / group;
  // Measured on M4 Pro: grouped heads need more KV partitions to fill the GPU.
  const int splits = std::min<int64_t>(16, std::max<int64_t>(1, 128 / groups));
  const Stream stream = default_stream(Device::gpu);
  for (auto& input : inputs)
    input = contiguous(input, false, stream);
  inputs[6] = astype(inputs[6], int64, stream);
  inputs[7] = astype(inputs[7], int64, stream);
  const Plan plan{dim, heads, kv_heads, splits, group, scale, tail};
  const auto dtype = inputs[0].dtype();
  const Shape shape = splits > 1 ? Shape{batch, heads, splits, dim + 2} : inputs[0].shape();
  array result(shape, splits > 1 ? float32 : dtype, std::make_shared<RadixAttention>(stream, plan, false), inputs);
  if (splits > 1)
    result = array({batch, heads, dim}, dtype, std::make_shared<RadixAttention>(stream, plan, true), {result});
  return wrap_array(std::move(result));
}

}  // namespace

void register_radix_attention(nb::module_& module) {
  module.def(
      "radix_attention",
      &radix_py,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("kp"),
      nb::arg("vp"),
      nb::arg("table"),
      nb::arg("requests"),
      nb::arg("lengths"),
      nb::arg("scale"),
      nb::arg("tail_k") = nb::none(),
      nb::arg("tail_v") = nb::none());
}
