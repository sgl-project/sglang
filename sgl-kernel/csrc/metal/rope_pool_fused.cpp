// Combined optimal: real AOT .metallib + Primitive integration + optimized
// 3-kernel + 3D-grid dispatch + fused KV pool write.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

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

void register_library_impl(const std::string& path) {
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

uint32_t pick_heads_per_thread(uint32_t nh) {
  if (nh == 0) return 1;
  if (const char* e = std::getenv("SGLANG_RPF_N")) {
    uint32_t v = static_cast<uint32_t>(std::atoi(e));
    if (v >= 1 && nh % v == 0) return v;
  }
  return 1u;
}

class RopePoolFused : public Primitive {
 public:
  RopePoolFused(Stream stream, int head_dim, int num_qo_heads, int num_kv_heads, float rope_base)
      : Primitive(stream),
        head_dim_(head_dim),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        rope_base_(rope_base) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("rope_pool_fused: CPU eval not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    if (g_library == nullptr) {
      throw std::runtime_error("rope_pool_fused: register_library() not called yet");
    }
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& positions = inputs[3];
    auto& slots = inputs[4];
    auto& k_pool_in = inputs[5];
    auto& v_pool_in = inputs[6];

    auto& q_out = outputs[0];
    auto& k_out = outputs[1];
    auto& k_pool_out = outputs[2];
    auto& v_pool_out = outputs[3];

    q_out.set_data(allocator::malloc(q_out.nbytes()));
    k_out.set_data(allocator::malloc(k_out.nbytes()));
    // Donate input pool buffers to outputs - zero-copy in-place semantics.
    k_pool_out.copy_shared_buffer(k_pool_in);
    v_pool_out.copy_shared_buffer(v_pool_in);

    auto& d = metal::device(stream().device);

    const uint32_t hd = static_cast<uint32_t>(head_dim_);
    const uint32_t nq = static_cast<uint32_t>(num_qo_heads_);
    const uint32_t nk = static_cast<uint32_t>(num_kv_heads_);
    const uint32_t half_dim = hd / 2;
    const uint32_t num_tokens = static_cast<uint32_t>(q.shape(0));
    const uint32_t hpt_q = pick_heads_per_thread(nq);
    const uint32_t hpt_k = pick_heads_per_thread(nk);
    const uint32_t hpt_v = pick_heads_per_thread(nk);

    const float inv_dim_log2_base = std::log2(rope_base_) / static_cast<float>(head_dim_);

    auto build_consts = [&](const uint32_t& hpt) {
      return metal::MTLFCList{
          {&hd, MTL::DataType::DataTypeUInt, 0},
          {&nq, MTL::DataType::DataTypeUInt, 1},
          {&nk, MTL::DataType::DataTypeUInt, 2},
          {&inv_dim_log2_base, MTL::DataType::DataTypeFloat, 3},
          {&hpt, MTL::DataType::DataTypeUInt, 4},
      };
    };

    auto build_hash = [&](const std::string& kname, uint32_t hpt) {
      return kname + "_hd" + std::to_string(head_dim_) + "_q" + std::to_string(num_qo_heads_) + "_k" +
             std::to_string(num_kv_heads_) + "_n" + std::to_string(hpt) + "_b" +
             std::to_string(static_cast<int>(rope_base_));
    };

    const std::string rect_kname = std::string("rope_pool_rect_") + dtype_suffix(q.dtype());
    const std::string q_kname = std::string("rope_q_") + dtype_suffix(q.dtype());
    const std::string k_kname = std::string("rope_k_pool_") + dtype_suffix(k.dtype());
    const std::string v_kname = std::string("v_to_pool_") + dtype_suffix(v.dtype());

    // Single rectangular dispatch uses one HEADS_PER_THREAD value for both
    // Q and KV heads. Keep it valid for both head counts.
    uint32_t hpt = std::min(hpt_q, hpt_k);
    if (nq % hpt != 0 || nk % hpt != 0) hpt = 1;

    auto& enc = metal::get_command_encoder(stream());

    const bool use_rect_dispatch = q.dtype() == bfloat16 && hd >= 128 && nk >= 8 && num_tokens >= 256;
    if (use_rect_dispatch) {
      auto rect_consts = build_consts(hpt);
      auto* rect_pipe = d.get_kernel(rect_kname, g_library, build_hash(rect_kname, hpt), rect_consts);
      if (!rect_pipe) {
        throw std::runtime_error("rope_pool_fused: failed to resolve rectangular kernel");
      }
      const uint32_t max_heads = std::max(nq, nk);
      const uint32_t gz = (max_heads + hpt - 1) / hpt;
      enc.set_compute_pipeline_state(rect_pipe);
      enc.set_input_array(q, 0);
      enc.set_input_array(k, 1);
      enc.set_input_array(v, 2);
      enc.set_output_array(q_out, 3);
      enc.set_output_array(k_out, 4);
      enc.set_output_array(k_pool_out, 5);
      enc.set_output_array(v_pool_out, 6);
      enc.set_input_array(positions, 7);
      enc.set_input_array(slots, 8);
      enc.dispatch_threads(MTL::Size::Make(hd, num_tokens, gz), pick_tg(hd, num_tokens, gz));
    } else {
      auto q_consts = build_consts(hpt_q);
      auto k_consts = build_consts(hpt_k);
      auto v_consts = build_consts(hpt_v);

      auto* q_pipe = d.get_kernel(q_kname, g_library, build_hash(q_kname, hpt_q), q_consts);
      auto* k_pipe = d.get_kernel(k_kname, g_library, build_hash(k_kname, hpt_k), k_consts);
      auto* v_pipe = d.get_kernel(v_kname, g_library, build_hash(v_kname, hpt_v), v_consts);
      if (!q_pipe || !k_pipe || !v_pipe) {
        throw std::runtime_error("rope_pool_fused: failed to resolve kernels");
      }

      // Kernel 1: Q rope
      {
        enc.set_compute_pipeline_state(q_pipe);
        enc.set_input_array(q, 0);
        enc.set_output_array(q_out, 1);
        enc.set_input_array(positions, 2);
        const uint32_t gz = (nq + hpt_q - 1) / hpt_q;
        enc.dispatch_threads(MTL::Size::Make(half_dim, num_tokens, gz), pick_tg(half_dim, num_tokens, gz));
      }

      // Kernel 2: K rope + pool write
      {
        enc.set_compute_pipeline_state(k_pipe);
        enc.set_input_array(k, 0);
        enc.set_output_array(k_out, 1);
        enc.set_output_array(k_pool_out, 2);
        enc.set_input_array(positions, 3);
        enc.set_input_array(slots, 4);
        const uint32_t gz = (nk + hpt_k - 1) / hpt_k;
        enc.dispatch_threads(MTL::Size::Make(half_dim, num_tokens, gz), pick_tg(half_dim, num_tokens, gz));
      }

      // Kernel 3: V copy to pool
      {
        enc.set_compute_pipeline_state(v_pipe);
        enc.set_input_array(v, 0);
        enc.set_output_array(v_pool_out, 1);
        enc.set_input_array(slots, 2);
        enc.dispatch_threads(MTL::Size::Make(hd, num_tokens, nk), pick_tg(hd, num_tokens, nk));
      }
    }
    // No commit / synchronize - MLX's lazy graph batches into one buffer.
  }

  const char* name() const override {
    return "RopePoolFused";
  }

  bool is_equivalent(const Primitive& other) const override {
    auto* o = dynamic_cast<const RopePoolFused*>(&other);
    return o != nullptr && o->head_dim_ == head_dim_ && o->num_qo_heads_ == num_qo_heads_ &&
           o->num_kv_heads_ == num_kv_heads_ && o->rope_base_ == rope_base_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape(), inputs[5].shape(), inputs[6].shape()};
  }

 private:
  int head_dim_;
  int num_qo_heads_;
  int num_kv_heads_;
  float rope_base_;
};

// Python entry: returns 4 arrays (q_rot, k_rot, k_pool_new, v_pool_new).
nb::tuple rope_pool_fused_py(
    nb::handle q_h,
    nb::handle k_h,
    nb::handle v_h,
    nb::handle positions_h,
    nb::handle slots_h,
    nb::handle k_pool_h,
    nb::handle v_pool_h,
    int head_dim,
    int num_qo_heads,
    int num_kv_heads,
    float rope_base) {
  auto& q = *nb::inst_ptr<array>(q_h);
  auto& k = *nb::inst_ptr<array>(k_h);
  auto& v = *nb::inst_ptr<array>(v_h);
  auto& positions = *nb::inst_ptr<array>(positions_h);
  auto& slots = *nb::inst_ptr<array>(slots_h);
  auto& k_pool = *nb::inst_ptr<array>(k_pool_h);
  auto& v_pool = *nb::inst_ptr<array>(v_pool_h);

  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3) throw std::runtime_error("rope_pool_fused: q/k/v must be 3-D");
  if (positions.ndim() != 1 || slots.ndim() != 1)
    throw std::runtime_error("rope_pool_fused: positions/slots must be 1-D");
  if (k_pool.ndim() != 3 || v_pool.ndim() != 3) throw std::runtime_error("rope_pool_fused: pools must be 3-D");
  if (positions.dtype() != int32 || slots.dtype() != int32)
    throw std::runtime_error("rope_pool_fused: positions/slots must be int32");
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || q.dtype() != k_pool.dtype() || q.dtype() != v_pool.dtype())
    throw std::runtime_error("rope_pool_fused: all float arrays must share dtype");
  if ((head_dim & 1) != 0) throw std::runtime_error("rope_pool_fused: head_dim must be even");
  // Shape cross-checks (catch any drift between Python pre-flight state
  // and actual tensors at dispatch time).
  const int num_tokens = q.shape(0);
  if (k.shape(0) != num_tokens || v.shape(0) != num_tokens || positions.shape(0) != num_tokens ||
      slots.shape(0) != num_tokens)
    throw std::runtime_error("rope_pool_fused: q/k/v/positions/slots must agree on token dim");
  if (q.shape(1) != num_qo_heads || k.shape(1) != num_kv_heads || v.shape(1) != num_kv_heads)
    throw std::runtime_error("rope_pool_fused: head-count mismatch with num_qo_heads/num_kv_heads");
  if (q.shape(2) != head_dim || k.shape(2) != head_dim || v.shape(2) != head_dim)
    throw std::runtime_error("rope_pool_fused: head_dim mismatch with q/k/v last dim");
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads || k_pool.shape(2) != head_dim ||
      v_pool.shape(2) != head_dim)
    throw std::runtime_error("rope_pool_fused: pool layout must be [pool_size, num_kv_heads, head_dim]");

  auto stream = default_stream(Device::gpu);
  auto primitive = std::make_shared<RopePoolFused>(stream, head_dim, num_qo_heads, num_kv_heads, rope_base);

  auto outs = array::make_arrays(
      {q.shape(), k.shape(), k_pool.shape(), v_pool.shape()},
      {q.dtype(), k.dtype(), k_pool.dtype(), v_pool.dtype()},
      primitive,
      {q, k, v, positions, slots, k_pool, v_pool});

  // Cross-module nb cast doesn't work cleanly - explicitly construct.
  nb::module_ mx_core = nb::module_::import_("mlx.core");
  nb::object py_array_type = mx_core.attr("array");

  nb::list result;
  for (auto& a : outs) {
    nb::object py_obj = py_array_type(0);
    auto* dst = nb::inst_ptr<array>(py_obj);
    new (dst) array(std::move(a));
    nb::inst_mark_ready(py_obj);
    result.append(py_obj);
  }
  return nb::tuple(result);
}

}  // namespace

NB_MODULE(_metal, m) {
  m.def("register_library", &register_library_impl, nb::arg("path"));

  m.def(
      "rope_pool_fused",
      &rope_pool_fused_py,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("positions"),
      nb::arg("slots"),
      nb::arg("k_pool"),
      nb::arg("v_pool"),
      nb::arg("head_dim"),
      nb::arg("num_qo_heads"),
      nb::arg("num_kv_heads"),
      nb::arg("rope_base"));
}
