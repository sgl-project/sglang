#include <cmath>
#include <stdexcept>
#include <string>

#include "metal_common.h"
#include "mlx/allocator.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

using namespace mlx::core;
using namespace sglang::metal_common;

namespace {

class PagedAttentionDecode : public Primitive {
 public:
  PagedAttentionDecode(Stream stream, int head_dim, int num_qo_heads, int num_kv_heads, float sm_scale)
      : Primitive(stream),
        head_dim_(head_dim),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        sm_scale_(sm_scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("paged_attention_decode: CPU eval not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    if (g_library == nullptr) {
      throw std::runtime_error("paged_attention_decode: register_library() not called yet");
    }

    auto& q = inputs[0];
    auto& k_pool = inputs[1];
    auto& v_pool = inputs[2];
    auto& kv_indptr = inputs[3];
    auto& kv_indices = inputs[4];
    auto& out = outputs[0];

    out.set_data(allocator::malloc(out.nbytes()));

    auto& d = metal::device(stream().device);
    auto& enc = command_encoder(stream());

    const uint32_t hd = static_cast<uint32_t>(head_dim_);
    const uint32_t nq = static_cast<uint32_t>(num_qo_heads_);
    const uint32_t nk = static_cast<uint32_t>(num_kv_heads_);
    const float scale = sm_scale_;

    auto consts = metal::MTLFCList{
        {&hd, MTL::DataType::DataTypeUInt, 0},
        {&nq, MTL::DataType::DataTypeUInt, 1},
        {&nk, MTL::DataType::DataTypeUInt, 2},
        {&scale, MTL::DataType::DataTypeFloat, 3},
    };

    const std::string kname = std::string("paged_attention_decode_") + dtype_suffix(q.dtype());
    const std::string hash = kname + "_hd" + std::to_string(head_dim_) + "_q" + std::to_string(num_qo_heads_) + "_k" +
                             std::to_string(num_kv_heads_) + "_s" +
                             std::to_string(static_cast<int>(sm_scale_ * 1000000.0f));
    auto* pipe = d.get_kernel(kname, g_library, hash, consts);
    if (!pipe) {
      throw std::runtime_error("paged_attention_decode: failed to resolve kernel");
    }

    enc.set_compute_pipeline_state(pipe);
    enc.set_input_array(q, 0);
    enc.set_input_array(k_pool, 1);
    enc.set_input_array(v_pool, 2);
    enc.set_input_array(kv_indptr, 3);
    enc.set_input_array(kv_indices, 4);
    enc.set_output_array(out, 5);

    const uint32_t batch = static_cast<uint32_t>(q.shape(0));
    enc.dispatch_threads(MTL::Size::Make(batch, nq, 32), MTL::Size::Make(1, 1, 32));
  }

  const char* name() const override {
    return "PagedAttentionDecode";
  }

  bool is_equivalent(const Primitive& other) const override {
    auto* o = dynamic_cast<const PagedAttentionDecode*>(&other);
    return o != nullptr && o->head_dim_ == head_dim_ && o->num_qo_heads_ == num_qo_heads_ &&
           o->num_kv_heads_ == num_kv_heads_ && o->sm_scale_ == sm_scale_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int head_dim_;
  int num_qo_heads_;
  int num_kv_heads_;
  float sm_scale_;
};

nb::object paged_attention_decode_py(
    nb::handle q_h,
    nb::handle k_pool_h,
    nb::handle v_pool_h,
    nb::handle kv_indptr_h,
    nb::handle kv_indices_h,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale) {
  auto& q = *nb::inst_ptr<array>(q_h);
  auto& k_pool = *nb::inst_ptr<array>(k_pool_h);
  auto& v_pool = *nb::inst_ptr<array>(v_pool_h);
  auto& kv_indptr = *nb::inst_ptr<array>(kv_indptr_h);
  auto& kv_indices = *nb::inst_ptr<array>(kv_indices_h);

  if (q.ndim() != 3) throw std::runtime_error("paged_attention_decode: q must be 3-D");
  if (k_pool.ndim() != 3 || v_pool.ndim() != 3) throw std::runtime_error("paged_attention_decode: pools must be 3-D");
  if (kv_indptr.ndim() != 1 || kv_indices.ndim() != 1)
    throw std::runtime_error("paged_attention_decode: kv_indptr/kv_indices must be 1-D");
  if (kv_indptr.dtype() != int32 || kv_indices.dtype() != int32)
    throw std::runtime_error("paged_attention_decode: kv_indptr/kv_indices must be int32");
  if (q.dtype() != k_pool.dtype() || q.dtype() != v_pool.dtype())
    throw std::runtime_error("paged_attention_decode: q/k_pool/v_pool must share dtype");
  if (q.shape(1) != num_qo_heads || q.shape(2) != head_dim)
    throw std::runtime_error("paged_attention_decode: q shape must be [batch, num_qo_heads, head_dim]");
  if (k_pool.shape(1) != num_kv_heads || v_pool.shape(1) != num_kv_heads || k_pool.shape(2) != head_dim ||
      v_pool.shape(2) != head_dim)
    throw std::runtime_error("paged_attention_decode: pool layout must be [pool_size, num_kv_heads, head_dim]");
  if (v_pool.shape(0) != k_pool.shape(0))
    throw std::runtime_error("paged_attention_decode: k_pool and v_pool must have same pool size");
  if (kv_indptr.shape(0) != q.shape(0) + 1)
    throw std::runtime_error("paged_attention_decode: kv_indptr must have batch + 1 entries");
  if (num_qo_heads % num_kv_heads != 0)
    throw std::runtime_error("paged_attention_decode: num_qo_heads must be divisible by num_kv_heads");
  if (head_dim > 128)
    throw std::runtime_error("paged_attention_decode: head_dim > 128 is not supported by this kernel");

  auto stream = default_stream(Device::gpu);
  auto primitive = std::make_shared<PagedAttentionDecode>(stream, head_dim, num_qo_heads, num_kv_heads, sm_scale);
  auto outs = array::make_arrays({q.shape()}, {q.dtype()}, primitive, {q, k_pool, v_pool, kv_indptr, kv_indices});
  return wrap_array(std::move(outs[0]));
}

}  // namespace

void register_paged_attention(nb::module_& m) {
  m.def(
      "paged_attention_decode",
      &paged_attention_decode_py,
      nb::arg("q"),
      nb::arg("k_pool"),
      nb::arg("v_pool"),
      nb::arg("kv_indptr"),
      nb::arg("kv_indices"),
      nb::arg("num_qo_heads"),
      nb::arg("num_kv_heads"),
      nb::arg("head_dim"),
      nb::arg("sm_scale"));
}
