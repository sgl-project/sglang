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

class BlockPagedAttentionDecode : public Primitive {
 public:
  BlockPagedAttentionDecode(
      Stream stream, int head_dim, int num_qo_heads, int num_kv_heads, int block_size, float sm_scale)
      : Primitive(stream),
        head_dim_(head_dim),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        block_size_(block_size),
        sm_scale_(sm_scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("block_paged_attention_decode: CPU eval not supported");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    if (g_library == nullptr) {
      throw std::runtime_error("block_paged_attention_decode: register_library() not called yet");
    }

    auto& q = inputs[0];
    auto& k_blocks = inputs[1];
    auto& v_blocks = inputs[2];
    auto& block_tables = inputs[3];
    auto& seq_lens = inputs[4];
    auto& out = outputs[0];

    out.set_data(allocator::malloc(out.nbytes()));

    auto& d = metal::device(stream().device);
    auto& enc = command_encoder(stream());

    const uint32_t hd = static_cast<uint32_t>(head_dim_);
    const uint32_t nq = static_cast<uint32_t>(num_qo_heads_);
    const uint32_t nk = static_cast<uint32_t>(num_kv_heads_);
    const uint32_t bs = static_cast<uint32_t>(block_size_);
    const uint32_t mb = static_cast<uint32_t>(block_tables.shape(1));
    const float scale = sm_scale_;

    auto consts = metal::MTLFCList{
        {&hd, MTL::DataType::DataTypeUInt, 0},
        {&nq, MTL::DataType::DataTypeUInt, 1},
        {&nk, MTL::DataType::DataTypeUInt, 2},
        {&scale, MTL::DataType::DataTypeFloat, 3},
        {&bs, MTL::DataType::DataTypeUInt, 4},
    };

    const std::string kname = std::string("block_paged_attention_decode_") + dtype_suffix(q.dtype());
    const std::string hash = kname + "_hd" + std::to_string(head_dim_) + "_q" + std::to_string(num_qo_heads_) + "_k" +
                             std::to_string(num_kv_heads_) + "_bs" + std::to_string(block_size_) + "_s" +
                             std::to_string(static_cast<int>(sm_scale_ * 1000000.0f));
    auto* pipe = d.get_kernel(kname, g_library, hash, consts);
    if (!pipe) {
      throw std::runtime_error("block_paged_attention_decode: failed to resolve kernel");
    }

    enc.set_compute_pipeline_state(pipe);
    enc.set_input_array(q, 0);
    enc.set_input_array(k_blocks, 1);
    enc.set_input_array(v_blocks, 2);
    enc.set_input_array(block_tables, 3);
    enc.set_input_array(seq_lens, 4);
    enc.set_output_array(out, 5);
    enc.set_bytes(&mb, sizeof(uint32_t), 6);

    const uint32_t batch = static_cast<uint32_t>(q.shape(0));
    enc.dispatch_threads(MTL::Size::Make(batch, nq, 256), MTL::Size::Make(1, 1, 256));
  }

  const char* name() const override {
    return "BlockPagedAttentionDecode";
  }

  bool is_equivalent(const Primitive& other) const override {
    auto* o = dynamic_cast<const BlockPagedAttentionDecode*>(&other);
    return o != nullptr && o->head_dim_ == head_dim_ && o->num_qo_heads_ == num_qo_heads_ &&
           o->num_kv_heads_ == num_kv_heads_ && o->block_size_ == block_size_ && o->sm_scale_ == sm_scale_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  int head_dim_;
  int num_qo_heads_;
  int num_kv_heads_;
  int block_size_;
  float sm_scale_;
};

nb::object block_paged_attention_decode_py(
    nb::handle q_h,
    nb::handle k_blocks_h,
    nb::handle v_blocks_h,
    nb::handle block_tables_h,
    nb::handle seq_lens_h,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float sm_scale) {
  auto& q = *nb::inst_ptr<array>(q_h);
  auto& k_blocks = *nb::inst_ptr<array>(k_blocks_h);
  auto& v_blocks = *nb::inst_ptr<array>(v_blocks_h);
  auto& block_tables = *nb::inst_ptr<array>(block_tables_h);
  auto& seq_lens = *nb::inst_ptr<array>(seq_lens_h);

  if (q.ndim() != 3) throw std::runtime_error("block_paged_attention_decode: q must be 3-D");
  if (k_blocks.ndim() != 4 || v_blocks.ndim() != 4)
    throw std::runtime_error("block_paged_attention_decode: K/V blocks must be 4-D");
  if (block_tables.ndim() != 2 || seq_lens.ndim() != 1)
    throw std::runtime_error("block_paged_attention_decode: block_tables must be 2-D and seq_lens must be 1-D");
  if (block_tables.dtype() != int32 || seq_lens.dtype() != int32)
    throw std::runtime_error("block_paged_attention_decode: block_tables/seq_lens must be int32");
  if (q.dtype() != k_blocks.dtype() || q.dtype() != v_blocks.dtype())
    throw std::runtime_error("block_paged_attention_decode: q/k_blocks/v_blocks must share dtype");
  if (q.shape(1) != num_qo_heads || q.shape(2) != head_dim)
    throw std::runtime_error("block_paged_attention_decode: q shape must be [batch, num_qo_heads, head_dim]");
  if (k_blocks.shape(1) != block_size || v_blocks.shape(1) != block_size || k_blocks.shape(2) != num_kv_heads ||
      v_blocks.shape(2) != num_kv_heads || k_blocks.shape(3) != head_dim || v_blocks.shape(3) != head_dim)
    throw std::runtime_error(
        "block_paged_attention_decode: K/V layout must be [num_blocks, block_size, num_kv_heads, head_dim]");
  if (v_blocks.shape(0) != k_blocks.shape(0))
    throw std::runtime_error("block_paged_attention_decode: k_blocks and v_blocks must have same block count");
  if (block_tables.shape(0) != q.shape(0) || seq_lens.shape(0) != q.shape(0))
    throw std::runtime_error("block_paged_attention_decode: metadata batch dimension must match q");
  if (num_qo_heads % num_kv_heads != 0)
    throw std::runtime_error("block_paged_attention_decode: num_qo_heads must be divisible by num_kv_heads");
  if (head_dim > 256)
    throw std::runtime_error("block_paged_attention_decode: head_dim > 256 is not supported by this kernel");
  if (block_size <= 0) throw std::runtime_error("block_paged_attention_decode: block_size must be positive");

  auto stream = default_stream(Device::gpu);
  auto primitive =
      std::make_shared<BlockPagedAttentionDecode>(stream, head_dim, num_qo_heads, num_kv_heads, block_size, sm_scale);
  auto outs = array::make_arrays({q.shape()}, {q.dtype()}, primitive, {q, k_blocks, v_blocks, block_tables, seq_lens});
  return wrap_array(std::move(outs[0]));
}

}  // namespace

void register_paged_attention(nb::module_& m) {
  m.def(
      "block_paged_attention_decode",
      &block_paged_attention_decode_py,
      nb::arg("q"),
      nb::arg("k_blocks"),
      nb::arg("v_blocks"),
      nb::arg("block_tables"),
      nb::arg("seq_lens"),
      nb::arg("num_qo_heads"),
      nb::arg("num_kv_heads"),
      nb::arg("head_dim"),
      nb::arg("block_size"),
      nb::arg("sm_scale"));
}
