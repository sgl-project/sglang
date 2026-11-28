#pragma once

#include <sgl_kernel/distributed/push/prologue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView;

/// The plain prologue: loads this rank's `partial`.
struct Load {
 public:
  using Vec = VecStorage<bf16_t>;

  struct Params {
    const bf16_t* partial;  // [num_tokens, hidden_size]
  };

  SGL_DEVICE Load(const Params& p, uint32_t /*row*/, uint32_t /*col*/, uint32_t vid) : m_params(p), m_vid(vid) {}

  SGL_DEVICE Vec load() const {
    Vec v;
    v.load(assume_global(m_params.partial), m_vid);
    return v;
  }

  static Params make_params(CallShape& shape, TensorView partial, int64_t hidden_size) {
    shape.verify_tensor(partial, hidden_size);
    return {.partial = get_pointer<const bf16_t>(partial)};
  }

 private:
  const Params& m_params;
  uint32_t m_vid;
};

static_assert(Prologue<Load>);

}  // namespace sglang::device::distributed::push
