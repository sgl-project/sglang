#pragma once

#include <sgl_kernel/distributed/push/epilogue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView;

/// `out = x`.
template <uint32_t kHiddenSize>
struct Store {
 public:
  using Vec = VecStorage<bf16_t>;
  template <uint32_t, uint32_t>
  using Smem = NoSmem;

  struct Params {
    bf16_t* out;  // [num_tokens, kHiddenSize]
  };

  SGL_DEVICE Store(const Params& p, uint32_t /*row*/, uint32_t /*col*/, uint32_t vid) : m_params(p), m_vid(vid) {}

  template <typename SmemT>
  SGL_DEVICE void store(const Vec& x, SmemT&) const {
    x.store(assume_global(m_params.out), m_vid);
  }

  static Params make_params(CallShape& shape, TensorView out) {
    shape.verify_tensor(out, kHiddenSize);
    return {.out = get_pointer<bf16_t>(out)};
  }

 private:
  const Params& m_params;
  uint32_t m_vid;
};

static_assert(Epilogue<Store<2560>>);

}  // namespace sglang::device::distributed::push
