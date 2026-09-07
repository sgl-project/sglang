#pragma once
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

namespace sglang {

namespace host::ffi {

using tvm::ffi::Tensor, tvm::ffi::TensorView, tvm::ffi::ShapeView;

inline Tensor empty(ShapeView shape, DLDataType dtype, DLDevice device) {
  return Tensor::FromEnvAlloc(::TVMFFIEnvTensorAlloc, shape, dtype, device);
}

inline Tensor empty_like(TensorView tensor) {
  return empty(tensor.shape(), tensor.dtype(), tensor.device());
}

struct _dummy_deleter {
  void operator()(void*) const {}
};

// template <typename Fn = _dummy_deleter>

template <typename Fn>
struct FromBlobContext {
  [[no_unique_address]] Fn deleter;
  int64_t dimension;
  int64_t* get_shape() {
    return reinterpret_cast<int64_t*>(this + 1);
  }
  int64_t* get_stride() {
    return this->get_shape() + dimension;
  }
};

template <typename Fn = _dummy_deleter>
inline Tensor from_blob(
    void* data,
    ShapeView shape,
    DLDataType dtype,
    DLDevice device,
    Fn&& deleter = {},
    std::optional<ShapeView> stride = {},
    uint64_t byte_offset = 0) {
  using Context = FromBlobContext<std::decay_t<Fn>>;
  const auto ndim = shape.size();
  const auto ctx = [&] {
    auto ptr = std::malloc(sizeof(Context) + sizeof(int64_t) * ndim * 2);
    auto ctx = static_cast<Context*>(ptr);
    std::construct_at(ctx, std::forward<Fn>(deleter), static_cast<int64_t>(ndim));
    stdr::copy_n(shape.data(), ndim, ctx->get_shape());
    if (stride.has_value()) {
      RuntimeCheck(stride->size() == ndim, "Stride ndim mismatch!");
      stdr::copy_n(stride->data(), ndim, ctx->get_stride());
    } else {
      int64_t stride_val = 1;
      for (const auto i : irange(ndim)) {
        const auto j = ndim - 1 - i;
        ctx->get_stride()[j] = stride_val;
        stride_val *= shape[j];
      }
    }
    return ctx;
  }();
  const auto tensor = DLTensor{
      .data = data,
      .device = device,
      .ndim = static_cast<int32_t>(ndim),
      .dtype = dtype,
      .shape = ctx->get_shape(),
      .strides = ctx->get_stride(),
      .byte_offset = byte_offset,
  };
  const auto blob_deleter = [](DLManagedTensor* self) {
    auto ctx = static_cast<Context*>(self->manager_ctx);
    ctx->deleter(self->dl_tensor.data);
    std::destroy_at(ctx);
    std::free(ctx);
  };
  auto managed_tensor = DLManagedTensor{tensor, ctx, blob_deleter};
  return Tensor::FromDLPack(&managed_tensor);
}

template <typename Fn = _dummy_deleter>
inline Tensor from_blob_like(
    void* data,
    TensorView t,
    Fn&& deleter = {},
    bool is_contiguous = false,  // if override to true, the stride will be ignored
    uint64_t byte_offset = 0) {
  const auto stride = is_contiguous ? std::nullopt : std::optional{t.strides()};
  return from_blob(data, t.shape(), t.dtype(), t.device(), std::forward<Fn>(deleter), stride, byte_offset);
}

inline Tensor alloc_workspace_tensor(size_t required_bytes, DLDevice device) {
  if (required_bytes == 0) return {};
  DLDataType u8 = {kDLUInt, 8, 1};
  int64_t shape[] = {static_cast<int64_t>(required_bytes)};
  return ffi::empty(tvm::ffi::ShapeView(shape, 1), u8, device);
}

}  // namespace host::ffi

// Declare `REF`, the not-nullable ObjectRef handle for `OBJ`.
//
// Whether `ref->` hands out a mutable pointer is decided by
// `OBJ::_type_mutable` (see TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE),
// so it belongs on the object, not here.
#define SGLANG_REGISTER_FFI_REFERENCE_CLASS(REF, OBJ)                             \
  struct REF : public tvm::ffi::ObjectRef {                                       \
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(REF, tvm::ffi::ObjectRef, OBJ); \
  }

}  // namespace sglang
