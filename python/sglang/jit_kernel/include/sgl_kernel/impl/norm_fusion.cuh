#pragma once
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil, RuntimeCheck

#include <tvm/ffi/container/tensor.h>

namespace host::norm_fusion {

// IndexEnum for scale/shift/gate tensor indexing
// Scalar: [1]
// NoBroadcast: [B, S, D]
// BroadcastB: [1, S, D]
// BroadcastS: [B, D], [B, 1, D]
// BroadcastBS: [D], [1, D], [1, 1, D]
// BF1D: [B, F, 1, D]
enum class IndexEnum : int {
  NotATensor = -1,
  Scalar = 0,
  NoBroadcast = 1,
  BroadcastB = 2,
  BroadcastS = 3,
  BroadcastBS = 4,
  BF1D = 5,
};

/**
 * \brief Host-side shape and layout checker for parameter tensors.
 */
template <typename T>
struct Matcher {
  SymbolicSize B_ = {"B"}, S_ = {"S"}, F_ = {"F"}, D_ = {"D"};
  bool has_value_F = false;

  template <IndexEnum index_enum>
  void match(const tvm::ffi::TensorView& tensor) {
    if constexpr (index_enum == IndexEnum::NotATensor) {
      // No check
    } else if constexpr (index_enum == IndexEnum::Scalar) {
      TensorMatcher({1}).with_dtype<T>().template with_device<kDLCUDA>().verify(tensor);
    } else if constexpr (index_enum == IndexEnum::NoBroadcast) {
      SymbolicSize S0_;
      TensorMatcher({B_, S_, D_})
          .with_strides({S0_, D_, 1})
          .with_dtype<T>()
          .template with_device<kDLCUDA>()
          .verify(tensor);
    } else if constexpr (index_enum == IndexEnum::BroadcastB) {
      SymbolicSize S0_;
      TensorMatcher({1, S_, D_})
          .with_strides({S0_, D_, 1})
          .with_dtype<T>()
          .template with_device<kDLCUDA>()
          .verify(tensor);
    } else if constexpr (index_enum == IndexEnum::BroadcastS) {
      SymbolicSize S0_, S1_;
      if (tensor.ndim() == 2) {
        TensorMatcher({B_, D_}).with_strides({D_, 1}).with_dtype<T>().template with_device<kDLCUDA>().verify(tensor);
      } else if (tensor.ndim() == 3) {
        TensorMatcher({B_, 1, D_})
            .with_strides({S0_, D_, 1})
            .with_dtype<T>()
            .template with_device<kDLCUDA>()
            .verify(tensor);
      } else {
        RuntimeCheck(
            false, "Invalid tensor rank for index_enum=BroadcastS: expected ndim=2 ([B, D]) or ndim=3 ([B, 1, D]).");
      }
    } else if constexpr (index_enum == IndexEnum::BroadcastBS) {
      SymbolicSize S0_, S1_;
      if (tensor.ndim() == 1) {
        TensorMatcher({D_}).with_strides({1}).with_dtype<T>().template with_device<kDLCUDA>().verify(tensor);
      } else if (tensor.ndim() == 2) {
        TensorMatcher({1, D_}).with_strides({S0_, 1}).with_dtype<T>().template with_device<kDLCUDA>().verify(tensor);
      } else if (tensor.ndim() == 3) {
        TensorMatcher({1, 1, D_})
            .with_strides({S0_, S1_, 1})
            .with_dtype<T>()
            .template with_device<kDLCUDA>()
            .verify(tensor);
      } else {
        RuntimeCheck(
            false, "Invalid tensor rank for index_enum=BroadcastS: expected ndim=2 ([1, D]) or ndim=3 ([1, 1, D]).");
      }
    } else if constexpr (index_enum == IndexEnum::BF1D) {
      has_value_F = true;
      SymbolicSize S0, S1, S2, S3;
      TensorMatcher({B_, F_, S3, D_})
          .with_dtype<T>()
          .with_strides({S0, S1, S2, 1})
          .template with_device<kDLCUDA>()
          .verify(tensor);
      const auto S = S_.unwrap();
      const auto F = F_.unwrap();
      RuntimeCheck(S % F == 0, "S must be divisible by F for 4D scale/shift");
    } else {
      RuntimeCheck(false, "Unknown index_enum");
    }
  }
};

}  // namespace host::norm_fusion

namespace device::norm_fusion {

using host::norm_fusion::IndexEnum;

/**
 * \brief Compute the linear index offset for a parameter tensor under a
 *        given IndexEnum.
 */
template <IndexEnum index_enum>
SGL_DEVICE int get_offset(int S, int F, int b_id, int s_id) {
  if constexpr (index_enum == IndexEnum::NotATensor || index_enum == IndexEnum::Scalar) {
    return 0;
  } else if constexpr (index_enum == IndexEnum::NoBroadcast) {
    return b_id * S + s_id;
  } else if constexpr (index_enum == IndexEnum::BroadcastB) {
    return s_id;
  } else if constexpr (index_enum == IndexEnum::BroadcastS) {
    return b_id;
  } else if constexpr (index_enum == IndexEnum::BroadcastBS) {
    return 0;
  } else if constexpr (index_enum == IndexEnum::BF1D) {
    int frame_len = S / F;
    return b_id * F + s_id / frame_len;
  } else {
    static_assert(index_enum != index_enum, "Unsupported IndexEnum");
  }
}
}  // namespace device::norm_fusion
