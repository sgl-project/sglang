/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/ffi.h>     // For ffi::empty
#include <sgl_kernel/tensor.h>  // For TensorView, RuntimeCheck, Panic
#include <sgl_kernel/utils.h>   // For host utilities

#include <sgl_kernel/runtime.cuh>  // For device/runtime helpers
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, RuntimeDeviceCheck

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

using namespace host;

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

// Allocate a raw uint8 workspace tensor on the given device (empty when unused).
inline auto alloc_workspace_tensor(size_t required_bytes, DLDevice device) -> tvm::ffi::Tensor {
  if (required_bytes == 0) return {};
  DLDataType u8 = {kDLUInt, 8, 1};
  int64_t shape[] = {static_cast<int64_t>(required_bytes)};
  return ffi::empty(tvm::ffi::ShapeView(shape, 1), u8, device);
}

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}
