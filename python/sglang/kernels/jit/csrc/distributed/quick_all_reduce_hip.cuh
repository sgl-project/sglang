// TVM-FFI entry points for sgl-kernel's qr_* calls (kernels/aot/csrc/allreduce/quick_all_reduce.cu) over the
// JIT copy in quick_all_reduce/, which serves INT4 and INT8 at TP4 on bf16 cast to fp16 only.
#pragma once

#ifndef USE_ROCM
#error "quick_all_reduce_hip.cuh is ROCm-only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "quick_all_reduce/quick_all_reduce.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace sglang {
namespace quick_all_reduce_hip {

constexpr int64_t kHandleBytes = sizeof(hipIpcMemHandle_t);
// sgl-kernel's QuickReduceQuantLevel
constexpr int64_t kInt8 = 1;
constexpr int64_t kInt4 = 3;

struct QuickAllReduce {
  /// \brief Allocates this rank's communication buffer for a TP group of `world_size`.
  /// \param max_problem_size Largest all-reduce in bytes; <= 0 keeps sgl-kernel's 2 GiB default.
  /// \return An opaque DeviceComms handle, released by destroy().
  static int64_t init(int64_t world_size, int64_t rank, int64_t max_problem_size) {
    host::RuntimeCheck(world_size == 4, "the JIT quick all-reduce is built for TP4");
    host::RuntimeCheck(0 <= rank && rank < world_size, "invalid rank");
    auto* comms = new quickreduce::DeviceComms();
    comms->init(
        static_cast<int>(world_size),
        static_cast<int>(rank),
        max_problem_size > 0 ? std::optional<int64_t>(max_problem_size) : std::nullopt);
    return reinterpret_cast<int64_t>(comms);
  }

  /// \brief Writes this rank's IPC handle into `handle`, a CPU uint8 tensor of kHandleBytes.
  static void get_handle(int64_t comm, const tvm::ffi::TensorView handle) {
    using namespace host;
    TensorMatcher({kHandleBytes}).with_dtype<uint8_t>().with_device<kDLCPU>().verify(handle);
    const hipIpcMemHandle_t ipc = reinterpret_cast<quickreduce::DeviceComms*>(comm)->get_handle();
    std::memcpy(handle.data_ptr(), &ipc, kHandleBytes);
  }

  /// \brief Opens every rank's buffer from `handles`, CPU uint8 [world_size, kHandleBytes] in rank order.
  static void open_handles(int64_t comm, const tvm::ffi::TensorView handles) {
    using namespace host;
    auto* comms = reinterpret_cast<quickreduce::DeviceComms*>(comm);
    const int64_t world_size = comms->get_world_size();
    TensorMatcher({world_size, kHandleBytes}).with_dtype<uint8_t>().with_device<kDLCPU>().verify(handles);
    std::vector<hipIpcMemHandle_t> ipc(world_size);
    std::memcpy(ipc.data(), handles.data_ptr(), ipc.size() * kHandleBytes);
    comms->open_ipc_handles(ipc);
  }

  /// \brief out = sum of the bf16 `inp` over the group, out of place, on the current stream.
  /// \param quant_level 3 (INT4) or 1 (INT8) in sgl-kernel's QuickReduceQuantLevel.
  /// \param cast_bf2half must be true: the reduction runs in the fp16 kernels, as sgl-kernel's does.
  static void all_reduce(
      int64_t comm,
      const tvm::ffi::TensorView inp,
      const tvm::ffi::TensorView out,
      int64_t quant_level,
      bool cast_bf2half) {
    using namespace host;
    RuntimeCheck(
        (quant_level == kInt4 || quant_level == kInt8) && cast_bf2half,
        "the JIT quick all-reduce runs INT4 or INT8 on bf16 cast to fp16 only; unset SGLANG_ROCM_QUICK_ALL_REDUCE_JIT");
    auto N = SymbolicSize{"num_elements"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).verify(inp).verify(out);
    auto* comms = reinterpret_cast<quickreduce::DeviceComms*>(comm);
    const int64_t n = N.unwrap();
    RuntimeCheck(n * 2 <= comms->kMaxProblemSize, "all-reduce larger than the communication buffer");
    RuntimeCheck((n * 2) % 16 == 0, "quick all-reduce needs a multiple of 16 bytes");
    const auto stream = static_cast<hipStream_t>(LaunchKernel::resolve_device(device.unwrap()));
    const auto* a = static_cast<const half*>(inp.data_ptr());
    auto* b = static_cast<half*>(out.data_ptr());
    if (quant_level == kInt8) {
      comms->allreduce<quickreduce::CodecQ8>(a, b, static_cast<uint32_t>(n), stream);
    } else {
      comms->allreduce<quickreduce::CodecQ4>(a, b, static_cast<uint32_t>(n), stream);
    }
  }

  static void destroy(int64_t comm) {
    delete reinterpret_cast<quickreduce::DeviceComms*>(comm);
  }

  static int64_t handle_bytes() {
    return kHandleBytes;
  }
};

}  // namespace quick_all_reduce_hip
}  // namespace sglang
