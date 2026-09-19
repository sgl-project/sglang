#include <sgl_kernel/eplb_async_signal_runtime.h>

namespace sglang::eplb {

namespace {

__device__ __forceinline__ uint64_t load_signal(const EplbAsyncSignal* signal) {
  return *reinterpret_cast<const volatile uint64_t*>(&signal->step_and_owner);
}

__global__ void wait_signal_for_gpu_stage_kernel(
    EplbAsyncSignal* signal, int* enabled, int64_t* h2d_done_flag, int64_t expected_h2d_generation) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int enabled_value = 0;
  while (true) {
    const uint64_t signal_value = load_signal(signal);
    if (is_signal_disabled(signal_value)) {
      enabled_value = 0;
      break;
    }
    if (decode_signal_owner(signal_value) == kSignalOwnerGpu) {
      // Signal owner is GPU. Now check if H2D has completed.
      // h2d_done_flag is mapped pinned memory written by CPU after
      // cudaEventSynchronize guarantees H2D is done.  No kernel on
      // copy_stream_ is needed.
      volatile int64_t* done_ptr = reinterpret_cast<volatile int64_t*>(h2d_done_flag);
      if (*done_ptr >= expected_h2d_generation) {
        // Both signal ready AND H2D complete — safe to use new weights.
        enabled_value = !should_skip_signal_step(signal_value);
        break;
      }
      // H2D not yet complete, keep spinning.
    }
    // owner==CPU: worker hasn't written signal yet, keep spinning.
  }
  *enabled = enabled_value;
}

__global__ void set_signal_for_cpu_stage_kernel(EplbAsyncSignal* signal) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const uint64_t signal_value = load_signal(signal);
  if (is_signal_disabled(signal_value)) {
    return;
  }
  *reinterpret_cast<volatile uint64_t*>(&signal->step_and_owner) = signal_value | kSignalOwnerCpu;
  __threadfence_system();
}

}  // namespace

void wait_signal_for_gpu_stage_device(
    EplbAsyncSignal* signal,
    int* enabled,
    int64_t* h2d_done_flag,
    int64_t expected_h2d_generation,
    cudaStream_t stream) {
  wait_signal_for_gpu_stage_kernel<<<1, 1, 0, stream>>>(signal, enabled, h2d_done_flag, expected_h2d_generation);
}

void set_signal_for_cpu_stage_device(EplbAsyncSignal* signal, cudaStream_t stream) {
  set_signal_for_cpu_stage_kernel<<<1, 1, 0, stream>>>(signal);
}

}  // namespace sglang::eplb
