#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

// ----------------------------------------------------------------
// Kernel: resolve future token IDs in-place using vectorized 128-bit
// loads/stores with branchless per-element logic.
//
// Semantics: for each element in input_ids,
//   if val < 0: input_ids[i] = future_map[clamp(-val, 0)]
//   else:       input_ids[i] = val  (unchanged)
//
// T     = int32_t | int64_t
// kVecN = number of elements per 128-bit vector (4 for int32, 2 for int64)
// ----------------------------------------------------------------
template <typename T, int kVecN>
__global__ void
resolve_future_token_ids_kernel(T* __restrict__ input_ids, const T* __restrict__ future_map, uint32_t n_total) {
  using vec_t = device::AlignedVector<T, kVecN>;
  const uint32_t n_vecs = n_total / kVecN;

  // --- vectorised body ---
  const uint32_t vec_stride = blockDim.x * gridDim.x;
  for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vecs; vi += vec_stride) {
    vec_t v;
    v.load(input_ids, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      T val = v[i];
      T key = -val;
      if (key < 0) key = 0;  // clamp for signed overflow
      // Branchless: always gather, select via predicate (compiles to SELP)
      v[i] = (val < 0) ? future_map[key] : val;
    }
    v.store(input_ids, vi);
  }

  // --- scalar tail ---
  const uint32_t base = n_vecs * kVecN;
  const uint32_t scalar_stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; base + i < n_total; i += scalar_stride) {
    T val = input_ids[base + i];
    T key = -val;
    if (key < 0) key = 0;
    input_ids[base + i] = (val < 0) ? future_map[key] : val;
  }
}

// ----------------------------------------------------------------
// Launcher: validates tensors, selects vector width, launches kernel
// ----------------------------------------------------------------
template <typename T>
struct ResolveFutureTokenIds {
  static void run(tvm::ffi::TensorView input_ids, tvm::ffi::TensorView future_map) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize M = {"map_size"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    TensorMatcher({N})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(input_ids);

    TensorMatcher({M})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(future_map);

    const uint32_t num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;

    const DLDevice device = device_.unwrap();

    // 128-bit vector width: int32 -> 4 elements, int64 -> 2 elements
    constexpr int kVecN = 16 / sizeof(T);
    const uint32_t n_work_items = div_ceil(num_tokens, static_cast<uint32_t>(kVecN));

    constexpr uint32_t kBlockSize = 256;
    const uint32_t grid = div_ceil(n_work_items, kBlockSize);

    LaunchKernel(grid, kBlockSize, device)(
        resolve_future_token_ids_kernel<T, kVecN>,
        static_cast<T*>(input_ids.data_ptr()),
        static_cast<const T*>(future_map.data_ptr()),
        num_tokens);
  }
};

}  // namespace
