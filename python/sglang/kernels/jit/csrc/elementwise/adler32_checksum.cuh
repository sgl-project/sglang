// Adler-32 checksum CUDA kernels for tensor verification.
//
// Two modes:
//   1. Whole-tensor: checksum all bytes of a contiguous tensor
//   2. Strided: checksum selected items across multiple tensors,
//      given precomputed byte pointers and per-item lengths
//
// Adler-32:
//   A = 1 + sum(bytes)              (mod 65521)
//   B = sum of running A values     (mod 65521)
//   checksum = (B << 16) | A
//
// Combine (left || right):
//   A = A_left + A_right - 1             (mod 65521)
//   B = B_left + B_right + len_right * (A_left - 1)  (mod 65521)

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// ROCm's masked __shfl_*_sync traps on partial wavefronts (fewer than 64 lanes
// launched), so use the maskless intrinsic on ROCm.
#ifdef USE_ROCM
#define ADLER32_SHFL_DOWN(mask, var, delta) __shfl_down((var), (delta))
#else
#define ADLER32_SHFL_DOWN(mask, var, delta) __shfl_down_sync((mask), (var), (delta))
#endif

namespace sglang {

namespace {

constexpr uint32_t MOD_ADLER = 65521;
constexpr size_t kBlockSize = 256;

struct Adler32State {
  uint32_t a;
  uint32_t b;
  uint64_t len;
};

__device__ __forceinline__ Adler32State adler32_identity() {
  return {1, 0, 0};
}

__device__ Adler32State adler32_compute(const uint8_t* ptr, uint64_t nbytes) {
  uint64_t sum = 0;
  uint64_t weighted = 0;
  uint64_t offset = 0;
  while (offset < nbytes && (reinterpret_cast<uintptr_t>(ptr + offset) & 15) != 0) {
    uint64_t value = ptr[offset];
    sum += value;
    weighted += (nbytes - offset) * value;
    ++offset;
  }
  const uint4* vectors = reinterpret_cast<const uint4*>(ptr + offset);
  uint64_t num_vectors = (nbytes - offset) / 16;
  for (uint64_t i = 0; i < num_vectors; ++i) {
    uint4 value = vectors[i];
    uint32_t words[4] = {value.x, value.y, value.z, value.w};
    uint64_t word_sums[4];
    uint64_t within = 0;
#pragma unroll
    for (uint32_t word = 0; word < 4; ++word) {
      uint32_t value_word = words[word];
      uint64_t x0 = value_word & 0xffu;
      uint64_t x1 = (value_word >> 8) & 0xffu;
      uint64_t x2 = (value_word >> 16) & 0xffu;
      uint64_t x3 = value_word >> 24;
      word_sums[word] = x0 + x1 + x2 + x3;
      within += 3 * x0 + 2 * x1 + x2;
    }
    uint64_t vector_sum = word_sums[0] + word_sums[1] + word_sums[2] + word_sums[3];
    within += 12 * word_sums[0] + 8 * word_sums[1] + 4 * word_sums[2];
    uint64_t base = offset + i * 16;
    sum += vector_sum;
    weighted += (nbytes - base - 15) * vector_sum + within;
    if ((i & 4095) == 4095) {
      sum %= MOD_ADLER;
      weighted %= MOD_ADLER;
    }
  }
  offset += num_vectors * 16;
  while (offset < nbytes) {
    uint64_t value = ptr[offset];
    sum += value;
    weighted += (nbytes - offset) * value;
    ++offset;
  }
  return {
      static_cast<uint32_t>((1 + sum) % MOD_ADLER),
      static_cast<uint32_t>((nbytes % MOD_ADLER + weighted) % MOD_ADLER),
      nbytes};
}

__device__ __forceinline__ Adler32State adler32_combine(Adler32State left, Adler32State right) {
  uint64_t a = ((uint64_t)left.a + (uint64_t)right.a - 1 + MOD_ADLER) % MOD_ADLER;
  uint64_t len_mod = (uint64_t)right.len % MOD_ADLER;
  uint64_t a_minus_1 = ((uint64_t)left.a - 1 + MOD_ADLER) % MOD_ADLER;
  uint64_t b = ((uint64_t)left.b + (uint64_t)right.b + len_mod * a_minus_1) % MOD_ADLER;
  return {(uint32_t)a, (uint32_t)b, left.len + right.len};
}

__device__ Adler32State adler32_block_reduce(Adler32State state, Adler32State* smem, uint32_t num_valid) {
#ifdef USE_ROCM
  // ROCm wavefront is 64 lanes; __shfl_*_sync requires a 64-bit mask.
  constexpr uint64_t kFullMask = 0xffffffffffffffffULL;
#else
  constexpr uint32_t kFullMask = 0xffffffffu;
#endif
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t warp_id = threadIdx.x / warpSize;
  uint32_t num_warps = (num_valid + warpSize - 1) / warpSize;

  // Phase 1: intra-warp ordered reduction via __shfl_down_sync
  uint32_t valid_in_warp = warpSize;
  if (warp_id == num_warps - 1) {
    uint32_t remainder = num_valid % warpSize;
    if (remainder != 0) valid_in_warp = remainder;
  }

#pragma unroll
  for (uint32_t delta = 1; delta < warpSize; delta *= 2) {
    Adler32State right;
    right.a = ADLER32_SHFL_DOWN(kFullMask, state.a, delta);
    right.b = ADLER32_SHFL_DOWN(kFullMask, state.b, delta);
    uint32_t len_lo = ADLER32_SHFL_DOWN(kFullMask, (uint32_t)state.len, delta);
    uint32_t len_hi = ADLER32_SHFL_DOWN(kFullMask, (uint32_t)(state.len >> 32), delta);
    right.len = (uint64_t)len_lo | ((uint64_t)len_hi << 32);
    if (lane + delta < valid_in_warp) {
      state = adler32_combine(state, right);
    }
  }

  // Lane 0 of each warp writes result to shared memory
  if (lane == 0) {
    smem[warp_id] = state;
  }
  __syncthreads();

  // Phase 2: warp 0 reduces across warp results
  if (warp_id == 0) {
    state = (lane < num_warps) ? smem[lane] : adler32_identity();
#pragma unroll
    for (uint32_t delta = 1; delta < warpSize; delta *= 2) {
      Adler32State right;
      right.a = ADLER32_SHFL_DOWN(kFullMask, state.a, delta);
      right.b = ADLER32_SHFL_DOWN(kFullMask, state.b, delta);
      uint32_t len_lo = ADLER32_SHFL_DOWN(kFullMask, (uint32_t)state.len, delta);
      uint32_t len_hi = ADLER32_SHFL_DOWN(kFullMask, (uint32_t)(state.len >> 32), delta);
      right.len = (uint64_t)len_lo | ((uint64_t)len_hi << 32);
      if (lane + delta < num_warps) {
        state = adler32_combine(state, right);
      }
    }
  }

  return state;
}

// --- Kernel 1: Whole tensor ---

__global__ void adler32_whole_kernel(
    const uint8_t* __restrict__ data,
    uint32_t* __restrict__ out_a,
    uint32_t* __restrict__ out_b,
    uint64_t* __restrict__ out_len,
    uint64_t num_bytes,
    uint64_t chunk_size) {
  extern __shared__ Adler32State smem[];

  uint64_t global_tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = global_tid * chunk_size;

  Adler32State state;
  if (start < num_bytes) {
    uint64_t end = start + chunk_size;
    if (end > num_bytes) end = num_bytes;
    state = adler32_compute(data + start, end - start);
  } else {
    state = adler32_identity();
  }

  uint32_t threads_in_block = blockDim.x;
  uint64_t block_start = (uint64_t)blockIdx.x * blockDim.x;
  uint64_t total_threads = (num_bytes + chunk_size - 1) / chunk_size;
  if (block_start + threads_in_block > total_threads) {
    threads_in_block = (uint32_t)(total_threads - block_start);
  }

  Adler32State result = adler32_block_reduce(state, smem, threads_in_block);

  if (threadIdx.x == 0) {
    out_a[blockIdx.x] = result.a;
    out_b[blockIdx.x] = result.b;
    out_len[blockIdx.x] = result.len;
  }
}

struct Adler32WholeKernel {
  static void run(tvm::ffi::TensorView data, tvm::ffi::TensorView states, int64_t num_bytes, int64_t chunk_size) {
    using namespace host;

    const uint8_t* data_ptr = static_cast<const uint8_t*>(data.data_ptr());

    int64_t total_threads = (num_bytes + chunk_size - 1) / chunk_size;
    int64_t num_blocks = (total_threads + kBlockSize - 1) / kBlockSize;

    uint32_t* out_a = static_cast<uint32_t*>(states.data_ptr());
    uint32_t* out_b = out_a + num_blocks;
    uint64_t* out_len = reinterpret_cast<uint64_t*>(out_b + num_blocks);

    DLDevice device;
    device.device_type = kDLCUDA;
    device.device_id = data.device().device_id;

    size_t smem_bytes = (kBlockSize / 32) * sizeof(Adler32State);
    LaunchKernel(num_blocks, kBlockSize, device, smem_bytes)(
        adler32_whole_kernel,
        data_ptr,
        out_a,
        out_b,
        out_len,
        static_cast<uint64_t>(num_bytes),
        static_cast<uint64_t>(chunk_size));
  }
};

// --- Kernel 2: Strided items ---

__global__ void adler32_strided_kernel(
    const int64_t* __restrict__ ptrs,
    const int32_t* __restrict__ lens,
    uint32_t* __restrict__ out_a,
    uint32_t* __restrict__ out_b,
    uint64_t* __restrict__ out_len,
    int64_t num_items) {
  extern __shared__ Adler32State smem[];

  int64_t global_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  Adler32State state;
  if (global_tid < num_items) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(ptrs[global_tid]);
    uint32_t nbytes = static_cast<uint32_t>(lens[global_tid]);
    state = adler32_compute(ptr, nbytes);
  } else {
    state = adler32_identity();
  }

  uint32_t threads_in_block = blockDim.x;
  int64_t block_start = (int64_t)blockIdx.x * blockDim.x;
  if (block_start + threads_in_block > num_items) {
    threads_in_block = (uint32_t)(num_items - block_start);
  }

  Adler32State result = adler32_block_reduce(state, smem, threads_in_block);

  if (threadIdx.x == 0) {
    out_a[blockIdx.x] = result.a;
    out_b[blockIdx.x] = result.b;
    out_len[blockIdx.x] = result.len;
  }
}

struct Adler32StridedKernel {
  static void
  run(tvm::ffi::TensorView ptrs, tvm::ffi::TensorView lens, tvm::ffi::TensorView states, int64_t num_items) {
    using namespace host;

    const int64_t* ptrs_ptr = static_cast<const int64_t*>(ptrs.data_ptr());
    const int32_t* lens_ptr = static_cast<const int32_t*>(lens.data_ptr());

    int64_t num_blocks = (num_items + kBlockSize - 1) / kBlockSize;

    uint32_t* out_a = static_cast<uint32_t*>(states.data_ptr());
    uint32_t* out_b = out_a + num_blocks;
    uint64_t* out_len = reinterpret_cast<uint64_t*>(out_b + num_blocks);

    DLDevice device;
    device.device_type = kDLCUDA;
    device.device_id = ptrs.device().device_id;

    size_t smem_bytes = (kBlockSize / 32) * sizeof(Adler32State);
    LaunchKernel(num_blocks, kBlockSize, device, smem_bytes)(
        adler32_strided_kernel, ptrs_ptr, lens_ptr, out_a, out_b, out_len, num_items);
  }
};

// --- Kernel 3: Final reduction ---

__global__ void adler32_reduce_kernel(
    const uint32_t* __restrict__ in_a,
    const uint32_t* __restrict__ in_b,
    const uint64_t* __restrict__ in_len,
    int64_t* __restrict__ output,
    int64_t num_blocks_to_reduce) {
  extern __shared__ Adler32State smem[];

  // Each thread loads one block result; if more blocks than threads,
  // sequentially combine multiple entries first.
  Adler32State state = adler32_identity();
  int64_t items_per_thread = (num_blocks_to_reduce + blockDim.x - 1) / blockDim.x;
  int64_t start = (int64_t)threadIdx.x * items_per_thread;
  int64_t end = start + items_per_thread;
  if (end > num_blocks_to_reduce) end = num_blocks_to_reduce;

  for (int64_t i = start; i < end; ++i) {
    Adler32State s = {in_a[i], in_b[i], in_len[i]};
    state = adler32_combine(state, s);
  }

  uint32_t num_valid = blockDim.x;
  int64_t total_active = (num_blocks_to_reduce + items_per_thread - 1) / items_per_thread;
  if (total_active < num_valid) num_valid = (uint32_t)total_active;

  Adler32State result = adler32_block_reduce(state, smem, num_valid);

  if (threadIdx.x == 0) {
    int64_t checksum = ((int64_t)result.b << 16) | (int64_t)result.a;
    output[0] = checksum;
  }
}

struct Adler32ReduceKernel {
  static void run(tvm::ffi::TensorView states, tvm::ffi::TensorView output, int64_t num_blocks_to_reduce) {
    using namespace host;

    uint32_t* in_a = static_cast<uint32_t*>(states.data_ptr());
    uint32_t* in_b = in_a + num_blocks_to_reduce;
    uint64_t* in_len = reinterpret_cast<uint64_t*>(in_b + num_blocks_to_reduce);

    int64_t* out_ptr = static_cast<int64_t*>(output.data_ptr());

    uint32_t threads = kBlockSize;
    if (num_blocks_to_reduce < (int64_t)threads) threads = (uint32_t)num_blocks_to_reduce;
    // Round up to next power of 2 for reduction
    uint32_t t = 1;
    while (t < threads)
      t *= 2;
    threads = t;
    if (threads > kBlockSize) threads = kBlockSize;

    DLDevice device;
    device.device_type = kDLCUDA;
    device.device_id = states.device().device_id;

    size_t smem_bytes = ((threads + 31) / 32) * sizeof(Adler32State);
    LaunchKernel(1, threads, device, smem_bytes)(
        adler32_reduce_kernel, in_a, in_b, in_len, out_ptr, num_blocks_to_reduce);
  }
};

}  // namespace

}  // namespace sglang
