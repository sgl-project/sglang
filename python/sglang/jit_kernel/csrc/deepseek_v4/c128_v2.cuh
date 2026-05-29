/**
 * \brief Here's some dimension info for the main buffer used in C128 prefill and decode.
 *
 * kv_buffer: [num_indices, 128, head_dim * 2]
 * - last dimension layout: | kv | score |
 * kv_input: [batch_size, head_dim * 2]
 * kv_output: [batch_size, head_dim]
 * score_bias (ape): [128, head_dim]
 * plan_c/plan_w: [variable length]
 *
 * For prefill, batch_size = num_q_tokens
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;
using PlanW = device::compress::WritePlan;

/// \brief Block-level configuration shared by all C128 kernels (independent of head_dim).
struct C128Config {
  /// \brief Each thread loads/stores this many elements (split along head_dim).
  static constexpr int32_t kTileElements = 2;
  /// \brief Each warp handles this many elements (split along the softmax dim of 128).
  static constexpr int32_t kElementsPerWarp = 8;
  static constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
  static constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;
  /// \brief Block size used by the prefill write kernel (one warp per write plan tile).
  static constexpr uint32_t kWriteBlockSize = 128;
  static constexpr uint32_t kNumWriteWarps = kWriteBlockSize / device::kWarpThreads;
  /// \brief Per-warp scratch buffer used to stage partial softmax results before the
  /// final block-level reduction. Padded to avoid bank conflicts.
  using SharedStorage = device::AlignedVector<float, kTileElements>;
  using SharedBuffer = SharedStorage[kNumWarps][device::kWarpThreads];
};

template <int64_t kHeadDim_>
struct C128Trait : public C128Config {
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr int64_t kHeadDim = kHeadDim_;
  static constexpr int64_t kScoreOffset = kHeadDim;
  static constexpr int64_t kElementSize = kHeadDim * 2;
  static constexpr int64_t kPageElementSize = 128 * kElementSize;  // page size = 128
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

/// \brief Need to reduce register usage to increase occupancy
#define C128_KERNEL __global__ __launch_bounds__(C128Config::kBlockSize, 2)
#define WRITE_KERNEL __global__ __launch_bounds__(C128Config::kWriteBlockSize, 16)

struct Compress128DecodeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanD* __restrict__ plan_d;
  uint32_t batch_size;
};

struct Compress128PrefillParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanC* __restrict__ plan_c;
  const PlanW* __restrict__ plan_w;
  uint32_t num_compress;
  uint32_t num_write;
};

template <typename Trait, bool kUsePDL, typename BufferFloat, typename InputFloat, typename OutFloat>
SGL_DEVICE void c128_forward(
    const BufferFloat* kv_buf,  // [128n, 128n + 127]
    const InputFloat* kv_src,   // ragged pointer at position = 128n + 127
    OutFloat* kv_out,
    const InputFloat* score_bias,
    const int32_t buffer_len) {
  using namespace device;

  constexpr uint32_t kTileElements = Trait::kTileElements;
  constexpr uint32_t kElementsPerWarp = Trait::kElementsPerWarp;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InputFloat, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];
  StorageIn score[kElementsPerWarp];
  StorageIn bias[kElementsPerWarp];
  const int32_t warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_in.load(score_bias + j * Trait::kHeadDim);
  }

  const auto kv_start = kv_src - 127 * Trait::kElementSize;  // point to start

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
#pragma unroll
    for (int32_t i = 0; i < kElementsPerWarp; ++i) {
      const int32_t j = i + warp_offset;
      __builtin_assume(j < 128);
      const auto src = j < buffer_len ? kv_buf : kv_start;
      kv[i] = gmem_in.load(src + j * Trait::kElementSize);
      score[i] = gmem_in.load(src + j * Trait::kElementSize + Trait::kScoreOffset);
    }
  } else {  // mixed dtype
    using StorageBuffer = AlignedVector<BufferFloat, Trait::kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>{lane_id, kWarpThreads};

#pragma unroll
    for (int32_t i = 0; i < kElementsPerWarp; ++i) {
      const int32_t j = i + warp_offset;
      __builtin_assume(j < 128);
      if (j < buffer_len) {
        const auto src = kv_buf + j * Trait::kElementSize;
        const auto kv_tmp = gmem_buffer.load(src);
        const auto score_tmp = gmem_buffer.load(src + Trait::kScoreOffset);
#pragma unroll
        for (int32_t k = 0; k < kTileElements; ++k) {
          kv[i][k] = cast<InputFloat>(kv_tmp[k]);
          score[i][k] = cast<InputFloat>(score_tmp[k]);
        }
      } else {
        const auto src = kv_start + j * Trait::kElementSize;
        kv[i] = gmem_in.load(src);
        score[i] = gmem_in.load(src + Trait::kScoreOffset);
      }
    }
  }

  /// NOTE: part 2: per-warp partial softmax (online softmax stats + weighted sum)
  using SharedBuffer = typename Trait::SharedBuffer;
  using TmpStorage = typename Trait::SharedStorage;
  __shared__ SharedBuffer s_local_val_max;
  __shared__ SharedBuffer s_local_exp_sum;
  __shared__ SharedBuffer s_local_product;

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

  float score_fp32[kTileElements][kElementsPerWarp];

  // convert to fp32 and apply bias first
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto& score = score_fp32[i];
    float max_value = score[0];
#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      max_value = fmaxf(max_value, score[j]);
    }

    float sum_exp_value = 0.0f;
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      const auto exp_score = expf(score[j] - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // naturally aligned, so no bank conflict
  s_local_val_max[warp_id][lane_id] = tmp_val_max;
  s_local_exp_sum[warp_id][lane_id] = tmp_exp_sum;
  s_local_product[warp_id][lane_id] = tmp_product;

  __syncthreads();

  PDLTriggerSecondary<kUsePDL>();

  /// NOTE: part 3: final reduction + write-back.
  /// Only the first `kTileElements` warps participate; each thread reduces over
  /// `kNumWarps` partial values entirely in registers and writes one output element.
  /// The remaining warps exit early, freeing issue slots and avoiding redundant writes.
  if (warp_id < kTileElements) {
    const uint32_t tx = threadIdx.x;
    const uint32_t local_lane_id = tx / kTileElements;  // [0, kWarpThreads)
    const uint32_t local_tile_id = tx % kTileElements;  // [0, kTileElements)

    float local_val_max[kNumWarps];
    float local_exp_sum[kNumWarps];
    float local_product[kNumWarps];
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      local_val_max[i] = s_local_val_max[i][local_lane_id][local_tile_id];
      local_exp_sum[i] = s_local_exp_sum[i][local_lane_id][local_tile_id];
      local_product[i] = s_local_product[i][local_lane_id][local_tile_id];
    }

    float global_max = local_val_max[0];
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      global_max = fmaxf(global_max, local_val_max[i]);
    }

    float global_exp_sum = 0.0f;
    float global_product = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      const auto exp_val = expf(local_val_max[i] - global_max);
      global_exp_sum += local_exp_sum[i] * exp_val;
      global_product += local_product[i] * exp_val;
    }
    kv_out[tx] = cast<OutFloat>(global_product / global_exp_sum);
  }
}

template <typename Trait, typename BufferFloat, typename InputFloat>
SGL_DEVICE void c128_write_decode(BufferFloat* kv_buf, const InputFloat* kv_src) {
  using namespace device;

  using StorageInput = AlignedVector<InputFloat, Trait::kTileElements>;
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  StorageInput data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem_input.load(kv_src + Trait::kHeadDim * i);
  }

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
      gmem_input.store(kv_buf + Trait::kHeadDim * i, data[i]);
    }
  } else {
    using StorageBuffer = AlignedVector<BufferFloat, Trait::kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();

    StorageBuffer data_cast[2];
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
      for (int32_t j = 0; j < Trait::kTileElements; ++j) {
        data_cast[i][j] = cast<BufferFloat>(data[i][j]);
      }
      gmem_buffer.store(kv_buf + Trait::kHeadDim * i, data_cast[i]);
    }
  }
}

/// \brief Need to reduce register usage to increase occupancy.
template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_decode(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t global_bid = blockIdx.x / Trait::kNumSplit;  // batch id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_bid >= params.batch_size) return;

  const auto plan = params.plan_d[global_bid];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  // the write warp must match the load warp in the following `c128_forward`
  if (warp_id == Trait::kNumWarps - 1) {
    c128_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);
  }
  if (plan.write_loc % 128 == 127) {
    c128_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(kv_buf, kv_src, kv_out, score_bias, 128);
  }
}

// compress kernel
template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t global_pid = blockIdx.x / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_pid >= params.num_compress) return;

  const auto plan = params.plan_c[global_pid];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;
  if (plan.is_invalid()) return;

  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // Compact output: one row per compress plan, indexed by `global_pid`.
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  PDLWaitPrimary<kUsePDL>();
  c128_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(kv_buf, kv_src, kv_out, score_bias, plan.buffer_len);
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;
  using StorageInput = AlignedVector<InputFloat, Trait::kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 2` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 2);
  if (global_pid >= params.num_write) return;

  const auto plan = params.plan_w[global_pid];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  if (plan.is_invalid()) return;

  // each warp will handle a contiguous region
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  PDLWaitPrimary<kUsePDL>();
  StorageInput data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem_input.load(kv_src, i);
  }

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
      gmem_input.store(kv_buf, data[i], i);
    }
  } else {
    using StorageBuffer = AlignedVector<BufferFloat, Trait::kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();

    StorageBuffer data_cast[2];
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
      for (int32_t j = 0; j < Trait::kTileElements; ++j) {
        data_cast[i][j] = cast<BufferFloat>(data[i][j]);
      }
    }
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
      gmem_buffer.store(kv_buf, data_cast[i], i);
    }
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress128Kernel {
  using Trait = C128Trait<kHeadDim>;
  static constexpr auto decode_kernel = flash_c128_decode<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c128_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c128_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto params = Compress128DecodeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .batch_size = batch_size,
    };
    const uint32_t num_blocks = batch_size * Trait::kNumSplit;
    LaunchKernel(num_blocks, Trait::kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input (ragged)
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({C, kHeadDim})  // kv compressed output (compact)
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    const auto params = Compress128PrefillParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,
        .plan_w = plan_w,
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");
    if (const auto num_c_blocks = num_c * Trait::kNumSplit) {
      LaunchKernel(num_c_blocks, Trait::kBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w * Trait::kNumSplit, Trait::kNumWriteWarps)) {
      LaunchKernel(num_w_blocks, Trait::kWriteBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace
