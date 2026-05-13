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

#include <cstdint>

namespace {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;
using PlanW = device::compress::WritePlan;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int32_t kTileElements = 2;
/// \brief Each warp will handle this many elements (split along 128)
constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;
constexpr uint32_t kWriteBlockSize = 128;  // one warp per write

/// \brief Need to reduce register usage to increase occupancy
#define C128_KERNEL __global__ __launch_bounds__(kBlockSize, 2)
#define WRITE_KERNEL __global__ __launch_bounds__(kWriteBlockSize, 16)

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

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, kTileElements>;
  Storage data[kNumWarps][device::kWarpThreads + 1];  // padding to avoid bank conflict
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) {
    return data[warp_id][lane_id];
  }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};

template <int64_t kHeadDim_>
struct C128Trait {
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr int64_t kHeadDim = kHeadDim_;
  static constexpr int64_t kScoreOffset = kHeadDim;
  static constexpr int64_t kElementSize = kHeadDim * 2;
  static constexpr int64_t kPageElementSize = 128 * kElementSize;  // page size = 128
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
SGL_DEVICE void c128_forward(
    const InFloat* kv_buf,  // [128n, 128n + 127]
    const InFloat* kv_src,  // ragged pointer at position = 128n + 127
    OutFloat* kv_out,
    const InFloat* score_bias,
    const int32_t buffer_len) {
  using namespace device;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InFloat, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];
  StorageIn score[kElementsPerWarp];
  StorageIn bias[kElementsPerWarp];
  const int32_t warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_in.load(score_bias + j * Trait::kHeadDim);
  }

  const auto kv_start = kv_src - 127 * Trait::kElementSize;  // point to start

#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    __builtin_assume(j < 128);
    const auto src = j < buffer_len ? kv_buf : kv_start;
    kv[i] = gmem_in.load(src + j * Trait::kElementSize);
    score[i] = gmem_in.load(src + j * Trait::kElementSize + Trait::kScoreOffset);
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using TmpStorage = typename Compress128SharedBuffer::Storage;
  __shared__ Compress128SharedBuffer s_local_val_max;
  __shared__ Compress128SharedBuffer s_local_exp_sum;
  __shared__ Compress128SharedBuffer s_local_product;

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

  float score_fp32[kTileElements][kElementsPerWarp];

  // convert to fp32 and apply bias first
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto& score = score_fp32[i];
    float max_value = score[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      const auto fp32_score = score[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // naturally aligned, so no bank conflict
  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();

  /// NOTE: part 3: online softmax
  /// NOTE: We have `kTileElements * kWarpThreads * kNumWarps` values to reduce
  /// each reduce will consume `kNumWarps` threads (use partial warp reduction)
  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kBlockSize;

  PDLTriggerSecondary<kUsePDL>();

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    /// NOTE: Range `[0, kTileElements * kWarpThreads * kNumWarps)`
    const uint32_t j = i * kBlockSize + warp_id * kWarpThreads + lane_id;
    /// NOTE: Range `[0, kNumWarps)`
    const uint32_t local_warp_id = j % kNumWarps;
    /// NOTE: Range `[0, kTileElements * kWarpThreads)`
    const uint32_t local_elem_id = j / kNumWarps;
    /// NOTE: Range `[0, kTileElements)`
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    /// NOTE: Range `[0, kWarpThreads)`
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    /// NOTE: each warp will access the whole tile (all `kTileElements`)
    /// and for different lanes, the memory access only differ in `local_warp_id`
    /// so there's no bank conflict in shared memory access.
    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    kv_out[local_elem_id] = cast<OutFloat>(global_product);
  }
}

template <typename Trait, typename InFloat>
SGL_DEVICE void c128_write_decode(InFloat* kv_buf, const InFloat* kv_src) {
  using namespace device;

  using Storage = AlignedVector<InFloat, kTileElements>;
  const auto gmem = tile::Memory<Storage>::warp();

  Storage data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem.load(kv_src + Trait::kHeadDim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_buf + Trait::kHeadDim * i, data[i]);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_decode(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t global_bid = blockIdx.x / Trait::kNumSplit;  // batch id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_bid >= params.batch_size) return;

  const auto plan = params.plan_d[global_bid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  // the write warp must match the load warp in the following `c128_forward`
  if (warp_id == kNumWarps - 1) {
    c128_write_decode<Trait>(kv_dst, kv_src);
  }
  if (plan.write_loc % 128 == 127) {
    c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, 128);
  }
}

// compress kernel
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t global_pid = blockIdx.x / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_pid >= params.num_compress) return;

  const auto plan = params.plan_c[global_pid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;
  if (plan.is_invalid()) return;

  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // Compact output: one row per compress plan, indexed by `global_pid`.
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  PDLWaitPrimary<kUsePDL>();
  c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, plan.buffer_len);
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;
  using StorageIn = AlignedVector<InFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 2` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 2);
  if (global_pid >= params.num_write) return;

  const auto plan = params.plan_w[global_pid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  if (plan.is_invalid()) return;

  // each warp will handle a contiguous region
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem = tile::Memory<StorageIn>::warp();

  PDLWaitPrimary<kUsePDL>();
  StorageIn data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem.load(kv_src, i);
  }
  PDLTriggerSecondary<kUsePDL>();
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_buf, data[i], i);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress128Kernel {
  static constexpr auto decode_kernel = flash_c128_decode<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  using Trait = C128Trait<kHeadDim>;

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InFloat>()
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
    const uint32_t num_blocks = batch_size * kNumSplit;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
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
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input (ragged)
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({C, kHeadDim})  // kv compressed output (compact)
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InFloat>()
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
    if (const auto num_c_blocks = num_c * kNumSplit) {
      constexpr auto kBlockSize_C = kBlockSize;
      LaunchKernel(num_c_blocks, kBlockSize_C, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    constexpr uint32_t kWarpsPerWriteBlock = kWriteBlockSize / device::kWarpThreads;
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerWriteBlock)) {
      constexpr auto kBlockSize_W = kWriteBlockSize;
      LaunchKernel(num_w_blocks, kBlockSize_W, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace
