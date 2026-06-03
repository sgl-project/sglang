/**
 * \brief Here's some dimension info for the main buffer used in C4 prefill and decode.
 *
 * kv_buffer: [num_indices, 8, head_dim * 4]
 * - last dimension layout: | kv overlap | kv | score overlap | score |
 * kv_input: [batch_size, head_dim * 4]
 * kv_output: [batch_size, head_dim]
 * score_bias (ape): [8, head_dim]
 * plan_c/plan_w: [variable length]
 *
 * For prefill, batch_size = num_q_tokens
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

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

namespace {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;
using PlanW = device::compress::WritePlan;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int32_t kTileElements = 4;

/// \brief Need to improve register usage to reduce latency
#define C4_KERNEL __global__ __launch_bounds__(128, 4)
#define WRITE_KERNEL __global__ __launch_bounds__(128, 16)

struct Compress4DecodeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanD* __restrict__ plan_d;
  uint32_t batch_size;
};

struct Compress4PrefillParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  void* __restrict__ kv_output;
  const void* __restrict__ score_bias;
  const PlanC* __restrict__ plan_c;
  const PlanW* __restrict__ plan_w;
  uint32_t num_compress;
  uint32_t num_write;
};

template <int64_t kHeadDim_>
struct C4Trait {
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 128
  static constexpr int64_t kHeadDim = kHeadDim_;
  static constexpr int64_t kOverlapOffset = kHeadDim;
  static constexpr int64_t kScoreOffset = kHeadDim * 2;
  static constexpr int64_t kElementSize = kHeadDim * 4;
  static constexpr int64_t kPageElementSize = 4 * kElementSize;  // page size = 4
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

template <typename Trait, bool kUsePDL, typename BufferFloat, typename InputFloat, typename OutFloat>
SGL_DEVICE void c4_forward(
    const BufferFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
    const BufferFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
    const InputFloat* kv_src,     // ragged pointer at position = 4n + 3
    OutFloat* kv_out,
    const InputFloat* score_bias,
    const bool should_overlap,
    const int32_t buffer_len) {
  using namespace device;

  /// NOTE: part 1: load kv + score
  using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
  using StorageInput = AlignedVector<InputFloat, kTileElements>;
  /// NOTE: load one tile_dim (< head_dim) at at time
  const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();
  const auto gmem_input = tile::Memory<StorageInput>::warp();
  StorageBuffer kv_hist[8];
  StorageBuffer score_hist[8];
  StorageInput kv_live[8];
  StorageInput score_live[8];
  StorageInput bias[8];

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bias[i] = gmem_input.load(score_bias + i * Trait::kHeadDim);
  }

  if (should_overlap) {
    const auto kv_start = kv_src - 7 * Trait::kElementSize;  // point to start
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      if (i < buffer_len) {
        const auto base = kv_buf_0 + i * Trait::kElementSize;
        kv_hist[i] = gmem_buffer.load(base);
        score_hist[i] = gmem_buffer.load(base + Trait::kScoreOffset);
      } else {
        const auto base = kv_start + i * Trait::kElementSize;
        kv_live[i] = gmem_input.load(base);
        score_live[i] = gmem_input.load(base + Trait::kScoreOffset);
      }
    }
  } else {
    [[unlikely]];
    constexpr float kFloatNegInf = -FLT_MAX;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      kv_live[i].fill(cast<InputFloat>(0.0f));
      score_live[i].fill(cast<InputFloat>(kFloatNegInf));
    }
  }

  const auto kv_start = kv_src - 3 * Trait::kElementSize;  // point to start
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    if (i + 4 < buffer_len) {
      const auto base = kv_buf_1 + i * Trait::kElementSize + Trait::kOverlapOffset;
      kv_hist[i + 4] = gmem_buffer.load(base);
      score_hist[i + 4] = gmem_buffer.load(base + Trait::kScoreOffset);
    } else {
      const auto base = kv_start + i * Trait::kElementSize + Trait::kOverlapOffset;
      kv_live[i + 4] = gmem_input.load(base);
      score_live[i + 4] = gmem_input.load(base + Trait::kScoreOffset);
    }
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using StorageOut = AlignedVector<OutFloat, kTileElements>;
  const auto gmem_out = tile::Memory<StorageOut>::warp();
  StorageOut result;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float score_fp32[8];

    float max_value = -INFINITY;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const bool use_hist = j < 4 ? (should_overlap && j < buffer_len) : (j < buffer_len);
      const float score_value =
          use_hist ? cast<float>(score_hist[j][i]) : cast<float>(score_live[j][i]);
      const float score = score_value + cast<float>(bias[j][i]);
      score_fp32[j] = score;
      max_value = fmaxf(max_value, score);
    }

    float sum_exp_value = 0.0f;
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      const auto exp_score = expf(fp32_score - max_value);
      const bool use_hist = j < 4 ? (should_overlap && j < buffer_len) : (j < buffer_len);
      const float kv_value = use_hist ? cast<float>(kv_hist[j][i]) : cast<float>(kv_live[j][i]);
      sum_product += kv_value * exp_score;
      sum_exp_value += exp_score;
    }

    result[i] = cast<OutFloat>(sum_product / sum_exp_value);
  }

  // overlap the store with the next iteration's load
  PDLTriggerSecondary<kUsePDL>();
  gmem_out.store(kv_out, result);
}

template <typename Trait, typename BufferFloat, typename InputFloat>
SGL_DEVICE void c4_write_decode(BufferFloat* kv_buf, const InputFloat* kv_src) {
  using namespace device;

  using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
  using StorageInput = AlignedVector<InputFloat, kTileElements>;
  const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  StorageInput data[4];
  StorageBuffer data_cast[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem_input.load(kv_src + Trait::kHeadDim * i);
#pragma unroll
    for (int32_t j = 0; j < kTileElements; ++j) {
      data_cast[i][j] = cast<BufferFloat>(data[i][j]);
    }
    gmem_buffer.store(kv_buf + Trait::kHeadDim * i, data_cast[i]);
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C4_KERNEL void flash_c4_decode(const __grid_constant__ Compress4DecodeParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_bid = global_wid / Trait::kNumSplit;  // batch id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_bid >= params.batch_size) return;

  const auto plan = params.plan_d[global_bid];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);
  if (plan.seq_len % 4 == 0) {
    const auto need_overlap = plan.seq_len > 4;
    c4_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(
        kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, 8);
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
C4_KERNEL void flash_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
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
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const bool need_overlap = plan.seq_len > 4;
  PDLWaitPrimary<kUsePDL>();
  c4_forward<Trait, kUsePDL, BufferFloat, InputFloat, OutFloat>(
      kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, plan.buffer_len);
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;
  using StorageBuffer = AlignedVector<BufferFloat, kTileElements>;
  using StorageInput = AlignedVector<InputFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 4` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 4);
  if (global_pid >= params.num_write) return;

  const auto plan = params.plan_w[global_pid];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  if (plan.is_invalid()) return;

  // each warp will handle a contiguous region
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  PDLWaitPrimary<kUsePDL>();
  StorageInput data[4];
  StorageBuffer data_cast[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem_input.load(kv_src, i);
#pragma unroll
    for (int32_t j = 0; j < kTileElements; ++j) {
      data_cast[i][j] = cast<BufferFloat>(data[i][j]);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    gmem_buffer.store(kv_buf, data_cast[i], i);
  }
}

template <int64_t kHeadDim, typename BufferFloat, typename InputFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress4Kernel {
  static constexpr auto decode_kernel =
      flash_c4_decode<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel =
      flash_c4_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel =
      write_c4_prefill<kHeadDim, BufferFloat, InputFloat, OutFloat, kUsePDL>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kWarpsPerBlock = kBlockSize / device::kWarpThreads;
  using Trait = C4Trait<kHeadDim>;

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

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
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
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto params = Compress4DecodeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .batch_size = batch_size,
    };
    const uint32_t num_blocks = div_ceil(batch_size * kNumSplit, kWarpsPerBlock);
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
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
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
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    const auto params = Compress4PrefillParams{
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
    if (const auto num_c_blocks = div_ceil(num_c * kNumSplit, kWarpsPerBlock)) {
      LaunchKernel(num_c_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerBlock)) {
      LaunchKernel(num_w_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace
