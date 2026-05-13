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

template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
SGL_DEVICE void c4_forward(
    const InFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
    const InFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
    const InFloat* kv_src,    // ragged pointer at position = 4n + 3
    OutFloat* kv_out,
    const InFloat* score_bias,
    const bool should_overlap,
    const int32_t buffer_len) {
  using namespace device;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InFloat, kTileElements>;
  /// NOTE: load one tile_dim (< head_dim) at at time
  const auto gmem_in = tile::Memory<StorageIn>::warp();
  StorageIn kv[8];
  StorageIn score[8];
  StorageIn bias[8];

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bias[i] = gmem_in.load(score_bias + i * Trait::kHeadDim);
  }

  if (should_overlap) {
    const auto kv_start = kv_src - 7 * Trait::kElementSize;  // point to start
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      const auto src = i < buffer_len ? kv_buf_0 : kv_start;
      const auto base = src + i * Trait::kElementSize;
      kv[i] = gmem_in.load(base);
      score[i] = gmem_in.load(base + Trait::kScoreOffset);
    }
  } else {
    [[unlikely]];
    constexpr float kFloatNegInf = -FLT_MAX;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      kv[i].fill(cast<InFloat>(0.0f));
      score[i].fill(cast<InFloat>(kFloatNegInf));
    }
  }

  const auto kv_start = kv_src - 3 * Trait::kElementSize;  // point to start
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    const auto src = i + 4 < buffer_len ? kv_buf_1 : kv_start;
    const auto base = src + i * Trait::kElementSize + Trait::kOverlapOffset;
    kv[i + 4] = gmem_in.load(base);
    score[i + 4] = gmem_in.load(base + Trait::kScoreOffset);
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using StorageOut = AlignedVector<OutFloat, kTileElements>;
  const auto gmem_out = tile::Memory<StorageOut>::warp();
  StorageOut result;

  // consume 32 fp registers
  float score_fp32[kTileElements][8];

  // convert to fp32 and apply bias first
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    for (int32_t j = 0; j < 8; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto& score = score_fp32[i];
    float max_value = score[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
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

    result[i] = cast<OutFloat>(sum_product / sum_exp_value);
  }

  // overlap the store with the next iteration's load
  PDLTriggerSecondary<kUsePDL>();
  gmem_out.store(kv_out, result);
}

template <typename Trait, typename InFloat>
SGL_DEVICE void c4_write_decode(InFloat* kv_buf, const InFloat* kv_src) {
  using namespace device;

  using StorageIn = AlignedVector<InFloat, kTileElements>;
  const auto gmem = tile::Memory<StorageIn>::warp();

  StorageIn data[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem.load(kv_src + Trait::kHeadDim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    gmem.store(kv_buf + Trait::kHeadDim * i, data[i]);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
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
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait>(kv_dst, kv_src);
  if (plan.seq_len % 4 == 0) {
    const auto need_overlap = plan.seq_len > 4;
    c4_forward<Trait, kUsePDL>(kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, 8);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
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
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;
  if (plan.is_invalid()) return;

  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // Compact output: one row per compress plan, indexed by `global_pid`.
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const bool need_overlap = plan.seq_len > 4;
  PDLWaitPrimary<kUsePDL>();
  c4_forward<Trait, kUsePDL>(kv_buf_0, kv_buf_1, kv_src, kv_out, score_bias, need_overlap, plan.buffer_len);
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;
  using Trait = C4Trait<kHeadDim>;
  using StorageIn = AlignedVector<InFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 4` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 4);
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
  StorageIn data[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem.load(kv_src, i);
  }
  PDLTriggerSecondary<kUsePDL>();
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    gmem.store(kv_buf, data[i], i);
  }
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress4Kernel {
  static constexpr auto decode_kernel = flash_c4_decode<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c4_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c4_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
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
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
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
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InFloat>()
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
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
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
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InFloat>()
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
