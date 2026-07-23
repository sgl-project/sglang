#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace {

using Plan = device::compress::PrefillPlan;

/// \brief common block size for memory-bound kernel
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct FusedNormRopeParams {
  void* __restrict__ input;
  const void* __restrict__ weight;
  float eps;
  uint32_t num_works;
  const void* __restrict__ handle;
  const float* __restrict__ freqs_cis;
  uint32_t compress_ratio;
};

enum class ForwardMode {
  CompressExtend = 0,
  CompressDecode = 1,
  DefaultForward = 2,
};

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, ForwardMode kMode, bool kUsePDL>
__global__ void fused_norm_rope(const __grid_constant__ FusedNormRopeParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kMaxVecSize = 16 / sizeof(DType);
  constexpr int64_t kVecSize = std::min(kMaxVecSize, kHeadDim / kWarpThreads);
  constexpr int64_t kLocalSize = kHeadDim / (kWarpThreads * kVecSize);
  constexpr int64_t kRopeVecSize = kRopeDim / (kWarpThreads * 2);
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  static_assert(kLocalSize * kVecSize * kWarpThreads == kHeadDim);
  static_assert(kRopeDim % (kWarpThreads * 2) == 0);
  static_assert(kRopeDim % (kVecSize * kLocalSize) == 0);
  static_assert(kRopeSize <= kWarpThreads);
  static_assert(kRopeVecSize == 1, "only support rope dim = 64");

  const auto& [
    _input, _weight, eps, num_works, // norm
    handle, freqs_cis, compress_ratio // rope
  ] = params;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;

  if (work_id >= num_works) return;

  DType* input;
  int32_t position;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const Plan*>(handle)[work_id];
    input = static_cast<DType*>(_input) + plan.ragged_id * kHeadDim;
    position = plan.position + 1 - compress_ratio;
    if (plan.ragged_id == 0xFFFFFFFF) [[unlikely]]
      return;
  } else if constexpr (kMode == CompressDecode) {
    input = static_cast<DType*>(_input) + work_id * kHeadDim;
    const auto seq_len = static_cast<const int32_t*>(handle)[work_id];
    if (seq_len % compress_ratio != 0) return;
    position = seq_len - compress_ratio;
  } else if constexpr (kMode == DefaultForward) {
    input = static_cast<DType*>(_input) + work_id * kHeadDim;
    position = static_cast<const int64_t*>(handle)[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }

  using Storage = AlignedVector<DType, kVecSize>;
  __shared__ Storage s_rope_input[kNumWarps][kRopeSize];

  // prefetch freq
  const auto mem_freq = tile::Memory<fp32x2_t>::warp();
  const auto freq = mem_freq.load(freqs_cis + position * kRopeDim);

  PDLWaitPrimary<kUsePDL>();

  // part 1: norm
  {
    const auto gmem = tile::Memory<Storage>::warp();
    Storage input_vec[kLocalSize];
    Storage weight_vec[kLocalSize];
#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      input_vec[i] = gmem.load(input, i);
    }

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      weight_vec[i] = gmem.load(_weight, i);
    }

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const auto fp32_input = cast<float>(input_vec[i][j]);
        sum_of_squares += fp32_input * fp32_input;
      }
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + eps);

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const auto fp32_input = cast<float>(input_vec[i][j]);
        const auto fp32_weight = cast<float>(weight_vec[i][j]);
        input_vec[i][j] = cast<DType>(fp32_input * norm_factor * fp32_weight);
      }
    }

    const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      if (i == kLocalSize - 1 && is_rope_lane) {
        const auto rope_id = lane_id - (kWarpThreads - kRopeSize);
        s_rope_input[warp_id][rope_id] = input_vec[i];
      } else {
        gmem.store(input, input_vec[i], i);
      }
    }

    __syncwarp();
  }

  // part 2: rope
  {
    // mem elem = DType x 2
    using DTypex2_t = packed_t<DType>;
    const auto mem_elem = tile::Memory<DTypex2_t>::warp();
    const auto elem = mem_elem.load(s_rope_input[warp_id]);
    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);
    const auto [freq_real, freq_imag] = freq;
    const fp32x2_t output = {
        x_real * freq_real - x_imag * freq_imag,
        x_real * freq_imag + x_imag * freq_real,
    };
    mem_elem.store(input + (kHeadDim - kRopeDim), cast<DTypex2_t>(output));
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct FusedNormRopeKernel {
  template <ForwardMode kMode>
  static constexpr auto fused_kernel = fused_norm_rope<DType, kHeadDim, kRopeDim, kMode, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView handle,
      const tvm::ffi::TensorView freqs_cis,
      int32_t _mode,
      float eps,
      uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    const auto mode = static_cast<ForwardMode>(_mode);

    auto B = SymbolicSize{"num_q_tokens"};
    auto N = SymbolicSize{"num_compress_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  // input
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({kHeadDim})  // weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({-1, kRopeDim})  // freqs_cis
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    switch (mode) {
      case CompressExtend:
        TensorMatcher({N, compress::kPrefillPlanDim})  // plan
            .with_dtype<compress::PrefillPlanTensorDtype>()
            .with_device(device_)
            .verify(handle);
        RuntimeCheck(compress_ratio > 0);
        break;
      case CompressDecode:
        TensorMatcher({N})  // seq_len
            .with_dtype<int32_t>()
            .with_device(device_)
            .verify(handle);
        RuntimeCheck(compress_ratio > 0);
        break;
      case DefaultForward:
        TensorMatcher({N})  // position
            .with_dtype<int64_t>()
            .with_device(device_)
            .verify(handle);
        RuntimeCheck(compress_ratio == 0);
        break;
      default:
        Panic("unsupported forward mode: ", static_cast<int>(mode));
    }

    // launch kernel
    const auto num_compress_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_compress_tokens == 0) return;
    const auto params = FusedNormRopeParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .eps = eps,
        .num_works = num_compress_tokens,
        .handle = handle.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .compress_ratio = compress_ratio,
    };
    const auto num_blocks = div_ceil(num_compress_tokens, kNumWarps);
    using KernelType = std::decay_t<decltype(fused_norm_rope<DType, kHeadDim, kRopeDim, CompressExtend, kUsePDL>)>;
    static constexpr KernelType kernel_table[3] = {
        [static_cast<int>(CompressExtend)] = fused_kernel<CompressExtend>,
        [static_cast<int>(CompressDecode)] = fused_kernel<CompressDecode>,
        [static_cast<int>(DefaultForward)] = fused_kernel<DefaultForward>,
    };
    const auto kernel = kernel_table[static_cast<int>(mode)];
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
