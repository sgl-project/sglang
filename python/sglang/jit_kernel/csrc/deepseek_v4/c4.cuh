#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <cstdint>

namespace {

using Plan4 = device::compress::PrefillPlan;
using IndiceT = int32_t;

/// \brief Each thread will handle this many elements (split along head_dim)
constexpr int kTileElements = 4;

/// \brief Need to improve register usage to reduce latency
#define C4_KERNEL __global__ __launch_bounds__(128, 4)

enum class PageMode {
  RingBuffer = 8,
  Page4Align = 4,
};

struct alignas(16) C4IndexBundle {
  int32_t load_first_page;
  int32_t load_second_page;
  int32_t write_first_page;
  int32_t last_position;
};

struct Compress4DecodeParams {
  /**
   * \brief Shape: `[num_indices, 8, head_dim * 4]` \n
   * last dimension layout:
   * | kv overlap | kv | score overlap | score |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[batch_size, head_dim * 4]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[batch_size, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[8, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndiceT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]` */
  const IndiceT* __restrict__ seq_lens;
  /** \brief Shape: `[batch_size, 1]` */
  const int32_t* __restrict__ extra;
  /** \NOTE: `batch_size` <= `num_indices` */
  uint32_t batch_size;
};

struct Compress4PrefillParams {
  /**
   * \brief Shape: `[num_indices, 8, head_dim * 4]` \n
   * last dimension layout:
   * | kv overlap | kv | score overlap | score |
   */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[num_q_tokens, head_dim * 4]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[num_q_tokens, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[8, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndiceT* __restrict__ indices;
  /** \brief Shape: `[batch_size, 4]` */
  const C4IndexBundle* __restrict__ extra;
  /** \brief The following part is plan info. */

  const Plan4* __restrict__ compress_plan;
  const Plan4* __restrict__ write_plan;
  uint32_t num_compress;
  uint32_t num_write;
};

template <typename T>
SGL_DEVICE void c4_write(
    T* kv_score_buf,  //
    const T* kv_score_src,
    const int64_t head_dim,
    const int32_t write_pos) {
  using namespace device;

  using Storage = AlignedVector<T, kTileElements>;
  const auto element_size = head_dim * 4;
  const auto gmem = tile::Memory<Storage>::warp();
  kv_score_buf += write_pos * element_size;

  /// NOTE: Layout | [0] = kv overlap | [1] = kv | [2] = score overlap | [3] = score |
  Storage kv_score[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    kv_score[i] = gmem.load(kv_score_src + head_dim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    gmem.store(kv_score_buf + head_dim * i, kv_score[i]);
  }
}

template <bool kPaged, typename InFloat, typename OutFloat>
SGL_DEVICE void c4_forward(
    const InFloat* kv_score_buf,
    const InFloat* kv_score_src,
    OutFloat* kv_out,
    const InFloat* score_bias,
    const int64_t head_dim,
    const int32_t seq_len,
    const int32_t window_len,
    [[maybe_unused]] const InFloat* kv_score_overlap_buf = nullptr) {
  using namespace device;

  const auto element_size = head_dim * 4;
  const auto score_offset = head_dim * 2;
  const auto overlap_stride = head_dim;

  /// NOTE: part 1: load kv + score
  using StorageIn = AlignedVector<InFloat, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>::warp();
  StorageIn kv[8];
  StorageIn score[8];
  StorageIn bias[8];

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bias[i] = gmem_in.load(score_bias + i * head_dim);
  }

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const bool is_overlap = i < 4;
    const InFloat* src;
    if (i < window_len) {
      /// NOTE: `seq_len` must be a multiple of 4 here
      if constexpr (kPaged) {
        const auto kv_score_ptr = is_overlap ? kv_score_overlap_buf : kv_score_buf;
        const int32_t k = i % 4;
        src = kv_score_ptr + k * element_size;
      } else {
        const int32_t k = (seq_len + i) % 8;
        src = kv_score_buf + k * element_size;
      }
    } else {
      /// NOTE: k in [-7, 0]. We'll load from the ragged `kv_score_src`
      const int32_t k = i - 7;
      src = kv_score_src + k * element_size;
    }
    src += (is_overlap ? 0 : overlap_stride);
    kv[i] = gmem_in.load(src);
    score[i] = gmem_in.load(src + score_offset);
  }

  if (seq_len == 4) {
    [[unlikely]];
    constexpr float kFloatNegInf = -1e9f;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      kv[i].fill(cast<InFloat>(0.0f));
      score[i].fill(cast<InFloat>(kFloatNegInf));
    }
  }

  /// NOTE: part 2: safe online softmax + weighted sum
  using StorageOut = AlignedVector<OutFloat, kTileElements>;
  const auto gmem_out = tile::Memory<StorageOut>::warp();
  StorageOut result;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float score_fp32[8];

#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      score_fp32[j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }

    float max_value = score_fp32[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    result[i] = cast<OutFloat>(sum_product / sum_exp_value);
  }

  gmem_out.store(kv_out, result);
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, PageMode kMode, bool kUsePDL>
C4_KERNEL void flash_c4_decode(const __grid_constant__ Compress4DecodeParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;  // 128
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 4;  // `* 4` due to overlap transform + score
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const auto& [
    _kv_score_buffer, _kv_score_input, _kv_compressed_output, _score_bias, // kv score
    indices, seq_lens, extra, batch_size // decode info
  ] = params;
  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;  // warp id
  const uint32_t global_bid = global_wid / kNumSplit;     // batch id
  const uint32_t global_sid = global_wid % kNumSplit;     // split id

  if (global_bid >= batch_size) return;

  const int32_t index = indices[global_bid];
  const int32_t seq_len = seq_lens[global_bid];
  const int64_t split_offset = global_sid * kTileDim;

  // kv score
  const auto kv_score_buffer = static_cast<InFloat*>(_kv_score_buffer);

  // kv input
  const auto kv_score_input = static_cast<const InFloat*>(_kv_score_input);
  const auto kv_src = kv_score_input + global_bid * kElementSize + split_offset;

  // kv output
  const auto kv_compressed_output = static_cast<OutFloat*>(_kv_compressed_output);
  const auto kv_out = kv_compressed_output + global_bid * kHeadDim + split_offset;

  // score bias (ape)
  const auto score_bias = static_cast<const InFloat*>(_score_bias) + split_offset;

  PDLWaitPrimary<kUsePDL>();

  /// NOTE: `position` = `seq_len - 1`. To avoid underflow, we use `seq_len + page_size - 1`
  if constexpr (kMode == PageMode::Page4Align) {
    const auto index_prev = extra[global_bid];
    const auto kv_buf = kv_score_buffer + index * (kElementSize * 4) + split_offset;
    c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/(seq_len + 3) % 4);
    if (seq_len % 4 == 0) {
      const auto kv_overlap = kv_buf + (index_prev - index) * (kElementSize * 4);
      c4_forward<true>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, 8, kv_overlap);
    }
  } else {
    static_assert(kMode == PageMode::RingBuffer, "Unsupported PageMode");
    const auto kv_buf = kv_score_buffer + index * (kElementSize * 8) + split_offset;
    c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/(seq_len + 7) % 8);
    if (seq_len % 4 == 0) {
      c4_forward<false>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, /*window_size=*/8);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, PageMode kMode, bool kWrite, bool kUsePDL>
C4_KERNEL void flash_c4_prefill(const __grid_constant__ Compress4PrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;  // 128
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  constexpr int64_t kElementSize = kHeadDim * 4;  // `* 4` due to overlap transform + score
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  const auto& [
    _kv_score_buffer, _kv_score_input, _kv_compressed_output, _score_bias, // kv score
    indices, extra, compress_plan, write_plan, num_compress, num_write // prefill plan
  ] = params;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;  // warp id
  const uint32_t global_pid = global_wid / kNumSplit;     // plan id
  const uint32_t global_sid = global_wid % kNumSplit;     // split id

  /// NOTE: compiler can optimize this if-else at compile time
  const auto num_plans = kWrite ? num_write : num_compress;
  const auto plan_ptr = kWrite ? write_plan : compress_plan;
  if (global_pid >= num_plans) return;

  const auto& [ragged_id, global_bid, position, window_len] = plan_ptr[global_pid];
  const int64_t split_offset = global_sid * kTileDim;

  // kv score
  const auto kv_score_buffer = static_cast<InFloat*>(_kv_score_buffer);

  // kv input
  const auto kv_score_input = static_cast<const InFloat*>(_kv_score_input);
  const auto kv_src = kv_score_input + ragged_id * kElementSize + split_offset;

  // kv output
  const auto kv_compressed_output = static_cast<OutFloat*>(_kv_compressed_output);
  const auto kv_out = kv_compressed_output + ragged_id * kHeadDim + split_offset;

  if (ragged_id == 0xFFFFFFFF) [[unlikely]]
    return;

  // score bias (ape)
  const auto score_bias = static_cast<const InFloat*>(_score_bias) + split_offset;
  const auto seq_len = position + 1;
  const int32_t index = indices[global_bid];

  PDLWaitPrimary<kUsePDL>();

  if constexpr (kMode == PageMode::Page4Align) {
    const auto write_second_page = index;
    const auto [load_first_page, load_second_page, write_first_page, last_pos] = extra[global_bid];
    if constexpr (kWrite) {
      int32_t index;
      if (position < static_cast<uint32_t>(last_pos)) {
        index = write_first_page;
      } else {
        index = write_second_page;
      }
      const auto kv_buf = kv_score_buffer + index * (kElementSize * 4) + split_offset;
      c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/position % 4);
    } else {
      int32_t index_overlap, index_normal;
      if (window_len <= 4) {
        index_overlap = load_second_page;
        index_normal = load_second_page;  // not used
      } else {
        index_overlap = load_first_page;
        index_normal = load_second_page;
      }
      const auto kv_buf = kv_score_buffer + index_normal * (kElementSize * 4) + split_offset;
      const auto kv_overlap = kv_score_buffer + index_overlap * (kElementSize * 4) + split_offset;
      c4_forward<true>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, window_len, kv_overlap);
    }
  } else {
    static_assert(kMode == PageMode::RingBuffer, "Unsupported PageMode");
    const auto kv_buf = kv_score_buffer + index * (kElementSize * 8) + split_offset;
    if constexpr (kWrite) {
      c4_write(kv_buf, kv_src, kHeadDim, /*write_pos=*/position % 8);
    } else {
      c4_forward<false>(kv_buf, kv_src, kv_out, score_bias, kHeadDim, seq_len, window_len);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress4Kernel {
  template <PageMode kMode>
  static constexpr auto decode_kernel = flash_c4_decode<kHeadDim, InFloat, OutFloat, kMode, kUsePDL>;
  template <PageMode kMode, bool kWrite>
  static constexpr auto prefill_kernel = flash_c4_prefill<kHeadDim, InFloat, OutFloat, kMode, kWrite, kUsePDL>;
  template <PageMode kMode>
  static constexpr auto prefill_c_kernel = prefill_kernel<kMode, /*kWrite=*/false>;
  template <PageMode kMode>
  static constexpr auto prefill_w_kernel = prefill_kernel<kMode, /*kWrite=*/true>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kWarpsPerBlock = kBlockSize / device::kWarpThreads;

  using Self = FlashCompress4Kernel;

  static void run_decode(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> extra) {
    using namespace host;

    // this should not happen in practice
    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    const auto extra_ptr = _get_extra_pointer(B, device_, extra);
    const auto page_size = extra_ptr != nullptr ? 4 : 8;

    TensorMatcher({-1, page_size, kHeadDim * 4})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 4})  // kv score input
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({B, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({B})  // indices
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({B})  // seq lens
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(seq_lens);

    const auto device = device_.unwrap();
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = Compress4DecodeParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .seq_lens = static_cast<const IndiceT*>(seq_lens.data_ptr()),
        .extra = static_cast<const int32_t*>(extra_ptr),
        .batch_size = batch_size,
    };
    const auto kernel = extra_ptr != nullptr ? decode_kernel<PageMode::Page4Align>  //
                                             : decode_kernel<PageMode::RingBuffer>;
    const uint32_t num_blocks = div_ceil(batch_size * kNumSplit, kWarpsPerBlock);
    LaunchKernel(num_blocks, kBlockSize, device)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView compress_plan,
      const tvm::ffi::TensorView write_plan,
      const tvm::ffi::Optional<tvm::ffi::TensorView> extra) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_q_tokens"};
    auto X = SymbolicSize{"compress_tokens"};
    auto Y = SymbolicSize{"write_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    const auto extra_ptr = _get_extra_pointer(B, device_, extra, /*is_prefill=*/true);
    const auto page_size = extra_ptr != nullptr ? 4 : 8;

    TensorMatcher({-1, page_size, kHeadDim * 4})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({N, kHeadDim * 4})  // kv score input
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({B})  // indices
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({X, compress::kPrefillPlanDim})  // compress plan
        .with_dtype<compress::PrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(compress_plan);
    TensorMatcher({Y, compress::kPrefillPlanDim})  // write plan
        .with_dtype<compress::PrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(write_plan);

    const auto device = device_.unwrap();
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(X.unwrap());
    const auto num_w = static_cast<uint32_t>(Y.unwrap());
    const auto params = Compress4PrefillParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .extra = static_cast<const C4IndexBundle*>(extra_ptr),
        .compress_plan = static_cast<const Plan4*>(compress_plan.data_ptr()),
        .write_plan = static_cast<const Plan4*>(write_plan.data_ptr()),
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= batch_size, "num_q_tokens must be >= batch_size");
    RuntimeCheck(num_q_tokens >= std::max(num_c, num_w), "invalid prefill plan");
    if (const auto num_c_blocks = div_ceil(num_c * kNumSplit, kWarpsPerBlock)) {
      const auto c_kernel = extra_ptr != nullptr ? prefill_c_kernel<PageMode::Page4Align>  //
                                                 : prefill_c_kernel<PageMode::RingBuffer>;
      LaunchKernel(num_c_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(c_kernel, params);
    }
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerBlock)) {
      const auto w_kernel = extra_ptr != nullptr ? prefill_w_kernel<PageMode::Page4Align>  //
                                                 : prefill_w_kernel<PageMode::RingBuffer>;
      LaunchKernel(num_w_blocks, kBlockSize, device)  //
          .enable_pdl(kUsePDL)(w_kernel, params);
    }
  }

  // some auxiliary functions
 private:
  static const void* _get_extra_pointer(
      host::SymbolicSize& B,  // batch_size
      host::SymbolicDevice& device,
      const tvm::ffi::Optional<tvm::ffi::TensorView>& extra,
      bool is_prefill = false) {
    // only have value when using page-aligned mode
    if (!extra.has_value()) return nullptr;
    const auto& extra_tensor = extra.value();
    /// NOTE: the metadata layout is different for prefill and decode:
    /// for prefill, last 4 are:
    /// load overlap | load normal | write overlap | last written page
    /// for decode, last 1 is the write (also load) overlap
    host::TensorMatcher({B, is_prefill ? 4 : 1})  // extra tensor
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(extra_tensor);
    const auto data_ptr = extra_tensor.data_ptr();
    host::RuntimeCheck(data_ptr != nullptr, "extra tensor data ptr is null");
    if (is_prefill) {
      static_assert(alignof(C4IndexBundle) == 16);
      host::RuntimeCheck(std::bit_cast<uintptr_t>(data_ptr) % 16 == 0, "extra tensor is not properly aligned");
    }
    return data_ptr;
  }
};

}  // namespace
