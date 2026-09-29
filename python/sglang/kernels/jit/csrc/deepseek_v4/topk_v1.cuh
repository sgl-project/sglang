#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/dsa/legacy_radix_topk.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

// `topk` is a *runtime* value (<= kMaxTopK), so one module serves every k. It
// used to be baked in via -DSGL_TOPK, which built a separate module per k --
// and because `kTopK` came from a macro rather than a template parameter, both
// modules exported identically mangled symbols. The function-local static in
// setup_kernel_smem_once() is emitted as STB_GNU_UNIQUE, which the loader
// merges across every loaded object, so whichever module was used second
// skipped its cudaFuncSetAttribute opt-in and then failed to launch with 64 KB
// of dynamic shared memory ("invalid argument").
constexpr uint32_t kMaxTopK = 1024;
// Fixed, and deliberately not tied to `topk`: run_cumsum() and the histogram
// init below index up to RADIX + 1 == 257 threads, so a block sized after a
// small topk would silently skip part of the histogram.
constexpr uint32_t kTopKBlockSize = kMaxTopK;
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)

struct TopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
  uint32_t topk;
};

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

[[maybe_unused]]
SGL_DEVICE void naive_transform(
    const float* __restrict__,  // unused
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,  // optional: output raw abs position indices
    const uint32_t length,
    const uint32_t page_bits,
    const uint32_t topk) {
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    if (raw_indices != nullptr) {
      raw_indices[tx] = tx;
    }
  } else if (tx < topk) {
    indices[tx] = -1;  // fill invalid indices to -1
    if (raw_indices != nullptr) {
      raw_indices[tx] = -1;
    }
  }
}

[[maybe_unused]]
SGL_DEVICE void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length, const uint32_t topk) {
  ::sglang::device::legacy_radix_topk::
      select<kTopKBlockSize, kMaxTopK, static_cast<int>(kSMEM / (2 * sizeof(int32_t)))>(
          input, output, 0, static_cast<int>(length), static_cast<int>(topk));
  __syncthreads();
}

template <bool kUsePDL>
__global__ void topk_transform_kernel(const __grid_constant__ TopKParams params) {
  const auto &[
    scores, seq_lens, page_table, page_indices, raw_indices, // pointers
    score_stride, page_table_stride, page_bits, topk // sizes
  ] = params;
  const uint32_t work_id = blockIdx.x;

  /// NOTE: dangerous prefetch seq_len before PDL wait
  const uint32_t seq_len = seq_lens[work_id];
  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * topk;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * topk : nullptr;

  device::PDLWaitPrimary<kUsePDL>();

  if (seq_len <= topk) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits, topk);
  } else {
    __shared__ int32_t s_topk_indices[kMaxTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len, topk);
    const auto tx = threadIdx.x;
    if (tx < topk) {
      const auto raw = s_topk_indices[tx];
      indices_ptr[tx] = raw < 0 ? -1 : page_to_indices(page_ptr, raw, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = raw;
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

template <bool kUsePDL>
struct TopKKernel {
  static constexpr auto kernel = topk_transform_kernel<kUsePDL>;

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B})  // seq_lens, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // strided page table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})  // output, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K})  // optional raw indices output, must be contiguous
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 1024]");
    const auto params = TopKParams{
        .scores = static_cast<float*>(scores.data_ptr()),
        .seq_lens = static_cast<int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
        .topk = topk,
    };
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);  // align up a little
    setup_kernel_smem_once<kernel, kSMEM_>();
    LaunchKernel(batch_size, kTopKBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
