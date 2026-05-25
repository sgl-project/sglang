#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kSplitKV = 256;  // const for both SM90 and SM100

struct MetadataParams {
  /// NOTE: batch_size > 0
  uint32_t batch_size;
  uint32_t num_sm;
  const uint32_t* __restrict__ context_lens;
  uint32_t* __restrict__ schedule_metadata;
  bool use_smem = true;
};

__global__ __launch_bounds__(kBlockSize, 1)  //
    void smxx_paged_mqa_logits_metadata(const MetadataParams params) {
  using namespace device;
  extern __shared__ uint32_t s_length[];
  static constexpr auto kNumWarps = kBlockSize / kWarpThreads;
  static_assert(kNumWarps == kWarpThreads);

  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = tx / kWarpThreads;

  __shared__ uint32_t s_warp_sum[kNumWarps];

  uint32_t local_sum = 0;
  for (uint32_t i = tx; i < params.batch_size; i += kBlockSize) {
    const auto length = params.context_lens[i];
    local_sum += (length + kSplitKV - 1) / kSplitKV;
    if (params.use_smem) s_length[i] = length;
  }

  s_warp_sum[warp_id] = warp::reduce_sum(local_sum);
  __syncthreads();

  const auto global_sum = warp::reduce_sum(s_warp_sum[lane_id]);
  if (lane_id != 0) return;

  const auto length_ptr = params.use_smem ? s_length : params.context_lens;

  const auto avg = global_sum / params.num_sm;
  const auto ret = global_sum % params.num_sm;
  uint32_t q = 0;
  uint32_t num_work = (length_ptr[0] + kSplitKV - 1) / kSplitKV;
  uint32_t sum_work = num_work;
  for (auto i = warp_id; i <= params.num_sm; i += kNumWarps) {
    const auto target = i * avg + min(i, ret);
    while (sum_work <= target) {
      if (++q >= params.batch_size) break;
      num_work = (length_ptr[q] + kSplitKV - 1) / kSplitKV;
      sum_work += num_work;
    }
    if (q >= params.batch_size) {
      params.schedule_metadata[2 * i + 0] = params.batch_size;
      params.schedule_metadata[2 * i + 1] = 0;
    } else {
      // sum > target && (sum - length) <= target
      params.schedule_metadata[2 * i + 0] = q;
      params.schedule_metadata[2 * i + 1] = target - (sum_work - num_work);
    }
  }
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

struct IndexerMetadataKernel {
  static constexpr auto kMaxBatchSizeInSmem = 16384 * 2;  // 128 KB smeme
  static void run(tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView metadata) {
    using namespace host;
    auto N = SymbolicSize{"batch_size"};
    auto M = SymbolicSize{"num_sm"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({M, 2})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(metadata);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto num_sm = static_cast<uint32_t>(M.unwrap()) - 1;
    RuntimeCheck(num_sm <= 1024);
    const auto use_smem = batch_size <= kMaxBatchSizeInSmem;
    const auto params = MetadataParams{
        .batch_size = batch_size,
        .num_sm = num_sm,
        .context_lens = static_cast<uint32_t*>(seq_lens.data_ptr()),
        .schedule_metadata = static_cast<uint32_t*>(metadata.data_ptr()),
        .use_smem = use_smem,
    };
    constexpr auto kernel = smxx_paged_mqa_logits_metadata;
    setup_kernel_smem_once<kernel, (kMaxBatchSizeInSmem + 1) * sizeof(uint32_t)>();
    const auto smem = use_smem ? (batch_size + 1) * sizeof(uint32_t) : 0;
    LaunchKernel(1, kBlockSize, device.unwrap(), smem)(kernel, params);
  }
};

}  // namespace
