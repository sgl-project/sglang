#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

struct QKNormParams {
  void* __restrict__ q;
  void* __restrict__ k;  // k is offset by (-num_qo_heads * head_dim) elements
  int64_t q_stride;
  int64_t k_stride;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  float eps;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  uint32_t num_tokens;
};

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreadsPerBlock = kWarpsPerBlock * device::kWarpThreads;

template <int64_t kHeadDim, bool kUsePDL, typename Float>
__global__ void fused_qknorm(const QKNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kHeadDim>;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_blks = gridDim.x;
  const auto num_workers = num_blks * kWarpsPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    const auto input_vec = gmem.load(input);
    const auto weight_vec = gmem.load(weight);
    const auto output_vec = norm::apply_norm_warp<kHeadDim>(input_vec, weight_vec, eps);
    gmem.store(input, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <int64_t kHeadDim, bool kUsePDL, typename DType>
struct QKNormKernel {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(!host::norm::should_use_cta<DType, kHeadDim>(), "Head dim too large for QKNorm");
  static constexpr auto kernel = fused_qknorm<kHeadDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto Sq = SymbolicSize{"q_stride"};
    auto Sk = SymbolicSize{"k_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Sq, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Sk, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight)
        .verify(k_weight);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const auto params = QKNormParams{
        .q = q.data_ptr(),
        .k = pointer::offset(k.data_ptr(), -2 * static_cast<int64_t>(num_qo_heads) * kHeadDim),
        .q_stride = static_cast<int64_t>(Sq.unwrap()),
        .k_stride = static_cast<int64_t>(Sk.unwrap()),
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .eps = eps,
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .num_tokens = num_tokens,
    };

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kThreadsPerBlock);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);

    // choose kernel based on dtype
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);

    // we use persistent kernel, which limit the number of blocks to reduce overhead
    const auto num_blocks = std::min(kNumSM * max_occupancy, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
