#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <numeric>

namespace {

struct FusedRopeParams {
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;  // NOTE: this k is pre-offset in host code to reduce computation in kernel
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
};

struct FusedRopeStoreParams {
  FusedRopeParams base_params;
  void* v_ptr;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ out_loc;
  int64_t v_stride_bytes;
  int64_t cache_stride_bytes;
};

constexpr uint32_t kBlockSize = 128;

[[maybe_unused]]
constexpr auto next_pow2(uint32_t target, uint32_t factor = 1) {
  uint32_t power = 1;
  while (power * factor < target)
    power *= 2;
  return power;
}

template <bool kIsNeox, int64_t kRopeDim, bool kUsePDL, typename DType, typename IdType, uint32_t kWorkThreads>
__global__ void fused_rope_kernel(const __grid_constant__ FusedRopeParams params) {
  using namespace device;

  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);
  constexpr int64_t kVecSize = next_pow2(kRopeDim, (2 * kWorkThreads * (1 + kIsNeox)));
  using DType2 = packed_t<DType>;
  using InputStorage = AlignedVector<DType2, kVecSize>;
  constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
  constexpr uint32_t kLaneCount = kRopeDim / kDimPerThread;
  static_assert(kRopeDim % kDimPerThread == 0 && kLaneCount <= kWorkThreads);

  const auto &[
    q, k, cos_sin_cache_ptr, positions, // pointers
    q_stride_bytes, k_stride_bytes, head_stride_bytes,  // strides
    num_qo_heads, num_kv_heads, num_tokens // dimensions
  ] = params;

  const auto num_blks = gridDim.x;
  constexpr auto kWorkersPerBlock = kBlockSize / kWorkThreads;
  const auto num_workers = num_blks * kWorkersPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = (blockIdx.x * kBlockSize + threadIdx.x) / kWorkThreads;
  const auto cos_cache_ptr = cos_sin_cache_ptr;
  const auto sin_cache_ptr = pointer::offset(cos_sin_cache_ptr, kCosSinStrideBytes / 2);

  uint32_t lane_id = threadIdx.x % kWorkThreads;
  if constexpr (kLaneCount < kWorkThreads) {
    if (lane_id >= kLaneCount) return;
  }

  PDLWaitPrimary<kUsePDL>();

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto pos = static_cast<const IdType*>(positions)[token_id];
    const auto load_q = head_id < num_qo_heads;
    const auto input_ = load_q ? pointer::offset(q, token_id * q_stride_bytes)  //
                               : pointer::offset(k, token_id * k_stride_bytes);
    const auto input = pointer::offset(input_, head_id * head_stride_bytes);
    const auto cos_ptr = pointer::offset(cos_cache_ptr, pos * kCosSinStrideBytes);
    const auto sin_ptr = pointer::offset(sin_cache_ptr, pos * kCosSinStrideBytes);
    if constexpr (kIsNeox) {
      using CacheStorage = AlignedVector<fp32x2_t, kVecSize>;
      const auto input_x = input;
      const auto input_y = pointer::offset(input, (kRopeDim / 2) * sizeof(DType));
      auto input_vec_x = load_as<InputStorage>(input_x, lane_id);
      auto input_vec_y = load_as<InputStorage>(input_y, lane_id);
      const auto cos_pair = load_as<CacheStorage>(cos_ptr, lane_id);
      const auto sin_pair = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
      for (int64_t j = 0; j < kVecSize; ++j) {
        const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);
        const auto [y0, y1] = cast<fp32x2_t>(input_vec_y[j]);
        const auto [cos_0, cos_1] = cos_pair[j];
        const auto [sin_0, sin_1] = sin_pair[j];
        const auto out_x0 = x0 * cos_0 - y0 * sin_0;
        const auto out_y0 = x0 * sin_0 + y0 * cos_0;
        const auto out_x1 = x1 * cos_1 - y1 * sin_1;
        const auto out_y1 = x1 * sin_1 + y1 * cos_1;
        input_vec_x[j] = cast<DType2, fp32x2_t>({out_x0, out_x1});
        input_vec_y[j] = cast<DType2, fp32x2_t>({out_y0, out_y1});
      }
      store_as<InputStorage>(input_x, input_vec_x, lane_id);
      store_as<InputStorage>(input_y, input_vec_y, lane_id);
    } else {
      using CacheStorage = AlignedVector<float, kVecSize>;
      auto input_vec = load_as<InputStorage>(input, lane_id);
      const auto cos_vec = load_as<CacheStorage>(cos_ptr, lane_id);
      const auto sin_vec = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
      for (int64_t j = 0; j < kVecSize; ++j) {
        const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
        const auto cos = cos_vec[j];
        const auto sin = sin_vec[j];
        const auto out_x = x * cos - y * sin;
        const auto out_y = x * sin + y * cos;
        input_vec[j] = cast<DType2, fp32x2_t>({out_x, out_y});
      }
      store_as<InputStorage>(input, input_vec, lane_id);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kIsNeox, int64_t kRopeDim, bool kUsePDL, typename DType, typename IdType, uint32_t kWorkThreads>
__global__ void fused_rope_store_kernel(const __grid_constant__ FusedRopeStoreParams params) {
  using namespace device;

  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);
  constexpr int64_t kVecSize = kRopeDim / (2 * kWorkThreads * (1 + kIsNeox));
  using DType2 = packed_t<DType>;
  using InputStorage = AlignedVector<DType2, kVecSize>;
  constexpr int64_t kDimPerThread = kVecSize * 2 * (1 + kIsNeox);
  static_assert(kRopeDim == kDimPerThread * kWorkThreads);

  const auto& [base_params, v_ptr, k_cache, v_cache, out_loc, v_stride_bytes, cache_stride_bytes] = params;
  const auto &[
    q, k, cos_sin_cache_ptr, positions, // pointers
    q_stride_bytes, k_stride_bytes, head_stride_bytes,  // strides
    num_qo_heads, num_kv_heads, num_tokens // dimensions
  ] = base_params;

  const auto num_blks = gridDim.x;
  constexpr auto kWorkersPerBlock = kBlockSize / kWorkThreads;
  const auto num_workers = num_blks * kWorkersPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto num_extra_works = num_kv_heads * num_tokens;  // rope works + v store works
  const auto start_worker_id = (blockIdx.x * kBlockSize + threadIdx.x) / kWorkThreads;
  const auto lane_id = threadIdx.x % kWorkThreads;
  const auto cos_cache_ptr = cos_sin_cache_ptr;
  const auto sin_cache_ptr = pointer::offset(cos_sin_cache_ptr, kCosSinStrideBytes / 2);

  auto idx = start_worker_id;

  PDLWaitPrimary<kUsePDL>();
  // in this case, head_dim = rope_dim must be true
  __builtin_assume(head_stride_bytes == kRopeDim * sizeof(DType));

  for (; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto pos = static_cast<const IdType*>(positions)[token_id];
    const auto loc = static_cast<const IdType*>(out_loc)[token_id];
    const auto load_q = head_id < num_qo_heads;
    const auto input_ = load_q ? pointer::offset(q, token_id * q_stride_bytes)  //
                               : pointer::offset(k, token_id * k_stride_bytes);
    const auto input = pointer::offset(input_, head_id * head_stride_bytes);
    const auto cos_ptr = pointer::offset(cos_cache_ptr, pos * kCosSinStrideBytes);
    const auto sin_ptr = pointer::offset(sin_cache_ptr, pos * kCosSinStrideBytes);
    if constexpr (kIsNeox) {
      using CacheStorage = AlignedVector<fp32x2_t, kVecSize>;
      const auto input_x = input;
      const auto input_y = pointer::offset(input, (kRopeDim / 2) * sizeof(DType));
      auto input_vec_x = load_as<InputStorage>(input_x, lane_id);
      auto input_vec_y = load_as<InputStorage>(input_y, lane_id);
      const auto cos_pair = load_as<CacheStorage>(cos_ptr, lane_id);
      const auto sin_pair = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
      for (int64_t j = 0; j < kVecSize; ++j) {
        const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);
        const auto [y0, y1] = cast<fp32x2_t>(input_vec_y[j]);
        const auto [cos_0, cos_1] = cos_pair[j];
        const auto [sin_0, sin_1] = sin_pair[j];
        const auto out_x0 = x0 * cos_0 - y0 * sin_0;
        const auto out_y0 = x0 * sin_0 + y0 * cos_0;
        const auto out_x1 = x1 * cos_1 - y1 * sin_1;
        const auto out_y1 = x1 * sin_1 + y1 * cos_1;
        input_vec_x[j] = cast<DType2, fp32x2_t>({out_x0, out_x1});
        input_vec_y[j] = cast<DType2, fp32x2_t>({out_y0, out_y1});
      }
      const auto k_out = pointer::offset(k_cache, loc * cache_stride_bytes, head_id * head_stride_bytes);
      const auto output_x = load_q ? input : k_out;
      store_as<InputStorage>(output_x, input_vec_x, lane_id);
      const auto output_y = pointer::offset(output_x, (kRopeDim / 2) * sizeof(DType));
      store_as<InputStorage>(output_y, input_vec_y, lane_id);
    } else {
      using CacheStorage = AlignedVector<float, kVecSize>;
      auto input_vec = load_as<InputStorage>(input, lane_id);
      const auto cos_vec = load_as<CacheStorage>(cos_ptr, lane_id);
      const auto sin_vec = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
      for (int64_t j = 0; j < kVecSize; ++j) {
        const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
        const auto cos = cos_vec[j];
        const auto sin = sin_vec[j];
        const auto out_x = x * cos - y * sin;
        const auto out_y = x * sin + y * cos;
        input_vec[j] = cast<DType2, fp32x2_t>({out_x, out_y});
      }
      const auto k_out = pointer::offset(k_cache, loc * cache_stride_bytes, head_id * head_stride_bytes);
      const auto output = load_q ? input : k_out;
      store_as<InputStorage>(output, input_vec, lane_id);
    }
  }

  __syncwarp();  // to avoid warp divergence
  idx -= num_works;
  for (; idx < num_extra_works; idx += num_workers) {
    using VStorage = AlignedVector<DType, kRopeDim / kWorkThreads>;
    const int64_t token_id = idx / num_kv_heads;
    const int64_t head_id = idx % num_kv_heads;
    const auto loc = static_cast<const IdType*>(out_loc)[token_id];
    const auto input = pointer::offset(v_ptr, token_id * v_stride_bytes, head_id * head_stride_bytes);
    const auto input_vec = load_as<VStorage>(input, lane_id);
    const auto output = pointer::offset(v_cache, loc * cache_stride_bytes, head_id * head_stride_bytes);
    store_as<VStorage>(output, input_vec, lane_id);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kIsNeox, int64_t kRopeDim, bool kUsePDL, typename DType>
struct FusedRopeKernel {
  static constexpr uint32_t kDimPerThread = std::gcd(16 / sizeof(DType), kRopeDim);
  static constexpr uint32_t kWorkThreads = next_pow2(kRopeDim, kDimPerThread);
  static constexpr bool kSupportFused = kWorkThreads * kDimPerThread == kRopeDim;
  static_assert(kRopeDim % kDimPerThread == 0);
  static_assert(kBlockSize % kWorkThreads == 0);

  template <typename IdType>
  static constexpr auto _kernel_0 = fused_rope_kernel<kIsNeox, kRopeDim, kUsePDL, DType, IdType, kWorkThreads>;
  template <typename IdType>
  static constexpr auto _kernel_1 = fused_rope_store_kernel<kIsNeox, kRopeDim, kUsePDL, DType, IdType, kWorkThreads>;

  static auto get_num_sm(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kRopeDim);
    device.set_options<kDLCUDA>();
    TensorMatcher({N, Q, D})  // q input
        .with_strides({Dq, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Dk, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({-1, D})  // cos_sin_cache
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N})  // positions
        .with_dtype<int32_t, int64_t>(id_type)
        .with_device(device)
        .verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = FusedRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel_0<int32_t> : _kernel_0<int64_t>;
    const uint32_t kNumSM = get_num_sm(device.unwrap());
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(_kernel_0<int32_t>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel_0<int64_t>, kBlockSize),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_works, (kBlockSize / kWorkThreads));
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void run_fused(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc) {
    if constexpr (kSupportFused) {
      return _run_fused_impl(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc);
    } else {
      host::Panic("Fused rope + store is not supported for rope_dim ", kRopeDim);
    }
  }

  static void _run_fused_impl(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"rope_dim"};
    auto R = SymbolicSize{"row_size"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dv = SymbolicSize{"v_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto Dc = SymbolicSize{"cache_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Dq, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Dk, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({N, K, D})  // v input
        .with_strides({Dv, Dd, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, D})  // cos_sin_cache
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N})  // positions, out_loc
        .with_dtype<int32_t, int64_t>(id_type)
        .with_device(device)
        .verify(positions)
        .verify(out_loc);
    TensorMatcher({-1, R})  // k_cache
        .with_strides({Dc, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_cache)
        .verify(v_cache);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride = Dd.unwrap();
    const auto row_dim = R.unwrap();
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    RuntimeCheck(kRopeDim == head_stride, "rope_dim ", kRopeDim, " should = head_stride ", head_stride);
    RuntimeCheck(num_kv_heads * kRopeDim == row_dim, "invalid kvcache");

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = FusedRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
    };

    const auto v_stride_bytes = static_cast<int64_t>(Dv.unwrap() * sizeof(DType));
    const auto cache_stride_bytes = static_cast<int64_t>(Dc.unwrap() * sizeof(DType));
    const auto store_params = FusedRopeStoreParams{
        .base_params = params,
        .v_ptr = v.data_ptr(),
        .k_cache = pointer::offset(k_cache.data_ptr(), -k_offset),
        .v_cache = v_cache.data_ptr(),
        .out_loc = out_loc.data_ptr(),
        .v_stride_bytes = v_stride_bytes,
        .cache_stride_bytes = cache_stride_bytes,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel_1<int32_t> : _kernel_1<int64_t>;
    const uint32_t kNumSM = get_num_sm(device.unwrap());
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(_kernel_1<int32_t>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel_1<int64_t>, kBlockSize),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    // rope works for q+k heads, plus v store works for kv heads
    const auto num_total_works = (num_qo_heads + 2 * num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_total_works, (kBlockSize / kWorkThreads));
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, store_params);
  }
};

}  // namespace
