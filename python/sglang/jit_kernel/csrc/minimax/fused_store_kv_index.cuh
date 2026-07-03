#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize/DType/Device
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil, pointer::offset

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct StoreKVIndexParams {
  const void* __restrict__ input_ptrs[4];
  void* __restrict__ cache_ptrs[4];
  const void* __restrict__ indices;
  int64_t input_stride_bytes[4];  // k stride, v stride, idx_k stride, idx_v stride
  int64_t cache_stride_bytes[4];  // k stride, v stride, idx_k stride, idx_v stride
  uint32_t num_k_heads;
  uint32_t num_total_heads;  // 2 * num_k_heads + 1 + has_index_v
  uint32_t total_jobs;       // batch_size * heads_per_token
};

template <int64_t kHeadBytes, bool kUsePDL>
struct StoreTrait {
  static_assert(kHeadBytes % 16 == 0, "head bytes must be a multiple of 16 (128-bit vector)");
  using vec_t = device::AlignedVector<uint32_t, 4>;         // 16 bytes / thread
  static constexpr uint32_t kWorkerSize = kHeadBytes / 16;  // threads per head

  template <typename T>
  SGL_DEVICE static void forward(const StoreKVIndexParams& params) {
    using namespace device;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t work_id = tid / kWorkerSize;  // which (token, head) job
    const uint32_t lane = tid % kWorkerSize;     // which 16-byte chunk of the head
    if (work_id >= params.total_jobs) return;

    const uint32_t num_total_heads = params.num_total_heads;
    const uint32_t token_id = work_id / num_total_heads;
    const uint32_t slot_id = work_id % num_total_heads;
    const uint32_t num_k_heads = params.num_k_heads;

    const auto loc = static_cast<const T*>(params.indices)[token_id];
    uint32_t head_id, which;
    if (slot_id < num_k_heads) {
      head_id = slot_id;
      which = 0;  // K
    } else if (slot_id < 2 * num_k_heads) {
      head_id = slot_id - num_k_heads;
      which = 1;  // V
    } else if (slot_id == 2 * num_k_heads) {
      head_id = 0;
      which = 2;  // idx K
    } else {
      head_id = 0;
      which = 3;  // idx V
    }
    const auto cache_ptr = static_cast<char*>(params.cache_ptrs[which]);
    const auto input_ptr = static_cast<const char*>(params.input_ptrs[which]);
    const auto cache_stride = params.cache_stride_bytes[which];
    const auto input_stride = params.input_stride_bytes[which];
    const auto src = pointer::offset(input_ptr, token_id * input_stride, head_id * kHeadBytes);
    const auto dst = pointer::offset(cache_ptr, loc * cache_stride, head_id * kHeadBytes);
    PDLWaitPrimary<kUsePDL>();
    vec_t chunk;
    chunk.load(src, lane);
    chunk.store(dst, lane);
    PDLTriggerSecondary<kUsePDL>();
  }
};

template <typename Trait, typename T>
__global__ void store_kv_index_kernel(const __grid_constant__ StoreKVIndexParams params) {
  Trait::template forward<T>(params);
}

// idx_v / idx_v_cache may be dummies (== idx_k / idx_k_cache) when the layer
// has no index value; the caller signals "no V" via heads_per_token == 2*hkv+1
// so the index-V branch is never taken.
template <int64_t kHeadBytes, bool kUsePDL>
void store_kv_index(
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView idx_k,
    tvm::ffi::TensorView idx_k_cache,
    tvm::ffi::TensorView idx_v,
    tvm::ffi::TensorView idx_v_cache,
    tvm::ffi::TensorView indices,
    int64_t num_kv_heads,
    int64_t heads_per_token) {
  using namespace host;
  auto B = SymbolicSize{"batch"};
  auto Mrow = SymbolicSize{"main_row"};  // num_kv_heads * head_dim
  auto Drow = SymbolicSize{"idx_row"};   // head_dim
  auto dtype = SymbolicDType{};
  auto indice_dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // All cache/source tensors share the same store dtype (fast-path precondition).
  TensorMatcher({B, Mrow})  //
      .with_strides({-1, 1})
      .with_dtype(dtype)
      .with_device(device)
      .verify(k)
      .verify(v);
  TensorMatcher({-1, Mrow})  //
      .with_strides({-1, 1})
      .with_dtype(dtype)
      .with_device(device)
      .verify(k_cache)
      .verify(v_cache);
  TensorMatcher({B, Drow})  //
      .with_strides({-1, 1})
      .with_dtype(dtype)
      .with_device(device)
      .verify(idx_k)
      .verify(idx_v);
  TensorMatcher({-1, Drow})  //
      .with_strides({-1, 1})
      .with_dtype(dtype)
      .with_device(device)
      .verify(idx_k_cache)
      .verify(idx_v_cache);
  TensorMatcher({B})  //
      .with_dtype<int32_t, int64_t>(indice_dtype)
      .with_device(device)
      .verify(indices);

  const int64_t dsize = dtype_bytes(dtype.unwrap());
  RuntimeCheck(kHeadBytes == Drow.unwrap() * dsize);
  RuntimeCheck(Mrow.unwrap() == num_kv_heads * Drow.unwrap());

  const auto params = StoreKVIndexParams{
      .input_ptrs = {k.data_ptr(), v.data_ptr(), idx_k.data_ptr(), idx_v.data_ptr()},
      .cache_ptrs = {k_cache.data_ptr(), v_cache.data_ptr(), idx_k_cache.data_ptr(), idx_v_cache.data_ptr()},
      .indices = indices.data_ptr(),
      .input_stride_bytes =
          {
              k.stride(0) * dsize,
              v.stride(0) * dsize,
              idx_k.stride(0) * dsize,
              idx_v.stride(0) * dsize,
          },
      .cache_stride_bytes =
          {
              k_cache.stride(0) * dsize,
              v_cache.stride(0) * dsize,
              idx_k_cache.stride(0) * dsize,
              idx_v_cache.stride(0) * dsize,
          },
      .num_k_heads = static_cast<uint32_t>(num_kv_heads),
      .num_total_heads = static_cast<uint32_t>(heads_per_token),
      .total_jobs = static_cast<uint32_t>(B.unwrap() * heads_per_token),
  };
  if (params.total_jobs == 0) return;

  using Trait = StoreTrait<kHeadBytes, kUsePDL>;
  constexpr uint32_t kBlockSize = 256u;
  const uint32_t num_blocks = div_ceil(params.total_jobs * Trait::kWorkerSize, kBlockSize);
  const auto kernel = indice_dtype.is_type<int32_t>()  //
                          ? store_kv_index_kernel<Trait, int32_t>
                          : store_kv_index_kernel<Trait, int64_t>;
  LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
      .enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace
