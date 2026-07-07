#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct FusedQkvParams {
  const void* __restrict__ q;
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ q_out;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ cache_loc;
  const float* __restrict__ k_scale;
  const float* __restrict__ v_scale;
  int64_t q_stride;
  int64_t k_stride;
  int64_t v_stride;
  uint32_t num_tokens;
  uint32_t q_dim;
  uint32_t kv_dim;
};

constexpr uint32_t kBlockSize = 128;

template <typename T, int kVecN>
SGL_DEVICE void quant_row(const T* __restrict__ src, fp8_e4m3_t* __restrict__ dst, uint32_t n, float inv_scale) {
  using namespace device;
  using in_vec = AlignedVector<T, kVecN>;
  using out_vec = AlignedVector<fp8_e4m3_t, kVecN>;

  const uint32_t n_vec = n / kVecN;
  for (uint32_t vi = threadIdx.x; vi < n_vec; vi += blockDim.x) {
    in_vec iv;
    iv.load(src, vi);
    out_vec ov;
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      ov[i] = static_cast<fp8_e4m3_t>(static_cast<float>(iv[i]) * inv_scale);
    }
    ov.store(dst, vi);
  }

  const uint32_t base = n_vec * kVecN;
  for (uint32_t i = base + threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = static_cast<fp8_e4m3_t>(static_cast<float>(src[i]) * inv_scale);
  }
}

template <typename T, typename IdxT, int kVecN, bool kUsePDL, bool kQuantizeQ>
__global__ void fused_fp8_qkv_kv_cache_kernel(const __grid_constant__ FusedQkvParams params) {
  using namespace device;
  const uint32_t token = blockIdx.x;
  if (token >= params.num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const IdxT slot = static_cast<const IdxT*>(params.cache_loc)[token];
  const float inv_k = 1.0f / (*params.k_scale);
  const float inv_v = 1.0f / (*params.v_scale);

  if constexpr (kQuantizeQ) {
    quant_row<T, kVecN>(
        static_cast<const T*>(params.q) + static_cast<size_t>(token) * params.q_stride,
        static_cast<fp8_e4m3_t*>(params.q_out) + static_cast<size_t>(token) * params.q_dim,
        params.q_dim,
        1.0f);
  }
  quant_row<T, kVecN>(
      static_cast<const T*>(params.k) + static_cast<size_t>(token) * params.k_stride,
      static_cast<fp8_e4m3_t*>(params.k_cache) + static_cast<size_t>(slot) * params.kv_dim,
      params.kv_dim,
      inv_k);
  quant_row<T, kVecN>(
      static_cast<const T*>(params.v) + static_cast<size_t>(token) * params.v_stride,
      static_cast<fp8_e4m3_t*>(params.v_cache) + static_cast<size_t>(slot) * params.kv_dim,
      params.kv_dim,
      inv_v);

  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct FusedFp8QkvKvCache {
  static constexpr int kVecWide = device::kMaxVecBytes / sizeof(T);
  static constexpr int kVec128 = 16 / sizeof(T);

  template <typename IdxT, int kVecN, bool kQuantizeQ>
  static constexpr auto kernel = fused_fp8_qkv_kv_cache_kernel<T, IdxT, kVecN, kUsePDL, kQuantizeQ>;

  template <typename IdxT, bool kQuantizeQ>
  static auto get_kernel(int vec_n) {
    if (vec_n == kVecWide) return kernel<IdxT, kVecWide, kQuantizeQ>;
    if (vec_n == kVec128) return kernel<IdxT, kVec128, kQuantizeQ>;
    return kernel<IdxT, 1, kQuantizeQ>;
  }

  static bool aligned(const void* p, int bytes) {
    return reinterpret_cast<uintptr_t>(p) % bytes == 0;
  }

  static void
  run(const tvm::ffi::Optional<tvm::ffi::TensorView> q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::Optional<tvm::ffi::TensorView> q_out,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView cache_loc,
      const tvm::ffi::TensorView k_scale,
      const tvm::ffi::TensorView v_scale) {
    using namespace host;
    const bool quantize_q = q.has_value();
    RuntimeCheck(quantize_q == q_out.has_value(), "fused_fp8_qkv_kv_cache: q and q_out must both be given or both omitted");

    auto N = SymbolicSize{"num_tokens"};
    auto Dkv = SymbolicSize{"kv_dim"};
    auto S = SymbolicSize{"num_slots"};
    auto SK = SymbolicSize{"k_stride"};
    auto SV = SymbolicSize{"v_stride"};
    auto device = SymbolicDevice{};
    auto idx_dtype = SymbolicDType{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Dkv}).with_strides({SK, 1}).with_dtype<T>().with_device(device).verify(k);
    TensorMatcher({N, Dkv}).with_strides({SV, 1}).with_dtype<T>().with_device(device).verify(v);
    TensorMatcher({S, Dkv}).with_dtype<fp8_e4m3_t>().with_device(device).verify(k_cache).verify(v_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(idx_dtype).with_device(device).verify(cache_loc);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(k_scale).verify(v_scale);

    uint32_t q_dim = 0;
    int64_t q_stride = 0;
    const void* q_ptr = nullptr;
    void* q_out_ptr = nullptr;
    if (quantize_q) {
      auto Dq = SymbolicSize{"q_dim"};
      auto SQ = SymbolicSize{"q_stride"};
      TensorMatcher({N, Dq}).with_strides({SQ, 1}).with_dtype<T>().with_device(device).verify(q.value());
      TensorMatcher({N, Dq}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q_out.value());
      q_dim = static_cast<uint32_t>(Dq.unwrap());
      q_stride = SQ.unwrap();
      q_ptr = q.value().data_ptr();
      q_out_ptr = q_out.value().data_ptr();
    }

    const uint32_t num_tokens = static_cast<uint32_t>(N.unwrap());
    const uint32_t kv_dim = static_cast<uint32_t>(Dkv.unwrap());
    const int64_t k_stride = SK.unwrap();
    const int64_t v_stride = SV.unwrap();
    RuntimeCheck(num_tokens > 0, "fused_fp8_qkv_kv_cache: num_tokens must be > 0, got ", num_tokens);

    auto fits = [&](int vec) {
      const int in_bytes = vec * static_cast<int>(sizeof(T));
      bool ok = kv_dim % vec == 0 && k_stride % vec == 0 && v_stride % vec == 0 &&
                aligned(k.data_ptr(), in_bytes) && aligned(v.data_ptr(), in_bytes);
      if (quantize_q) {
        ok = ok && q_dim % vec == 0 && q_stride % vec == 0 && aligned(q_ptr, in_bytes);
      }
      return ok;
    };
    const int vec_n = fits(kVecWide) ? kVecWide : (fits(kVec128) ? kVec128 : 1);

    const auto params = FusedQkvParams{
        .q = q_ptr,
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .q_out = q_out_ptr,
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .cache_loc = cache_loc.data_ptr(),
        .k_scale = static_cast<const float*>(k_scale.data_ptr()),
        .v_scale = static_cast<const float*>(v_scale.data_ptr()),
        .q_stride = q_stride,
        .k_stride = k_stride,
        .v_stride = v_stride,
        .num_tokens = num_tokens,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
    };

    auto launch = [&](auto kernel) {
      LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel, params);
    };
    if (quantize_q) {
      launch(idx_dtype.is_type<int32_t>() ? get_kernel<int32_t, true>(vec_n) : get_kernel<int64_t, true>(vec_n));
    } else {
      launch(idx_dtype.is_type<int32_t>() ? get_kernel<int32_t, false>(vec_n) : get_kernel<int64_t, false>(vec_n));
    }
  }
};

}  // namespace
