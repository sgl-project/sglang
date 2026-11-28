#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/e2m1_software.cuh>
#include <sgl_kernel/deepseek_v4/packed_main_kv.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>
#include <optional>

namespace sglang {

struct PackedMainKVStoreParams {
  const bf16_t* __restrict__ latent;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ slots;
  uint8_t* __restrict__ storage;
  int32_t* __restrict__ error_flag;
};

template <int64_t kPageSlots, typename LocT>
__global__ __launch_bounds__(256) void pack_dsv41_main_kv_fp4_kernel(const PackedMainKVStoreParams params) {
  using namespace device;
  using Paged = deepseek_v4::PackedMainKVPaged<kPageSlots>;

  constexpr uint32_t kHeadDim = 512;
  constexpr uint32_t kRopeDim = 64;
  constexpr uint32_t kNopePairs = (kHeadDim - kRopeDim) / 2;
  constexpr uint32_t kBlockLanes = 16 / 2;

  const uint32_t row_id = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const auto slot = static_cast<const LocT*>(params.slots)[row_id];
  if (slot <= 0) return;

  AlignedVector<bf16x2_t, 1> input;
  input.load(params.latent + static_cast<int64_t>(row_id) * kHeadDim, lane);
  auto values = cast<fp32x2_t>(input[0]);

  if (lane >= kNopePairs) {
    AlignedVector<float, 2> freq;
    freq.load(params.freqs_cis + static_cast<int64_t>(row_id) * kRopeDim, lane - kNopePairs);
    const auto rotated = cast<bf16x2_t>(fp32x2_t{
        __fsub_rn(__fmul_rn(values.x, freq[0]), __fmul_rn(values.y, freq[1])),
        __fadd_rn(__fmul_rn(values.x, freq[1]), __fmul_rn(values.y, freq[0])),
    });
    values = cast<fp32x2_t>(rotated);
  }

  if (!isfinite(values.x) || !isfinite(values.y)) {
    if (params.error_flag != nullptr) atomicExch(params.error_flag, 1);
    values = {0.0f, 0.0f};
  }

  const float local_amax = fmaxf(fabsf(values.x), fabsf(values.y));
  const float amax = warp::reduce_max<kBlockLanes>(local_amax);
  const auto scale_e4m3 = __nv_fp8_e4m3{fminf(fmaxf(amax * (1.0f / 6.0f), 0x1p-9f), 448.0f)};
  const float scale = static_cast<float>(scale_e4m3);
  const uint8_t low = deepseek_v4::e2m1::encode_rne_satfinite(__fdiv_rn(__fadd_rn(values.x, 0.0f), scale));
  const uint8_t high = deepseek_v4::e2m1::encode_rne_satfinite(__fdiv_rn(__fadd_rn(values.y, 0.0f), scale));
  const uint8_t packed = low | (high << 4);

  const auto output = Paged::row(params.storage, slot);
  if (lane < kNopePairs) {
    output.payload[lane] = packed;
    if ((lane & (kBlockLanes - 1)) == 0) {
      output.scales[lane / kBlockLanes] = scale_e4m3.__x;
    }
    if (lane == 0) {
      *reinterpret_cast<uint32_t*>(output.scales + deepseek_v4::PackedMainKVTraits::kValidScalesPerSlot) = 0;
    }
  } else {
    const fp32x2_t dequantized{
        deepseek_v4::e2m1::decode(low) * scale,
        deepseek_v4::e2m1::decode(high) * scale,
    };
    reinterpret_cast<bf16x2_t*>(output.rope)[lane - kNopePairs] = cast<bf16x2_t>(dequantized);
  }
}

template <int64_t kPageSlots>
struct PackDSV41MainKVKernel {
  static constexpr int64_t kPageBytes = deepseek_v4::PackedMainKVPaged<kPageSlots>::kPageBytes;
  static constexpr uint32_t kBlockSize = 256;

  static void
  run(tvm::ffi::TensorView latent,
      tvm::ffi::TensorView freqs_cis,
      tvm::ffi::TensorView slots,
      tvm::ffi::TensorView storage) {
    launch(latent, freqs_cis, slots, storage, std::nullopt);
  }

  static void run_debug(
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView freqs_cis,
      tvm::ffi::TensorView slots,
      tvm::ffi::TensorView storage,
      tvm::ffi::TensorView error_flag) {
    launch(latent, freqs_cis, slots, storage, error_flag);
  }

 private:
  using MaybeTensor = std::optional<tvm::ffi::TensorView>;

  template <typename LocT>
  static constexpr auto kernel = pack_dsv41_main_kv_fp4_kernel<kPageSlots, LocT>;

  static void launch(
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView freqs_cis,
      tvm::ffi::TensorView slots,
      tvm::ffi::TensorView storage,
      MaybeTensor error_flag) {
    using namespace host;

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens, 512}).with_dtype<bf16_t>().with_device(device).verify(latent);
    TensorMatcher({num_tokens, 64}).with_dtype<fp32_t>().with_device(device).verify(freqs_cis);
    auto slot_dtype = SymbolicDType{};
    TensorMatcher({num_tokens}).with_dtype<int32_t, int64_t>(slot_dtype).with_device(device).verify(slots);
    TensorMatcher({-1, kPageBytes})
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(storage);
    CHECK_HOST(reinterpret_cast<uintptr_t>(storage.data_ptr()) % 16 == 0)
        << "packed Main KV storage must be at least 16-byte aligned";

    int32_t* error_ptr = nullptr;
    if (error_flag.has_value()) {
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(*error_flag);
      error_ptr = static_cast<int32_t*>(error_flag->data_ptr());
    }

    const auto rows = static_cast<uint32_t>(num_tokens.unwrap());
    if (rows == 0) return;
    const auto params = PackedMainKVStoreParams{
        .latent = static_cast<const bf16_t*>(latent.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .slots = slots.data_ptr(),
        .storage = static_cast<uint8_t*>(storage.data_ptr()),
        .error_flag = error_ptr,
    };
    const auto fn_int32 = kernel<int32_t>;
    const auto fn_int64 = kernel<int64_t>;
    const auto fn = slot_dtype.is_type<int32_t>() ? fn_int32 : fn_int64;
    LaunchKernel(rows, kBlockSize, device.unwrap())(fn, params);
  }
};

}  // namespace sglang
