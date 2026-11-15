#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::mxfp4_dequant {

/// \brief Decode one e2m1 code (low 4 bits of `code`) to fp32.
///
/// Magnitudes 0..7 are {0, 0.5, 1, 1.5, 2, 3, 4, 6}: codes 0/1 are the subnormals
/// 0 and 0.5, codes 2..7 have exponent (code >> 1) - 1 and one mantissa bit. Bit 3
/// is the sign (code 8 is -0). Built from bits rather than a lookup table so lanes
/// reading different codes do not serialize on constant memory.
SGL_DEVICE float e2m1_to_float(uint32_t code) {
  const uint32_t mag = code & 0x7u;
  const uint32_t mag_bits =
      mag < 2 ? __float_as_uint(0.5f * static_cast<float>(mag)) : (((mag >> 1) + 126u) << 23) | ((mag & 1u) << 22);
  return __uint_as_float(mag_bits | ((code & 0x8u) << 28));
}

}  // namespace device::mxfp4_dequant

/**
 * \brief Dequantize MXFP4 (packed e2m1 pairs + one E8M0 scale per 32 values) to bf16.
 *
 * One thread owns one 32-value block: 16 packed bytes, one scale byte, 64 output
 * bytes. Byte j holds value 2j in its low nibble and value 2j+1 in its high nibble.
 * The result matches `triton_kernels.upcast_from_mxfp(..., bfloat16, axis=-1)`:
 * value * 2^(scale - 127), scale byte 0 -> 0, clamped to the bf16 finite range,
 * and NaN where the scale byte is 0xFF.
 */
template <bool kUsePDL>
__global__ void mxfp4_dequant_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scale,
    bf16_t* __restrict__ out,
    uint64_t num_blocks) {
  using namespace device;
  using InVec = AlignedVector<uint8_t, 16>;
  using OutVec = AlignedVector<bf16x2_t, 4>;  // 8 values, 16 bytes
  constexpr float kBf16MaxFinite = 3.3895313892515355e38f;

  const uint64_t block_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks) return;

  PDLWaitPrimary<kUsePDL>();

  InVec in;
  in.load(packed, block_id);
  const uint32_t s = scale[block_id];
  const float block_scale = __uint_as_float(s << 23);  // s == 0 gives +0

#pragma unroll
  for (uint32_t chunk = 0; chunk < 4; ++chunk) {
    OutVec o;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t byte = in[chunk * 4 + i];
      float lo = mxfp4_dequant::e2m1_to_float(byte) * block_scale;
      float hi = mxfp4_dequant::e2m1_to_float(byte >> 4) * block_scale;
      lo = fminf(fmaxf(lo, -kBf16MaxFinite), kBf16MaxFinite);
      hi = fminf(fmaxf(hi, -kBf16MaxFinite), kBf16MaxFinite);
      if (s == 0xFFu) {
        lo = hi = __uint_as_float(0x7FC00000u);
      }
      o[i] = cast<bf16x2_t>(fp32x2_t{lo, hi});
    }
    o.store(out + block_id * 32, chunk);
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate and launch `mxfp4_dequant_kernel`.
 *
 * \param packed [M, K / 2] uint8, contiguous
 * \param scale  [M, K / 32] uint8, contiguous
 * \param out    [M, K] bf16, contiguous
 */
template <bool kUsePDL>
void mxfp4_dequant(tvm::ffi::TensorView packed, tvm::ffi::TensorView scale, tvm::ffi::TensorView out) {
  using namespace host;

  auto M = SymbolicSize{"num_rows"};
  auto KP = SymbolicSize{"packed_cols"};
  auto KB = SymbolicSize{"num_blocks"};
  auto K = SymbolicSize{"hidden"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, KP})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(packed);
  TensorMatcher({M, KB})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(scale);
  TensorMatcher({M, K})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(out);

  const int64_t num_rows = M.unwrap();
  const int64_t hidden = K.unwrap();
  CHECK_HOST(hidden % 32 == 0) << "mxfp4_dequant: hidden " << hidden << " is not a multiple of 32";
  CHECK_HOST(KP.unwrap() * 2 == hidden) << "mxfp4_dequant: packed has " << KP.unwrap() << " columns, expected "
                                        << hidden / 2;
  CHECK_HOST(KB.unwrap() * 32 == hidden) << "mxfp4_dequant: scale has " << KB.unwrap() << " columns, expected "
                                         << hidden / 32;

  const uint64_t num_blocks = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(hidden) / 32;
  if (num_blocks == 0) return;

  constexpr uint32_t kBlockSize = 256;
  const uint64_t grid = div_ceil(num_blocks, static_cast<uint64_t>(kBlockSize));
  CHECK_HOST(grid <= 0x7FFFFFFFull) << "mxfp4_dequant: input too large (" << num_blocks << " blocks)";

  LaunchKernel(static_cast<uint32_t>(grid), kBlockSize, device.unwrap())
      .enable_pdl(kUsePDL)(
          mxfp4_dequant_kernel<kUsePDL>,
          static_cast<const uint8_t*>(packed.data_ptr()),
          static_cast<const uint8_t*>(scale.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          num_blocks);
}

}  // namespace sglang
