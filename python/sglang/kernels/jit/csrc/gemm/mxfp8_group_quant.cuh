#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::mxfp8_group_quant::ptx {

/// \brief `div.full.f32`: the instruction Triton lowers an fp32 `/` to. Using it
/// here keeps the scale bit-identical to `triton_kernels.downcast_to_mxfp`.
SGL_DEVICE float div_full_f32(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

}  // namespace device::mxfp8_group_quant::ptx

/**
 * \brief MXFP8 (OCP 1x32 block, E8M0 scale) quantization of a row-major tensor.
 *
 * Each 32-element block along the last dim gets the scale
 * 2^ceil(log2(amax / 448)) (ROUND_UP: the block amax never clips) and its values
 * become e4m3 = rn_satfinite(x / scale). Because the input is contiguous and the
 * last dim is a multiple of 32, blocks are consecutive 32-element chunks of the
 * flat buffer, so the kernel walks it as 1-D: one thread owns one 16-byte vector,
 * `kThreadsPerBlock` adjacent lanes own one block.
 *
 * \tparam T       Input type: fp32_t | fp16_t | bf16_t
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 */
template <typename T, bool kUsePDL>
__global__ void mxfp8_group_quant_kernel(
    const T* __restrict__ x, fp8_e4m3_t* __restrict__ q, uint8_t* __restrict__ scale, uint64_t num_vecs) {
  using namespace device;
  namespace ptx = device::mxfp8_group_quant::ptx;

  constexpr uint32_t kBlockElems = 32;
  constexpr uint32_t kVecElems = 16 / sizeof(T);
  constexpr uint32_t kThreadsPerBlock = kBlockElems / kVecElems;
  static_assert(kVecElems % 2 == 0 && kThreadsPerBlock <= kWarpThreads);
  using InVec = AlignedVector<T, kVecElems>;
  using OutVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint64_t vec_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // num_vecs is a multiple of kThreadsPerBlock and blockDim.x a multiple of 32,
  // so a block's lanes are all in range or all out; out-of-range lanes still join
  // the shuffle below instead of exiting early.
  const bool valid = vec_id < num_vecs;

  PDLWaitPrimary<kUsePDL>();

  float vals[kVecElems];
  float local_max = 0.0f;
  if (valid) {
    InVec in;
    in.load(x, vec_id);
#pragma unroll
    for (uint32_t i = 0; i < kVecElems; ++i) {
      vals[i] = static_cast<float>(in[i]);
      local_max = fmaxf(local_max, fabsf(vals[i]));
    }
  }
  const float amax = warp::reduce_max<kThreadsPerBlock>(local_max);

  // Round the dequant scale up to a power of two: adding 0x007FFFFF carries into
  // the exponent unless the mantissa is already zero.
  const float dequant = ptx::div_full_f32(amax, 448.0f);
  const uint32_t scale_bits = (__float_as_uint(dequant) + 0x007FFFFFu) & 0x7F800000u;
  const float dequant_rounded = __uint_as_float(scale_bits);
  const float quant_scale = dequant_rounded == 0.0f ? 0.0f : ptx::div_full_f32(1.0f, dequant_rounded);

  if (valid) {
    OutVec out;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      out[i] = fp8x2_e4m3_t{fp32x2_t{vals[2 * i] * quant_scale, vals[2 * i + 1] * quant_scale}};
    }
    out.store(q, vec_id);
    if (vec_id % kThreadsPerBlock == 0) {
      scale[vec_id / kThreadsPerBlock] = static_cast<uint8_t>(scale_bits >> 23);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate and launch `mxfp8_group_quant_kernel`.
 *
 * \param x     [M, K] contiguous input, K % 32 == 0
 * \param q     [M, K] float8_e4m3fn output
 * \param scale [M, K / 32] uint8 output (biased E8M0 exponents, row-major)
 */
template <typename T, bool kUsePDL>
void mxfp8_group_quant(tvm::ffi::TensorView x, tvm::ffi::TensorView q, tvm::ffi::TensorView scale) {
  using namespace host;

  auto M = SymbolicSize{"num_rows"};
  auto K = SymbolicSize{"hidden"};
  auto KB = SymbolicSize{"num_blocks"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, K})  //
      .with_dtype<T>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x);
  TensorMatcher({M, K})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(q);
  TensorMatcher({M, KB})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(scale);

  const int64_t num_rows = M.unwrap();
  const int64_t hidden = K.unwrap();
  CHECK_HOST(hidden % 32 == 0) << "mxfp8_group_quant: hidden " << hidden << " is not a multiple of 32";
  CHECK_HOST(KB.unwrap() == hidden / 32) << "mxfp8_group_quant: scale has " << KB.unwrap() << " columns, expected "
                                         << hidden / 32;

  const uint64_t num_vecs = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(hidden) / (16 / sizeof(T));
  if (num_vecs == 0) return;

  constexpr uint32_t kBlockSize = 256;
  const uint64_t grid = div_ceil(num_vecs, static_cast<uint64_t>(kBlockSize));
  CHECK_HOST(grid <= 0x7FFFFFFFull) << "mxfp8_group_quant: input too large (" << num_vecs << " vectors)";

  LaunchKernel(static_cast<uint32_t>(grid), kBlockSize, device.unwrap())
      .enable_pdl(kUsePDL)(
          mxfp8_group_quant_kernel<T, kUsePDL>,
          static_cast<const T*>(x.data_ptr()),
          static_cast<fp8_e4m3_t*>(q.data_ptr()),
          static_cast<uint8_t*>(scale.data_ptr()),
          num_vecs);
}

}  // namespace sglang
