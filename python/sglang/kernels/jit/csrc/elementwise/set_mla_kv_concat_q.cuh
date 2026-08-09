// MLA KV-cache write fused with the Q concat, bf16 and fp8 entry points.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::copy_bytes, elect_one_lane, inclusive_sum

#include <cuda/ptx>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

struct SetMlaKVConcatQParams {
  // KV scatter side (byte-typed: dtype-agnostic row copies).
  const void* __restrict__ k_nope;
  const void* __restrict__ k_rope;
  void* __restrict__ kv_buffer;
  const void* __restrict__ loc;
  int64_t stride_nope_bytes;
  int64_t stride_rope_bytes;
  int64_t stride_buffer_bytes;
  uint32_t batch_size;
  // Q concat side (bf16, element strides).
  const bf16_t* __restrict__ q_nope;
  const bf16_t* __restrict__ q_rope;
  bf16_t* __restrict__ q_out;
  uint32_t num_q_items;  // batch_size * num_heads
  uint32_t q_dim_1;      // num_heads
  int64_t qn_stride_0;
  int32_t qn_stride_1;
  int64_t qr_stride_0;
  int32_t qr_stride_1;
  int64_t qo_stride_0;
  int32_t qo_stride_1;
};

template <int64_t kNopeBytes, int64_t kRopeBytes, int kNumWarps, bool kUsePDL, typename TLoc>
__global__ void set_mla_kv_concat_q_kernel(const __grid_constant__ SetMlaKVConcatQParams params) {
  using namespace device;
  static_assert((kNopeBytes + kRopeBytes) % 16 == 0, "TMA bulk store requires total row to be 16-byte aligned");

  constexpr int64_t kRowBytes = kNopeBytes + kRopeBytes;
  constexpr int kQNopeDim = static_cast<int>(kNopeBytes / sizeof(bf16_t));
  constexpr int kQRopeDim = static_cast<int>(kRopeBytes / sizeof(bf16_t));

  // Per-warp smem slots for the KV scatter role; concat warps leave theirs idle.
  __shared__ alignas(16) uint8_t smem[kNumWarps][kRowBytes];

  const uint32_t warp_in_cta = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t flat_warp = blockIdx.x * kNumWarps + warp_in_cta;

  PDLWaitPrimary<kUsePDL>();

  if (flat_warp < params.batch_size) {
    // --- KV scatter role: one warp per token (smem staging + TMA bulk store) ---
    const uint32_t item_id = flat_warp;
    const int64_t loc = static_cast<int64_t>(static_cast<const TLoc*>(params.loc)[item_id]);

    const auto nope_src = pointer::offset(params.k_nope, item_id * params.stride_nope_bytes);
    const auto rope_src = pointer::offset(params.k_rope, item_id * params.stride_rope_bytes);
    void* const gmem_dst = pointer::offset(params.kv_buffer, loc * params.stride_buffer_bytes);

    warp::copy_bytes<kNopeBytes>(nope_src, &smem[warp_in_cta][0]);
    warp::copy_bytes<kRopeBytes>(rope_src, &smem[warp_in_cta][kNopeBytes]);

    // TMA reads smem via the async proxy; fence so it can't observe stale sts.
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // elect.sync rather than `lane_id == 0`: the TMA issue must not sit
    // behind a lane-index predicate (see PR review).
    if (device::warp::elect_one_lane()) {
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          gmem_dst,
          &smem[warp_in_cta][0],
          static_cast<uint32_t>(kRowBytes));
    }

    // ``wait_group`` (not ``_read``): waits for gmem commit, not just smem reuse.
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
  } else if (flat_warp - params.batch_size < params.num_q_items) {
    // --- Q concat role: one warp per (token, head) row ---
    const uint32_t q_item = flat_warp - params.batch_size;
    const uint32_t idx_0 = q_item / params.q_dim_1;
    const uint32_t idx_1 = q_item % params.q_dim_1;

    using ABufType = int4;
    constexpr int kAVecElems = static_cast<int>(sizeof(ABufType) / sizeof(bf16_t));
    constexpr int kANumUnroll = kQNopeDim / (kAVecElems * kWarpThreads);
    static_assert(kANumUnroll * kAVecElems * kWarpThreads == kQNopeDim, "nope dim must fill whole int4 warp rounds");
    using BBufType = int;
    constexpr int kBVecElems = static_cast<int>(sizeof(BBufType) / sizeof(bf16_t));
    static_assert(kBVecElems * kWarpThreads == kQRopeDim, "rope dim must be exactly one int warp round");

    const bf16_t* a_row = params.q_nope + idx_0 * params.qn_stride_0 + idx_1 * params.qn_stride_1;
    const bf16_t* b_row = params.q_rope + idx_0 * params.qr_stride_0 + idx_1 * params.qr_stride_1;
    bf16_t* o_row = params.q_out + idx_0 * params.qo_stride_0 + idx_1 * params.qo_stride_1;

    ABufType a_buf[kANumUnroll];
#pragma unroll
    for (int i = 0; i < kANumUnroll; ++i) {
      a_buf[i] = reinterpret_cast<const ABufType*>(a_row)[i * kWarpThreads + lane_id];
    }
    const BBufType b_buf = reinterpret_cast<const BBufType*>(b_row)[lane_id];

#pragma unroll
    for (int i = 0; i < kANumUnroll; ++i) {
      reinterpret_cast<ABufType*>(o_row)[i * kWarpThreads + lane_id] = a_buf[i];
    }
    reinterpret_cast<BBufType*>(o_row + kQNopeDim)[lane_id] = b_buf;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL>
struct SetMlaKVConcatQKernel {
  static_assert(kNopeBytes > 0 && kNopeBytes % 4 == 0, "kNopeBytes must be a positive multiple of 4");
  static_assert(kRopeBytes > 0 && kRopeBytes % 4 == 0, "kRopeBytes must be a positive multiple of 4");
  static_assert(
      (kNopeBytes + kRopeBytes) % 16 == 0, "TMA bulk store requires (kNopeBytes + kRopeBytes) to be a multiple of 16");

  static constexpr int64_t kQNopeDim = kNopeBytes / static_cast<int64_t>(sizeof(bf16_t));
  static constexpr int64_t kQRopeDim = kRopeBytes / static_cast<int64_t>(sizeof(bf16_t));

  template <int kNumWarps, typename TLoc>
  static constexpr auto kernel = set_mla_kv_concat_q_kernel<kNopeBytes, kRopeBytes, kNumWarps, kUsePDL, TLoc>;

  static void
  run(tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView q_out,
      int64_t num_warps_per_block) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto D_nope = SymbolicSize{"nope_dim"};
    auto D_rope = SymbolicSize{"rope_dim"};
    auto D_buf = SymbolicSize{"buffer_last_dim"};
    auto D_qn = SymbolicSize{"q_nope_dim"};
    auto D_qr = SymbolicSize{"q_rope_dim"};
    auto D_qo = SymbolicSize{"q_out_dim"};
    auto S_nope = SymbolicSize{"nope_stride"};
    auto S_rope = SymbolicSize{"rope_stride"};
    auto S_buf = SymbolicSize{"buffer_stride"};
    auto S_loc = SymbolicSize{"loc_stride"};
    auto S0_qn = SymbolicSize{"q_nope_stride_0"};
    auto S1_qn = SymbolicSize{"q_nope_stride_1"};
    auto S0_qr = SymbolicSize{"q_rope_stride_0"};
    auto S1_qr = SymbolicSize{"q_rope_stride_1"};
    auto S0_qo = SymbolicSize{"q_out_stride_0"};
    auto S1_qo = SymbolicSize{"q_out_stride_1"};
    auto loc_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    D_qn.set_value(kQNopeDim);
    D_qr.set_value(kQRopeDim);
    D_qo.set_value(kQNopeDim + kQRopeDim);

    TensorMatcher({B, D_nope})  //
        .with_strides({S_nope, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_nope);
    TensorMatcher({B, D_rope})  //
        .with_strides({S_rope, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_rope);
    TensorMatcher({-1, D_buf})  //
        .with_strides({S_buf, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(kv_buffer);
    TensorMatcher({B})  //
        .with_strides({S_loc})
        .with_dtype<int32_t, int64_t>(loc_dtype)
        .with_device(device)
        .verify(loc);
    TensorMatcher({B, H, D_qn})  //
        .with_strides({S0_qn, S1_qn, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({B, H, D_qr})  //
        .with_strides({S0_qr, S1_qr, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({B, H, D_qo})  //
        .with_strides({S0_qo, S1_qo, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_out);

    constexpr int64_t kDtypeSize = static_cast<int64_t>(sizeof(bf16_t));
    CHECK_HOST(kNopeBytes == kDtypeSize * D_nope.unwrap())
        << "kNopeBytes mismatch: expected " << kNopeBytes << ", got " << kDtypeSize * D_nope.unwrap();
    CHECK_HOST(kRopeBytes == kDtypeSize * D_rope.unwrap())
        << "kRopeBytes mismatch: expected " << kRopeBytes << ", got " << kDtypeSize * D_rope.unwrap();
    CHECK_HOST(kDtypeSize * D_buf.unwrap() >= kNopeBytes + kRopeBytes) << "kv_buffer last dim too small";
    CHECK_HOST(S_loc.unwrap() == 1) << "loc must be contiguous; got stride " << S_loc.unwrap();

    // Alignment tripwires. The device code does 16-byte vector accesses on the
    // kv row / nope rows / q rows and 4-byte accesses on the rope rows; the
    // python-side ``covered()`` mirrors these so uncovered layouts fall back
    // instead of faulting (do NOT assume "PyTorch tensors are aligned" — views
    // and odd pool pitches break that).
    const auto aligned = [](const void* ptr, int64_t align) {
      return reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(align) == 0;
    };
    CHECK_HOST(aligned(kv_buffer.data_ptr(), 16) && (S_buf.unwrap() * kDtypeSize) % 16 == 0)
        << "kv_buffer base/row-stride must be 16-byte aligned for TMA bulk store";
    CHECK_HOST(aligned(k_nope.data_ptr(), 16) && (S_nope.unwrap() * kDtypeSize) % 16 == 0)
        << "k_nope base/row-stride must be 16-byte aligned";
    CHECK_HOST(aligned(k_rope.data_ptr(), 4) && (S_rope.unwrap() * kDtypeSize) % 4 == 0)
        << "k_rope base/row-stride must be 4-byte aligned";
    CHECK_HOST(
        aligned(q_nope.data_ptr(), 16) && (S0_qn.unwrap() * kDtypeSize) % 16 == 0 &&
        (S1_qn.unwrap() * kDtypeSize) % 16 == 0)
        << "q_nope base/strides must be 16-byte aligned";
    CHECK_HOST(
        aligned(q_rope.data_ptr(), 4) && (S0_qr.unwrap() * kDtypeSize) % 4 == 0 &&
        (S1_qr.unwrap() * kDtypeSize) % 4 == 0)
        << "q_rope base/strides must be 4-byte aligned";
    CHECK_HOST(
        aligned(q_out.data_ptr(), 16) && (S0_qo.unwrap() * kDtypeSize) % 16 == 0 &&
        (S1_qo.unwrap() * kDtypeSize) % 16 == 0)
        << "q_out base/strides must be 16-byte aligned";

    const uint32_t batch = static_cast<uint32_t>(B.unwrap());
    const uint32_t num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch == 0) return;

    const auto params = SetMlaKVConcatQParams{
        .k_nope = k_nope.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .kv_buffer = kv_buffer.data_ptr(),
        .loc = loc.data_ptr(),
        .stride_nope_bytes = S_nope.unwrap() * kDtypeSize,
        .stride_rope_bytes = S_rope.unwrap() * kDtypeSize,
        .stride_buffer_bytes = S_buf.unwrap() * kDtypeSize,
        .batch_size = batch,
        .q_nope = static_cast<const bf16_t*>(q_nope.data_ptr()),
        .q_rope = static_cast<const bf16_t*>(q_rope.data_ptr()),
        .q_out = static_cast<bf16_t*>(q_out.data_ptr()),
        .num_q_items = batch * num_heads,
        .q_dim_1 = num_heads,
        .qn_stride_0 = S0_qn.unwrap(),
        .qn_stride_1 = static_cast<int32_t>(S1_qn.unwrap()),
        .qr_stride_0 = S0_qr.unwrap(),
        .qr_stride_1 = static_cast<int32_t>(S1_qr.unwrap()),
        .qo_stride_0 = S0_qo.unwrap(),
        .qo_stride_1 = static_cast<int32_t>(S1_qo.unwrap()),
    };

    const auto use_int32 = loc_dtype.is_type<int32_t>();
    const uint32_t total_warps = params.batch_size + params.num_q_items;

    auto launch = [&]<int kNW>() {
      const auto kernel_ptr = use_int32 ? kernel<kNW, int32_t> : kernel<kNW, int64_t>;
      const uint32_t num_blocks = div_ceil(total_warps, static_cast<uint32_t>(kNW));
      const uint32_t threads_per_block = static_cast<uint32_t>(kNW) * device::kWarpThreads;
      LaunchKernel(num_blocks, threads_per_block, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel_ptr, params);
    };

    switch (num_warps_per_block) {
      case 1:
        launch.template operator()<1>();
        break;
      case 2:
        launch.template operator()<2>();
        break;
      case 4:
        launch.template operator()<4>();
        break;
      case 8:
        launch.template operator()<8>();
        break;
      default:
        Panic("Unsupported num_warps_per_block=", num_warps_per_block);
    }
  }
};

// ---------------------------------------------------------------------------
// fp8 variant. Shares the translation unit, not the kernel: dims are runtime
// rather than template parameters, it shards DCP slots (vloc % world != rank),
// converts per lane instead of bulk-copying, and counts strides in elements.
// Only the module that instantiates it pays for it.
// ---------------------------------------------------------------------------
constexpr int kFp8NopeDim = 512;
constexpr int kFp8RopeDim = 64;
constexpr int kFp8RowBytes = kFp8NopeDim + kFp8RopeDim;  // fp8: 1 byte/elem

struct SetMlaKVConcatQFp8Params {
  // KV quantize + scatter side.
  const bf16_t* __restrict__ k_nope;
  const bf16_t* __restrict__ k_rope;
  uint8_t* __restrict__ kv_buffer;
  const void* __restrict__ loc;
  int64_t stride_nope;          // elements
  int64_t stride_rope;          // elements
  int64_t stride_buffer_bytes;  // bytes
  uint32_t batch_size;
  // DCP cyclic sharding of the KV pool: ``loc`` is VIRTUAL; the physical
  // row on the owner rank is loc / world, and only the owner
  // (loc % world == rank) writes. world=1/rank=0 = identity (non-DCP).
  int32_t dcp_world_size;
  int32_t dcp_rank;
  // Q quantize + concat side.
  const bf16_t* __restrict__ q_nope;
  const bf16_t* __restrict__ q_rope;
  uint8_t* __restrict__ q_out;
  uint32_t num_q_items;  // batch_size * num_heads
  uint32_t q_dim_1;      // num_heads
  int64_t qn_stride_0;
  int32_t qn_stride_1;
  int64_t qr_stride_0;
  int32_t qr_stride_1;
  int64_t qo_stride_0;  // elements (== bytes for fp8)
  int32_t qo_stride_1;
};

// 2x bf16 -> 2x fp8 e4m3, float-mediated cvt.rn NOSAT (matches aten: overflow -> NaN).
SGL_DEVICE uint16_t bf16x2_to_fp8x2(const bf16x2_t v) {
  const float2 f = __bfloat1622float2(v);
  return __nv_cvt_float2_to_fp8x2(f, __NV_NOSAT, __NV_E4M3);
}

// Convert 8 bf16 (one int4 load) to 8 fp8 packed in a uint2.
SGL_DEVICE uint2 bf16x8_to_fp8x8(const int4 v) {
  const bf16x2_t* p = reinterpret_cast<const bf16x2_t*>(&v);
  uint2 out;
  out.x = static_cast<uint32_t>(bf16x2_to_fp8x2(p[0])) | (static_cast<uint32_t>(bf16x2_to_fp8x2(p[1])) << 16);
  out.y = static_cast<uint32_t>(bf16x2_to_fp8x2(p[2])) | (static_cast<uint32_t>(bf16x2_to_fp8x2(p[3])) << 16);
  return out;
}

template <int kNumWarps, bool kUsePDL, typename TLoc>
__global__ void set_mla_kv_concat_q_fp8_kernel(const __grid_constant__ SetMlaKVConcatQFp8Params params) {
  using namespace device;

  // Per-warp smem slots for the KV role (fp8 rows); concat warps leave
  // theirs idle. 576 % 16 == 0 satisfies the TMA bulk-store requirement.
  __shared__ alignas(16) uint8_t smem[kNumWarps][kFp8RowBytes];

  const uint32_t warp_in_cta = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t flat_warp = blockIdx.x * kNumWarps + warp_in_cta;

  PDLWaitPrimary<kUsePDL>();

  if (flat_warp < params.batch_size) {
    // --- KV role: quantize one token's row into smem, TMA-scatter it ---
    const uint32_t item_id = flat_warp;
    const int64_t vloc = static_cast<int64_t>(static_cast<const TLoc*>(params.loc)[item_id]);
    // DCP ownership: non-owner ranks write nothing for this token (mirrors
    // the triton writer's is_valid mask + loc // world translation).
    if (vloc % params.dcp_world_size != params.dcp_rank) {
      PDLTriggerSecondary<kUsePDL>();
      return;
    }
    const int64_t loc = vloc / params.dcp_world_size;
    const bf16_t* nope_src = params.k_nope + item_id * params.stride_nope;
    const bf16_t* rope_src = params.k_rope + item_id * params.stride_rope;

    // nope: 512 bf16 -> 512 fp8; 16 elems/lane (2 int4 loads -> 1 int4 store).
    {
      const int4* src = reinterpret_cast<const int4*>(nope_src);
      uint2 lo = bf16x8_to_fp8x8(src[lane_id * 2]);
      uint2 hi = bf16x8_to_fp8x8(src[lane_id * 2 + 1]);
      reinterpret_cast<int4*>(&smem[warp_in_cta][0])[lane_id] =
          make_int4(static_cast<int>(lo.x), static_cast<int>(lo.y), static_cast<int>(hi.x), static_cast<int>(hi.y));
    }
    // rope: 64 bf16 -> 64 fp8; 2 elems/lane.
    {
      const bf16x2_t v = reinterpret_cast<const bf16x2_t*>(rope_src)[lane_id];
      reinterpret_cast<uint16_t*>(&smem[warp_in_cta][kFp8NopeDim])[lane_id] = bf16x2_to_fp8x2(v);
    }

    // TMA reads smem via the async proxy; fence so it can't observe stale sts.
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // elect.sync rather than `lane_id == 0`: the TMA issue must not sit
    // behind a lane-index predicate (same review point as the bf16 variant).
    if (device::warp::elect_one_lane()) {
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          params.kv_buffer + loc * params.stride_buffer_bytes,
          &smem[warp_in_cta][0],
          static_cast<uint32_t>(kFp8RowBytes));
    }
    // ``wait_group`` (not ``_read``): waits for gmem commit, not just smem reuse.
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
  } else if (flat_warp - params.batch_size < params.num_q_items) {
    // --- Q role: quantize one (token, head) row into the fp8 query ---
    const uint32_t q_item = flat_warp - params.batch_size;
    const uint32_t idx_0 = q_item / params.q_dim_1;
    const uint32_t idx_1 = q_item % params.q_dim_1;
    const bf16_t* a_row = params.q_nope + idx_0 * params.qn_stride_0 + idx_1 * params.qn_stride_1;
    const bf16_t* b_row = params.q_rope + idx_0 * params.qr_stride_0 + idx_1 * params.qr_stride_1;
    uint8_t* o_row = params.q_out + idx_0 * params.qo_stride_0 + idx_1 * params.qo_stride_1;

    {
      const int4* src = reinterpret_cast<const int4*>(a_row);
      uint2 lo = bf16x8_to_fp8x8(src[lane_id * 2]);
      uint2 hi = bf16x8_to_fp8x8(src[lane_id * 2 + 1]);
      reinterpret_cast<int4*>(o_row)[lane_id] =
          make_int4(static_cast<int>(lo.x), static_cast<int>(lo.y), static_cast<int>(hi.x), static_cast<int>(hi.y));
    }
    {
      const bf16x2_t v = reinterpret_cast<const bf16x2_t*>(b_row)[lane_id];
      reinterpret_cast<uint16_t*>(o_row + kFp8NopeDim)[lane_id] = bf16x2_to_fp8x2(v);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct SetMlaKVConcatQFp8Kernel {
  template <int kNumWarps, typename TLoc>
  static constexpr auto kernel = set_mla_kv_concat_q_fp8_kernel<kNumWarps, kUsePDL, TLoc>;

  static void
  run(tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView q_out,
      int64_t num_warps_per_block,
      int64_t dcp_world_size,
      int64_t dcp_rank) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto D_nope = SymbolicSize{"nope_dim"};
    auto D_rope = SymbolicSize{"rope_dim"};
    auto D_buf = SymbolicSize{"buffer_last_dim"};
    auto D_qn = SymbolicSize{"q_nope_dim"};
    auto D_qr = SymbolicSize{"q_rope_dim"};
    auto D_qo = SymbolicSize{"q_out_dim"};
    auto S_nope = SymbolicSize{"nope_stride"};
    auto S_rope = SymbolicSize{"rope_stride"};
    auto S_buf = SymbolicSize{"buffer_stride"};
    auto S_loc = SymbolicSize{"loc_stride"};
    auto S0_qn = SymbolicSize{"q_nope_stride_0"};
    auto S1_qn = SymbolicSize{"q_nope_stride_1"};
    auto S0_qr = SymbolicSize{"q_rope_stride_0"};
    auto S1_qr = SymbolicSize{"q_rope_stride_1"};
    auto S0_qo = SymbolicSize{"q_out_stride_0"};
    auto S1_qo = SymbolicSize{"q_out_stride_1"};
    auto loc_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    D_nope.set_value(kFp8NopeDim);
    D_rope.set_value(kFp8RopeDim);
    D_qn.set_value(kFp8NopeDim);
    D_qr.set_value(kFp8RopeDim);
    D_qo.set_value(kFp8RowBytes);

    TensorMatcher({B, D_nope})  //
        .with_strides({S_nope, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_nope);
    TensorMatcher({B, D_rope})  //
        .with_strides({S_rope, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_rope);
    TensorMatcher({-1, D_buf})  //
        .with_strides({S_buf, 1})
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .verify(kv_buffer);
    TensorMatcher({B})  //
        .with_strides({S_loc})
        .with_dtype<int32_t, int64_t>(loc_dtype)
        .with_device(device)
        .verify(loc);
    TensorMatcher({B, H, D_qn})  //
        .with_strides({S0_qn, S1_qn, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({B, H, D_qr})  //
        .with_strides({S0_qr, S1_qr, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({B, H, D_qo})  //
        .with_strides({S0_qo, S1_qo, 1})
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .verify(q_out);

    CHECK_HOST(D_buf.unwrap() >= kFp8RowBytes) << "kv_buffer last dim too small";
    CHECK_HOST(dcp_world_size >= 1 && dcp_rank >= 0 && dcp_rank < dcp_world_size)
        << "invalid dcp world/rank: " << dcp_world_size << "/" << dcp_rank;
    CHECK_HOST(S_loc.unwrap() == 1) << "loc must be contiguous; got stride " << S_loc.unwrap();

    // Alignment tripwires (mirrored by python covered() so uncovered layouts
    // fall back instead of faulting): 16B vector loads on the bf16 nope/q
    // rows, 4B on the rope rows, 16B TMA dst rows, 16B int4 stores on q_out.
    const auto aligned = [](const void* ptr, int64_t align) {
      return reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(align) == 0;
    };
    CHECK_HOST(aligned(kv_buffer.data_ptr(), 16) && S_buf.unwrap() % 16 == 0)
        << "kv_buffer base/row-stride must be 16-byte aligned for TMA bulk store";
    CHECK_HOST(aligned(k_nope.data_ptr(), 16) && (S_nope.unwrap() * 2) % 16 == 0)
        << "k_nope base/row-stride must be 16-byte aligned";
    CHECK_HOST(aligned(k_rope.data_ptr(), 4) && (S_rope.unwrap() * 2) % 4 == 0)
        << "k_rope base/row-stride must be 4-byte aligned";
    CHECK_HOST(aligned(q_nope.data_ptr(), 16) && (S0_qn.unwrap() * 2) % 16 == 0 && (S1_qn.unwrap() * 2) % 16 == 0)
        << "q_nope base/strides must be 16-byte aligned";
    CHECK_HOST(aligned(q_rope.data_ptr(), 4) && (S0_qr.unwrap() * 2) % 4 == 0 && (S1_qr.unwrap() * 2) % 4 == 0)
        << "q_rope base/strides must be 4-byte aligned";
    CHECK_HOST(aligned(q_out.data_ptr(), 16) && S0_qo.unwrap() % 16 == 0 && S1_qo.unwrap() % 16 == 0)
        << "q_out base/strides must be 16-byte aligned";

    const uint32_t batch = static_cast<uint32_t>(B.unwrap());
    const uint32_t num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch == 0) return;

    const auto params = SetMlaKVConcatQFp8Params{
        .k_nope = static_cast<const bf16_t*>(k_nope.data_ptr()),
        .k_rope = static_cast<const bf16_t*>(k_rope.data_ptr()),
        .kv_buffer = static_cast<uint8_t*>(kv_buffer.data_ptr()),
        .loc = loc.data_ptr(),
        .stride_nope = S_nope.unwrap(),
        .stride_rope = S_rope.unwrap(),
        .stride_buffer_bytes = S_buf.unwrap(),
        .batch_size = batch,
        .dcp_world_size = static_cast<int32_t>(dcp_world_size),
        .dcp_rank = static_cast<int32_t>(dcp_rank),
        .q_nope = static_cast<const bf16_t*>(q_nope.data_ptr()),
        .q_rope = static_cast<const bf16_t*>(q_rope.data_ptr()),
        .q_out = static_cast<uint8_t*>(q_out.data_ptr()),
        .num_q_items = batch * num_heads,
        .q_dim_1 = num_heads,
        .qn_stride_0 = S0_qn.unwrap(),
        .qn_stride_1 = static_cast<int32_t>(S1_qn.unwrap()),
        .qr_stride_0 = S0_qr.unwrap(),
        .qr_stride_1 = static_cast<int32_t>(S1_qr.unwrap()),
        .qo_stride_0 = S0_qo.unwrap(),
        .qo_stride_1 = static_cast<int32_t>(S1_qo.unwrap()),
    };

    const auto use_int32 = loc_dtype.is_type<int32_t>();
    const uint32_t total_warps = params.batch_size + params.num_q_items;

    auto launch = [&]<int kNW>() {
      const auto kernel_ptr = use_int32 ? kernel<kNW, int32_t> : kernel<kNW, int64_t>;
      const uint32_t num_blocks = div_ceil(total_warps, static_cast<uint32_t>(kNW));
      LaunchKernel(num_blocks, static_cast<uint32_t>(kNW) * device::kWarpThreads, device.unwrap())
          .enable_pdl(kUsePDL)(kernel_ptr, params);
    };

    switch (num_warps_per_block) {
      case 1:
        launch.template operator()<1>();
        break;
      case 2:
        launch.template operator()<2>();
        break;
      case 4:
        launch.template operator()<4>();
        break;
      case 8:
        launch.template operator()<8>();
        break;
      default:
        Panic("Unsupported num_warps_per_block=", num_warps_per_block);
    }
  }
};

}  // namespace sglang
