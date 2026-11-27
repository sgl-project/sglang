// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <limits>

#if !defined(USE_ROCM) && defined(SGL_CUDA_ARCH) && (SGL_CUDA_ARCH < 1200 || SGL_CUDA_ARCH >= 1300)
#error "moe_lowm_sm120.cuh requires SM12x (compute capability 12.x)"
#endif

namespace sglang::inkling2_moe_lowm_sm120 {

namespace {

constexpr int32_t kMaxExperts = 256;

SGL_DEVICE uint4 ld_w_mxfp8(void const* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

SGL_DEVICE uint32_t ld_sf_nc(void const* ptr) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

SGL_DEVICE void transpose_4x4(uint4 v, int32_t tid_g, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  int32_t const odd = tid_g & 1;
  uint32_t const p0 = __shfl_xor_sync(0xFFFFFFFFu, odd ? v.x : v.y, 1);
  uint32_t const p1 = __shfl_xor_sync(0xFFFFFFFFu, odd ? v.z : v.w, 1);
  uint32_t const u0 = odd ? p0 : v.x;
  uint32_t const u1 = odd ? v.y : p0;
  uint32_t const u2 = odd ? p1 : v.z;
  uint32_t const u3 = odd ? v.w : p1;
  int32_t const hi = (tid_g >> 1) & 1;
  uint32_t const q0 = __shfl_xor_sync(0xFFFFFFFFu, hi ? u0 : u2, 2);
  uint32_t const q1 = __shfl_xor_sync(0xFFFFFFFFu, hi ? u1 : u3, 2);
  r0 = hi ? q0 : u0;
  r1 = hi ? q1 : u1;
  r2 = hi ? u2 : q0;
  r3 = hi ? u3 : q1;
}

SGL_DEVICE void mxfp8_mma_m16n8k32(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t sfa,
    uint32_t sfb,
    uint16_t bid) {
#if defined(__CUDA_ARCH__)
  uint16_t const zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0),
        "r"(a1),
        "r"(a2),
        "r"(a3),
        "r"(b0),
        "r"(b1),
        "r"(sfa),
        "h"(bid),
        "h"(zero),
        "r"(sfb),
        "h"(bid),
        "h"(zero));
#endif
}

template <int32_t kRowShift>
SGL_DEVICE void build_ntile_prefix(int32_t const* s_indptr, int32_t* s_ntile_prefix, int32_t num_groups, int32_t lane) {
  int32_t run = 0;
  int32_t const groups32 = (num_groups + 31) >> 5;
  constexpr int32_t kMask = (1 << kRowShift) - 1;
  for (int32_t g = 0; g < groups32; ++g) {
    int32_t const e = g * 32 + lane;
    int32_t const cnt = (e < num_groups) ? (s_indptr[e + 1] - s_indptr[e]) : 0;
    int32_t const nt = (cnt + kMask) >> kRowShift;
    int32_t x = nt;
#pragma unroll
    for (int32_t s = 1; s < 32; s <<= 1) {
      int32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, s);
      if (lane >= s) x += y;
    }
    if (e < num_groups) s_ntile_prefix[e] = run + (x - nt);
    run += __shfl_sync(0xFFFFFFFFu, x, 31);
  }
  if (lane == 0) s_ntile_prefix[num_groups] = run;
}

}  // namespace

__global__ void __launch_bounds__(128, 2) mxfp8_gemm_lowm_kernel(
    uint8_t const* __restrict__ A_acts,
    uint8_t const* __restrict__ B_weights,
    uint32_t const* __restrict__ SFA_cute_u32,
    uint32_t const* __restrict__ SFW_raw_u32,
    bf16_t* __restrict__ D_out,
    int32_t const* __restrict__ m_indptr,
    int32_t n,
    int32_t k,
    int32_t num_groups,
    int32_t m_padded) {
  constexpr int32_t kWarps = 4;
  constexpr int32_t kTileM = kWarps * 16;
  constexpr int32_t kMaxRows = 32;
  constexpr int32_t kMaxN8 = kMaxRows / 8;
  constexpr int32_t kUnrollK128 = 3;

  __shared__ int32_t s_indptr[kMaxExperts + 1];
  __shared__ int32_t s_ntile_prefix[kMaxExperts + 1];
  __shared__ alignas(16) bf16_t s_out[kMaxRows][kTileM + 8];

  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const warp = tid >> 5;
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const grp = lane >> 2;
  int32_t const tid_g = lane & 3;
  bool const load_sfa = (tid_g < 2);

  device::PDLWaitPrimary<true>();

  for (int32_t i = tid; i <= num_groups; i += kWarps * 32) {
    s_indptr[i] = m_indptr[i];
  }
  __syncthreads();

  if (warp == 0) {
    build_ntile_prefix<5>(s_indptr, s_ntile_prefix, num_groups, lane);
  }
  __syncthreads();

  int32_t const total_ntiles = s_ntile_prefix[num_groups];
  int32_t const m_tiles_per_n = n / kTileM;
  int32_t const total_tiles = total_ntiles * m_tiles_per_n;
  int32_t const num_k128 = k >> 7;

  for (int32_t tile_id = static_cast<int32_t>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int32_t>(gridDim.x)) {
    int32_t const ntile_global = tile_id / m_tiles_per_n;
    int32_t const m_tile = tile_id - ntile_global * m_tiles_per_n;

    int32_t lo = 0, hi = num_groups;
    while (hi - lo > 1) {
      int32_t const mid = (lo + hi) >> 1;
      if (s_ntile_prefix[mid] <= ntile_global)
        lo = mid;
      else
        hi = mid;
    }
    int32_t const e = lo;
    int32_t const local_nt = ntile_global - s_ntile_prefix[e];
    int32_t const m_off_e = s_indptr[e];
    int32_t const cnt_e = s_indptr[e + 1] - m_off_e;
    int32_t const row_start_local = local_nt * kMaxRows;
    int32_t const rows_this_tile = min(kMaxRows, cnt_e - row_start_local);
    int32_t const row_start_global = m_off_e + row_start_local;
    int32_t const m_base = m_tile * kTileM + warp * 16;

    uint8_t const* W_e = B_weights + static_cast<int64_t>(e) * n * k;
    uint8_t const* W_row0 = W_e + static_cast<int64_t>(m_base + grp) * k + tid_g * 16;
    uint8_t const* W_row1 = W_e + static_cast<int64_t>(m_base + grp + 8) * k + tid_g * 16;
    int32_t const sf_r = m_base + grp + (lane & 1) * 8;
    uint32_t const* SFW_row = SFW_raw_u32 + (static_cast<int64_t>(e) * n + sf_r) * num_k128;

    uint8_t const* B_tile_base = A_acts + static_cast<int64_t>(row_start_global) * k + tid_g * 16;
    int32_t const scale_row_base = ((m_off_e + 3 * e) / 4) * 4 + row_start_local + grp;

    float acc[kMaxN8][4];
#pragma unroll
    for (int32_t nb = 0; nb < kMaxN8; ++nb) {
      acc[nb][0] = 0.0f;
      acc[nb][1] = 0.0f;
      acc[nb][2] = 0.0f;
      acc[nb][3] = 0.0f;
    }

    int32_t ks = 0;
    for (; ks + kUnrollK128 <= num_k128; ks += kUnrollK128) {
      uint4 w0_lo[kUnrollK128], w0_hi[kUnrollK128], w1_lo[kUnrollK128], w1_hi[kUnrollK128];
      uint32_t sfa_buf[kUnrollK128];

#pragma unroll
      for (int32_t u = 0; u < kUnrollK128; ++u) {
        int32_t const kb = (ks + u) * 128;
        int32_t const blk = ks + u;
        w0_lo[u] = ld_w_mxfp8(W_row0 + kb);
        w0_hi[u] = ld_w_mxfp8(W_row0 + kb + 64);
        w1_lo[u] = ld_w_mxfp8(W_row1 + kb);
        w1_hi[u] = ld_w_mxfp8(W_row1 + kb + 64);
        sfa_buf[u] = load_sfa ? ld_sf_nc(SFW_row + blk) : 0u;
      }

#pragma unroll
      for (int32_t u = 0; u < kUnrollK128; ++u) {
        int32_t const kb = (ks + u) * 128;
        int32_t const blk = ks + u;
        uint32_t const sfa = __shfl_sync(0xFFFFFFFFu, sfa_buf[u], (lane & ~2));
        uint32_t a0_0, a2_0, a0_1, a2_1, a1_0, a3_0, a1_1, a3_1;
        uint32_t a0_2, a2_2, a0_3, a2_3, a1_2, a3_2, a1_3, a3_3;
        transpose_4x4(w0_lo[u], tid_g, a0_0, a2_0, a0_1, a2_1);
        transpose_4x4(w1_lo[u], tid_g, a1_0, a3_0, a1_1, a3_1);
        transpose_4x4(w0_hi[u], tid_g, a0_2, a2_2, a0_3, a2_3);
        transpose_4x4(w1_hi[u], tid_g, a1_2, a3_2, a1_3, a3_3);
        uint32_t const* sfb_blk = SFA_cute_u32 + static_cast<int64_t>(blk) * m_padded + scale_row_base;
#pragma unroll
        for (int32_t nb = 0; nb < kMaxN8; ++nb) {
          if (nb == 0 || nb * 8 < rows_this_tile) {
            bool const valid = (nb * 8 + grp < rows_this_tile);
            uint8_t const* bptr = B_tile_base + static_cast<int64_t>(nb * 8 + grp) * k + kb;
            uint4 const blo = valid ? *reinterpret_cast<uint4 const*>(bptr) : make_uint4(0, 0, 0, 0);
            uint4 const bhi = valid ? *reinterpret_cast<uint4 const*>(bptr + 64) : make_uint4(0, 0, 0, 0);
            uint32_t const sfb = valid ? ld_sf_nc(sfb_blk + nb * 8) : 0u;
            uint32_t b0_0, b1_0, b0_1, b1_1;
            transpose_4x4(blo, tid_g, b0_0, b1_0, b0_1, b1_1);
            mxfp8_mma_m16n8k32(
                acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_0, a1_0, a2_0, a3_0, b0_0, b1_0, sfa, sfb, 0);
            mxfp8_mma_m16n8k32(
                acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_1, a1_1, a2_1, a3_1, b0_1, b1_1, sfa, sfb, 1);
            transpose_4x4(bhi, tid_g, b0_0, b1_0, b0_1, b1_1);
            mxfp8_mma_m16n8k32(
                acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_2, a1_2, a2_2, a3_2, b0_0, b1_0, sfa, sfb, 2);
            mxfp8_mma_m16n8k32(
                acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_3, a1_3, a2_3, a3_3, b0_1, b1_1, sfa, sfb, 3);
          }
        }
      }
    }
    for (; ks < num_k128; ++ks) {
      int32_t const kb = ks * 128;
      uint4 const w0_lo = ld_w_mxfp8(W_row0 + kb);
      uint4 const w0_hi = ld_w_mxfp8(W_row0 + kb + 64);
      uint4 const w1_lo = ld_w_mxfp8(W_row1 + kb);
      uint4 const w1_hi = ld_w_mxfp8(W_row1 + kb + 64);
      uint32_t const sfa_r = load_sfa ? ld_sf_nc(SFW_row + ks) : 0u;
      uint32_t const sfa = __shfl_sync(0xFFFFFFFFu, sfa_r, (lane & ~2));
      uint32_t a0_0, a2_0, a0_1, a2_1, a1_0, a3_0, a1_1, a3_1;
      uint32_t a0_2, a2_2, a0_3, a2_3, a1_2, a3_2, a1_3, a3_3;
      transpose_4x4(w0_lo, tid_g, a0_0, a2_0, a0_1, a2_1);
      transpose_4x4(w1_lo, tid_g, a1_0, a3_0, a1_1, a3_1);
      transpose_4x4(w0_hi, tid_g, a0_2, a2_2, a0_3, a2_3);
      transpose_4x4(w1_hi, tid_g, a1_2, a3_2, a1_3, a3_3);
      uint32_t const* sfb_blk = SFA_cute_u32 + static_cast<int64_t>(ks) * m_padded + scale_row_base;
#pragma unroll
      for (int32_t nb = 0; nb < kMaxN8; ++nb) {
        if (nb == 0 || nb * 8 < rows_this_tile) {
          bool const valid = (nb * 8 + grp < rows_this_tile);
          uint8_t const* bptr = B_tile_base + static_cast<int64_t>(nb * 8 + grp) * k + kb;
          uint4 const blo = valid ? *reinterpret_cast<uint4 const*>(bptr) : make_uint4(0, 0, 0, 0);
          uint4 const bhi = valid ? *reinterpret_cast<uint4 const*>(bptr + 64) : make_uint4(0, 0, 0, 0);
          uint32_t const sfb = valid ? ld_sf_nc(sfb_blk + nb * 8) : 0u;
          uint32_t b0_0, b1_0, b0_1, b1_1;
          transpose_4x4(blo, tid_g, b0_0, b1_0, b0_1, b1_1);
          mxfp8_mma_m16n8k32(
              acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_0, a1_0, a2_0, a3_0, b0_0, b1_0, sfa, sfb, 0);
          mxfp8_mma_m16n8k32(
              acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_1, a1_1, a2_1, a3_1, b0_1, b1_1, sfa, sfb, 1);
          transpose_4x4(bhi, tid_g, b0_0, b1_0, b0_1, b1_1);
          mxfp8_mma_m16n8k32(
              acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_2, a1_2, a2_2, a3_2, b0_0, b1_0, sfa, sfb, 2);
          mxfp8_mma_m16n8k32(
              acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3], a0_3, a1_3, a2_3, a3_3, b0_1, b1_1, sfa, sfb, 3);
        }
      }
    }

    int32_t const n0 = tid_g * 2;
    int32_t const n1 = tid_g * 2 + 1;
#pragma unroll
    for (int32_t nb = 0; nb < kMaxN8; ++nb) {
      if (nb * 8 < rows_this_tile) {
        int32_t const r0 = nb * 8 + n0;
        int32_t const r1 = nb * 8 + n1;
        if (r0 < rows_this_tile) {
          s_out[r0][warp * 16 + grp] = __float2bfloat16_rn(acc[nb][0]);
          s_out[r0][warp * 16 + grp + 8] = __float2bfloat16_rn(acc[nb][2]);
        }
        if (r1 < rows_this_tile) {
          s_out[r1][warp * 16 + grp] = __float2bfloat16_rn(acc[nb][1]);
          s_out[r1][warp * 16 + grp + 8] = __float2bfloat16_rn(acc[nb][3]);
        }
      }
    }
    __syncthreads();

    constexpr int32_t kU4PerRow = kTileM / 8;
    int32_t const total_u4 = rows_this_tile * kU4PerRow;
    for (int32_t idx = tid; idx < total_u4; idx += kWarps * 32) {
      int32_t const r = idx / kU4PerRow;
      int32_t const c = (idx - r * kU4PerRow) * 8;
      *reinterpret_cast<uint4*>(D_out + static_cast<int64_t>(row_start_global + r) * n + m_tile * kTileM + c) =
          *reinterpret_cast<uint4 const*>(&s_out[r][c]);
    }
    __syncthreads();
  }
  device::PDLTriggerSecondary<true>();
}

template <int32_t kTopK>
__global__ void moe_combine_bf16_kernel(
    bf16_t const* __restrict__ down_bf16,
    int32_t const* __restrict__ src2dst,
    bf16_t* __restrict__ out,
    int32_t T,
    int32_t H,
    int32_t down_rows) {
  int32_t const t = static_cast<int32_t>(blockIdx.x);
  if (t >= T) return;
  device::PDLWaitPrimary<true>();
  __shared__ int32_t s_dst[kTopK];
  if (threadIdx.x < static_cast<uint32_t>(kTopK)) {
    s_dst[threadIdx.x] = src2dst[static_cast<int64_t>(t) * kTopK + threadIdx.x];
  }
  __syncthreads();

  int32_t const col = static_cast<int32_t>(threadIdx.x) * 8;
  if (col < H) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int32_t k = 0; k < kTopK; ++k) {
      int32_t const d = s_dst[k];
      if (static_cast<uint32_t>(d) < static_cast<uint32_t>(down_rows)) {
        uint4 const raw = *reinterpret_cast<uint4 const*>(down_bf16 + static_cast<int64_t>(d) * H + col);
        bf16_t const* h = reinterpret_cast<bf16_t const*>(&raw);
#pragma unroll
        for (int32_t i = 0; i < 8; ++i)
          acc[i] += __bfloat162float(h[i]);
      }
    }
    alignas(16) bf16_t out8[8];
#pragma unroll
    for (int32_t i = 0; i < 8; ++i)
      out8[i] = __float2bfloat16_rn(acc[i]);
    *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * H + col) = *reinterpret_cast<uint4 const*>(out8);
  }
  device::PDLTriggerSecondary<true>();
}

/**
 * \brief SM12x low-M MXFP8 grouped GEMM reading CuTe activation scales and raw UE8M0 weight scales.
 * \param values FP8 E4M3 activations viewed as [rows, K] uint8.
 * \param weight FP8 E4M3 weights viewed as [E, N, K] uint8.
 * \param scales_cute Contiguous [K / 128, m_padded] int32 packed UE8M0 activation scales.
 * \param weight_scales_raw Contiguous [E, N, K / 32] uint8 raw UE8M0 weight scales.
 * \param out Output [rows, N] BF16.
 * \param offsets [E + 1] int32 CSR expert row offsets.
 * \param sm_count Device SM count.
 */
inline void mxfp8_gemm_lowm(
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scales_cute,
    tvm::ffi::TensorView weight_scales_raw,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView offsets,
    int64_t sm_count) {
  auto rows = host::SymbolicSize{"routed_rows"};
  auto E = host::SymbolicSize{"experts"};
  auto N = host::SymbolicSize{"out_features"};
  auto K = host::SymbolicSize{"in_features"};
  auto k_blocks = host::SymbolicSize{"k_div_128"};
  auto m_padded_sym = host::SymbolicSize{"m_padded"};
  auto w_sf_k = host::SymbolicSize{"k_div_32"};
  auto off_len = host::SymbolicSize{"experts_plus_1"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({rows, K}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(values);
  host::TensorMatcher({E, N, K}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(weight);
  host::TensorMatcher({k_blocks, m_padded_sym})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(scales_cute);
  host::TensorMatcher({E, N, w_sf_k})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(weight_scales_raw);
  host::TensorMatcher({rows, N}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);
  host::TensorMatcher({off_len}).with_dtype<int32_t>().with_device(device).verify(offsets);

  int64_t const num_groups = E.unwrap();
  int64_t const n = N.unwrap();
  int64_t const k = K.unwrap();
  int64_t const num_rows = rows.unwrap();
  int64_t const m_padded = m_padded_sym.unwrap();
  CHECK_HOST(off_len.unwrap() == num_groups + 1) << "mxfp8_gemm_lowm: offsets length mismatch";
  CHECK_HOST(num_groups > 0 && num_groups <= kMaxExperts) << "mxfp8_gemm_lowm: unsupported experts " << num_groups;
  CHECK_HOST(n > 0 && n % 64 == 0 && k > 0 && k % 128 == 0)
      << "mxfp8_gemm_lowm: requires N % 64 == 0 and K % 128 == 0, got N=" << n << " K=" << k;
  CHECK_HOST(k_blocks.unwrap() == k / 128 && w_sf_k.unwrap() == k / 32)
      << "mxfp8_gemm_lowm: scale K dimension mismatch";
  CHECK_HOST(m_padded >= ((num_rows + 3 * num_groups) / 4) * 4 && m_padded <= std::numeric_limits<int32_t>::max())
      << "mxfp8_gemm_lowm: invalid m_padded " << m_padded;
  CHECK_HOST(sm_count > 0) << "mxfp8_gemm_lowm: sm_count must be positive, got " << sm_count;
  if (num_rows == 0) return;

  int32_t const grid = static_cast<int32_t>(sm_count * 4);
  host::LaunchKernel(static_cast<uint32_t>(grid), 128, device.unwrap())
      .enable_pdl(true)(
          mxfp8_gemm_lowm_kernel,
          static_cast<uint8_t const*>(values.data_ptr()),
          static_cast<uint8_t const*>(weight.data_ptr()),
          static_cast<uint32_t const*>(scales_cute.data_ptr()),
          static_cast<uint32_t const*>(weight_scales_raw.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          static_cast<int32_t>(num_groups),
          static_cast<int32_t>(m_padded));
}

/**
 * \brief Sum top-8 expert-order BF16 rows per token in FP32 and round to BF16.
 * \param down_bf16 Input [rows, H] BF16 expert-order W2 outputs.
 * \param src2dst [T * 8] int32 token-slot to expert-order row indices.
 * \param out Output [T, H] BF16 combined hidden states.
 * \param top_k Experts per token (must be 8).
 */
inline void moe_combine_bf16(
    tvm::ffi::TensorView down_bf16, tvm::ffi::TensorView src2dst, tvm::ffi::TensorView out, int64_t top_k) {
  auto rows = host::SymbolicSize{"routed_rows"};
  auto T = host::SymbolicSize{"tokens"};
  auto H = host::SymbolicSize{"hidden_size"};
  auto map_len = host::SymbolicSize{"src2dst_len"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({rows, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(down_bf16);
  host::TensorMatcher({map_len}).with_dtype<int32_t>().with_device(device).verify(src2dst);
  host::TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);

  int64_t const tokens = T.unwrap();
  int64_t const hidden = H.unwrap();
  CHECK_HOST(top_k == 8) << "moe_combine_bf16: requires top_k == 8, got " << top_k;
  CHECK_HOST(map_len.unwrap() == tokens * top_k) << "moe_combine_bf16: src2dst length mismatch";
  CHECK_HOST(rows.unwrap() == map_len.unwrap()) << "moe_combine_bf16: down_bf16 rows (" << rows.unwrap()
                                                << ") must equal src2dst length (" << map_len.unwrap() << ")";
  CHECK_HOST(hidden > 0 && hidden % 8 == 0 && hidden / 8 <= 1024)
      << "moe_combine_bf16: requires 0 < H <= 8192 and H % 8 == 0, got " << hidden;
  if (tokens == 0) return;

  host::LaunchKernel(static_cast<uint32_t>(tokens), static_cast<uint32_t>(hidden / 8), device.unwrap())
      .enable_pdl(true)(
          moe_combine_bf16_kernel<8>,
          static_cast<bf16_t const*>(down_bf16.data_ptr()),
          static_cast<int32_t const*>(src2dst.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<int32_t>(tokens),
          static_cast<int32_t>(hidden),
          static_cast<int32_t>(rows.unwrap()));
}

}  // namespace sglang::inkling2_moe_lowm_sm120
