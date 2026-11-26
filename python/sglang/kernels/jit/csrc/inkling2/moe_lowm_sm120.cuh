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

SGL_DEVICE uint4 ld_w_cs(void const* ptr) {
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
  return v;
}

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

SGL_DEVICE void nvfp4_mma_m16n8k64(
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
    uint32_t sfb) {
#if defined(__CUDA_ARCH__)
  uint16_t const zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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
        "h"(zero),
        "h"(zero),
        "r"(sfb),
        "h"(zero),
        "h"(zero));
#endif
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

// Match baseline SM120 NVFP4 path: CUTLASS rounds raw FP32 accumulators to BF16
// before routed_quant_sm120 / routed_rows rescales by (x_amax * w_amax) * rcp.
SGL_DEVICE bf16_t round_and_scale_bf16(float acc, float scale) {
  float const raw_bf16 = __bfloat162float(__float2bfloat16_rn(acc));
  return __float2bfloat16_rn(__fmul_rn(raw_bf16, scale));
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

inline void verify_scalar_fp32(tvm::ffi::TensorView tensor, host::SymbolicDevice const& device) {
  DLDevice const expected = device.unwrap();
  DLDevice const actual = tensor.device();
  CHECK_HOST(actual.device_type == expected.device_type && actual.device_id == expected.device_id)
      << "scalar tensor device mismatch";
  CHECK_HOST(host::is_type<fp32_t>(tensor.dtype())) << "scalar tensor must be float32";
  CHECK_HOST(tensor.is_contiguous() && tensor.numel() == 1) << "scalar tensor must have 1 contiguous element";
}

}  // namespace

__global__ void __launch_bounds__(128, 2) nvfp4_gemm_lowm_scaled_kernel(
    uint8_t const* __restrict__ A_acts,
    uint8_t const* __restrict__ B_weights,
    uint8_t const* __restrict__ SFA_acts,
    uint8_t const* __restrict__ SFB_weights,
    bf16_t* __restrict__ D_out,
    int32_t const* __restrict__ m_indptr,
    float const* __restrict__ x_amax,
    float const* __restrict__ w_amax,
    float const* __restrict__ w_denom,
    int32_t n,
    int32_t k,
    int32_t num_groups) {
  constexpr int32_t kWarps = 4;
  constexpr int32_t kTileM = kWarps * 16;
  constexpr int32_t kMaxRows = 16;
  constexpr int32_t kUnrollK = 3;

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

  float const rcp = __frcp_rn(__fmul_rn(1536.0f, *w_denom));

  for (int32_t i = tid; i <= num_groups; i += kWarps * 32) {
    s_indptr[i] = m_indptr[i];
  }
  __syncthreads();

  if (warp == 0) {
    build_ntile_prefix<4>(s_indptr, s_ntile_prefix, num_groups, lane);
  }
  __syncthreads();

  int32_t const total_ntiles = s_ntile_prefix[num_groups];
  int32_t const m_tiles_per_n = n / kTileM;
  int32_t const total_tiles = total_ntiles * m_tiles_per_n;
  int32_t const k_bytes = k >> 1;
  int32_t const sf_n = ((n + 127) >> 7) << 7;
  int32_t const sf_k = (((k + 63) >> 6) << 6) >> 4;
  int32_t const sf_k_div4 = sf_k >> 2;
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

    uint8_t const* W_e = B_weights + static_cast<int64_t>(e) * n * k_bytes;
    uint8_t const* SFW_e = SFB_weights + static_cast<int64_t>(e) * sf_n * sf_k;
    int64_t const sf_m_off_e = ((static_cast<int64_t>(m_off_e) + static_cast<int64_t>(e) * 127) >> 7) << 7;
    uint8_t const* SFA_e = SFA_acts + sf_m_off_e * sf_k;

    uint8_t const* W_row0 = W_e + static_cast<int64_t>(m_base + grp) * k_bytes + tid_g * 16;
    uint8_t const* W_row1 = W_e + static_cast<int64_t>(m_base + grp + 8) * k_bytes + tid_g * 16;
    int32_t const sf_r = m_base + grp + (lane & 1) * 8;
    int64_t const sfa_base =
        (static_cast<int64_t>(sf_r >> 7) * sf_k_div4) * 512 + (sf_r & 31) * 16 + ((sf_r >> 5) & 3) * 4;

    int32_t const b0_local = row_start_local + grp;
    bool const b0_valid = (grp < rows_this_tile);
    uint8_t const* B0_ptr =
        A_acts + static_cast<int64_t>(m_off_e + (b0_valid ? b0_local : row_start_local)) * k_bytes + tid_g * 16;
    int64_t const sfb0_base =
        (static_cast<int64_t>(b0_local >> 7) * sf_k_div4) * 512 + (b0_local & 31) * 16 + ((b0_local >> 5) & 3) * 4;

    bool const has_second = (rows_this_tile > 8);
    int32_t const b1_local = row_start_local + 8 + grp;
    bool const b1_valid = has_second && (8 + grp < rows_this_tile);
    uint8_t const* B1_ptr =
        A_acts + static_cast<int64_t>(m_off_e + (b1_valid ? b1_local : row_start_local)) * k_bytes + tid_g * 16;
    int64_t const sfb1_base =
        (static_cast<int64_t>(b1_local >> 7) * sf_k_div4) * 512 + (b1_local & 31) * 16 + ((b1_local >> 5) & 3) * 4;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;

    int32_t ks = 0;
    for (; ks + kUnrollK <= num_k128; ks += kUnrollK) {
      uint4 w0_buf[kUnrollK], w1_buf[kUnrollK], b0_buf[kUnrollK], b1_buf[kUnrollK];
      uint32_t sfa0_buf[kUnrollK], sfa1_buf[kUnrollK];
      uint32_t sfb0_0_buf[kUnrollK], sfb0_1_buf[kUnrollK], sfb1_0_buf[kUnrollK], sfb1_1_buf[kUnrollK];

#pragma unroll
      for (int32_t u = 0; u < kUnrollK; ++u) {
        int32_t const kb = (ks + u) * 64;
        int32_t const blk = (ks + u) * 2;
        w0_buf[u] = ld_w_cs(W_row0 + kb);
        w1_buf[u] = ld_w_cs(W_row1 + kb);
        sfa0_buf[u] = load_sfa ? ld_sf_nc(SFW_e + sfa_base + static_cast<int64_t>(blk) * 512) : 0u;
        sfa1_buf[u] = load_sfa ? ld_sf_nc(SFW_e + sfa_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
        b0_buf[u] = b0_valid ? *reinterpret_cast<uint4 const*>(B0_ptr + kb) : make_uint4(0, 0, 0, 0);
        sfb0_0_buf[u] = b0_valid ? ld_sf_nc(SFA_e + sfb0_base + static_cast<int64_t>(blk) * 512) : 0u;
        sfb0_1_buf[u] = b0_valid ? ld_sf_nc(SFA_e + sfb0_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
        if (has_second) {
          b1_buf[u] = b1_valid ? *reinterpret_cast<uint4 const*>(B1_ptr + kb) : make_uint4(0, 0, 0, 0);
          sfb1_0_buf[u] = b1_valid ? ld_sf_nc(SFA_e + sfb1_base + static_cast<int64_t>(blk) * 512) : 0u;
          sfb1_1_buf[u] = b1_valid ? ld_sf_nc(SFA_e + sfb1_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
        }
      }

#pragma unroll
      for (int32_t u = 0; u < kUnrollK; ++u) {
        uint32_t const sfa0 = __shfl_sync(0xFFFFFFFFu, sfa0_buf[u], (lane & ~2));
        uint32_t const sfa1 = __shfl_sync(0xFFFFFFFFu, sfa1_buf[u], (lane & ~2));
        uint32_t a0_0, a2_0, a0_1, a2_1, a1_0, a3_0, a1_1, a3_1, b0_0, b1_0, b0_1, b1_1;
        transpose_4x4(w0_buf[u], tid_g, a0_0, a2_0, a0_1, a2_1);
        transpose_4x4(w1_buf[u], tid_g, a1_0, a3_0, a1_1, a3_1);
        transpose_4x4(b0_buf[u], tid_g, b0_0, b1_0, b0_1, b1_1);
        nvfp4_mma_m16n8k64(d0, d1, d2, d3, a0_0, a1_0, a2_0, a3_0, b0_0, b1_0, sfa0, sfb0_0_buf[u]);
        nvfp4_mma_m16n8k64(d0, d1, d2, d3, a0_1, a1_1, a2_1, a3_1, b0_1, b1_1, sfa1, sfb0_1_buf[u]);
        if (has_second) {
          uint32_t c0_0, c1_0, c0_1, c1_1;
          transpose_4x4(b1_buf[u], tid_g, c0_0, c1_0, c0_1, c1_1);
          nvfp4_mma_m16n8k64(f0, f1, f2, f3, a0_0, a1_0, a2_0, a3_0, c0_0, c1_0, sfa0, sfb1_0_buf[u]);
          nvfp4_mma_m16n8k64(f0, f1, f2, f3, a0_1, a1_1, a2_1, a3_1, c0_1, c1_1, sfa1, sfb1_1_buf[u]);
        }
      }
    }
    for (; ks < num_k128; ++ks) {
      int32_t const kb = ks * 64;
      int32_t const blk = ks * 2;
      uint4 const w0 = ld_w_cs(W_row0 + kb);
      uint4 const w1 = ld_w_cs(W_row1 + kb);
      uint32_t const sfa0_r = load_sfa ? ld_sf_nc(SFW_e + sfa_base + static_cast<int64_t>(blk) * 512) : 0u;
      uint32_t const sfa1_r = load_sfa ? ld_sf_nc(SFW_e + sfa_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
      uint32_t const sfa0 = __shfl_sync(0xFFFFFFFFu, sfa0_r, (lane & ~2));
      uint32_t const sfa1 = __shfl_sync(0xFFFFFFFFu, sfa1_r, (lane & ~2));
      uint4 const b0 = b0_valid ? *reinterpret_cast<uint4 const*>(B0_ptr + kb) : make_uint4(0, 0, 0, 0);
      uint32_t const sfb0_0 = b0_valid ? ld_sf_nc(SFA_e + sfb0_base + static_cast<int64_t>(blk) * 512) : 0u;
      uint32_t const sfb0_1 = b0_valid ? ld_sf_nc(SFA_e + sfb0_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
      uint32_t a0_0, a2_0, a0_1, a2_1, a1_0, a3_0, a1_1, a3_1, b0_0, b1_0, b0_1, b1_1;
      transpose_4x4(w0, tid_g, a0_0, a2_0, a0_1, a2_1);
      transpose_4x4(w1, tid_g, a1_0, a3_0, a1_1, a3_1);
      transpose_4x4(b0, tid_g, b0_0, b1_0, b0_1, b1_1);
      nvfp4_mma_m16n8k64(d0, d1, d2, d3, a0_0, a1_0, a2_0, a3_0, b0_0, b1_0, sfa0, sfb0_0);
      nvfp4_mma_m16n8k64(d0, d1, d2, d3, a0_1, a1_1, a2_1, a3_1, b0_1, b1_1, sfa1, sfb0_1);
      if (has_second) {
        uint4 const b1 = b1_valid ? *reinterpret_cast<uint4 const*>(B1_ptr + kb) : make_uint4(0, 0, 0, 0);
        uint32_t const sfb1_0 = b1_valid ? ld_sf_nc(SFA_e + sfb1_base + static_cast<int64_t>(blk) * 512) : 0u;
        uint32_t const sfb1_1 = b1_valid ? ld_sf_nc(SFA_e + sfb1_base + static_cast<int64_t>(blk + 1) * 512) : 0u;
        uint32_t c0_0, c1_0, c0_1, c1_1;
        transpose_4x4(b1, tid_g, c0_0, c1_0, c0_1, c1_1);
        nvfp4_mma_m16n8k64(f0, f1, f2, f3, a0_0, a1_0, a2_0, a3_0, c0_0, c1_0, sfa0, sfb1_0);
        nvfp4_mma_m16n8k64(f0, f1, f2, f3, a0_1, a1_1, a2_1, a3_1, c0_1, c1_1, sfa1, sfb1_1);
      }
    }

    int32_t const n0 = tid_g * 2;
    int32_t const n1 = tid_g * 2 + 1;
    float const wa = w_amax[e];
    if (n0 < rows_this_tile) {
      float const sc0 = __fmul_rn(__fmul_rn(x_amax[row_start_global + n0], wa), rcp);
      s_out[n0][warp * 16 + grp] = round_and_scale_bf16(d0, sc0);
      s_out[n0][warp * 16 + grp + 8] = round_and_scale_bf16(d2, sc0);
    }
    if (n1 < rows_this_tile) {
      float const sc1 = __fmul_rn(__fmul_rn(x_amax[row_start_global + n1], wa), rcp);
      s_out[n1][warp * 16 + grp] = round_and_scale_bf16(d1, sc1);
      s_out[n1][warp * 16 + grp + 8] = round_and_scale_bf16(d3, sc1);
    }
    if (has_second) {
      if (8 + n0 < rows_this_tile) {
        float const sc0 = __fmul_rn(__fmul_rn(x_amax[row_start_global + 8 + n0], wa), rcp);
        s_out[8 + n0][warp * 16 + grp] = round_and_scale_bf16(f0, sc0);
        s_out[8 + n0][warp * 16 + grp + 8] = round_and_scale_bf16(f2, sc0);
      }
      if (8 + n1 < rows_this_tile) {
        float const sc1 = __fmul_rn(__fmul_rn(x_amax[row_start_global + 8 + n1], wa), rcp);
        s_out[8 + n1][warp * 16 + grp] = round_and_scale_bf16(f1, sc1);
        s_out[8 + n1][warp * 16 + grp + 8] = round_and_scale_bf16(f3, sc1);
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
 * \brief SM12x low-M NVFP4 grouped GEMM with fused baseline-faithful BF16 rescaling.
 * \param values Packed E2M1 activations [rows, K / 2] uint8.
 * \param weight Packed E2M1 expert weights [E, N, K / 2] uint8.
 * \param scales Swizzled 128x4 activation scales [scale_rows, K / 16] uint8.
 * \param weight_scales Swizzled weight scales [E, sf_n * sf_k] uint8.
 * \param out Output [rows, N] BF16.
 * \param offsets [E + 1] int32 CSR expert row offsets.
 * \param x_amax [rows] FP32 per-row activation absmax.
 * \param w_amax [E] FP32 per-expert weight absmax.
 * \param w_denom Scalar FP32 outer scale denominator.
 * \param sm_count Device SM count.
 */
inline void nvfp4_gemm_lowm_scaled(
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView weight_scales,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView x_amax,
    tvm::ffi::TensorView w_amax,
    tvm::ffi::TensorView w_denom,
    int64_t sm_count) {
  auto rows = host::SymbolicSize{"routed_rows"};
  auto E = host::SymbolicSize{"experts"};
  auto N = host::SymbolicSize{"out_features"};
  auto K_half = host::SymbolicSize{"k_half"};
  auto scale_rows = host::SymbolicSize{"scale_rows"};
  auto sf_k_dim = host::SymbolicSize{"sf_k"};
  auto w_sf_cols = host::SymbolicSize{"w_sf_cols"};
  auto off_len = host::SymbolicSize{"experts_plus_1"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({rows, K_half}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(values);
  host::TensorMatcher({E, N, K_half}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(weight);
  host::TensorMatcher({scale_rows, sf_k_dim})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(scales);
  host::TensorMatcher({E, w_sf_cols})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(weight_scales);
  host::TensorMatcher({rows, N}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);
  host::TensorMatcher({off_len}).with_dtype<int32_t>().with_device(device).verify(offsets);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(x_amax);
  host::TensorMatcher({E}).with_dtype<fp32_t>().with_device(device).verify(w_amax);
  verify_scalar_fp32(w_denom, device);

  int64_t const num_groups = E.unwrap();
  int64_t const n = N.unwrap();
  int64_t const k = K_half.unwrap() * 2;
  int64_t const num_rows = rows.unwrap();
  CHECK_HOST(off_len.unwrap() == num_groups + 1) << "nvfp4_gemm_lowm_scaled: offsets length mismatch";
  CHECK_HOST(num_groups > 0 && num_groups <= kMaxExperts)
      << "nvfp4_gemm_lowm_scaled: unsupported experts " << num_groups;
  CHECK_HOST(n > 0 && n % 64 == 0 && k > 0 && k % 128 == 0)
      << "nvfp4_gemm_lowm_scaled: requires N % 64 == 0 and K % 128 == 0, got N=" << n << " K=" << k;
  CHECK_HOST(sf_k_dim.unwrap() == k / 16) << "nvfp4_gemm_lowm_scaled: activation scale width mismatch";
  int64_t const sf_n = ((n + 127) / 128) * 128;
  int64_t const sf_k = (((k + 63) / 64) * 64) / 16;
  CHECK_HOST(w_sf_cols.unwrap() == sf_n * sf_k) << "nvfp4_gemm_lowm_scaled: weight scale size mismatch";
  CHECK_HOST(scale_rows.unwrap() >= host::div_ceil(num_rows + 127 * num_groups, 128) * 128)
      << "nvfp4_gemm_lowm_scaled: insufficient activation scale rows " << scale_rows.unwrap();
  CHECK_HOST(sm_count > 0) << "nvfp4_gemm_lowm_scaled: sm_count must be positive, got " << sm_count;
  if (num_rows == 0) return;

  int32_t const grid = static_cast<int32_t>(sm_count * 4);
  host::LaunchKernel(static_cast<uint32_t>(grid), 128, device.unwrap())
      .enable_pdl(true)(
          nvfp4_gemm_lowm_scaled_kernel,
          static_cast<uint8_t const*>(values.data_ptr()),
          static_cast<uint8_t const*>(weight.data_ptr()),
          static_cast<uint8_t const*>(scales.data_ptr()),
          static_cast<uint8_t const*>(weight_scales.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          static_cast<float const*>(x_amax.data_ptr()),
          static_cast<float const*>(w_amax.data_ptr()),
          static_cast<float const*>(w_denom.data_ptr()),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          static_cast<int32_t>(num_groups));
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
