// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// DFlash2 selector tail in one launch (SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH, default off): dflash_selector_select's
// candidate top-k (its warp code, verbatim, one warp per draft slot), the lattice einsum (mma.sync BF16 -> FP32, K in
// increasing order, BF16 out; the same bits as the served torch.einsum bmm) and selector_walk_fused (the served Triton
// walk's instruction sequence: redux max, sub, mul log2e, ex2.approx, shfl.bfly 8/4/2/1 sum, div.full, shfl.up 1/2/4/8
// scan). Every output (candidates, tokens, q_rows) is bitwise equal to the three-launch chain; unary / pred_h / keys /
// lattice stay on chip. One CTA per request b, 32 x kMaxSlots threads; warp e owns slot e (draft row n = b * slots + e).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>

#include <cstdint>

namespace sglang {

namespace dflash_selector_one_launch {

inline constexpr uint32_t kSortSize = 32;
inline constexpr uint32_t kKChunk = 64;  // K columns staged per warp per round

SGL_DEVICE uint32_t radix_key(float v) {
  const uint32_t x = __float_as_uint(v);
  const uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (v == v) ? (x ^ mask) : 0xffffffffu;
}

SGL_DEVICE bool greater_nan_first(float lhs, float rhs) {
  return (isnan(lhs) && !isnan(rhs)) || (lhs > rhs);
}

SGL_DEVICE float div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

SGL_DEVICE float ex2_approx(float a) {
  float d;
  asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

SGL_DEVICE float redux_max_f32(float a) {
  float d;
  asm volatile("redux.sync.max.f32 %0, %1, -1;" : "=f"(d) : "f"(a));
  return d;
}

SGL_DEVICE int redux_max_s32(int a) {
  int d;
  asm volatile("redux.sync.max.s32 %0, %1, -1;" : "=r"(d) : "r"(a));
  return d;
}

SGL_DEVICE int redux_min_s32(int a) {
  int d;
  asm volatile("redux.sync.min.s32 %0, %1, -1;" : "=r"(d) : "r"(a));
  return d;
}

SGL_DEVICE int redux_add_s32(int a) {
  int d;
  asm volatile("redux.sync.add.s32 %0, %1, -1;" : "=r"(d) : "r"(a));
  return d;
}

SGL_DEVICE float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

SGL_DEVICE float sub_rn(float a, float b) {
  float d;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

SGL_DEVICE float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

SGL_DEVICE void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct Params {
  const float* vals;          // [rows, width] gathered candidate logits
  const void* ids;            // [rows, width] gathered global ids (IdT)
  const bf16_t* hidden;       // [bs, slots, rank]
  const int64_t* anchors;     // [bs], stride anchor_stride
  const bf16_t* pred_table;   // [vocab, rank]
  const bf16_t* succ_table;   // [vocab, rank]
  const float* uniforms;      // [bs, slots]
  const float* temperatures;  // [bs]
  const uint8_t* greedy;      // [bs] (uint8: 0 = sample)
  int64_t* candidates;        // [rows, k]
  int64_t* tokens;            // [rows]
  float* q_rows;              // [bs, slots, k]
  bf16_t* lattice;            // [bs, slots, k, k] debug copy of the on-chip lattice, or nullptr (served: nullptr)
  int64_t anchor_stride;
  int64_t vocab;
  uint32_t slots;
};

}  // namespace dflash_selector_one_launch

template <int kWidth, int kTopK, int kRank, typename IdT, int kMaxSlots>
__global__ void __launch_bounds__(32 * kMaxSlots) dflash_selector_one_launch_kernel(const __grid_constant__ dflash_selector_one_launch::Params p) {
  using namespace dflash_selector_one_launch;
  constexpr int kPerLane = kWidth / 32;
  constexpr int kIdVecs = kPerLane * static_cast<int>(sizeof(IdT)) / 16;
  static_assert(kWidth % 32 == 0 && kPerLane % 4 == 0 && kPerLane <= 32, "width: 32 lanes x float4 loads");
  static_assert(kIdVecs >= 1 && kPerLane * sizeof(IdT) % 16 == 0, "ids: whole 16-byte loads per lane");
  static_assert(kTopK == 16 && kSortSize == 32, "torch's 32-slot sort; one 16 x 16 lattice per slot");
  static_assert(kRank % kKChunk == 0, "rank: whole K chunks");

  __shared__ float s_key[kMaxSlots][kTopK];
  __shared__ int64_t s_id[kMaxSlots][kTopK];
  __shared__ int64_t s_cand[kMaxSlots][kTopK];
  __shared__ float s_unary[kMaxSlots][kTopK];
  extern __shared__ __align__(16) unsigned char one_launch_dsmem[];  // per warp: pred rows [16][rank], succ rows [16][rank], hidden [rank]
  __shared__ bf16_t s_lat[kMaxSlots][kTopK][kTopK];

  const uint32_t b = blockIdx.x;
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t slots = p.slots;
  const bool active = warp < slots;
  const uint32_t e = warp;
  const uint32_t n = b * slots + e;

  if (active) {
    // ---- selection of row n: dflash_selector_select_kernel's warp-0 code, unchanged ----
    float v[kPerLane];
    uint32_t key[kPerLane];
    union {
      uint4 vec[kIdVecs];
      IdT id[kPerLane];
    } ids;
    const float4* src = reinterpret_cast<const float4*>(p.vals + static_cast<int64_t>(n) * kWidth) + lane * (kPerLane / 4);
    const uint4* id_src = reinterpret_cast<const uint4*>(static_cast<const IdT*>(p.ids) + static_cast<int64_t>(n) * kWidth) +
                          lane * kIdVecs;
#pragma unroll
    for (int j = 0; j < kIdVecs; ++j) ids.vec[j] = id_src[j];
#pragma unroll
    for (int j = 0; j < kPerLane / 4; ++j) {
      const float4 q = src[j];
      v[4 * j] = q.x;
      v[4 * j + 1] = q.y;
      v[4 * j + 2] = q.z;
      v[4 * j + 3] = q.w;
    }
    uint32_t any = 0u, all = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      key[i] = radix_key(v[i]);
      any |= key[i];
      all &= key[i];
    }
    any = __reduce_or_sync(0xffffffffu, any);
    all = __reduce_and_sync(0xffffffffu, all);
    static_assert(kWidth < 1024, "a digit count fits a 10-bit field");
    uint32_t kth = all;
    if (any != all) {
      const int top = 31 - __clz(any ^ all);
      uint32_t mask = ~((2u << top) - 1u), prefix = all & mask, rank = kTopK;
      auto step1 = [&](int bit) {
        const uint32_t m2 = mask | (1u << bit), want = prefix | (1u << bit);
        uint32_t cnt = 0;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) cnt += (key[i] & m2) == want;
        cnt = __reduce_add_sync(0xffffffffu, cnt);
        if (cnt >= rank) {
          prefix = want;
        } else {
          rank -= cnt;
        }
        mask = m2;
      };
      auto step2 = [&](int bit) {
        const uint32_t m2 = mask | (3u << bit);
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const uint32_t k = key[i] & m2;
          packed += k == (prefix | (3u << bit)) ? 1u : k == (prefix | (2u << bit)) ? (1u << 10) :
                    k == (prefix | (1u << bit)) ? (1u << 20) : 0u;
        }
        packed = __reduce_add_sync(0xffffffffu, packed);
        const uint32_t c3 = packed & 1023u, c2 = (packed >> 10) & 1023u, c1 = packed >> 20;
        uint32_t digit = 0;
        if (c3 >= rank) {
          digit = 3;
        } else if (c3 + c2 >= rank) {
          digit = 2;
          rank -= c3;
        } else if (c3 + c2 + c1 >= rank) {
          digit = 1;
          rank -= c3 + c2;
        } else {
          rank -= c3 + c2 + c1;
        }
        prefix |= digit << bit;
        mask = m2;
      };
      int bit = top;
      auto run_down_to = [&](int stop) {
        if (((bit - stop + 1) & 1) != 0) step1(bit--);
#pragma unroll 1
        for (; bit > stop; bit -= 2) step2(bit - 1);
      };
      bool done = false;
      if (top >= 16) {
        run_down_to(16);
        uint32_t lo_any = 0u, lo_all = 0xffffu;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          if ((key[i] & mask) == prefix) {
            lo_any |= key[i] & 0xffffu;
            lo_all &= key[i] & 0xffffu;
          }
        }
        lo_any = __reduce_or_sync(0xffffffffu, lo_any);
        lo_all = __reduce_and_sync(0xffffffffu, lo_all);
        if (lo_any == lo_all) {
          prefix |= lo_any;
          done = true;
        }
      }
      if (!done) run_down_to(0);
      kth = prefix;
    }
    uint32_t above = 0, equal = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      above |= static_cast<uint32_t>(key[i] > kth) << i;
      equal |= static_cast<uint32_t>(key[i] == kth) << i;
    }
    const uint32_t lower_lanes = (1u << lane) - 1u;
    uint32_t above_before = 0, equal_before = 0, num_above = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const uint32_t ba = __ballot_sync(0xffffffffu, (above >> i) & 1u);
      const uint32_t be = __ballot_sync(0xffffffffu, (equal >> i) & 1u);
      above_before += __popc(ba & lower_lanes);
      equal_before += __popc(be & lower_lanes);
      num_above += __popc(ba);
    }
    const uint32_t num_equal = kTopK - num_above;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const uint32_t below_i = (1u << i) - 1u;
      uint32_t pos = kSortSize;
      if ((above >> i) & 1u) {
        pos = above_before + __popc(above & below_i);
      } else if ((equal >> i) & 1u) {
        const uint32_t r = equal_before + __popc(equal & below_i);
        if (r < num_equal) pos = num_above + r;
      }
      if (pos < kSortSize) {
        const int64_t id = static_cast<int64_t>(ids.id[i]);
        if (id < 0 || id >= p.vocab) __trap();
        s_key[e][pos] = v[i];
        s_id[e][pos] = id;
      }
    }
    __syncwarp();
    // torch's bitonicSort<32>, one slot per lane (served code, warp 0 there).
    float skey = lane < static_cast<uint32_t>(kTopK) ? s_key[e][lane] : 0.0f;
    uint32_t slot = lane < static_cast<uint32_t>(kTopK) ? lane : 0u;
    uint32_t valid = lane < static_cast<uint32_t>(kTopK) ? 1u : 0u;
    auto compare_exchange = [&](uint32_t stride, bool dir) {
      const float other_key = __shfl_xor_sync(0xffffffffu, skey, stride);
      const uint32_t other_slot = __shfl_xor_sync(0xffffffffu, slot, stride);
      const uint32_t other_valid = __shfl_xor_sync(0xffffffffu, valid, stride);
      const bool lower = (lane & stride) == 0;
      const float key_a = lower ? skey : other_key, key_b = lower ? other_key : skey;
      const bool valid_a = (lower ? valid : other_valid) != 0, valid_b = (lower ? other_valid : valid) != 0;
      const bool swap = (greater_nan_first(key_a, key_b) && valid_a) || !valid_b;
      if (swap == dir) {
        skey = other_key;
        slot = other_slot;
        valid = other_valid;
      }
    };
#pragma unroll
    for (uint32_t size = 2; size < kSortSize; size *= 2) {
#pragma unroll
      for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
        compare_exchange(stride, ((lane & ~stride) & size) != 0);
      }
    }
#pragma unroll
    for (uint32_t stride = kSortSize / 2; stride > 0; stride /= 2) compare_exchange(stride, false);
    if (lane < static_cast<uint32_t>(kTopK)) {
      const int64_t cid = s_id[e][slot];
      s_cand[e][lane] = cid;
      s_unary[e][lane] = skey;
      p.candidates[static_cast<int64_t>(n) * kTopK + lane] = cid;
    }
  }
  __syncthreads();  // every slot's sorted candidates (slot e's lattice needs slot e - 1's)

  if (active) {
    // ---- lattice of slot e: pred_h[e] (16 x rank) . keys[e]^T, K chunks of 64 staged in shared memory ----
    // pred_h[e][p] = bf16(pred_table[id] * hidden[b, e]), id = anchor (e = 0) or slot e - 1's sorted candidate p;
    // keys[e][c] = succ_table[slot e's sorted candidate c] (the served select kernel's operands).
    const int64_t anchor = e == 0 ? p.anchors[static_cast<int64_t>(b) * p.anchor_stride] : 0;
    if (e == 0 && (anchor < 0 || anchor >= p.vocab)) __trap();
    const uint32_t g = lane / 4, t4 = lane % 4;
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    constexpr int kStride = kRank + 8;  // 16 B pad per row: rows g = 0..7 land in different banks
    bf16_t* s_pred = reinterpret_cast<bf16_t*>(one_launch_dsmem) + static_cast<int64_t>(e) * (32 * kStride + kRank);
    bf16_t* s_succ = s_pred + 16 * kStride;
    bf16_t* s_h = s_succ + 16 * kStride;
    const bf16_t* h_row = p.hidden + static_cast<int64_t>(n) * kRank;
    constexpr int kVecRow = kRank / 8;  // uint4 per row
    // Every load in flight at once (cp.async.cg 16 B, global -> shared), one wait.
#pragma unroll
    for (int j = 0; j < 16 * kVecRow / 32; ++j) {
      const uint32_t idx = lane + 32 * j, row = idx / kVecRow, c8 = (idx % kVecRow) * 8;
      const int64_t pid = e == 0 ? anchor : s_cand[e - 1][row];
      const uint32_t dp = static_cast<uint32_t>(__cvta_generic_to_shared(s_pred + row * kStride + c8));
      const uint32_t ds = static_cast<uint32_t>(__cvta_generic_to_shared(s_succ + row * kStride + c8));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dp), "l"(p.pred_table + pid * kRank + c8) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(ds), "l"(p.succ_table + s_cand[e][row] * kRank + c8) : "memory");
    }
#pragma unroll
    for (int j = 0; j < kVecRow / 32; ++j) {
      const uint32_t c8 = (lane + 32 * j) * 8;
      const uint32_t dh = static_cast<uint32_t>(__cvta_generic_to_shared(s_h + c8));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dh), "l"(h_row + c8) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncwarp();
    auto pred_h_pair = [&](uint32_t row, uint32_t col) -> uint32_t {  // bf16(pred * hidden) for 2 columns, as mul_bf16x8
      const float2 pf = __bfloat1622float2(*reinterpret_cast<const bf16x2_t*>(s_pred + row * kStride + col));
      const float2 hf = __bfloat1622float2(*reinterpret_cast<const bf16x2_t*>(s_h + col));
      const bf16x2_t o = __floats2bfloat162_rn(pf.x * hf.x, pf.y * hf.y);
      return *reinterpret_cast<const uint32_t*>(&o);
    };
#pragma unroll
    for (int kk = 0; kk < kRank; kk += 16) {
      uint32_t a[4];
      a[0] = pred_h_pair(g, kk + 2 * t4);
      a[1] = pred_h_pair(g + 8, kk + 2 * t4);
      a[2] = pred_h_pair(g, kk + 8 + 2 * t4);
      a[3] = pred_h_pair(g + 8, kk + 8 + 2 * t4);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(s_succ + (nt * 8 + g) * kStride + kk + 2 * t4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(s_succ + (nt * 8 + g) * kStride + kk + 8 + 2 * t4);
        mma_bf16(acc[nt], a, b0, b1);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const uint32_t col = nt * 8 + 2 * t4;
      s_lat[e][g][col] = __float2bfloat16_rn(acc[nt][0]);
      s_lat[e][g][col + 1] = __float2bfloat16_rn(acc[nt][1]);
      s_lat[e][g + 8][col] = __float2bfloat16_rn(acc[nt][2]);
      s_lat[e][g + 8][col + 1] = __float2bfloat16_rn(acc[nt][3]);
      if (p.lattice != nullptr) {
        bf16_t* out = p.lattice + static_cast<int64_t>(n) * kTopK * kTopK;
        out[g * kTopK + col] = __float2bfloat16_rn(acc[nt][0]);
        out[g * kTopK + col + 1] = __float2bfloat16_rn(acc[nt][1]);
        out[(g + 8) * kTopK + col] = __float2bfloat16_rn(acc[nt][2]);
        out[(g + 8) * kTopK + col + 1] = __float2bfloat16_rn(acc[nt][3]);
      }
    }
  }
  __syncthreads();  // every slot's lattice

  if (warp == 0) {
    // ---- the walk: selector_walk_fused's instruction sequence (selw3 walk.ptx), lanes 16-31 mirror lanes 0-15 ----
    const uint32_t i = lane & 15u;
    const float temperature = p.temperatures[b];
    const bool greedy = p.greedy[b] != 0;
    int previous = 0;
    for (uint32_t s = 0; s < slots; ++s) {
      const int64_t base = (static_cast<int64_t>(b) * slots + s) * kTopK;
      const float scores = add_rn(s_unary[s][i], __bfloat162float(s_lat[s][previous][i]));
      float prob;
      int index;
      if (greedy) {
        const bool is_nan = scores != scores;
        const float best = redux_max_f32(is_nan ? __int_as_float(0xff800000) : scores);
        const int has_nan = redux_max_s32(is_nan ? 1 : 0);
        const int r17 = scores == best ? static_cast<int>(i) : kTopK;
        const int r18 = has_nan < 1 ? r17 : kTopK;
        const int r19 = is_nan ? static_cast<int>(i) : r18;
        const int r20 = has_nan > 0 ? r19 : r18;
        index = redux_min_s32(r20);
        prob = static_cast<int>(i) == index ? 1.0f : 0.0f;
      } else {
        const float scaled = div_full(scores, temperature);
        const float m = redux_max_f32(scaled);
        const float ex = ex2_approx(mul_rn(sub_rn(scaled, m), 1.44269502162933349609375f));  // 0f3FB8AA3B
        float sum = ex;
        sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 8));
        sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 4));
        sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
        sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
        prob = div_full(ex, sum);
        float cum = prob;
        float up = __shfl_up_sync(0xffffffffu, cum, 1);
        cum = i == 0 ? cum : add_rn(cum, up);
        up = __shfl_up_sync(0xffffffffu, cum, 2);
        cum = i > 1 ? add_rn(cum, up) : cum;
        up = __shfl_up_sync(0xffffffffu, cum, 4);
        cum = i > 3 ? add_rn(cum, up) : cum;
        up = __shfl_up_sync(0xffffffffu, cum, 8);
        cum = i > 7 ? add_rn(cum, up) : cum;
        const float u = p.uniforms[static_cast<int64_t>(b) * slots + s];
        const int hit = (lane < 16 && cum <= u) ? 1 : 0;
        index = redux_add_s32(hit);
        index = index < kTopK - 1 ? index : kTopK - 1;
      }
      if (lane < 16) p.q_rows[base + i] = prob;
      if (lane == 0) p.tokens[static_cast<int64_t>(b) * slots + s] = s_cand[s][index];
      previous = index;
    }
  }
}

template <int kWidth, int kTopK, int kRank, typename IdT, int kMaxSlots>
struct DFlashSelectorOneLaunch {
  static void
  run(tvm::ffi::TensorView vals,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView anchors,
      tvm::ffi::TensorView pred_table,
      tvm::ffi::TensorView succ_table,
      tvm::ffi::TensorView uniforms,
      tvm::ffi::TensorView temperatures,
      tvm::ffi::TensorView greedy,
      tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView tokens,
      tvm::ffi::TensorView q_rows,
      tvm::ffi::TensorView lattice,
      int64_t want_lattice) {
    using namespace host;
    auto Rows = SymbolicSize{"rows"};
    auto Bs = SymbolicSize{"bs"};
    auto Slots = SymbolicSize{"slots"};
    auto Vocab = SymbolicSize{"vocab"};
    auto AnchorStride = SymbolicSize{"anchor_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({Rows, kWidth}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(vals);
    TensorMatcher({Rows, kWidth}).with_dtype<IdT>().with_device(device).ensure_alignment(16).verify(ids);
    TensorMatcher({Bs, Slots, kRank}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(hidden);
    TensorMatcher({Bs}).with_strides({AnchorStride}).with_dtype<int64_t>().with_device(device).verify(anchors);
    TensorMatcher({Vocab, kRank}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(pred_table).verify(succ_table);
    TensorMatcher({Bs, Slots}).with_dtype<fp32_t>().with_device(device).verify(uniforms);
    TensorMatcher({Bs}).with_dtype<fp32_t>().with_device(device).verify(temperatures);
    TensorMatcher({Bs}).with_dtype<uint8_t>().with_device(device).verify(greedy);
    TensorMatcher({Rows, kTopK}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({Rows}).with_dtype<int64_t>().with_device(device).verify(tokens);
    TensorMatcher({Bs, Slots, kTopK}).with_dtype<fp32_t>().with_device(device).verify(q_rows);
    // lattice is a debug output (tests); the served call passes want_lattice = 0 and any tensor.
    if (want_lattice != 0) TensorMatcher({Bs, Slots, kTopK, kTopK}).with_dtype<bf16_t>().with_device(device).verify(lattice);
    const int64_t rows = Rows.unwrap(), bs = Bs.unwrap(), slots = Slots.unwrap();
    CHECK_HOST(rows == bs * slots) << "rows " << rows << " != bs " << bs << " x slots " << slots;
    CHECK_HOST(slots >= 1 && slots <= kMaxSlots) << "slots " << slots;
    if (rows == 0) return;
    const auto params = dflash_selector_one_launch::Params{
        .vals = static_cast<const float*>(vals.data_ptr()),
        .ids = ids.data_ptr(),
        .hidden = static_cast<const bf16_t*>(hidden.data_ptr()),
        .anchors = static_cast<const int64_t*>(anchors.data_ptr()),
        .pred_table = static_cast<const bf16_t*>(pred_table.data_ptr()),
        .succ_table = static_cast<const bf16_t*>(succ_table.data_ptr()),
        .uniforms = static_cast<const float*>(uniforms.data_ptr()),
        .temperatures = static_cast<const float*>(temperatures.data_ptr()),
        .greedy = static_cast<const uint8_t*>(greedy.data_ptr()),
        .candidates = static_cast<int64_t*>(candidates.data_ptr()),
        .tokens = static_cast<int64_t*>(tokens.data_ptr()),
        .q_rows = static_cast<float*>(q_rows.data_ptr()),
        .lattice = want_lattice ? static_cast<bf16_t*>(lattice.data_ptr()) : nullptr,
        .anchor_stride = AnchorStride.unwrap(),
        .vocab = Vocab.unwrap(),
        .slots = static_cast<uint32_t>(slots),
    };
    constexpr std::size_t kSmem = static_cast<std::size_t>(kMaxSlots) * (32 * (kRank + 8) + kRank) * sizeof(bf16_t);
    // One module per process instantiates this template (the struct name is unique to this file): a second .so with the
    // same instantiation would share this static (STB_GNU_UNIQUE) and skip its own attribute (H-SELW-CAPTURE).
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(dflash_selector_one_launch_kernel<kWidth, kTopK, kRank, IdT, kMaxSlots>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
      configured = true;
    }
    LaunchKernel(static_cast<uint32_t>(bs), 32u * kMaxSlots, device.unwrap(), kSmem)(
        dflash_selector_one_launch_kernel<kWidth, kTopK, kRank, IdT, kMaxSlots>, params);
  }
};

}  // namespace sglang
