// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// DFlash2 selector after the candidate gather (SGLANG_OPT_DFLASH_FUSED_SELECTOR): torch.topk of every draft row's
// gathered candidates with torch's own tie order, the candidate-id gather, and the codebook rows that feed the
// selector's lattice GEMM, in one launch of one CTA per draft row n = (request b, slot e).
//
// The selection reproduces torch 2.15 (TensorTopK.cu sbtopk::gatherTopK, then Sort.cu SmallBitonicSort): the
// candidates whose radix key is above the k-th largest key, in index order, then the first ones equal to it, in
// index order, sorted by the 32-slot bitonic network on 16 threads with GTOp<float, true>. Values, indices and the
// order of ties therefore match torch.topk(gathered, k) bitwise. k is 16 (a top-16 drafter, 16 invalid sort slots)
// or 32 (the public DFlash2 drafter, every sort slot valid); both sort through the same 32-slot network.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>

#include <cstdint>

namespace sglang {

namespace dflash_selector {

inline constexpr uint32_t kThreads = 256;
// torch's SmallBitonicSort: sort_size 32, 16 threads holding two slots each, unused slots invalid.
inline constexpr uint32_t kSortSize = 32;
inline constexpr uint32_t kSortThreads = 16;

// TopKTypeConfig<float>::convert: an order-preserving uint32 key; every NaN maps to the largest key.
SGL_DEVICE uint32_t radix_key(float v) {
  const uint32_t x = __float_as_uint(v);
  const uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (v == v) ? (x ^ mask) : 0xffffffffu;
}

// GTOp<float, true>: NaN ranks above every number.
SGL_DEVICE bool greater_nan_first(float lhs, float rhs) {
  return (isnan(lhs) && !isnan(rhs)) || (lhs > rhs);
}

struct SelectParams {
  const float* vals;        // [rows, width] gathered candidate logits
  const void* ids;          // [rows, width] gathered global token ids (IdT)
  const bf16_t* hidden;     // [bs, slots, rank] hidden_projection output
  const int64_t* anchors;   // [bs] verified anchor token, stride anchor_stride
  const bf16_t* pred_table;  // [vocab, rank] predecessor codebook
  const bf16_t* succ_table;  // [vocab, rank] successor codebook
  int64_t* candidates;      // [rows, k] global token ids
  float* unary;             // [rows, k] candidate logits
  bf16_t* pred_h;           // [rows, k, rank] predecessor row * hidden (bf16, rounded once)
  bf16_t* keys;             // [rows, k, rank] successor row
  int64_t anchor_stride;
  int64_t vocab;
  uint32_t slots;
};

}  // namespace dflash_selector

// bf16(f32(a) * f32(h)) for 8 BF16 pairs: the product of two BF16 is exact in FP32, so this rounds once, as the model's
// BF16 multiply does.
SGL_DEVICE uint4 mul_bf16x8(const uint4 a, const uint4 h) {
  uint4 out;
  const bf16x2_t* pa = reinterpret_cast<const bf16x2_t*>(&a);
  const bf16x2_t* ph = reinterpret_cast<const bf16x2_t*>(&h);
  bf16x2_t* po = reinterpret_cast<bf16x2_t*>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 af = __bfloat1622float2(pa[j]);
    const float2 hf = __bfloat1622float2(ph[j]);
    po[j] = __floats2bfloat162_rn(af.x * hf.x, af.y * hf.y);
  }
  return out;
}

// grid rows, kThreads threads. Warp 0 selects (pre-sort order) while every thread already holds its hidden and anchor
// chunks; after barrier A all threads fetch the selected candidates' codebook rows while warp 0 runs torch's bitonic
// network in registers; after barrier B every thread stores its rows at their sorted positions.
template <int kWidth, int kTopK, int kRank, typename IdT>
__global__ void __launch_bounds__(dflash_selector::kThreads)
    dflash_selector_select_kernel(const __grid_constant__ dflash_selector::SelectParams p) {
  using namespace dflash_selector;
  constexpr int kPerLane = kWidth / 32;
  constexpr int kChunks = kRank / 8;  // 16-byte chunks per codebook row
  constexpr int kRowsPerPass = kThreads / kChunks;
  constexpr int kPasses = kTopK / kRowsPerPass;
  constexpr int kIdVecs = kPerLane * static_cast<int>(sizeof(IdT)) / 16;
  static_assert(kWidth % 32 == 0 && kPerLane % 4 == 0 && kPerLane <= 32, "width: 32 lanes x float4 loads");
  static_assert(kIdVecs >= 1 && kPerLane * sizeof(IdT) % 16 == 0, "ids: whole 16-byte loads per lane");
  // torch.topk sorts k <= 32 with SmallBitonicSort (Sort.cu sortKeyValueInplace, sort_size 32): k 16 pads 16 invalid
  // slots, k 32 fills all 32. Larger k takes WarpMergeSort / radix sort, whose tie order this network does not reproduce.
  static_assert(
      (kTopK == static_cast<int>(kSortThreads) || kTopK == static_cast<int>(kSortSize)) && kSortSize == 32,
      "torch's 32-slot sort, one slot per lane: k 16 or 32");
  static_assert(kTopK <= kWidth, "k within the gathered width");
  static_assert(kRank % 8 == 0 && kThreads % kChunks == 0 && kTopK % kRowsPerPass == 0, "whole rows per pass");

  __shared__ float s_key[kTopK];     // value of pre-sort slot j
  __shared__ int64_t s_id[kTopK];    // token id of pre-sort slot j
  __shared__ uint32_t s_pos[kTopK];  // sorted position of pre-sort slot j

  const uint32_t n = blockIdx.x;
  const uint32_t b = n / p.slots, e = n % p.slots;
  const uint32_t t = threadIdx.x, lane = t % 32;
  const uint32_t chunk = t % kChunks, row0 = t / kChunks;
  const bool has_next = e + 1 < p.slots;

  // Operands that do not depend on the selection.
  uint4 h_next = make_uint4(0, 0, 0, 0), anchor_row = make_uint4(0, 0, 0, 0);
  if (has_next) h_next = reinterpret_cast<const uint4*>(p.hidden + (static_cast<int64_t>(n) + 1) * kRank)[chunk];
  if (e == 0) {
    const int64_t anchor = p.anchors[static_cast<int64_t>(b) * p.anchor_stride];
    if (anchor < 0 || anchor >= p.vocab) __trap();
    const uint4 pred = reinterpret_cast<const uint4*>(p.pred_table + anchor * kRank)[chunk];
    const uint4 h0 = reinterpret_cast<const uint4*>(p.hidden + static_cast<int64_t>(n) * kRank)[chunk];
    anchor_row = mul_bf16x8(pred, h0);
  }

  if (t < 32) {
    // Lane owns gathered columns [lane * kPerLane, (lane + 1) * kPerLane): index order is lane-major.
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

    // The kTopK-th largest key, counted with multiplicity (radixSelect's result), two bits per round: one packed warp
    // sum carries the counts of digits 3, 2 and 1 (10 bits each; a count is at most kWidth). Bits every key shares are
    // taken as they are; once bits 31-16 are fixed, equal low halves among the remaining keys (BF16-valued logits) end
    // the search.
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
      auto step2 = [&](int bit) {  // digit = bits (bit + 1, bit)
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
      auto run_down_to = [&](int stop) {  // bits [stop, bit], highest first
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

    // gatherTopK's output order: keys above kth in index order, then the first keys equal to kth in index order.
    // Index order is lane-major, so a lane's prefix is the set bits of lower lanes in each element's ballot.
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
        // Out-of-range ids fail loudly, as torch's index kernel asserts.
        if (id < 0 || id >= p.vocab) __trap();
        s_key[pos] = v[i];
        s_id[pos] = id;
      }
    }
  }
  __syncthreads();  // A: the selection, in pre-sort order

  // The selected candidates' codebook rows, fetched while warp 0 sorts.
  uint4 succ[kPasses], pred[kPasses];
#pragma unroll
  for (int q = 0; q < kPasses; ++q) {
    const int64_t id = s_id[row0 + q * kRowsPerPass];
    succ[q] = reinterpret_cast<const uint4*>(p.succ_table + id * kRank)[chunk];
    pred[q] = has_next ? reinterpret_cast<const uint4*>(p.pred_table + id * kRank)[chunk] : make_uint4(0, 0, 0, 0);
  }

  if (t < 32) {
    // SortUtils.cuh bitonicSort<32> with bitonicSwap, one slot per lane: torch's thread j compares slots pos and
    // pos + stride (pos = 2j - (j & (stride - 1)), i.e. pos ^ stride is the partner and bit `stride` of pos is clear)
    // with direction (j & (size / 2)) != 0 = (pos & size) != 0, and false in the final merge. Slots past kTopK are
    // bitonicSortKVInPlace's invalid padding (key 0, value 0). The value is the pre-sort slot.
    float key = lane < static_cast<uint32_t>(kTopK) ? s_key[lane] : 0.0f;
    uint32_t slot = lane < static_cast<uint32_t>(kTopK) ? lane : 0u;
    uint32_t valid = lane < static_cast<uint32_t>(kTopK) ? 1u : 0u;
    auto compare_exchange = [&](uint32_t stride, bool dir) {
      const float other_key = __shfl_xor_sync(0xffffffffu, key, stride);
      const uint32_t other_slot = __shfl_xor_sync(0xffffffffu, slot, stride);
      const uint32_t other_valid = __shfl_xor_sync(0xffffffffu, valid, stride);
      const bool lower = (lane & stride) == 0;
      const float key_a = lower ? key : other_key, key_b = lower ? other_key : key;
      const bool valid_a = (lower ? valid : other_valid) != 0, valid_b = (lower ? other_valid : valid) != 0;
      const bool swap = (greater_nan_first(key_a, key_b) && valid_a) || !valid_b;
      if (swap == dir) {
        key = other_key;
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
      s_pos[slot] = lane;
      p.candidates[static_cast<int64_t>(n) * kTopK + lane] = s_id[slot];
      p.unary[static_cast<int64_t>(n) * kTopK + lane] = key;
    }
  }
  __syncthreads();  // B: sorted positions

  // keys[n, c] = succ[id_c]; pred_h[n + 1, c] = bf16(pred[id_c] * hidden[b, e + 1]) for e + 1 < slots;
  // slot 0 also writes pred_h[n, p] = bf16(pred[anchor_b] * hidden[b, 0]) for every p.
#pragma unroll
  for (int q = 0; q < kPasses; ++q) {
    const uint32_t c = s_pos[row0 + q * kRowsPerPass];
    reinterpret_cast<uint4*>(p.keys + (static_cast<int64_t>(n) * kTopK + c) * kRank)[chunk] = succ[q];
    if (has_next) {
      reinterpret_cast<uint4*>(p.pred_h + ((static_cast<int64_t>(n) + 1) * kTopK + c) * kRank)[chunk] =
          mul_bf16x8(pred[q], h_next);
    }
    if (e == 0) {
      reinterpret_cast<uint4*>(p.pred_h + (static_cast<int64_t>(n) * kTopK + row0 + q * kRowsPerPass) * kRank)[chunk] =
          anchor_row;
    }
  }
}

template <int kWidth, int kTopK, int kRank, typename IdT>
struct DFlashSelectorSelect {
  /// \brief Select every draft row's top-kTopK gathered candidates (torch.topk order) and write the lattice
  ///        GEMM operands; see the file comment.
  /// \param vals fp32 [rows, kWidth] gathered candidate logits (rows = bs * slots).
  /// \param ids IdT [rows, kWidth] gathered global token ids.
  /// \param hidden bf16 [bs, slots, kRank] hidden_projection output.
  /// \param anchors int64 [bs] (any stride) anchor tokens.
  /// \param pred_table bf16 [vocab, kRank]; \param succ_table bf16 [vocab, kRank].
  /// \param candidates int64 [rows, kTopK]; \param unary fp32 [rows, kTopK].
  /// \param pred_h bf16 [rows, kTopK, kRank]; \param keys bf16 [rows, kTopK, kRank].
  static void
  run(tvm::ffi::TensorView vals,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView anchors,
      tvm::ffi::TensorView pred_table,
      tvm::ffi::TensorView succ_table,
      tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView unary,
      tvm::ffi::TensorView pred_h,
      tvm::ffi::TensorView keys) {
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
    TensorMatcher({Vocab, kRank})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(pred_table)
        .verify(succ_table);
    TensorMatcher({Rows, kTopK}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({Rows, kTopK}).with_dtype<fp32_t>().with_device(device).verify(unary);
    TensorMatcher({Rows, kTopK, kRank})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(pred_h)
        .verify(keys);
    const int64_t rows = Rows.unwrap(), bs = Bs.unwrap(), slots = Slots.unwrap();
    CHECK_HOST(rows == bs * slots) << "rows " << rows << " != bs " << bs << " x slots " << slots;
    if (rows == 0) return;
    const auto params = dflash_selector::SelectParams{
        .vals = static_cast<const float*>(vals.data_ptr()),
        .ids = ids.data_ptr(),
        .hidden = static_cast<const bf16_t*>(hidden.data_ptr()),
        .anchors = static_cast<const int64_t*>(anchors.data_ptr()),
        .pred_table = static_cast<const bf16_t*>(pred_table.data_ptr()),
        .succ_table = static_cast<const bf16_t*>(succ_table.data_ptr()),
        .candidates = static_cast<int64_t*>(candidates.data_ptr()),
        .unary = static_cast<float*>(unary.data_ptr()),
        .pred_h = static_cast<bf16_t*>(pred_h.data_ptr()),
        .keys = static_cast<bf16_t*>(keys.data_ptr()),
        .anchor_stride = AnchorStride.unwrap(),
        .vocab = Vocab.unwrap(),
        .slots = static_cast<uint32_t>(slots),
    };
    LaunchKernel(static_cast<uint32_t>(rows), dflash_selector::kThreads, device.unwrap())(
        dflash_selector_select_kernel<kWidth, kTopK, kRank, IdT>, params);
  }
};

// The rank's local top-k of its logits shard, as flashinfer.top_k(logits, k, sorted=True, deterministic=True) returns
// it, already widened for the gather: int64 global ids (shard index + offset) and FP32 values. flashinfer's
// single-CTA deterministic radix kernel (1024 threads, topk.cuh RadixCollectIndicesDeterministic) collects the keys
// above the pivot and then the pivot ties in thread-strided order (index i belongs to thread i % 1024, which visits
// i / 1024 in order), and StableSortTopKByValueKernel sorts by the ordered BF16 key, stable. So the result is the first
// k under (ordered key desc, i % 1024 asc, i / 1024 asc); one 32-bit key per element encodes exactly that order.
// kIndexOrder = flashinfer's deterministic FilteredTopK path (BF16 rows > len / 256): the same set (its pivot ties are
// collected thread-strided over 1024 threads too, DeterministicThreadStridedCollect), but FinalizeTopKIndicesKernel
// sorts the set by index before the stable value sort, so equal values come out in index order.
namespace dflash_selector {

inline constexpr uint32_t kLocalThreads = 1024;     // flashinfer's BLOCK_THREADS: fixes its tie order
inline constexpr uint32_t kLocalCandidates = 512;   // keys at or above the warp-maximum bound kept in shared memory

// RadixTopKTraits<nv_bfloat16>::ToOrdered / FromOrdered.
SGL_DEVICE uint32_t bf16_ordered(uint32_t bits) {
  return (bits & 0x8000u) ? (~bits & 0xffffu) : (bits ^ 0x8000u);
}
SGL_DEVICE uint32_t bf16_from_ordered(uint32_t ordered) {
  return (ordered & 0x8000u) ? (ordered ^ 0x8000u) : (~ordered & 0xffffu);
}

struct LocalTopkParams {
  const uint16_t* logits;  // [rows, row_stride] BF16 bits
  float* vals;             // [rows, k]
  int64_t* ids;            // [rows, k]
  int64_t row_stride;
  int64_t offset;
  uint32_t len;
};

}  // namespace dflash_selector

// grid rows, 1024 threads. Thread t holds the keys of indices t + 1024 j. The kTopK-th largest warp maximum bounds the
// kTopK-th largest key from below (kTopK warps each hold a key at or above it), so the keys at or above it are the
// candidates; they are ranked by counting (keys are distinct). More than kLocalCandidates candidates (thousands of BF16
// ties at the top) take an exact block-wide radix select instead.
template <int kPerThread, int kTopK, bool kIndexOrder>
__global__ void __launch_bounds__(dflash_selector::kLocalThreads)
    dflash_selector_local_topk_kernel(const __grid_constant__ dflash_selector::LocalTopkParams p) {
  using namespace dflash_selector;
  constexpr uint32_t kWarps = kLocalThreads / 32;
  static_assert(kPerThread >= 2 && kPerThread * kLocalThreads <= 0x10000u, "positions fit the key's low 16 bits");
  static_assert(kTopK <= static_cast<int>(kWarps) && kTopK <= static_cast<int>(kLocalCandidates), "bound needs kTopK warps");

  __shared__ uint32_t s_warp[kWarps];
  __shared__ uint32_t s_cand[kLocalCandidates];
  __shared__ uint32_t s_top[kTopK];
  __shared__ uint32_t s_count, s_bound;

  const uint32_t row = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
  const uint16_t* x = p.logits + static_cast<int64_t>(row) * p.row_stride;
  // Key: ordered BF16 << 16 | (0xffff - collect position), collect position = (i % 1024) * kPerThread + i / 1024.
  uint32_t key[kPerThread];
  uint32_t local_max = 0u;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const uint32_t i = t + static_cast<uint32_t>(j) * kLocalThreads;
    key[j] = i < p.len ? (bf16_ordered(x[i]) << 16) | (0xffffu - (t * kPerThread + j)) : 0u;
    local_max = max(local_max, key[j]);
  }
  if (t == 0) s_count = 0u;
  const uint32_t warp_max = __reduce_max_sync(0xffffffffu, local_max);
  if (lane == 0) s_warp[warp] = warp_max;
  __syncthreads();
  if (t < kWarps) {
    const uint32_t m = s_warp[t];
    uint32_t above = 0;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) above += s_warp[w] > m;
    if (above == kTopK - 1) s_bound = m;
  }
  __syncthreads();
  const uint32_t bound = s_bound;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    if (key[j] >= bound) {
      const uint32_t pos = atomicAdd(&s_count, 1u);
      if (pos < kLocalCandidates) s_cand[pos] = key[j];
    }
  }
  __syncthreads();
  uint32_t count = s_count;
  if (count > kLocalCandidates) {
    // The kTopK-th largest key exactly, one bit per round over every key of the row.
    uint32_t prefix = 0u, mask = 0u, rank = kTopK;
#pragma unroll 1
    for (int bit = 31; bit >= 0; --bit) {
      const uint32_t m2 = mask | (1u << bit), want = prefix | (1u << bit);
      uint32_t cnt = 0;
#pragma unroll
      for (int j = 0; j < kPerThread; ++j) cnt += (key[j] & m2) == want;
      cnt = __reduce_add_sync(0xffffffffu, cnt);
      if (lane == 0) s_warp[warp] = cnt;
      __syncthreads();
      uint32_t total = 0;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) total += s_warp[w];
      __syncthreads();
      if (total >= rank) {
        prefix = want;
      } else {
        rank -= total;
      }
      mask = m2;
    }
    if (t == 0) s_count = 0u;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      if (key[j] >= prefix) s_cand[atomicAdd(&s_count, 1u)] = key[j];
    }
    __syncthreads();
    count = kTopK;
  }
  if (t < count) {
    const uint32_t c = s_cand[t];
    uint32_t above = 0;
    for (uint32_t u = 0; u < count; ++u) above += s_cand[u] > c;
    if (above < kTopK) s_top[above] = c;
  }
  __syncthreads();
  if (t < kTopK) {
    const uint32_t k = s_top[t];
    const uint32_t pos = 0xffffu - (k & 0xffffu);
    const uint32_t i = pos / kPerThread + (pos % kPerThread) * kLocalThreads;
    uint32_t slot = t;
    if constexpr (kIndexOrder) {
      // Rank in (ordered key desc, index asc) among the kTopK selected keys (distinct indices).
      slot = 0u;
#pragma unroll
      for (int u = 0; u < kTopK; ++u) {
        const uint32_t ku = s_top[u];
        const uint32_t pu = 0xffffu - (ku & 0xffffu);
        const uint32_t iu = pu / kPerThread + (pu % kPerThread) * kLocalThreads;
        slot += ((ku >> 16) > (k >> 16)) || ((ku >> 16) == (k >> 16) && iu < i);
      }
    }
    const bf16_t v = __ushort_as_bfloat16(static_cast<unsigned short>(bf16_from_ordered(k >> 16)));
    p.vals[static_cast<int64_t>(row) * kTopK + slot] = __bfloat162float(v);
    p.ids[static_cast<int64_t>(row) * kTopK + slot] = static_cast<int64_t>(i) + p.offset;
  }
}

template <int kPerThread, int kTopK, bool kIndexOrder>
struct DFlashSelectorLocalTopk {
  /// \brief flashinfer.top_k(logits, kTopK, sorted=True, deterministic=True) of every row, as FP32 values and int64
  ///        global ids (index + offset).
  /// \param logits BF16 [rows, len] (row stride free), (kPerThread - 1) * 1024 < len <= kPerThread * 1024.
  /// \param vals FP32 [rows, kTopK]; \param ids int64 [rows, kTopK]; \param offset the shard's first vocabulary id.
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView vals, tvm::ffi::TensorView ids, int64_t offset) {
    using namespace host;
    auto Rows = SymbolicSize{"rows"};
    auto Len = SymbolicSize{"len"};
    auto Stride = SymbolicSize{"row_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({Rows, Len}).with_strides({Stride, 1}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({Rows, kTopK}).with_dtype<fp32_t>().with_device(device).verify(vals);
    TensorMatcher({Rows, kTopK}).with_dtype<int64_t>().with_device(device).verify(ids);
    const int64_t rows = Rows.unwrap(), len = Len.unwrap();
    constexpr int64_t kT = dflash_selector::kLocalThreads;
    CHECK_HOST(len > (kPerThread - 1) * kT && len <= kPerThread * kT) << "len " << len << " for kPerThread " << kPerThread;
    if (rows == 0) return;
    const auto params = dflash_selector::LocalTopkParams{
        .logits = static_cast<const uint16_t*>(logits.data_ptr()),
        .vals = static_cast<float*>(vals.data_ptr()),
        .ids = static_cast<int64_t*>(ids.data_ptr()),
        .row_stride = Stride.unwrap(),
        .offset = offset,
        .len = static_cast<uint32_t>(len),
    };
    LaunchKernel(static_cast<uint32_t>(rows), dflash_selector::kLocalThreads, device.unwrap())(
        dflash_selector_local_topk_kernel<kPerThread, kTopK, kIndexOrder>, params);
  }
};

}  // namespace sglang
