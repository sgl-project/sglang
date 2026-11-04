#pragma once
#include <sgl_kernel/utils.cuh>

namespace sglang::deterministic_topk_detail {

SGL_DEVICE uint32_t ordered_key(float value) {
  uint32_t bits = __float_as_uint(value), key;
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  const uint32_t flip = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  asm("xor.b32 %0, %1, %2;" : "=r"(key) : "r"(bits), "r"(flip));
  return key;
}

template <typename T>
SGL_DEVICE T warp_exclusive(T value) {
  T inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const T other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if ((threadIdx.x & 31) >= offset) inclusive += other;
  }
  return inclusive - value;
}

template <int kThreads, int kRounds>
SGL_DEVICE uint32_t publish_prefixes(uint32_t* counts, int rounds) {
  constexpr int kWarps = kThreads / 32;
  const int lane = threadIdx.x & 31;
  uint32_t carry = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
#pragma unroll
    for (int r = 0; r < kRounds; ++r) if (r < rounds) {
      const auto mine = lane < kWarps ? counts[r * kWarps + lane] : 0u;
      const auto prior = warp_exclusive(mine);
      if (lane < kWarps) counts[r * kWarps + lane] = carry + prior;
      carry += __shfl_sync(0xffffffffu, prior + mine, kWarps - 1);
    }
  }
  return carry;
}

SGL_DEVICE void histogram_add_short(int* histogram, uint32_t bin) {
  const unsigned mask = __activemask();
  const unsigned peers = __match_any_sync(mask, bin);
  if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(histogram + bin, __popc(peers));
}

SGL_DEVICE void histogram_add(int* histogram, uint32_t bin) {
  const unsigned mask = __activemask();
  const int leader = __ffs(mask) - 1;
  const unsigned first = __shfl_sync(mask, bin, leader);
  if (__ballot_sync(mask, bin == first) != mask) atomicAdd(histogram + bin, 1);
  else if ((threadIdx.x & 31) == leader) atomicAdd(histogram + bin, __popc(mask));
}

// Monotone FP32-key buckets retain three fraction bits between saturated tails.
SGL_DEVICE uint32_t coarse_key(uint32_t key) {
  const int sign = int(key >> 31), begin = sign * 256;
  return uint32_t(min(begin + 255, max(begin, int(key >> 20) - 896 - sign * 1792)));
}

template <int kThreads, int kShortItems>
SGL_DEVICE void emit_short(uint32_t chosen, int* output, uint32_t* warp_sel) {
  constexpr int kWarps = kThreads / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const uint32_t count = __popc(chosen), lane_base = warp_exclusive(count);
  if (lane == 31) warp_sel[warp] = lane_base + count;
  __syncthreads();
  uint32_t slot;
  if constexpr (kThreads == 512) {
    // Counts stay read-only until return; each warp computes its own prefix.
    const uint32_t before = lane < warp ? warp_sel[lane] : 0;
    slot = __reduce_add_sync(0xffffffffu, before) + lane_base;
  } else {
    if (tid < 32) {
      const uint32_t mine = lane < kWarps ? warp_sel[lane] : 0;
      const uint32_t prior = warp_exclusive(mine);
      if (lane < kWarps) warp_sel[lane] = prior;
    }
    __syncthreads();
    slot = warp_sel[warp] + lane_base;
  }
#pragma unroll
  for (int i = 0; i < kShortItems; ++i)
    if ((chosen >> i) & 1u) output[slot++] = i * kThreads + tid;
}

SGL_DEVICE uint32_t coarse_lower(uint32_t digit) {
  if (digit == 0) return 0;
  if (digit == 256) return 0x80000000u;
  return (digit + (digit < 256 ? 896u : 2688u)) << 20;
}

SGL_DEVICE unsigned lowest_bits(unsigned bits, int take) {
  if (take <= 0) return 0;
  if (take >= __popc(bits)) return bits;
  unsigned remaining = bits;
  int index = 0;
#pragma unroll
  for (int step = 16; step > 0; step /= 2) {
    const unsigned lower = remaining & ((1u << step) - 1u);
    const int count = __popc(lower);
    if (take > count) { index += step; remaining >>= step; take -= count; }
    else remaining = lower;
  }
  return bits & (0xffffffffu >> (31 - index));
}

template <int kTopK, int kThreads, bool kWarpScan, bool kShortRow>
SGL_DEVICE void radix_select_impl(const float* input, int* output, int start, int length) {
  constexpr int kBins = 8192, kWarps = kThreads / 32;
  constexpr int kTile = kBins * 16, kWordRounds = kBins / (2 * kThreads);
  extern __shared__ int bins[];
  __shared__ int groups[256], cutoff, remain, boundary_count;
  __shared__ uint32_t min_key, max_key, warp_eq[kWarps], warp_sel[kWarps];
  __shared__ uint32_t total_eq, total_sel;
  __shared__ unsigned long long loser;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  if (kShortRow && length == kTopK + 1) {
    if (tid == 0) loser = ~0ull;
    __syncthreads();
    unsigned long long mine = ~0ull;
    for (int id = tid; id < length; id += kThreads) {
      const auto packed = (static_cast<unsigned long long>(ordered_key(input[start + id])) << 32) | (0xffffffffu - uint32_t(id));
      mine = mine < packed ? mine : packed;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const auto other = __shfl_down_sync(0xffffffffu, mine, offset);
      mine = mine < other ? mine : other;
    }
    if (lane == 0) atomicMin(&loser, mine);
    __syncthreads();
    const int skip = int(0xffffffffu - uint32_t(loser));
    for (int i = tid; i < kTopK; i += kThreads) output[i] = i + (i >= skip);
    return;
  }
  constexpr int kShortItems = 4096 / kThreads;
  constexpr bool short_row = kShortRow;
  uint32_t short_keys[kShortItems];
  if (short_row) {
#pragma unroll
    for (int i = 0; i < kShortItems; ++i) {
      const int id = i * kThreads + tid;
      short_keys[i] = id < length ? ordered_key(input[start + id]) : 0;
    }
  }
  uint32_t prefix = 0, mask = 0;
  int need = kTopK, high = 31, low = 0;
  uint32_t varying = 0xffffffffu;
  uint32_t root_equal = 0xffffffffu, root_above = 0;
  bool cutoff_ready = false, all_boundary = false;
  // Native short rows retain their original bit-consensus root.
  if constexpr (kShortRow && kThreads != 512) {
    uint32_t all = 0xffffffffu, any = 0;
#pragma unroll
    for (int i = 0; i < kShortItems; ++i) if (i * kThreads + tid < length) {
      all &= short_keys[i]; any |= short_keys[i];
    }
    if (tid == 0) { min_key = 0xffffffffu; max_key = 0; }
    __syncthreads();
    all = __reduce_and_sync(0xffffffffu, all);
    any = __reduce_or_sync(0xffffffffu, any);
    if (lane == 0) { atomicAnd(&min_key, all); atomicOr(&max_key, any); }
    __syncthreads();
    all = min_key; any = max_key;
    varying = all ^ any;
    if (!varying) {
      constexpr int full = kTopK / kThreads, extra = kTopK % kThreads;
      const int base = tid * full + min(tid, extra);
      for (int i = 0; i < full + (tid < extra); ++i) output[base + i] = i * kThreads + tid;
      return;
    }
    mask = ~varying; prefix = all;
    high = 31 - __clz(varying); low = __ffs(varying) - 1;
  }
  if constexpr (kShortRow && kThreads == 512) {
    bins[tid] = 0;
    if (tid == 0) { min_key = 0xffffffffu; max_key = 0; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kShortItems; ++i)
      if (i * kThreads + tid < length) histogram_add_short(bins, coarse_key(short_keys[i]));
    __syncthreads();
    const int own = bins[tid];
    const int group_total = __reduce_add_sync(0xffffffffu, own);
    if (lane == 0) groups[warp] = group_total;
    __syncthreads();
    // Every warp tests its own bin range against the immutable group totals.
    const int later_group = lane < kWarps && lane > warp ? groups[lane] : 0;
    const int above = __reduce_add_sync(0xffffffffu, later_group);
    int inclusive = own;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const int other = __shfl_down_sync(0xffffffffu, inclusive, offset);
      if (lane + offset < 32) inclusive += other;
    }
    if (above + inclusive >= need && above + inclusive - own < need) {
      cutoff = tid; remain = need - above - (inclusive - own); boundary_count = own;
    }
    __syncthreads();
    if (boundary_count == remain) {
      const uint32_t lower = coarse_lower(uint32_t(cutoff));
      uint32_t chosen = 0;
#pragma unroll
      for (int i = 0; i < kShortItems; ++i)
        if (i * kThreads + tid < length && short_keys[i] >= lower) chosen |= 1u << i;
      emit_short<kThreads, kShortItems>(chosen, output, warp_sel);
      return;
    }
    need = remain;
    root_equal = 0;
    uint32_t all = 0xffffffffu, any = 0;
#pragma unroll
    for (int i = 0; i < kShortItems; ++i) {
      const uint32_t key = short_keys[i], digit = coarse_key(key);
      const bool valid = i * kThreads + tid < length;
      if (valid && digit > uint32_t(cutoff)) root_above |= 1u << i;
      if (valid && digit == uint32_t(cutoff)) {
        root_equal |= 1u << i; all &= key; any |= key;
      }
    }
    all = __reduce_and_sync(0xffffffffu, all);
    any = __reduce_or_sync(0xffffffffu, any);
    if (lane == 0) { atomicAnd(&min_key, all); atomicOr(&max_key, any); }
    __syncthreads();
    all = min_key; any = max_key;
    varying = all ^ any; prefix = all; mask = ~varying;
    cutoff_ready = varying == 0;
    if (!cutoff_ready) { high = 31 - __clz(varying); low = __ffs(varying) - 1; }
  }
  for (int pass = 0; pass < (short_row ? 4 : 3) && !cutoff_ready; ++pass) {
    const int span = high - low + 1;
    const int width = short_row ? min(9, span) : min(13, high + 1);
    const int shift = high - width + 1;
    const int digit_bins = 1 << width, group_count = (digit_bins + 31) / 32;
    const uint32_t digit_mask = uint32_t(digit_bins - 1);
    for (int b = tid; b < digit_bins; b += kThreads) bins[b] = 0;
    if (!short_row && tid == 0) { min_key = 0xffffffffu; max_key = 0; }
    __syncthreads();
    uint32_t lo = 0xffffffffu, hi = 0;
    if (short_row) {
#pragma unroll
      for (int i = 0; i < kShortItems; ++i) {
        const uint32_t key = short_keys[i];
        if (i * kThreads + tid < length && ((root_equal >> i) & 1u) && (key & mask) == prefix) {
          histogram_add_short(bins, (key >> shift) & digit_mask);
        }
      }
    } else {
      for (int base = 0; base < length; base += 4 * kThreads) {
        float values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int id = base + i * kThreads + tid;
          values[i] = id < length ? input[start + id] : 0;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t key = ordered_key(values[i]);
          if (base + i * kThreads + tid < length && (key & mask) == prefix) {
            lo = min(lo, key); hi = max(hi, key);
            histogram_add(bins, (key >> shift) & digit_mask);
          }
        }
      }
    }
    if (!short_row) {
      lo = __reduce_min_sync(0xffffffffu, lo);
      hi = __reduce_max_sync(0xffffffffu, hi);
      if (lane == 0) { atomicMin(&min_key, lo); atomicMax(&max_key, hi); }
    }
    __syncthreads();
    if (!short_row) {
      lo = min_key; hi = max_key;
      if (lo == hi) {
        if (pass == 0 && (kTopK <= 1024 || length <= 32 * kThreads)) {
          for (int i = tid; i < kTopK; i += kThreads) output[i] = i;
          return;
        }
        prefix = lo; mask = 0xffffffffu;
        break;
      }
    }
    if constexpr (kShortRow && kThreads == 512) {
      const int own = tid < digit_bins ? bins[tid] : 0;
      const int group_total = __reduce_add_sync(0xffffffffu, own);
      if (lane == 0) groups[warp] = group_total;
      __syncthreads();
      // Each warp tests its bin range against immutable higher-group counts.
      const int later = lane < group_count && lane > warp ? groups[lane] : 0;
      const int above = __reduce_add_sync(0xffffffffu, later);
      int inclusive = own;
#pragma unroll
      for (int offset = 1; offset < 32; offset *= 2) {
        const int other = __shfl_down_sync(0xffffffffu, inclusive, offset);
        if (lane + offset < 32) inclusive += other;
      }
      if (tid < digit_bins && above + inclusive >= need && above + inclusive - own < need) {
        cutoff = tid; remain = need - above - (inclusive - own); boundary_count = own;
      }
    } else {
      for (int g = warp; g < group_count; g += kWarps) {
        const int b = g * 32 + lane;
        int count = b < digit_bins ? bins[b] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          count += __shfl_down_sync(0xffffffffu, count, offset);
        if (lane == 0) groups[g] = count;
      }
      __syncthreads();
      if (tid < 32) {
        int above = 0, group = 0;
        for (int tile = (group_count - 1) / 32; tile >= 0; --tile) {
          const int pos = tile * 32 + lane;
          int inclusive = pos < group_count ? groups[pos] : 0;
          const int own = inclusive;
#pragma unroll
          for (int offset = 1; offset < 32; offset *= 2) {
            const int other = __shfl_down_sync(0xffffffffu, inclusive, offset);
            if (lane + offset < 32) inclusive += other;
          }
          const int total = __shfl_sync(0xffffffffu, inclusive, 0);
          if (above + total >= need) {
            const unsigned hit = __ballot_sync(0xffffffffu, above + inclusive >= need && above + inclusive - own < need);
            const int winner = __ffs(hit) - 1;
            group = tile * 32 + winner;
            above += __shfl_sync(0xffffffffu, inclusive - own, winner);
            break;
          }
          above += total;
        }
        const int b = group * 32 + lane;
        const int own = b < digit_bins ? bins[b] : 0;
        int inclusive = own;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
          const int other = __shfl_down_sync(0xffffffffu, inclusive, offset);
          if (lane + offset < 32) inclusive += other;
        }
        if (above + inclusive >= need && above + inclusive - own < need) {
          cutoff = b; remain = need - above - (inclusive - own); boundary_count = own;
        }
      }
    }
    __syncthreads();
    prefix |= uint32_t(cutoff) << shift;
    mask |= digit_mask << shift;
    need = remain;
    if (boundary_count == need || shift == 0) {
      all_boundary = boundary_count == need;
      break;
    }
    if (short_row) {
      uint32_t all = 0xffffffffu, any = 0;
#pragma unroll
      for (int i = 0; i < kShortItems; ++i) {
        const uint32_t key = short_keys[i];
        if (i * kThreads + tid < length && ((root_equal >> i) & 1u) && (key & mask) == prefix) { all &= key; any |= key; }
      }
      if (tid == 0) { min_key = 0xffffffffu; max_key = 0; }
      __syncthreads();
      all = __reduce_and_sync(0xffffffffu, all);
      any = __reduce_or_sync(0xffffffffu, any);
      if (lane == 0) { atomicAnd(&min_key, all); atomicOr(&max_key, any); }
      __syncthreads();
      all = min_key; any = max_key;
      varying = all ^ any; prefix = all; mask = ~varying;
      if (!varying) break;
      high = 31 - __clz(varying);
      low = __ffs(varying) - 1;
    } else {
      high = min(shift - 1, 31 - __clz(lo ^ hi));
      const uint32_t known = 0xffffffffu << (high + 1);
      prefix |= lo & known & ~mask;
      mask |= known;
    }
  }
  if (short_row) {
    auto* counts = reinterpret_cast<uint32_t*>(bins);
    uint32_t chosen = root_above;
    if (all_boundary) {
#pragma unroll
      for (int i = 0; i < kShortItems; ++i)
        if (i * kThreads + tid < length && ((root_equal >> i) & 1u) && (short_keys[i] & mask) >= prefix) chosen |= 1u << i;
    } else {
#pragma unroll
      for (int i = 0; i < kShortItems; ++i) if (i * kThreads < length) {
        const bool equal = i * kThreads + tid < length && ((root_equal >> i) & 1u) && (short_keys[i] & mask) == prefix;
        const unsigned e = __ballot_sync(0xffffffffu, equal);
        if (lane == 0) counts[i * kWarps + warp] = uint32_t(__popc(e));
      }
      publish_prefixes<kThreads, kShortItems>(counts, (length + kThreads - 1) / kThreads);
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kShortItems; ++i) if (i * kThreads < length) {
        const int id = i * kThreads + tid;
        const uint32_t key = short_keys[i] & mask;
        const bool equal = id < length && ((root_equal >> i) & 1u) && key == prefix;
        const unsigned e = __ballot_sync(0xffffffffu, equal);
        const uint32_t eq_before = counts[i * kWarps + warp] + __popc(e & ((1u << lane) - 1u));
        if (id < length && ((root_equal >> i) & 1u) && (key > prefix || (equal && eq_before < uint32_t(need)))) chosen |= 1u << i;
      }
    }
    emit_short<kThreads, kShortItems>(chosen, output, warp_sel);
    return;
  }
  // Reuse histogram storage for predicates; scores are read once for emission.
  uint32_t* bits = reinterpret_cast<uint32_t*>(bins);
  int output_base = 0;
  for (int begin = 0; begin < length; begin += kTile) {
    const int n = min(kTile, length - begin), words = (n + 31) / 32;
    const int rounds = (words + kThreads - 1) / kThreads;
    for (int off = 0; off < n; off += 4 * kThreads) {
      float values[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int id = off + i * kThreads + tid;
        values[i] = id < n ? input[start + begin + id] : 0;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int id = off + i * kThreads + tid;
        const uint32_t key = ordered_key(values[i]) & mask;
        const unsigned a = __ballot_sync(0xffffffffu, id < n && key > prefix);
        const unsigned e = __ballot_sync(0xffffffffu, id < n && key == prefix);
        if (lane == 0 && id < n) { bits[2 * (id / 32)] = a; bits[2 * (id / 32) + 1] = e; }
      }
    }
    __syncthreads();
    uint32_t a[kWordRounds], e[kWordRounds], eq_count = 0;
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = warp * rounds * 32 + r * 32 + lane;
      const bool valid = r < rounds && word < words;
      a[r] = valid ? bits[2 * word] : 0;
      e[r] = valid ? bits[2 * word + 1] : 0;
      eq_count += __popc(e[r]);
    }
    const uint32_t warp_equal = __reduce_add_sync(0xffffffffu, eq_count);
    if (lane == 0) warp_eq[warp] = warp_equal;
    __syncthreads();
    if (tid < 32) {
      const uint32_t count = lane < kWarps ? warp_eq[lane] : 0;
      const uint32_t prior = warp_exclusive(count);
      if (lane < kWarps) warp_eq[lane] = prior;
      if (lane == kWarps - 1) total_eq = prior + count;
    }
    __syncthreads();
    uint32_t equal_before = warp_eq[warp], chosen_count = 0;
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      if (r < rounds) {
        const uint32_t count = __popc(e[r]);
        const uint32_t prior = warp_exclusive(count);
        const int take = max(0, need - int(equal_before + prior));
        a[r] |= lowest_bits(e[r], take);
        chosen_count += __popc(a[r]);
        equal_before += __shfl_sync(0xffffffffu, prior + count, 31);
      }
    }
    const uint32_t lane_base = warp_exclusive(chosen_count);
    if (lane == 31) warp_sel[warp] = lane_base + chosen_count;
    __syncthreads();
    if (tid < 32) {
      const uint32_t count = lane < kWarps ? warp_sel[lane] : 0;
      const uint32_t prior = warp_exclusive(count);
      if (lane < kWarps) warp_sel[lane] = prior;
      if (lane == kWarps - 1) total_sel = prior + count;
    }
    __syncthreads();
    uint32_t slot = uint32_t(output_base) + warp_sel[warp] + lane_base;
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = warp * rounds * 32 + r * 32 + lane;
      unsigned chosen = a[r];
      while (chosen) {
        const int bit = __ffs(chosen) - 1;
        output[slot++] = begin + word * 32 + bit;
        chosen &= chosen - 1;
      }
    }
    __syncthreads();
    need = max(0, need - int(total_eq));
    output_base += int(total_sel);
  }
}

template <int kTopK, int kThreads, bool kWarpScan, bool kLongRows = false>
SGL_DEVICE void radix_select(const float* input, int* output, int start, int length) {
  if (length <= 4096) radix_select_impl<kTopK, kThreads, kWarpScan, true>(input, output, start, length);
  else radix_select_impl<kTopK, kThreads, kWarpScan, false>(input, output, start, length);
}

}  // namespace sglang::deterministic_topk_detail
