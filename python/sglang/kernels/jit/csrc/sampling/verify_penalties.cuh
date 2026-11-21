#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/ptx/async.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sglang {

struct VerifyPenaltiesParams {
  void* __restrict__ logits;
  int64_t logits_stride;
  const void* __restrict__ draft_tokens;
  int64_t draft_tokens_stride;
  bool draft_tokens_are_int64;
  // Both null for a chain, whose node j has parent j - 1.
  const int64_t* __restrict__ next_child;
  const int64_t* __restrict__ next_sibling;
  int64_t tree_stride;
  const int64_t* __restrict__ batch_indices;
  const int64_t* __restrict__ penalty_indices;
  const int32_t* __restrict__ counts;
  const int32_t* __restrict__ lens;
  // Null when no request of the step uses that penalty.
  const float* __restrict__ frequency;
  const float* __restrict__ presence;
  const float* __restrict__ repetition;
  const int32_t* __restrict__ min_new_tokens;
  const int64_t* __restrict__ stop_token_ids;
  int64_t num_stop_tokens;
  int32_t width;
  int32_t vocab_size;
  int32_t rows_per_cta;
  // Logits and counts rows are 16-byte aligned.
  bool vectorized;
};

constexpr uint32_t kVerifyPenaltiesThreads = 128;
constexpr int32_t kVerifyPenaltiesRowsInFlight = 4;
constexpr uint32_t kVerifyPenaltiesVectorBytes = 16;

template <typename T>
struct VerifyPenaltiesLayout {
  static constexpr uint32_t kVec = kVerifyPenaltiesVectorBytes / sizeof(T);
  static constexpr uint32_t kChunk = kVerifyPenaltiesThreads * kVec;
  static constexpr uint32_t kChunkWords = kChunk / 32;
  static_assert(32 % kVec == 0, "a thread's bitmap bits sit in one word");
};

SGL_DEVICE void verify_penalties_mark(uint32_t* bits, int64_t token, int32_t chunk_begin, uint32_t chunk) {
  const int64_t local = token - chunk_begin;
  if (0 <= local && local < chunk) {
    atomicOr(&bits[local / 32], 1u << (local % 32));
  }
}

template <int N>
SGL_DEVICE void verify_penalties_wait_copies() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

SGL_DEVICE void verify_penalties_commit_copies() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

/// The division of torch.compile's `apply_scaling_penalties`, which is not IEEE-rounded.
SGL_DEVICE float verify_penalties_div(float x, float y) {
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
  return result;
}

struct VerifyPenalty {
  float additive;
  float scaling;
};

struct VerifyPenaltyCoefficients {
  float frequency;
  float presence;
  float repetition;
};

/// The _rn intrinsics keep the compiler from contracting these into an FMA, to
/// match the eager tables of `_compute_logit_penalties`.
SGL_DEVICE VerifyPenalty
verify_penalty(const VerifyPenaltiesParams& p, const VerifyPenaltyCoefficients& c, int32_t count, bool blocked) {
  float additive = 0.0f;
  if (p.frequency != nullptr) additive = __fadd_rn(additive, -__fmul_rn(static_cast<float>(count), c.frequency));
  if (p.presence != nullptr) additive = __fadd_rn(additive, -(count > 0 ? c.presence : 0.0f));
  if (p.min_new_tokens != nullptr) additive = __fadd_rn(additive, blocked ? -INFINITY : 0.0f);
  return {additive, count > 0 ? c.repetition : 1.0f};
}

SGL_DEVICE float verify_scale(float x, float scaling) {
  return x < 0.0f ? __fmul_rn(x, scaling) : verify_penalties_div(x, scaling);
}

/// One logit as non-speculative decoding penalizes it: `logits.add_` rounded to
/// T, then the compiled `apply_scaling_penalties`, rounded on store.
template <typename T>
SGL_DEVICE T verify_penalize(T logit, VerifyPenalty penalty, bool has_additive, bool has_repetition) {
  float x = device::cast<float>(logit);
  if (has_additive) x = device::cast<float>(device::cast<T>(__fadd_rn(x, penalty.additive)));
  if (has_repetition) x = verify_scale(x, penalty.scaling);
  return device::cast<T>(x);
}

/// `verify_penalize` of two logits converted as a pair; bit i of `scaled` marks logit i scaled.
template <typename T>
SGL_DEVICE packed_t<T> verify_penalize_pair(
    packed_t<T> logits, VerifyPenalty first, VerifyPenalty second, bool has_additive, uint32_t scaled) {
  if (has_additive) {
    const fp32x2_t x = device::cast<fp32x2_t>(logits);
    logits = device::cast<packed_t<T>>(fp32x2_t{__fadd_rn(x.x, first.additive), __fadd_rn(x.y, second.additive)});
  }
  if (scaled != 0) {
    fp32x2_t x = device::cast<fp32x2_t>(logits);
    if (scaled & 1u) x.x = verify_scale(x.x, first.scaling);
    if (scaled & 2u) x.y = verify_scale(x.y, second.scaling);
    logits = device::cast<packed_t<T>>(x);
  }
  return logits;
}

/// One CTA per (vocab chunk, penalized request, group of `rows_per_cta` rows).
/// A row counts its request's history plus the in-vocab tokens on its path (root
/// exclusive, the row's node inclusive). Entries off every path and outside the
/// stop set take the history's penalty in all rows, applied as whole vectors; a
/// bitmap marks the rest, which are then penalized per row.
template <typename T>
__global__ void verify_penalties_kernel(const __grid_constant__ VerifyPenaltiesParams p) {
  using Layout = VerifyPenaltiesLayout<T>;
  constexpr uint32_t kVec = Layout::kVec;
  constexpr uint32_t kChunk = Layout::kChunk;
  using LogitVec = device::AlignedVector<packed_t<T>, kVec / 2>;
  using CountVec = device::AlignedVector<int32_t, 4>;

  extern __shared__ int32_t smem[];
  int32_t* parent = smem;              // [width], -1 at the root and at nodes off the tree
  int32_t* token = smem + p.width;     // [width], -1 when out of vocab
  int32_t* depth = token + p.width;    // [width], in-vocab tokens on the node's path
  int32_t* sibling = depth + p.width;  // [width], next sibling, -1 for none
  __shared__ uint32_t path_bits[Layout::kChunkWords];
  __shared__ uint32_t stop_bits[Layout::kChunkWords];
  __shared__ uint32_t own_entries[kChunk];
  __shared__ uint32_t num_own_entries;

  const uint32_t tid = threadIdx.x;
  const int32_t width = p.width;
  const int32_t vocab_size = p.vocab_size;
  const int64_t request = blockIdx.y;
  const int64_t batch_index = p.batch_indices[request];
  const int64_t penalty_index = p.penalty_indices[request];
  const int32_t chunk_begin = blockIdx.x * kChunk;
  const int32_t row_begin = blockIdx.z * p.rows_per_cta;
  const int32_t row_end = min(width, row_begin + p.rows_per_cta);
  const uint32_t local = tid * kVec;
  const int32_t v0 = chunk_begin + local;
  const bool whole_vector = p.vectorized && v0 + static_cast<int32_t>(kVec) <= vocab_size;

  // Loads that need no tree go first, to overlap the tree walk.
  const int32_t* counts = p.counts + penalty_index * vocab_size;
  int32_t history_count[kVec];
  if (whole_vector) {
#pragma unroll
    for (uint32_t i = 0; i < kVec / 4; ++i) {
      CountVec vec;
      vec.load(counts + v0, i);
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e)
        history_count[i * 4 + e] = vec[e];
    }
  } else {
#pragma unroll
    for (uint32_t e = 0; e < kVec; ++e)
      history_count[e] = v0 + e < vocab_size ? counts[v0 + e] : 0;
  }
  const VerifyPenaltyCoefficients coefficients = {
      .frequency = p.frequency != nullptr ? p.frequency[penalty_index] : 0.0f,
      .presence = p.presence != nullptr ? p.presence[penalty_index] : 0.0f,
      .repetition = p.repetition != nullptr ? p.repetition[penalty_index] : 1.0f,
  };
  const int32_t min_new_tokens = p.min_new_tokens != nullptr ? p.min_new_tokens[penalty_index] : 0;
  const int32_t history_len = p.lens[penalty_index];

  T* rows = static_cast<T*>(p.logits) + batch_index * width * p.logits_stride + v0;
  const auto load_row_elements = [&](LogitVec& x, int32_t j) {
    const T* row = rows + j * p.logits_stride;
    T* elements = reinterpret_cast<T*>(&x);
#pragma unroll
    for (uint32_t e = 0; e < kVec; ++e) {
      if (v0 + e < vocab_size) elements[e] = row[e];
    }
  };
  const auto store_row = [&](const LogitVec& x, int32_t j) {
    T* row = rows + j * p.logits_stride;
    if (whole_vector) {
      x.store(row);
    } else {
      const T* elements = reinterpret_cast<const T*>(&x);
#pragma unroll
      for (uint32_t e = 0; e < kVec; ++e) {
        if (v0 + e < vocab_size) row[e] = elements[e];
      }
    }
  };
  // Slot r stages rows j with (j - row_begin) % slots == r: one cp.async commit
  // group per row, refilled with the row `slots` ahead once penalized.
  __shared__ LogitVec staging[kVerifyPenaltiesRowsInFlight][kVerifyPenaltiesThreads];
  const auto stage_row = [&](int32_t r, int32_t j) {
    if (whole_vector && j < row_end) {
      device::ptx::cp_async_16B(
          static_cast<uint32_t>(__cvta_generic_to_shared(&staging[r][tid])), rows + j * p.logits_stride);
    }
    verify_penalties_commit_copies();
  };
#pragma unroll
  for (int32_t r = 0; r < kVerifyPenaltiesRowsInFlight; ++r) {
    stage_row(r, row_begin + r);
  }

  for (uint32_t i = tid; i < Layout::kChunkWords; i += kVerifyPenaltiesThreads) {
    path_bits[i] = 0;
    stop_bits[i] = 0;
  }
  if (tid == 0) num_own_entries = 0;
  // depth[a] holds node a's first child until the tree is linked.
  const int64_t tokens_offset = batch_index * p.draft_tokens_stride;
  const int64_t tree_offset = batch_index * p.tree_stride;
  for (int32_t a = tid; a < width; a += kVerifyPenaltiesThreads) {
    const int64_t t = p.draft_tokens_are_int64 ? static_cast<const int64_t*>(p.draft_tokens)[tokens_offset + a]
                                               : static_cast<const int32_t*>(p.draft_tokens)[tokens_offset + a];
    token[a] = (0 <= t && t < vocab_size) ? static_cast<int32_t>(t) : -1;
    parent[a] = p.next_child == nullptr ? a - 1 : -1;
    if (p.next_child != nullptr) {
      const int64_t c = p.next_child[tree_offset + a];
      const int64_t s = p.next_sibling[tree_offset + a];
      depth[a] = (0 <= c && c < width) ? static_cast<int32_t>(c) : -1;
      sibling[a] = (0 <= s && s < width) ? static_cast<int32_t>(s) : -1;
    }
  }
  const int64_t* stops = p.stop_token_ids + penalty_index * p.num_stop_tokens;
  const int64_t num_stops = p.min_new_tokens != nullptr ? p.num_stop_tokens : 0;
  const int64_t first_stop = tid < num_stops ? stops[tid] : vocab_size;
  __syncthreads();

  if (p.next_child != nullptr) {
    // Node a is the parent of its first child and of that child's siblings. The
    // step bound keeps a malformed sibling cycle from hanging the kernel.
    for (int32_t a = tid; a < width; a += kVerifyPenaltiesThreads) {
      for (int32_t child = depth[a], step = 0; child >= 0 && step < width; ++step) {
        parent[child] = a;
        child = sibling[child];
      }
    }
    __syncthreads();
  }

  // A node is on some path exactly when it has a parent: it ends its own path.
  for (int32_t j = tid; j < width; j += kVerifyPenaltiesThreads) {
    int32_t d = 0;
    for (int32_t a = j, step = 0; parent[a] >= 0 && step < width; a = parent[a], ++step) {
      d += token[a] >= 0;
    }
    depth[j] = d;
    if (parent[j] >= 0 && token[j] >= 0) verify_penalties_mark(path_bits, token[j], chunk_begin, kChunk);
  }
  // The host pads stop ids with vocab_size, which a partial last chunk covers.
  if (first_stop < vocab_size) verify_penalties_mark(stop_bits, first_stop, chunk_begin, kChunk);
  for (int64_t i = tid + kVerifyPenaltiesThreads; i < num_stops; i += kVerifyPenaltiesThreads) {
    if (stops[i] < vocab_size) verify_penalties_mark(stop_bits, stops[i], chunk_begin, kChunk);
  }
  __syncthreads();

  constexpr uint32_t kThreadMask = (1u << kVec) - 1u;
  const uint32_t on_path = (path_bits[local / 32] >> (local % 32)) & kThreadMask;
  const uint32_t is_stop = (stop_bits[local / 32] >> (local % 32)) & kThreadMask;
  const uint32_t own_penalty = on_path | is_stop;
  if (own_penalty != 0) {
    uint32_t slot = atomicAdd(&num_own_entries, __popc(own_penalty));
    for (uint32_t pending = own_penalty; pending != 0; pending &= pending - 1) {
      own_entries[slot++] = local + __ffs(pending) - 1;
    }
  }
  const bool has_additive = p.frequency != nullptr || p.presence != nullptr || p.min_new_tokens != nullptr;
  const bool has_repetition = p.repetition != nullptr;
  float history_additive[kVec];
  uint32_t seen = 0;
  // After the additive step, whose sum also makes any NaN canonical, a scaling
  // of 1 leaves every bit pattern as it is (checked over all fp32 patterns):
  // the vector pass scales only the entries seen before.
  uint32_t scaled = 0;
#pragma unroll
  for (uint32_t e = 0; e < kVec; ++e) {
    history_additive[e] = verify_penalty(p, coefficients, history_count[e], false).additive;
    if (history_count[e] > 0) seen |= 1u << e;
  }
  if (has_repetition) scaled = has_additive ? seen : kThreadMask;
  const auto history_penalty = [&](uint32_t e) -> VerifyPenalty {
    return {history_additive[e], (seen >> e) & 1u ? coefficients.repetition : 1.0f};
  };

  // Entries with a penalty of their own are left for the per-row pass.
  const auto penalize_row = [&](const LogitVec& x) {
    LogitVec y;
#pragma unroll
    for (uint32_t i = 0; i < kVec / 2; ++i) {
      y[i] = verify_penalize_pair<T>(
          x[i], history_penalty(2 * i), history_penalty(2 * i + 1), has_additive, (scaled >> (2 * i)) & 3u);
    }
    if (own_penalty != 0) {
#pragma unroll
      for (uint32_t e = 0; e < kVec; ++e) {
        if ((own_penalty >> e) & 1u) reinterpret_cast<T*>(&y)[e] = reinterpret_cast<const T*>(&x)[e];
      }
    }
    return y;
  };

  // Not unrolled: the slot is picked at run time to keep code and registers small.
  if (v0 < vocab_size) {
#pragma unroll 1
    for (int32_t j = row_begin; j < row_end; ++j) {
      const int32_t r = (j - row_begin) % kVerifyPenaltiesRowsInFlight;
      LogitVec x;
      if (whole_vector) {
        // Row j's commit group is the oldest of the ones in flight.
        verify_penalties_wait_copies<kVerifyPenaltiesRowsInFlight - 1>();
        x = staging[r][tid];
      } else {
        load_row_elements(x, j);
      }
      const LogitVec y = penalize_row(x);
      stage_row(r, j + kVerifyPenaltiesRowsInFlight);
      store_row(y, j);
    }
  }

  // The barrier makes the vector pass's stores visible.
  __syncthreads();
  const int32_t num_entries = num_own_entries;
  const int32_t num_pairs = num_entries * (row_end - row_begin);
  for (int32_t pair = tid; pair < num_pairs; pair += kVerifyPenaltiesThreads) {
    const uint32_t entry = own_entries[pair % num_entries];
    const int32_t j = row_begin + pair / num_entries;
    const int32_t v = chunk_begin + entry;
    const bool entry_on_path = (path_bits[entry / 32] >> (entry % 32)) & 1u;
    const bool entry_is_stop = (stop_bits[entry / 32] >> (entry % 32)) & 1u;
    T* logit = static_cast<T*>(p.logits) + (batch_index * width + j) * p.logits_stride + v;
    const T original = *logit;
    int32_t count = counts[v];
    if (entry_on_path) {
      for (int32_t a = j, step = 0; parent[a] >= 0 && step < width; a = parent[a], ++step) {
        count += token[a] == v;
      }
    }
    const bool blocked = entry_is_stop && history_len + depth[j] < min_new_tokens;
    *logit = verify_penalize(original, verify_penalty(p, coefficients, count, blocked), has_additive, has_repetition);
  }
}

/// Apply each verify row's own sampling penalties to `logits` (`[bs * width, vocab]`)
/// in place; rows of requests outside `batch_indices` are untouched. Shapes: see
/// `verify_penalties.py`.
template <typename T>
void apply_verify_penalties(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView draft_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> next_child,
    tvm::ffi::Optional<tvm::ffi::TensorView> next_sibling,
    tvm::ffi::TensorView batch_indices,
    tvm::ffi::TensorView penalty_indices,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView lens,
    tvm::ffi::Optional<tvm::ffi::TensorView> frequency,
    tvm::ffi::Optional<tvm::ffi::TensorView> presence,
    tvm::ffi::Optional<tvm::ffi::TensorView> repetition,
    tvm::ffi::Optional<tvm::ffi::TensorView> min_new_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> stop_token_ids) {
  using namespace host;
  using Layout = VerifyPenaltiesLayout<T>;
  SymbolicSize bs{"bs"}, width{"width"}, vocab{"vocab"}, n{"num_penalized"}, num_rows{"num_history_rows"};
  SymbolicSize logits_stride{"logits_stride"}, tokens_stride{"tokens_stride"};
  SymbolicDevice device;
  device.set_options<kDLGPU>();

  TensorMatcher({bs, width})  //
      .with_strides({tokens_stride, 1})
      .with_dtype<int32_t, int64_t>()
      .with_device(device)
      .verify(draft_tokens);
  const int64_t num_logit_rows = bs.unwrap() * width.unwrap();
  TensorMatcher({num_logit_rows, vocab})  //
      .with_strides({logits_stride, 1})
      .with_dtype<T>()
      .with_device(device)
      .verify(logits);
  TensorMatcher({n})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(batch_indices)
      .verify(penalty_indices);
  TensorMatcher({num_rows, vocab})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(counts);
  TensorMatcher({num_rows})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(lens);

  CHECK_HOST(next_child.has_value() == next_sibling.has_value())
      << "apply_verify_penalties: next_child and next_sibling go together";
  const int64_t* next_child_ptr = nullptr;
  const int64_t* next_sibling_ptr = nullptr;
  int64_t tree_stride = 0;
  if (next_child.has_value()) {
    SymbolicSize stride{"tree_stride"};
    TensorMatcher({bs, width})  //
        .with_strides({stride, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(next_child.value())
        .verify(next_sibling.value());
    next_child_ptr = static_cast<const int64_t*>(next_child.value().data_ptr());
    next_sibling_ptr = static_cast<const int64_t*>(next_sibling.value().data_ptr());
    tree_stride = stride.unwrap();
  }

  auto per_row_float = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& t) -> const float* {
    if (!t.has_value()) return nullptr;
    TensorMatcher({num_rows}).with_dtype<float>().with_device(device).verify(t.value());
    return static_cast<const float*>(t.value().data_ptr());
  };
  const float* frequency_ptr = per_row_float(frequency);
  const float* presence_ptr = per_row_float(presence);
  const float* repetition_ptr = per_row_float(repetition);

  CHECK_HOST(min_new_tokens.has_value() == stop_token_ids.has_value())
      << "apply_verify_penalties: min_new_tokens and stop_token_ids go together";
  const int32_t* min_new_tokens_ptr = nullptr;
  const int64_t* stop_token_ids_ptr = nullptr;
  int64_t num_stop_tokens = 0;
  if (min_new_tokens.has_value()) {
    SymbolicSize k{"num_stop_tokens"};
    TensorMatcher({num_rows}).with_dtype<int32_t>().with_device(device).verify(min_new_tokens.value());
    TensorMatcher({num_rows, k}).with_dtype<int64_t>().with_device(device).verify(stop_token_ids.value());
    min_new_tokens_ptr = static_cast<const int32_t*>(min_new_tokens.value().data_ptr());
    stop_token_ids_ptr = static_cast<const int64_t*>(stop_token_ids.value().data_ptr());
    num_stop_tokens = k.unwrap();
  }

  const int64_t num_requests = n.unwrap();
  const int64_t vocab_size = vocab.unwrap();
  const int64_t width_value = width.unwrap();
  if (num_requests == 0 || width_value == 0) return;
  CHECK_HOST(vocab_size < (int64_t{1} << 31)) << "apply_verify_penalties: vocab " << vocab_size << " overflows int32";
  CHECK_HOST(num_requests <= 65535) << "apply_verify_penalties: " << num_requests << " requests exceed grid.y";
  // Keeps the shared memory under the 48 KB default.
  CHECK_HOST(width_value <= 2048) << "apply_verify_penalties: width " << width_value << " exceeds 2048";

  const auto is_aligned = [](const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % kVerifyPenaltiesVectorBytes == 0;
  };
  const bool vectorized = is_aligned(logits.data_ptr()) && logits_stride.unwrap() % Layout::kVec == 0 &&
                          is_aligned(counts.data_ptr()) && vocab_size % 4 == 0;

  // Rows split over grid.z up to one wave of CTAs; each split rereads the history
  // counts. Registers bound residency, so the tree's shared memory is ignored.
  const int32_t device_id = device.unwrap().device_id;
  static runtime::DeviceCacheMap<uint32_t, 0> blocks_per_sm;
  const int64_t wave = int64_t{runtime::get_sm_count(device_id)} * blocks_per_sm.get_cached(device_id, [](int32_t) {
    return runtime::get_blocks_per_sm(verify_penalties_kernel<T>, kVerifyPenaltiesThreads);
  });
  const int64_t num_chunks = div_ceil(vocab_size, int64_t{Layout::kChunk});
  const int64_t splits = std::clamp(wave / (num_chunks * num_requests), int64_t{1}, width_value);
  const int64_t rows_per_cta = div_ceil(width_value, splits);

  const auto params = VerifyPenaltiesParams{
      .logits = logits.data_ptr(),
      .logits_stride = logits_stride.unwrap(),
      .draft_tokens = draft_tokens.data_ptr(),
      .draft_tokens_stride = tokens_stride.unwrap(),
      .draft_tokens_are_int64 = draft_tokens.dtype().bits == 64,
      .next_child = next_child_ptr,
      .next_sibling = next_sibling_ptr,
      .tree_stride = tree_stride,
      .batch_indices = static_cast<const int64_t*>(batch_indices.data_ptr()),
      .penalty_indices = static_cast<const int64_t*>(penalty_indices.data_ptr()),
      .counts = static_cast<const int32_t*>(counts.data_ptr()),
      .lens = static_cast<const int32_t*>(lens.data_ptr()),
      .frequency = frequency_ptr,
      .presence = presence_ptr,
      .repetition = repetition_ptr,
      .min_new_tokens = min_new_tokens_ptr,
      .stop_token_ids = stop_token_ids_ptr,
      .num_stop_tokens = num_stop_tokens,
      .width = static_cast<int32_t>(width_value),
      .vocab_size = static_cast<int32_t>(vocab_size),
      .rows_per_cta = static_cast<int32_t>(rows_per_cta),
      .vectorized = vectorized,
  };
  const dim3 grid(
      static_cast<uint32_t>(num_chunks),
      static_cast<uint32_t>(num_requests),
      static_cast<uint32_t>(div_ceil(width_value, rows_per_cta)));
  const size_t smem_bytes = 4 * width_value * sizeof(int32_t);
  LaunchKernel(grid, kVerifyPenaltiesThreads, device.unwrap(), smem_bytes)(verify_penalties_kernel<T>, params);
}

}  // namespace sglang
