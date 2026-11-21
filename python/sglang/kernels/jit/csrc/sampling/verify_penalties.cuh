#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

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
  // Each null when no request of the step uses that penalty.
  const float* __restrict__ frequency;
  const float* __restrict__ presence;
  const float* __restrict__ repetition;
  const int32_t* __restrict__ min_new_tokens;
  const int64_t* __restrict__ stop_token_ids;  // set with min_new_tokens
  int64_t num_stop_tokens;
  int32_t width;
  int32_t vocab_size;
};

constexpr uint32_t kVerifyPenaltiesThreads = 256;
constexpr uint32_t kVerifyPenaltiesItems = 8;
constexpr uint32_t kVerifyPenaltiesChunk = kVerifyPenaltiesThreads * kVerifyPenaltiesItems;
constexpr uint32_t kVerifyPenaltiesChunkWords = kVerifyPenaltiesChunk / 32;
static_assert(kVerifyPenaltiesChunk % 32 == 0);

SGL_DEVICE void verify_penalties_mark(uint32_t* bits, int64_t token, int32_t chunk_begin) {
  const int64_t local = token - chunk_begin;
  if (0 <= local && local < kVerifyPenaltiesChunk) {
    atomicOr(&bits[local / 32], 1u << (local % 32));
  }
}

SGL_DEVICE bool verify_penalties_marked(const uint32_t* bits, uint32_t local) {
  return (bits[local / 32] >> (local % 32)) & 1u;
}

/// \brief One CTA per (vocab chunk, penalized request); it rewrites that chunk of
/// all `width` verify rows of the request.
///
/// A row counts its request's history plus the in-vocab tokens of its path: the
/// nodes from the root, exclusive, to the row's node, inclusive. Only tokens on
/// some path of the request need more than the history count, so a bitmap of
/// them keeps the path walk off every other element.
template <typename T>
__global__ void verify_penalties_kernel(const __grid_constant__ VerifyPenaltiesParams p) {
  extern __shared__ int32_t smem[];
  int32_t* parent = smem;            // [width], -1 at the root and at nodes off the tree
  int32_t* token = smem + p.width;   // [width], -1 when out of vocab
  int32_t* depth = token + p.width;  // [width], in-vocab tokens on the node's path
  __shared__ uint32_t path_bits[kVerifyPenaltiesChunkWords];
  __shared__ uint32_t stop_bits[kVerifyPenaltiesChunkWords];

  const uint32_t tid = threadIdx.x;
  const int32_t width = p.width;
  const int32_t vocab_size = p.vocab_size;
  const int64_t request = blockIdx.y;
  const int64_t batch_index = p.batch_indices[request];
  const int64_t penalty_index = p.penalty_indices[request];
  const int32_t chunk_begin = blockIdx.x * kVerifyPenaltiesChunk;

  for (uint32_t i = tid; i < kVerifyPenaltiesChunkWords; i += kVerifyPenaltiesThreads) {
    path_bits[i] = 0;
    stop_bits[i] = 0;
  }
  const int64_t tokens_offset = batch_index * p.draft_tokens_stride;
  for (int32_t a = tid; a < width; a += kVerifyPenaltiesThreads) {
    const int64_t t = p.draft_tokens_are_int64 ? static_cast<const int64_t*>(p.draft_tokens)[tokens_offset + a]
                                               : static_cast<const int32_t*>(p.draft_tokens)[tokens_offset + a];
    token[a] = (0 <= t && t < vocab_size) ? static_cast<int32_t>(t) : -1;
    parent[a] = p.next_child == nullptr ? a - 1 : -1;
  }
  __syncthreads();

  if (p.next_child != nullptr) {
    // Node a is the parent of its first child and of that child's siblings. The
    // step bound keeps a malformed sibling cycle from hanging the kernel.
    const int64_t tree_offset = batch_index * p.tree_stride;
    for (int32_t a = tid; a < width; a += kVerifyPenaltiesThreads) {
      int64_t child = p.next_child[tree_offset + a];
      for (int32_t step = 0; 0 <= child && child < width && step < width; ++step) {
        parent[child] = a;
        child = p.next_sibling[tree_offset + child];
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
    if (parent[j] >= 0 && token[j] >= 0) verify_penalties_mark(path_bits, token[j], chunk_begin);
  }
  if (p.min_new_tokens != nullptr) {
    // Host-padded ids at vocab_size fall outside every chunk.
    for (int64_t i = tid; i < p.num_stop_tokens; i += kVerifyPenaltiesThreads) {
      const int64_t stop = p.stop_token_ids[request * p.num_stop_tokens + i];
      if (stop < vocab_size) verify_penalties_mark(stop_bits, stop, chunk_begin);
    }
  }
  __syncthreads();

  const bool has_additive = p.frequency != nullptr || p.presence != nullptr || p.min_new_tokens != nullptr;
  const float frequency = p.frequency != nullptr ? p.frequency[request] : 0.0f;
  const float presence = p.presence != nullptr ? p.presence[request] : 0.0f;
  const float repetition = p.repetition != nullptr ? p.repetition[request] : 1.0f;
  const int32_t min_new_tokens = p.min_new_tokens != nullptr ? p.min_new_tokens[request] : 0;
  const int32_t history_len = p.lens[penalty_index];
  const int32_t* counts = p.counts + penalty_index * vocab_size;
  T* rows = static_cast<T*>(p.logits) + batch_index * width * p.logits_stride;

#pragma unroll
  for (uint32_t i = 0; i < kVerifyPenaltiesItems; ++i) {
    const uint32_t local = i * kVerifyPenaltiesThreads + tid;
    const int32_t v = chunk_begin + local;
    if (v >= vocab_size) break;
    const int32_t history_count = counts[v];
    const bool on_path = verify_penalties_marked(path_bits, local);
    const bool is_stop = verify_penalties_marked(stop_bits, local);
    for (int32_t j = 0; j < width; ++j) {
      int32_t count = history_count;
      if (on_path) {
        for (int32_t a = j, step = 0; parent[a] >= 0 && step < width; a = parent[a], ++step) {
          count += token[a] == v;
        }
      }
      // The same ops as StepPenalties' tables, rounded where torch rounds: the
      // _rn intrinsics keep the compiler from contracting them into an FMA.
      T* x_ptr = rows + j * p.logits_stride + v;
      float x = device::cast<float>(*x_ptr);
      if (has_additive) {
        float additive = 0.0f;
        if (p.frequency != nullptr) additive = __fadd_rn(additive, -__fmul_rn(static_cast<float>(count), frequency));
        if (p.presence != nullptr) additive = __fadd_rn(additive, -(count > 0 ? presence : 0.0f));
        if (p.min_new_tokens != nullptr) {
          const bool blocked = is_stop && history_len + depth[j] < min_new_tokens;
          additive = __fadd_rn(additive, blocked ? -INFINITY : 0.0f);
        }
        x = device::cast<float>(device::cast<T>(__fadd_rn(x, additive)));
      }
      if (p.repetition != nullptr) {
        const float scaling = count > 0 ? repetition : 1.0f;
        x = x < 0.0f ? __fmul_rn(x, scaling) : __fdiv_rn(x, scaling);
      }
      *x_ptr = device::cast<T>(x);
    }
  }
}

/**
 * \brief Apply each verify row's own sampling penalties to `logits` in place.
 *
 * Rows of requests outside `batch_indices` are left as they are.
 *
 * \tparam T             Logits dtype: fp32_t | fp16_t | bf16_t
 * \param logits         `[bs * width, vocab]`, `width` consecutive verify rows per request
 * \param draft_tokens   `[bs, width]` int32 or int64 verify input tokens, column 0 the root
 * \param next_child     `[bs, width]` int64 first child per node, -1 for none; absent for a chain
 * \param next_sibling   `[bs, width]` int64 next sibling per node, -1 for none; set with next_child
 * \param batch_indices  `[n]` int64 batch position of each penalized request
 * \param penalty_indices `[n]` int64 history row of each penalized request
 * \param counts         `[num_rows, vocab]` int32 per-token history counts
 * \param lens           `[num_rows]` int32 history lengths
 * \param frequency      `[n, 1]` fp32, or absent
 * \param presence       `[n, 1]` fp32, or absent
 * \param repetition     `[n, 1]` fp32, or absent
 * \param min_new_tokens `[n, 1]` int32, or absent
 * \param stop_token_ids `[n, k]` int64 padded with `vocab`; given with min_new_tokens
 */
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

  auto per_request_float = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& t) -> const float* {
    if (!t.has_value()) return nullptr;
    TensorMatcher({n, 1}).with_dtype<float>().with_device(device).verify(t.value());
    return static_cast<const float*>(t.value().data_ptr());
  };
  const float* frequency_ptr = per_request_float(frequency);
  const float* presence_ptr = per_request_float(presence);
  const float* repetition_ptr = per_request_float(repetition);

  CHECK_HOST(min_new_tokens.has_value() == stop_token_ids.has_value())
      << "apply_verify_penalties: min_new_tokens and stop_token_ids go together";
  const int32_t* min_new_tokens_ptr = nullptr;
  const int64_t* stop_token_ids_ptr = nullptr;
  int64_t num_stop_tokens = 0;
  if (min_new_tokens.has_value()) {
    SymbolicSize k{"num_stop_tokens"};
    TensorMatcher({n, 1}).with_dtype<int32_t>().with_device(device).verify(min_new_tokens.value());
    TensorMatcher({n, k}).with_dtype<int64_t>().with_device(device).verify(stop_token_ids.value());
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
  // Keeps the dynamic shared memory under the 48 KB default.
  CHECK_HOST(width_value <= 2048) << "apply_verify_penalties: width " << width_value << " exceeds 2048";

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
  };
  const dim3 grid(static_cast<uint32_t>(div_ceil(vocab_size, int64_t{kVerifyPenaltiesChunk})), num_requests);
  const size_t smem_bytes = 3 * width_value * sizeof(int32_t);
  LaunchKernel(grid, kVerifyPenaltiesThreads, device.unwrap(), smem_bytes)(verify_penalties_kernel<T>, params);
}

}  // namespace sglang
