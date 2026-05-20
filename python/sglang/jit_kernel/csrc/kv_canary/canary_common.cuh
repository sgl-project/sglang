// Shared device helpers for the KV cache canary verify + write kernels.
//
// Real-KV source ABI (tvm-ffi cannot pass tuple[RealKvSource, ...] directly): the host wrapper unpacks the
// tuple into a fixed-size array of 4 sources and passes 4 separate uint8 tensors plus a single int32 array
// of (page_size, num_bytes_per_token, read_bytes) triplets of length 12 (4 sources x 3 fields). Unused
// padding sources have read_bytes = 0 so fold_real_kv_sources skips them.

#pragma once

#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE

#include "consts.cuh"
#include <cstdint>

namespace canary {

// Device-side handle for one real-KV source.
struct RealKvSourceHandle {
  const uint8_t* tensor;     // raw uint8 byte pointer to the source tensor
  int32_t row_stride_bytes;  // tensor.shape[1] in bytes (may exceed page_size * num_bytes_per_token)
  int32_t page_size;
  int32_t num_bytes_per_token;
  int32_t read_bytes;
};

// Standard splitmix64 finalizer. Bit-equivalent to the Python splitmix64 in
// kv_canary/verify_ref.py.
SGL_DEVICE uint64_t splitmix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// 4-arg chain step: nested splitmix64 of the four uint64 inputs. Each input is folded into the running
// accumulator via `acc = splitmix64(acc ^ next)`, so order is significant and a pair of equal inputs no
// longer self-cancels. Matches the Python helper splitmix64_mix4 in kv_canary/verify_ref.py.
SGL_DEVICE uint64_t splitmix64_mix4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  uint64_t h = splitmix64(a);
  h = splitmix64(h ^ b);
  h = splitmix64(h ^ c);
  h = splitmix64(h ^ d);
  return h;
}

// Read one byte from a source following the RealKvSource access invariant. The invariant (from
// kv_canary/verify.py docstring) is:
//
//     tensor[slot_idx // page_size,
//            (slot_idx % page_size) * num_bytes_per_token + byte_offset]
//
// row_stride_bytes is the dim-1 size of the underlying tensor in bytes (which may exceed
// page_size * num_bytes_per_token; trailing bytes are skipped).
SGL_DEVICE uint8_t real_kv_load_byte(const RealKvSourceHandle& src, int64_t slot_idx, int64_t byte_offset) {
  const int64_t row = slot_idx / src.page_size;
  const int64_t col_within_page = slot_idx % src.page_size;
  const int64_t col = col_within_page * src.num_bytes_per_token + byte_offset;
  const int64_t flat_index = row * static_cast<int64_t>(src.row_stride_bytes) + col;
  return src.tensor[flat_index];
}

// Fold one source's read_bytes into a uint64 hash, mode-dispatching.
//
// PARTIAL mode: pack the first min(read_bytes, 16) bytes little-endian into 8-byte words and
// splitmix64-fold them (at most 2 words). When read_bytes <= 16, PARTIAL produces the same hash as ALL.
// ALL mode: pack bytes little-endian into 8-byte words (zero-padded if read_bytes is not a multiple of 8)
// and splitmix64-fold them iteratively.
SGL_DEVICE uint64_t real_kv_fold_one_source(const RealKvSourceHandle& src, int64_t slot_idx, RealKvHashMode mode) {
  if (src.read_bytes <= 0) {
    return 0ULL;
  }
  const int64_t effective_read_bytes =
      (mode == RealKvHashMode::kPartial) ? (src.read_bytes < 16 ? src.read_bytes : 16) : src.read_bytes;
  // ALL mode (and PARTIAL, which shares the same word-pack + splitmix64 loop with a capped length).
  uint64_t acc = 0ULL;
  int64_t i = 0;
  while (i < effective_read_bytes) {
    uint64_t word = 0ULL;
    const int64_t take = (effective_read_bytes - i) >= 8 ? 8 : (effective_read_bytes - i);
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      if (b < take) {
        word |= static_cast<uint64_t>(real_kv_load_byte(src, slot_idx, i + b)) << (b * 8);
      }
    }
    acc = splitmix64(acc ^ word);
    i += 8;
  }
  return acc;
}

// Fold all configured real-KV sources for a given slot. Iterates sequentially and combines each source's
// contribution via acc = splitmix64(acc XOR source_hash); matches _compute_real_kv_hash_vec in
// kv_canary/verify_ref.py.
//
// In OFF mode the function returns 0 unconditionally (the running real_kv_hash field is always 0). Sources
// with read_bytes == 0 contribute nothing (their source_hash is 0 and acc = splitmix64(acc ^ 0)).
//
// IMPORTANT: the Python ref ONLY enters the fold loop when read_bytes > 0 (see _compute_real_kv_hash_vec
// in kv_canary/verify_ref.py: `if source.read_bytes <= 0: continue`). To stay byte-equal, this
// helper must skip the splitmix64 step entirely for read_bytes == 0 sources rather than treating them as
// "fold 0".
SGL_DEVICE uint64_t
fold_real_kv_sources(const RealKvSourceHandle* sources, int num_sources, int64_t slot_idx, RealKvHashMode mode) {
  if (mode == RealKvHashMode::kOff || num_sources <= 0) {
    return 0ULL;
  }
  uint64_t acc = 0ULL;
  for (int s = 0; s < num_sources; ++s) {
    if (sources[s].read_bytes <= 0) {
      continue;
    }
    const uint64_t source_hash = real_kv_fold_one_source(sources[s], slot_idx, mode);
    acc = splitmix64(acc ^ source_hash);
  }
  return acc;
}

// Append a violation row to the ring (fill-once) and bump the monotonic counter unconditionally.
//
// Ordering of columns must match kViolationField* in consts.cuh exactly. The "ExpectedAux"
// column (index 6) is reason-agnostic: verify launches pass expected_chain_hash, write launches pass
// expected_position. The "StoredChainHash" column (index 5) carries running_prev_hash on the write path.
//
// atomicAdd on violation_write_index serializes arrivals; only writers with idx < ring_capacity store a
// row. The __threadfence_system after the store guarantees any host observer that reads the post-increment
// counter also sees the committed row.
SGL_DEVICE void record_violation(
    int64_t* __restrict__ violation_ring,
    int32_t* __restrict__ violation_write_index,
    int32_t ring_capacity,
    int32_t kernel_kind,
    int64_t slot_idx,
    int64_t position,
    int64_t stored_token,
    int64_t expected_token,
    int64_t stored_chain_hash,
    int64_t expected_aux,
    int64_t fail_reason_bits) {
  const int32_t seq = atomicAdd(violation_write_index, 1);
  if (seq < ring_capacity) {
    int64_t* row = violation_ring + static_cast<int64_t>(seq) * kViolationFields;
    row[kViolationFieldKernelKind] = static_cast<int64_t>(kernel_kind);
    row[kViolationFieldSlotIdx] = slot_idx;
    row[kViolationFieldPosition] = position;
    row[kViolationFieldStoredToken] = stored_token;
    row[kViolationFieldExpectedToken] = expected_token;
    row[kViolationFieldStoredChainHash] = stored_chain_hash;
    row[kViolationFieldExpectedAux] = expected_aux;
    row[kViolationFieldFailReasonBits] = fail_reason_bits;
    __threadfence_system();
  }
}

SGL_DEVICE int64_t canary_load_field(const uint8_t* buf, int64_t slot_idx, int64_t slot_stride_bytes, int field) {
  const int64_t* p = reinterpret_cast<const int64_t*>(buf + slot_idx * slot_stride_bytes);
  return p[field];
}

SGL_DEVICE void
canary_store_field(uint8_t* buf, int64_t slot_idx, int64_t slot_stride_bytes, int field, int64_t value) {
  int64_t* p = reinterpret_cast<int64_t*>(buf + slot_idx * slot_stride_bytes);
  p[field] = value;
}

}  // namespace canary
