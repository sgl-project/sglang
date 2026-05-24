// Shared device helpers for the KV cache canary verify + write kernels.
//
// Real-KV source ABI (tvm-ffi cannot pass tuple[RealKvSource, ...] directly): the host wrapper unpacks the
// tuple into a fixed-size array of 4 sources and passes 4 separate uint8 tensors plus a single int32 array
// of (page_size, num_bytes_per_token, read_bytes) triplets of length 12 (4 sources x 3 fields). The host
// also passes ``num_sources`` (count of valid leading entries); the kernel iterates only
// ``sources[0..num_sources)`` and never reads the padding tail.

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

// 3-arg chain step: nested splitmix64 of the three uint64 inputs. Each input is folded into the running
// accumulator via `acc = splitmix64(acc ^ next)`, so order is significant and a pair of equal inputs no
// longer self-cancels. Matches the Python helper splitmix64_mix3 in kv_canary/consts.py.
SGL_DEVICE uint64_t splitmix64_mix3(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t h = splitmix64(a);
  h = splitmix64(h ^ b);
  h = splitmix64(h ^ c);
  return h;
}

// Read 16 aligned bytes from a source as two uint64 little-endian words, following the RealKvSource access
// invariant. The invariant (from kv_canary/verify.py docstring) is:
//
//     tensor[slot_idx // page_size,
//            (slot_idx % page_size) * num_bytes_per_token + byte_offset]
//
// row_stride_bytes is the dim-1 size of the underlying tensor in bytes (which may exceed
// page_size * num_bytes_per_token; trailing bytes are skipped).
//
// 16B-alignment precondition (enforced host-side in kv_canary/verify.py::RealKvSource.__post_init__):
// num_bytes_per_token, row_stride_bytes, and read_bytes are all multiples of 16, and byte_offset is always a
// multiple of 16. Combined with PyTorch's CUDA-allocator alignment (>= 256B) of src.tensor, every flat_index
// computed here is 16B-aligned, so the uint4 load below is a single coalesced LDG.E.128.
SGL_DEVICE void real_kv_load_uint4(
    const RealKvSourceHandle& src, int64_t slot_idx, int64_t byte_offset, uint64_t& word_lo, uint64_t& word_hi) {
  const int64_t row = slot_idx / src.page_size;
  const int64_t col_within_page = slot_idx % src.page_size;
  const int64_t col = col_within_page * src.num_bytes_per_token + byte_offset;
  const int64_t flat_index = row * static_cast<int64_t>(src.row_stride_bytes) + col;
  const uint4 vec = *reinterpret_cast<const uint4*>(src.tensor + flat_index);
  word_lo = static_cast<uint64_t>(vec.x) | (static_cast<uint64_t>(vec.y) << 32);
  word_hi = static_cast<uint64_t>(vec.z) | (static_cast<uint64_t>(vec.w) << 32);
}

// Fold one source's read_bytes into a uint64 hash, mode-dispatching.
//
// PARTIAL mode: pack exactly the first 16 bytes (one 16B chunk = two 8B little-endian words) and
// splitmix64-fold them. ALL mode: pack every 16 bytes little-endian and splitmix64-fold iteratively.
//
// Loads run in 16B (uint4) chunks; the host-side __post_init__ guarantees read_bytes % 16 == 0 so no
// tail-padding logic is needed inside the kernel. With the 16B-aligned contract, read_bytes is always
// >= 16 when non-zero, so PARTIAL collapses to a constant 16B prefix. Output is byte-equal to the
// Python ref helper _splitmix64_fold_bytes_scalar, which loops over 8B little-endian words.
SGL_DEVICE uint64_t real_kv_fold_one_source(const RealKvSourceHandle& src, int64_t slot_idx, RealKvHashMode mode) {
  const int64_t effective_read_bytes = (mode == RealKvHashMode::kPartial) ? static_cast<int64_t>(16) : src.read_bytes;
  uint64_t acc = 0ULL;
  for (int64_t byte_offset = 0; byte_offset < effective_read_bytes; byte_offset += 16) {
    uint64_t word_lo;
    uint64_t word_hi;
    real_kv_load_uint4(src, slot_idx, byte_offset, word_lo, word_hi);
    acc = splitmix64(acc ^ word_lo);
    acc = splitmix64(acc ^ word_hi);
  }
  return acc;
}

// Fold all configured real-KV sources for a given slot. Iterates sequentially and combines each source's
// contribution via acc = splitmix64(acc XOR source_hash); matches _compute_real_kv_hash_scalar in
// kv_canary/verify_ref.py. In OFF mode the function returns 0 unconditionally (the running real_kv_hash
// field is always 0).
SGL_DEVICE uint64_t
real_kv_fold_sources(const RealKvSourceHandle* sources, int num_sources, int64_t slot_idx, RealKvHashMode mode) {
  if (mode == RealKvHashMode::kOff || num_sources <= 0) {
    return 0ULL;
  }
  uint64_t acc = 0ULL;
  for (int s = 0; s < num_sources; ++s) {
    const uint64_t source_hash = real_kv_fold_one_source(sources[s], slot_idx, mode);
    acc = splitmix64(acc ^ source_hash);
  }
  return acc;
}

// Kernel-wide sink for violation rows. ``ring`` / ``write_index`` are device pointers, ``ring_capacity``
// caps how many rows physically land in the ring (overflow rows bump ``write_index`` but are not stored),
// and ``kernel_kind`` is stamped into every row so a host observer can attribute it to its source launch.
struct ViolationSink {
  int64_t* __restrict__ ring;
  int32_t* __restrict__ write_index;
  int32_t ring_capacity;
  int32_t kernel_kind;
};

// One violation row's payload. Field order is meaning-only (not the on-ring column order — that is set by
// kViolationField* in consts.cuh). ``expected_aux`` is reason-agnostic: verify launches pass
// expected_chain_hash, write launches pass expected_position. ``stored_chain_hash`` carries
// running_prev_hash on the write path.
struct ViolationRow {
  int64_t slot_idx;
  int64_t position;
  int64_t stored_token;
  int64_t expected_token;
  int64_t stored_chain_hash;
  int64_t expected_aux;
  int64_t fail_reason_bits;
};

// Append a violation row to the sink's ring (fill-once) and bump the monotonic counter unconditionally.
//
// atomicAdd on ``sink.write_index`` serializes arrivals; only writers with idx < ring_capacity store a row.
// The __threadfence_system after the store guarantees any host observer that reads the post-increment
// counter also sees the committed row.
SGL_DEVICE void record_violation(const ViolationSink& sink, const ViolationRow& row) {
  const int32_t seq = atomicAdd(sink.write_index, 1);
  if (seq < sink.ring_capacity) {
    int64_t* dst = sink.ring + static_cast<int64_t>(seq) * kViolationFields;
    dst[kViolationFieldKernelKind] = static_cast<int64_t>(sink.kernel_kind);
    dst[kViolationFieldSlotIdx] = row.slot_idx;
    dst[kViolationFieldPosition] = row.position;
    dst[kViolationFieldStoredToken] = row.stored_token;
    dst[kViolationFieldExpectedToken] = row.expected_token;
    dst[kViolationFieldStoredChainHash] = row.stored_chain_hash;
    dst[kViolationFieldExpectedAux] = row.expected_aux;
    dst[kViolationFieldFailReasonBits] = row.fail_reason_bits;
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

// Compute the chain-step hash of ``source_slot_idx``: load (token, position, prev_hash) from that slot
// and fold them through splitmix64_mix3. The result is the chain hash that the slot immediately following
// ``source_slot_idx`` should store as its prev_hash.
//
// real_kv_hash is intentionally NOT folded in: under radix prefix sharing, ``RadixCache.cache_unfinished_req``
// remaps ``req_to_token[req, P]`` from a req's own freshly-written slot to a shared slot owned by another
// req. The shared slot's (token, position, prev_hash) match by definition (same prompt → same content
// chain), but its real_kv_hash reflects the donor req's KV values and would not equal the consumer req's
// stored chain. Excluding real_kv_hash keeps the chain pure-content and immune to legitimate radix folding.
// real_kv_hash field corruption is still caught by the standalone real_kv_hash field check on each slot
// (per-forward verify + periodic sweep).
//
// ``source_slot_idx < 0`` signals "no predecessor"; the chain anchors on splitmix64(kCanaryChainAnchor).
// Shared by the verify path (recompute expected prev_hash from a slot's predecessor) and the write path
// (seed running_prev_hash from a chain-prefix slot).
SGL_DEVICE uint64_t compute_slot_hash(const uint8_t* canary_buf, int64_t slot_stride_bytes, int64_t source_slot_idx) {
  if (source_slot_idx < 0) {
    return splitmix64(kCanaryChainAnchor);
  }
  const int64_t token = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldToken);
  const int64_t position = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldPosition);
  const int64_t prev_hash = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldPrevHash);
  return splitmix64_mix3(
      static_cast<uint64_t>(prev_hash),
      static_cast<uint64_t>(token),
      static_cast<uint64_t>(position));
}

}  // namespace canary
