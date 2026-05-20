// Shared device helpers for the KV cache canary verify + write kernels.
//
// Mirrors module-level constants and helpers from
// python/sglang/jit_kernel/kv_canary/verify.py and
// python/sglang/jit_kernel/kv_canary/write.py. Value parity is enforced by test_unit_const_sync.py.
//
// Real-KV source ABI (tvm-ffi cannot pass tuple[RealKvSource, ...] directly): the host wrapper unpacks the
// tuple into a fixed-size array of 4 sources and passes 4 separate uint8 tensors plus a single int32 array
// of (page_size, num_bytes_per_token, read_bytes) triplets of length 12 (4 sources x 3 fields). Unused
// padding sources have read_bytes = 0 so fold_real_kv_sources skips them.

#pragma once

#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE

#include <cstdint>

namespace canary {

// Frozen chain anchor used wherever a slot has no predecessor. Mirror of the Python CANARY_CHAIN_ANCHOR
// in kv_canary/verify.py.
constexpr uint64_t kCanaryChainAnchor = 0xC0FFEE1234567890ULL;

// Canary slot field offsets within the 4-int64 layout.
constexpr int kCanaryFieldsPerSlot = 4;
constexpr int kCanaryFieldToken = 0;
constexpr int kCanaryFieldPosition = 1;
constexpr int kCanaryFieldPrevHash = 2;
constexpr int kCanaryFieldRealKvHash = 3;

// Violation-row column layout. Mirrors _VIOLATION_FIELD_* in kv_canary/verify.py. Column 6
// (ExpectedAux) is reason-agnostic: verify launches store expected_chain_hash there, write launches store
// expected_position.
constexpr int kViolationFields = 8;
constexpr int kViolationFieldKernelKind = 0;
constexpr int kViolationFieldSlotIdx = 1;
constexpr int kViolationFieldPosition = 2;
constexpr int kViolationFieldStoredToken = 3;
constexpr int kViolationFieldExpectedToken = 4;
constexpr int kViolationFieldStoredChainHash = 5;
constexpr int kViolationFieldExpectedAux = 6;
constexpr int kViolationFieldFailReasonBits = 7;

// Fail-reason bit positions. Bitfield (not enum) because a single verify entry may set multiple reasons.
// Verify-launch bits mirror _FAIL_REASON_BIT_* in kv_canary/verify.py; write-launch bits mirror
// _FAIL_REASON_BIT_WRITE_* in kv_canary/write.py.
constexpr int64_t kFailReasonChainHash = 1LL << 0;
constexpr int64_t kFailReasonPosition = 1LL << 1;
constexpr int64_t kFailReasonRealKvHash = 1LL << 2;
constexpr int64_t kFailReasonWriteTokenMismatch = 1LL << 3;
constexpr int64_t kFailReasonWritePositionMismatch = 1LL << 4;

// Mirror of the Python RealKvHashMode IntEnum in kv_canary/verify.py. Values must match exactly.
enum class RealKvHashMode : int32_t {
  kOff = 0,
  kPartial = 1,
  kAll = 2,
};

// Mirror of the Python CanaryPseudoMode IntEnum in kv_canary/write.py.
enum class CanaryPseudoMode : int32_t {
  kOff = 0,
  kOn = 1,
};

// Maximum number of real-KV sources the C++ ABI supports per launch. Host wrapper pads any shorter tuple
// up to this length with dummy entries (read_bytes = 0) and rejects longer tuples.
constexpr int kMaxRealKvSources = 4;

// Per-source ABI fields packed into a length-(kMaxRealKvSources * 3) int32 tensor.
constexpr int kRealKvSourceFieldsPerEntry = 3;
constexpr int kRealKvSourceFieldPageSize = 0;
constexpr int kRealKvSourceFieldNumBytesPerToken = 1;
constexpr int kRealKvSourceFieldReadBytes = 2;

// Device-side handle for one real-KV source.
struct RealKvSourceHandle {
  const uint8_t* tensor;     // raw uint8 byte pointer to the source tensor
  int32_t row_stride_bytes;  // tensor.shape[1] in bytes (may exceed page_size * num_bytes_per_token)
  int32_t page_size;
  int32_t num_bytes_per_token;
  int32_t read_bytes;
};

// Standard splitmix64 finalizer. Bit-equivalent to the Python _splitmix64_python in
// kv_canary/verify_ref.py.
SGL_DEVICE uint64_t splitmix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// 4-arg chain step: XOR all four uint64 inputs, then splitmix64-finalize. Matches the Python helper
// _splitmix64_mix4_vec in kv_canary/verify_ref.py.
SGL_DEVICE uint64_t splitmix64_mix4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return splitmix64(a ^ b ^ c ^ d);
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
// Ordering of columns must match _VIOLATION_FIELD_* in kv_canary/verify.py exactly. The "ExpectedAux"
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
