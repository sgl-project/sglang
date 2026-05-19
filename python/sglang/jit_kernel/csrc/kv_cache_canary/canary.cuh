#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kCanaryFieldsPerSlot = 5;
constexpr int kCanaryFieldReqId = 0;
constexpr int kCanaryFieldTokenId = 1;
constexpr int kCanaryFieldPosition = 2;
constexpr int kCanaryFieldPrevHash = 3;
constexpr int kCanaryFieldRealKvHash = 4;

constexpr int kViolationFields = 8;
constexpr int kViolationFieldKernelKind = 0;
constexpr int kViolationFieldFailReason = 1;
constexpr int kViolationFieldSlotIdx = 2;
constexpr int kViolationFieldReqId = 3;
constexpr int kViolationFieldTokenId = 4;
constexpr int kViolationFieldPosition = 5;
constexpr int kViolationFieldExpectedHash = 6;
constexpr int kViolationFieldActualHash = 7;

constexpr int kFailReasonReqId = 1;
constexpr int kFailReasonTokenId = 2;
constexpr int kFailReasonPosition = 3;
constexpr int kFailReasonHash = 4;
constexpr int kFailReasonPositionMonotonic = 5;
constexpr int kFailReasonRealKvHash = 6;

// Mirror of the Python REAL_KV_HASH_MODE_* constants in
// jit_kernel/kv_cache_canary.py. ``OFF`` disables the real-KV
// fingerprint entirely; ``BIT`` mixes the first 16 bytes of the
// real-KV slot, ``ALL`` mixes the full real-KV slot stride.
constexpr int kRealKvHashModeOff = 0;
constexpr int kRealKvHashModeBit = 1;
constexpr int kRealKvHashModeAll = 2;

struct CanaryParams {
  // Source buffer: read prior canary slots from here (verification target,
  // and the prev-slot read source for chain hash recomputation).
  const uint8_t* __restrict__ src_buf;
  // Destination buffer: write the new canary slots here.
  uint8_t* __restrict__ dst_buf;
  // Bytes per slot in both src and dst (= per-slot stride of the underlying real-layer-shaped tensor).
  int64_t slot_stride_bytes;

  // Per-verify-entry arrays, length == num_verify. NOTE: we do NOT carry
  // verify_token_ids — the historical tokens were written in prior forwards
  // and are not in the current ForwardBatch.input_ids; recovering them
  // would require maintaining host-side per-req history, which is exactly
  // what the stateless redesign drops. Token tampering is caught indirectly
  // via the splitmix64 chain (a corrupt token_id propagates into prev_hash
  // for the next position) and req_id / position cross-checks below.
  const int64_t* __restrict__ verify_slot_indices;
  const int64_t* __restrict__ verify_positions;
  const int64_t* __restrict__ verify_req_ids;
  // For verify entry i, the slot index of position (verify_positions[i] - 1)
  // for the same req. -1 means "this is position 0; expected prev_hash =
  // kSeed". The kernel reads (prev_token, prev_position, prev_prev_hash) from
  // that slot and recomputes splitmix64_mix on-device.
  const int64_t* __restrict__ verify_prev_slot_indices;
  // 1 = active verify entry, 0 = skip-sentinel padding (cuda graph fixed
  // buffer capacity exceeds plan size).
  const int32_t* __restrict__ verify_active_mask;
  uint32_t num_verify;

  // Per-write-entry arrays, length == num_write. Pure data; read by the
  // per-write-req driver thread that walks the chain.
  const int64_t* __restrict__ write_slot_indices;
  const int64_t* __restrict__ write_token_ids;
  const int64_t* __restrict__ write_positions;
  const int64_t* __restrict__ write_req_ids;

  // Per-write-req arrays, length == num_write_reqs. One driver thread per row
  // walks the splitmix64 chain across this req's writes in sequence.
  const int64_t* __restrict__ write_req_seed_slot_indices;  // -1 = kSeed
  const int64_t* __restrict__ write_req_entry_starts;
  const int64_t* __restrict__ write_req_entry_counts;
  const int32_t* __restrict__ write_req_active_mask;
  uint32_t num_write_reqs;

  // splitmix64 chain seed (== CanaryConfig.seed).
  uint64_t seed;

  // Violation ring + first-violation slot + is_errored flag.
  int64_t* __restrict__ violation_ring;          // [ring_capacity, kViolationFields]
  int32_t* __restrict__ violation_ring_valid;    // [ring_capacity] per-row valid latch
  uint32_t* __restrict__ violation_write_index;  // [1], atomicAdd target
  int64_t* __restrict__ first_violation;         // [kViolationFields]
  uint32_t* __restrict__ first_violation_set;    // [1], 0/1
  int32_t* __restrict__ is_errored;              // [1] (32-bit so atomicOr is in-bounds)
  uint64_t* __restrict__ slot_run_counter;       // [1], +num_slots
  uint64_t* __restrict__ kernel_run_counter;     // [1], +1
  uint32_t ring_capacity;
  int32_t kernel_kind;  // 0 = head, 1 = tail

  // Optional real-KV "canary-with-real-data" inputs. When ``real_kv_hash_mode``
  // is ``kRealKvHashModeOff`` (or ``real_kv_buf`` is null / ``real_kv_read_bytes``
  // is 0), the write path stores ``real_kv_hash = 0`` and the verify path
  // skips the new field. Otherwise the kernel reads ``read_bytes`` bytes
  // starting at ``slot_idx * slot_stride_bytes`` (the real-KV pool's
  // per-slot stride), folds them through ``splitmix64_mix`` in 8-byte
  // chunks, and either stores the fingerprint (write path) or compares
  // against the stored value (verify path).
  const uint8_t* __restrict__ real_kv_buf;
  int64_t real_kv_slot_stride_bytes;
  int64_t real_kv_read_bytes;
  int32_t real_kv_hash_mode;
};

SGL_DEVICE int64_t load_field(const uint8_t* buf, int64_t slot_idx, int64_t stride, int field) {
  const int64_t* p = reinterpret_cast<const int64_t*>(buf + slot_idx * stride);
  return p[field];
}

SGL_DEVICE void store_field(uint8_t* buf, int64_t slot_idx, int64_t stride, int field, int64_t value) {
  int64_t* p = reinterpret_cast<int64_t*>(buf + slot_idx * stride);
  p[field] = value;
}

// Standard splitmix64 finalizer. Bit-wise equivalent to the Python
// mirror in ``jit_kernel/kv_cache_canary_ref.py``;
// ``test_splitmix64_consistency.py`` cross-validates the two
// implementations.
SGL_DEVICE uint64_t splitmix64_finalize(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

SGL_DEVICE uint64_t splitmix64_mix(uint64_t prev_hash, uint64_t token_id, uint64_t position) {
  return splitmix64_finalize(prev_hash ^ token_id ^ position);
}

// Fold a real-KV slot's bytes through splitmix64 in 8-byte chunks. The last
// chunk is zero-padded if ``read_bytes`` is not a multiple of 8. Returns 0
// when the feature is disabled (mode == OFF) or when no buffer/byte budget is
// configured. Bit-exact reference lives in ``_RealKvView.hash_slot`` in
// ``jit_kernel/kv_cache_canary.py``.
SGL_DEVICE uint64_t compute_real_kv_hash(const CanaryParams& p, int64_t slot_idx) {
  if (p.real_kv_hash_mode == kRealKvHashModeOff || p.real_kv_buf == nullptr || p.real_kv_read_bytes <= 0 ||
      p.real_kv_slot_stride_bytes <= 0) {
    return 0ULL;
  }
  const uint8_t* slot_base = p.real_kv_buf + slot_idx * p.real_kv_slot_stride_bytes;
  uint64_t acc = 0ULL;
  int64_t i = 0;
  while (i < p.real_kv_read_bytes) {
    int64_t remaining = p.real_kv_read_bytes - i;
    int64_t take = remaining >= 8 ? 8 : remaining;
    uint64_t word = 0ULL;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      if (b < take) {
        word |= static_cast<uint64_t>(slot_base[i + b]) << (b * 8);
      }
    }
    acc = splitmix64_mix(acc, word, 0ULL);
    i += take;
  }
  return acc;
}

SGL_DEVICE void record_violation(
    const CanaryParams& p,
    int32_t fail_reason,
    int64_t slot_idx,
    int64_t req_id,
    int64_t token_id,
    int64_t position,
    uint64_t expected_hash,
    uint64_t actual_hash) {
  int64_t entry[kViolationFields];
  entry[kViolationFieldKernelKind] = static_cast<int64_t>(p.kernel_kind);
  entry[kViolationFieldFailReason] = static_cast<int64_t>(fail_reason);
  entry[kViolationFieldSlotIdx] = slot_idx;
  entry[kViolationFieldReqId] = req_id;
  entry[kViolationFieldTokenId] = token_id;
  entry[kViolationFieldPosition] = position;
  entry[kViolationFieldExpectedHash] = static_cast<int64_t>(expected_hash);
  entry[kViolationFieldActualHash] = static_cast<int64_t>(actual_hash);

  // First-violation latch: only the first writer wins; ensures the very first
  // mismatch is preserved verbatim even if subsequent cascades overrun the ring.
  // We CAS-claim the set flag BEFORE writing the 8 fields, then __threadfence_system
  // after the writes — but importantly we use a 2-stage latch (claimed → committed)
  // so a host that races to read after seeing ``is_errored == 1`` can detect a
  // partial write. Stage 1: CAS 0→1 (claimed). Stage 2: __threadfence_system +
  // CAS 1→2 (committed). The later atomicOr on is_errored is fenced by the same
  // __threadfence_system, so host that reads is_errored==1 sees ``first_violation_set
  // >= 2`` and a complete 8-field write.
  unsigned int prev = atomicCAS(p.first_violation_set, 0u, 1u);
  if (prev == 0u) {
#pragma unroll
    for (int i = 0; i < kViolationFields; ++i) {
      p.first_violation[i] = entry[i];
    }
    __threadfence_system();
    atomicExch(p.first_violation_set, 2u);
  }

  // Ring buffer append: atomicAdd reserves a unique sequence. Two threads with
  // colliding ``seq % capacity`` would otherwise tear each other's 8-field
  // stores, so we per-row CAS-latch the valid-flag (0=empty→1=writing) and
  // only flip to 2 (=readable) after all 8 fields are stored.
  unsigned int seq = atomicAdd(p.violation_write_index, 1u);
  unsigned int idx = seq % p.ring_capacity;
  int32_t* row_valid = p.violation_ring_valid + idx;
  unsigned int prev_valid = atomicCAS(reinterpret_cast<unsigned int*>(row_valid), 0u, 1u);
  if (prev_valid == 0u) {
    int64_t* row = p.violation_ring + static_cast<int64_t>(idx) * kViolationFields;
#pragma unroll
    for (int i = 0; i < kViolationFields; ++i) {
      row[i] = entry[i];
    }
    __threadfence_system();
    atomicExch(reinterpret_cast<unsigned int*>(row_valid), 2u);
  }

  // Order: violation buffer writes must be visible before is_errored is set,
  // otherwise the host can read is_errored=true but see torn / empty entries.
  __threadfence_system();
  atomicOr(reinterpret_cast<unsigned int*>(p.is_errored), 1u);
}

SGL_DEVICE void run_verify_entry(const CanaryParams& p, uint32_t tid) {
  if (p.verify_active_mask[tid] == 0) {
    return;
  }
  const int64_t slot_idx = p.verify_slot_indices[tid];
  const int64_t expected_req_id = p.verify_req_ids[tid];
  const int64_t expected_position = p.verify_positions[tid];
  const int64_t prev_slot_idx = p.verify_prev_slot_indices[tid];

  const int64_t actual_req_id = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldReqId);
  const int64_t actual_token_id = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId);
  const int64_t actual_position = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
  const uint64_t actual_prev_hash =
      static_cast<uint64_t>(load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash));

  uint64_t expected_prev_hash;
  if (prev_slot_idx < 0) {
    expected_prev_hash = p.seed;
  } else {
    const uint64_t prev_prev_hash =
        static_cast<uint64_t>(load_field(p.src_buf, prev_slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash));
    const int64_t prev_token = load_field(p.src_buf, prev_slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId);
    const int64_t prev_position = load_field(p.src_buf, prev_slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
    expected_prev_hash =
        splitmix64_mix(prev_prev_hash, static_cast<uint64_t>(prev_token), static_cast<uint64_t>(prev_position));
  }

  // Independent verify-path fail_reason categories:
  // (b1) req_id cross-check, (b3) position monotonic, (a) chain hash,
  // (c) real-KV fingerprint (when ``--kv-cache-canary-real-data`` is on).
  // Token-id check applies only to the write path (where the host has the
  // raw input_ids from forward_batch); on verify it's elided. Token-field
  // tampering propagates into the chain hash via splitmix64_mix and is
  // caught at the next position's verify.
  const uint64_t actual_real_kv_hash =
      static_cast<uint64_t>(load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldRealKvHash));
  const uint64_t expected_real_kv_hash = compute_real_kv_hash(p, slot_idx);

  int32_t fail_reason = 0;
  uint64_t reported_expected = expected_prev_hash;
  uint64_t reported_actual = actual_prev_hash;
  if (actual_req_id != expected_req_id) {
    fail_reason = kFailReasonReqId;
  } else if (actual_position != expected_position) {
    fail_reason = kFailReasonPositionMonotonic;
  } else if (actual_prev_hash != expected_prev_hash) {
    fail_reason = kFailReasonHash;
  } else if (actual_real_kv_hash != expected_real_kv_hash) {
    fail_reason = kFailReasonRealKvHash;
    reported_expected = expected_real_kv_hash;
    reported_actual = actual_real_kv_hash;
  }
  if (fail_reason != 0) {
    record_violation(
        p, fail_reason, slot_idx, actual_req_id, actual_token_id, actual_position, reported_expected, reported_actual);
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(p.slot_run_counter), 1ULL);
}

SGL_DEVICE void run_write_req_chain(const CanaryParams& p, uint32_t req_tid) {
  if (p.write_req_active_mask[req_tid] == 0) {
    return;
  }
  const int64_t entry_start = p.write_req_entry_starts[req_tid];
  const int64_t entry_count = p.write_req_entry_counts[req_tid];
  if (entry_count <= 0) {
    return;
  }
  const int64_t seed_slot_idx = p.write_req_seed_slot_indices[req_tid];

  // Seed of the chain at the first write entry's position. If seed_slot < 0,
  // K_req_old == 0 so we start from kSeed; otherwise read the canary slot at
  // (K_req_old - 1) and recompute splitmix64_mix from its stored
  // (prev_hash, token_id, position) to derive the prev_hash field for the
  // first new write.
  uint64_t prev_hash;
  if (seed_slot_idx < 0) {
    prev_hash = p.seed;
  } else {
    const uint64_t seed_prev_hash =
        static_cast<uint64_t>(load_field(p.src_buf, seed_slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash));
    const int64_t seed_token = load_field(p.src_buf, seed_slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId);
    const int64_t seed_position = load_field(p.src_buf, seed_slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
    prev_hash = splitmix64_mix(seed_prev_hash, static_cast<uint64_t>(seed_token), static_cast<uint64_t>(seed_position));
  }

  for (int64_t k = 0; k < entry_count; ++k) {
    const int64_t i = entry_start + k;
    const int64_t slot_idx = p.write_slot_indices[i];
    const int64_t req_id = p.write_req_ids[i];
    const int64_t token_id = p.write_token_ids[i];
    const int64_t position = p.write_positions[i];

    const uint64_t real_kv_hash = compute_real_kv_hash(p, slot_idx);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldReqId, req_id);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId, token_id);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPosition, position);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash, static_cast<int64_t>(prev_hash));
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldRealKvHash, static_cast<int64_t>(real_kv_hash));

    prev_hash = splitmix64_mix(prev_hash, static_cast<uint64_t>(token_id), static_cast<uint64_t>(position));
    atomicAdd(reinterpret_cast<unsigned long long*>(p.slot_run_counter), 1ULL);
  }
}

__global__ void canary_kernel(const CanaryParams __grid_constant__ p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_threads = p.num_verify + p.num_write_reqs;

  // Unconditional liveness counter: block 0 thread 0 ALWAYS increments the
  // kernel-run counter before the early-return guard, even when the plan
  // is fully inactive (skip-sentinel masks) or the launch has zero verify
  // and zero write-req entries. The host-side health monitor reads this
  // counter to detect "canary is actually executing" — tying the
  // increment to active work would silently zero it during server warmup
  // forwards (and any other no-work step) and trip a spurious "kernel
  // never ran" panic.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p.kernel_run_counter), 1ULL);
  }

  if (tid >= total_threads) {
    return;
  }

  if (tid < p.num_verify) {
    run_verify_entry(p, tid);
  } else {
    const uint32_t req_tid = tid - p.num_verify;
    run_write_req_chain(p, req_tid);
  }
}

void canary_step(
    tvm::ffi::TensorView src_buf,
    tvm::ffi::TensorView dst_buf,
    int64_t slot_stride_bytes,
    tvm::ffi::TensorView verify_slot_indices,
    tvm::ffi::TensorView verify_positions,
    tvm::ffi::TensorView verify_req_ids,
    tvm::ffi::TensorView verify_prev_slot_indices,
    tvm::ffi::TensorView verify_active_mask,
    tvm::ffi::TensorView write_slot_indices,
    tvm::ffi::TensorView write_token_ids,
    tvm::ffi::TensorView write_positions,
    tvm::ffi::TensorView write_req_ids,
    tvm::ffi::TensorView write_req_seed_slot_indices,
    tvm::ffi::TensorView write_req_entry_starts,
    tvm::ffi::TensorView write_req_entry_counts,
    tvm::ffi::TensorView write_req_active_mask,
    int64_t seed,
    tvm::ffi::TensorView violation_ring,
    tvm::ffi::TensorView violation_ring_valid,
    tvm::ffi::TensorView violation_write_index,
    tvm::ffi::TensorView first_violation,
    tvm::ffi::TensorView first_violation_set,
    tvm::ffi::TensorView is_errored,
    tvm::ffi::TensorView slot_run_counter,
    tvm::ffi::TensorView kernel_run_counter,
    int64_t kernel_kind,
    tvm::ffi::TensorView real_kv_buf,
    int64_t real_kv_slot_stride_bytes,
    int64_t real_kv_read_bytes,
    int64_t real_kv_hash_mode) {
  using namespace host;

  SymbolicSize N_verify = {"num_verify"};
  SymbolicSize N_write = {"num_write"};
  SymbolicSize N_write_reqs = {"num_write_reqs"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N_verify})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(verify_slot_indices)
      .verify(verify_positions)
      .verify(verify_req_ids)
      .verify(verify_prev_slot_indices);
  TensorMatcher({N_verify}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(verify_active_mask);

  TensorMatcher({N_write})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(write_slot_indices)
      .verify(write_token_ids)
      .verify(write_positions)
      .verify(write_req_ids);

  TensorMatcher({N_write_reqs})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(write_req_seed_slot_indices)
      .verify(write_req_entry_starts)
      .verify(write_req_entry_counts);
  TensorMatcher({N_write_reqs}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(write_req_active_mask);

  const uint32_t num_verify = static_cast<uint32_t>(N_verify.unwrap());
  const uint32_t num_write_reqs = static_cast<uint32_t>(N_write_reqs.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      slot_stride_bytes >= static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      "canary: slot_stride_bytes must hold at least ",
      static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      " bytes per slot, got ",
      slot_stride_bytes);
  RuntimeCheck(
      violation_ring.size(1) == kViolationFields,
      "canary: violation_ring last dim must be ",
      kViolationFields,
      ", got ",
      violation_ring.size(1));
  const uint32_t ring_capacity = static_cast<uint32_t>(violation_ring.size(0));
  RuntimeCheck(ring_capacity > 0, "canary: violation_ring capacity must be positive");
  RuntimeCheck(
      violation_ring_valid.size(0) == static_cast<int64_t>(ring_capacity),
      "canary: violation_ring_valid first dim must equal ring_capacity");

  const uint32_t total_threads = num_verify + num_write_reqs;
  // No early return on total_threads == 0: the kernel still has the
  // unconditional kernel_run_counter atomicAdd at its entry, which the
  // host-side health monitor relies on to detect "canary actually ran"
  // across warmup / no-work forwards.

  CanaryParams p{};
  p.src_buf = static_cast<const uint8_t*>(src_buf.data_ptr());
  p.dst_buf = static_cast<uint8_t*>(dst_buf.data_ptr());
  p.slot_stride_bytes = slot_stride_bytes;
  p.verify_slot_indices = static_cast<const int64_t*>(verify_slot_indices.data_ptr());
  p.verify_positions = static_cast<const int64_t*>(verify_positions.data_ptr());
  p.verify_req_ids = static_cast<const int64_t*>(verify_req_ids.data_ptr());
  p.verify_prev_slot_indices = static_cast<const int64_t*>(verify_prev_slot_indices.data_ptr());
  p.verify_active_mask = static_cast<const int32_t*>(verify_active_mask.data_ptr());
  p.num_verify = num_verify;
  p.write_slot_indices = static_cast<const int64_t*>(write_slot_indices.data_ptr());
  p.write_token_ids = static_cast<const int64_t*>(write_token_ids.data_ptr());
  p.write_positions = static_cast<const int64_t*>(write_positions.data_ptr());
  p.write_req_ids = static_cast<const int64_t*>(write_req_ids.data_ptr());
  p.write_req_seed_slot_indices = static_cast<const int64_t*>(write_req_seed_slot_indices.data_ptr());
  p.write_req_entry_starts = static_cast<const int64_t*>(write_req_entry_starts.data_ptr());
  p.write_req_entry_counts = static_cast<const int64_t*>(write_req_entry_counts.data_ptr());
  p.write_req_active_mask = static_cast<const int32_t*>(write_req_active_mask.data_ptr());
  p.num_write_reqs = num_write_reqs;
  p.seed = static_cast<uint64_t>(seed);
  p.violation_ring = static_cast<int64_t*>(violation_ring.data_ptr());
  p.violation_ring_valid = static_cast<int32_t*>(violation_ring_valid.data_ptr());
  p.violation_write_index = static_cast<uint32_t*>(violation_write_index.data_ptr());
  p.first_violation = static_cast<int64_t*>(first_violation.data_ptr());
  p.first_violation_set = static_cast<uint32_t*>(first_violation_set.data_ptr());
  p.is_errored = static_cast<int32_t*>(is_errored.data_ptr());
  p.slot_run_counter = static_cast<uint64_t*>(slot_run_counter.data_ptr());
  p.kernel_run_counter = static_cast<uint64_t*>(kernel_run_counter.data_ptr());
  p.ring_capacity = ring_capacity;
  p.kernel_kind = static_cast<int32_t>(kernel_kind);
  // Real-KV plumbing: pass through the host's pointer + stride + mode.
  // ``real_kv_buf`` may be a zero-byte tensor when the feature is OFF;
  // ``data_ptr()`` is still safe to call (it just returns a pointer that
  // the kernel never dereferences in OFF mode).
  p.real_kv_buf = static_cast<const uint8_t*>(real_kv_buf.data_ptr());
  p.real_kv_slot_stride_bytes = real_kv_slot_stride_bytes;
  p.real_kv_read_bytes = real_kv_read_bytes;
  p.real_kv_hash_mode = static_cast<int32_t>(real_kv_hash_mode);

  constexpr uint32_t kBlockSize = 128;
  // Always launch at least one block so the liveness atomicAdd at the
  // kernel entry runs even when total_threads == 0 (no verify entries
  // and no write-req chains).
  const uint32_t effective_threads = total_threads == 0 ? 1u : total_threads;
  const uint32_t grid = (effective_threads + kBlockSize - 1) / kBlockSize;
  LaunchKernel(grid, kBlockSize, device)(canary_kernel, p);
}

}  // namespace
