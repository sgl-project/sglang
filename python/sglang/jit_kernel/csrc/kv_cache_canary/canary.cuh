#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kCanaryFieldsPerSlot = 4;
constexpr int kCanaryFieldReqId = 0;
constexpr int kCanaryFieldTokenId = 1;
constexpr int kCanaryFieldPosition = 2;
constexpr int kCanaryFieldPrevHash = 3;

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

struct CanaryParams {
  // Source buffer: read prior canary slots from here (verification target).
  const uint8_t* __restrict__ src_buf;
  // Destination buffer: write the new canary slots here.
  uint8_t* __restrict__ dst_buf;
  // Bytes per slot in both src and dst (= per-slot stride of the underlying real-layer-shaped tensor).
  int64_t slot_stride_bytes;

  // Per-token-slot inputs. All have shape [num_slots].
  const int64_t* __restrict__ slot_indices;
  const int64_t* __restrict__ expected_req_ids;
  const int64_t* __restrict__ expected_token_ids;
  const int64_t* __restrict__ expected_positions;
  const int64_t* __restrict__ expected_prev_hashes;
  // 1 = verify-then-write, 0 = write-only (first-write skip)
  const int32_t* __restrict__ verify_mask;
  // Per-verify-slot 0..K monotonic expected position; -1 for write-only slots.
  // Position-monotonic check (README §3 (b)) compares the slot's stored
  // ``input_position`` against this value WITHOUT consulting any expected
  // table that derives from the request→token map. It is the indirection-free
  // check that catches map-pointing-wrong bugs.
  const int64_t* __restrict__ verify_seq_positions;
  uint32_t num_slots;

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
};

__device__ inline int64_t load_field(const uint8_t* buf, int64_t slot_idx, int64_t stride, int field) {
  const int64_t* p = reinterpret_cast<const int64_t*>(buf + slot_idx * stride);
  return p[field];
}

__device__ inline void store_field(uint8_t* buf, int64_t slot_idx, int64_t stride, int field, int64_t value) {
  int64_t* p = reinterpret_cast<int64_t*>(buf + slot_idx * stride);
  p[field] = value;
}

__device__ inline void record_violation(
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

__global__ void canary_kernel(const CanaryParams __grid_constant__ p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= p.num_slots) {
    return;
  }

  const int64_t slot_idx = p.slot_indices[tid];
  const int64_t expected_req_id = p.expected_req_ids[tid];
  const int64_t expected_token_id = p.expected_token_ids[tid];
  const int64_t expected_position = p.expected_positions[tid];
  const uint64_t expected_prev_hash = static_cast<uint64_t>(p.expected_prev_hashes[tid]);
  const int32_t do_verify = p.verify_mask[tid];
  const int64_t verify_seq_position = p.verify_seq_positions[tid];

  if (do_verify != 0) {
    const int64_t actual_req_id = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldReqId);
    const int64_t actual_token_id = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId);
    const int64_t actual_position = load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
    const uint64_t actual_prev_hash =
        static_cast<uint64_t>(load_field(p.src_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash));

    int32_t fail_reason = 0;
    // (b) position monotonic check. Independent of any req→token table:
    // the slot's stored ``input_position`` field must equal the expected
    // sequence index 0..K we baked into the verify entry.
    if (verify_seq_position >= 0 && actual_position != verify_seq_position) {
      fail_reason = kFailReasonPositionMonotonic;
    } else if (actual_req_id != expected_req_id) {
      fail_reason = kFailReasonReqId;
    } else if (actual_token_id != expected_token_id) {
      fail_reason = kFailReasonTokenId;
    } else if (actual_position != expected_position) {
      fail_reason = kFailReasonPosition;
    } else if (actual_prev_hash != expected_prev_hash) {
      fail_reason = kFailReasonHash;
    }
    if (fail_reason != 0) {
      record_violation(
          p,
          fail_reason,
          slot_idx,
          actual_req_id,
          actual_token_id,
          actual_position,
          expected_prev_hash,
          actual_prev_hash);
    }
  } else {
    // do_verify == 0 means "write-only" entry. Per README §2, slot[i] stores
    // the chain hash *before* position i (= ``expected_prev_hash``); the host
    // advances the chain itself, so we do NOT mix the current token into the
    // value we store.
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldReqId, expected_req_id);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldTokenId, expected_token_id);
    store_field(p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPosition, expected_position);
    store_field(
        p.dst_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash, static_cast<int64_t>(expected_prev_hash));
  }

  atomicAdd(reinterpret_cast<unsigned long long*>(p.slot_run_counter), 1ULL);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p.kernel_run_counter), 1ULL);
  }
}

void canary_step(
    tvm::ffi::TensorView src_buf,
    tvm::ffi::TensorView dst_buf,
    int64_t slot_stride_bytes,
    tvm::ffi::TensorView slot_indices,
    tvm::ffi::TensorView expected_req_ids,
    tvm::ffi::TensorView expected_token_ids,
    tvm::ffi::TensorView expected_positions,
    tvm::ffi::TensorView expected_prev_hashes,
    tvm::ffi::TensorView verify_mask,
    tvm::ffi::TensorView verify_seq_positions,
    tvm::ffi::TensorView violation_ring,
    tvm::ffi::TensorView violation_ring_valid,
    tvm::ffi::TensorView violation_write_index,
    tvm::ffi::TensorView first_violation,
    tvm::ffi::TensorView first_violation_set,
    tvm::ffi::TensorView is_errored,
    tvm::ffi::TensorView slot_run_counter,
    tvm::ffi::TensorView kernel_run_counter,
    int64_t kernel_kind) {
  using namespace host;

  SymbolicSize N = {"num_slots"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_indices)
      .verify(expected_req_ids)
      .verify(expected_token_ids)
      .verify(expected_positions)
      .verify(expected_prev_hashes)
      .verify(verify_seq_positions);
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(verify_mask);

  const uint32_t num_slots = static_cast<uint32_t>(N.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      slot_stride_bytes >= static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      "canary: slot_stride_bytes must hold at least 32 bytes per slot, got ",
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

  if (num_slots == 0) {
    return;
  }

  CanaryParams p{};
  p.src_buf = static_cast<const uint8_t*>(src_buf.data_ptr());
  p.dst_buf = static_cast<uint8_t*>(dst_buf.data_ptr());
  p.slot_stride_bytes = slot_stride_bytes;
  p.slot_indices = static_cast<const int64_t*>(slot_indices.data_ptr());
  p.expected_req_ids = static_cast<const int64_t*>(expected_req_ids.data_ptr());
  p.expected_token_ids = static_cast<const int64_t*>(expected_token_ids.data_ptr());
  p.expected_positions = static_cast<const int64_t*>(expected_positions.data_ptr());
  p.expected_prev_hashes = static_cast<const int64_t*>(expected_prev_hashes.data_ptr());
  p.verify_mask = static_cast<const int32_t*>(verify_mask.data_ptr());
  p.verify_seq_positions = static_cast<const int64_t*>(verify_seq_positions.data_ptr());
  p.num_slots = num_slots;
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

  constexpr uint32_t kBlockSize = 128;
  const uint32_t grid = (num_slots + kBlockSize - 1) / kBlockSize;
  LaunchKernel(grid, kBlockSize, device)(canary_kernel, p);
}

}  // namespace
