#pragma once

#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE

#include "consts.cuh"
#include <cstdint>

namespace canary {

SGL_DEVICE uint64_t splitmix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

SGL_DEVICE uint64_t splitmix64_mix3(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t h = splitmix64(a);
  h = splitmix64(h ^ b);
  h = splitmix64(h ^ c);
  return h;
}

struct ViolationSink {
  int64_t* __restrict__ ring;
  int32_t* __restrict__ write_index;
  int32_t ring_capacity;
  int32_t kernel_kind;
};

struct ViolationRow {
  int64_t slot_idx;
  int64_t position;
  int64_t stored_token;
  int64_t expected_token;
  int64_t stored_chain_hash;
  int64_t expected_aux;
  int64_t fail_reason_bits;
};

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

SGL_DEVICE uint64_t compute_slot_hash(const uint8_t* canary_buf, int64_t slot_stride_bytes, int64_t source_slot_idx) {
  if (source_slot_idx < 0) {
    return splitmix64(kCanaryChainAnchor);
  }
  const int64_t token = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldToken);
  const int64_t position = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldPosition);
  const int64_t prev_hash = canary_load_field(canary_buf, source_slot_idx, slot_stride_bytes, kCanaryFieldPrevHash);
  return splitmix64_mix3(
      static_cast<uint64_t>(prev_hash), static_cast<uint64_t>(token), static_cast<uint64_t>(position));
}

}  // namespace canary
