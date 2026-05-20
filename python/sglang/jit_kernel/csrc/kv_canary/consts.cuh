#pragma once

#include <cstdint>

namespace canary {

constexpr uint64_t kCanaryChainAnchor = 0xC0FFEE1234567890ULL;

// Slot 0 of every canary buffer is a reserved padding sentinel. Pools that attach a canary MUST reserve
// slot 0 (free_slots starts at 1) so unfilled req_to_token entries (zero-initialized) translate to this
// slot and the verify kernel skips them instead of raising spurious chain_hash / position violations.
constexpr int64_t kCanaryReservedSlot = 0;

constexpr int kCanaryFieldsPerSlot = 4;
constexpr int kCanaryFieldToken = 0;
constexpr int kCanaryFieldPosition = 1;
constexpr int kCanaryFieldPrevHash = 2;
constexpr int kCanaryFieldRealKvHash = 3;

constexpr int kViolationFields = 8;
constexpr int kViolationFieldKernelKind = 0;
constexpr int kViolationFieldSlotIdx = 1;
constexpr int kViolationFieldPosition = 2;
constexpr int kViolationFieldStoredToken = 3;
constexpr int kViolationFieldExpectedToken = 4;
constexpr int kViolationFieldStoredChainHash = 5;
constexpr int kViolationFieldExpectedAux = 6;
constexpr int kViolationFieldFailReasonBits = 7;

enum class FailReason : int64_t {
  kChainHash = 1LL << 0,
  kPosition = 1LL << 1,
  kRealKvHash = 1LL << 2,
  kWriteTokenMismatch = 1LL << 3,
  kWritePositionMismatch = 1LL << 4,
};

constexpr FailReason operator|(FailReason a, FailReason b) {
  return static_cast<FailReason>(static_cast<int64_t>(a) | static_cast<int64_t>(b));
}

constexpr FailReason& operator|=(FailReason& a, FailReason b) {
  a = a | b;
  return a;
}

enum class RealKvHashMode : int32_t {
  kOff = 0,
  kPartial = 1,
  kAll = 2,
};

enum class CanaryPseudoMode : int32_t {
  kOff = 0,
  kOn = 1,
};

constexpr int kMaxRealKvSources = 4;

constexpr int kRealKvSourceFieldsPerEntry = 3;
constexpr int kRealKvSourceFieldPageSize = 0;
constexpr int kRealKvSourceFieldNumBytesPerToken = 1;
constexpr int kRealKvSourceFieldReadBytes = 2;

}  // namespace canary
