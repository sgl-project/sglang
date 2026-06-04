#pragma once

#include <cstdint>

namespace canary {

constexpr uint64_t kCanaryChainAnchor = 0xC0FFEE1234567890ULL;

// Mirrors SGLang's TokenToKVPoolAllocator contract: token-to-KV slot 0 is reserved for padded-token dummy
// writes. Since req_to_token stores token-to-KV slot ids and is zero-initialized, canary slot 0 is skipped
// instead of treating unfilled entries as real KV slots.
constexpr int64_t kTokenToKvSlotPadding = 0;
constexpr int64_t kReqPoolIdxPadding = 0;

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
  kVerifyChainHashMismatch = 1LL << 0,
  kVerifyPositionMismatch = 1LL << 1,
  kVerifyRealKvHashMismatch = 1LL << 2,
  kWriteTokenMismatch = 1LL << 3,
  kWritePositionMismatch = 1LL << 4,
  kVerifyTokenMismatch = 1LL << 5,
};

constexpr FailReason operator|(FailReason a, FailReason b) {
  return static_cast<FailReason>(static_cast<int64_t>(a) | static_cast<int64_t>(b));
}

constexpr FailReason& operator|=(FailReason& a, FailReason b) {
  a = a | b;
  return a;
}

enum class RealKvHashMode : int32_t {
  kNone = 0,
  kPartial = 1,
  kAll = 2,
};

constexpr int kMaxRealKvSources = 4;

constexpr int kRealKvSourceFieldsPerEntry = 3;
constexpr int kRealKvSourceFieldPageSize = 0;
constexpr int kRealKvSourceFieldNumBytesPerToken = 1;
constexpr int kRealKvSourceFieldReadBytes = 2;

}  // namespace canary
