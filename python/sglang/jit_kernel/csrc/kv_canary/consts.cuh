#pragma once

#include <cstdint>

namespace canary {

constexpr uint64_t kCanaryChainAnchor = 0xC0FFEE1234567890ULL;

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

constexpr int64_t kFailReasonChainHash = 1LL << 0;
constexpr int64_t kFailReasonPosition = 1LL << 1;
constexpr int64_t kFailReasonRealKvHash = 1LL << 2;
constexpr int64_t kFailReasonWriteTokenMismatch = 1LL << 3;
constexpr int64_t kFailReasonWritePositionMismatch = 1LL << 4;

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
