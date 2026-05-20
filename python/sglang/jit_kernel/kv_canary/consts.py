from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Final

CANARY_CHAIN_ANCHOR: Final[int] = 0xC0FFEE1234567890

CANARY_FIELDS_PER_SLOT: Final[int] = 4
CANARY_FIELD_TOKEN: Final[int] = 0
CANARY_FIELD_POSITION: Final[int] = 1
CANARY_FIELD_PREV_HASH: Final[int] = 2
CANARY_FIELD_REAL_KV_HASH: Final[int] = 3

VIOLATION_FIELDS: Final[int] = 8
VIOLATION_FIELD_KERNEL_KIND: Final[int] = 0
VIOLATION_FIELD_SLOT_IDX: Final[int] = 1
VIOLATION_FIELD_POSITION: Final[int] = 2
VIOLATION_FIELD_STORED_TOKEN: Final[int] = 3
VIOLATION_FIELD_EXPECTED_TOKEN: Final[int] = 4
VIOLATION_FIELD_STORED_CHAIN_HASH: Final[int] = 5
VIOLATION_FIELD_EXPECTED_AUX: Final[int] = 6
VIOLATION_FIELD_FAIL_REASON_BITS: Final[int] = 7


class FailReason(IntFlag):
    CHAIN_HASH = 1 << 0
    POSITION = 1 << 1
    REAL_KV_HASH = 1 << 2
    WRITE_TOKEN_MISMATCH = 1 << 3
    WRITE_POSITION_MISMATCH = 1 << 4


MAX_REAL_KV_SOURCES: Final[int] = 4

REAL_KV_SOURCE_FIELDS_PER_ENTRY: Final[int] = 3
REAL_KV_SOURCE_FIELD_PAGE_SIZE: Final[int] = 0
REAL_KV_SOURCE_FIELD_NUM_BYTES_PER_TOKEN: Final[int] = 1
REAL_KV_SOURCE_FIELD_READ_BYTES: Final[int] = 2


class RealKvHashMode(IntEnum):
    OFF = 0
    PARTIAL = 1
    ALL = 2


class CanaryPseudoMode(IntEnum):
    OFF = 0
    ON = 1
