from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Final

CANARY_CHAIN_ANCHOR: Final[int] = 0xC0FFEE1234567890

# Mirrors SGLang's ReqToTokenPool contract: row 0 is reserved as the CUDA-graph padding row and real
# req_pool_idx values start at 1. Canary-attached pools reserve the same slot so zero-initialized
# req_to_token entries translate to a skipped padding sentinel instead of spurious violations.
CANARY_RESERVED_SLOT: Final[int] = 0

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


_U64_MASK: int = (1 << 64) - 1


def splitmix64(value: int) -> int:
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def splitmix64_mix4(a: int, b: int, c: int, d: int) -> int:
    """Chained 4-input splitmix64. Folds each input into a running accumulator via
    ``acc = splitmix64(acc ^ next)``; order-sensitive, no XOR self-cancellation between equal inputs.
    Must stay byte-equal to ``splitmix64_mix4`` in csrc/kv_canary/canary_common.cuh.
    """
    h = splitmix64(a & _U64_MASK)
    h = splitmix64(h ^ (b & _U64_MASK))
    h = splitmix64(h ^ (c & _U64_MASK))
    h = splitmix64(h ^ (d & _U64_MASK))
    return h
