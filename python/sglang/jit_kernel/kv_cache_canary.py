from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Dict, Final, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary_plan_ref_legacy import BatchPlanGpu
from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# Frozen chain anchor used wherever a slot has no predecessor. Mirrored
# in C++ as ``kCanaryChainAnchor`` in ``canary.cuh``; the const-sync test
# pins them together.
CANARY_CHAIN_ANCHOR: Final[int] = 0xC0FFEE1234567890

VIOLATION_FIELDS: int = 8
CANARY_FIELDS_PER_SLOT: int = 4
CANARY_SLOT_BYTES: int = CANARY_FIELDS_PER_SLOT * 8

# Slot field offsets (in 8-byte words). Mirrors the kCanaryField* constants
# in ``canary.cuh``.
_CANARY_FIELD_TOKEN_ID: int = 0
_CANARY_FIELD_POSITION: int = 1
_CANARY_FIELD_PREV_HASH: int = 2
# real_kv_hash: splitmix64 mix of a few bytes of the real KV pool's slot
# data captured at write time. Verified by reading the same bytes again
# at verify time; mismatch implies the real KV slot changed underneath
# the canary (the attn-kernel-config / PD-transfer corruption modes
# canary-with-real-data is designed to catch).
_CANARY_FIELD_REAL_KV_HASH: int = 3

# Modes for ``--kv-cache-canary-real-data``. ``OFF`` disables the
# real-KV mix entirely (the real_kv_hash field stays zero); ``PORTION``
# mixes a 16-byte prefix of the real slot; ``ALL`` mixes the full
# real-slot stride. Mirrored in C++ as ``kRealKvHashMode*`` constants.
REAL_KV_HASH_MODE_OFF: int = 0
REAL_KV_HASH_MODE_PORTION: int = 1
REAL_KV_HASH_MODE_ALL: int = 2
REAL_KV_HASH_PORTION_BYTES: int = 16

# Skip-sentinel value for expected_write_{token_ids,positions}. Mirrored
# as ``kCanaryExpectedSkipSentinel`` in canary.cuh.
CANARY_EXPECTED_SKIP_SENTINEL: int = -1

# Sentinel for verify_prev_slot_indices: skip the chain hash check on
# this entry. Distinct from -1 (chain head, expected_prev_hash =
# CANARY_CHAIN_ANCHOR). Mirrored as ``kSkipChainSentinel`` in canary.cuh.
SKIP_CHAIN_SENTINEL: int = -2

# Violation-row field offsets. Mirrors the kViolationField* constants.
_VIOLATION_FIELD_KERNEL_KIND: int = 0
_VIOLATION_FIELD_FAIL_REASON: int = 1
_VIOLATION_FIELD_SLOT_IDX: int = 2
_VIOLATION_FIELD_TOKEN_ID: int = 3
_VIOLATION_FIELD_POSITION: int = 4
_VIOLATION_FIELD_EXPECTED_HASH: int = 5
_VIOLATION_FIELD_ACTUAL_HASH: int = 6
_VIOLATION_FIELD_EXPECTED_POSITION: int = 7

KERNEL_KIND_HEAD: int = 0
KERNEL_KIND_TAIL: int = 1
KERNEL_KIND_SWEEP: int = 2

_U64_MASK: int = (1 << 64) - 1


def to_signed_int64(value: int) -> int:
    """Reinterpret an unsigned uint64 as signed int64 (for torch.int64 storage).

    Python ints are arbitrary-precision, so a uint64 value above 2^63 - 1
    overflows ``torch.tensor(..., dtype=torch.int64)`` with
    ``OverflowError: Python int too large to convert to C long``. Used at
    every uint64 -> int64 boundary (e.g. mirroring
    :data:`CANARY_CHAIN_ANCHOR` into a tensor).
    """
    value &= _U64_MASK
    if value >= (1 << 63):
        value -= 1 << 64
    return value


class FailReason(enum.IntEnum):
    """Mirror of C++ ``kFailReason*`` constants in ``canary.cuh``.

    Single source of truth — kernel and Python must agree on these integers.
    """

    NONE = 0
    TOKEN_ID = 1
    POSITION = 2
    HASH = 3
    POSITION_MONOTONIC = 4
    REAL_KV_HASH = 5
    INPUT_TOKEN_MISMATCH = 6
    INPUT_POSITION_MISMATCH = 7


@cache_once
def _jit_canary_module() -> "Module":
    return load_jit(
        "kv_cache_canary",
        cuda_files=["kv_cache_canary/canary.cuh"],
        cuda_wrappers=[
            ("canary_step", "canary_step"),
            ("canary_get_constants", "canary_get_constants"),
        ],
    )


# Names ordered exactly as ``canary_get_constants`` writes them in
# canary.cuh. Changing one side without the other is caught by
# ``test_unit_const_sync.py`` (the kernel-side fill is the source of
# truth; this list just labels the slots).
_CANARY_CONSTANT_LAYOUT: Tuple[str, ...] = (
    "kCanaryFieldsPerSlot",
    "kCanaryFieldTokenId",
    "kCanaryFieldPosition",
    "kCanaryFieldPrevHash",
    "kCanaryFieldRealKvHash",
    "kViolationFields",
    "kViolationFieldKernelKind",
    "kViolationFieldFailReason",
    "kViolationFieldSlotIdx",
    "kViolationFieldTokenId",
    "kViolationFieldPosition",
    "kViolationFieldExpectedHash",
    "kViolationFieldActualHash",
    "kViolationFieldExpectedPosition",
    "kFailReasonTokenId",
    "kFailReasonPosition",
    "kFailReasonHash",
    "kFailReasonPositionMonotonic",
    "kFailReasonRealKvHash",
    "kFailReasonInputTokenMismatch",
    "kFailReasonInputPositionMismatch",
    "kRealKvHashModeOff",
    "kRealKvHashModePortion",
    "kRealKvHashModeAll",
    "kCanaryExpectedSkipSentinel",
    "kSkipChainSentinel",
    "kKernelKindHead",
    "kKernelKindTail",
    "kKernelKindSweep",
    "kCanaryChainAnchor",
)


def get_cpp_constants() -> Dict[str, int]:
    """Read the canonical canary constants directly from the C++ kernel.

    Allocates a small CPU int64 tensor, invokes the kernel module's
    ``canary_get_constants`` host function (which fills the tensor with
    every shared ``constexpr int k...`` value), and returns a dict keyed
    by the C++ identifier. Use this as the ground truth in tests that
    want to assert Python's mirror integers have not drifted.
    """
    module = _jit_canary_module()
    out = torch.zeros(len(_CANARY_CONSTANT_LAYOUT), dtype=torch.int64, device="cpu")
    module.canary_get_constants(out)
    return dict(zip(_CANARY_CONSTANT_LAYOUT, (int(x) for x in out.tolist())))


def canary_step(
    *,
    buf: torch.Tensor,
    plan: BatchPlanGpu,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    kernel_kind: int,
    real_kv_buf: torch.Tensor,
    real_kv_read_bytes: int,
    real_kv_hash_mode: int,
) -> None:
    """One step of the KV-cache canary protocol against the canary buffer.

    Shadow slot layout — each logical token tracked by the canary occupies one slot of ``slot_bytes`` bytes
    in ``buf`` (the second dim of the 2D buffer), holding 4 ``int64`` fields
    (see ``_CANARY_FIELD_*`` for offsets):

    - ``token_id``     — vocab id of the token this slot represents.
    - ``position``     — sequence position of the token within its request.
    - ``prev_hash``    — running fingerprint over the entire 4-field tuple of every chain element *strictly before*
                         this slot. Anchored on :data:`CANARY_CHAIN_ANCHOR` for the chain head. Successive slots are
                         linked by
                         ``next.prev_hash == hash(this.prev_hash, this.token_id, this.position, this.real_kv_hash)``;
                         that link is exactly what Verify recomputes and asserts on. Folding ``real_kv_hash`` into
                         the chain means any tampering with a slot's *full* contents (not just token/position)
                         propagates forward and gets caught at every later slot's verify.
    - ``real_kv_hash`` — fingerprint of a configurable prefix of the *real* KV-pool slot at the same index, captured
                         at write time and re-checked at verify time. Zero when the real-KV mixin is disabled
                         (in which case the chain collapses to a token/position-only fingerprint).

    A call does two independent operations on the canary buffer:

    1. **Verify** — for each active verify entry, recompute the expected ``prev_hash`` from the slot's predecessor
       and check it against the slot's stored fields (plus ``position`` and, if enabled, the live ``real_kv_hash``).
       Mismatches are *recorded* (not raised) into the violation sink; verification never throws.
    2. **Write** — for each active write-req, populate a contiguous run of slots, walking the chain forward from
       either a designated predecessor slot (carrying chain state forward from a prior step) or
       :data:`CANARY_CHAIN_ANCHOR` (chain start), and stamping each slot's 4 fields.

    The two are independent: a call may verify only, write only, or do both. Returns ``None``; all effects are
    in-place mutation of the output tensors listed below.

    Args:
        buf:                         ``uint8 [num_slots, slot_bytes]`` — canary tensor that both verify reads from
                                     and write writes into (each endpoint is self-verifying). Bytes per slot are
                                     taken from ``buf.shape[1]``.
        plan:                        :class:`~sglang.jit_kernel.kv_cache_canary_plan_ref_legacy.BatchPlanGpu` bundling the
                                     13 per-verify-entry / per-write-entry / per-write-req plan tensors plus the
                                     ``verify_num_valid`` and ``write_req_num_valid`` active-count scalars. See the
                                     ``BatchPlanGpu`` docstring for the per-field layout, sentinel encodings, and the
                                     cuda-graph fixed-capacity lifecycle.
        violation_ring:              ``int64 [ring_capacity, VIOLATION_FIELDS]`` — fill-once append sink. Row ``0`` is
                                     the first violation (atomic ``seq == 0`` writer wins and never gets overwritten).
                                     Rows ``1..ring_capacity-1`` hold subsequent arrivals; once the ring fills,
                                     further violations are dropped from the ring but still advance
                                     ``violation_write_index``.
        violation_write_index:       ``int32 [1]`` — monotonic violation counter. Advanced on every recorded violation
                                     regardless of whether the ring still has room. ``> 0`` is the canonical "any
                                     violation has landed" signal; the host derives all of ``is_errored`` /
                                     ``first_violation`` / valid-ring-row count from this counter plus the ring.
        slot_run_counter:            ``int64 [1]`` — incremented by the total number of active verify entries plus
                                     active write entries processed.
        kernel_run_counter:          ``int64 [1]`` — incremented by exactly 1 per call, unconditionally. The host
                                     health monitor uses this to prove the canary hook actually ran.
        kernel_kind:                 :data:`KERNEL_KIND_HEAD` or :data:`KERNEL_KIND_TAIL` — stamped into every
                                     violation row so downstream reporting can tell which hook fired.
        real_kv_buf:                 ``uint8 [num_slots, real_kv_slot_stride_bytes]`` — view of the *real* KV pool
                                     layer this canary is mirroring. Read by the real-KV fingerprint mixin. In OFF
                                     mode the kernel never dereferences this buffer; a 2D placeholder of any shape
                                     is accepted.
        real_kv_read_bytes:          Number of leading bytes per real-KV slot folded into the fingerprint. ``0``
                                     disables the mixin.
        real_kv_hash_mode:           One of :data:`REAL_KV_HASH_MODE_OFF` / :data:`REAL_KV_HASH_MODE_PORTION` /
                                     :data:`REAL_KV_HASH_MODE_ALL`. ``OFF`` makes the mixin a no-op regardless of
                                     ``real_kv_read_bytes``.

    Calling contract:

    - Pure side-effect: returns ``None``; only the output tensors listed above are mutated. Inputs are read-only.
    - Verification is out-of-band: failed checks never raise and never block. Callers inspect
      ``violation_write_index`` / ``violation_ring`` asynchronously.
    - ``violation_write_index[0] >= 1`` is the canonical "any violation has landed" signal; ``violation_ring[0]`` is
      the first violation row whenever that condition holds. The atomic ``seq == 0`` writer is the unique first-
      violation owner and never gets overwritten, so the row 0 latch is consistent across cascading mismatches.
    - ``kernel_run_counter`` is bumped on every call — even one with zero active verify + zero active write entries
      — and is the canonical "did the canary actually run" signal.
    - When multiple violations land in one call, each advances ``violation_write_index`` and (if still in range)
      writes a distinct ring row; the relative ordering of rows in the ring across concurrent threads is unspecified.
    - Safe to invoke inside cuda-graph capture. Replay reuses the buffer addresses captured here, so callers must
      refill the input tensors in-place before ``graph.replay()``.

    The full per-field semantics (chain hashing, fail-reason priority, ring/latch update rules) are pinned by the
    executable reference :func:`sglang.jit_kernel.kv_cache_canary_ref.canary_step_torch_reference`, which the CUDA
    path must match byte-for-byte.
    """
    module = _jit_canary_module()
    module.canary_step(
        buf,
        plan.verify_slot_indices,
        plan.verify_positions,
        plan.verify_prev_slot_indices,
        plan.verify_num_valid,
        plan.write_slot_indices,
        plan.write_token_ids,
        plan.write_positions,
        plan.write_req_seed_slot_indices,
        plan.write_req_entry_starts,
        plan.write_req_entry_counts,
        plan.write_req_num_valid,
        plan.expected_write_token_ids,
        plan.expected_write_positions,
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
        kernel_kind,
        real_kv_buf,
        real_kv_read_bytes,
        real_kv_hash_mode,
    )
