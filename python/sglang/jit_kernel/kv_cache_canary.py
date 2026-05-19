from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Dict, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


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
# real-KV mix entirely (the real_kv_hash field stays zero); ``BIT`` mixes
# a 16-byte prefix of the real slot; ``ALL`` mixes the full real-slot
# stride. Mirrored in C++ as ``kRealKvHashMode*`` constants.
REAL_KV_HASH_MODE_OFF: int = 0
REAL_KV_HASH_MODE_BIT: int = 1
REAL_KV_HASH_MODE_ALL: int = 2
REAL_KV_HASH_BIT_BYTES: int = 16

# Skip-sentinel value for expected_write_{token_ids,positions}. Mirrored
# as ``kCanaryExpectedSkipSentinel`` in canary.cuh.
CANARY_EXPECTED_SKIP_SENTINEL: int = -1

# Sentinel for verify_prev_slot_indices: skip the chain hash check on
# this entry. Distinct from -1 (chain head, expected_prev_hash = seed).
# Mirrored as ``kSkipChainSentinel`` in canary.cuh.
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
    every uint64 -> int64 boundary (e.g. passing ``CanaryConfig.seed`` into
    the C++ kernel).
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
    "kRealKvHashModeBit",
    "kRealKvHashModeAll",
    "kCanaryExpectedSkipSentinel",
    "kSkipChainSentinel",
    "kKernelKindHead",
    "kKernelKindTail",
    "kKernelKindSweep",
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
    src_buf: torch.Tensor,
    dst_buf: torch.Tensor,
    slot_stride_bytes: int,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    verify_num_valid: torch.Tensor,
    write_slot_indices: torch.Tensor,
    write_token_ids: torch.Tensor,
    write_positions: torch.Tensor,
    write_req_seed_slot_indices: torch.Tensor,
    write_req_entry_starts: torch.Tensor,
    write_req_entry_counts: torch.Tensor,
    write_req_num_valid: torch.Tensor,
    expected_write_token_ids: torch.Tensor,
    expected_write_positions: torch.Tensor,
    seed: int,
    violation_ring: torch.Tensor,
    violation_ring_valid: torch.Tensor,
    violation_write_index: torch.Tensor,
    first_violation: torch.Tensor,
    first_violation_set: torch.Tensor,
    is_errored: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    kernel_kind: int,
    real_kv_buf: torch.Tensor,
    real_kv_slot_stride_bytes: int,
    real_kv_read_bytes: int,
    real_kv_hash_mode: int,
) -> None:
    """One step of the KV-cache canary protocol against the canary buffer.

    Shadow slot layout — each logical token tracked by the canary occupies one slot of ``slot_stride_bytes`` bytes
    in ``src_buf`` / ``dst_buf``, holding 4 ``int64`` fields (see ``_CANARY_FIELD_*`` for offsets):

    - ``token_id``     — vocab id of the token this slot represents.
    - ``position``     — sequence position of the token within its request.
    - ``prev_hash``    — running fingerprint over the entire 4-field tuple of every chain element *strictly before*
                         this slot. Anchored on ``seed`` for the chain head. Successive slots are linked by
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
       either a designated predecessor slot (carrying chain state forward from a prior step) or the global ``seed``
       (chain start), and stamping each slot's 4 fields.

    The two are independent: a call may verify only, write only, or do both. Returns ``None``; all effects are
    in-place mutation of the output tensors listed below.

    Args:
        src_buf:                     Flat ``uint8`` view of the canary canary tensor that verify reads from.
        dst_buf:                     Flat ``uint8`` view of the canary canary tensor that writes land in. Aliasing
                                     ``src_buf`` is allowed.
        slot_stride_bytes:           Bytes per logical slot in the canary buffer.
        verify_slot_indices:         ``int64 [N_verify]`` — slot of each verify entry.
        verify_positions:            ``int64 [N_verify]`` — position the caller expects that slot to carry.
        verify_prev_slot_indices:    ``int64 [N_verify]`` — slot of the predecessor in the chain, or ``-1`` to anchor
                                     the chain on ``seed`` (no predecessor available, e.g. position 0 or an SWA
                                     window head).
        verify_num_valid:            ``int32 [1]`` — number of leading verify entries to process. Remaining entries
                                     are cuda-graph padding and are skipped.
        write_slot_indices,
        write_token_ids,
        write_positions:             ``int64 [N_write]`` — per-slot payload to install, flattened across all
                                     write-reqs.
        write_req_seed_slot_indices: ``int64 [N_write_reqs]`` — for each write-req, the slot to anchor its chain on,
                                     or ``-1`` to anchor on ``seed``.
        write_req_entry_starts,
        write_req_entry_counts:      ``int64 [N_write_reqs]`` — slice of the per-slot arrays owned by each write-req.
        write_req_num_valid:         ``int32 [1]`` — number of leading write-req rows to process.
        expected_write_token_ids,
        expected_write_positions:    ``int64 [N_write]`` — per-write-entry oracle predictions for the input token and
                                     position. A value of ``-1`` is the skip-sentinel: the kernel skips that entry's
                                     input-mismatch check. Non-pseudo callers fill both buffers with ``-1`` and pay
                                     no per-entry cost beyond two loads and two compares.
        seed:                        Chain anchor used wherever a slot has no predecessor (``CanaryConfig.seed``).
        violation_ring:              ``int64 [ring_capacity, VIOLATION_FIELDS]`` — append-only sink. Each populated
                                     row is fill-once: never overwritten.
        violation_ring_valid:        ``int32 [ring_capacity]`` — per-row "occupied" flag.
        violation_write_index:       ``int32 [1]`` — monotonic violation counter. Advanced on every recorded violation
                                     regardless of whether the ring still has room.
        first_violation:             ``int64 [VIOLATION_FIELDS]`` — sticky copy of the first violation ever seen on
                                     this buffer (latched across calls; never cleared by the kernel).
        first_violation_set:         ``int32 [1]`` — latch flag for ``first_violation``.
        is_errored:                  ``int32 [1]`` — set to 1 the moment any violation lands; never cleared by the
                                     kernel.
        slot_run_counter:            ``int64 [1]`` — incremented by the total number of active verify entries plus
                                     active write entries processed.
        kernel_run_counter:          ``int64 [1]`` — incremented by exactly 1 per call, unconditionally. The host
                                     health monitor uses this to prove the canary hook actually ran.
        kernel_kind:                 :data:`KERNEL_KIND_HEAD` or :data:`KERNEL_KIND_TAIL` — stamped into every
                                     violation row so downstream reporting can tell which hook fired.
        real_kv_buf:                 Flat ``uint8`` view of the *real* KV pool layer this canary is mirroring. Read
                                     by the real-KV fingerprint mixin.
        real_kv_slot_stride_bytes:   Bytes per slot in ``real_kv_buf``.
        real_kv_read_bytes:          Number of leading bytes per real-KV slot folded into the fingerprint. ``0``
                                     disables the mixin.
        real_kv_hash_mode:           One of :data:`REAL_KV_HASH_MODE_OFF` / :data:`REAL_KV_HASH_MODE_BIT` /
                                     :data:`REAL_KV_HASH_MODE_ALL`. ``OFF`` makes the mixin a no-op regardless of
                                     ``real_kv_read_bytes``.

    Calling contract:

    - Pure side-effect: returns ``None``; only the output tensors listed above are mutated. Inputs are read-only.
    - Verification is out-of-band: failed checks never raise and never block. Callers inspect ``is_errored`` /
      ``first_violation`` / ``violation_ring`` asynchronously.
    - ``kernel_run_counter`` is bumped on every call — even one with zero active verify + zero active write entries
      — and is the canonical "did the canary actually run" signal.
    - When multiple violations land in one call, all of them set ``is_errored`` and each advances
      ``violation_write_index``; which one wins the ``first_violation`` latch and the relative ordering of rows in
      the ring are unspecified.
    - Safe to invoke inside cuda-graph capture. Replay reuses the buffer addresses captured here, so callers must
      refill the input tensors in-place before ``graph.replay()``.

    The full per-field semantics (chain hashing, fail-reason priority, ring/latch update rules) are pinned by the
    executable reference :func:`sglang.jit_kernel.kv_cache_canary_ref.canary_step_torch_reference`, which the CUDA
    path must match byte-for-byte.
    """
    module = _jit_canary_module()
    module.canary_step(
        src_buf,
        dst_buf,
        slot_stride_bytes,
        verify_slot_indices,
        verify_positions,
        verify_prev_slot_indices,
        verify_num_valid,
        write_slot_indices,
        write_token_ids,
        write_positions,
        write_req_seed_slot_indices,
        write_req_entry_starts,
        write_req_entry_counts,
        write_req_num_valid,
        expected_write_token_ids,
        expected_write_positions,
        to_signed_int64(int(seed)),
        violation_ring,
        violation_ring_valid,
        violation_write_index,
        first_violation,
        first_violation_set,
        is_errored,
        slot_run_counter,
        kernel_run_counter,
        kernel_kind,
        real_kv_buf,
        real_kv_slot_stride_bytes,
        real_kv_read_bytes,
        real_kv_hash_mode,
    )
