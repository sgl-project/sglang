from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


VIOLATION_FIELDS: int = 10
CANARY_FIELDS_PER_SLOT: int = 5
CANARY_SLOT_BYTES: int = CANARY_FIELDS_PER_SLOT * 8

# Slot field offsets (in 8-byte words). Mirrors the kCanaryField* constants
# in ``canary.cuh``.
_CANARY_FIELD_REQ_ID: int = 0
_CANARY_FIELD_TOKEN_ID: int = 1
_CANARY_FIELD_POSITION: int = 2
_CANARY_FIELD_PREV_HASH: int = 3
# real_kv_hash: splitmix64 mix of a few bytes of the real KV pool's slot
# data captured at write time. Verified by reading the same bytes again
# at verify time; mismatch implies the real KV slot changed underneath
# the canary (the attn-kernel-config / PD-transfer corruption modes
# canary-with-real-data is designed to catch).
_CANARY_FIELD_REAL_KV_HASH: int = 4

# Modes for ``--kv-cache-canary-real-data``. ``OFF`` disables the
# real-KV mix entirely (the real_kv_hash field stays zero); ``BIT`` mixes
# a 16-byte prefix of the real slot; ``ALL`` mixes the full real-slot
# stride. Mirrored in C++ as ``kRealKvHashMode*`` constants.
REAL_KV_HASH_MODE_OFF: int = 0
REAL_KV_HASH_MODE_BIT: int = 1
REAL_KV_HASH_MODE_ALL: int = 2
REAL_KV_HASH_BIT_BYTES: int = 16

# Violation-row field offsets. Mirrors the kViolationField* constants.
_VIOLATION_FIELD_KERNEL_KIND: int = 0
_VIOLATION_FIELD_FAIL_REASON: int = 1
_VIOLATION_FIELD_SLOT_IDX: int = 2
_VIOLATION_FIELD_REQ_ID: int = 3
_VIOLATION_FIELD_TOKEN_ID: int = 4
_VIOLATION_FIELD_POSITION: int = 5
_VIOLATION_FIELD_EXPECTED_HASH: int = 6
_VIOLATION_FIELD_ACTUAL_HASH: int = 7
_VIOLATION_FIELD_EXPECTED_REQ_ID: int = 8
_VIOLATION_FIELD_EXPECTED_POSITION: int = 9

KERNEL_KIND_HEAD: int = 0
KERNEL_KIND_TAIL: int = 1

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
    REQ_ID = 1
    TOKEN_ID = 2
    POSITION = 3
    HASH = 4
    POSITION_MONOTONIC = 5
    REAL_KV_HASH = 6


@cache_once
def _jit_canary_module() -> "Module":
    return load_jit(
        "kv_cache_canary",
        cuda_files=["kv_cache_canary/canary.cuh"],
        cuda_wrappers=[("canary_step", "canary_step")],
    )


def canary_step(
    *,
    src_buf: torch.Tensor,
    dst_buf: torch.Tensor,
    slot_stride_bytes: int,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_req_ids: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    verify_active_mask: torch.Tensor,
    write_slot_indices: torch.Tensor,
    write_token_ids: torch.Tensor,
    write_positions: torch.Tensor,
    write_req_ids: torch.Tensor,
    write_req_seed_slot_indices: torch.Tensor,
    write_req_entry_starts: torch.Tensor,
    write_req_entry_counts: torch.Tensor,
    write_req_active_mask: torch.Tensor,
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
    """Launch one KV-cache canary step against the shadow buffers.

    Per-forward host hook into the compiled CUDA kernel. Each call
    optionally verifies a set of slots and optionally writes a set of
    new slots, in a single grid launch on the current stream. All output
    state (the destination buffer, violation ring, is_errored flag,
    counters) is mutated in-place; this function returns ``None``.

    Args:
        src_buf:                Flattened ``uint8`` view of the source
                                shadow tensor (verify reads + prev-slot
                                reads come from here).
        dst_buf:                Flattened ``uint8`` view of the
                                destination shadow tensor (writes land
                                here). Aliasing ``src_buf`` is allowed.
        slot_stride_bytes:      Bytes per logical slot in both buffers.
        verify_slot_indices:    ``int64 [N_verify]`` — slots to verify.
        verify_positions:       ``int64 [N_verify]`` — expected positions.
        verify_req_ids:         ``int64 [N_verify]`` — expected req ids.
        verify_prev_slot_indices: ``int64 [N_verify]`` — slot index of
                                position ``p - 1`` for each verify entry,
                                or ``-1`` for "position 0, seed from
                                kSeed".
        verify_active_mask:     ``int32 [N_verify]`` — 1 = process, 0 =
                                skip (cuda-graph padding).
        write_slot_indices:     ``int64 [N_write]`` — slots to write.
        write_token_ids:        ``int64 [N_write]`` — token IDs to store.
        write_positions:        ``int64 [N_write]`` — positions to store.
        write_req_ids:          ``int64 [N_write]`` — req IDs to store.
        write_req_seed_slot_indices: ``int64 [N_write_reqs]`` — for each
                                write-req, the slot to seed the chain
                                from (``-1`` = seed from kSeed).
        write_req_entry_starts: ``int64 [N_write_reqs]`` — start offset
                                into the per-write-entry arrays.
        write_req_entry_counts: ``int64 [N_write_reqs]`` — count of
                                entries in the per-write-entry arrays.
        write_req_active_mask:  ``int32 [N_write_reqs]`` — 1 = process,
                                0 = skip.
        seed:                   Chain seed (``CanaryConfig.seed``).
                                Cast through :func:`to_signed_int64`
                                before being handed to the kernel.
        violation_ring:         ``int64 [ring_capacity, VIOLATION_FIELDS]``.
        violation_ring_valid:   ``int32 [ring_capacity]``.
        violation_write_index:  ``int32 [1]`` — ring write cursor.
        first_violation:        ``int64 [VIOLATION_FIELDS]`` — latched
                                first violation row.
        first_violation_set:    ``int32 [1]`` — CAS latch flag.
        is_errored:             ``int32 [1]`` — set to 1 on any violation.
        slot_run_counter:       ``int64 [1]`` — ``+= num_active_slots``.
        kernel_run_counter:     ``int64 [1]`` — ``+= 1`` unconditionally
                                (host-side health monitor relies on this).
        kernel_kind:            :data:`KERNEL_KIND_HEAD` or
                                :data:`KERNEL_KIND_TAIL`.
        real_kv_buf:            Flattened ``uint8`` view of the real-KV
                                pool's source layer (used by the optional
                                real-data fingerprint).
        real_kv_slot_stride_bytes: Byte stride of the real-KV pool.
        real_kv_read_bytes:     Number of leading bytes per slot to fold
                                into the fingerprint (0 disables).
        real_kv_hash_mode:      One of :data:`REAL_KV_HASH_MODE_OFF`,
                                :data:`REAL_KV_HASH_MODE_BIT`,
                                :data:`REAL_KV_HASH_MODE_ALL`.

    Calling contract:

    - Safe to call inside cuda-graph capture; the recorded kernel uses
      the buffer addresses passed here. Replay-side code must refill the
      buffers in-place before ``graph.replay()``.
    - Always launches at least one thread (the kernel performs an
      unconditional ``kernel_run_counter`` atomicAdd in block 0 / thread
      0 before any other work); skipping the launch breaks the host
      health monitor.
    - For the per-forward write/verify semantics see
      :func:`sglang.jit_kernel.kv_cache_canary_ref.canary_step_torch_reference`,
      which is a byte-equal reference implementation.
    """
    module = _jit_canary_module()
    module.canary_step(
        src_buf,
        dst_buf,
        slot_stride_bytes,
        verify_slot_indices,
        verify_positions,
        verify_req_ids,
        verify_prev_slot_indices,
        verify_active_mask,
        write_slot_indices,
        write_token_ids,
        write_positions,
        write_req_ids,
        write_req_seed_slot_indices,
        write_req_entry_starts,
        write_req_entry_counts,
        write_req_active_mask,
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
