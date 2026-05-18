from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


VIOLATION_FIELDS: int = 8
CANARY_FIELDS_PER_SLOT: int = 4
CANARY_SLOT_BYTES: int = CANARY_FIELDS_PER_SLOT * 8

KERNEL_KIND_HEAD: int = 0
KERNEL_KIND_TAIL: int = 1


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


FAIL_REASON_REQ_ID: int = FailReason.REQ_ID.value
FAIL_REASON_TOKEN_ID: int = FailReason.TOKEN_ID.value
FAIL_REASON_POSITION: int = FailReason.POSITION.value
FAIL_REASON_HASH: int = FailReason.HASH.value
FAIL_REASON_POSITION_MONOTONIC: int = FailReason.POSITION_MONOTONIC.value


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
    slot_indices: torch.Tensor,
    expected_req_ids: torch.Tensor,
    expected_token_ids: torch.Tensor,
    expected_positions: torch.Tensor,
    expected_prev_hashes: torch.Tensor,
    verify_mask: torch.Tensor,
    verify_seq_positions: torch.Tensor,
    violation_ring: torch.Tensor,
    violation_ring_valid: torch.Tensor,
    violation_write_index: torch.Tensor,
    first_violation: torch.Tensor,
    first_violation_set: torch.Tensor,
    is_errored: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    kernel_kind: int,
) -> None:
    """Launch one KV-cache canary step.

    Each entry is either a verify (``verify_mask == 1``, reads ``src_buf``
    at the given slot index, checks the stored 4-int64 fingerprint against
    expected values, including the monotonic ``input_position`` from
    ``verify_seq_positions``) or a write (``verify_mask == 0``, writes the
    expected fingerprint into ``dst_buf`` at the slot index).

    Args:
        src_buf:               uint8 flatten view of the source layer-shaped shadow
                               tensor (verify path reads from here).
        dst_buf:               uint8 flatten view of the destination shadow tensor
                               (write path stores into here).
        slot_stride_bytes:     bytes per logical slot in both buffers.
        slot_indices:          int64 [num_slots]. Per-entry slot index in
                               ``src_buf`` / ``dst_buf``.
        expected_req_ids:      int64 [num_slots]. Expected ``req_pool_idx``.
        expected_token_ids:    int64 [num_slots]. Expected ``input_token_id``.
        expected_positions:    int64 [num_slots]. Expected ``input_position``.
        expected_prev_hashes:  int64 [num_slots]. Expected chain ``prev_hash``.
        verify_mask:           int32 [num_slots]. ``1`` = verify-then-skip-write,
                               ``0`` = write-only.
        verify_seq_positions:  int64 [num_slots]. Expected 0..K sequence
                               position for verify entries (``-1`` for writes).
                               Checked independently of any req→slot table.
        violation_ring:        int64 [ring_capacity, 8]. Circular log of
                               violations (ring buffer).
        violation_ring_valid:  int32 [ring_capacity]. Per-row valid latch
                               (0 = empty, 1 = writing, 2 = readable).
        violation_write_index: int32 [1]. ``atomicAdd`` target for sequence #.
        first_violation:       int64 [8]. First-violation latch (never
                               overwritten by cascades).
        first_violation_set:   int32 [1]. CAS guard for ``first_violation``.
        is_errored:            int32 [1]. ``atomicOr`` flag; set when ANY
                               thread records a violation.
        slot_run_counter:      int64 [1]. ``+= num_slots`` per launch.
        kernel_run_counter:    int64 [1]. ``+= 1`` per launch.
        kernel_kind:           ``KERNEL_KIND_HEAD`` or ``KERNEL_KIND_TAIL``.
    """
    module = _jit_canary_module()
    module.canary_step(
        src_buf,
        dst_buf,
        slot_stride_bytes,
        slot_indices,
        expected_req_ids,
        expected_token_ids,
        expected_positions,
        expected_prev_hashes,
        verify_mask,
        verify_seq_positions,
        violation_ring,
        violation_ring_valid,
        violation_write_index,
        first_violation,
        first_violation_set,
        is_errored,
        slot_run_counter,
        kernel_run_counter,
        kernel_kind,
    )
