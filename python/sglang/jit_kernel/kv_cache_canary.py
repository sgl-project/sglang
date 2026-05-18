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
) -> None:
    """Launch one KV-cache canary step.

    The kernel runs two distinct workloads sharing one grid:

    1. **Verify entries** (per-thread): read a slot's stored fingerprint,
       compute the expected ``prev_hash`` by reading the previous slot and
       running ``splitmix64_mix`` on-device, then compare against three
       expected fields (``req_id``, ``position``, ``prev_hash``) — one
       independent fail_reason per field. ``token_id`` is NOT verified on
       this path (historical tokens are not in the current
       ``ForwardBatch.input_ids``); token tampering is caught indirectly
       via the splitmix64 chain at the next position.
    2. **Write-req chains** (per-thread, one driver thread per request): walk
       the splitmix64 chain across all writes for that req, seeded from
       either ``kSeed`` (``write_req_seed_slot_indices == -1``) or the
       canary slot at ``K_req - 1``, and store ``(req_id, token_id, position,
       prev_hash)`` into each slot in order.

    The host emits **raw input data** (``token_id``, ``position``) from the
    sglang ``ForwardBatch`` — no host-side hash computation. ``seed`` is
    ``CanaryConfig.seed``.

    Args:
        src_buf:             uint8 flatten view of the source shadow tensor
                             (verify reads + prev-slot reads here).
        dst_buf:             uint8 flatten view of the destination shadow.
        slot_stride_bytes:   bytes per logical slot in both buffers.
        verify_*:            per-verify-entry arrays. ``verify_prev_slot_indices``
                             at ``-1`` means "position 0, expected prev_hash =
                             kSeed". ``verify_active_mask`` at 0 means
                             "skip-sentinel padding for cuda graph".
        write_*:             per-write-entry arrays (pure data; driver threads
                             read these).
        write_req_*:         per-write-req arrays, one row per req with new
                             writes. The driver thread walks
                             ``[entry_starts, entry_starts + entry_counts)``
                             sequentially, advancing splitmix64.
        seed:                ``CanaryConfig.seed`` (chain head).
        violation_*:         GPU ring + first-violation latch + is_errored
                             flag for error reporting.
        slot_run_counter:    int64 [1]. ``+= num_active_slots`` per launch.
        kernel_run_counter:  int64 [1]. ``+= 1`` per launch.
        kernel_kind:         ``KERNEL_KIND_HEAD`` or ``KERNEL_KIND_TAIL``.
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
        seed,
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
