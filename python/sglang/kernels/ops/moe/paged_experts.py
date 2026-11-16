"""Paged-experts residency kernels: the per-step decision and the expert page-in on the GPU.

Used by ``sglang.srt.layers.moe.paged_experts`` so a decode step pages experts without a host
sync and can be captured in a CUDA graph. All index tensors are int32 on the GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "paged_experts",
        cuda_files=["moe/paged_experts.cuh"],
        cuda_wrappers=[
            ("paged_experts_decide", "paged_experts_decide"),
            ("paged_experts_gather", "paged_experts_gather"),
            ("paged_experts_rows_to_slots", "paged_experts_rows_to_slots"),
            ("paged_experts_tiered_plan", "paged_experts_tiered_plan"),
            ("paged_experts_await_rows", "paged_experts_await_rows"),
            (
                "paged_experts_host_device_pointer",
                "paged_experts_host_device_pointer",
            ),
        ],
    )


def paged_experts_decide(
    topk: torch.Tensor,
    topk_weights: torch.Tensor,
    step: torch.Tensor,
    slot_expert: torch.Tensor,
    expert_slot: torch.Tensor,
    slot_lastuse: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    count: torch.Tensor,
    run_ids: torch.Tensor,
    run_w: torch.Tensor,
) -> None:
    """Make every expert in ``topk`` (``[T]`` routed ids, negative = padding, ``T <= K``) resident.

    Keeps the resident ones, gives each missing one the least recently used slot the step does
    not need, and updates the residency state in place: ``step`` ``[1]``, ``slot_expert`` and
    ``slot_lastuse`` ``[K]``, ``expert_slot`` ``[E]``. Writes the page-in plan to ``src``/``dst``
    ``[K]`` (expert -> slot) and its length to ``count`` ``[1]``, and the runner's inputs:
    ``run_ids`` ``[T]`` int32, the slot holding each entry's expert, and ``run_w`` ``[T]``
    float32, its weight from ``topk_weights`` ``[T]`` float32; padding entries (id outside
    ``[0, E)``) get slot 0 and weight 0.
    """
    _jit_module().paged_experts_decide(
        topk,
        topk_weights,
        step,
        slot_expert,
        expert_slot,
        slot_lastuse,
        src,
        dst,
        count,
        run_ids,
        run_w,
    )


def paged_experts_gather(
    hosts: torch.Tensor,
    gpus: torch.Tensor,
    row_bytes: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Copy experts ``src[i]`` into slots ``dst[i]`` for ``i < count[0]``, for every paged tensor.

    ``hosts``/``gpus``/``row_bytes`` are int64 GPU tensors with one entry per paged tensor: the UVA
    device pointer of its pinned host store (``paged_experts_host_device_pointer``), the address of
    its GPU table, and its size per expert in bytes, even. Rows are copied in 16-byte loads when
    their size allows, else in 4- or 2-byte loads.
    """
    _jit_module().paged_experts_gather(hosts, gpus, row_bytes, src, dst, count)


def paged_experts_rows_to_slots(
    hosts: torch.Tensor,
    gpus: torch.Tensor,
    row_bytes: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    count: torch.Tensor,
) -> None:
    """Copy host row ``rows[i]`` into GPU slot ``slots[i]`` for ``i < count[0]``, for every
    paged tensor, one way: the row keeps its expert (inclusive host rows). Entries with a
    negative ``rows`` are skipped. Arguments as for ``paged_experts_gather``."""
    _jit_module().paged_experts_rows_to_slots(
        hosts, gpus, row_bytes, rows, slots, count
    )


def paged_experts_tiered_plan(
    src: torch.Tensor,
    dst: torch.Tensor,
    count: torch.Tensor,
    expert_row: torch.Tensor,
    row_expert: torch.Tensor,
    row_stamp: torch.Tensor,
    clock: torch.Tensor,
    doorbell: torch.Tensor,
    rows_out: torch.Tensor,
    publish_row_address: int = 0,
    publish_flag_address: int = 0,
    row_len: int = 0,
) -> None:
    """A device-planned step's page-in with the experts in host rows (the rest on disk), before
    its copies: each planned expert ``src[i]`` (``i < count[0]``) found in a host row
    (``expert_row`` ``[E]``) has that row stamped (``row_stamp`` ``[R]`` with the advanced
    ``clock`` ``[1]``) and written to ``rows_out`` ``[K]`` (-1: on disk or unused), for
    ``paged_experts_rows_to_slots(..., rows_out, dst, count)``; experts only on disk are
    recorded in ``doorbell`` ``[3, K]`` (missing, src, dst ``[K]``) for the host. The maps keep
    every expert where it is (inclusive rows; ``row_expert`` ``[R]`` completes the state row).
    With ``publish_row_address`` (pinned host memory, UVA) the same kernel then writes the
    layer's state row (``row_len`` int32 from ``doorbell``'s first entry: doorbell, maps, the
    clock last) there and the new clock to ``publish_flag_address``, after the row."""
    _jit_module().paged_experts_tiered_plan(
        src,
        dst,
        count,
        expert_row,
        row_expert,
        row_stamp,
        clock,
        doorbell,
        rows_out,
        int(publish_row_address),
        int(publish_flag_address),
        int(row_len),
    )


def paged_experts_await_rows(
    doorbell: torch.Tensor,
    maps: torch.Tensor,
    clock: torch.Tensor,
    reply_address: int,
    ack_address: int,
    plan: torch.Tensor,
    timeout_ns: int,
) -> None:
    """In a captured step, after ``paged_experts_tiered_plan``: if the layer left experts on disk
    (``doorbell`` ``[3, K]``), wait until the host's reply row (UVA address ``reply_address``:
    ticket, n, rows ``[K]``, slots ``[K]``, maps ``[E + 2R + 1]``) carries the ticket ``clock``
    ``[1]``, then write its plan to ``plan`` ``[1 + 2K]`` (n, rows, slots; for
    ``paged_experts_rows_to_slots``), its maps to ``maps``, clear the missing flags and write the
    ticket to ``ack_address``. Without a reply within ``timeout_ns`` the plan is empty and the
    flags stay."""
    _jit_module().paged_experts_await_rows(
        doorbell, maps, clock, reply_address, ack_address, plan, timeout_ns
    )


def paged_experts_host_device_pointer(pinned: torch.Tensor) -> int:
    """The address under which the GPU reads the pinned host tensor ``pinned``."""
    return int(_jit_module().paged_experts_host_device_pointer(pinned))
