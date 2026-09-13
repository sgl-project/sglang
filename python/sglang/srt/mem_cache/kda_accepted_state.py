"""Request-owned KDA checkpoints with explicit committed-state materialization.

Scratch remains sized by active requests. Pending metadata maps a physical
state slot to its request scratch row and selected post-token checkpoint.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_selected(
    Dst,
    Src,
    Slots,
    Rows,
    Steps,
    N: tl.constexpr,
    DL: tl.constexpr,
    DS: tl.constexpr,
    SL: tl.constexpr,
    SR: tl.constexpr,
    ST: tl.constexpr,
    WIDTH: tl.constexpr,
    DST_SIZE: tl.constexpr,
    SRC_SIZE: tl.constexpr,
    STEP_SIZE: tl.constexpr,
    POOL_INDEXED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, block = tl.program_id(0), tl.program_id(2)
    # All-layer pools exceed 2^31 elements at normal GLM request counts.
    layer = tl.program_id(1).to(tl.int64)
    slot = tl.load(Slots + request).to(tl.int64)
    if slot < 0 or slot >= DST_SIZE:
        return
    meta = slot if POOL_INDEXED else request
    step = tl.load(Steps + meta).to(tl.int64)
    row = tl.load(Rows + meta).to(tl.int64)
    if step < 0 or step >= STEP_SIZE or row < 0 or row >= SRC_SIZE:
        return
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        Src + layer * SL + row * SR + step * ST + offsets, offsets < WIDTH, 0
    )
    tl.store(Dst + layer * DL + slot * DS + offsets, value, offsets < WIDTH)


@triton.jit
def _record(
    Slots,
    Rows,
    Steps,
    PendingRows,
    PendingSteps,
    N: tl.constexpr,
    SIZE: tl.constexpr,
    INVALIDATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + x, x < N, -1).to(tl.int64)
    valid = (x < N) & (slot >= 0) & (slot < SIZE)
    if INVALIDATE:
        tl.store(PendingSteps + slot, -1, valid)
    else:
        row = tl.load(Rows + x, x < N, 0)
        step = tl.load(Steps + x, x < N, -1)
        tl.store(PendingRows + slot, row, valid)
        tl.store(PendingSteps + slot, step, valid)


class KDAAcceptedState:
    """Own pending state identities; never resize or replace the scratch pool.

    Callers must materialize before reading ``temporal`` or recycling a live
    request's scratch row, and invalidate after replacing a physical state.
    All copies and metadata updates run in stream order and are graph-safe.
    """

    def __init__(self, temporal: torch.Tensor, scratch: torch.Tensor):
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            _require_entry_contiguous_dst,
        )

        if temporal.ndim != 5 or scratch.ndim != 6:
            raise ValueError("expected [layers, slots, H, V, K] and per-step scratch")
        if (
            temporal.shape[0] != scratch.shape[0]
            or temporal.shape[2:] != scratch.shape[3:]
        ):
            raise ValueError("committed and scratch state shapes must match")
        if temporal.dtype != torch.float32 or scratch.dtype != torch.float32:
            raise ValueError(
                "accepted KDA state requires FP32 committed and scratch state"
            )
        if not temporal.is_cuda or temporal.device != scratch.device:
            raise ValueError("state tensors must be on the same CUDA device")
        _require_entry_contiguous_dst(temporal, 2, "KDAAcceptedState")
        _require_entry_contiguous_dst(scratch, 3, "KDAAcceptedState")
        self.temporal, self.scratch = temporal, scratch
        self.rows = torch.zeros(
            temporal.shape[1], dtype=torch.int32, device=temporal.device
        )
        self.steps = torch.full_like(self.rows, -1)
        self.all_slots = torch.arange(
            temporal.shape[1], dtype=torch.int32, device=temporal.device
        )

    def _copy(self, slots, rows, steps, *, pool_indexed):
        if slots.numel() == 0:
            return
        width = self.temporal[0, 0].numel()
        _copy_selected[
            (slots.numel(), self.temporal.shape[0], triton.cdiv(width, 1024))
        ](
            self.temporal,
            self.scratch,
            slots,
            rows,
            steps,
            slots.numel(),
            self.temporal.stride(0),
            self.temporal.stride(1),
            self.scratch.stride(0),
            self.scratch.stride(1),
            self.scratch.stride(2),
            width,
            self.temporal.shape[1],
            self.scratch.shape[1],
            self.scratch.shape[2],
            pool_indexed,
            1024,
        )

    def record(self, slots, rows, steps):
        if slots.numel() == 0:
            return
        _record[(triton.cdiv(slots.numel(), 256),)](
            slots,
            rows,
            steps,
            self.rows,
            self.steps,
            slots.numel(),
            self.steps.numel(),
            False,
            256,
        )

    def invalidate(self, slots=None):
        slots = self.all_slots if slots is None else slots
        if slots.numel() == 0:
            return
        _record[(triton.cdiv(slots.numel(), 256),)](
            slots,
            None,
            None,
            self.rows,
            self.steps,
            slots.numel(),
            self.steps.numel(),
            True,
            256,
        )

    def materialize(self, slots=None):
        slots = self.all_slots if slots is None else slots
        self._copy(slots, self.rows, self.steps, pool_indexed=True)
        # Separate launch: every state-copy CTA must finish reading metadata
        # before it is invalidated. Resetting inside the copy grid races.
        self.invalidate(slots)

    def track(self, destinations, request_rows, steps):
        self._copy(destinations, request_rows, steps, pool_indexed=False)
        self.invalidate(destinations)
