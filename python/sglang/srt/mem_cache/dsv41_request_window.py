from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.dsv4_request_window_copy import (
    commit_packed_rows,
    copy_packed_rows,
)
from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.utils.invariants import Bucket, Invariant, IsTrue, expect, resolve_level

# A window row the replay expects but no longer holds; that request's attention
# would read the wrong rows.
_SWA_HISTORY_PRESENT = Invariant(
    "dsv41.swa_history_present", Bucket.FATAL_CONTAINABLE, IsTrue()
)


class WindowLayout(msgspec.Struct, frozen=True):
    req: torch.Tensor
    pos: torch.Tensor
    write_loc: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    history_req: torch.Tensor
    history_pos: torch.Tensor
    history_loc: torch.Tensor
    history_valid: torch.Tensor
    commit_mask: torch.Tensor
    size: int

    def leading_rows(self, num_rows: int) -> "WindowLayout":
        """The layout of the live queries; a prefill CUDA graph pads the rest."""
        if num_rows == self.pos.numel():
            return self
        return msgspec.structs.replace(
            self,
            req=self.req[:num_rows],
            pos=self.pos[:num_rows],
            write_loc=self.write_loc[:num_rows],
            indices=self.indices[:num_rows],
            lengths=self.lengths[:num_rows],
            commit_mask=self.commit_mask[:num_rows],
        )

    def copy_(self, other: "WindowLayout") -> None:
        # Captured copy kernels read these tensors by address, so a graph replay
        # must refresh their contents in place, not rebind the object.
        assert self.size == other.size, (self.size, other.size)
        self.req.copy_(other.req)
        self.pos.copy_(other.pos)
        self.write_loc.copy_(other.write_loc)
        self.indices.copy_(other.indices)
        self.lengths.copy_(other.lengths)
        self.history_req.copy_(other.history_req)
        self.history_pos.copy_(other.history_pos)
        self.history_loc.copy_(other.history_loc)
        self.history_valid.copy_(other.history_valid)
        self.commit_mask.copy_(other.commit_mask)


def _first_row_offsets(
    req: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = req.numel()
    offset = torch.arange(n, device=req.device)
    starts = torch.ones(n, dtype=torch.bool, device=req.device)
    starts[1:] = req[1:] != req[:-1]
    group = starts.cumsum(0) - 1
    group_first = torch.cummax(torch.where(starts, offset, 0), dim=0).values
    ends = torch.ones(n, dtype=torch.bool, device=req.device)
    ends[:-1] = starts[1:]
    group_last = torch.cummin(
        torch.where(ends, offset, n - 1).flip(0), dim=0
    ).values.flip(0)
    return group, group_first, group_last


def window_layout(
    req,
    pos,
    *,
    window: int = 128,
    capacity: int = 256,
    floor: Optional[torch.Tensor] = None,
    num_groups: Optional[int] = None,
    padded_rows: Optional[int] = None,
):
    """padded_rows is set for the prefill CUDA graph, which pads the layout to that
    many query rows past the live ones in req and pos."""
    n = pos.numel()
    if n == 0:
        raise ValueError("request-window layout needs at least one query")
    req = req.to(torch.int64)
    pos = pos.to(torch.int64)
    device = pos.device
    groups = n if num_groups is None else int(num_groups)
    offset = torch.arange(n, device=device)
    group, group_first, group_last = _first_row_offsets(req)
    first_pos = pos - (offset - group_first)
    history_rows = groups * window

    write_loc = (history_rows + offset).to(torch.int32)
    lookback = torch.arange(window, device=device)
    seen_pos = pos[:, None] - lookback
    old = seen_pos < first_pos[:, None]
    old_loc = group[:, None] * window + (seen_pos - (first_pos[:, None] - window))
    new_loc = history_rows + group_first[:, None] + seen_pos - first_pos[:, None]
    valid = seen_pos >= 0
    if floor is not None:
        floor = floor.to(torch.int64)
        valid &= seen_pos >= floor[:, None]
    indices = torch.where(valid, torch.where(old, old_loc, new_loc), -1).to(torch.int32)
    lengths = valid.sum(-1).to(torch.int32)

    g_req = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
        0, group, req
    )
    g_first = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
        0, group, first_pos
    )
    g_live = torch.zeros(groups, dtype=torch.bool, device=device).scatter_(
        0, group, torch.ones_like(group, dtype=torch.bool)
    )
    history_pos = (g_first[:, None] - window + lookback[None, :]).flatten()
    history_valid = (history_pos >= 0) & g_live.repeat_interleave(window)
    if floor is not None:
        g_floor = torch.zeros(groups, dtype=torch.int64, device=device).scatter_(
            0, group, floor
        )
        history_valid &= history_pos >= g_floor.repeat_interleave(window)
    history_req = g_req.repeat_interleave(window)
    history_loc = torch.arange(history_rows, device=device)

    commit_mask = (group_last - offset) < capacity
    if padded_rows is not None and padded_rows > n:
        # Padding rows write the workspace past the live rows and commit nowhere.
        padding = padded_rows - n
        zeros = req.new_zeros(padding)
        req, pos = torch.cat([req, zeros]), torch.cat([pos, zeros])
        write_loc = torch.cat(
            [
                write_loc,
                torch.arange(
                    history_rows + n,
                    history_rows + padded_rows,
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        )
        indices = torch.cat([indices, indices.new_full((padding, window), -1)])
        lengths = torch.cat([lengths, lengths.new_zeros(padding)])
        commit_mask = torch.cat([commit_mask, commit_mask.new_zeros(padding)])
        n = padded_rows
    return WindowLayout(
        req,
        pos,
        write_loc,
        indices,
        lengths,
        history_req,
        history_pos,
        history_loc,
        history_valid,
        commit_mask,
        history_rows + n,
    )


def copy_packed_tokens(src, dst, src_loc, dst_loc, *, page_size, layout=KVLayout.V4):
    """Move tokens between paged buffers of ``layout``: a data row and a scale row."""
    if not src_loc.numel():
        return
    copy_packed_rows(src, dst, src_loc, dst_loc, page_size=page_size, layout=layout)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class RequestWindow:
    def __init__(
        self,
        pool_factory,
        *,
        num_slots,
        layers,
        page_size,
        capacity,
        workspace_rows: Optional[int] = None,
    ):
        self.capacity = ((capacity + page_size - 1) // page_size) * page_size
        self.page_size = page_size
        self.pool_factory = pool_factory
        self.num_slots = num_slots
        self.rows = num_slots * self.capacity

        self.state = pool_factory(self.rows + page_size, layers)
        self.zero_row = self.rows
        self.sink_row = self.rows + 1
        self.tags = torch.full(
            (layers, self.rows + page_size),
            -1,
            dtype=torch.int64,
            device=self.state.kv_buffer[0].device,
        )
        self.workspace = None
        if workspace_rows:
            self._ensure_workspace(workspace_rows)
        self.layout = None
        self.prepared = None

    def _ensure_workspace(self, rows: int) -> None:
        if self.workspace is not None and self.workspace.size >= rows:
            return
        assert not _capturing(), "request-window workspace must be sized before capture"
        size = ((rows + self.page_size - 1) // self.page_size) * self.page_size
        self.workspace = self.pool_factory(size, 1)

    def row_loc(self, slots, positions):
        return slots * self.capacity + positions % self.capacity

    def page_indices(self, slot: int, start: int, end: int):
        assert start % self.page_size == 0, (start, self.page_size)
        positions = torch.arange(start, end, self.page_size)
        # capacity is whole pages, so a page's rows keep a paged pool's offsets
        # and pair with decode's paged SWA pages one for one.
        return (self.row_loc(slot, positions) // self.page_size).to(torch.int32).numpy()

    def reset(self, slots):
        loc = slots.to(torch.int64)[:, None] * self.capacity + torch.arange(
            self.capacity, device=slots.device
        )
        # index_fill_ stays on the device; a scalar index_put waits for the stream.
        self.tags.index_fill_(1, loc.flatten(), -1)
        self.prepared = None

    def activate(self, layout):
        if self.layout is layout:
            return
        self.layout = layout
        self.prepared = None
        if self.workspace is None:
            self._ensure_workspace(layout.size)
        elif self.workspace.size < layout.size:
            # Captured graphs hold the workspace address; growing it strands them.
            raise RuntimeError(
                f"request-window workspace too small: {self.workspace.size} rows "
                f"for a layout of {layout.size}"
            )

    def initialize_dummy_history(self):
        layout = self.layout
        self.tags.fill_(-1)
        loc = self.row_loc(layout.history_req, layout.history_pos)
        for buf in self.state.kv_buffer:
            buf.zero_()
        self.tags[:, loc] = layout.history_pos
        self.prepared = None

    def _history_src(self, layout):
        return torch.where(
            layout.history_valid,
            self.row_loc(layout.history_req, layout.history_pos),
            self.zero_row,
        )

    def buffer(self, layer):
        # The runner's capture scope includes eager warmups before CUDA capture
        # starts, so the phase is part of the key: leaving the scope revalidates.
        in_capture = get_is_capture_mode() or _capturing()
        prepared_key = (layer, in_capture)
        if self.prepared != prepared_key:
            layout = self.layout
            if layout is None:
                raise RuntimeError("request-window metadata was not activated")
            src = self._history_src(layout)
            # Signal-only: skip building the predicate when detection is off.
            if not in_capture and resolve_level() >= InvariantCheckLevel.WARN:
                expect(
                    _SWA_HISTORY_PRESENT,
                    (self.tags[layer, src] == layout.history_pos)
                    | ~layout.history_valid,
                    msg="replay or window ownership is invalid",
                )
            copy_packed_tokens(
                self.state.kv_buffer[layer],
                self.workspace.kv_buffer[0],
                src,
                layout.history_loc,
                page_size=self.page_size,
                layout=self.state.kv_layout,
            )
            self.prepared = prepared_key
        return self.workspace.kv_buffer[0]

    def commit(self, layer):
        layout = self.layout
        commit_packed_rows(
            self.buffer(layer),
            self.state.kv_buffer[layer],
            self.tags[layer],
            layout,
            page_size=self.page_size,
            capacity=self.capacity,
            layout=self.state.kv_layout,
        )
