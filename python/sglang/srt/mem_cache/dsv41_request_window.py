from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode


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

    def copy_(self, other: "WindowLayout") -> None:
        # Graph replay refreshes a captured layout in place: the captured copy
        # kernels read these tensors by address, so their contents move, not
        # the object.
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
):
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
    """Move tokens between two paged buffers of ``layout``: a token is a data row
    and a scale row (576 + 8 bytes for V4, 512 + 16 for V41, 256 + 32 for V41_FP4)."""
    if not src_loc.numel():
        return
    src_loc, dst_loc = src_loc.long(), dst_loc.long()
    for width, base in (
        (layout.data_bytes, 0),
        (layout.scale_bytes, page_size * layout.data_bytes),
    ):
        cols = torch.arange(width, device=src.device)
        values = src[
            src_loc[:, None] // page_size,
            base + (src_loc[:, None] % page_size) * width + cols,
        ]
        dst[
            dst_loc[:, None] // page_size,
            base + (dst_loc[:, None] % page_size) * width + cols,
        ] = values


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

    def reset(self, slots):
        loc = slots.to(torch.int64)[:, None] * self.capacity + torch.arange(
            self.capacity, device=slots.device
        )
        self.tags[:, loc.flatten()] = -1
        self.prepared = None

    def activate(self, layout):
        if self.layout is layout:
            return
        self.layout = layout
        self.prepared = None
        if self.workspace is None:
            self._ensure_workspace(layout.size)
        elif self.workspace.size < layout.size:
            # Captured graphs hold the workspace address; growing it here would
            # leave them writing into a freed buffer. Size it at construction.
            raise RuntimeError(
                f"request-window workspace too small: {self.workspace.size} rows "
                f"for a layout of {layout.size}"
            )

    def initialize_dummy_history(self):
        layout = self.layout
        self.tags.fill_(-1)
        loc = layout.history_req * self.capacity + layout.history_pos % self.capacity
        for buf in self.state.kv_buffer:
            buf.zero_()
        self.tags[:, loc] = layout.history_pos
        self.prepared = None

    def _history_src(self, layout):
        return torch.where(
            layout.history_valid,
            layout.history_req * self.capacity + layout.history_pos % self.capacity,
            self.zero_row,
        )

    def buffer(self, layer):
        # The runner's capture scope includes eager warmups, before CUDA capture
        # starts. Include the phase in the key so leaving that scope revalidates
        # ownership even when the layout and layer have not changed.
        in_capture = get_is_capture_mode() or _capturing()
        prepared_key = (layer, in_capture)
        if self.prepared != prepared_key:
            layout = self.layout
            if layout is None:
                raise RuntimeError("request-window metadata was not activated")
            src = self._history_src(layout)
            if not in_capture:
                valid = layout.history_valid
                if not torch.equal(
                    self.tags[layer, src][valid], layout.history_pos[valid]
                ):
                    raise RuntimeError(
                        "SWA history is missing: replay or window ownership is invalid"
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
        dst = torch.where(
            layout.commit_mask,
            layout.req * self.capacity + layout.pos % self.capacity,
            self.sink_row,
        )
        copy_packed_tokens(
            self.buffer(layer),
            self.state.kv_buffer[layer],
            layout.write_loc,
            dst,
            page_size=self.page_size,
            layout=self.state.kv_layout,
        )
        self.tags[layer, dst] = layout.pos
