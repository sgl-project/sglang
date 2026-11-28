"""Row-style request window for encoder SWA bounded replay on NPU.

The CUDA request window stores packed pages and copies/commits them with Triton
kernels. NPU rows are plain ``[dim]`` BF16 (A3) or quantized (A5) rows, so this
keeps the same window semantics but moves rows with torch gather/scatter:

- ``state[layer]``: ``num_slots * capacity`` request rows plus a zero row and a
  sink row;
- ``workspace``: shared staging rows; eager layouts pack replayable history
  rows first, graph layouts own one page-aligned segment per request;
- ``tags[layer]``: the position each request row holds, used to validate a
  history row before it is copied.

The attention backend consumes ``workspace`` as the layer's SWA buffer and
writes the extend's rows there; ``commit`` stores them back into the request
ring after attention.
"""

from typing import Optional

import msgspec
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.hardware_backend.npu.dsv4.ring_layout import (
    RingPageTableCache,
    reset_tag_rows,
    ring_page_indices,
    ring_row_loc,
)
from sglang.srt.mem_cache.dsv41_request_window import (
    _SWA_HISTORY_PRESENT,
    WindowLayout,
    window_layout,
)
from sglang.srt.utils.common import ceil_align
from sglang.srt.utils.invariants import expect, resolve_level
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class NPURequestWindowSpec(msgspec.Struct, frozen=True, kw_only=True):
    num_slots: int
    layers: int
    page_size: int
    capacity: int
    workspace_rows: int
    row_dim: int
    row_dtype: torch.dtype

    def __post_init__(self):
        if (
            min(
                self.num_slots,
                self.layers,
                self.page_size,
                self.capacity,
                self.workspace_rows,
                self.row_dim,
            )
            < 1
        ):
            raise ValueError("request-window extents must be positive")
        if self.capacity % self.page_size or self.workspace_rows % self.page_size:
            raise ValueError("request-window extents must contain whole pages")

    @property
    def tag_rows(self) -> int:
        return self.num_slots * self.capacity + self.page_size

    @property
    def state_rows(self) -> int:
        # Page-aligned so decode can view the ring as a paged PA_ND buffer; the
        # zero and sink rows sit right after the request rows.
        return self.tag_rows

    @property
    def nbytes(self) -> int:
        row_bytes = torch.empty(0, dtype=self.row_dtype).element_size() * self.row_dim
        return (
            self.layers * self.state_rows + self.workspace_rows
        ) * row_bytes + self.layers * self.tag_rows * 8

    @classmethod
    def create(
        cls,
        *,
        num_slots: int,
        layers: int,
        page_size: int,
        capacity: int,
        workspace_rows: int,
        row_dim: int,
        row_dtype: torch.dtype,
    ):
        return cls(
            num_slots=num_slots,
            layers=layers,
            page_size=page_size,
            capacity=ceil_align(capacity, page_size),
            workspace_rows=ceil_align(workspace_rows, page_size),
            row_dim=row_dim,
            row_dtype=row_dtype,
        )


class NPUWindowLayout(msgspec.Struct, frozen=True):
    """One staged window: per-request history rows followed by live rows.

    A3's sparse-attention operator drops explicit indices >= ``seqused_kv``, so
    the fixed 128-row history region of the CUDA layout cannot be used: eager
    prefill sizes the history to the replayable rows, and graph layouts fix one
    page-aligned segment per request. ``kv_lens`` is each request's staged row
    span; the window side feeds it as ``seqused_ori_kv``.
    """

    indices: torch.Tensor
    lengths: torch.Tensor
    kv_lens: torch.Tensor
    write_loc: torch.Tensor
    commit_mask: torch.Tensor
    history_src: torch.Tensor
    history_loc: torch.Tensor
    history_pos: torch.Tensor
    history_valid: torch.Tensor
    size: int
    req: torch.Tensor
    positions: torch.Tensor
    # Precomputed commit addresses: workspace sources, ring destinations and
    # tag positions, fixed shape for a replay and filtered once when eager.
    commit_src: torch.Tensor
    commit_dst: torch.Tensor
    commit_tags: torch.Tensor
    # Per-request page-aligned workspace segment: row base and segment pages.
    base: torch.Tensor
    group_pages: torch.Tensor
    # Widest segment in pages; the page table never needs more columns.
    group_pages_max: int


def _cpu_max(values, rows: int) -> int:
    """Largest of the first ``rows`` host values (``None`` and empty are 0)."""
    if values is None:
        return 0
    values = values[:rows]
    if torch.is_tensor(values):
        return int(values.max()) if values.numel() else 0
    values = list(values)
    return max(values) if values else 0


def group_first_offsets(starts: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Per-row index of its request run's first row."""
    return torch.cummax(
        torch.where(starts, offset, torch.zeros_like(offset)), dim=0
    ).values


def _group_ends(
    starts: torch.Tensor,
    group: torch.Tensor,
    offset: torch.Tensor,
    first_pos: torch.Tensor,
    floor: torch.Tensor,
    req: torch.Tensor,
    groups: int,
):
    """Per-group first position, floor, request, row count and last row.

    Derived from the request-run boundaries. A duplicate-index ``scatter_``
    leaves the winner implementation-defined, which would make the commit mask
    depend on it; the run starts are unique, so ``index_copy_`` is exact.
    Groups without rows (a request with no token in this batch) stay zero.
    """
    n = offset.numel()
    start_idx = torch.nonzero(starts, as_tuple=False).flatten()
    gid = group[start_idx]
    g_first = offset.new_zeros(groups)
    g_floor = offset.new_zeros(groups)
    g_req = req.new_zeros(groups)
    g_count = offset.new_zeros(groups)
    g_last = offset.new_zeros(groups)
    g_first.index_copy_(0, gid, first_pos[start_idx])
    g_floor.index_copy_(0, gid, floor[start_idx])
    g_req.index_copy_(0, gid, req[start_idx])
    # Run ``i`` spans rows [start_i, start_{i+1}); its last row is one before
    # the next run's start (or the batch end), so no per-row scan is needed.
    next_start = torch.cat([start_idx[1:], offset.new_full((1,), n)])
    g_count.index_copy_(0, gid, next_start - start_idx)
    g_last.index_copy_(0, gid, next_start - 1)
    return g_first, g_floor, g_req, g_count, g_last


def _history_rows(
    *,
    total: int,
    hist_start: torch.Tensor,
    hist_len: torch.Tensor,
    group_req: torch.Tensor,
    base: torch.Tensor,
    row_loc,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack every request's replayable history rows into flat gather lists.

    ``hist_len`` is a device tensor, so the group of each flat row comes from a
    device-side search over its prefix sum: one host read of the caller's total
    replaces one device-to-host scalar read per request.
    """
    if total == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty, torch.empty(0, dtype=torch.bool, device=device)
    group_end = hist_len.cumsum(0)
    flat = torch.arange(total, dtype=torch.int64, device=device)
    group = torch.searchsorted(group_end, flat, right=True)
    offset = flat - (group_end - hist_len)[group]
    positions = hist_start[group] + offset
    return (
        row_loc(group_req[group], positions),
        base[group] + offset,
        positions,
        torch.ones(total, dtype=torch.bool, device=device),
    )


def _row_view(t: torch.Tensor) -> torch.Tensor:
    """Row gather/scatter view of a request-window tensor.

    A5 keeps fp8 rows, and the NPU ``index_select`` / ``index_copy_`` kernels
    reject ``DT_FLOAT8_E4M3FN``.  A byte view moves the same bytes and supports
    both ops; wider rows are returned unchanged.
    """
    return t.view(torch.uint8) if t.element_size() == 1 else t


def npu_workspace_rows(shared_rows: int, *, num_slots: int, page_size: int) -> int:
    """Shared workspace rows plus one alignment page per request slot.

    Each request stages into its own page-aligned segment; rounding every
    segment up adds at most ``page_size - 1`` rows per request.
    """
    return int(shared_rows) + int(num_slots) * int(page_size)


def dsv41_request_window_spec(
    *,
    num_slots: int,
    layers: int,
    page_size: int,
    capacity: int,
    workspace_rows: int,
    row_dim: int,
    row_dtype: torch.dtype,
) -> NPURequestWindowSpec:
    """The one spec both the pool allocation and the memory budget derive from."""
    return NPURequestWindowSpec.create(
        num_slots=num_slots,
        layers=layers,
        page_size=page_size,
        capacity=capacity,
        workspace_rows=npu_workspace_rows(
            workspace_rows, num_slots=num_slots, page_size=page_size
        ),
        row_dim=row_dim,
        row_dtype=row_dtype,
    )


class NPURequestWindowMemory:
    def __init__(
        self,
        spec: NPURequestWindowSpec,
        *,
        device: torch.device,
        enable_memory_saver: bool = False,
    ):
        self.spec = spec
        # The window outlives every forward, like the KV pools: put it in the
        # same memory-saver region so pause/resume covers its allocation.
        adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        with adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.state = torch.empty(
                (spec.layers, spec.state_rows, spec.row_dim),
                dtype=spec.row_dtype,
                device=device,
            )
            # Invalid history reads the zero row; zero it through a byte view
            # because A5 fp8 rows do not implement fill_.
            self.state.view(torch.uint8)[:, spec.num_slots * spec.capacity :].zero_()
            self.tags = torch.full(
                (spec.layers, spec.tag_rows), -1, dtype=torch.int64, device=device
            )
            self.workspace = torch.empty(
                (spec.workspace_rows, spec.row_dim),
                dtype=spec.row_dtype,
                device=device,
            )
            # The operator may scan pages past staged rows; zero workspace once,
            # via a byte view because A5 fp8 rows do not implement fill_.
            self.workspace.view(torch.uint8).zero_()
        self.view = NPURequestWindow(
            spec, state=self.state, tags=self.tags, workspace=self.workspace
        )

    def reset_rows(self, rows) -> None:
        indices = torch.as_tensor(rows, dtype=torch.int64, device=self.view.tags.device)
        self.view.reset(indices)


class NPURequestWindow:
    def __init__(self, spec, *, state, tags, workspace):
        self.spec = spec
        self.capacity = spec.capacity
        self.page_size = spec.page_size
        self.num_slots = spec.num_slots
        self.rows = spec.num_slots * spec.capacity
        self.zero_row = self.rows
        self.sink_row = self.rows + 1
        self.state = state
        self.tags = tags
        self.workspace = workspace
        self.layout: Optional[WindowLayout | NPUWindowLayout] = None
        self.prepared = None
        self._idle_layout = None
        # Captured graphs cannot branch on the commit mask; eager commits do.
        self.graph_capture = False
        # Prefill stages history in the workspace; decode reads and writes the
        # persistent request ring directly.
        self.prefill = True

    def set_phase(self, *, prefill: bool) -> None:
        if self.prefill != prefill:
            self.prefill = prefill
            self.prepared = None

    def state_view(self, layer: int) -> torch.Tensor:
        """The layer's request ring as a paged PA_ND buffer."""
        page_size = self.page_size
        return self.state[layer].view(-1, page_size, 1, self.state.shape[-1])

    def row_loc(self, slots, positions):
        return ring_row_loc(slots, positions, self.capacity)

    def page_indices(self, slot: int, start: int, end: int):
        """Physical page ids of ``slot``'s ring rows in [start, end)."""
        return ring_page_indices(self.capacity, self.page_size, slot, start, end)

    def activate(self, layout: WindowLayout | NPUWindowLayout) -> None:
        # Every layout stages through the workspace; direct-ring decode does not
        # create one. Bind the phase even when the layout object is unchanged.
        self.set_phase(prefill=True)
        if self.layout is layout:
            return
        if self.workspace.shape[0] < layout.size:
            raise RuntimeError(
                f"request-window workspace too small: {self.workspace.shape[0]} rows "
                f"for a layout of {layout.size}"
            )
        self.layout = layout
        self.prepared = None

    def build_prefill_layout(self, req, positions, *, num_groups, window, floor=None):
        capacity = self.capacity
        page = self.page_size
        # The ring stores ``capacity`` positions (window + draft rows), but the
        # attention window stays ``window`` wide; conflating them over-attends.
        window = int(window)
        device = positions.device
        n = positions.numel()
        req = req.to(torch.int64)
        pos = positions.to(torch.int64)
        offset = torch.arange(n, dtype=torch.int64, device=device)
        starts = torch.ones(n, dtype=torch.bool, device=device)
        starts[1:] = req[1:] != req[:-1]
        group = starts.cumsum(0) - 1
        first_pos = pos - (offset - group_first_offsets(starts, offset))
        floor = (
            torch.zeros(n, dtype=torch.int64, device=device)
            if floor is None
            else floor.reshape(-1).to(torch.int64)
        )
        groups = int(num_groups)
        g_first, g_floor, g_req, g_count, group_last = _group_ends(
            starts, group, offset, first_pos, floor, req, groups
        )
        # History is [max(0, floor, first-window), first); a fresh or folded
        # request (floor == first) has none.
        hist_start = torch.clamp(g_first - window, min=0)
        hist_start = torch.maximum(hist_start, torch.clamp(g_floor, min=0))
        hist_len = torch.clamp(g_first - hist_start, min=0, max=window)
        spans = hist_len + g_count
        # Each request owns a page-aligned workspace segment, addressed by a
        # per-request page table; no two requests share a page.
        seg = (spans + page - 1) // page * page
        base = torch.cumsum(seg, 0) - seg
        # One device-to-host read for the three layout-wide extents: a separate
        # int() per scalar would stall the stream once each.
        size, history_rows, seg_max = torch.stack(
            (seg.sum(), hist_len.sum(), seg.max())
        ).tolist()

        history_src, history_loc, history_pos, history_valid = _history_rows(
            total=history_rows,
            hist_start=hist_start,
            hist_len=hist_len,
            group_req=g_req,
            base=base,
            row_loc=self.row_loc,
            device=device,
        )

        base_row = base[group]
        hist_row = hist_len[group]
        hist_lo = hist_start[group]
        write_loc = (base_row + hist_row + (pos - first_pos)).to(torch.int32)
        offsets = torch.arange(window, dtype=torch.int64, device=device)
        # The valid window is the suffix [max(hist_lo, pos-window+1), pos]; the
        # left-aligned indices are that range shifted to lane 0, so no sort.
        win_start = torch.maximum(hist_lo, torch.clamp(pos - (window - 1), min=0))
        lengths = torch.clamp(pos - win_start + 1, min=0, max=window)
        seen = win_start.unsqueeze(1) + offsets.unsqueeze(0)
        # Attention indices are frame positions: the per-request page table adds
        # the request's workspace base, so indices stay below ``seqused``.
        old_rows = seen - hist_lo.unsqueeze(1)
        new_rows = hist_row.unsqueeze(1) + (seen - first_pos.unsqueeze(1))
        rows = torch.where(seen < first_pos.unsqueeze(1), old_rows, new_rows)
        # A5 reads indices[:length], so valid frame rows are a left-aligned
        # prefix; their values remain frame offsets for the page table.
        indices = torch.where(
            offsets.unsqueeze(0) < lengths.unsqueeze(1),
            rows,
            rows.new_full((), -1),
        ).to(torch.int32)
        lengths = lengths.to(torch.int32)
        commit_mask = (group_last[group] - offset) < capacity
        # Eager commits compact to the masked rows once per layout instead of
        # repeating the mask filter and ring modulo in every layer.
        commit_src = write_loc[commit_mask].to(torch.int64)
        commit_dst = self.row_loc(req[commit_mask], pos[commit_mask])
        commit_tags = pos[commit_mask].to(torch.int64)
        return NPUWindowLayout(
            indices=indices,
            lengths=lengths,
            kv_lens=spans.to(torch.int32),
            write_loc=write_loc,
            commit_mask=commit_mask,
            history_src=history_src,
            history_loc=history_loc,
            history_pos=history_pos,
            history_valid=history_valid,
            size=size,
            req=req,
            positions=pos,
            commit_src=commit_src,
            commit_dst=commit_dst,
            commit_tags=commit_tags,
            base=base,
            group_pages=(seg // page).to(torch.int64),
            group_pages_max=seg_max // page,
        )

    def _graph_window(self, positions, *, num_draft: int, window: int):
        """Staged window rows and lengths of a query block.

        Each query's window starts at ``max(0, pos - window + 1)``; it lies at
        or after the staged context start by construction. Decode is the
        ``num_draft == 1`` case.
        """
        device = positions.device
        rows = positions.numel()
        first = positions[::num_draft]
        group = torch.arange(rows, device=device) // num_draft
        step = torch.arange(rows, device=device) % num_draft
        ctx_start = torch.clamp(first - (window - 1), min=0)
        ctx_len = first - ctx_start
        offsets = torch.arange(window, dtype=torch.int64, device=device)
        local_lo = torch.clamp(positions - (window - 1), min=0) - ctx_start[group]
        row_idx = local_lo.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < (ctx_len[group] + step + 1).unsqueeze(1)
        indices = torch.where(valid, row_idx, row_idx.new_full((), -1)).to(torch.int32)
        lengths = valid.sum(-1, dtype=torch.int32)
        return ctx_start, ctx_len, step, group, indices, lengths

    def _graph_scratch_row(self) -> int:
        """The workspace row graph layouts scatter their padding into."""
        return self.workspace.shape[0] - 1

    def build_graph_layout(
        self, batch: int, num_draft: int, *, window: int
    ) -> NPUWindowLayout:
        """Fixed-shape graph layout: context + block per request.

        Decode is the ``num_draft == 1`` case. Every tensor is graph-owned;
        ``refresh_graph_layout`` rewrites its contents before each replay.
        """
        device = self.tags.device
        page = self.page_size
        span = window - 1 + num_draft
        stride = (span + page - 1) // page * page
        rows = batch * num_draft
        hist = batch * (window - 1)
        if batch * stride > self._graph_scratch_row():
            raise RuntimeError(
                f"request-window workspace too small for a graph layout: "
                f"{batch * stride} rows for {self.workspace.shape[0]}"
            )
        scratch = self._graph_scratch_row()
        base = torch.arange(batch, dtype=torch.int64, device=device) * stride
        return NPUWindowLayout(
            indices=torch.full((rows, window), -1, dtype=torch.int32, device=device),
            lengths=torch.zeros(rows, dtype=torch.int32, device=device),
            kv_lens=torch.zeros(batch, dtype=torch.int32, device=device),
            write_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            commit_mask=torch.zeros(rows, dtype=torch.bool, device=device),
            history_src=torch.full(
                (hist,), self.zero_row, dtype=torch.int64, device=device
            ),
            history_loc=torch.full((hist,), scratch, dtype=torch.int64, device=device),
            history_pos=torch.zeros(hist, dtype=torch.int64, device=device),
            history_valid=torch.zeros(hist, dtype=torch.bool, device=device),
            size=batch * stride,
            req=torch.zeros(rows, dtype=torch.int64, device=device),
            positions=torch.zeros(rows, dtype=torch.int64, device=device),
            commit_src=torch.zeros(rows, dtype=torch.int64, device=device),
            commit_dst=torch.zeros(rows, dtype=torch.int64, device=device),
            commit_tags=torch.zeros(rows, dtype=torch.int64, device=device),
            base=base,
            group_pages=torch.full(
                (batch,), stride // page, dtype=torch.int64, device=device
            ),
            group_pages_max=stride // page,
        )

    @torch.no_grad()
    def refresh_graph_layout(
        self,
        layout: NPUWindowLayout,
        req,
        positions,
        live,
        *,
        num_draft: int,
        window: int,
    ) -> None:
        device = layout.indices.device
        req = req.to(device=device, dtype=torch.int64).reshape(-1)
        positions = positions.to(device=device, dtype=torch.int64).reshape(-1)
        rows = positions.numel()
        groups = rows // num_draft
        ctx_start, ctx_len, step, group, indices, lengths = self._graph_window(
            positions, num_draft=num_draft, window=window
        )
        g_base = layout.base.to(torch.int64)
        live_rows = live.to(device=device, dtype=torch.bool).reshape(-1)
        req_live = live_rows.reshape(groups, num_draft).any(dim=1)
        # Dead rows read a zero-length window, gather the zero row, and stage
        # their write on the scratch row; only the commit mask still sees them.
        lengths = torch.where(live_rows, lengths, lengths.new_zeros(()))
        kv_lens = (ctx_len + num_draft).to(torch.int32)
        kv_lens = torch.where(req_live, kv_lens, kv_lens.new_zeros(()))
        write_loc = (g_base[group] + ctx_len[group] + step).to(torch.int32)
        write_loc = torch.where(
            live_rows, write_loc, write_loc.new_full((), self._graph_scratch_row())
        )
        layout.req.copy_(req)
        layout.positions.copy_(positions)
        layout.kv_lens.copy_(kv_lens)
        layout.write_loc.copy_(write_loc)
        layout.commit_mask.copy_(live_rows)
        layout.indices.copy_(indices)
        layout.lengths.copy_(lengths)
        self.refresh_graph_commit_plan(layout)

        # History slots: each request owns ``window - 1`` rows; unused ones
        # gather the zero row into the scratch workspace row.
        hist_rows = window - 1
        hist_flat = torch.arange(groups * hist_rows, dtype=torch.int64, device=device)
        h_group = hist_flat // hist_rows
        h_off = hist_flat % hist_rows
        h_valid = (h_off < ctx_len[h_group]) & req_live[h_group]
        h_pos = ctx_start[h_group] + h_off
        h_src = self.row_loc(req[h_group * num_draft], h_pos)
        scratch = h_pos.new_full((), self._graph_scratch_row())
        layout.history_src.copy_(
            torch.where(h_valid, h_src, h_src.new_full((), self.zero_row))
        )
        layout.history_loc.copy_(torch.where(h_valid, g_base[h_group] + h_off, scratch))
        layout.history_pos.copy_(torch.where(h_valid, h_pos, torch.zeros_like(h_pos)))
        layout.history_valid.copy_(h_valid)

    def refresh_graph_commit_plan(self, layout: NPUWindowLayout) -> None:
        """Rewrite a fixed-shape graph commit plan from the live layout rows."""
        dst = self.row_loc(layout.req, layout.positions)
        dst = torch.where(layout.commit_mask, dst, dst.new_full((), self.sink_row))
        layout.commit_src.copy_(layout.write_loc.to(torch.int64))
        layout.commit_dst.copy_(dst)
        layout.commit_tags.copy_(layout.positions.to(torch.int64))

    def _scatter_commit(self, layer: int, layout: NPUWindowLayout) -> None:
        # The layout precomputes the commit addresses once per step; every
        # layer reuses them.
        _row_view(self.state[layer]).index_copy_(
            0,
            layout.commit_dst,
            _row_view(self.workspace).index_select(0, layout.commit_src),
        )
        self.tags[layer].index_copy_(0, layout.commit_dst, layout.commit_tags)
        self.prepared = None

    def activate_idle(self, num_rows: int) -> torch.Tensor:
        layout = self._idle_layout
        if layout is None or layout.pos.numel() != num_rows:
            zeros = torch.zeros(num_rows, dtype=torch.int64, device=self.tags.device)
            layout = window_layout(
                zeros,
                zeros,
                zero_row=self.zero_row,
                capacity=self.capacity,
                num_groups=1,
            )
            self._idle_layout = layout
        self.activate(layout)
        return layout.write_loc

    def reset(self, slots) -> None:
        reset_tag_rows(self.tags, slots, self.capacity)
        self.prepared = None

    def initialize_dummy_history(self) -> None:
        layout = self.layout
        self.tags.fill_(-1)
        # See __init__: float8 rows cannot use fill_ on A5.
        self.state.view(torch.uint8).zero_()
        if layout.history_src.numel():
            self.tags[:, layout.history_src.to(torch.int64)] = layout.history_pos
        self.prepared = None

    def buffer(self, layer: int) -> torch.Tensor:
        """Copy the layer's history rows into the workspace and return it."""
        if self.prepared != layer:
            layout = self.layout
            if layout is None:
                raise RuntimeError("request-window metadata was not activated")
            if layout.history_src.numel() == 0:
                # No replayable history in this layout (e.g. a fresh/folded
                # request): nothing to gather.
                self.prepared = layer
                return self.workspace
            src = layout.history_src.to(torch.int64)
            if not self.graph_capture and resolve_level() >= InvariantCheckLevel.WARN:
                # A reset slot or a reused ring row must not be read as history;
                # the tags record the position each row currently holds.
                expect(
                    _SWA_HISTORY_PRESENT,
                    (self.tags[layer, src] == layout.history_pos)
                    | ~layout.history_valid,
                    msg="replay or window ownership is invalid",
                )
            _row_view(self.workspace).index_copy_(
                0,
                layout.history_loc.to(torch.int64),
                _row_view(self.state[layer]).index_select(0, src),
            )
            self.prepared = layer
        return self.workspace

    def commit(self, layer: int) -> None:
        layout = self.layout
        if not isinstance(layout, NPUWindowLayout):
            raise RuntimeError("request-window commit needs a staged layout")
        self._scatter_commit(layer, layout)


class NPURequestWindowController:
    """Bind forward metadata to the row-style window for one attention backend."""

    def __init__(
        self,
        *,
        window: NPURequestWindow,
        page_size: int,
        sliding_window: int,
        device: torch.device,
        graph_metadata,
        graph_tokens_per_req: int,
        copy_page_table,
    ) -> None:
        self.window = window
        self.page_size = page_size
        self.sliding_window = sliding_window
        self.device = device
        # The backend rebinds graph_metadata when graph buffers are built.
        self.graph_metadata = graph_metadata
        self.graph_tokens_per_req = graph_tokens_per_req
        self.copy_page_table = copy_page_table
        self.ring_pages = window.capacity // page_size
        self._ring_page_tables = RingPageTableCache()
        self._graph_page_tables: dict[tuple, torch.Tensor] = {}

    def _page_table(self, layout, batch_size, positions, max_seq: int):
        window = self.window
        page = window.page_size
        seq_pages = -(-int(max_seq) // page)
        # Each request has its own page-aligned segment, so the table is per
        # request; a broadcast table would alias the first request's rows.
        base = layout.base
        group_pages = layout.group_pages
        assert int(base.numel()) == int(batch_size), (base.numel(), batch_size)
        num_pages = max(int(layout.group_pages_max), seq_pages, 1)
        base_pages = base.to(torch.int64) // page
        pages_per_req = group_pages.to(torch.int64).clamp_min(1)
        q = torch.arange(num_pages, dtype=torch.int64, device=positions.device)
        pages = base_pages[:, None] + (q[None, :] % pages_per_req[:, None])
        pages = torch.where(group_pages[:, None] > 0, pages, torch.zeros_like(pages))
        return pages.to(torch.int32).contiguous()

    def _max_seq_len(self, fm, forward_batch, rows: int) -> int:
        """Largest sequence length from host metadata.

        Reads the device only when no host mirror exists; the CPU lengths are
        the scheduler's own copy, so this does not add a D2H copy.
        """
        max_seq = _cpu_max(forward_batch.seq_lens_cpu, rows)
        mirror = getattr(fm, "seq_lens_cpu_int", None)
        max_seq = max(max_seq, _cpu_max(mirror, rows))
        if max_seq > 0:
            return max_seq
        seq_lens = forward_batch.seq_lens
        return int(seq_lens.max()) if seq_lens.numel() else 1

    def init_prefill(self, fm, forward_batch) -> None:
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=True)
        # Metadata is host-side: expand requests from the CPU lengths instead of
        # a device repeat_interleave, which is not reliable on NPU.
        lens_cpu = forward_batch.extend_seq_lens_cpu
        if not isinstance(lens_cpu, list):
            lens_cpu = lens_cpu.tolist()
        num_rows = sum(lens_cpu)
        req_cpu = forward_batch.req_pool_indices_cpu
        if req_cpu is None:
            req_cpu = forward_batch.req_pool_indices.cpu()
        req = torch.repeat_interleave(
            req_cpu.to(torch.int64),
            torch.tensor(lens_cpu, dtype=torch.int64),
        ).to(forward_batch.positions.device)
        positions = forward_batch.positions[:num_rows].to(torch.int64)
        floor = forward_batch.encoder_swa_row_floor
        if floor is not None:
            floor = floor.reshape(-1).to(torch.int64)
            if floor.numel() < num_rows:
                floor = torch.cat([floor, floor.new_zeros(num_rows - floor.numel())])
            floor = floor[:num_rows]
        layout = window.build_prefill_layout(
            req,
            positions,
            floor=floor,
            num_groups=forward_batch.batch_size,
            window=self.sliding_window,
        )
        window.activate(layout)
        fm.request_window_layout = layout
        fm.dsv41_window_kv_lens = layout.kv_lens
        fm.swa_loc = layout.write_loc
        indices = layout.indices.unsqueeze(1).contiguous()
        lengths = layout.lengths.unsqueeze(1).contiguous()
        fm.dsv41_win_sparse_indices = indices
        fm.dsv41_win_topk_length = lengths
        fm.swa_page_table = self._page_table(
            layout,
            forward_batch.batch_size,
            positions,
            self._max_seq_len(fm, forward_batch, forward_batch.batch_size),
        )

    def _logical_pages(self, fm, forward_batch, rows: int) -> int:
        # seqused_kv can exceed committed length by a speculative step; one
        # spare page keeps the scan in bounds (extra entries wrap onto valid pages).
        return max(
            1, -(-self._max_seq_len(fm, forward_batch, rows) // self.page_size) + 1
        )

    def init_decode_direct(self, fm, forward_batch) -> None:
        """Decode reads and writes the request ring directly (no staging)."""
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=False)
        positions = forward_batch.positions.reshape(-1).to(torch.int64)
        req = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        if req.numel() != positions.numel():
            req = torch.repeat_interleave(req, positions.numel() // req.numel())
        fm.swa_loc = window.row_loc(req, positions).to(torch.int64)
        fm.swa_page_table = self._ring_page_tables.table(
            req_pool_indices=req,
            ring_pages=self.ring_pages,
            logical_pages=self._logical_pages(fm, forward_batch, int(req.numel())),
        )

    def init_verify_direct(self, fm, forward_batch) -> None:
        """Verify writes its block rows and reads the ring directly."""
        window = self.window
        window.graph_capture = False
        window.set_phase(prefill=False)
        num_draft = int(forward_batch.spec_info.draft_token_num)
        bs = int(forward_batch.batch_size)
        rows = bs * num_draft
        req_pool = forward_batch.req_pool_indices.reshape(-1).to(torch.int64)
        req = torch.repeat_interleave(req_pool, num_draft)[:rows]
        positions = forward_batch.positions.reshape(-1)[:rows].to(torch.int64)
        live = forward_batch.out_cache_loc.reshape(-1)[:rows] >= self.page_size
        row_loc = window.row_loc(req, positions).to(torch.int64)
        fm.swa_loc = torch.where(live, row_loc, row_loc.new_full((), -1))
        fm.swa_page_table = self._ring_page_tables.table(
            req_pool_indices=req_pool,
            ring_pages=self.ring_pages,
            logical_pages=self._logical_pages(fm, forward_batch, bs),
        )

    def init_graph_metadata(self, metadata, forward_batch) -> None:
        window = self.window
        window.graph_capture = True
        bs = int(forward_batch.batch_size)
        mode = forward_batch.forward_mode
        if mode.is_decode():
            num_draft = 1
            live = (
                torch.arange(bs, device=self.device)
                < bs - int(getattr(forward_batch, "num_padding", 0) or 0)
            ) & (forward_batch.out_cache_loc[:bs] >= self.page_size)
        elif mode.is_target_verify():
            num_draft = self.graph_tokens_per_req
            live = forward_batch.out_cache_loc[: bs * num_draft] >= self.page_size
        else:
            return
        rows = bs * num_draft
        layout = window.build_graph_layout(bs, num_draft, window=self.sliding_window)
        window.refresh_graph_layout(
            layout,
            forward_batch.req_pool_indices[:bs].repeat_interleave(num_draft),
            forward_batch.positions[:rows],
            live,
            num_draft=num_draft,
            window=self.sliding_window,
        )
        window.activate(layout)
        metadata.request_window_layout = layout
        metadata.swa_loc = layout.write_loc
        metadata.dsv41_window_kv_lens = layout.kv_lens
        metadata.dsv41_win_sparse_indices = layout.indices.unsqueeze(1).contiguous()
        metadata.dsv41_win_topk_length = layout.lengths.unsqueeze(1).contiguous()
        self.refresh_graph_page_table(
            bs, layout, self._max_seq_len(None, forward_batch, rows)
        )

    def refresh_graph_page_table(self, bs: int, layout, max_seq: int = 0) -> None:
        window = self.window
        page = window.page_size
        # The operator scans to the request's sequence length, not just the
        # staged block; columns wrap onto its segment, so never leave -1 sentinels.
        seq_pages = max(1, -(-int(max_seq) // page))
        base = layout.base
        group_pages = layout.group_pages
        num_pages = max(int(layout.group_pages_max), seq_pages)
        table_width = int(self.graph_metadata()["swa_page_table"].shape[1])
        width = min(table_width, num_pages)
        # A graph layout's bases and segment sizes are fixed, so one full-width
        # table serves every replay; the live slice follows the scanned width.
        key = (int(bs), base.data_ptr(), group_pages.data_ptr(), self.device)
        table = self._graph_page_tables.get(key)
        if table is None:
            base_pages = base.to(torch.int64) // page
            pages_per_req = group_pages.to(torch.int64).clamp_min(1)
            q = torch.arange(table_width, dtype=torch.int64, device=self.device)
            pages = base_pages[:, None] + (q[None, :] % pages_per_req[:, None])
            pages = torch.where(
                group_pages[:, None] > 0, pages, torch.zeros_like(pages)
            )
            table = pages.to(torch.int32).contiguous()
            self._graph_page_tables[key] = table
        self.copy_page_table("swa_page_table", table[:, :width])

    def _live_row_mask(self, ctx, fb, rows: int):
        """Live rows of a replay layout, or None for a replay with no work.

        ``positions`` is padded to the graph bucket, but ``out_cache_loc`` is
        the unpadded live batch: a partial bucket (fewer local requests than
        the DP-wide graph bucket) still refreshes its live prefix, and the
        padded tail stays dead so it can neither read a stale window nor
        commit.
        """
        if ctx.raw_bs <= 0 or fb.positions.numel() < rows:
            return None
        loc = fb.out_cache_loc.reshape(-1)
        if loc.numel() >= rows:
            # The common full bucket: no padding rows to mask off.
            return loc[:rows] >= self.page_size
        live = torch.zeros(rows, dtype=torch.bool, device=ctx.device)
        if loc.numel():
            live[: loc.numel()] = loc >= self.page_size
        return live

    def _neutralize_idle_layout(self, layout) -> None:
        """Make an idle replay read nothing and commit nothing.

        Clearing only the commit mask would leave the previous step's lengths
        as ``seqused`` and its indexed rows as attention sources. Neutralize
        every read instead: a legal first index with zero length, history from
        the zero row, and dead writes parked on the workspace scratch row.
        """
        layout.commit_mask.fill_(False)
        layout.lengths.zero_()
        layout.kv_lens.zero_()
        layout.indices.fill_(-1)
        layout.indices[..., 0] = 0
        layout.history_valid.zero_()
        layout.history_src.fill_(self.window.zero_row)
        layout.write_loc.fill_(self.window.workspace.shape[0] - 1)
        layout.commit_src.zero_()
        layout.commit_dst.fill_(self.window.sink_row)
        layout.commit_tags.zero_()

    def refresh_graph(self, ctx) -> None:
        layout = getattr(ctx.fm, "request_window_layout", None)
        if layout is None:
            return
        fb = ctx.forward_batch
        if ctx.graph_mode.is_decode():
            num_draft = 1
        elif ctx.graph_mode.is_target_verify():
            num_draft = ctx.tokens_per_bs
        else:
            return
        rows = ctx.bs * num_draft
        live = self._live_row_mask(ctx, fb, rows)
        if live is None:
            # Idle replay: keep the captured shape, read and commit nothing.
            self._neutralize_idle_layout(layout)
            return
        self.window.refresh_graph_layout(
            layout,
            fb.req_pool_indices[: ctx.bs].repeat_interleave(num_draft),
            fb.positions[:rows],
            live,
            num_draft=num_draft,
            window=self.sliding_window,
        )

    def activate(self, metadata) -> None:
        layout = getattr(metadata, "request_window_layout", None)
        if layout is not None:
            self.window.activate(layout)
