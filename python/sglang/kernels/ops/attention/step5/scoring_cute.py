from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class Step5ScoringWorkspace:
    request_indices: torch.Tensor
    table_indices: torch.Tensor
    block_table: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    request_limits: torch.Tensor
    valid_tokens: torch.Tensor
    q_runtime: torch.Tensor
    kernel_weights: torch.Tensor
    rows_per_group: int


def allocate_scoring_workspace(rows, regions, group_rows, device):
    group = 1 if group_rows == 1 else 4
    groups = triton.cdiv(rows, group)
    packed_rows = groups * 4
    empty = lambda *shape, dtype=torch.int32: torch.empty(
        shape, dtype=dtype, device=device
    )
    return Step5ScoringWorkspace(
        request_indices=empty(packed_rows),
        table_indices=empty(packed_rows),
        block_table=empty(groups, triton.cdiv(regions, 8)),
        row_starts=torch.zeros(packed_rows, dtype=torch.int32, device=device),
        row_ends=empty(packed_rows),
        request_limits=empty(2),
        valid_tokens=empty(1),
        q_runtime=empty(groups * 16, 256, dtype=torch.float8_e4m3fn),
        kernel_weights=empty(groups, 32, 4, dtype=torch.float32),
        rows_per_group=group,
    )


@triton.jit
def _prepare_scoring(
    Q,
    W,
    Requests,
    Positions,
    PackedQ,
    PackedW,
    RowReq,
    RowTable,
    RowEnds,
    RequestLimits,
    ValidTokens,
    Out,
    Mapping,
    Table,
    MAP_STRIDE: tl.constexpr,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    CLEAR_FUTURE: tl.constexpr,
    REDUCE_ROWS: tl.constexpr,
    PACK_QW: tl.constexpr = True,
):
    pid = tl.program_id(0)
    groups = tl.cdiv(ROWS, GROUP)
    if pid < groups * 4:
        group = pid // 4
        slot = pid % 4
        row = group * GROUP + slot
        valid_row = (slot < GROUP) & (row < ROWS)
        request = tl.load(Requests + row, valid_row, other=0)
        position = tl.load(Positions + row, valid_row, other=0)
        valid = valid_row & (request > 0)
        regions = tl.maximum(position // 8, 0)
        if not CLEAR_FUTURE:
            regions = tl.where(regions > 512, regions, 0)
        if PACK_QW:
            h = tl.arange(0, 4)
            d = tl.arange(0, 256)
            q = tl.load(
                Q + (row * HEADS + h[:, None]) * 256 + d[None, :],
                valid & (h[:, None] < HEADS),
                other=0,
            )
            if Q.dtype.element_ty != tl.uint8:
                q = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            w = tl.load(W + row * HEADS + h, valid & (h < HEADS), other=0)
            packed_h = group * 16 + (h // 2) * 8 + slot * 2 + h % 2
            tl.store(PackedQ + packed_h[:, None] * 256 + d[None, :], q)
            tl.store(
                PackedW + group * 128 + slot * 4 + h, w.to(tl.float32) * WEIGHT_SCALE
            )
        slots = tl.arange(0, 4)
        owners = tl.load(
            Requests + group * GROUP + slots,
            (slots < GROUP) & (group * GROUP + slots < ROWS),
            other=0,
        )
        tl.store(RowReq + pid, group)
        owner = tl.max(owners, 0)
        owner_positions = tl.load(
            Positions + group * GROUP + slots,
            (slots < GROUP) & (group * GROUP + slots < ROWS),
            other=0,
        )
        live_pages = tl.cdiv(
            tl.max(tl.where(owners > 0, tl.maximum(owner_positions // 8, 0), 0), 0), 8
        )
        for page_start in range(slot * 256, live_pages, 1024):
            col = page_start + tl.arange(0, 256)
            physical = tl.load(
                Mapping + owner.to(tl.int64) * MAP_STRIDE + col * 64,
                (owner > 0) & (col < live_pages) & (col < COLS),
                other=0,
            )
            tl.store(
                Table + group * COLS + col,
                physical // 64,
                (col < COLS) & (col < live_pages),
            )
        tl.store(RowTable + pid, group)
        tl.store(RowEnds + pid, tl.where(valid, regions, 0))
        if pid == 0:
            tl.store(RequestLimits, groups)
            all_rows = tl.arange(0, REDUCE_ROWS)
            all_requests = tl.load(Requests + all_rows, all_rows < ROWS, other=0)
            all_positions = tl.load(Positions + all_rows, all_rows < ROWS, other=0)
            live = tl.max(
                tl.where(all_requests > 0, tl.maximum(all_positions // 8, 0), 0), 0
            )
            if not CLEAR_FUTURE:
                live = tl.where(live > 512, live, 0)
            tl.store(RequestLimits + 1, live)
            tl.store(ValidTokens, groups * 4)
    elif CLEAR_FUTURE:
        index = (pid - groups * 4) * 1024 + tl.arange(0, 1024)
        fill_row = index // WIDTH
        fill_col = index % WIDTH
        fill_req = tl.load(Requests + fill_row, fill_row < ROWS, other=0)
        fill_pos = tl.load(Positions + fill_row, fill_row < ROWS, other=0)
        tl.store(
            Out + index,
            -float("inf"),
            (fill_row < ROWS) & ((fill_req <= 0) | (fill_col >= fill_pos // 8)),
        )


def prepare_scoring(
    workspace,
    mapping,
    requests,
    positions,
    *,
    q=None,
    weights=None,
    out=None,
    weight_scale=1.0,
    clear_future=False,
):
    """Prepare scoring inputs, or only metadata when Q/W come from fused norm.

    Without q, the packing branch is compiled out. Call once from the forward's
    metadata build so CUDA Graph replay refreshes metadata shared across layers.
    """
    w = workspace
    rows = requests.numel()
    group = w.rows_per_group
    groups = triton.cdiv(rows, group)
    width = out.shape[1] if out is not None else 0
    heads = q.shape[1] if q is not None else 0
    if q is not None and q.dtype == torch.float8_e4m3fn:
        q = q.view(torch.uint8)
    fill_blocks = triton.cdiv(rows * width, 1024) if clear_future else 0
    _prepare_scoring[(groups * 4 + fill_blocks,)](
        q,
        weights,
        requests,
        positions,
        w.q_runtime.view(torch.uint8),
        w.kernel_weights,
        w.request_indices,
        w.table_indices,
        w.row_ends,
        w.request_limits,
        w.valid_tokens,
        out,
        mapping,
        w.block_table,
        mapping.shape[1],
        w.block_table.shape[1],
        rows,
        width,
        heads,
        group,
        weight_scale,
        clear_future,
        triton.next_power_of_2(rows),
        PACK_QW=q is not None,
    )


def launch(
    q,
    weights,
    keys,
    mapping,
    requests,
    positions,
    out,
    *,
    clear_future,
    group_rows,
    weight_scale,
    workspace=None,
):
    rows = q.shape[0]
    if workspace is None:
        workspace = allocate_scoring_workspace(rows, out.shape[1], group_rows, q.device)
    prepare_scoring(
        workspace,
        mapping,
        requests,
        positions,
        q=q,
        weights=weights,
        out=out,
        weight_scale=weight_scale,
        clear_future=clear_future,
    )
    launch_prepared_scoring(keys, out, workspace, rows)


def launch_prepared_scoring(keys, out, workspace, rows):
    """Score using Q/W and metadata already written into the workspace."""
    from .cute.scoring_sm100 import launch_paged_scoring

    w = workspace
    groups = triton.cdiv(rows, w.rows_per_group)
    launch_paged_scoring(
        keys,
        w.block_table,
        w.request_indices[: groups * 4],
        w.table_indices[: groups * 4],
        w.row_starts[: groups * 4],
        w.row_ends[: groups * 4],
        w.q_runtime[: groups * 16],
        w.kernel_weights[:groups],
        out,
        w.request_limits,
        w.valid_tokens,
        w.rows_per_group,
    )
