"""Multi-row PLE gather from the host table (SGLANG_PLE_GATHER_ROWS, default off).

The stock ``_gather_ple_embedding_from_pinned_kernel`` runs one program per looked-up row: a
16K-token prefill launches 262,144 programs that each read one 160-byte row over the host link,
so the pass waits on per-program latency rather than on bytes. This kernel reads ROWS rows per
program, all issued before the first store. Each element goes through the stock kernel's
load, cast to bf16 and out-of-shard zero, so the output is the same bits.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _rows_from_env() -> int:
    raw = os.environ.get("SGLANG_PLE_GATHER_ROWS", "0").strip()
    rows = int(raw) if raw else 0
    if rows and (rows < 2 or rows > 128 or rows & (rows - 1)):
        raise ValueError(f"SGLANG_PLE_GATHER_ROWS must be a power of two in 2..128, got {rows}")
    return rows


ROWS = _rows_from_env()
# Programs launched (0: one per ROWS rows). A capped grid loops over the row tiles and leaves
# the other SMs to the forward that runs beside the gather.
PROGRAMS = int(os.environ.get("SGLANG_PLE_GATHER_PROGRAMS", "0") or 0)
_logged = [False]


@triton.jit
def _gather_ple_rows_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    num_rows,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    offsets = tl.arange(0, BLOCK_D)
    num_tiles = tl.cdiv(num_rows, ROWS)
    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        rows = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_live = rows < num_rows
        global_idx = tl.load(ids_ptr + rows, mask=row_live, other=0)
        in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
        local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
        mask = row_live[:, None] & (offsets < embedding_dim)[None, :]
        values = tl.load(
            weight_ptr + local_idx[:, None] * embedding_dim + offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
        tl.store(
            output_ptr + rows[:, None] * embedding_dim + offsets[None, :],
            tl.where(in_range[:, None], values, 0.0),
            mask=mask,
        )
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


def gather_rows(
    *,
    weight_ptr: int,
    flat_ids: torch.Tensor,
    output: torch.Tensor,
    embedding_dim: int,
    tp_vocab_start: int,
    tp_vocab_end: int,
    is_fp8: bool,
    block_d: int,
    pdl_small: bool,
    rows: int = 0,
    programs: int = -1,
) -> None:
    """output[i] = bf16(table[flat_ids[i] - tp_vocab_start]), zero outside the shard."""
    rows = rows or ROWS
    programs = PROGRAMS if programs < 0 else programs
    num_rows = flat_ids.numel()
    tiles = triton.cdiv(num_rows, rows)
    _gather_ple_rows_kernel[(min(tiles, programs) if programs > 0 else tiles,)](
        weight_ptr,
        flat_ids,
        output,
        num_rows,
        embedding_dim=embedding_dim,
        tp_vocab_start=tp_vocab_start,
        tp_vocab_end=tp_vocab_end,
        is_fp8=is_fp8,
        BLOCK_D=block_d,
        ROWS=rows,
        launch_pdl=pdl_small,
        PDL_SMALL=pdl_small,
    )
    if not _logged[0]:
        _logged[0] = True
        logger.info(
            "SGLANG_PLE_GATHER_ROWS: PLE host gather at %d rows per program, %s programs "
            "(%d rows, dim %d, %s)",
            rows,
            programs if programs > 0 else "uncapped",
            num_rows,
            embedding_dim,
            "fp8" if is_fp8 else "bf16",
        )
