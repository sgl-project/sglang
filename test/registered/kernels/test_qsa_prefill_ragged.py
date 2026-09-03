"""Ragged-boundary correctness tests for the TileLang QSA prefill MQA kernel.

The TileLang prefill kernel iterates compressed-K positions in tiles of
block_n=64 rows. Previous versions copied a full block_n rows from K and
wrote a full block_n entries per query into Logits even when the last
tile was only partially covered by the row's [Starts, Ends) window. That
produced out-of-bounds K reads past the end of the K allocation (visible
as NVIDIA Xid 31 / MMU Fault when the overshoot crosses a page boundary
under production ragged batching) and intra-buffer stray writes into
Logits positions that belong to other packed rows or to the invalid
region of the current row.

These tests exercise tail positions around every 64-row boundary
(63/64/65, 127/128/129, 191/192/193, 255/256/257, ...) with a variety of
start offsets, non-block_q-aligned row counts, and padding rows. Each
case compares the public TileLang entry point against the torch
reference, with torch.cuda.synchronize() after each call to
deterministically surface illegal accesses. A separate test inspects
the pre-mask raw-kernel output to directly assert that no writes fall
outside each row's [Starts, Ends) window.
"""

import math

import pytest
import torch

from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    _tilelang_qsa_mqa_mask_kernel,
    _tilelang_qsa_mqa_prefill_kernel,
    tilelang_qsa_mqa_prefill,
    torch_qsa_mqa_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TILELANG,
    reason="TileLang + CUDA required",
)

HEADS = 4
HEAD_DIM = 128
SCORE_SCALE = math.sqrt(HEAD_DIM)
BLOCK_N = 64  # kernel tile, kept in sync with mqa.py
BLOCK_Q = max(1, 128 // HEADS)


def _assert_matches_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    label: str,
):
    ref = torch_qsa_mqa_prefill(q, k, starts, ends, score_scale=SCORE_SCALE)
    torch.cuda.synchronize()
    out = tilelang_qsa_mqa_prefill(q, k, starts, ends, score_scale=SCORE_SCALE)
    torch.cuda.synchronize()
    valid = torch.isfinite(ref)
    invalid_finite = (torch.isfinite(out) & ~valid).sum().item()
    valid_nonfinite = (~torch.isfinite(out) & valid).sum().item()
    assert invalid_finite == 0, (
        f"{label}: TileLang wrote finite values into {int(invalid_finite)} "
        f"masked-invalid positions"
    )
    assert valid_nonfinite == 0, (
        f"{label}: TileLang produced non-finite values at "
        f"{int(valid_nonfinite)} valid positions"
    )
    if valid.any():
        maxdiff = (out - ref).masked_fill(~valid, 0.0).abs().max().item()
        assert maxdiff < 5e-3, f"{label}: max abs diff {maxdiff} exceeds tolerance"


@pytest.mark.parametrize(
    "keys",
    [
        # Around every 64-row boundary up through one chunked-prefill
        # compressed-window size (chunk=4096 -> 1024 compressed rows).
        *[
            base + d
            for base in range(0, 1088, BLOCK_N)
            for d in (-1, 0, 1)
            if base + d > 0
        ],
    ],
)
@pytest.mark.parametrize(
    "rows",
    [1, 4, 32, 33, 64],
)
def test_tilelang_prefill_mqa_tail_boundaries(rows: int, keys: int):
    """Single-pack rows, all starting at 0, ending at keys.

    Exercises the last-tile overshoot on both K and Logits at every 64-row
    boundary with different row counts (including block_q=32 padding cases).
    """
    torch.manual_seed(1234 + rows * 10000 + keys)
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    _assert_matches_reference(q, k, starts, ends, label=f"rows={rows} keys={keys}")


@pytest.mark.parametrize(
    "keys",
    [65, 129, 193, 257, 321, 513, 769],
)
def test_tilelang_prefill_mqa_nonzero_starts(keys: int):
    """Rows with non-zero, varied starts inside the K buffer.

    This is the production-realistic ragged case: multiple sequences packed
    into one K tensor, each observing a different [start,end) window. The
    previous kernel wrote tile entries across all of [start_min,end_max)
    even for rows whose Ends sat before the current tile.
    """
    torch.manual_seed(2345 + keys)
    rows = 6
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor(
        [0, keys // 5, keys // 3, keys // 2, (2 * keys) // 3, keys - 7],
        dtype=torch.int32,
        device="cuda",
    )
    ends = starts + torch.tensor(
        [keys // 6, keys // 5, keys // 5, keys // 6, keys // 8, 7],
        dtype=torch.int32,
        device="cuda",
    )
    ends = ends.clamp_max(keys)
    ends = torch.maximum(ends, starts + 1)
    _assert_matches_reference(q, k, starts, ends, label=f"nonzero_starts keys={keys}")


def test_tilelang_prefill_mqa_mixed_pack_exact_boundaries():
    """A hand-curated mix that lands every tail boundary on a different row."""
    torch.manual_seed(3456)
    keys = 577  # > 9 * 64 = 576, hits a 576->577 tail
    rows = 10
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor(
        [0, 0, 32, 65, 128, 193, 257, 320, 384, 513],
        dtype=torch.int32,
        device="cuda",
    )
    ends = torch.tensor(
        [63, 65, 96, 129, 192, 255, 321, 383, 449, 577],
        dtype=torch.int32,
        device="cuda",
    )
    _assert_matches_reference(q, k, starts, ends, label="mixed_pack_577")


@pytest.mark.parametrize("rows", [1, 33, 35, 63, 65])
def test_tilelang_prefill_mqa_non_block_q_rows_small_keys(rows: int):
    """Rows counts that are not a multiple of block_q=32, including the
    padding-row case. K deliberately small (65 rows) so the last tile is
    only partially filled and the padding rows' [start,end) is set to a
    degenerate window.
    """
    torch.manual_seed(4567 + rows)
    keys = 65
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), keys, dtype=torch.int32, device="cuda")
    _assert_matches_reference(q, k, starts, ends, label=f"rows={rows} keys={keys}")


def test_tilelang_prefill_mqa_large_random_pack():
    """Stress test at production-like scale (up to ~1024 compressed keys,
    up to 63 rows, arbitrary non-zero starts/ends)."""
    torch.manual_seed(5678)
    for _ in range(10):
        rows = int(torch.randint(1, 64, (1,)).item())
        keys = int(torch.randint(1, 1088, (1,)).item())
        q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        starts = torch.randint(
            0, max(1, keys), (rows,), dtype=torch.int32, device="cuda"
        )
        ends = starts + torch.randint(
            1, max(2, keys), (rows,), dtype=torch.int32, device="cuda"
        )
        ends = ends.clamp_max(keys)
        ends = torch.maximum(ends, starts + 1)
        _assert_matches_reference(
            q, k, starts, ends, label=f"random rows={rows} keys={keys}"
        )


def test_tilelang_prefill_mqa_single_token_keys():
    """Degenerate edge: 1 key (smaller than one tile)."""
    torch.manual_seed(6789)
    q = torch.randn(1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.ones(1, dtype=torch.int32, device="cuda")
    _assert_matches_reference(q, k, starts, ends, label="single_key")


def _run_raw_prefill_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Call the raw prefill kernel (without the follow-up mask kernel) and
    return the pre-mask logits tensor of shape [padded_rows, keys].

    Callers MUST call torch.cuda.synchronize() afterwards to surface any
    async illegal-memory-access errors.
    """
    rows, keys = q.shape[0], k.shape[0]
    padded_rows = rows + ((-rows) % BLOCK_Q)
    logits = torch.full(
        (padded_rows, keys), float("nan"), dtype=torch.float32, device=q.device
    )
    q_padded = q.contiguous()
    starts_pad = starts.contiguous()
    ends_pad = ends.contiguous()
    pad = padded_rows - rows
    if pad:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(pad, HEADS, HEAD_DIM)])
        starts_pad = torch.cat([starts_pad, starts_pad[-1:].expand(pad)])
        ends_pad = torch.cat([ends_pad, ends_pad[-1:].expand(pad)])
    _tilelang_qsa_mqa_prefill_kernel(heads=HEADS, head_dim=HEAD_DIM, block_q=BLOCK_Q)(
        q_padded.reshape(-1, HEAD_DIM),
        k[:, 0].contiguous(),
        logits,
        starts_pad,
        ends_pad,
    )
    return logits


def _stray_write_count(logits: torch.Tensor, starts, ends) -> int:
    """Count finite values written outside [start, end) for each row."""
    bad = 0
    for r in range(starts.shape[0]):
        s, e = int(starts[r]), int(ends[r])
        bad += int(torch.isfinite(logits[r, :s]).sum().item())
        bad += int(torch.isfinite(logits[r, e : logits.shape[1]]).sum().item())
    return bad


def test_tilelang_prefill_raw_kernel_no_stray_writes_mixed_pack():
    """Directly inspect the pre-mask kernel output on a staggered pack.

    The follow-up mask kernel _tilelang_qsa_mqa_mask_kernel masks
    [-inf, start) U [end, keys) AFTER the prefill kernel runs, so an
    end-to-end comparison against the torch reference can pass even when
    the prefill kernel writes stray finite values into invalid regions
    (those values then get overwritten to -inf by the mask). A production
    crash happens when those stray writes land OUTSIDE the Logits/K
    allocation (past the end of a row's last tile and off the buffer),
    which this direct inspection cannot catch—but intra-buffer stray
    writes into another row's valid window silently corrupt scores and
    must also be forbidden.
    """
    torch.manual_seed(3456)
    keys = 577
    rows = 10
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor(
        [0, 0, 32, 65, 128, 193, 257, 320, 384, 513],
        dtype=torch.int32,
        device="cuda",
    )
    ends = torch.tensor(
        [63, 65, 96, 129, 192, 255, 321, 383, 449, 577],
        dtype=torch.int32,
        device="cuda",
    )
    logits = _run_raw_prefill_kernel(q, k, starts, ends)
    torch.cuda.synchronize()
    stray = _stray_write_count(logits, starts, ends)
    assert stray == 0, (
        f"Raw prefill kernel wrote {stray} finite values outside "
        f"[Starts,Ends) windows before the mask kernel ran"
    )


@pytest.mark.parametrize("keys", [65, 129, 193, 257, 321, 513, 769, 1025])
def test_tilelang_prefill_raw_kernel_no_stray_writes_nonzero_starts(keys: int):
    """Per-nonzero-start case: no pre-mask stray writes."""
    torch.manual_seed(7890 + keys)
    rows = 6
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor(
        [0, keys // 5, keys // 3, keys // 2, (2 * keys) // 3, keys - 7],
        dtype=torch.int32,
        device="cuda",
    )
    ends = starts + torch.tensor(
        [keys // 6, keys // 5, keys // 5, keys // 6, keys // 8, 7],
        dtype=torch.int32,
        device="cuda",
    )
    ends = ends.clamp_max(keys)
    ends = torch.maximum(ends, starts + 1)
    logits = _run_raw_prefill_kernel(q, k, starts, ends)
    torch.cuda.synchronize()
    stray = _stray_write_count(logits, starts, ends)
    assert (
        stray == 0
    ), f"Raw prefill kernel (keys={keys}) wrote {stray} stray pre-mask values"


def test_tilelang_prefill_mask_kernel_is_idempotent_on_clean_logits():
    """Sanity: running the mask kernel on already-correct logits leaves
    valid positions untouched and invalid positions at -inf."""
    torch.manual_seed(9012)
    keys, rows = 257, 5
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(keys, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor([0, 32, 65, 128, 193], dtype=torch.int32, device="cuda")
    ends = torch.tensor([63, 96, 129, 192, 257], dtype=torch.int32, device="cuda")
    out = tilelang_qsa_mqa_prefill(q, k, starts, ends, score_scale=SCORE_SCALE)
    torch.cuda.synchronize()
    # Run mask a second time on the result
    _tilelang_qsa_mqa_mask_kernel()(out, starts, ends)
    torch.cuda.synchronize()
    ref = torch_qsa_mqa_prefill(q, k, starts, ends, score_scale=SCORE_SCALE)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=0)
