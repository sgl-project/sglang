"""Owned DSA DCP route on ROCm: the decode indexer row split.

Covers:
1. A row slice of the packed top-k with ``row_to_batch`` gives the same rows as
   the whole batch, which is what the per-rank decode indexer split relies on.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)

TOPK, DCP = 2048, 4


@pytest.mark.parametrize("rows", [24, 144])
def test_packed_topk_row_slice_matches_whole_batch(rows):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_packed_v2

    torch.manual_seed(2)
    page_size, max_len = 64, 16384
    lengths = torch.randint(
        TOPK // 2, max_len, (rows,), device="cuda", dtype=torch.int32
    )
    pages = max_len // page_size
    page_table = torch.stack(
        [torch.randperm(4 * pages, device="cuda")[:pages] for _ in range(rows)]
    ).int()
    scores = torch.randn(rows, max_len, device="cuda")

    whole = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    all_rows = torch.arange(rows, dtype=torch.int32, device="cuda")
    topk_transform_packed_v2(
        scores.clone(),
        lengths,
        page_table,
        whole,
        page_size,
        row_starts=torch.zeros_like(all_rows),
        row_to_batch=all_rows,
    )

    per = -(-rows // DCP)
    split = torch.empty(per * DCP, TOPK, dtype=torch.int32, device="cuda")
    for rank in range(DCP):
        lo, hi = min(rank * per, rows), min(rank * per + per, rows)
        if hi == lo:
            continue
        row_ids = torch.arange(lo, hi, dtype=torch.int32, device="cuda")
        topk_transform_packed_v2(
            scores[lo:hi].clone(),
            lengths[lo:hi],
            page_table,
            split[rank * per : rank * per + hi - lo],
            page_size,
            row_starts=torch.zeros_like(row_ids),
            row_to_batch=row_ids,
        )

    # The top-k order within a row is unspecified; the selected set is not.
    torch.testing.assert_close(
        split[:rows].sort(-1).values, whole.sort(-1).values, rtol=0, atol=0
    )
    # Each selected slot is the page-table translation of one of the row's top-k.
    for r in (0, rows - 1):
        n = lengths[r].item()
        pos = torch.topk(scores[r, :n], TOPK).indices
        expect = page_table[r, pos // page_size] * page_size + pos % page_size
        assert set(whole[r].tolist()) == set(expect.int().tolist())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
