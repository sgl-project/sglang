"""topk_page_table_transform_ against the page_size=1 table it replaces."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.topk_page_table import topk_page_table_transform_
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("page_size", [1, 48, 64])
@pytest.mark.parametrize("topk", [512, 2048, 2052])
def test_matches_page_size_1_table(page_size: int, topk: int) -> None:
    _check(page_size, topk, repeat=1)


def test_more_rows_than_grid_y() -> None:
    # 65536 rows, past the 65535 limit of a grid's y dimension.
    _check(page_size=64, topk=512, repeat=8192)


def _check(page_size: int, topk: int, repeat: int) -> None:
    torch.manual_seed(0)
    batches, pages = 5, 40
    # Shuffled, non-contiguous page ids in a wider (strided) table.
    table = torch.randperm(batches * pages * 3, device="cuda", dtype=torch.int32)[
        : batches * pages
    ]
    table = torch.cat(
        [
            table.view(batches, pages),
            torch.full((batches, 7), -9, device="cuda", dtype=torch.int32),
        ],
        1,
    )
    table = table[:, :pages]
    width = pages * page_size
    lens = torch.tensor([1, 63, 64, 1000, width], device="cuda").clamp(max=width)
    row_batch = torch.tensor(
        [0, 1, 2, 3, 4, 4, 3, 1], device="cuda", dtype=torch.int32
    ).repeat(repeat)
    rows = row_batch.numel()
    pos = (torch.rand(rows, topk, device="cuda") * lens[row_batch.long()][:, None]).to(
        torch.int32
    )
    pos[:, ::3] = -1
    out = pos.clone()

    slots_1 = (
        table.long()[:, :, None] * page_size + torch.arange(page_size, device="cuda")
    ).view(batches, -1)
    ref = torch.where(
        pos >= 0, slots_1[row_batch.long()[:, None], pos.clamp(min=0).long()].int(), pos
    )
    topk_page_table_transform_(out, table, row_batch, page_size)
    assert torch.equal(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
