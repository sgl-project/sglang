"""Functional tests for DeepSelect top-k against a torch.topk reference.

Checks behavior only: the selected values match the reference as a multiset
(ties may pick different columns), indices are distinct and inside the row's
valid range, and slots past a row's ``end`` hold the fill values.
"""

from __future__ import annotations

import sys
from typing import Optional

import pytest
import torch

from sglang.kernels.ops.attention.deep_select import (
    get_input_stride_alignment_bytes,
    get_output_stride_alignment_bytes,
    is_deep_select_supported,
    topk,
    topk_page_transform,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_deep_select_supported(), reason="requires CUDA SM90, SM100, or SM103"
)

DEFAULT_IDX_FILL = 2147483647


def _make_input(
    rows: int, width: int, dtype: torch.dtype, seed: int = 0
) -> torch.Tensor:
    """Random scores whose rows are padded to the documented input alignment."""
    step = get_input_stride_alignment_bytes() // dtype.itemsize
    padded = (width + step - 1) // step * step
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.randn(rows, padded, device="cuda", generator=g).to(dtype)
    return buf[:, :width]


def _check_topk(
    input: torch.Tensor,
    k: int,
    indices: torch.Tensor,
    *,
    end: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    idx_fill: int = DEFAULT_IDX_FILL,
    value_fill: Optional[float] = None,
    offset: Optional[torch.Tensor] = None,
) -> None:
    rows, width = input.shape
    assert indices.shape == (rows, k)
    scores = input.float()
    for r in range(rows):
        row_end = width if end is None else int(end[r])
        num_valid = min(row_end, k)
        row_idx = indices[r].long()
        valid = row_idx != idx_fill
        assert int(valid.sum()) == num_valid, f"row {r}"
        local = row_idx[valid] - (0 if offset is None else int(offset[r]))
        if num_valid > 0:
            assert 0 <= int(local.min()) and int(local.max()) < row_end, f"row {r}"
        assert local.unique().numel() == num_valid, f"row {r}"
        picked = scores[r, local]
        expected = torch.topk(scores[r, :row_end], num_valid).values
        torch.testing.assert_close(
            picked.sort(descending=True).values, expected, rtol=0, atol=0
        )
        if values is not None:
            row_values = values[r].float()
            torch.testing.assert_close(row_values[valid], picked, rtol=0, atol=0)
            if value_fill is not None:
                assert torch.all(row_values[~valid] == value_fill), f"row {r}"


@pytest.mark.parametrize("rows,width", [(4, 8192), (200, 30000)])
def test_topk_matches_reference(rows, width):
    input = _make_input(rows, width, torch.bfloat16)
    values, indices = topk(input, 512)
    assert indices.dtype == torch.int32
    _check_topk(input, 512, indices, values=values)


def test_sorted_index_int64():
    input = _make_input(16, 30000, torch.bfloat16)
    _, indices = topk(
        input, 1024, indices_type=torch.int64, sorted_index=True, return_value=False
    )
    assert indices.dtype == torch.int64
    assert torch.all(indices[:, 1:] > indices[:, :-1])
    _check_topk(input, 1024, indices)


def test_sorted_values_fp32():
    input = _make_input(8, 20000, torch.float32)
    values, indices = topk(input, 1024, sorted=True)
    expected = torch.topk(input, 1024)
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(
        input.gather(1, indices.long()), expected.values, rtol=0, atol=0
    )


def test_variable_row_end():
    k, width = 512, 8192
    ends = [0, 1, 100, k - 1, k, k + 1, 4000, width]
    input = _make_input(len(ends), width, torch.bfloat16)
    end = torch.tensor(ends, dtype=torch.int32, device="cuda")
    values, indices = topk(
        input, k, end=end, idx_oob_fill_value=-1, value_oob_fill_value=-5.0
    )
    _check_topk(input, k, indices, end=end, values=values, idx_fill=-1, value_fill=-5.0)


def test_caller_output_and_index_offset():
    rows, k, width = 5, 512, 8192
    input = _make_input(rows, width, torch.bfloat16)
    pad = get_output_stride_alignment_bytes() // torch.int32.itemsize
    backing = torch.full((rows, k + pad), -7, dtype=torch.int32, device="cuda")
    output = backing[:, :k]
    offset = torch.randint(0, 1 << 20, (rows,), dtype=torch.int32, device="cuda")

    values, indices = topk(input, k, output_idx=output, output_idx_offset=offset)

    assert indices.data_ptr() == output.data_ptr()
    assert torch.all(backing[:, k:] == -7), "wrote past the output row"
    _check_topk(input, k, indices, values=values, offset=offset)


def _random_page_table(rows: int, width: int, page_size: int):
    """Distinct page ids per slot, plus the inverse map from page id to logical page."""
    num_pages = (width + page_size - 1) // page_size
    ids = torch.randperm(rows * num_pages * 2, device="cuda", dtype=torch.int32)
    table = ids[: rows * num_pages].view(rows, num_pages)
    inverse = torch.full((rows * num_pages * 2,), -1, dtype=torch.int64, device="cuda")
    inverse[table.flatten().long()] = torch.arange(num_pages, device="cuda").repeat(
        rows
    )
    return table, inverse


@pytest.mark.parametrize(
    "k,rows,width,with_end",
    [
        (2048, 150, 33000, True),
        (1024, 2, 600_000, False),
    ],
)
def test_page_transform_matches_reference(k, rows, width, with_end):
    page_size = 64
    input = _make_input(rows, width, torch.bfloat16)
    end = None
    if with_end:
        g = torch.Generator(device="cuda").manual_seed(1)
        end = torch.randint(
            1, width + 1, (rows,), dtype=torch.int32, device="cuda", generator=g
        )
        end[0] = k // 2
    table, inverse = _random_page_table(rows, width, page_size)

    slots = topk_page_transform(
        input, k, page_table=table, page_size=page_size, end=end
    )

    assert slots.dtype == torch.int32 and slots.shape == (rows, k)
    valid = slots != -1
    wide = slots.long()
    logical = inverse[(wide // page_size).clamp(min=0)] * page_size + wide % page_size
    logical = torch.where(valid, logical, torch.full_like(logical, -1))
    _check_topk(input, k, logical, end=end, idx_fill=-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
