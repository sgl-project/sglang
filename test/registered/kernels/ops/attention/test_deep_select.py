"""Regression tests for the DeepSelect JIT wrapper and vendored kernels."""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

from sglang.kernels.ops.attention.deep_select import is_deep_select_supported, topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_deep_select_supported(), reason="requires CUDA SM90, SM100, or SM103"
)


def _aligned_input(
    rows: int, width: int, dtype: torch.dtype, device: torch.device | str = "cuda"
) -> torch.Tensor:
    alignment = 1024 // dtype.itemsize
    padded_width = (width + alignment - 1) // alignment * alignment
    return torch.empty((rows, padded_width), dtype=dtype, device=device)[:, :width]


def test_deepselect_topk_matches_official_signature():
    assert list(inspect.signature(topk).parameters) == [
        "input",
        "topk",
        "sorted",
        "begin",
        "end",
        "indices_type",
        "sorted_index",
        "hint",
        "output_idx",
        "output_idx_offset",
        "idx_oob_fill_value",
        "value_oob_fill_value",
        "return_value",
        "abort_when_nan_found",
    ]


def test_out_of_range_oob_fill_is_rejected():
    input = _aligned_input(1, 1024, torch.float32)
    with pytest.raises(RuntimeError, match="idx_oob_fill_value must fit in int32"):
        topk(input, 8, idx_oob_fill_value=1 << 40)


def test_misaligned_caller_output_is_staged_without_overwrite():
    input = _aligned_input(1, 1024, torch.float32)
    input.copy_(torch.arange(1024, device=input.device))
    backing = torch.full((528,), -77, dtype=torch.int64, device=input.device)
    output = backing.as_strided((1, 513), (520, 1), 1)

    values, indices = topk(input, 513, sorted=True, output_idx=output)

    assert values is not None
    assert indices is output
    expected = torch.topk(input, 513)
    torch.testing.assert_close(indices, expected.indices, rtol=0, atol=0)
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    assert backing[0] == -77
    assert torch.all(backing[514:] == -77)


@pytest.mark.parametrize(
    "layout,error",
    [
        ("misaligned", "input address must be aligned"),
        ("missing-tail-padding", "input storage must include"),
    ],
)
def test_input_storage_validation(layout, error):
    dtype = torch.bfloat16
    width = 1024 // dtype.itemsize + 1
    stride = 2 * (1024 // dtype.itemsize)
    if layout == "misaligned":
        backing = torch.empty(stride + 1, dtype=dtype, device="cuda")
        input = backing.as_strided((1, width), (stride, 1), 1)
    else:
        input = torch.empty_strided((1, width), (stride, 1), dtype=dtype, device="cuda")

    with pytest.raises(RuntimeError, match=error):
        topk(input, 8)


def test_begin_is_rejected_by_cpp():
    input = _aligned_input(1, 1024, torch.float32)
    begin = torch.zeros(1, dtype=torch.int32, device=input.device)
    with pytest.raises(RuntimeError, match="`begin` is not supported"):
        topk(input, 8, begin=begin)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
