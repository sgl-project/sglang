"""window_unpartition_add must stay bit-identical to unpartition + residual add,
for both BHWC and BCHW-permuted shortcut layouts."""

import sys

import pytest
import torch

from sglang.kernels.ops.elementwise.window_unpartition_add import (
    can_use_window_unpartition_add,
    window_unpartition_add,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sam import make_windows, window_unpartition_add_reference

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("batch,height,width", [(1, 40, 40), (8, 40, 40), (2, 64, 64)])
def test_window_unpartition_add_bit_identity(batch: int, height: int, width: int):
    torch.manual_seed(20260923)
    attention = torch.randn(
        batch, height, width, 768, device="cuda", dtype=torch.bfloat16
    )
    shortcut = torch.randn_like(attention)
    layouts = [shortcut]
    layouts.append(
        torch.randn(
            batch, 768, height, width, device="cuda", dtype=torch.bfloat16
        ).permute(0, 2, 3, 1)
    )
    windows = make_windows(attention, 14)
    for shortcut_layout in layouts:
        assert can_use_window_unpartition_add(windows, shortcut_layout, 14)
        expected = window_unpartition_add_reference(windows, shortcut_layout, 14)
        actual = window_unpartition_add(windows, shortcut_layout, 14)
        assert actual.stride() == shortcut_layout.stride()
        assert torch.equal(actual, expected)


def _inputs(dtype=torch.bfloat16, hidden=768, device="cuda"):
    shortcut = torch.randn(1, 40, 40, hidden, device=device, dtype=dtype)
    return make_windows(torch.randn_like(shortcut), 14), shortcut


@pytest.mark.parametrize(
    "case",
    [
        "fp16",
        "fp32",
        "cpu",
        "hidden_not_multiple_of_8",
        "unsupported_shortcut_layout",
        "non_contiguous_windows",
        "window_count_mismatch",
    ],
)
def test_unsupported_inputs_take_the_eager_path(case: str):
    """Inputs the kernel rejects must take the eager unpartition + add."""
    if case == "fp16":
        windows, shortcut = _inputs(dtype=torch.float16)
    elif case == "fp32":
        windows, shortcut = _inputs(dtype=torch.float32)
    elif case == "cpu":
        windows, shortcut = _inputs(device="cpu")
    elif case == "hidden_not_multiple_of_8":
        windows, shortcut = _inputs(hidden=12)
    elif case == "unsupported_shortcut_layout":
        windows, _ = _inputs()
        # [B, H, C, W] storage: neither BHWC nor BCHW-permuted.
        shortcut = torch.randn(
            1, 40, 768, 40, device="cuda", dtype=torch.bfloat16
        ).permute(0, 1, 3, 2)
    elif case == "non_contiguous_windows":
        windows, shortcut = _inputs()
        windows = windows.transpose(1, 2)
    else:
        windows, shortcut = _inputs()
        windows = windows[1:]
    assert not can_use_window_unpartition_add(windows, shortcut, 14)


def test_kernel_rejects_non_bf16_inputs():
    windows, shortcut = _inputs(dtype=torch.float16)
    with pytest.raises(Exception, match="dtype|Tensor match failed"):
        window_unpartition_add(windows, shortcut, 14)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
