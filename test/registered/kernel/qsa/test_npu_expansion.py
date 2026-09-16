"""Exact comparisons for stable NPU block expansion and graph replay."""

import pytest
import torch

from sglang.srt.hardware_backend.npu.kernels.qwen3_8_flash_next.expansion import (
    can_run_block_expansion,
    expand_blocks,
)
from sglang.srt.layers.attention.qsa import kernel
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,ratio,topk",
    [(0, 4, 17), (1, 1, 1), (8, 3, 17), (32, 4, 2048),
     (128, 8, 4097), (4, 8, 8185)],
)
def test_exact_expansion(rows, ratio, topk, dtype):
    torch.manual_seed(73)
    width = (topk + ratio - 1) // ratio
    blocks = torch.randint(-3, 100, (rows, width * 2), dtype=dtype)[:, ::2]
    positions = torch.arange(rows * 2, dtype=dtype)[::2] - 5
    lengths = torch.arange(rows * 2, dtype=dtype)[::2] * 7
    # Preserve non-contiguous strides on device, including the column stride.
    blocks = blocks.t().contiguous().to("npu").t()
    positions = torch.stack((positions, positions), dim=1).to("npu")[:, 0]
    lengths = torch.stack((lengths, lengths), dim=1).to("npu")[:, 0]
    args = blocks, positions, lengths, ratio, topk
    assert can_run_block_expansion(*args)
    expected = kernel.torch_expand_qsa_block_indices(*args)
    actual = expand_blocks(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_holes_duplicates_large_indices_and_graph():
    blocks = torch.tensor(
        [
            [3, -1, 0, 3, 2],
            [-1, -2, -1, -1, -1],
            [2**30, 0, 1, 2**29, -1],
        ],
        device="npu",
        dtype=torch.int64,
    )
    positions = torch.tensor([19, 7, 2**32], device="npu", dtype=torch.int64)
    lengths = torch.tensor([14, 0, 2**34], device="npu", dtype=torch.int64)
    args = blocks, positions, lengths, 4, 19
    for _ in range(2):
        expand_blocks(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = expand_blocks(*args)
    for length in (0, 1, 14, 2**34):
        lengths.fill_(length)
        positions.sub_(1)
        blocks[0, 1] = 2 if length else -1
        graph.replay()
        torch.npu.synchronize()
        expected = kernel.torch_expand_qsa_block_indices(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_dispatch_and_fallback(monkeypatch):
    args = (
        torch.tensor([[2, -1, 0]], device="npu", dtype=torch.int32),
        torch.tensor([9], device="npu"),
        torch.tensor([10], device="npu"),
        4,
        9,
    )
    expected = kernel.torch_expand_qsa_block_indices(*args)
    original = kernel.torch_expand_qsa_block_indices

    def forbidden(*args):
        raise AssertionError("unexpected Torch fallback")

    monkeypatch.setattr(kernel, "torch_expand_qsa_block_indices", forbidden)
    actual = kernel.expand_qsa_block_indices(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    monkeypatch.setattr(kernel, "torch_expand_qsa_block_indices", original)
    cpu_positions = (args[0], args[1].cpu(), *args[2:])
    assert not can_run_block_expansion(*cpu_positions)
    actual = kernel.expand_qsa_block_indices(*cpu_positions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 3, 4, 8])
def test_mixed_integer_types_and_broadcast(ratio):
    blocks = torch.tensor(
        [[0, -1, 2**30, 2**31 - 1, 3]], device="npu", dtype=torch.int32
    ).expand(4, -1)
    positions = torch.tensor([-5, 15, 2**31, 2**34], device="npu")
    lengths = torch.tensor([0, 17, 2**31 - 1, 2**35], device="npu")
    args = blocks, positions, lengths, ratio, 5 * ratio - (ratio > 1)
    torch.testing.assert_close(
        expand_blocks(*args), kernel.torch_expand_qsa_block_indices(*args),
        atol=0, rtol=0,
    )


def test_unsupported_width_keeps_reference():
    args = (
        torch.zeros((1, 1025), device="npu", dtype=torch.int32),
        torch.tensor([20], device="npu"),
        torch.tensor([21], device="npu"),
        1,
        1025,
    )
    assert not can_run_block_expansion(*args)
    with pytest.raises(ValueError, match="Unsupported NPU"):
        expand_blocks(*args)
    torch.testing.assert_close(
        kernel.expand_qsa_block_indices(*args),
        kernel.torch_expand_qsa_block_indices(*args), atol=0, rtol=0,
    )
