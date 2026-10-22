"""BF16 sparse top-k must resolve ties independently of warp arrival order."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("length", [512, 961, 4096, 16384])
def test_sparse_topk_ties_are_reproducible(length):
    scores = torch.ones(16, 16384, dtype=torch.bfloat16, device="cuda")
    lens = torch.full((16,), length, dtype=torch.int32, device="cuda")
    blocks = (
        torch.arange(2048, dtype=torch.int32, device="cuda")[None]
        .expand(16, -1)
        .contiguous()
    )
    output = torch.empty(16, 512, dtype=torch.int32, device="cuda")
    topk_transform_sparse(scores, lens, blocks, output)
    reference = output.clone()
    for _ in range(20):
        topk_transform_sparse(scores, lens, blocks, output)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert torch.all(output >= 0) and torch.all(output < length)
    assert torch.all(output.sort(-1).values[:, 1:] != output.sort(-1).values[:, :-1])
    scores[:, length:] = float("nan")
    topk_transform_sparse(scores, lens, blocks, output)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("length", [961, 4096, 16384])
def test_sparse_topk_selected_values_match_reference(length):
    torch.manual_seed(45)
    scores = torch.randn(16, 16384, dtype=torch.bfloat16, device="cuda")
    lens = torch.full((16,), length, dtype=torch.int32, device="cuda")
    blocks = (
        torch.arange(2048, dtype=torch.int32, device="cuda")[None]
        .expand(16, -1)
        .contiguous()
    )
    output = torch.empty(16, 512, dtype=torch.int32, device="cuda")
    topk_transform_sparse(scores, lens, blocks, output)
    selected = scores.gather(1, output.long()).sort(-1).values
    expected = scores[:, :length].topk(512).values.sort(-1).values
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [961, 4096, 8192])
def test_fp32_ragged_topk_has_reproducible_order(length):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    torch.manual_seed(75)
    scores = torch.randn(16, (length + 3) // 4 * 4, dtype=torch.float32, device="cuda")
    lens = torch.full((16,), length, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(16, dtype=torch.int32, device="cuda")
    indices = torch.empty(16, 512, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(scores, lens, out_offsets=offsets, out_indices=indices)
    reference = indices.clone()
    for _ in range(20):
        topk_transform_ragged_v2(scores, lens, out_offsets=offsets, out_indices=indices)
        torch.testing.assert_close(indices, reference, rtol=0, atol=0)
    actual = scores.gather(1, indices.long()).sort(-1).values
    expected = scores[:, :length].topk(512).values.sort(-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
