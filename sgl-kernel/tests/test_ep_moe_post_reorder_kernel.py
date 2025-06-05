import itertools

import pytest
import torch
from sgl_kernel import ep_moe_post_reorder

from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel


def create_test_tensors(
    batch_size: int,
    hidden_size: int,
    topk: int,
    start_expert_id: int,
    end_expert_id: int,
    dtype: torch.dtype,
    device: torch.device,
):
    down_output = torch.randn(
        batch_size * topk, hidden_size, dtype=dtype, device=device
    )

    # Ensure src2dst has no duplicate destinations to avoid race conditions
    total_tokens = batch_size * topk
    dst_indices = torch.randperm(total_tokens, device=device, dtype=torch.int32)
    src2dst = dst_indices.view(batch_size, topk)

    topk_ids = torch.randint(
        start_expert_id,
        end_expert_id + 1,
        (batch_size, topk),
        dtype=torch.int32,
        device=device,
    )

    topk_weights = torch.rand(batch_size, topk, dtype=dtype, device=device)

    return down_output, src2dst, topk_ids, topk_weights


def run_cuda_kernel(
    down_output: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    topk: int,
):
    ep_moe_post_reorder(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
    )
    return output


def run_triton_kernel(
    down_output: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    topk: int,
    hidden_size: int,
):
    batch_size = down_output.size(0)
    block_size = 512

    post_reorder_triton_kernel[(batch_size,)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
        hidden_size,
        block_size,
    )
    return output


def assert_close(a, b):
    a32, b32 = a.float(), b.float()
    if a.dtype is torch.float16:
        torch.testing.assert_close(a32, b32, rtol=1e-5, atol=1e-2)
    elif a.dtype is torch.bfloat16:
        torch.testing.assert_close(a32, b32, rtol=1e-4, atol=1e-1)
    else:
        torch.testing.assert_close(a32, b32, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch_size,hidden_size,topk",
    list(itertools.product([32, 64], [128, 256, 512], [2, 4, 8])),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ep_moe_post_reorder_vs_triton(
    batch_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
):
    device = torch.device("cuda")
    start_expert_id = 0
    end_expert_id = 15

    (
        down_output,
        src2dst,
        topk_ids,
        topk_weights,
    ) = create_test_tensors(
        batch_size,
        hidden_size,
        topk,
        start_expert_id,
        end_expert_id,
        dtype,
        device,
    )

    output_cuda = torch.empty(batch_size, hidden_size, dtype=dtype, device=device)
    output_triton = torch.empty(batch_size, hidden_size, dtype=dtype, device=device)

    cuda_output = run_cuda_kernel(
        down_output,
        output_cuda,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
    )

    triton_output = run_triton_kernel(
        down_output,
        output_triton,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
        hidden_size,
    )

    assert_close(cuda_output, triton_output)


if __name__ == "__main__":
    pytest.main([__file__])
