import itertools

import pytest
import torch
from sgl_kernel import ep_moe_silu_and_mul

from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_triton_kernel


def create_test_tensors(
    total_tokens: int,
    hidden_size: int,
    start_expert_id: int,
    end_expert_id: int,
    dtype: torch.dtype,
    device: torch.device,
):
    gateup_output = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    reorder_topk_ids = torch.randint(
        start_expert_id,
        end_expert_id + 1,
        (total_tokens,),
        dtype=torch.int32,
        device=device,
    )

    num_experts = end_expert_id - start_expert_id + 1
    scales = torch.rand(num_experts, dtype=torch.float32, device=device) * 0.8 + 0.5

    half_hidden = hidden_size // 2
    down_input = torch.empty(total_tokens, half_hidden, dtype=dtype, device=device)

    return gateup_output, down_input, reorder_topk_ids, scales


def run_cuda_kernel(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    reorder_topk_ids: torch.Tensor,
    scales: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
):
    ep_moe_silu_and_mul(
        gateup_output,
        down_input,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
    )
    return down_input


def run_triton_kernel(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    reorder_topk_ids: torch.Tensor,
    scales: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    hidden_size: int,
):
    total_tokens = gateup_output.size(0)
    block_size = 512

    silu_and_mul_triton_kernel[(total_tokens,)](
        gateup_output,
        down_input,
        hidden_size,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
        block_size,
    )
    return down_input


@pytest.mark.parametrize(
    "total_tokens,hidden_size",
    list(itertools.product([32, 256, 1024], [128, 256, 512])),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ep_moe_silu_and_mul_vs_triton(
    total_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    device = torch.device("cuda")
    start_expert_id = 0
    end_expert_id = 15

    (
        gateup_output,
        _,
        reorder_topk_ids,
        scales,
    ) = create_test_tensors(
        total_tokens,
        hidden_size,
        start_expert_id,
        end_expert_id,
        dtype,
        device,
    )

    down_input_cuda = torch.empty(
        total_tokens, hidden_size // 2, dtype=dtype, device=device
    )
    down_input_triton = torch.empty_like(down_input_cuda)

    cuda_output = run_cuda_kernel(
        gateup_output,
        down_input_cuda,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
    )

    triton_output = run_triton_kernel(
        gateup_output,
        down_input_triton,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
        hidden_size,
    )

    torch.testing.assert_close(
        cuda_output,
        triton_output,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
