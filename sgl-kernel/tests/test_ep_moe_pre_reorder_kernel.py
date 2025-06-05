import itertools

import pytest
import torch
from sgl_kernel import ep_moe_pre_reorder

from sglang.srt.layers.moe.ep_moe.kernels import pre_reorder_triton_kernel


def create_test_tensors(
    batch_size: int,
    hidden_size: int,
    topk: int,
    start_expert_id: int,
    end_expert_id: int,
    dtype: torch.dtype,
    device: torch.device,
    use_per_token_if_dynamic: bool = True,
):
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

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

    if use_per_token_if_dynamic:
        a1_scales = (
            torch.rand(batch_size, dtype=torch.float32, device=device) * 0.8 + 0.2
        )
    else:
        a1_scales = (
            torch.rand(
                end_expert_id - start_expert_id + 1, dtype=torch.float32, device=device
            )
            * 0.8
            + 0.2
        )

    return input_tensor, src2dst, topk_ids, a1_scales


def run_cuda_kernel(
    input_tensor: torch.Tensor,
    gateup_input: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scales: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    topk: int,
    use_per_token_if_dynamic: bool,
):
    ep_moe_pre_reorder(
        input_tensor,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        use_per_token_if_dynamic,
    )
    return gateup_input


def run_triton_kernel(
    input_tensor: torch.Tensor,
    gateup_input: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scales: torch.Tensor,
    start_expert_id: int,
    end_expert_id: int,
    topk: int,
    hidden_size: int,
    use_per_token_if_dynamic: bool,
):
    batch_size = input_tensor.size(0)
    block_size = 512

    pre_reorder_triton_kernel[(batch_size,)](
        input_tensor,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        hidden_size,
        block_size,
        use_per_token_if_dynamic,
    )
    return gateup_input


@pytest.mark.parametrize(
    "batch_size,hidden_size,topk",
    list(itertools.product([32, 64, 128], [512, 1024, 2048], [4, 8])),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_per_token_if_dynamic", [True, False])
def test_ep_moe_pre_reorder_vs_triton(
    batch_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_per_token_if_dynamic: bool,
):
    device = torch.device("cuda")
    start_expert_id = 0
    end_expert_id = 15

    (
        input_tensor,
        src2dst,
        topk_ids,
        a1_scales,
    ) = create_test_tensors(
        batch_size,
        hidden_size,
        topk,
        start_expert_id,
        end_expert_id,
        dtype,
        device,
        use_per_token_if_dynamic,
    )

    gateup_input_cuda = torch.empty(
        batch_size * topk, hidden_size, dtype=dtype, device=device
    )
    gateup_input_triton = torch.empty(
        batch_size * topk, hidden_size, dtype=dtype, device=device
    )

    cuda_output = run_cuda_kernel(
        input_tensor,
        gateup_input_cuda,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        use_per_token_if_dynamic,
    )

    triton_output = run_triton_kernel(
        input_tensor,
        gateup_input_triton,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        hidden_size,
        use_per_token_if_dynamic,
    )

    torch.testing.assert_close(
        cuda_output.float(),
        triton_output.float(),
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
