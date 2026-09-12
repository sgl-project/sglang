import sys

import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import deepep_permute_triton_kernel
from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import (
    per_tensor_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _routing(num_tokens: int, topk: int):
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int64)
    invalid = (torch.arange(num_tokens * topk).reshape(num_tokens, topk) + 1) % 5 == 0
    topk_ids[invalid] = -1

    src2dst = torch.full((num_tokens, topk), -1, dtype=torch.int64)
    num_valid = int((~invalid).sum().item())
    src2dst[~invalid] = torch.randperm(num_valid, dtype=torch.int64)
    return topk_ids.cuda(), src2dst.cuda(), num_valid


@pytest.mark.parametrize("num_tokens", [1, 7, 33])
@pytest.mark.parametrize("hidden_size", [128, 512, 7168])
@pytest.mark.parametrize("topk", [1, 2, 8])
def test_deepep_quant_permute_matches_permute_then_quant(
    num_tokens: int,
    hidden_size: int,
    topk: int,
):
    torch.manual_seed(num_tokens * 1000 + hidden_size + topk)
    input = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    input[0, 0] = 500.0
    scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    topk_ids, src2dst, num_valid = _routing(num_tokens, topk)

    permuted = torch.empty(
        (num_valid, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    deepep_permute_triton_kernel[(num_tokens,)](
        input,
        permuted,
        src2dst,
        topk_ids,
        None,
        topk,
        hidden_size,
        BLOCK_SIZE=512,
    )
    flat_src2dst = src2dst.view(-1)
    valid = flat_src2dst >= 0
    source_rows = torch.arange(num_tokens, device="cuda").repeat_interleave(topk)
    expected_permuted = torch.empty_like(permuted)
    expected_permuted[flat_src2dst[valid]] = input[source_rows[valid]]
    assert torch.equal(permuted, expected_permuted)

    reference = torch.empty_like(permuted, dtype=torch.float8_e4m3fn)
    per_tensor_quant_fp8(permuted, reference, scale, is_static=True)

    fused = torch.empty_like(reference)
    deepep_permute_triton_kernel[(num_tokens,)](
        input,
        fused,
        src2dst,
        topk_ids,
        scale,
        topk,
        hidden_size,
        BLOCK_SIZE=512,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused.view(torch.uint8), reference.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
