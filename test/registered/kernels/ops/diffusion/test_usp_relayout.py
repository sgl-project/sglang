"""Bitwise tests for the generic Ulysses output head-merge fast path."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion.usp_relayout import (
    can_use_usp_merge_heads,
    usp_merge_heads,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


@pytest.mark.parametrize(
    "world,seq,batch,h_local,head_dim",
    [
        (4, 7936, 1, 14, 128),  # H3 768p production shape (Ulysses 4)
        (2, 64, 3, 4, 64),  # batched
        (4, 33, 2, 4, 100),  # scalar fallback inside the CUDA kernel
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_usp_merge_heads_bitwise(dtype, world, seq, batch, h_local, head_dim):
    generator = torch.Generator(device=DEVICE).manual_seed(4321)
    x = torch.randn(
        world,
        seq,
        batch,
        h_local,
        head_dim,
        dtype=dtype,
        device=DEVICE,
        generator=generator,
    )
    assert can_use_usp_merge_heads(x)
    out = usp_merge_heads(x)
    ref = x.permute(2, 1, 0, 3, 4).contiguous()
    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def test_usp_merge_heads_unsupported_inputs_use_exact_fallback():
    x = torch.randn(2, 4, 1, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    unsupported = [x.transpose(0, 1), x[:0]]

    for value in unsupported:
        assert not can_use_usp_merge_heads(value)
        assert torch.equal(
            usp_merge_heads(value), value.permute(2, 1, 0, 3, 4).contiguous()
        )

    with patch.object(torch.version, "hip", "6.3"):
        assert not can_use_usp_merge_heads(x)
        assert torch.equal(usp_merge_heads(x), x.permute(2, 1, 0, 3, 4).contiguous())


def test_usp_merge_heads_fast_path_rejects_wrong_rank():
    x = torch.randn(2, 4, 1, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    assert not can_use_usp_merge_heads(x[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
