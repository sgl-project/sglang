"""Clover right convolution parity and read-only history contracts."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_right_convolution_matches_reference_and_preserves_history():
    block_size = 32
    from sglang.kernels.ops.mamba.clover_conv_triton import (
        clover_local_memory_conv,
    )

    torch.manual_seed(23)
    raw = torch.randn(block_size * 2, 128, device="cuda", dtype=torch.bfloat16)
    history = torch.randn(3, 128, 3, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(128, 1, 3, device="cuda", dtype=torch.bfloat16)
    before = history.clone()
    kw = dict(
        clean_conv_states=history,
        cache_indices=torch.tensor([2, 0], device="cuda"),
        conv_weights=weights,
        bias=None,
        activation="silu",
        block_size=block_size,
        active_valid_mask=torch.ones(block_size * 2, device="cuda", dtype=torch.bool),
        conv_mode="independent_symmetric",
        conv_right_weight=right,
    )
    actual = clover_local_memory_conv(raw, **kw)
    current = raw.reshape(2, block_size, 128).float()
    joined = torch.cat((history[[2, 0]].transpose(1, 2).float(), current), dim=1)
    expected = torch.zeros_like(current)
    for pos in range(block_size):
        expected[:, pos] = (joined[:, pos : pos + 4] * weights.T.float()).sum(1)
        for distance in range(1, min(4, block_size - pos)):
            expected[:, pos] += (
                current[:, pos + distance] * right[:, 0, distance - 1].float()
            )
    expected = torch.nn.functional.silu(expected).reshape_as(raw).to(raw.dtype)
    torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.03)
    torch.testing.assert_close(history, before, atol=0, rtol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
