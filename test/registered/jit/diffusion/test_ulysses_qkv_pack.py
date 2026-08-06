"""Bit-exact tests for the fused peer-first Ulysses QKV pack kernel."""

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.ulysses_qkv_pack import (
    fused_pack_peer_first_qkv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("shape", "world_size"),
    [
        ((1, 17, 24, 128), 2),
        ((2, 5, 8, 64), 4),
    ],
)
def test_fused_pack_peer_first_qkv_matches_torch(dtype, shape, world_size):
    torch.manual_seed(17)
    query = torch.randn(shape, dtype=dtype, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output_buffer = torch.empty(3 * query.numel(), dtype=dtype, device="cuda")

    output = fused_pack_peer_first_qkv(query, key, value, world_size, output_buffer)

    batch, sequence, global_heads, head_dim = shape
    local_heads = global_heads // world_size
    expected = torch.cat(
        tuple(
            tensor.unflatten(2, (world_size, local_heads)).permute(2, 0, 1, 3, 4)
            for tensor in (query, key, value)
        ),
        dim=-1,
    )
    assert output.shape == (
        world_size,
        batch,
        sequence,
        local_heads,
        3 * head_dim,
    )
    assert output.data_ptr() == output_buffer.data_ptr()
    assert torch.equal(output, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
