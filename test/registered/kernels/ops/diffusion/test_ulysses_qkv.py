import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
    pack_qkv_destination_major,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pack_qkv_destination_major_is_bit_exact(dtype):
    torch.manual_seed(0)
    rows, world_size, global_heads, head_size = 17, 4, 12, 64
    q, k, v = (
        torch.randn(rows, global_heads, head_size, device="cuda", dtype=dtype)
        for _ in range(3)
    )

    local_heads = global_heads // world_size
    expected = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_size,
        device="cuda",
        dtype=dtype,
    )
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.view(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )
        expected[..., index * head_size : (index + 1) * head_size].copy_(shards)

    actual = pack_qkv_destination_major(q, k, v, world_size)
    assert torch.equal(actual, expected)


def test_pack_qkv_destination_major_validates_inputs():
    q = torch.empty(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same 3D shape"):
        pack_qkv_destination_major(q, q[:, :-1], q, 2)
    with pytest.raises(ValueError, match="divide global_heads"):
        pack_qkv_destination_major(q, q, q, 3)
    with pytest.raises(ValueError, match="expected shape"):
        pack_qkv_destination_major(q, q, q, 2, out=torch.empty_like(q))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
