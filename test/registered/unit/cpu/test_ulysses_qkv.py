import sys

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pack_qkv_destination_major_cpu_is_bit_exact(dtype):
    torch.manual_seed(0)

    rows, world_size, global_heads, head_size = 17, 4, 12, 64

    q, k, v = (
        torch.randn(global_heads, rows, head_size, device="cpu", dtype=dtype).transpose(
            0, 1
        )
        for _ in range(3)
    )

    local_heads = global_heads // world_size

    expected = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_size,
        device="cpu",
        dtype=dtype,
    )

    for index, tensor in enumerate((q, k, v)):
        shards = tensor.view(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )

        expected[..., index * head_size : (index + 1) * head_size].copy_(shards)

    actual = torch.empty_like(expected)

    torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(q, k, v, world_size, actual)

    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
