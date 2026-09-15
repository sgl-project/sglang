import sys

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pack_qkv_destination_major_cpu_is_bit_exact(dtype):
    torch.manual_seed(0)

    rows, world_size, global_heads, head_size = 17, 4, 12, 64

    q, k, v = (
        torch.randn(
            rows,
            global_heads,
            head_size,
            device="cpu",
            dtype=dtype,
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

    torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(
        q,
        k,
        v,
        world_size,
        actual,
    )

    assert torch.equal(actual, expected)


def test_pack_qkv_destination_major_cpu_validates_inputs():
    q = torch.empty(2, 4, 8, device="cpu", dtype=torch.bfloat16)

    # Q/K/V must have the same 3D shape.
    with pytest.raises(RuntimeError, match="same 3D shape"):
        output = torch.empty(
            2,
            2,
            2,
            24,
            device="cpu",
            dtype=torch.bfloat16,
        )

        torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(
            q,
            q[:, :-1],
            q,
            2,
            output,
        )

    # world_size must divide global_heads.
    with pytest.raises(RuntimeError, match="divide global_heads"):
        output = torch.empty(
            3,
            2,
            1,
            24,
            device="cpu",
            dtype=torch.bfloat16,
        )

        torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(q, q, q, 3, output)

    # Output shape must match
    # [world_size, rows, local_heads, 3 * head_size].
    with pytest.raises(RuntimeError, match="invalid output shape"):
        torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(
            q,
            q,
            q,
            2,
            torch.empty_like(q),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
