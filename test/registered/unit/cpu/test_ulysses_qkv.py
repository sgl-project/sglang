import sys
from unittest.mock import patch

import pytest
import sgl_kernel  # noqa: F401
import torch

import sglang.multimodal_gen.runtime.layers.usp as usp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_3d_cpu_packed_qkv_dispatch(dtype):
    world_size = 2
    s_local, h_global, d = 4, 8, 32
    h_local = h_global // world_size

    q = torch.randn(s_local, h_global, d, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def identity_a2a(x, role=None):
        return x

    def accelerator_kernel_must_not_run(*args, **kwargs):
        raise AssertionError("Accelerator QKV kernel called on CPU")

    with (
        torch.no_grad(),
        patch.object(usp, "get_ulysses_parallel_world_size", return_value=world_size),
        patch.object(usp, "_usp_all_to_all_single", side_effect=identity_a2a),
        patch.object(
            usp,
            "pack_qkv_destination_major",
            side_effect=accelerator_kernel_must_not_run,
        ),
    ):
        q_out, k_out, v_out = usp._usp_input_all_to_all_packed_qkv(q, k, v)

    expected = torch.stack(
        [
            torch.cat(
                (
                    q[:, r * h_local : (r + 1) * h_local],
                    k[:, r * h_local : (r + 1) * h_local],
                    v[:, r * h_local : (r + 1) * h_local],
                ),
                dim=-1,
            )
            for r in range(world_size)
        ],
        dim=0,
    ).reshape(s_local * world_size, h_local, 3 * d)

    eq, ek, ev = expected.split(d, dim=-1)
    assert torch.equal(q_out, eq)
    assert torch.equal(k_out, ek)
    assert torch.equal(v_out, ev)


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

    result = torch.ops.sgl_kernel.pack_qkv_destination_major_cpu(
        q, k, v, world_size, out=actual
    )

    assert torch.equal(result, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
