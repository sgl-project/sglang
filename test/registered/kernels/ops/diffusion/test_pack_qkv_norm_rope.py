# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
    can_use_pack_qkv_norm_rope,
    pack_qkv_destination_major,
    pack_qkv_norm_rope_destination_major,
)
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _eager_heads(x, weight, rope, eps):
    """Eager RMSNorm then interleaved RoPE, ``[rows, heads, 128]`` with ``[rows, 64]``."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight
    z = torch.view_as_complex(normed.float().unflatten(-1, (-1, 2)))
    return torch.view_as_real(z * rope[:, None]).flatten(-2).to(x.dtype)


def _fp64_heads(x, weight, rope, eps):
    x = x.double()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.double()
    z = torch.view_as_complex(normed.unflatten(-1, (-1, 2)))
    return torch.view_as_real(z * rope.to(torch.complex128)[:, None]).flatten(-2)


def _projection_segment(rows, heads, activations):
    """Q/K/V as strided views of one fused projection, with this segment's weights."""
    qkv = torch.randn(rows, 3 * heads * 128, device="cuda", dtype=torch.bfloat16)
    if activations == "outliers":
        # DiT-style: a constant offset plus a few large outlier channels.
        qkv = qkv + 4.0
        qkv[:, torch.arange(8, device="cuda") * 47] *= 32
    q, k, v = qkv.unflatten(-1, (3, heads, 128)).unbind(1)
    q_weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    k_weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    return q, k, v, q_weight, k_weight


def _unpack(packed):
    """Destination-major ``[W, rows, h_local, 3 * 128]`` back to ``[rows, H, 128]`` x3."""
    world, rows, local_heads, _ = packed.shape
    return [
        packed[..., i * 128 : (i + 1) * 128]
        .permute(1, 0, 2, 3)
        .reshape(rows, world * local_heads, 128)
        for i in range(3)
    ]


@pytest.mark.parametrize(
    "segment_rows,heads,world",
    [((64, 864), 48, 8), ((928,), 48, 8), ((3, 5), 8, 2), ((7,), 4, 1)],
)
@pytest.mark.parametrize("activations", ["randn", "outliers"])
def test_pack_matches_eager_norm_rope_then_pack(
    segment_rows, heads, world, activations
):
    torch.manual_seed(0)
    segments = [_projection_segment(rows, heads, activations) for rows in segment_rows]
    total = sum(segment_rows)
    angle = torch.rand(total, 64, device="cuda") * 6.28
    rope = torch.polar(torch.ones_like(angle), angle)
    eps = 1e-6
    assert all(not t.is_contiguous() for segment in segments for t in segment[:3])
    rope_real = torch.view_as_real(rope).contiguous()
    assert can_use_pack_qkv_norm_rope(segments, rope_real, world)

    actual = pack_qkv_norm_rope_destination_major(segments, rope_real, eps, world)

    expected = {"eager": [[], [], []], "fp64": [[], []]}
    start = 0
    for (q, k, v, q_weight, k_weight), rows in zip(segments, segment_rows):
        table = rope[start : start + rows]
        for i, (x, weight) in enumerate(((q, q_weight), (k, k_weight))):
            expected["eager"][i].append(_eager_heads(x, weight, table, eps))
            expected["fp64"][i].append(_fp64_heads(x, weight, table, eps))
        expected["eager"][2].append(v)
        start += rows
    eager_q, eager_k, eager_v = (torch.cat(t) for t in expected["eager"])
    fp64_q, fp64_k = (torch.cat(t) for t in expected["fp64"])
    actual_q, actual_k, actual_v = _unpack(actual)

    # The packed layout is the one the unfused pack writes; V is only moved.
    assert torch.equal(
        actual, pack_qkv_destination_major(actual_q, actual_k, actual_v, world)
    )
    assert torch.equal(actual_v, eager_v)
    for name, fp64, eager, candidate in (
        ("q", fp64_q, eager_q, actual_q),
        ("k", fp64_k, eager_k, actual_k),
    ):
        assert_error_no_worse_than_reference(
            reference_fp64=fp64,
            baseline=eager,
            candidate=candidate,
            label=f"{name} {segment_rows} {heads}x{world} {activations}",
        )


def test_pack_rejects_unsupported_inputs():
    torch.manual_seed(0)
    q, k, v, q_weight, k_weight = _projection_segment(6, 8, "randn")
    rope = torch.view_as_real(
        torch.ones(6, 64, device="cuda", dtype=torch.complex64)
    ).contiguous()
    assert can_use_pack_qkv_norm_rope([(q, k, v, q_weight, k_weight)], rope, 4)
    # heads must divide across ranks
    assert not can_use_pack_qkv_norm_rope([(q, k, v, q_weight, k_weight)], rope, 3)
    # the rotation table must cover every row
    assert not can_use_pack_qkv_norm_rope([(q, k, v, q_weight, k_weight)], rope[:5], 4)
    # heads must be 128 apart
    assert not can_use_pack_qkv_norm_rope(
        [(q.transpose(0, 1), k, v, q_weight, k_weight)], rope, 4
    )
    assert not can_use_pack_qkv_norm_rope(
        [(q.float(), k, v, q_weight, k_weight)], rope, 4
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
