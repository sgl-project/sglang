import sys

import pytest
import torch

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_contiguous_supported,
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


# (num_k_heads, num_v_heads, head_k_dim, head_v_dim): pow2 group widths
# (ratios 1/2/4) take the kernel's wide-block path, non-pow2 (3/5/6) the
# per-head loop. Ratio 3 is Qwen3.6-27B's GDN (48/16).
SHAPES = [
    (8, 8, 64, 32),
    (8, 16, 64, 64),
    (16, 48, 128, 128),
    (4, 16, 128, 128),
    (4, 20, 64, 64),
    (4, 24, 128, 128),
]


def _reference_split(qkvz, ba, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    q, k, v, z = qkvz.split(
        [
            num_k_heads * head_k_dim,
            num_k_heads * head_k_dim,
            num_v_heads * head_v_dim,
            num_v_heads * head_v_dim,
        ],
        dim=-1,
    )
    b, a = ba.split([num_v_heads, num_v_heads], dim=-1)
    mixed_qkv = torch.cat([q, k, v], dim=-1)
    return mixed_qkv, z, b, a


@pytest.mark.parametrize("num_k_heads,num_v_heads,head_k_dim,head_v_dim", SHAPES)
@pytest.mark.parametrize("seq_len", [1, 7, 64, 333])
def test_fused_qkvzba_split_contiguous(
    num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int, seq_len: int
) -> None:
    torch.manual_seed(0)
    qkvz_dim = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
    qkvz = torch.randn(seq_len, qkvz_dim, dtype=torch.bfloat16, device="cuda")
    ba = torch.randn(seq_len, 2 * num_v_heads, dtype=torch.bfloat16, device="cuda")

    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz, ba, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    ref_qkv, ref_z, ref_b, ref_a = _reference_split(
        qkvz, ba, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )

    # The kernel is a pure copy/reshape — outputs must be bit-exact.
    assert torch.equal(mixed_qkv.reshape(seq_len, -1), ref_qkv)
    assert torch.equal(z.reshape(seq_len, -1), ref_z)
    assert torch.equal(b.reshape(seq_len, -1), ref_b)
    assert torch.equal(a.reshape(seq_len, -1), ref_a)


def test_supported_shapes() -> None:
    for num_k_heads, num_v_heads, head_k_dim, head_v_dim in SHAPES:
        assert fused_qkvzba_split_contiguous_supported(
            num_k_heads, num_v_heads, head_k_dim, head_v_dim
        )
    # inexact ratio
    assert not fused_qkvzba_split_contiguous_supported(16, 40, 128, 128)
    # non-power-of-2 head dims need the torch fallback (tl.arange widths)
    assert not fused_qkvzba_split_contiguous_supported(16, 48, 128, 96)
    assert not fused_qkvzba_split_contiguous_supported(16, 48, 96, 128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
