"""Tests for the Qwen3.5 GDN fused split/reshape/cat kernel, with dense inputs and with
column slices of a wider projection."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=6, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
HEAD_K_DIM = HEAD_V_DIM = 128


def reference_split_reshape_cat(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v):
    k_dim = num_heads_qk * HEAD_K_DIM
    v_dim = num_heads_v * HEAD_V_DIM
    query, key, value, z = mixed_qkvz.split([k_dim, k_dim, v_dim, v_dim], dim=-1)
    b, a = mixed_ba.split([num_heads_v, num_heads_v], dim=-1)
    z = z.reshape(z.size(0), -1, HEAD_V_DIM)
    mixed_qkv = torch.cat(
        [x.reshape(x.shape[0], -1) for x in (query, key, value)], dim=-1
    )
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def build_inputs(seq_len, num_heads_qk, num_heads_v, merged):
    """The qkvz/ba pair, either as separate tensors or as slices of one projection."""
    qkvz_width = 2 * num_heads_qk * HEAD_K_DIM + 2 * num_heads_v * HEAD_V_DIM
    ba_width = 2 * num_heads_v
    torch.manual_seed(0)
    if not merged:
        return (
            torch.randn(seq_len, qkvz_width, dtype=torch.bfloat16, device=DEVICE),
            torch.randn(seq_len, ba_width, dtype=torch.bfloat16, device=DEVICE),
        )
    # 64 extra columns stand in for the alignment padding the merged layer carries.
    projected = torch.randn(
        seq_len, qkvz_width + ba_width + 64, dtype=torch.bfloat16, device=DEVICE
    )
    return projected[:, :qkvz_width], projected[:, qkvz_width : qkvz_width + ba_width]


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("seq_len", [1, 4, 37, 256])
@pytest.mark.parametrize("num_heads_qk,num_heads_v", [(16, 32), (8, 32), (4, 16)])
def test_split_reshape_cat_contiguous(seq_len, num_heads_qk, num_heads_v, merged):
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    mixed_qkvz, mixed_ba = build_inputs(seq_len, num_heads_qk, num_heads_v, merged)
    assert (mixed_qkvz.stride(0) != mixed_qkvz.shape[1]) == merged

    out = fused_qkvzba_split_reshape_cat_contiguous(
        mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, HEAD_K_DIM, HEAD_V_DIM
    )
    ref = reference_split_reshape_cat(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v)
    for got, want in zip(out, ref):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
