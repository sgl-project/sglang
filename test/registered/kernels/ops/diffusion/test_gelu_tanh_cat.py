"""Exact GELU rounding, layout, validation and replay for the concat fusion."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fused_gelu_tanh_cat,
    fused_gelu_tanh_cat,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
def test_all_finite_bf16_values():
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    mlp = values[torch.isfinite(values)].view(1, -1, 8)
    attn = torch.randn_like(mlp)
    actual = fused_gelu_tanh_cat(attn, mlp)
    expected = torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=-1)
    # Compare bit patterns so signed zero is covered too.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "batch,tokens,a,m", [(2, 17, 8, 24), (1, 859, 1536, 6144), (1, 9233, 3072, 12288)]
)
@torch.inference_mode()
def test_shapes_and_unchanged_inputs(batch, tokens, a, m):
    attn = torch.randn(batch, tokens, a, device="cuda", dtype=torch.bfloat16)
    mlp = torch.randn(batch, tokens, m, device="cuda", dtype=torch.bfloat16)
    saved_attn, saved_mlp = attn.clone(), mlp.clone()
    assert can_use_fused_gelu_tanh_cat(attn, mlp)
    actual = fused_gelu_tanh_cat(attn, mlp)
    assert actual.is_contiguous()
    assert torch.equal(
        actual, torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=-1)
    )
    assert torch.equal(attn, saved_attn)
    assert torch.equal(mlp, saved_mlp)


@torch.inference_mode()
def test_changed_inputs_in_graph_replay():
    attn = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16)
    mlp = torch.randn(2, 17, 256, device="cuda", dtype=torch.bfloat16)
    fused_gelu_tanh_cat(attn, mlp)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_gelu_tanh_cat(attn, mlp)
    attn.neg_()
    mlp.add_(0.75)
    graph.replay()
    assert torch.equal(
        actual, torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=-1)
    )


@torch.inference_mode()
def test_unsupported_inputs():
    a = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
    bad_pairs = [
        (a.cpu(), a.cpu()),
        (a.float(), a.float()),
        (a, a.float()),
        (a, a[:1]),
        (a, a[..., :63]),
        (a, a.transpose(0, 1)),
        (a[:0], a[:0]),
    ]
    for attn, mlp in bad_pairs:
        assert not can_use_fused_gelu_tanh_cat(attn, mlp)
    unaligned = torch.empty(a.numel() + 1, device="cuda", dtype=a.dtype)[1:].view_as(a)
    assert not can_use_fused_gelu_tanh_cat(a, unaligned)
    with pytest.raises(RuntimeError, match="aligned"):
        fused_gelu_tanh_cat(a, unaligned)
    with pytest.raises(RuntimeError):
        fused_gelu_tanh_cat(a, a.float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
