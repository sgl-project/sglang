"""Edge's strided GQA views preserve the split BF16 norm/RoPE and UND cache."""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.cosmos3video as cosmos3
from sglang.kernels.ops.activation.activation import relu2
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    _apply_qwen3_qk_norm_rope_pack_kv,
    _apply_qwen3_qk_norm_rope_split,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "batch,tokens,prefix", [(1, 400, 32), (1, 1024, 64), (2, 401, 33)]
)
@torch.inference_mode()
def test_edge_qk_rope_pack_matches_split(batch, tokens, prefix):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Split-path bitwise parity covers the newly enabled Hopper path")
    torch.manual_seed(42)
    qkv = torch.randn(batch, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[:, :, :16], qkv[:, :, 16:24], qkv[:, :, 24:]
    k_und = torch.randn(batch, prefix, 8, 128, device="cuda", dtype=torch.bfloat16)
    v_und = torch.randn_like(k_und)
    before = qkv.clone(), k_und.clone(), v_und.clone()
    q_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    q_norm.weight.copy_(torch.rand_like(q_norm.weight) + 0.5)
    k_norm.weight.copy_(torch.rand_like(k_norm.weight) + 0.5)
    angles = torch.randn(batch * tokens, 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(torch.bfloat16)
    positions = torch.arange(batch * tokens, device="cuda")
    q_ref, k_ref = _apply_qwen3_qk_norm_rope_split(
        q, k, q_norm, k_norm, 128, cache.float()
    )
    k_ref = torch.cat((k_und, k_ref), dim=1)
    v_ref = torch.cat((v_und, v), dim=1)
    q_out, k_out, v_out = _apply_qwen3_qk_norm_rope_pack_kv(
        q,
        k,
        v,
        k_und,
        v_und,
        q_norm,
        k_norm,
        128,
        cache,
        positions,
        round_norm_before_rope=True,
    )
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)
    # The in-place Q/K suffix belongs to this forward; V and cached UND K/V
    # must remain reusable, including Edge's separately normalized UND keys.
    assert torch.equal(v, before[0][:, :, 24:])
    assert torch.equal(k_und, before[1])
    assert torch.equal(v_und, before[2])


class _IdentityProjection(nn.Module):
    def forward(self, x):
        return x, None


def _activation_only_mlp():
    mlp = cosmos3.Cosmos3DenseMLP.__new__(cosmos3.Cosmos3DenseMLP)
    nn.Module.__init__(mlp)
    mlp.up_proj = _IdentityProjection()
    mlp.down_proj = _IdentityProjection()
    return mlp


@torch.inference_mode()
def test_edge_relu2_all_finite_bfloat16_encodings():
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)].cuda().reshape(1, -1)
    expected = F.relu(values)
    expected = expected * expected
    with patch.object(cosmos3, "relu2", wraps=relu2) as fused:
        actual = _activation_only_mlp()(values)
    fused.assert_called_once()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    mlp = _activation_only_mlp()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = mlp(values)
    values.neg_()
    graph.replay()
    expected = F.relu(values)
    expected = expected * expected
    assert torch.equal(replayed.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tokens", [400, 8190])
@torch.inference_mode()
def test_edge_relu2_native_shapes(tokens):
    values = torch.randn(1, tokens, 9216, device="cuda", dtype=torch.bfloat16)
    before = values.clone()
    expected = F.relu(values)
    expected = expected * expected
    with patch.object(cosmos3, "relu2", wraps=relu2) as fused:
        actual = _activation_only_mlp()(values)
    fused.assert_called_once()
    assert torch.equal(actual, expected)
    assert torch.equal(values, before)


def test_edge_relu2_grad_falls_back_to_torch():
    values = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    with patch.object(
        cosmos3, "relu2", side_effect=AssertionError("unexpected fusion")
    ):
        actual = _activation_only_mlp()(values)
    actual.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.0, 0.0, 6.0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
