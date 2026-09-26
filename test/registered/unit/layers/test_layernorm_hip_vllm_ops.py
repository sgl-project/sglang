"""RMSNorm forward_hip with aiter disabled, which runs on vLLM's in-place ops."""

import pytest
import torch

import sglang.srt.layers.layernorm as ln
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rms(x, weight, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def _vllm_rms_norm(out, input, weight, epsilon):
    out.copy_(_rms(input, weight, epsilon))


def _vllm_fused_add_rms_norm(input, residual, weight, epsilon):
    residual.add_(input)
    input.copy_(_rms(residual, weight, epsilon))


@pytest.fixture
def vllm_ops(monkeypatch):
    monkeypatch.setattr(ln, "_use_aiter", False)
    monkeypatch.setattr(ln, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(ln, "rms_norm", _vllm_rms_norm, raising=False)
    monkeypatch.setattr(
        ln, "fused_add_rms_norm", _vllm_fused_add_rms_norm, raising=False
    )


def _make_norm(norm_cls, hidden):
    norm = norm_cls(hidden, eps=1e-6)
    weight = torch.randn(hidden)
    loader = getattr(norm.weight, "weight_loader", None)
    if loader is not None:
        loader(norm.weight, weight)
    else:
        norm.weight.data.copy_(weight)
    return norm


@pytest.mark.parametrize("norm_cls", [ln.RMSNorm, ln.GemmaRMSNorm])
@pytest.mark.parametrize("post_add", [False, True])
def test_forward_hip_matches_native(vllm_ops, norm_cls, post_add):
    torch.manual_seed(0)
    norm = _make_norm(norm_cls, 64)
    x = torch.randn(4, 64)
    residual = torch.randn(4, 64)
    post = torch.randn(4, 64) if post_add else None
    x_in, residual_in = x.clone(), residual.clone()

    out, residual_out = norm.forward_hip(x, residual, post)
    ref_out, ref_residual = norm.forward_native(x.clone(), residual.clone(), post)

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(residual_out, ref_residual)
    torch.testing.assert_close(x, x_in)
    torch.testing.assert_close(residual, residual_in)
