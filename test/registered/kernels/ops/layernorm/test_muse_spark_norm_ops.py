"""Muse Spark fused norm ops: Triton/JIT backends against their torch reference."""

import pytest
import torch

from sglang.kernels.ops.layernorm.muse_spark_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.layernorm.muse_spark_norm_residual import (
    FusedPostNormResidualPreNorm,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.muse_spark.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BACKENDS = [KernelBackend.TRITON, KernelBackend.JIT]


def _bf16_module(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(0.1 * torch.randn_like(p))
    return module.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("num_tokens", [1, 9, 300])
@pytest.mark.parametrize(
    "has_norm, has_weight", [(True, False), (True, True), (False, False)]
)
@pytest.mark.parametrize("gate_groups", [None, 1, 4])
def test_rms_norm_gate_matches_native(
    backend, num_tokens, has_norm, has_weight, gate_groups
):
    torch.manual_seed(num_tokens)
    head_dim, num_heads = 64, 16
    o_norm = (
        _bf16_module(
            MuseSpark1xRMSNorm(
                head_dim, eps=1e-6, has_weight=has_weight, zero_centered_gamma=True
            )
        )
        if has_norm
        else None
    )
    op = FusedRMSNormGate(o_norm, use_o_gate=gate_groups is not None, head_dim=head_dim)
    width = num_heads * head_dim
    x = torch.randn(num_tokens, width, device="cuda", dtype=torch.bfloat16)
    gate = None
    if gate_groups is not None:
        # A strided grouped view, as the fused QKV-O projection hands it over.
        packed = torch.randn(
            num_tokens,
            gate_groups,
            width // gate_groups + 32,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = packed[..., : width // gate_groups]
        if gate_groups == 1:
            gate = gate.reshape(num_tokens, width)
    ref = op.forward_native(x, gate)
    out = op(x, gate, backend=backend)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def _boundary(post_input_scale, hidden):
    post_cls = (
        MuseSpark1xRMSNormWithInputScale if post_input_scale else MuseSpark1xRMSNorm
    )
    post_norm = _bf16_module(
        post_cls(
            hidden, eps=1e-5, has_weight=post_input_scale, zero_centered_gamma=True
        )
    )
    pre_norm = _bf16_module(
        MuseSpark1xRMSNorm(hidden, eps=1e-6, has_weight=True, zero_centered_gamma=True)
    )
    residual = _bf16_module(MuseSpark1xNormPreservingResidualConnection(hidden))
    return FusedPostNormResidualPreNorm(
        post_norm,
        residual,
        pre_norm,
        post_norm_is_input_scale=post_input_scale,
        use_residual_gate=True,
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@pytest.mark.parametrize("post_input_scale", [False, True])
@pytest.mark.parametrize("router_dtype", [None, torch.float32])
def test_norm_residual_matches_native(
    backend, num_tokens, post_input_scale, router_dtype
):
    torch.manual_seed(num_tokens)
    hidden = 4096
    op = _boundary(post_input_scale, hidden)
    stream = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.float32)
    branch = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ref = op.forward_native(stream, branch, router_dtype=router_dtype)
    out = op(stream, branch, router_dtype=router_dtype, backend=backend)
    torch.testing.assert_close(out[0], ref[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out[1], ref[1], rtol=1e-2, atol=1e-2)
    if router_dtype is None:
        assert out[2] is None and ref[2] is None
    else:
        torch.testing.assert_close(out[2], ref[2], rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
