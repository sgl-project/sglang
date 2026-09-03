"""``InklingBatchDenseMLP.forward`` dispatch parity: the generic per-expert bmm
path and the linearized-bf16 fast path must both compute the same weighted
sum of shared-expert SwiGLU MLPs, and the gamma-width guard must reject a
malformed weight tensor before either path runs.

The Triton ``silu_and_mul_triton`` kernel this forward delegates to is CUDA
kernel numerics covered on-device by
``test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py``; here it is
replaced by its documented fp32 contract (``silu(gate) * up * weight``, gate
and up interleaved) so the *orchestration* around it -- per-expert batching,
gamma weighting, fp32-accumulated summation across experts, and the
generic-vs-linearized dispatch -- is checked on the host.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest
import torch

import sglang.kernels.ops.moe.inkling_moe as inkling_moe_module
from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_N_SHARED = 3
_D_MODEL = 4
_SHARED_D_MLP = 8


def _fake_silu_and_mul_triton(gate_up: torch.Tensor, weight: torch.Tensor):
    """Host stand-in for the kernel's documented contract: fp32
    ``silu(gate) * up * weight`` on the interleaved [gate, up, gate, up, ...]
    layout ``InklingBatchDenseMLP`` always feeds it (see
    ``_swiglu``'s ``inference_moe_w13_interleaved`` assertion)."""
    gu = gate_up.float()
    gate, up = gu[..., 0::2], gu[..., 1::2]
    out = gate * torch.sigmoid(gate) * up * weight.float()[:, None]
    return out.to(gate_up.dtype)


def _new_layer(*, linearized_bf16: bool = False) -> InklingBatchDenseMLP:
    torch.manual_seed(0)
    layer = InklingBatchDenseMLP(
        n_shared_experts=_N_SHARED,
        d_model=_D_MODEL,
        shared_d_mlp=_SHARED_D_MLP,
        layer_id=0,
        prefix="model.layers.0.mlp.shared_experts",
        quant_config=None,
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        linearized_bf16=linearized_bf16,
    )
    with torch.no_grad():
        layer.w13_weight.copy_(torch.randn_like(layer.w13_weight))
        layer.w2_weight.copy_(torch.randn_like(layer.w2_weight))
    if linearized_bf16:
        # Normally triggered by the w2_weight loader; done directly here since
        # the weights above were written by copy_(), not weight_loader_fused.
        layer._refresh_bf16_linearized()
    return layer


def _reference_forward(
    layer: InklingBatchDenseMLP, x: torch.Tensor, gammas: torch.Tensor
) -> torch.Tensor:
    """Independent per-expert reference: for each shared expert, SwiGLU the
    x @ w13 projection, scale by that expert's gamma, project through w2, and
    fp32-sum across experts -- the same weighted MoE mixture the generic and
    linearized forward paths must both compute."""
    x_td = x.reshape(-1, x.shape[-1])
    gammas_ts = gammas.reshape(-1, gammas.shape[-1])
    acc = torch.zeros(x_td.shape[0], layer.hidden_size, dtype=torch.float32)
    for s in range(layer.n_shared_experts):
        z = x_td @ layer.w13_weight[s].T
        gate, up = z[..., 0::2].float(), z[..., 1::2].float()
        act = (gate * torch.sigmoid(gate) * up) * gammas_ts[:, s : s + 1].float()
        acc = acc + (act.to(x.dtype) @ layer.w2_weight[s].T).float()
    return acc.to(x.dtype)


@pytest.mark.parametrize("linearized_bf16", [False, True])
@pytest.mark.parametrize("x_ndim", [2, 3])
def test_forward_matches_weighted_shared_expert_reference(
    linearized_bf16: bool, x_ndim: int
):
    """Both the generic bmm path (``linearized_bf16=False``) and the
    linearized fast path (``linearized_bf16=True``) must compute the same
    gamma-weighted sum of shared-expert SwiGLU MLPs; reds if either path
    drops the gamma weighting, mixes up an expert's w13/w2, or sums the
    experts outside fp32."""
    layer = _new_layer(linearized_bf16=linearized_bf16)
    assert layer._linearized_bf16_enabled is linearized_bf16
    assert layer._fp4_strategy.serves_fp4 is False

    torch.manual_seed(1)
    tokens = (2, 3) if x_ndim == 3 else (5,)
    x = torch.randn(*tokens, _D_MODEL)
    gammas = torch.rand(*tokens, _N_SHARED)

    with mock.patch.object(
        inkling_moe_module, "silu_and_mul_triton", _fake_silu_and_mul_triton
    ):
        out = layer.forward(x, gammas)

    expected = _reference_forward(layer, x, gammas)
    # A 3D input is flattened to [T, D] for the mixture and returned flat
    # (only a 2D input round-trips through view_as); the reference is
    # computed the same way, so shapes must match exactly, not just values.
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_forward_rejects_gamma_width_mismatched_with_shared_experts():
    """``gammas``' last dim must equal ``n_shared_experts``; reds if the guard
    is dropped and a mismatched weight tensor is silently broadcast or
    truncated instead of rejected."""
    layer = _new_layer()
    x = torch.randn(5, _D_MODEL)
    wrong_gammas = torch.rand(5, _N_SHARED + 1)

    with mock.patch.object(
        inkling_moe_module, "silu_and_mul_triton", _fake_silu_and_mul_triton
    ):
        with pytest.raises(AssertionError):
            layer.forward(x, wrong_gammas)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
