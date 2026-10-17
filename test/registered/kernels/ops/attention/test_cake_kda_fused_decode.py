"""Cake fused Kimi KDA decode (conv + recurrent KDA + RMSNorm) through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend; that the
facade output, the SD-layout conv state and the recurrent state pool are
bitwise identical to ``flashinfer.kda_decode.fused_kda_decode(backend="cake",
state_indices_mode=...)``; and that the result matches a pure-torch pipeline
(4-tap depthwise conv + SiLU, L2-normalized q/k, bounded gate, delta-rule
update, RMSNorm gated by ``sigmoid(output_gate)``) within BF16 tolerance.
Skips with the reason when FlashInfer lacks the module, the GPU is outside
sm_100a / sm_103a, or FlashInfer's frozen variant registry has no route for
the test layout (the registry is FlashInfer's, not SGLang's).
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_kda as cake_kda
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import cake_kda_fused_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "attention.kda_fused_decode"
HEAD_DIM = cake_kda.HEAD_DIM
LOWER_BOUND = -5.0
NORM_EPS = 1e-5


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_linear_kda:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_kda.FI_MODULE_DECODE, cake_kda.FI_JIT_MODULE_FUSED
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.jit.cake_fused_kda_decode")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_kda.ARCHS:
        pytest.skip(f"Cake fused KDA decode is built for sm_100a/sm_103a, got {cc}")


def _reference(inputs, conv_state, state):
    x = inputs["x"]
    rows = x.shape[0]
    heads = inputs["A_log"].numel()
    hidden = heads * HEAD_DIM
    slots = inputs["state_indices"].long()
    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden, 4)
    history = conv_state.index_select(0, slots).float()
    window = torch.cat((history, x.float().unsqueeze(-1)), dim=-1)
    conv_state.index_copy_(0, slots, window[:, :, 1:].to(torch.bfloat16))
    mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16).float()
    q, k, v = mixed.view(rows, 3, heads, HEAD_DIM).unbind(1)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * HEAD_DIM**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = inputs["raw_gate"][0].float() + inputs["dt_bias"].view(heads, HEAD_DIM)
    gate = LOWER_BOUND * torch.sigmoid(inputs["A_log"].exp()[None, :, None] * gate)
    selected = state.index_select(0, slots).float() * gate.exp().unsqueeze(-2)
    delta = v - torch.einsum("nhvk,nhk->nhv", selected, k)
    delta = delta * inputs["raw_beta"][0].float().sigmoid().unsqueeze(-1)
    selected = selected + delta.unsqueeze(-1) * k.unsqueeze(-2)
    state.index_copy_(0, slots, selected.to(state.dtype))
    out = torch.einsum("nhvk,nhk->nhv", selected, q).to(torch.bfloat16).float()
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + NORM_EPS)
    out = out * inputs["norm_weight"] * inputs["output_gate"].float().sigmoid()
    return out.unsqueeze(0)


@pytest.mark.parametrize("heads", [12, 8])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_matches_flashinfer_and_reference(heads, state_dtype):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    rows, slots = 3, 6
    hidden = heads * HEAD_DIM
    inputs = dict(
        x=torch.randn(rows, 3 * hidden, device=device, dtype=torch.bfloat16),
        weight=(torch.randn(3, 4, hidden, device=device) * 0.3).contiguous(),
        raw_gate=torch.randn(1, rows, heads, HEAD_DIM, device=device).to(
            torch.bfloat16
        ),
        raw_beta=torch.randn(1, rows, heads, device=device).to(torch.bfloat16),
        A_log=torch.randn(heads, device=device) * 0.1,
        dt_bias=torch.randn(hidden, device=device) - 2.0,
        state_indices=torch.tensor([4, 1, 2], device=device, dtype=torch.int32),
        output_gate=torch.randn(rows, heads, HEAD_DIM, device=device).to(
            torch.bfloat16
        ),
        norm_weight=torch.rand(HEAD_DIM, device=device) + 0.5,
    )
    # SD cache layout: [slots, 3*hidden, 3] with strides (9*hidden, 1, 3*hidden).
    conv_state = (
        torch.randn(slots, 3, 3 * hidden, device=device).to(torch.bfloat16)
    ).transpose(1, 2)
    state = (torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, device=device) * 0.1).to(
        state_dtype
    )
    assert cake_kda.supports_kda_fused_decode(
        inputs["x"],
        inputs["weight"],
        conv_state,
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_indices"],
        state,
        inputs["output_gate"],
        inputs["norm_weight"],
        lower_bound=LOWER_BOUND,
        norm_eps=NORM_EPS,
        state_indices_mode="positive_unique",
    )
    conv_fi, state_fi = conv_state.clone(), state.clone()
    conv_ref, state_ref = conv_state.clone(), state.clone()
    args = (
        inputs["x"],
        inputs["weight"],
    )
    tail = (
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_indices"],
    )
    kwargs = dict(lower_bound=LOWER_BOUND, norm_eps=NORM_EPS)
    try:
        out = cake_kda_fused_decode(
            *args,
            conv_state,
            *tail,
            state,
            inputs["output_gate"],
            inputs["norm_weight"],
            state_indices_mode="positive_unique",
            **kwargs,
        )
    except RuntimeError as error:
        if "does not have a route" in str(error):
            pytest.skip(f"FlashInfer has no frozen fused-decode variant: {error}")
        raise
    from flashinfer.kda_decode import fused_kda_decode as fi_direct

    out_fi = fi_direct(
        *args,
        conv_fi,
        *tail,
        state_fi,
        inputs["output_gate"],
        inputs["norm_weight"],
        backend="cake",
        state_indices_mode="positive_unique",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(conv_state, conv_fi)
    assert torch.equal(state, state_fi)

    expected = _reference(inputs, conv_ref, state_ref)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        conv_state.float(), conv_ref.float(), atol=1e-2, rtol=1e-2
    )
    state_tol = 1e-3 if state_dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        state.float(), state_ref.float(), atol=state_tol, rtol=state_tol
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
