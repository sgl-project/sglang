"""Kimi-K3 fused KDA decode at TP16 (6 local heads,
SGLANG_OPT_K3_FUSED_KDA_DECODE_TP16) against the unfused chain a TP16 KDA layer
runs otherwise: causal_conv1d_update -> fused_sigmoid_gating_delta_rule_update
(safe gate, lower bound -5) -> sigmoid-gated RMSNorm.

With ``round_like_unfused`` the fused kernel rounds q/k/v and the recurrence
output to bf16 where the unfused chain materializes them, so outputs and states
stay within fp32 evaluation-order noise of that chain, also over a 64-step
rollout from the same state.
"""

import types

import pytest
import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D = 128
LOWER_BOUND = -5.0
EPS = 1e-5
SLOTS = 64
STEPS = 64
_CUDA = torch.cuda.is_available()


def _make_static(heads: int, seed: int) -> dict:
    """Layer weights, and the fused kernel's static inputs built as
    KimiK3DeltaAttention._prepare_fused_decode builds them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * D
    conv_w = torch.randn(3 * seg, 4, generator=g, device="cuda") * 0.3
    a_log = torch.log(torch.rand(1, 1, heads, 1, generator=g, device="cuda") * 15 + 1)
    dt_bias = torch.randn(seg, generator=g, device="cuda") * 0.5
    onorm_w = (1 + 0.1 * torch.randn(D, generator=g, device="cuda")).bfloat16()
    wt = conv_w.t().contiguous()
    fused_static = (
        wt[:, :seg].contiguous(),
        wt[:, seg : 2 * seg].contiguous(),
        wt[:, 2 * seg :].contiguous(),
        torch.zeros(3 * seg, device="cuda"),
        a_log.reshape(-1),
        onorm_w.float().contiguous(),
    )
    return dict(
        heads=heads,
        seg=seg,
        conv_w=conv_w,
        a_log=a_log,
        dt_bias=dt_bias,
        onorm_w=onorm_w,
        fused_static=fused_static,
    )


def _make_step(heads: int, rows: int, seed: int) -> dict:
    """Per-step activations sliced from fused projection outputs, as in the
    model: q|k|v and the output gate from one [rows, 4 * seg] GEMM, beta from
    the [f_a | b | pad] GEMV."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * D
    proj = torch.randn(rows, 4 * seg, generator=g, device="cuda").bfloat16()
    a = (torch.randn(rows, seg, generator=g, device="cuda") * 2).bfloat16()
    pad = (-(D + heads)) % 8
    fab = (
        torch.randn(rows, D + heads + pad, generator=g, device="cuda") * 2
    ).bfloat16()
    return dict(
        qkv=proj[:, : 3 * seg], gate=proj[:, 3 * seg :], a=a, beta=fab[:, D : D + heads]
    )


def _unfused(static, step, conv, ssm, indices, query_start_loc):
    seg = static["seg"]
    qkv = causal_conv1d_update(
        step["qkv"],
        conv.transpose(-1, -2),
        static["conv_w"],
        None,
        activation="silu",
        conv_state_indices=indices,
    )
    q, k, v = (
        t.unflatten(-1, (-1, D)).unsqueeze(0) for t in qkv.split([seg] * 3, dim=-1)
    )
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=static["a_log"],
        dt_bias=static["dt_bias"],
        q=q,
        k=k,
        v=v,
        a=step["a"],
        b=step["beta"].unsqueeze(0),
        initial_state_source=ssm,
        initial_state_indices=indices,
        cu_seqlens=query_start_loc,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=LOWER_BOUND,
    )
    o = rms_norm_gated(
        o,
        step["gate"].unflatten(-1, (-1, D)),
        static["onorm_w"],
        None,
        "sigmoid",
        eps=EPS,
    )
    return o.squeeze(0).flatten(-2)


def _fused(static, step, conv, ssm, indices):
    w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_w = static["fused_static"]
    assert kda_fused_decode.covered(
        step["qkv"], step["a"], step["beta"], conv, ssm, indices, step["gate"]
    )
    o = kda_fused_decode.kda_fused_decode(
        step["qkv"],
        step["a"],
        step["beta"],
        conv,
        w_q_t,
        w_k_t,
        w_v_t,
        conv_bias,
        a_log,
        static["dt_bias"],
        step["gate"],
        onorm_w,
        ssm,
        indices,
        scale=D**-0.5,
        onorm_eps=EPS,
        lower_bound=LOWER_BOUND,
        round_like_unfused=True,
    )
    return o.view(o.shape[1], -1)


def _within_2_ulp(ref: torch.Tensor, got: torch.Tensor) -> float:
    r, o = ref.float(), got.float()
    ulp = torch.where(r == 0, torch.full_like(r, 2.0**-133), r.abs() * 2.0**-7)
    return ((r - o).abs() <= 2 * ulp).float().mean().item()


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("live", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("graph_padding", [0, 2])
def test_matches_unfused_tp16_chain(live: int, graph_padding: int):
    heads = 6
    static = _make_static(heads, seed=heads)
    g = torch.Generator(device="cuda").manual_seed(100 + live)
    conv0 = torch.randn(SLOTS, 3, 3 * heads * D, generator=g, device="cuda").bfloat16()
    ssm0 = torch.randn(SLOTS, heads, D, D, generator=g, device="cuda") * 0.05
    slots = torch.randperm(SLOTS, generator=torch.Generator().manual_seed(live))[:live]
    # CUDA-graph padding rows carry slot index -1
    indices = torch.cat(
        [slots.int(), torch.full((graph_padding,), -1, dtype=torch.int32)]
    ).cuda()
    rows = live + graph_padding
    query_start_loc = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    used = slots.cuda().long()
    unused = torch.tensor(
        [s for s in range(SLOTS) if s not in set(slots.tolist())], device="cuda"
    )

    conv_u, ssm_u = conv0.clone(), ssm0.clone()
    conv_f, ssm_f = conv0.clone(), ssm0.clone()
    for t in range(STEPS):
        step = _make_step(heads, rows, seed=1000 * live + t)
        out_u = _unfused(static, step, conv_u, ssm_u, indices, query_start_loc)
        out_f = _fused(static, step, conv_f, ssm_f, indices)
        if t in (0, STEPS - 1):
            assert _within_2_ulp(out_u[:live], out_f[:live]) > 0.999, t
            assert torch.equal(conv_u[used], conv_f[used]), t
            ssm_diff = (ssm_u[used] - ssm_f[used]).norm() / ssm_u[used].norm()
            assert ssm_diff.item() < 1e-6, (t, ssm_diff.item())
            assert not out_f[live:].any(), "padding rows must stay zero"
            assert torch.equal(conv_f[unused], conv0[unused])
            assert torch.equal(ssm_f[unused], ssm0[unused])

    # same inputs and state twice: identical outputs and states
    step = _make_step(heads, rows, seed=7)
    conv_a, ssm_a, conv_b, ssm_b = (
        conv0.clone(),
        ssm0.clone(),
        conv0.clone(),
        ssm0.clone(),
    )
    out_a = _fused(static, step, conv_a, ssm_a, indices)
    out_b = _fused(static, step, conv_b, ssm_b, indices)
    assert torch.equal(out_a, out_b) and torch.equal(ssm_a, ssm_b)


def _fake_kda_layer(heads: int):
    """The attributes KimiK3DeltaAttention._prepare_fused_decode reads."""
    seg = heads * D
    attn = types.SimpleNamespace(
        conv_weights=torch.randn(3 * seg, 4, device="cuda"),
        bias=None,
        A_log=torch.randn(1, 1, heads, 1, device="cuda"),
        dt_bias=torch.randn(seg, device="cuda"),
        lower_bound=LOWER_BOUND,
    )
    o_norm = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.randn(D, device="cuda").bfloat16()), eps=EPS
    )
    return types.SimpleNamespace(
        attn=attn,
        o_norm=o_norm,
        f_b_proj=None,
        _kda_fused_decode_ready=False,
        _kda_hip_fused_decode_ready=False,
    )


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("heads", [3, 6, 12])
def test_prepare_fused_decode(monkeypatch, enabled: bool, heads: int):
    """12 local heads (TP8) take the fused path as before; 6 (TP16) only with
    the flag, and with the unfused chain's rounding; 3 (TP32) never."""
    from sglang.srt.models import kimi_k3

    monkeypatch.setattr(kimi_k3, "_FUSED_KDA_DECODE_TP16", enabled)
    layer = _fake_kda_layer(heads)
    kimi_k3.KimiK3DeltaAttention._prepare_fused_decode(layer)
    expect_ready = heads == 12 or (enabled and heads == 6)
    assert layer._kda_fused_decode_ready == expect_ready
    if expect_ready:
        *_, round_like_unfused = layer.attn._k3_fused_decode_args
        assert round_like_unfused == (heads != 12)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
