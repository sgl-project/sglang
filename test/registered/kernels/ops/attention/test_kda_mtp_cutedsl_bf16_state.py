# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""BF16 recurrent state on the CuTe KDA MTP verify (ReplaySSM ring mode).

With ``--mamba-ssm-dtype bfloat16`` the verify kernel stages the state as BF16
and widens it to FP32 in registers. Widening is exact, so for the same BF16
state values the kernel must produce BIT-IDENTICAL outputs and ring records to
the FP32 kernel run on ``state.float()`` (the served FP32 path is the
reference for the new load path). An independent FP64 sequential KDA
reference checks the shared math (max-norm error <= 2e-2, cosine >= 0.9995,
bf16 I/O). The wrapper must refuse BF16 state without the rings and slot
offsets that break the 16-byte cp.async alignment; the backend takes the
CuTe path for BF16 state only behind SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("CuTe KDA MTP verify needs SM100-class GPUs.", allow_module_level=True)

from sglang.kernels.ops.attention.kda_decode_mtp import (  # noqa: E402
    fused_kda_decode_mtp_dspark,
)

DEV = "cuda"
K = 128
W = 4
L = 16
NUM_SPEC = 7
LB = -5.0


def _inputs(*, N, H, seed, onorm):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*shape, s=1.0, dt=torch.bfloat16):
        return (torch.randn(*shape, device=DEV, generator=g) * s).to(dt)

    T = N * (1 + NUM_SPEC)
    slots = N + 2
    d = dict(
        x_q=rn(1, T, H, K), x_k=rn(1, T, H, K), x_v=rn(1, T, H, K), g=rn(1, T, H, K), beta=rn(1, T, H),
        w_q=rn(H * K, W, s=0.3, dt=torch.float32), w_k=rn(H * K, W, s=0.3, dt=torch.float32),
        w_v=rn(H * K, W, s=0.3, dt=torch.float32), cs_q=rn(slots, H * K, W - 1), cs_k=rn(slots, H * K, W - 1),
        cs_v=rn(slots, H * K, W - 1), A_log=(torch.rand(H, device=DEV, generator=g) * 2 - 1).float(),
        dt_bias=rn(H * K, s=0.5, dt=torch.float32),
        ssm_state_indices=torch.arange(1, N + 1, device=DEV, dtype=torch.int32),
        intermediate_state_indices=torch.arange(N, device=DEV, dtype=torch.int32),
        cu_seqlens=torch.arange(0, T + 1, 1 + NUM_SPEC, device=DEV, dtype=torch.int32),
        lower_bound=LB,
    )
    if onorm:
        d.update(onorm_gate=rn(1, T, H, K), onorm_weight=(1 + 0.1 * rn(K, dt=torch.float32)), onorm_eps=1e-5)
    state16 = rn(slots, H, K, K, s=0.05)
    return d, state16


def _run(d, state, *, N, H):
    T = N * (1 + NUM_SPEC)
    slots = state.shape[0]
    bufs = dict(
        intermediate_conv_q=torch.zeros(N, 1 + NUM_SPEC, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
        intermediate_conv_k=torch.zeros(N, 1 + NUM_SPEC, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
        intermediate_conv_v=torch.zeros(N, 1 + NUM_SPEC, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
        replayssm_rawv=torch.zeros(slots, H, L, K, device=DEV, dtype=torch.bfloat16),
        replayssm_rawk=torch.zeros(slots, H, L, K, device=DEV, dtype=torch.bfloat16),
        replayssm_g=torch.zeros(slots, H, L, K, device=DEV, dtype=torch.float32),
        replayssm_beta=torch.zeros(slots, H, L, device=DEV, dtype=torch.float32),
    )
    out = fused_kda_decode_mtp_dspark(recurrent_state=state, intermediate_ssm=None, **bufs, **d)
    torch.cuda.synchronize()
    assert out.shape == (1, T, H, K)
    return out, bufs


def _ref(d, state16, *, n, H, onorm):
    """FP64 sequential KDA definition for request n: conv4 + SiLU, L2-normed q (x K^-0.5) and k, safe gate
    LB * sigmoid(exp(A_log) * (a + dt_bias)), decay along K, delta rule with sigmoid(b), o = S q, optional
    gated RMSNorm with a sigmoid gate."""
    T1 = 1 + NUM_SPEC
    sl = slice(n * T1, (n + 1) * T1)
    slot = int(d["ssm_state_indices"][n])
    S = state16[slot].double().clone()
    win = {c: d["cs_" + c][slot].double().t().clone() for c in "qkv"}  # [W-1, H*K], oldest first
    outs = []
    for t in range(T1):
        y = {}
        for c in "qkv":
            x = d["x_" + c][0, sl][t].double().reshape(1, H * K)
            full = torch.cat([win[c], x], 0)
            y[c] = F.silu((full * d["w_" + c].double().t()).sum(0)).view(H, K)
            win[c] = full[1:]
        q = y["q"] / torch.sqrt((y["q"] ** 2).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
        k = y["k"] / torch.sqrt((y["k"] ** 2).sum(-1, keepdim=True) + 1e-6)
        a = d["g"][0, sl][t].double()
        gate = LB * torch.sigmoid(torch.exp(d["A_log"].double()).view(H, 1) * (a + d["dt_bias"].double().view(H, K)))
        beta = torch.sigmoid(d["beta"][0, sl][t].double())
        S = S * torch.exp(gate).unsqueeze(1)
        r = (y["v"] - torch.einsum("hvk,hk->hv", S, k)) * beta.unsqueeze(1)
        S = S + r.unsqueeze(2) * k.unsqueeze(1)
        o = torch.einsum("hvk,hk->hv", S, q)
        if onorm:
            o = (o * torch.rsqrt((o * o).mean(-1, keepdim=True) + d["onorm_eps"]) * d["onorm_weight"].double()
                 * torch.sigmoid(d["onorm_gate"][0, sl][t].double()))
        outs.append(o)
    return torch.stack(outs)


@pytest.mark.parametrize("onorm", [False, True])
@pytest.mark.parametrize("N,H", [(1, 6), (4, 6), (16, 6), (64, 6), (16, 12)])
def test_bf16_state_bitwise_equals_fp32_on_widened_state(N, H, onorm):
    d, s16 = _inputs(N=N, H=H, seed=7 + N + H, onorm=onorm)
    out16, b16 = _run(d, s16, N=N, H=H)
    out32, b32 = _run(d, s16.float(), N=N, H=H)
    assert torch.equal(out16, out32), "BF16-state output differs from the FP32 kernel on the widened state"
    for name in b16:
        assert torch.equal(b16[name], b32[name]), f"{name} differs"


@pytest.mark.parametrize("onorm", [False, True])
@pytest.mark.parametrize("N,H", [(4, 6), (64, 6)])
def test_bf16_state_matches_fp64_reference(N, H, onorm):
    d, s16 = _inputs(N=N, H=H, seed=101 + N, onorm=onorm)
    out, _ = _run(d, s16, N=N, H=H)
    T1 = 1 + NUM_SPEC
    for n in sorted({0, N - 1}):
        ref = _ref(d, s16, n=n, H=H, onorm=onorm).reshape(-1)
        got = out[0, n * T1:(n + 1) * T1].double().reshape(-1)
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        cos = F.cosine_similarity(got, ref, dim=0).item()
        assert err <= 2e-2 and cos >= 0.9995, (n, err, cos)


def test_bf16_state_refused_without_rings():
    N, H = 4, 6
    d, s16 = _inputs(N=N, H=H, seed=3, onorm=False)
    T1 = 1 + NUM_SPEC

    def snapshot_call(state):
        return fused_kda_decode_mtp_dspark(
            recurrent_state=state,
            intermediate_ssm=torch.zeros(N, T1, H, K, K, device=DEV, dtype=torch.float32),
            intermediate_conv_q=torch.zeros(N, T1, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
            intermediate_conv_k=torch.zeros(N, T1, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
            intermediate_conv_v=torch.zeros(N, T1, H * K, W - 1, device=DEV, dtype=torch.bfloat16),
            **d,
        )

    snapshot_call(s16.float())  # control: the same snapshot-mode call runs with FP32 state
    with pytest.raises(ValueError):
        snapshot_call(s16)


def test_bf16_state_misaligned_slot_refused():
    # A pool view whose slot base is 8-byte but not 16-byte aligned (4 BF16 elements in) must be refused.
    N, H = 4, 6
    d, s16 = _inputs(N=N, H=H, seed=5, onorm=False)
    flat = torch.zeros(s16.numel() + 8, device=DEV, dtype=torch.bfloat16)
    aligned = flat[8:].view(s16.shape)  # control: 16-byte aligned offset runs and matches the base run
    aligned.copy_(s16)
    assert torch.equal(_run(d, aligned, N=N, H=H)[0], _run(d, s16, N=N, H=H)[0])
    shifted = flat[4:4 + s16.numel()].view(s16.shape)
    shifted.copy_(s16)
    assert shifted.storage_offset() % 8 == 4
    with pytest.raises(ValueError):
        _run(d, shifted, N=N, H=H)


def test_backend_gate_bf16_state_behind_knob():
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

    N, H = 4, 6
    d, s16 = _inputs(N=N, H=H, seed=9, onorm=False)
    stub = types.SimpleNamespace(
        kernel_dispatcher=types.SimpleNamespace(verify_backend=types.SimpleNamespace(is_nv_cutedsl=lambda: True))
    )
    layer = types.SimpleNamespace(
        bias=None, lower_bound=LB, head_q_dim=K, head_k_dim=K, head_v_dim=K, num_q_heads=H, num_k_heads=H,
        num_v_heads=H, conv_weights=torch.zeros(3 * H * K, W, device=DEV, dtype=torch.float32),
        q_dim=H * K, k_dim=H * K, v_dim=H * K,
    )
    mixed = torch.zeros(N * (1 + NUM_SPEC), 3 * H * K, device=DEV, dtype=torch.bfloat16)
    conv = torch.zeros(N + 2, 3, 3 * H * K, device=DEV, dtype=torch.bfloat16)
    icw = torch.zeros(N + 2, 1 + NUM_SPEC, 3, 3 * H * K, device=DEV, dtype=torch.bfloat16)
    rawv = torch.zeros(N + 2, H, L, K, device=DEV, dtype=torch.bfloat16)

    def gate(state, inter):
        return KDAAttnBackend._can_run_dspark_cutedsl_mtp(
            stub, layer=layer, mixed_qkv=mixed, a=mixed, b=mixed, draft_token_num=1 + NUM_SPEC,
            ragged_layout=None, conv_states=conv, ssm_states=state, intermediate_state_cache=inter,
            intermediate_conv_window_cache=icw, retrieve_parent_token=None, replayssm_rawv=rawv,
        )

    inter = torch.zeros(N, 1 + NUM_SPEC, H, K, K, device=DEV, dtype=torch.float32)
    with envs.SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE.override(False):
        assert gate(s16.float(), None) is True
        assert gate(s16, None) is False
    with envs.SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE.override(True):
        assert gate(s16, None) is True
        assert gate(s16, inter) is False  # snapshot (non-ring) path stays FP32-only
