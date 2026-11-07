"""M3.1 NVFP4 experts on the DeepEP low-latency flashinfer_cutedsl masked path."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# Import quantization before the MoE runners; the reverse is a circular import.
# isort: off
from sglang.srt.layers.quantization import minimax_m3_nvfp4
from sglang.srt.layers.moe import flashinfer_cutedsl_moe
from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    _quantize_per_token_nvfp4,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK, T = 8, 1024, 512, 4, 96
ALPHA, LIMIT = 1.702, 7.0
_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

requires_fused_masked_gemm = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not flashinfer_cutedsl_moe.supports_masked_gemm_fused_swiglu(
        torch.device("cuda")
    ),
    reason="needs SM100/SM103 and FlashInfer's fused masked GEMM epilogue",
)


def _dequant(q, sf, gs, shape):
    lut = torch.tensor(_E2M1, device=q.device)
    vals = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1)
    return vals.reshape(shape) * sf.float().repeat_interleave(16, -1) / gs


def _quantize(w):
    """Compressed-tensors NVFP4: packed e2m1, e4m3 block scales, encode global."""
    from flashinfer import fp4_quantize

    gs = (448 * 6 / w.abs().amax()).float().reshape(1)
    q, sf = fp4_quantize(w, gs, 16, False, False)
    sf = sf.view(torch.float8_e4m3fn).reshape(w.shape[0], -1)
    return q.view(torch.uint8), sf, gs, _dequant(q.view(torch.uint8), sf, gs, w.shape)


def _fake_quant_per_token(a):
    """What dynamic per-token NVFP4 activation quantization keeps of `a`."""
    return torch.cat([_quantize(row[None].bfloat16())[3] for row in a])


def _swigluoai(x):
    gate, up = x.chunk(2, -1)
    gate, up = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
    return gate * torch.sigmoid(ALPHA * gate) * (up + 1)


def _build_layer(gate_up_scale_ratio: float):
    layer = torch.nn.Module()
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / 4
    # Distinct gate and up global scales catch a dropped alpha_up.
    w13[:, I:] *= gate_up_scale_ratio
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / 8
    gate = [_quantize(w[:I]) for w in w13]
    up = [_quantize(w[I:]) for w in w13]
    q2, s2, g2, d2 = zip(*(_quantize(w) for w in w2))
    params = dict(
        w13_weight_packed=torch.stack(
            [torch.cat((g[0], u[0])) for g, u in zip(gate, up)]
        ),
        w13_weight_scale=torch.stack(
            [torch.cat((g[1], u[1])) for g, u in zip(gate, up)]
        ),
        w13_weight_global_scale=torch.stack(
            [torch.cat((g[2], u[2])) for g, u in zip(gate, up)]
        ),
        w13_input_global_scale=torch.full((E, 2), 1.0, device="cuda"),
        w2_weight_packed=torch.stack(q2),
        w2_weight_scale=torch.stack(s2),
        w2_weight_global_scale=torch.cat(g2),
        # Non-unit so a dropped or double-applied fc2 input scale shows up.
        w2_input_global_scale=torch.full((E,), 2.0, device="cuda"),
    )
    for name, t in params.items():
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.num_experts = layer.num_local_experts = E
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition = I
    layer._minimax_activation = "swigluoai"
    layer.dispatcher = mock.Mock()
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=E,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size_per_partition=I,
        top_k=TOPK,
        gemm1_alpha=ALPHA,
        gemm1_beta=1.0,
        gemm1_clamp_limit=LIMIT,
        inplace=False,
    )
    d13 = torch.stack([torch.cat((g[3], u[3])) for g, u in zip(gate, up)])
    return layer, d13, torch.stack(d2)


def _reference(x, ids, weights, d13, d2):
    out = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    xq = _fake_quant_per_token(x)
    for k in range(TOPK):
        h = _swigluoai(torch.einsum("th,tnh->tn", xq, d13[ids[:, k]]))
        # GEMM2 input uses the static per-expert FC2 input scale (2.0).
        gs = 2.0
        q, sf = _fp4_static(h, gs)
        h = _dequant(q, sf, gs, h.shape)
        out += weights[:, k : k + 1] * torch.einsum("ti,thi->th", h, d2[ids[:, k]])
    return out


def _fp4_static(h, gs):
    from flashinfer import fp4_quantize

    q, sf = fp4_quantize(
        h.bfloat16(), torch.tensor([gs], device="cuda"), 16, False, False
    )
    return q.view(torch.uint8), sf.view(torch.float8_e4m3fn).reshape(h.shape[0], -1)


def _ll_dispatch(x, ids, weights, capacity):
    """Single-rank stand-in for DeepEP low_latency_dispatch_fp4(swizzle_scales=True)."""
    q, sf, pts = _quantize_per_token_nvfp4(x)
    recv_x = torch.zeros(E, capacity, H // 2, dtype=torch.uint8, device="cuda")
    recv_sf = torch.zeros(E, capacity, H // 16, dtype=torch.uint8, device="cuda")
    # Garbage in unused rows must not leak into valid outputs.
    recv_pts = torch.full((E, capacity), float("nan"), device="cuda")
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    slots = []
    for t in range(x.shape[0]):
        for k in range(TOPK):
            e = int(ids[t, k])
            r = int(counts[e])
            recv_x[e, r], recv_sf[e, r], recv_pts[e, r] = q[t], sf[t], pts[t]
            counts[e] += 1
            slots.append((t, k, e, r))
    rm, rk = capacity // 128, H // 64
    swizzled = (
        swizzle_blockscale(recv_sf.view(torch.float8_e4m3fn))
        .view(E, rm, rk, 32, 4, 4)
        .permute(3, 4, 1, 5, 2, 0)
    )
    dispatch = DeepEPLLDispatchOutput(
        hidden_states=recv_x,
        hidden_states_scale=swizzled,
        topk_ids=ids,
        topk_weights=weights,
        masked_m=counts,
        expected_m=capacity,
        per_token_scale=recv_pts,
    )
    return dispatch, slots


def _scheme():
    with (
        mock.patch.object(
            minimax_m3_nvfp4,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ),
        mock.patch.object(
            minimax_m3_nvfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
        ),
    ):
        return minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()


@requires_fused_masked_gemm
@pytest.mark.parametrize("fused_nvfp4_output", [True, False])
def test_cutedsl_masked_matches_dequantized_reference(fused_nvfp4_output, monkeypatch):
    if not fused_nvfp4_output:
        # Falls back to a BF16 GEMM1 output and a separate quantize kernel.
        monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")
    torch.manual_seed(0)
    scheme = _scheme()
    assert scheme.use_flashinfer_cutedsl and not scheme.load_up_proj_weight_first
    layer, d13, d2 = _build_layer(gate_up_scale_ratio=3.0)
    scheme.process_weights_after_loading(layer)
    layer.dispatcher.set_quant_config.assert_called_once_with(
        {"dispatcher_output_dtype": "nvfp4", "nvfp4_per_token_activation": True}
    )

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    # Uneven routing: expert E - 1 receives no tokens.
    ids = torch.stack(
        [torch.randperm(E - 1, device="cuda")[:TOPK] for _ in range(T)]
    ).int()
    weights = torch.rand(T, TOPK, device="cuda").softmax(-1)
    ref = _reference(x, ids, weights, d13, d2)
    pre_act = torch.einsum("th,tnh->tn", x.float(), d13[ids[:, 0]])
    assert (pre_act.abs() > LIMIT).float().mean() > 0.2

    captured = {}
    scheme.runner = SimpleNamespace(run=lambda d, q: captured.setdefault("q", q))
    scheme.apply_weights(layer, None)
    dispatch, slots = _ll_dispatch(x, ids, weights, capacity=256)
    expert_out = flashinfer_cutedsl.fused_experts_deepep_to_flashinfer_cutedsl_fp4(
        dispatch, captured["q"], layer.moe_runner_config
    ).hidden_states
    assert expert_out.shape == (E, 256, H)

    out = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    for t, k, e, r in slots:
        out[t] += weights[t, k] * expert_out[e, r].float()
    rel = ((out - ref).norm() / ref.norm()).item()
    fit = ((out * ref).sum() / (ref * ref).sum()).item()
    assert rel < 0.13 and abs(fit - 1) < 0.02, (rel, fit)


def test_scheme_rejects_cutedsl_without_deepep():
    with (
        mock.patch.object(
            minimax_m3_nvfp4,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ),
        mock.patch.object(
            minimax_m3_nvfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ),
        mock.patch.object(
            minimax_m3_nvfp4,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=True),
        ),
        pytest.raises(ValueError, match="deepep"),
    ):
        minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
