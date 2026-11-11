"""Per-tensor FP8 provider against FP32 references from dequantized checkpoint
bytes, with static and dynamic per-tensor activation scales.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 512
_I = 512
_TOPK = 2
_T = 32
_FMAX = torch.finfo(torch.float8_e4m3fn).max
# Static checkpoint scales sit below this batch's range, so the provider
# saturates at them: a dynamic scale would not.
_STATIC_RANGE = 0.4


def _tensor_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert fp8 quant; returns (fp8 weight, fp32 scale [E])."""
    scale = weight.float().abs().amax(dim=(1, 2)).clamp(min=1e-6) / _FMAX
    return (weight.float() / scale[:, None, None]).to(torch.float8_e4m3fn), scale


def _fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round x through fp8 at one scale, saturating as the quantizer does."""
    q = (x.float() / scale).clamp(-_FMAX, _FMAX).to(torch.float8_e4m3fn)
    return q.float() * scale


def _case(device, hidden_size=_H):
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, 2 * _I, hidden_size)
    w2 = rand(_E, hidden_size, _I)
    hidden = rand(_T, hidden_size)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    return w13, w2, hidden, topk_ids


def _routes(topk_ids):
    """(pair index, expert, token) of the pairs that have a local expert."""
    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    return pair_idx, flat_ids[pair_idx].long(), (pair_idx // _TOPK).long()


def _act_pairs(num_pairs: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(11)
    return (torch.randn(num_pairs, _I, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )


def _quant_info(w13, w2, hidden, act_pairs, static: bool, hidden_size=_H):
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

    w13_q, w13_s = _tensor_quant(w13)
    w2_q, w2_s = _tensor_quant(w2)
    a13 = a2 = None
    if static:
        a13 = (hidden.float().abs().amax() * _STATIC_RANGE / _FMAX).reshape(())
        a2 = (act_pairs.float().abs().amax() * _STATIC_RANGE / _FMAX).reshape(())
    info = MoeLoraFp8TensorQuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        w13_input_scale=a13,
        w2_input_scale=a2,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=hidden_size,
    )
    return (
        info,
        w13_q.float() * w13_s[:, None, None],
        w2_q.float() * w2_s[:, None, None],
    )


def _run(quant_info, hidden, topk_ids, act_pairs, pair_idx, *, stale: float = 0.0):
    """Run both GEMMs; activation rows of pairs without an expert hold `stale`."""
    from sglang.srt.runtime_context import get_context

    device = hidden.device
    provider = select_provider_cls("route_major", "fp8_tensor", "triton")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.empty(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)
        rows = state.pair_to_row[pair_idx].long()
        act = torch.full(
            provider.act_out_shape(state), stale, dtype=torch.bfloat16, device=device
        )
        act.reshape(-1, _I)[rows] = act_pairs
        down = torch.empty(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.down(state, act, down)
    return (
        gateup.reshape(-1, gateup.shape[-1])[rows].float(),
        down.reshape(-1, down.shape[-1])[rows].float(),
    )


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def _references(hidden, act_pairs, w13_eff, w2_eff, experts, tokens):
    return (
        torch.einsum("ph,pnh->pn", hidden[tokens].float(), w13_eff[experts]),
        torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts]),
    )


@pytest.mark.parametrize("hidden_size", [_H, 7168])
def test_fp8_tensor_dynamic_scales_match_reference(hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    pair_idx, experts, tokens = _routes(topk_ids)
    act_pairs = _act_pairs(len(pair_idx), device)
    quant_info, w13_eff, w2_eff = _quant_info(
        w13, w2, hidden, act_pairs, False, hidden_size=hidden_size
    )
    got_gateup, got_down = _run(quant_info, hidden, topk_ids, act_pairs, pair_idx)
    ref_gateup, ref_down = _references(
        hidden, act_pairs, w13_eff, w2_eff, experts, tokens
    )
    # One activation scale per tensor: the coarsest FP8 quantization.
    assert _rel_l2(got_gateup, ref_gateup) < 8e-2
    assert _rel_l2(got_down, ref_down) < 8e-2


@pytest.mark.parametrize("hidden_size", [_H, 7168])
def test_fp8_tensor_static_scales_saturate_at_the_checkpoint(hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    pair_idx, experts, tokens = _routes(topk_ids)
    act_pairs = _act_pairs(len(pair_idx), device)
    quant_info, w13_eff, w2_eff = _quant_info(
        w13, w2, hidden, act_pairs, True, hidden_size=hidden_size
    )
    got_gateup, got_down = _run(quant_info, hidden, topk_ids, act_pairs, pair_idx)
    # The reference rounds and saturates the activations at the static scales.
    ref_gateup, ref_down = _references(
        _fake_quant(hidden, quant_info.w13_input_scale),
        _fake_quant(act_pairs, quant_info.w2_input_scale),
        w13_eff,
        w2_eff,
        experts,
        tokens,
    )
    assert _rel_l2(got_gateup, ref_gateup) < 1e-2
    assert _rel_l2(got_down, ref_down) < 1e-2
    # Saturation keeps the result away from the unclipped (dynamic) one.
    dyn_gateup, dyn_down = _references(
        hidden, act_pairs, w13_eff, w2_eff, experts, tokens
    )
    assert _rel_l2(got_gateup, dyn_gateup) > 5e-2
    assert _rel_l2(got_down, dyn_down) > 5e-2


@pytest.mark.parametrize("stale", [float("nan"), 1e4])
def test_fp8_tensor_dynamic_scale_skips_dropped_pairs(stale):
    """EP drops pairs without a local expert: their activation rows are never
    written and must not reach the dynamic per-tensor scale."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device)
    topk_ids[::2, 1] = -1
    topk_ids[5] = -1
    pair_idx, experts, tokens = _routes(topk_ids)
    assert len(pair_idx) < topk_ids.numel()
    act_pairs = _act_pairs(len(pair_idx), device)
    quant_info, w13_eff, w2_eff = _quant_info(w13, w2, hidden, act_pairs, False)
    got_gateup, got_down = _run(
        quant_info, hidden, topk_ids, act_pairs, pair_idx, stale=stale
    )
    ref_gateup, ref_down = _references(
        hidden, act_pairs, w13_eff, w2_eff, experts, tokens
    )
    assert _rel_l2(got_gateup, ref_gateup) < 8e-2
    assert _rel_l2(got_down, ref_down) < 8e-2


def test_fp8_tensor_rejects_unfolded_scales():
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

    w13, w2, _hidden, _topk_ids = _case(torch.device("cpu"))
    w13_q, w13_s = _tensor_quant(w13)
    w2_q, w2_s = _tensor_quant(w2)
    layer = type(
        "Layer",
        (),
        dict(
            w13_weight=w13_q,
            w13_weight_scale=torch.stack([w13_s, w13_s], dim=1),
            w2_weight=w2_q,
            w2_weight_scale=w2_s,
            w13_input_scale=None,
            w2_input_scale=None,
            num_local_experts=_E,
        ),
    )
    with pytest.raises(ValueError, match=r"w13_scale must be fp32 \[4\]"):
        MoeLoraFp8TensorQuantInfo.from_layer(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
