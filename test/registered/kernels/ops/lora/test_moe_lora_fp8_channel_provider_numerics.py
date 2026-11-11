"""Per-channel FP8 providers against FP32 references from dequantized checkpoint
bytes. Tolerances cover per-token activation quantization and BF16 outputs.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Also exercise the Blackwell row-scaled kernel in per-PR CI.
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E = 4
_H = 512
_I = 512
_TOPK = 2
_T = 32


def _skip_unless_supported(vendor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if vendor == "cutedsl" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the row-scaled FP8 CuTeDSL provider needs SM90+")


def _channel_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row fp8 quant; returns (fp8 weight, fp32 scale [E, rows, 1])."""
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / fmax
    return (weight.float() / scale).to(torch.float8_e4m3fn), scale


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


def _quant_info(w13, w2, hidden_size=_H):
    """Quantize weights locally; the FP32 reference uses the dequantized bytes."""
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

    w13_q, w13_s = _channel_quant(w13)
    w2_q, w2_s = _channel_quant(w2)
    info = MoeLoraFp8ChannelQuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=hidden_size,
    )
    return info, w13_q.float() * w13_s, w2_q.float() * w2_s


def _gather_pairs(rows, pair_to_row, pair_idx):
    return rows.reshape(-1, rows.shape[-1])[pair_to_row[pair_idx].long()]


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


# 7168 is a DeepSeek-V3-class hidden size: not a power of two.
@pytest.mark.parametrize("hidden_size", [_H, 7168])
@pytest.mark.parametrize(
    "vendor,rows",
    [
        ("triton", "route_major"),
        ("cutedsl", "expert_major"),
        ("cutedsl", "route_major"),
    ],
)
def test_fp8_channel_gateup_and_down_match_reference(vendor, rows, hidden_size):
    _skip_unless_supported(vendor)
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    quant_info, w13_eff, w2_eff = _quant_info(w13, w2, hidden_size=hidden_size)

    from sglang.srt.runtime_context import get_context

    provider = select_provider_cls(rows, "fp8_channel", vendor)(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.empty(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)

    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row

    # Per-token activation scales span the whole row, so the quantization
    # error exceeds the group-128 FP8 family's.
    ref_gateup = torch.einsum("ph,pnh->pn", hidden[tokens].float(), w13_eff[experts])
    got_gateup = _gather_pairs(gateup, pair_to_row, pair_idx).float()
    assert _rel_l2(got_gateup, ref_gateup) < 6e-2

    act_pairs = (torch.randn_like(ref_gateup[:, :_I]) * 0.1).to(torch.bfloat16)
    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down = torch.empty(
        provider.down_out_shape(state), dtype=torch.bfloat16, device=device
    )
    with get_context().override_server_args():
        provider.down(state, act, down)

    ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])
    got_down = _gather_pairs(down, pair_to_row, pair_idx).float()
    assert _rel_l2(got_down, ref_down) < 6e-2


@pytest.mark.parametrize("rows", ["expert_major", "route_major"])
def test_fp8_channel_cutedsl_fused_act_quant_matches_separate(rows):
    """Activation writes the per-row FP8 operands the down GEMM reads; they
    equal the separate quantizer's (the fused-B activation path) bitwise."""
    _skip_unless_supported("cutedsl")
    from sglang.srt.runtime_context import get_context

    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device)
    quant_info, _, _ = _quant_info(w13, w2)
    provider = select_provider_cls(rows, "fp8_channel", "cutedsl")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.empty(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)
        act = torch.zeros(
            provider.act_out_shape(state), dtype=torch.bfloat16, device=device
        )
        lora_in = torch.zeros(_T, _TOPK, _I, dtype=torch.bfloat16, device=device)
        provider.act_with_delta(state, gateup, None, topk_ids, act, lora_in)
        assert state.act_quant_ready
        fused_q, fused_s = state.act_fp8.clone(), state.act_scale.clone()
        state.act_quant_ready = False
        separate_q, separate_s = provider._down_operands(state, act)
    pair_rows = state.pair_to_row[topk_ids.flatten() >= 0].long()
    assert torch.equal(
        fused_q.view(-1, _I)[pair_rows].view(torch.uint8),
        separate_q.view(-1, _I)[pair_rows].view(torch.uint8),
    )
    assert torch.equal(fused_s.view(-1)[pair_rows], separate_s.view(-1)[pair_rows])


def test_fp8_channel_cutedsl_rejects_unaligned_widths():
    """The row-scaled kernel stages one weight-row scale per 128-row tile."""
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

    hidden_size = 192
    w13 = torch.zeros(_E, 2 * _I, hidden_size, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(_E, hidden_size, _I, dtype=torch.float8_e4m3fn)
    quant_info = MoeLoraFp8ChannelQuantInfo(
        w13_weight=w13,
        w13_scale=torch.ones(_E, 2 * _I, 1),
        w2_weight=w2,
        w2_scale=torch.ones(_E, hidden_size, 1),
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=hidden_size,
    )
    with pytest.raises(NotImplementedError, match="hidden_size % 128"):
        select_provider_cls("route_major", "fp8_channel", "cutedsl")(quant_info)


def test_fp8_channel_rejects_other_scale_layouts():
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

    w13, w2, _hidden, _topk_ids = _case(torch.device("cpu"))
    w13_q, w13_s = _channel_quant(w13)
    w2_q, w2_s = _channel_quant(w2)
    layer = type(
        "Layer",
        (),
        dict(
            w13_weight=w13_q,
            w13_weight_scale=w13_s.squeeze(-1),
            w2_weight=w2_q,
            w2_weight_scale=w2_s,
            num_local_experts=_E,
        ),
    )
    with pytest.raises(ValueError, match="w13_scale must be fp32"):
        MoeLoraFp8ChannelQuantInfo.from_layer(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
