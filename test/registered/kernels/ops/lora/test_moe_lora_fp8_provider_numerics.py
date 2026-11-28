"""Compare native FP8 MoE providers with dequantized references."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_lora import gather_pairs, rel_l2

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E = 4
_H = 512
_I = 512
_TOPK = 2
_T = 32
_BLOCK = 128
_FMAX = torch.finfo(torch.float8_e4m3fn).max


def _skip_unless_supported(vendor="cutedsl"):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if vendor == "cutedsl" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the FP8 CuTeDSL provider needs SM90+")


def _block_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, rows, cols = weight.shape
    blocks = weight.float().reshape(e, rows // _BLOCK, _BLOCK, cols // _BLOCK, _BLOCK)
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / _FMAX
    codes = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return codes.reshape_as(weight), scale


def _case(device, hidden_size=_H, num_tokens=_T):
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, 2 * _I, hidden_size)
    w2 = rand(_E, hidden_size, _I)
    hidden = rand(num_tokens, hidden_size)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(num_tokens)]
    ).to(device=device, dtype=torch.int32)
    return w13, w2, hidden, topk_ids


def _quant_info(w13, w2, hidden_size=_H):
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

    w13_q, w13_s = _block_quant(w13)
    w2_q, w2_s = _block_quant(w2)
    w13_eff = block_quant_dequant(w13_q, w13_s, [128, 128], torch.float32)
    w2_eff = block_quant_dequant(w2_q, w2_s, [128, 128], torch.float32)
    info = MoeLoraFp8QuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        block_shape=(128, 128),
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=hidden_size,
    )
    return info, w13_eff, w2_eff


def _channel_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / _FMAX
    return (weight.float() / scale).to(torch.float8_e4m3fn), scale


def _channel_info(w13, w2, hidden_size=_H):
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


def _tensor_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=(1, 2)).clamp(min=1e-6) / _FMAX
    return (weight.float() / scale[:, None, None]).to(torch.float8_e4m3fn), scale


def _fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = (x.float() / scale).clamp(-_FMAX, _FMAX).to(torch.float8_e4m3fn)
    return q.float() * scale


def _tensor_info(w13, w2, hidden, act_pairs, static: bool, hidden_size=_H):
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

    w13_q, w13_s = _tensor_quant(w13)
    w2_q, w2_s = _tensor_quant(w2)
    a13 = a2 = None
    if static:
        a13 = (hidden.float().abs().amax() * 0.4 / _FMAX).reshape(())
        a2 = (act_pairs.float().abs().amax() * 0.4 / _FMAX).reshape(())
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


def _check_gemms(
    provider,
    hidden,
    topk_ids,
    act_pairs,
    w13_eff,
    w2_eff,
    tol,
    *,
    input_scales=None,
    row_steps=None,
    poison_dropped=False,
    use_prequantized_down=False,
):
    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = pair_idx // _TOPK
    ref_hidden, ref_act = hidden, act_pairs
    if input_scales is not None:
        ref_hidden = _fake_quant(hidden, input_scales[0])
        ref_act = _fake_quant(act_pairs, input_scales[1])
    ref_gateup = torch.einsum(
        "ph,pnh->pn", ref_hidden[tokens].float(), w13_eff[experts]
    )
    ref_down = torch.einsum("pi,phi->ph", ref_act.float(), w2_eff[experts])
    gate_unit, down_unit = row_steps if row_steps is not None else (1, 1)

    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = hidden.new_empty(provider.gateup_out_shape(state))
        provider.gateup(state, gateup)
        act = hidden.new_zeros(provider.act_out_shape(state))
        if poison_dropped:
            dropped = torch.nonzero(flat_ids < 0, as_tuple=True)[0]
            act[dropped[::2]] = float("nan")
            act[dropped[1::2]] = 1e4
        act.reshape(-1, _I)[state.pair_to_row[pair_idx].long()] = act_pairs
        down = hidden.new_empty(provider.down_out_shape(state))
        provider.down(state, act, down)
        got_gateup = gather_pairs(gateup, state.pair_to_row, pair_idx).float()
        got_down = gather_pairs(down, state.pair_to_row, pair_idx).float()
        assert rel_l2(got_gateup / gate_unit, ref_gateup / gate_unit) < tol
        assert rel_l2(got_down / down_unit, ref_down / down_unit) < tol

        if use_prequantized_down:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            state.act_quant = sglang_per_token_group_quant_fp8(act, _BLOCK)
            state.act_quant_ready = True
            # The ready operands must bypass the BF16 activation.
            act.fill_(float("nan"))
            down.fill_(float("nan"))
            provider.down(state, act, down)
            assert (
                rel_l2(
                    gather_pairs(down, state.pair_to_row, pair_idx).float(), ref_down
                )
                < tol
            )

    return state, gateup, act


def _activations(num_pairs, device):
    g = torch.Generator(device="cpu").manual_seed(11)
    return (torch.randn(num_pairs, _I, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )


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
def test_fp8_gateup_and_down_match_reference(vendor, rows, hidden_size):
    _skip_unless_supported(vendor)
    w13, w2, hidden, topk_ids = _case("cuda", hidden_size)
    info, w13_eff, w2_eff = _quant_info(w13, w2, hidden_size)
    _check_gemms(
        select_provider_cls(rows, "fp8", vendor)(info),
        hidden,
        topk_ids,
        _activations(topk_ids.numel(), hidden.device),
        w13_eff,
        w2_eff,
        4e-2,
        use_prequantized_down=vendor == "triton",
    )


def test_fp8_channel_gateup_and_down_match_reference(
    vendor="triton", rows="route_major", hidden_size=_H
):
    _skip_unless_supported(vendor)
    w13, w2, hidden, topk_ids = _case("cuda", hidden_size)
    info, w13_eff, w2_eff = _channel_info(w13, w2, hidden_size)
    _check_gemms(
        select_provider_cls(rows, "fp8_channel", vendor)(info),
        hidden,
        topk_ids,
        _activations(topk_ids.numel(), hidden.device),
        w13_eff,
        w2_eff,
        6e-2,
    )


@pytest.mark.parametrize(
    "rows,token_width",
    [("expert_major", 8), ("route_major", 64), ("route_major", 128)],
)
def test_fp8_channel_cutedsl_pairs_scales(rows, token_width, monkeypatch):
    _skip_unless_supported()
    w13, w2, hidden, topk_ids = _case("cuda", num_tokens=320)

    # Distinct powers of two expose scales borrowed from a neighboring row.
    def steps(n):
        return 2.0 ** (torch.arange(n, device=hidden.device) % 7 - 3)

    w13_step = steps(w13.shape[1])
    w2_step = steps(w2.shape[1])
    token_step = steps(hidden.shape[0])
    pair_step = steps(topk_ids.numel())
    info, w13_eff, w2_eff = _channel_info(
        w13 * w13_step[:, None], w2 * w2_step[:, None], hidden.shape[1]
    )
    provider = select_provider_cls(rows, "fp8_channel", "cutedsl")(info)
    monkeypatch.setattr(provider, "_token_width_for", lambda *_: token_width)
    state, gateup, act = _check_gemms(
        provider,
        (hidden * token_step[:, None]).to(torch.bfloat16),
        topk_ids,
        (_activations(topk_ids.numel(), hidden.device) * pair_step[:, None]).to(
            torch.bfloat16
        ),
        w13_eff,
        w2_eff,
        6e-2,
        row_steps=(
            token_step.repeat_interleave(_TOPK)[:, None] * w13_step[None, :],
            pair_step[:, None] * w2_step[None, :],
        ),
    )
    assert state.token_width == token_width

    if rows == "expert_major":
        with get_context().override_server_args():
            act.zero_()
            # Distinct sentinels expose unwritten scales even with reused buffers.
            state.act_scale.fill_(-1)
            lora_in = act.new_zeros((*topk_ids.shape, _I))
            provider.act_with_delta(state, gateup, None, topk_ids, act, lora_in)
            assert state.act_quant_ready
            pair_rows = state.pair_to_row[topk_ids.flatten() >= 0].long()
            fused_q = state.act_fp8.view(-1, _I)[pair_rows].view(torch.uint8)
            fused_s = state.act_scale.view(-1)[pair_rows]
            state.act_quant_ready = False
            state.act_scale.fill_(-2)
            separate_q, separate_s = provider._down_operands(state, act)
            assert torch.equal(
                fused_q, separate_q.view(-1, _I)[pair_rows].view(torch.uint8)
            )
            assert torch.equal(fused_s, separate_s.view(-1)[pair_rows])


def _check_tensor(hidden_size, static):
    _skip_unless_supported("triton")
    w13, w2, hidden, topk_ids = _case("cuda", hidden_size)
    if not static:
        topk_ids[::2, 1] = -1
        topk_ids[5] = -1
    act_pairs = _activations(int((topk_ids >= 0).sum()), hidden.device)
    info, w13_eff, w2_eff = _tensor_info(
        w13, w2, hidden, act_pairs, static, hidden_size
    )
    _check_gemms(
        select_provider_cls("route_major", "fp8_tensor", "triton")(info),
        hidden,
        topk_ids,
        act_pairs,
        w13_eff,
        w2_eff,
        1e-2 if static else 8e-2,
        input_scales=(info.w13_input_scale, info.w2_input_scale) if static else None,
        poison_dropped=not static,
    )


def test_fp8_tensor_dynamic_scales_match_reference(hidden_size=_H):
    # Stale nonfinite and large finite rows must not affect the dynamic scale.
    _check_tensor(hidden_size, static=False)


def test_fp8_tensor_static_scales_saturate_at_the_checkpoint():
    _check_tensor(_H, static=True)


def test_fp8_kernel_rejects_tiles_the_scale_epilogue_cannot_index():
    cutlass = pytest.importorskip("cutlass")
    from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm100_fp8 import (
        GroupedGemmKernelSm100Fp8,
    )

    def build(mma_tiler_mn):
        return GroupedGemmKernelSm100Fp8(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=(1, 1),
            swap_ab=True,
        )

    for mma_tiler_mn in ((64, 128), (128, 256)):
        with pytest.raises(ValueError, match="mma_tiler_mn"):
            build(mma_tiler_mn)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
