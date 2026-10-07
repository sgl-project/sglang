"""FP8 providers against an FP32 reference, compared through each pair_to_row map.
Tolerances cover weight and per-token-group activation quantization.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 512
_I = 512
_TOPK = 2
_T = 32
_BLOCK = 128


def _skip_unless_supported(vendor="cutedsl"):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if vendor == "cutedsl" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the FP8 CuTeDSL provider needs SM90+")


def _block_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-[128,128]-block fp8 quant; returns (fp8 weight, fp32 inverse scale)."""
    e, rows, cols = weight.shape
    rb = (rows + _BLOCK - 1) // _BLOCK
    cb = (cols + _BLOCK - 1) // _BLOCK
    scale = torch.empty((e, rb, cb), dtype=torch.float32, device=weight.device)
    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    for i in range(rb):
        for j in range(cb):
            blk = weight[
                :, i * _BLOCK : (i + 1) * _BLOCK, j * _BLOCK : (j + 1) * _BLOCK
            ]
            amax = blk.float().abs().amax(dim=(1, 2)).clamp(min=1e-6)
            s = amax / fmax
            scale[:, i, j] = s
            q[:, i * _BLOCK : (i + 1) * _BLOCK, j * _BLOCK : (j + 1) * _BLOCK] = (
                blk.float() / s[:, None, None]
            ).to(torch.float8_e4m3fn)
    return q, scale


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


def _quant_info(w13, w2, scale_form="plain", hidden_size=_H):
    """Use serving quantization and return effective weights to isolate activation error.
    Plain scale_form retains the Triton provider's FP32 scales.
    """
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

    def finalize(w_q, w_s):
        assert scale_form == "plain"
        eff = block_quant_dequant(w_q, w_s, [128, 128], torch.float32)
        return w_q, w_s, eff

    w13_q, w13_s, w13_eff = finalize(*_block_quant(w13))
    w2_q, w2_s, w2_eff = finalize(*_block_quant(w2))
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


def _valid_pairs(topk_ids):
    flat = topk_ids.flatten()
    return torch.nonzero(flat >= 0, as_tuple=True)[0], flat


def _gather_pairs(slab_or_rows, pair_to_row, pair_idx):
    rows = (
        slab_or_rows.reshape(-1, slab_or_rows.shape[-1])
        if slab_or_rows.ndim == 3
        else slab_or_rows
    )
    return rows[pair_to_row[pair_idx].long()]


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


# 7168 is a DeepSeek-V3-class hidden size: not a power of two.
@pytest.mark.parametrize("hidden_size", [_H, 7168])
@pytest.mark.parametrize(
    "vendor,rows",
    [
        ("triton", "route_major"),
    ],
)
def test_fp8_gateup_and_down_match_reference(vendor, rows, hidden_size):
    _skip_unless_supported(vendor)
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    scale_form = "plain"  # every vendor serves the checkpoint bytes
    quant_info, w13_eff, w2_eff = _quant_info(
        w13, w2, scale_form=scale_form, hidden_size=hidden_size
    )

    from sglang.srt.runtime_context import get_context

    provider = select_provider_cls(rows, "fp8", vendor)(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.empty(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)

    pair_idx, flat_ids = _valid_pairs(topk_ids)
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row

    ref_gateup = torch.einsum("ph,pnh->pn", hidden[tokens].float(), w13_eff[experts])
    got_gateup = _gather_pairs(gateup, pair_to_row, pair_idx).float()
    assert _rel_l2(got_gateup, ref_gateup) < 4e-2

    # Down over a synthetic activation scattered into the provider's layout.
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
    assert _rel_l2(got_down, ref_down) < 4e-2


def test_fp8_admission_rejects_bad_geometry():
    _skip_unless_supported()
    device = torch.device("cuda")
    w13, w2, _, _ = _case(device)
    quant_info, _, _ = _quant_info(w13, w2)
    import msgspec

    bad = msgspec.structs.replace(quant_info, block_shape=(64, 64))
    from sglang.srt.lora.moe.quant_info import _admit_fp8_block_weights

    with pytest.raises(NotImplementedError, match=r"\[128, 128\]"):
        _admit_fp8_block_weights(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
