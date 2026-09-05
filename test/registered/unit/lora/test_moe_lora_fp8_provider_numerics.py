"""FP8 providers against an unquantized fp32 reference.

Pair-major comparison through each provider's own ``pair_to_row``, so the two row
dispatchers' within-expert row order never enters the check. Tolerances cover
fp8 weight + per-token-group activation rounding.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

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


def _case(device):
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, 2 * _I, _H)
    w2 = rand(_E, _H, _I)
    hidden = rand(_T, _H)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    return w13, w2, hidden, topk_ids


def _quant_info(w13, w2, scale_form="plain"):
    """Quantize like the serving load path, and return the exact effective
    (dequantized) weights so the reference isolates activation-quant error.
    scale_form mirrors what each vendor's runner branch leaves on the layer:
    'plain' = untouched
    fp32 scales (triton)."""
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
        hidden_size=_H,
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


@pytest.mark.parametrize(
    "vendor,rows",
    [
        ("triton", "route_major"),
        ("cutedsl", "expert_major"),
        ("cutedsl", "route_major"),
    ],
)
def test_fp8_gateup_and_down_match_reference(vendor, rows):
    _skip_unless_supported(vendor)
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device)
    scale_form = "plain"  # every vendor serves the checkpoint bytes
    quant_info, w13_eff, w2_eff = _quant_info(w13, w2, scale_form=scale_form)

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
    from sglang.srt.lora.moe.quant_info import admit_fp8_block_weights

    with pytest.raises(NotImplementedError, match=r"\[128, 128\]"):
        admit_fp8_block_weights(bad)


def test_fp8_kernel_rejects_tiles_the_scale_epilogue_cannot_index():
    """The SM100 FP8 epilogue indexes weight scales per 128-wide output tile and
    stages one column scale per epilogue thread (128 threads)."""
    cutlass = pytest.importorskip("cutlass")
    from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm100_fp8 import (
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

    build((128, 8))
    build((128, 128))
    for mma_tiler_mn in ((64, 128), (128, 256)):
        with pytest.raises(ValueError, match="mma_tiler_mn"):
            build(mma_tiler_mn)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
