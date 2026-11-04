"""FP8 providers against FP32 references from dequantized checkpoint bytes.
Tolerances cover per-token-group activation quantization and BF16 outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def _quant_info(w13, w2, hidden_size=_H):
    """Quantize weights locally and use serving dequantization for the FP32 reference."""
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
        ("cutedsl", "expert_major"),
        ("cutedsl", "route_major"),
    ],
)
def test_fp8_gateup_and_down_match_reference(vendor, rows, hidden_size):
    _skip_unless_supported(vendor)
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    quant_info, w13_eff, w2_eff = _quant_info(w13, w2, hidden_size=hidden_size)

    from sglang.srt.runtime_context import get_context

    provider = select_provider_cls(rows, "fp8", vendor)(quant_info)
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

    if vendor == "triton":
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        state.act_quant = sglang_per_token_group_quant_fp8(act, _BLOCK)
        # The supplied FP8 operands must bypass the unused BF16 activation.
        act.fill_(float("nan"))
        down.fill_(float("nan"))
        with get_context().override_server_args():
            provider.down(state, act, down)
        got_down = _gather_pairs(down, pair_to_row, pair_idx).float()
        assert _rel_l2(got_down, ref_down) < 4e-2


def test_fp8_admission_rejects_bad_geometry():
    from sglang.srt.lora.moe.quant_info import _admit_fp8_block_weights

    with pytest.raises(NotImplementedError, match=r"\[128, 128\]"):
        _admit_fp8_block_weights(SimpleNamespace(block_shape=(64, 64)))


def test_fp8_kernel_rejects_tiles_the_scale_epilogue_cannot_index():
    """The SM100 FP8 epilogue indexes weight scales per 128-wide output tile and
    stages one column scale per epilogue thread (128 threads)."""
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
