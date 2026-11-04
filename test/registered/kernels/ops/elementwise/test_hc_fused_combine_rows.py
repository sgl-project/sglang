"""HC boundary at 33-128 rows through the side-stream gate and fused combine + norm: bitwise to the default path.

Runs the decoder's HC loop (prepared norm and gate partials carried between boundaries, a PLE-like read without a
prepared norm, the terminal mixer) on real GatedResidual modules with SGLANG_HC_FUSED_COMBINE_MAX_ROWS off and at
128, at rows 1-160, and compares every boundary's output bit for bit.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the HC decode kernels are SM10x only",
    ),
]

HC, HS, LOWRANK, LAYERS = 4, 2560, 16, 3
PLE_LIKE = {1}
ROWS = list(range(1, 161))


def _build(hcm, max_rows: int, seed: int = 0):
    cfg = hcm.HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with mock.patch.object(hcm, "_FUSED_COMBINE_MAX_ROWS", max_rows):
            layers = [
                (hcm.GatedResidual(cfg), hcm.GatedResidual(cfg)) for _ in range(LAYERS)
            ]
            mixer = hcm.GatedResidual(cfg, use_combine=False)
    finally:
        torch.set_default_dtype(prev)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in [m for pair in layers for m in pair] + [mixer]:
            mod.input_mix_weight_down.weight.copy_(
                torch.randn(LOWRANK, HC * HS, generator=g) * 0.01
            )
            mod.input_mix_weight_up.weight.copy_(
                torch.randn(HC * HS, LOWRANK, generator=g) * 0.05
            )
            mod.hc_norm.weight.copy_(torch.randn(HC * HS, generator=g) * 0.1)
            if hasattr(mod, "block_inject_weight"):
                mod.block_inject_weight.weight.copy_(
                    torch.randn(HC, HC * HS, generator=g) * 0.01
                )
            mod.cuda().eval()
    return layers, mixer


def _forward(layers, mixer, x, pe):
    """Qwen4ExpModel's HC loop; returns every boundary's output and the final mix."""
    outs = []
    normalized = partials = None
    residual = x
    for i, (attn, ffn) in enumerate(layers):
        if i + 1 < len(layers) and i + 1 not in PLE_LIKE:
            next_norm = layers[i + 1][0].hc_norm
            next_inject = layers[i + 1][0].block_inject_weight.weight
        elif i + 1 < len(layers):
            next_norm = next_inject = None
        else:
            next_norm, next_inject = mixer.hc_norm, None
        if i in PLE_LIKE:
            residual = residual + pe
            h, res = attn.mix(residual)
        else:
            h, res = attn.mix(residual, normalized_input=normalized, partials=partials)
        residual, ffn_in, ffn_partials = attn.combine_and_normalize(
            h, res, ffn.hc_norm, ffn.block_inject_weight.weight
        )
        outs += [h, residual]
        h, res = ffn.mix(residual, normalized_input=ffn_in, partials=ffn_partials)
        residual, normalized, partials = ffn.combine_and_normalize(
            h, res, next_norm, next_inject
        )
        outs += [h, residual]
    out, _ = mixer.mix(residual, normalized_input=normalized)
    return outs + [out]


def _inputs(rows: int):
    g = torch.Generator(device="cuda").manual_seed(1000 + rows)
    scale = torch.exp(torch.randn(HC * HS, device="cuda", generator=g) * 0.5)
    scale[torch.randint(0, HC * HS, (16,), device="cuda", generator=g)] *= 20.0
    x = (torch.randn(rows, HC * HS, device="cuda", generator=g) * scale).to(
        torch.bfloat16
    )
    pe = (torch.randn(rows, HC * HS, device="cuda", generator=g) * 0.1).to(
        torch.bfloat16
    )
    return x, pe


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.fixture(scope="module")
def hcm():
    from sglang.srt.layers import hyperconnection

    with mock.patch.object(
        hyperconnection, "get_parallel", lambda: SimpleNamespace(tp_size=1)
    ):
        yield hyperconnection


def test_rows_1_to_160_bitwise(hcm):
    from sglang.kernels.ops.elementwise import hc_combine_decode

    stock = _build(hcm, max_rows=0)
    fused = _build(hcm, max_rows=128)
    assert not stock[0][0]._fused_combine_ok and fused[0][0]._fused_combine_ok
    fused_rows = []
    real_apply_norm = hc_combine_decode.hc_combine_apply_norm

    def apply_norm(block, *args, **kwargs):
        fused_rows.append(block.shape[0])
        return real_apply_norm(block, *args, **kwargs)

    with (
        mock.patch.object(hcm, "_FUSED_COMBINE_MAX_ROWS", 128),
        mock.patch.object(hc_combine_decode, "hc_combine_apply_norm", apply_norm),
        torch.inference_mode(),
    ):
        for rows in ROWS:
            x, pe = _inputs(rows)
            fused_rows.clear()
            ref = _forward(*stock, x, pe)
            stock_rows = list(fused_rows)
            fused_rows.clear()
            torch.full(
                (64 << 20,), float("nan"), device="cuda"
            )  # poison the allocator's free blocks
            got = _forward(*fused, x, pe)
            torch.cuda.synchronize()
            for step, (a, b) in enumerate(zip(ref, got)):
                assert torch.equal(_bits(a), _bits(b)), (rows, step)
            # the specialization that ran: the fused combine + norm at 33-128 rows only with the switch
            if 33 <= rows <= 128:
                assert rows in fused_rows and rows not in stock_rows, rows
            else:
                assert fused_rows == stock_rows, rows

    flipped = got[-1].clone()
    flipped.view(-1).view(torch.int16)[7] ^= 1
    assert (
        int((_bits(got[-1]) != _bits(flipped)).sum()) == 1
    )  # the comparison sees one bit


def test_control_fused_path_forced_wrong(hcm):
    """With the fused combine + norm forced wrong (another epsilon) the comparison must fail at 33-128 rows."""
    from sglang.kernels.ops.elementwise import hc_combine_decode

    stock = _build(hcm, max_rows=0)
    fused = _build(hcm, max_rows=128)
    real_apply_norm = hc_combine_decode.hc_combine_apply_norm

    def wrong_apply_norm(block, residual, partials, weight, eps, parts=2):
        return real_apply_norm(block, residual, partials, weight, 1e-2, parts=parts)

    with (
        mock.patch.object(hcm, "_FUSED_COMBINE_MAX_ROWS", 128),
        mock.patch.object(hc_combine_decode, "hc_combine_apply_norm", wrong_apply_norm),
        torch.inference_mode(),
    ):
        for rows in (33, 64, 128):
            x, pe = _inputs(rows)
            ref = _forward(*stock, x, pe)
            got = _forward(*fused, x, pe)
            assert any(not torch.equal(_bits(a), _bits(b)) for a, b in zip(ref, got)), (
                rows
            )
