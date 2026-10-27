import sys

import pytest
import torch

from sglang.kernels.ops.gemm.hc_seam import (
    HC,
    HS,
    LOWRANK,
    hc_gated_combine,
    hc_seam,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="the HC seam is enabled on ROCm only"
)

EPS = 1e-6
D = HC * HS


def _stage(gen):
    def rnd(*shape, std):
        return (torch.randn(*shape, generator=gen, device="cuda") * std).to(
            torch.bfloat16
        )

    return dict(
        norm_w=rnd(D, std=0.1),
        w_down=rnd(LOWRANK, D, std=0.04),
        w_up=rnd(D, LOWRANK, std=0.1),
        w_inject=rnd(HC, D, std=0.04),
    )


def _activations(gen, rows):
    y = torch.randn(rows, HS, generator=gen, device="cuda").to(torch.bfloat16)
    r = (torch.randn(rows, D, generator=gen, device="cuda") * 3.0).to(torch.bfloat16)
    gate = 2.0 * torch.sigmoid(torch.randn(rows, HC, generator=gen, device="cuda"))
    return y, r, gate.contiguous()


def _reference(y, residual, gate_prev, s):
    """fp64 seam from the same bf16 inputs; R' and the gate's xn are rounded to bf16 as in the stock kernels."""
    f = torch.float64
    rows = residual.shape[0]
    if y is not None:
        a = gate_prev.to(f).repeat_interleave(HS, dim=1)
        residual = (residual.to(f) + a * y.to(f).repeat(1, HC)).to(torch.bfloat16)
    x = residual.to(f).view(rows, HC, HS)
    rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    xn = (x * rs * (1.0 + s["norm_w"].to(f).view(HC, HS))).view(rows, D)
    d = xn @ s["w_down"].to(f).t() / HC
    g = torch.sigmoid((d * torch.sigmoid(d)) @ s["w_up"].to(f).t()).view(rows, HC, HS)
    mixed = (g * xn.view(rows, HC, HS)).mean(1)
    gate = 2.0 * torch.sigmoid(
        xn.to(torch.bfloat16).to(f) @ s["w_inject"].to(f).t() / HC
    )
    return residual, mixed, gate


def _rel_l2(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


# 1-64 rows run one row tile, 65+ rows several.
@pytest.mark.parametrize("rows", [1, 17, 64, 65, 193])
@pytest.mark.parametrize("combine", [True, False])
def test_hc_seam_matches_fp64_reference(rows, combine):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    s = _stage(gen)
    y, r, gate_prev = _activations(gen, rows)
    if not combine:
        y = gate_prev = None
    r_new, mixed, gate = hc_seam(
        y, r, gate_prev, s["norm_w"], s["w_down"], s["w_up"], s["w_inject"], EPS
    )
    r_ref, mixed_ref, gate_ref = _reference(y, r, gate_prev, s)
    # R' is one fma and one bf16 rounding of the same fp32 operands.
    assert torch.equal(r_new.view(torch.int16), r_ref.view(torch.int16))
    assert _rel_l2(mixed, mixed_ref) < 6e-3
    assert (gate.double() - gate_ref).abs().max().item() < 2e-4


@pytest.mark.parametrize("rows", [17, 193])
def test_fused_combine_equals_separate_combine(rows):
    """seam_combine_mix's fused form must equal hc_gated_combine plus a mix-only
    seam bit for bit, so either form can serve a boundary."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    s = _stage(gen)
    y, r, gate_prev = _activations(gen, rows)
    w = (s["norm_w"], s["w_down"], s["w_up"], s["w_inject"], EPS)
    fused = hc_seam(y, r, gate_prev, *w)
    r_split = hc_gated_combine(y, r, gate_prev)
    _, mixed, gate = hc_seam(None, r_split, None, *w)
    for a, b in zip(fused, (r_split, mixed, gate)):
        assert torch.equal(a, b)


def _gated_residual(s):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    # The model loader builds layers under a bf16 default dtype.
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = GatedResidual(cfg).cuda()
    finally:
        torch.set_default_dtype(default)
    with torch.no_grad():
        m.hc_norm.weight.copy_(s["norm_w"])
        m.input_mix_weight_down.weight.copy_(s["w_down"])
        m.input_mix_weight_up.weight.copy_(s["w_up"])
        m.block_inject_weight.weight.copy_(s["w_inject"])
    return m


@torch.no_grad()
def test_gated_residual_seam_matches_stock(monkeypatch):
    """Mix, combine, mix across one boundary: the seam path against the stock path."""
    import sglang.srt.layers.hyperconnection as hcm

    gen = torch.Generator(device="cuda").manual_seed(0)
    prev, cur = _gated_residual(_stage(gen)), _gated_residual(_stage(gen))
    y, r, _ = _activations(gen, 32)

    def boundary():
        mixed0, res0 = prev.mix(r)
        r1 = prev.combine(y, res0)
        mixed1, _ = cur.mix(r1)
        return res0[1], mixed0, r1, mixed1

    _, *stock = boundary()
    monkeypatch.setattr(hcm, "_HC_SEAM", True)
    normed, *seam = boundary()
    assert normed is hcm.SEAM_NORMED
    mixed0, r1, mixed1 = seam
    assert _rel_l2(mixed0, stock[0]) < 1e-2
    # R' differs from stock only where the two gates' fp32 sums round differently.
    assert (r1 == stock[1]).float().mean().item() > 0.99
    assert _rel_l2(mixed1, stock[2]) < 1e-2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
