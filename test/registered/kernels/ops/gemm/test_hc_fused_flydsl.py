import sys

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires ROCm gfx950",
)


def _layers():
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    # Qwen4-Exp shapes: HS=2560, HC=4, R=320.
    cfg = HyperConnectionConfig(
        hidden_size=2560, hc_lowrank=320, hc_per_branch_norm=True
    )
    torch.manual_seed(0)
    ref = GatedResidual(cfg).cuda().bfloat16()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.02)
    with envs.SGLANG_ROCM_OPT_QWEN4_FUSED_HC.override(True):
        fused = GatedResidual(cfg).cuda().bfloat16()
    fused.load_state_dict(ref.state_dict())
    assert fused._fused_hc_ok and not ref._fused_hc_ok
    return ref, fused


@pytest.mark.parametrize("rows", [1, 24, 96, 128])
def test_fused_hc_matches_unfused(rows):
    ref, fused = _layers()
    # Scale branch b by b + 1 so a kernel that mixes up per-branch norm scales fails.
    branch_scale = torch.arange(1, 5, device="cuda", dtype=torch.float32)[:, None]
    h = (torch.randn(rows, 4, 2560, device="cuda") * branch_scale).flatten(1).bfloat16()
    y = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        m_ref, res_ref = ref.mix(h)
        out_ref = ref.combine_and_normalize(y, res_ref, ref.hc_norm)[0]
        m, res = fused.mix(h)
        assert res[2] is not None
        out, sumsq = fused.combine_and_normalize(y, res, fused.hc_norm)
        # Mix again on the combined residual, reusing the combine's sums of squares.
        m2_ref, _ = ref.mix(out_ref)
        m2, _ = fused.mix(out, prepared_norm=sumsq)
    # Only the rounding points differ (normed input and (1 + w) are not rounded to
    # BF16 separately), so allow a few BF16 ulps.
    for a, b in ((m, m_ref), (out, out_ref), (m2, m2_ref)):
        torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)


def test_fused_hc_falls_back_past_max_rows():
    ref, fused = _layers()
    h = torch.randn(129, 10240, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(129, 2560, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        m, res = fused.mix(h)
        assert res[2] is None
        out = fused.combine_and_normalize(y, res, fused.hc_norm)[0]
        m_ref, res_ref = ref.mix(h)
        out_ref = ref.combine(y, res_ref)
    assert torch.equal(m, m_ref) and torch.equal(out, out_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
