import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.hc_mix_flydsl import flydsl_hc_mix_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires ROCm gfx950",
)


def test_disabled_does_not_require_gpu():
    with envs.SGLANG_AITER_HC_MIX.override(False):
        assert not flydsl_hc_mix_supported(None, None, None, 4, 2560)


def test_gated_residual_dispatch(monkeypatch):
    monkeypatch.delenv("SGLANG_AITER_HC_MIX", raising=False)
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    from aiter.ops.flydsl.hc_mix import hc_mix
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(hidden_size=2560, hc_lowrank=320)
    layer = GatedResidual(cfg, use_combine=False).cuda().bfloat16()
    x = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with patch("aiter.ops.flydsl.hc_mix.hc_mix", wraps=hc_mix) as call:
            result, (_, normed) = layer.mix(x)
            assert call.call_count == 1
        d, u = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
        with patch(
            "sglang.srt.layers.hc_mix_flydsl._aiter_hc_mix_available",
            return_value=False,
        ):
            assert not flydsl_hc_mix_supported(x, d, u, 4, 2560)
        t = torch.nn.functional.silu(normed.float() @ d.float().T / 4).bfloat16()
        ref = ((t.float() @ u.float().T).sigmoid() * normed.float()).view(4, 4, 2560).mean(1).bfloat16()
        torch.testing.assert_close(result, ref, atol=0.004, rtol=0.01)
        packed = layer._mix_flydsl_weights
        layer.mix(x)
        assert layer._mix_flydsl_weights is packed
        d.mul_(0.5)
        layer.mix(x)
        assert layer._mix_flydsl_weights is not packed
        for bad_x in [x[:0], x.repeat(5, 1), x.half(), x[:, ::2]]:
            assert not flydsl_hc_mix_supported(bad_x, d, u, 4, 2560)
        assert not flydsl_hc_mix_supported(x, d, u, 5, 2048)


def test_inference_tensor_weight_cache():
    from sglang.srt.layers.hc_mix_flydsl import weight_cache_key
    with torch.inference_mode():
        w = torch.ones(4)
        assert weight_cache_key(w, w)[1] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
