import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.hc_mix_flydsl import (
    flydsl_hc_mix_supported,
    flydsl_hc_mix_weights_supported,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires ROCm gfx950",
)


def test_gated_residual_dispatch():
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    from aiter.ops.flydsl.hc_mix import hc_mix

    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(hidden_size=2560, hc_lowrank=320)
    layer = GatedResidual(cfg, use_combine=False).cuda().bfloat16()
    x = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with patch("aiter.ops.flydsl.hc_mix.hc_mix", wraps=hc_mix) as call:
            result, (_, normed, _) = layer.mix(x)
            assert call.call_count == 1
        d, u = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
        assert layer._mix_flydsl_ok
        with patch(
            "sglang.srt.layers.hc_mix_flydsl._aiter_hc_mix_available",
            return_value=False,
        ):
            assert not flydsl_hc_mix_weights_supported(d, u, 4)
        t = torch.nn.functional.silu(normed.float() @ d.float().T / 4).bfloat16()
        ref = (
            ((t.float() @ u.float().T).sigmoid() * normed.float())
            .view(4, 4, 2560)
            .mean(1)
            .bfloat16()
        )
        torch.testing.assert_close(result, ref, atol=0.004, rtol=0.01)
        packed = layer._mix_flydsl_weights
        layer.mix(x)
        assert layer._mix_flydsl_weights is packed
        d.mul_(0.5)
        layer.mix(x)
        assert layer._mix_flydsl_weights is not packed
        assert flydsl_hc_mix_supported(x.repeat(32, 1), True)
        for bad_x in [x[:0], x.repeat(33, 1), x.half(), x[:, ::2]]:
            assert not flydsl_hc_mix_supported(bad_x, True)
        assert not flydsl_hc_mix_supported(x, False)
        assert not flydsl_hc_mix_weights_supported(d, u, 5)
        # K only needs whole 64-wide tiles, not the old 2048 multiple.
        k = 64 * 33
        assert flydsl_hc_mix_weights_supported(d[:, :k], u[:k], 4)
        assert not flydsl_hc_mix_weights_supported(d[:, : k - 32], u[: k - 32], 4)
        assert not flydsl_hc_mix_weights_supported(d[:16], u[:, :16], 4)
        assert not flydsl_hc_mix_weights_supported(d.half(), u.half(), 4)


@pytest.mark.parametrize("rows", [32, 96, 128])
def test_gated_residual_dispatch_many_rows(rows):
    # Speculative verify passes carry requests x draft tokens rows.
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    from aiter.ops.flydsl.hc_mix import hc_mix

    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(hidden_size=2560, hc_lowrank=320)
    layer = GatedResidual(cfg, use_combine=False).cuda().bfloat16()
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with patch("aiter.ops.flydsl.hc_mix.hc_mix", wraps=hc_mix) as call:
            result, (_, normed, _) = layer.mix(x)
            assert call.call_count == 1
        d, u = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
        t = torch.nn.functional.silu(normed.float() @ d.float().T / 4).bfloat16()
        ref = (
            ((t.float() @ u.float().T).sigmoid() * normed.float())
            .view(rows, 4, 2560)
            .mean(1)
            .bfloat16()
        )
        torch.testing.assert_close(result, ref, atol=0.004, rtol=0.01)


def test_inference_tensor_weight_cache():
    from sglang.srt.layers.hc_mix_flydsl import weight_cache_key

    with torch.inference_mode():
        w = torch.ones(4)
        assert weight_cache_key(w, w)[1] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
