import pytest
import torch
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Requires SM70",
)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32])
def test_combine_norm_matches_existing_boundaries_and_replays(rows, monkeypatch):
    from sglang.kernels.ops.layernorm.sm70_hc_combine_norm import (
        combine_norm,
        supported,
    )

    monkeypatch.setenv("SGLANG_SM70_HC_COMBINE_NORM", "1")
    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "1")
    torch.manual_seed(270 + rows)
    cfg = HyperConnectionConfig(
        hc_count=4,
        hidden_size=2560,
        hc_lowrank=320,
        params_dtype=torch.float16,
        hc_per_branch_norm=True,
    )
    layer = GatedResidual(cfg).cuda().half()
    following = GatedResidual(cfg).cuda().half()
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.float16)
    y = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        layer.hc_norm.weight.normal_(std=0.1)
        following.hc_norm.weight.normal_(std=0.1)
        _, residuals = layer.mix(x)
        assert supported(layer, y, residuals, following.hc_norm)
        combine_norm(layer, y, residuals, following.hc_norm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, normalized = layer.combine_and_norm(
                y, residuals, following.hc_norm
            )
        for scale in [1.0, -0.5, 0.0]:
            y.mul_(scale)
            graph.replay()
            expected = layer.combine(y, residuals)
            expected_norm = following.hc_norm(expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(normalized, expected_norm, rtol=0, atol=0)
        if rows in (2, 4):
            baseline_mix, _ = following.mix(actual)
            fused_mix, _ = following.mix(actual, normed_input=normalized)
            torch.testing.assert_close(fused_mix, baseline_mix, rtol=0, atol=0)
    monkeypatch.setenv("SGLANG_SM70_HC_COMBINE_NORM", "0")
    assert not supported(layer, y, residuals, following.hc_norm)
