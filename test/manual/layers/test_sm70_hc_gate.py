import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native HC injection gate requires V100",
)


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("seed", [7, 31, 113])
@torch.no_grad()
def test_precomputed_gate_matches_mix_combine_and_graph(batch, seed, monkeypatch):
    from sglang.srt.layers.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    torch.manual_seed(seed)
    layer = (
        GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560,
                hc_count=4,
                hc_lowrank=320,
                hc_per_branch_norm=True,
                params_dtype=torch.float16,
            )
        )
        .cuda()
        .half()
    )
    for parameter in layer.parameters():
        parameter.normal_(std=0.01)
    x = torch.randn(batch, 10240, device="cuda", dtype=torch.float16)
    block_output = torch.randn(batch, 2560, device="cuda", dtype=torch.float16)

    def run(enabled):
        monkeypatch.setenv("SGLANG_SM70_MTP_HC_GATE", str(int(enabled)))
        mixed, residual = layer.mix(x)
        assert len(residual) == 2
        assert (layer._pending_gate is not None) == enabled
        return mixed, layer.combine(block_output, residual)

    run(False)
    run(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(True)
    for _ in range(4):
        x.normal_()
        block_output.normal_()
        reference = run(False)
        graph.replay()
        for got, expected in zip(actual, reference):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
