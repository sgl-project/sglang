"""NPU HC normalization, mixing and residual injection."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


@pytest.mark.parametrize(
    "rows,capture,use_combine",
    [
        pytest.param(1, True, True, id="qwen3.8-flash-next-decode-t1-graph"),
        pytest.param(4, False, False, id="qwen3.8-flash-next-verify-t4-final-mix"),
        pytest.param(16, False, True, id="qwen3.8-flash-next-prefill-t16"),
    ],
)
@torch.no_grad()
def test_hc_outputs(rows, capture, use_combine):
    torch.manual_seed(17)
    # HC width stays 4 * 2560 on every TP rank.
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=2560, hc_lowrank=320, hc_per_branch_norm=True
        ),
        use_combine=use_combine,
    ).to(device="npu", dtype=torch.bfloat16)
    module.hc_norm.weight.normal_(std=0.1)
    x = torch.randn(rows, 10240, dtype=torch.bfloat16, device="npu")
    block = torch.randn(rows, 2560, dtype=x.dtype, device=x.device)

    def forward():
        mixed, residual = module.mix(x)
        return mixed, module.combine(block, residual) if use_combine else None

    for _ in range(2):
        forward()
    graph = None
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_mix, actual_combine = forward()

    for step in range(2 if capture else 1):
        if step:
            x.mul_(-0.5)
            block.add_(0.25)
        if graph is None:
            actual_mix, actual_combine = forward()
        else:
            graph.replay()
            torch.npu.synchronize()

        grouped = x.cpu().float().reshape(rows, 4, 2560)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + 1e-6
        )
        normalized = normalized.flatten(1) * (1 + module.hc_norm.weight.cpu().float())
        down = F.linear(normalized, module.input_mix_weight_down.weight.cpu().float())
        gates = F.linear(
            F.silu(down / 4), module.input_mix_weight_up.weight.cpu().float()
        )
        expected_mix = (normalized * gates.sigmoid()).reshape(rows, 4, 2560).mean(1)
        torch.testing.assert_close(
            actual_mix.cpu().float(), expected_mix, atol=2e-2, rtol=2e-2
        )
        if use_combine:
            gates = (
                2
                * (
                    F.linear(
                        normalized, module.block_inject_weight.weight.cpu().float()
                    )
                    / 4
                ).sigmoid()
            )
            expected_combine = (
                grouped + gates[..., None] * block.cpu().float()[:, None, :]
            )
            torch.testing.assert_close(
                actual_combine.cpu().float(),
                expected_combine.flatten(1),
                atol=2e-2,
                rtol=2e-2,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
