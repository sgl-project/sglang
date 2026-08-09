from __future__ import annotations

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl as cutedsl
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    CuteDslFp4MoeQuantInfo,
    fused_experts_flashinfer_to_flashinfer_cutedsl_fp4,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class _DirectWrapper:
    def __init__(self):
        self.output = None

    def run(self, *, x, moe_output=None, **kwargs):
        assert moe_output is not None
        self.output = moe_output
        return moe_output.fill_(3)


class _LegacyWrapper:
    def run(self, *, x, **kwargs):
        return torch.full((x.shape[0], 8), 5, dtype=torch.bfloat16)


def _run(wrapper):
    rows, hidden = 4, 8
    workspace = torch.empty(rows, hidden, dtype=torch.bfloat16)
    topk = StandardTopKOutput(
        torch.ones(rows, 2), torch.zeros(rows, 2, dtype=torch.int32), torch.empty(0)
    )
    dispatch = FlashinferDispatchOutput(
        torch.empty(rows, hidden // 2, dtype=torch.uint8),
        torch.ones(rows, 1),
        topk,
        workspace,
    )
    dummy = torch.empty(0)
    quant_info = CuteDslFp4MoeQuantInfo(
        w13_weight=dummy,
        w2_weight=dummy,
        w13_weight_sf=dummy,
        w2_weight_sf=dummy,
        w1_alpha=dummy,
        w2_alpha=dummy,
        a1_scale=dummy,
        a2_scale=dummy,
        wrapper=wrapper,
    )
    cutedsl._wrapper_run_supports_moe_output = None
    result = fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
        dispatch, quant_info, MoeRunnerConfig(activation="silu")
    )
    return result.hidden_states, workspace


def test_direct_output_uses_combine_workspace_without_copy():
    wrapper = _DirectWrapper()
    output, workspace = _run(wrapper)
    assert wrapper.output is workspace
    assert output.data_ptr() == workspace.data_ptr()
    torch.testing.assert_close(output, torch.full_like(output, 3))


def test_legacy_wrapper_falls_back_to_workspace_copy():
    output, workspace = _run(_LegacyWrapper())
    assert output.data_ptr() == workspace.data_ptr()
    torch.testing.assert_close(output, torch.full_like(output, 5))
