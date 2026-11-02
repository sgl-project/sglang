"""Native MiMo adapter/runner regression for supported unclamped configurations.

Defaults to AITER's Flash A16W4 CSV. Set AITER_CONFIG_FMOE before starting
pytest to repeat with the A8W4 CSV; kernel names are checked against it.
"""

import csv
import functools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.layers.quantization.mxfp4_aiter_a16w4_moe import (
    Mxfp4AiterA16W4MoEMethod,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)


def _normalized_error(expected, actual):
    x, y = expected.double(), actual.double()
    return float(
        (x - y).square().sum() / (x.square() + y.square()).sum().clamp_min(1e-12)
    )


@pytest.mark.parametrize("tokens", [1, 128])
@pytest.mark.parametrize("limit", [None, 0.0])
def test_native_adapter_unclamped_eager_and_graph(monkeypatch, tokens, limit):
    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import (
        fused_topk,
        get_2stage_cfgs,
        get_padded_M,
        torch_moe_stage1,
        torch_moe_stage2,
    )
    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl.moe_common import GateMode

    if get_gfx() != "gfx942":
        pytest.skip("Flash native recipes qualify gfx942")
    config_file = os.environ.get("AITER_CONFIG_FMOE") or str(
        Path(aiter.__file__).parent
        / "configs/model_configs/mimo_v2.6_flash_a16w4_tuned_fmoe.csv"
    )
    if not Path(config_file).is_file():
        pytest.skip("Set AITER_CONFIG_FMOE to the Flash native A16W4/A8W4 CSV")
    monkeypatch.setenv("AITER_CONFIG_FMOE", config_file)
    monkeypatch.setenv("SGLANG_USE_AITER_MOE_GU_ITLV", "1")
    monkeypatch.setattr(
        "sglang.srt.layers.moe.moe_runner.runner.get_moe_a2a_backend",
        lambda: MoeA2ABackend.NONE,
    )
    AITER_CONFIGS.get_config_file.cache_clear()
    get_2stage_cfgs.cache_clear()
    torch.manual_seed(505)
    hidden, inter, experts, topk = 4096, 256, 256, 8
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
    weights, ids = fused_topk(x, scores, topk, True)
    quantize = aiter.get_torch_quant(QuantType.per_1x32)
    q1, s1 = quantize(w1, quant_dtype=dtypes.fp4x2)
    q2, s2 = quantize(w2, quant_dtype=dtypes.fp4x2)
    q1 = q1.view(dtypes.fp4x2).view(experts, 2 * inter, hidden // 2)
    q2 = q2.view(dtypes.fp4x2).view(experts, hidden, inter // 2)

    def reference(clamp):
        mid = torch_moe_stage1(
            x,
            q1,
            q2,
            weights,
            ids,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1.view(experts, 2 * inter, hidden // 32),
            swiglu_limit=clamp,
            doweight=False,
        )
        return torch_moe_stage2(
            mid.view(tokens, topk, inter),
            q1,
            q2,
            weights,
            ids,
            dtype=torch.bfloat16,
            quant_type=QuantType.per_1x32,
            w2_scale=s2.view(experts, hidden, inter // 32),
            doweight=True,
        )

    expected = reference(None)
    layer = torch.nn.Module()
    for name, tensor in (
        ("w13_weight", q1),
        ("w2_weight", q2),
        ("w13_weight_scale_inv", s1.view(experts, 2 * inter, hidden // 32)),
        ("w2_weight_scale_inv", s2.view(experts, hidden, inter // 32)),
    ):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.dispatcher = SimpleNamespace(
        expert_mask_gpu=None, set_quant_config=lambda _: None
    )
    method = Mxfp4AiterA16W4MoEMethod(None, "test.native_clamp")
    method.process_weights_after_loading(layer)
    method.create_moe_runner(
        layer, MoeRunnerConfig(activation="silu", top_k=topk, swiglu_limit=limit)
    )
    dispatch = StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, None))

    def forward():
        return method.apply(layer, dispatch).hidden_states

    actual = forward()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    assert _normalized_error(expected, actual) < 1e-2
    meta = get_2stage_cfgs(
        get_padded_M(tokens),
        hidden,
        inter,
        experts,
        topk,
        torch.bfloat16,
        torch.bfloat16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value,
        swiglu_limit=limit,
    )
    with open(config_file) as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if int(r["token"]) == get_padded_M(tokens)
            and int(r["inter_dim"]) == inter
            and int(r["model_dim"]) == hidden
            and int(r["expert"]) == experts
        ]
    assert len(rows) == 1
    for stage, key in ((meta.stage1, "kernelName1"), (meta.stage2, "kernelName2")):
        assert isinstance(stage, functools.partial)
        assert rows[0][key] in repr(stage.keywords)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = forward()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(graph_output).all()
        assert _normalized_error(expected, graph_output) < 1e-2
