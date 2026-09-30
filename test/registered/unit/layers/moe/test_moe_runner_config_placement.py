"""A MoE layer's placement lives on its `MoeRunnerConfig`, and the fused funcs
read it from there."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-test-cpu-intel")


def _layer(monkeypatch) -> FusedMoE:
    method = UnquantizedFusedMoEMethod()
    monkeypatch.setattr(method, "create_weights", lambda **kwargs: None)
    monkeypatch.setattr(method, "create_moe_runner", lambda layer, config: None)
    monkeypatch.setattr(
        fused_moe_layer_module,
        "create_moe_dispatcher",
        lambda config, quant_method: SimpleNamespace(),
    )
    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.AUTO,
            a2a_backend=MoeA2ABackend.NONE,
        ),
        get_parallel().override(
            moe_ep_size=2,
            moe_ep_rank=1,
            moe_tp_size=2,
            moe_tp_rank=1,
            tp_size=4,
            tp_rank=3,
            attn_tp_size=4,
            attn_tp_rank=3,
        ),
    ):
        return FusedMoE(
            num_experts=4,
            hidden_size=4,
            intermediate_size=8,
            layer_id=0,
            quant_method=method,
        )


def _placement(obj):
    return (obj.moe_tp_size, obj.moe_tp_rank, obj.moe_ep_size, obj.moe_ep_rank)


def test_the_config_carries_the_layer_placement(monkeypatch) -> None:
    layer = _layer(monkeypatch)
    assert _placement(layer) == (2, 1, 2, 1)
    assert _placement(layer.moe_runner_config) == _placement(layer)


def test_a_full_expert_bind_collapses_ep_on_the_config(monkeypatch) -> None:
    layer = _layer(monkeypatch)
    layer.bind_full_expert_weights({})
    assert _placement(layer) == (2, 1, 1, 0)
    assert _placement(layer.moe_runner_config) == (2, 1, 1, 0)
    assert layer.moe_runner_config.num_local_experts == 4


def test_cutlass_passes_the_runner_config_placement(monkeypatch) -> None:
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [kwargs["output"]]

    monkeypatch.setattr(
        flashinfer_cutlass,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fake_kernel, None),
    )
    monkeypatch.setattr(flashinfer_cutlass, "_activation_type", lambda config: None)

    hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)
    dispatch_output = SimpleNamespace(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        ),
    )
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="bf16",
        w13_weight=torch.zeros(1),
        w2_weight=torch.zeros(1),
    )
    runner_config = MoeRunnerConfig(
        moe_tp_size=2, moe_tp_rank=1, moe_ep_size=4, moe_ep_rank=3
    )
    flashinfer_cutlass._run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=torch.empty_like(hidden_states),
    )

    (call,) = calls
    assert (call["tp_size"], call["tp_rank"], call["ep_size"], call["ep_rank"]) == (
        2,
        1,
        4,
        3,
    )
