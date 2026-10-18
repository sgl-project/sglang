"""A MoE layer owns its placement, and the fused funcs read it through the
layer's `MoeRunnerConfig`."""

import sys
from types import SimpleNamespace

import pytest
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


def test_a_config_without_a_layer_has_no_placement() -> None:
    config = MoeRunnerConfig()
    for name in ("moe_tp_size", "moe_tp_rank", "moe_ep_size", "moe_ep_rank"):
        with pytest.raises(AttributeError, match="has no layer"):
            getattr(config, name)


def test_the_config_reads_the_layer_placement(monkeypatch) -> None:
    layer = _layer(monkeypatch)
    assert _placement(layer) == (2, 1, 2, 1)
    assert _placement(layer.moe_runner_config) == _placement(layer)


def test_a_full_expert_bind_collapses_ep_on_the_config(monkeypatch) -> None:
    layer = _layer(monkeypatch)
    layer.bind_full_expert_weights({})
    assert _placement(layer) == (2, 1, 1, 0)
    assert _placement(layer.moe_runner_config) == (2, 1, 1, 0)
    assert layer.moe_runner_config.num_local_experts == 4


def test_the_kernel_gets_the_bound_placement(monkeypatch) -> None:
    """The reason the placement lives on the layer: after a full-expert bind
    the kernel must be told this rank owns every expert."""
    layer = _layer(monkeypatch)
    before = _kernel_placement(monkeypatch, layer.moe_runner_config)
    assert before == (2, 1, 2, 1)

    layer.bind_full_expert_weights({})

    # The layer kept its TP shard and now owns every expert.
    assert _kernel_placement(monkeypatch, layer.moe_runner_config) == (2, 1, 1, 0)


def _kernel_placement(monkeypatch, runner_config):
    """Run the CUTLASS fused func against a fake kernel and report the
    (tp_size, tp_rank, ep_size, ep_rank) it was handed."""
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
    flashinfer_cutlass._run_flashinfer_cutlass(
        dispatch_output=SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=SimpleNamespace(
                topk_weights=torch.ones(2, 1),
                topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            ),
        ),
        quant_info=FlashInferCutlassMoeQuantInfo(
            quant_type="bf16", w13_weight=torch.zeros(1), w2_weight=torch.zeros(1)
        ),
        runner_config=runner_config,
        output=torch.empty_like(hidden_states),
    )
    (call,) = calls
    return (call["tp_size"], call["tp_rank"], call["ep_size"], call["ep_rank"])


def test_cutlass_passes_the_runner_config_placement(monkeypatch) -> None:
    placement = _kernel_placement(
        monkeypatch,
        MoeRunnerConfig(
            layer=SimpleNamespace(
                moe_tp_size=2, moe_tp_rank=1, moe_ep_size=4, moe_ep_rank=3
            )
        ),
    )
    assert placement == (2, 1, 4, 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
