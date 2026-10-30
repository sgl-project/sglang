"""A MoE layer owns its placement, and the fused funcs read it through the
layer's `MoeRunnerConfig`."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.dwdp import dwdp_manager, replicate_routed_experts
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


def test_replicated_experts_are_complete_and_not_prefetched(monkeypatch) -> None:
    """DWDP feeds MoE DP-local tokens and sums nothing after the experts, so a
    MoE layer outside the target's prefetch (the MTP draft's) must hold every
    routed expert: each expert-sharded tensor, quant scales included, in global
    expert order and in the memory layout its kernel reads, and without waiting
    on the target's prefetch slots."""
    layer = _layer(monkeypatch)  # EP rank 1 of 2: local experts 2 and 3
    names_and_shapes = (
        ("w13_weight", (16, 4)),
        ("w2_weight", (4, 8)),
        ("w13_weight_scale_inv", (3, 4)),
    )
    for name, shape in names_and_shapes:
        # Every element of an expert's slice holds that expert's global id.
        local = torch.tensor([2.0, 3.0]).view(2, 1, 1).expand(2, *shape).clone()
        if name == "w13_weight_scale_inv":
            # Stored mn-major, as DeepGEMM keeps its scales: (E, 4, 3) in memory.
            local = local.transpose(1, 2).contiguous().transpose(1, 2)
        layer.register_parameter(name, torch.nn.Parameter(local, requires_grad=False))

    def all_gather_single(output, input, group):
        # The peer, EP rank 0, holds experts 0 and 1.
        output.copy_(torch.cat([input - 2, input]))

    monkeypatch.setattr(dwdp_manager, "all_gather_single", all_gather_single)
    with get_parallel().override(
        moe_ep_group=SimpleNamespace(world_size=2, device_group=None)
    ):
        replicate_routed_experts(torch.nn.ModuleDict({"experts": layer}))

    for name, shape in names_and_shapes:
        expected = torch.arange(4.0).view(4, 1, 1).expand(4, *shape)
        assert torch.equal(getattr(layer, name), expected), name
    assert layer.w13_weight_scale_inv.stride() == (12, 1, 3)
    assert _placement(layer) == (2, 1, 1, 0)
    assert not layer._dwdp_bound


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
    with get_context().override_server_args(model_path="dummy"):
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


def test_sm120_nvfp4_dispatch_uses_live_layer_placement(monkeypatch) -> None:
    from sglang.kernels.ops.gemm import sm120_nvfp4_moe

    calls = []
    monkeypatch.setattr(flashinfer_cutlass, "is_sm120", lambda: True)
    monkeypatch.setattr(flashinfer_cutlass, "_sm120_mtp_mma_supported", lambda: True)
    monkeypatch.setattr(
        flashinfer_cutlass, "_flashinfer_autotune_active", lambda: False
    )

    def native(*args, **kwargs):
        calls.append("native")
        return kwargs["out"]

    def fallback(**kwargs):
        calls.append((kwargs["tp_size"], kwargs["ep_size"]))
        return [kwargs["output"]]

    monkeypatch.setattr(sm120_nvfp4_moe, "nvfp4_moe_mtp_mma", native)
    monkeypatch.setattr(
        flashinfer_cutlass, "_flashinfer_cutlass_fused_moe", lambda: (fallback, None)
    )
    monkeypatch.setattr(flashinfer_cutlass, "_activation_type", lambda config: None)
    layer = SimpleNamespace(moe_tp_size=1, moe_tp_rank=0, moe_ep_size=1, moe_ep_rank=0)
    config = MoeRunnerConfig(activation="relu2", is_gated=False, layer=layer)
    quant = FlashInferCutlassMoeQuantInfo(
        "fp4",
        torch.empty((512, 2688, 512), device="meta", dtype=torch.uint8),
        torch.empty((512, 1024, 1344), device="meta", dtype=torch.uint8),
        quant_scales=[torch.ones(1)] * 6,
        output_dtype=torch.bfloat16,
    )
    x = torch.zeros((1, 1024), dtype=torch.bfloat16)
    dispatch = SimpleNamespace(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_ids=torch.zeros((1, 22), dtype=torch.int32),
            topk_weights=torch.ones((1, 22)),
        ),
    )
    for tp, ep in [(1, 1), (2, 1), (1, 2)]:
        layer.moe_tp_size, layer.moe_ep_size = tp, ep
        with get_context().override_server_args(model_path="dummy"):
            flashinfer_cutlass._run_flashinfer_cutlass(
                dispatch_output=dispatch,
                quant_info=quant,
                runner_config=config,
                output=torch.empty_like(x),
            )
    assert calls == ["native", (2, 1), (1, 2)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
