"""StandardDispatcher must give local expert ids to a runner that needs them.

Fp8MoEMethod runs its fp8 experts on triton when --moe-runner-backend is
flashinfer_cutlass, which NVFP4 checkpoints with fp8 MTP experts pin on sm120.
A dispatcher that only reads the global backend keeps global expert ids for
that layer, and under EP the triton kernel indexes experts the rank does not
hold (an illegal memory access on the draft extend of
nvidia/Qwen3.8-Flash-Next-NVFP4 at --tp 2 --ep 2).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import standard
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")

NUM_EXPERTS = 8
NUM_LOCAL_EXPERTS = 4


@pytest.fixture
def ep2_rank0(monkeypatch):
    monkeypatch.setattr(standard, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        standard,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=False, lora_paths=None),
    )
    with (
        get_context().override_server_args(model_path="dummy", device="cpu"),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE, runner_backend=MoeRunnerBackend.AUTO
        ),
        get_parallel().override(
            tp_size=2,
            tp_rank=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        ),
    ):
        yield get_flags().moe


@pytest.fixture
def cutlass_pin(ep2_rank0):
    ep2_rank0.runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS
    return ep2_rank0


def _config(runner_backend=None):
    return MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL_EXPERTS,
        num_fused_shared_experts=0,
        top_k=2,
        runner_backend=runner_backend,
    )


def _dispatch_ids(dispatcher, topk_ids):
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones(topk_ids.shape, device=topk_ids.device),
        topk_ids=topk_ids,
        router_logits=torch.zeros(
            topk_ids.shape[0], NUM_EXPERTS, device=topk_ids.device
        ),
    )
    hidden_states = torch.zeros(topk_ids.shape[0], 16, device=topk_ids.device)
    return dispatcher.dispatch(hidden_states, topk_output).topk_output.topk_ids


def test_direct_local_backend_is_independent_of_request(cutlass_pin):
    dispatcher = standard.StandardDispatcher(_config())
    assert not dispatcher.skip_local_expert_mapping


def test_triton_runner_gets_local_ids_under_cutlass_pin(cutlass_pin):
    dispatcher = standard.StandardDispatcher(_config(MoeRunnerBackend.TRITON))
    assert not dispatcher.skip_local_expert_mapping

    topk_ids = torch.tensor([[0, 5], [3, 7]], dtype=torch.int32, device="cpu")
    local_ids = _dispatch_ids(dispatcher, topk_ids)
    # Rank 0 holds global experts 0-3; experts on the other rank become the
    # -1 drop sentinel instead of indexing past the local weights.
    assert local_ids.tolist() == [[0, -1], [3, -1]]


def test_runner_that_keeps_global_ids_changes_nothing(cutlass_pin):
    dispatcher = standard.StandardDispatcher(
        _config(MoeRunnerBackend.FLASHINFER_CUTLASS)
    )
    assert dispatcher.skip_local_expert_mapping


def test_effective_global_backend_keeps_global_ids(ep2_rank0):
    # A resolved global-ID backend owns the layout even under an AUTO request.
    ep2_rank0.runner_backend = MoeRunnerBackend.AUTO
    dispatcher = standard.StandardDispatcher(
        _config(MoeRunnerBackend.FLASHINFER_TRTLLM)
    )
    assert dispatcher.skip_local_expert_mapping


def test_aiter_flag_follows_effective_backend(ep2_rank0, monkeypatch):
    # A Triton fallback uses local IDs even if AITER is available to other layers.
    ep2_rank0.runner_backend = MoeRunnerBackend.AUTO
    dispatcher = standard.StandardDispatcher(_config(MoeRunnerBackend.TRITON))
    assert not dispatcher.use_aiter_moe_runner


def test_lora_with_ep_is_rejected_on_a_remapped_layer(cutlass_pin, monkeypatch):
    monkeypatch.setattr(
        standard,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=True, lora_paths=None),
    )
    with pytest.raises(ValueError, match="LoRA"):
        standard.StandardDispatcher(_config(MoeRunnerBackend.TRITON))


def test_fp8_triton_fallback_reaches_the_dispatcher(cutlass_pin):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.is_fp4_expert = False
    method.use_mxfp8 = False
    with patch.object(method, "create_weights"):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=2,
            hidden_size=16,
            intermediate_size=128,
            layer_id=0,
            quant_method=method,
        )
    assert method.runner.runner_backend.is_triton()
    assert layer.moe_runner_config.runner_backend.is_triton()
    assert not layer.dispatcher.skip_local_expert_mapping


@pytest.mark.parametrize(
    "backend, requested, expected",
    [
        (MoeRunnerBackend.FLASHINFER_TRTLLM, None, RoutingMethodType.DeepSeekV3),
        (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, None, RoutingMethodType.DeepSeekV3),
        (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            RoutingMethodType.Default,
            RoutingMethodType.Default,
        ),
        (MoeRunnerBackend.TRITON, None, None),
        (MoeRunnerBackend.FLASHINFER_CUTLASS, None, None),
    ],
)
def test_fp8_publishes_only_the_selected_runners_policy(
    ep2_rank0, backend, requested, expected
):
    """The layer, router binding and consumer must share the actual FP8 policy."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    ep2_rank0.runner_backend = backend
    method = Fp8MoEMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    )
    with patch.object(method, "create_weights"):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=2,
            hidden_size=16,
            intermediate_size=128,
            layer_id=0,
            quant_method=method,
            routing_method_type=requested,
        )
    assert layer.routing_method_type == expected
    assert layer.moe_runner_config.routing_method_type == expected
    assert layer.runner_backend == method.get_moe_runner_backend(layer)
    assert layer.dispatcher.skip_local_expert_mapping == (
        layer.runner_backend.is_flashinfer_trtllm()
        or layer.runner_backend.is_flashinfer_trtllm_routed()
    )


def test_modelopt_undeclared_policy_keeps_triton_layout_and_local_ids(ep2_rank0):
    """Publishing another method's default must not enable ModelOpt native repacking."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization import modelopt_quant

    ep2_rank0.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM
    with patch.object(modelopt_quant, "cutlass_fp8_supported", return_value=False):
        method = modelopt_quant.ModelOptFp8MoEMethod(
            modelopt_quant.ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
        )
    with patch.object(method, "create_weights"):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=2,
            hidden_size=16,
            intermediate_size=128,
            layer_id=0,
            quant_method=method,
        )
    assert layer.routing_method_type is None
    assert layer.moe_runner_config.routing_method_type is None
    assert layer.runner_backend.is_triton()
    assert not layer.dispatcher.skip_local_expert_mapping
    ids = torch.tensor([[0, 5], [3, 7]], dtype=torch.int32)
    assert _dispatch_ids(layer.dispatcher, ids).tolist() == [[0, -1], [3, -1]]


@pytest.mark.parametrize(
    "backend, requested, expected",
    [
        (MoeRunnerBackend.FLASHINFER_TRTLLM, None, RoutingMethodType.DeepSeekV3),
        (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            RoutingMethodType.Default,
            RoutingMethodType.Default,
        ),
        (MoeRunnerBackend.TRITON, None, None),
    ],
)
def test_hybrid_method_preserves_its_gpu_policy(
    ep2_rank0, backend, requested, expected
):
    """The base query must not shadow the wrapped GPU method's routing contract."""
    from sglang.srt.layers.moe import kt_ep_wrapper
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    ep2_rank0.runner_backend = backend
    gpu = Fp8MoEMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    )
    with patch.object(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True):
        method = kt_ep_wrapper.KTEPWrapperMethod(
            gpu, SimpleNamespace(num_gpu_experts=4)
        )
    with patch.object(method, "create_weights"):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=2,
            hidden_size=16,
            intermediate_size=128,
            layer_id=0,
            quant_method=method,
            routing_method_type=requested,
        )
    assert layer.routing_method_type == expected
    assert layer.moe_runner_config.routing_method_type == expected
    assert layer.runner_backend == gpu.runner.runner_backend


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
