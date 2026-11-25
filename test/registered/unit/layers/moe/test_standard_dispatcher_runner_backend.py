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
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=8, suite="extra-test-cpu-intel")
register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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


@pytest.mark.parametrize(
    "opt_in, ep_size, backend, device, native_cuda",
    [
        (False, 2, MoeRunnerBackend.TRITON, "cuda", True),
        (True, 1, MoeRunnerBackend.TRITON, "cuda", True),
        (True, 2, MoeRunnerBackend.FLASHINFER_CUTLASS, "cuda", True),
        (True, 2, MoeRunnerBackend.TRITON, "cpu", True),
        (True, 2, MoeRunnerBackend.TRITON, "cuda", False),
    ],
)
def test_ineligible_layer_never_requests_cuda_mapper(
    ep2_rank0, monkeypatch, opt_in, ep_size, backend, device, native_cuda
):
    if device == "cuda" and not standard.is_cuda():
        pytest.skip("requires a CUDA tensor")
    monkeypatch.setattr(standard, "get_device", lambda: device)
    monkeypatch.setattr(standard, "is_cuda", lambda: native_cuda)
    config = _config(backend)
    if opt_in:
        config.enable_glm5_next_moe_kernels = True
    if ep_size == 1:
        config.num_local_experts = config.num_experts
    with (
        get_parallel().override(moe_ep_size=ep_size, moe_tp_size=2 // ep_size),
        patch.object(standard, "_map_glm5_next_cuda_expert_ids") as cuda_mapper,
    ):
        dispatcher = standard.StandardDispatcher(config)
        ids = torch.tensor([[-1, 3], [4, 0]], dtype=torch.int32, device=device)
        result = _dispatch_ids(dispatcher, ids)
        cuda_mapper.assert_not_called()
    if ep_size == 1 or backend.is_flashinfer_cutlass():
        assert result is ids
    else:
        assert result.tolist() == [[-1, 3], [-1, 0]]


@pytest.mark.skipif(not standard.is_cuda(), reason="requires native CUDA")
def test_opted_in_local_cuda_mapper_warms_once_and_replays(ep2_rank0, monkeypatch):
    monkeypatch.setattr(standard, "get_device", lambda: "cuda")
    monkeypatch.setattr(standard, "_compiled_local_expert_mapper", None)
    config = _config(MoeRunnerBackend.TRITON)
    config.enable_glm5_next_moe_kernels = True
    dispatcher = standard.StandardDispatcher(config)
    ids = torch.tensor([[-1, 3], [4, 0]], dtype=torch.int32, device="cuda")
    expected = [[-1, 3], [-1, 0]]

    hidden = torch.empty((2, 16), device="cuda")
    lazy_routes = BypassedTopKOutput(
        hidden, torch.empty((2, NUM_EXPERTS), device="cuda"), TopKConfig(top_k=2)
    )
    assert dispatcher.dispatch(hidden, lazy_routes).topk_output is lazy_routes
    assert standard._compiled_local_expert_mapper is None

    # A first-ever dispatch inside capture must not start the compiler.
    cold_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold_graph):
        cold = _dispatch_ids(dispatcher, ids)
    cold_graph.replay()
    assert cold.tolist() == expected
    assert standard._compiled_local_expert_mapper is None

    assert _dispatch_ids(dispatcher, ids).tolist() == expected
    compiled = standard._compiled_local_expert_mapper
    assert compiled is not None
    # A different layer and token count reuse the same wrapper.
    other_layer = standard.StandardDispatcher(config)
    assert _dispatch_ids(other_layer, ids.repeat(3, 1)).tolist() == expected * 3
    assert standard._compiled_local_expert_mapper is compiled

    graph = torch.cuda.CUDAGraph()
    with patch.object(standard, "_compiled_local_expert_mapper", wraps=compiled) as run:
        with torch.cuda.graph(graph):
            captured = _dispatch_ids(dispatcher, ids)
        run.assert_called_once()
    ids.copy_(torch.tensor([[2, 7], [-1, 1]], device="cuda", dtype=ids.dtype))
    graph.replay()
    assert captured.tolist() == [[2, -1], [-1, 1]]

    # Dynamo specializes singletons even with dynamic=True. Capture a row count
    # that neither layer warmed, then change its IDs to exercise actual replay.
    single_graph = torch.cuda.CUDAGraph()
    with torch.compiler.set_stance("fail_on_recompile"), torch.cuda.graph(single_graph):
        single = _dispatch_ids(dispatcher, ids[:1])
    ids[0].copy_(torch.tensor([7, 1], device="cuda", dtype=ids.dtype))
    single_graph.replay()
    assert single.tolist() == [[-1, 1]]


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
