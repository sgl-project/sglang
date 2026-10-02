"""Public NCCL EP contracts identified by review; native EP is replaced."""

import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.arg_groups.overrides import resolved_view

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from registered.unit.layers.moe.test_nccl_ep_graph_config import (  # noqa: F401
    ep_bindings,
    model_path,
    server_args,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The CUDA configuration path is required"
)


@pytest.fixture
def supported_gpu(monkeypatch):
    torch.cuda.init()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))


@pytest.mark.parametrize("graph", [False, True])
def test_default_ep_spans_tp(model_path, supported_gpu, graph):
    args = server_args(model_path, tp_size=2, enable_nccl_ep_cuda_graph=graph)
    assert resolved_view(args).ep_size == 2
    assert resolved_view(args).ep_size == 2


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("chunk", [None, 8192, 128])
def test_prefill_fits_native_capacity(model_path, supported_gpu, dp, chunk):
    kwargs = {} if chunk is None else {"chunked_prefill_size": chunk}
    args = server_args(
        model_path,
        tp_size=2,
        dp_size=dp,
        enable_dp_attention=dp == 2,
        nccl_ep_num_max_dispatch_tokens_per_rank=64,
        **kwargs,
    )
    assert 0 < resolved_view(args).chunked_prefill_size <= 64
    assert (
        resolved_view(args).chunked_prefill_size
        == resolved_view(args).chunked_prefill_size
    )


def test_disabled_chunking_is_rejected(model_path, supported_gpu):
    with pytest.raises(ValueError, match="NCCL EP.*chunked prefill"):
        server_args(model_path, chunked_prefill_size=-1)


@pytest.mark.parametrize("mixed_tokens", [0, 7])
def test_long_prefill_scheduler_respects_ll_capacity(
    model_path, supported_gpu, mixed_tokens
):
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.prefill_budget import PrefillBudget
    from sglang.srt.runtime_context import get_context
    from sglang.srt.sampling.sampling_params import SamplingParams

    allocator = SimpleNamespace(page_size=1, available_size=lambda: 10000)
    allocator.create_prefill_budget = lambda tree, **kw: PrefillBudget(
        allocator, tree, **kw
    )
    args = server_args(model_path, nccl_ep_num_max_dispatch_tokens_per_rank=64)
    context = get_context()
    with context.override_server_args():
        context.set_server_args(args)
        req = Req("long", "", list(range(257)), SamplingParams(max_new_tokens=8))
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.prefix_indices = []
        consumed = 0
        forwards = 0
        while consumed < 257:
            adder = PrefillAdder(
                page_size=1,
                tree_cache=SimpleNamespace(
                    supports_mamba=lambda: False, evictable_size=lambda: 0
                ),
                token_to_kv_pool_allocator=allocator,
                running_batch=None,
                new_token_ratio=1,
                rem_input_tokens=resolved_view(args).max_prefill_tokens,
                rem_chunk_tokens=resolved_view(args).chunked_prefill_size,
                num_mixed_decode_tokens=mixed_tokens,
            )
            remaining = adder.add_chunked_req(req)
            rows = req.extend_range.length
            assert 0 < rows + mixed_tokens <= 64
            consumed += rows
            req.prefix_indices = list(range(consumed))
            assert (remaining is None) == (consumed == 257)
            forwards += 1
        assert forwards > 1


@pytest.mark.parametrize("budget,expected", [(65, 64), (64, 64), (32, 32)])
def test_prefill_budget_respects_kv_pages(model_path, supported_gpu, budget, expected):
    args = server_args(
        model_path, page_size=16, nccl_ep_num_max_dispatch_tokens_per_rank=budget
    )
    assert resolved_view(args).chunked_prefill_size == expected


def test_budget_smaller_than_a_page_is_rejected(model_path, supported_gpu):
    with pytest.raises(ValueError, match="NCCL EP.*KV page"):
        server_args(
            model_path, page_size=16, nccl_ep_num_max_dispatch_tokens_per_rank=8
        )


def test_pp_dynamic_chunk_growth_is_rejected(model_path, supported_gpu):
    with pytest.raises(ValueError, match="NCCL EP.*dynamic chunking"):
        server_args(model_path, pp_size=2, enable_dynamic_chunking=True)


@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
def test_tbo_is_rejected(model_path, supported_gpu, runner):
    with pytest.raises(ValueError, match="NCCL EP.*overlap"):
        server_args(
            model_path, moe_runner_backend=runner, enable_two_batch_overlap=True
        )


def test_triton_unavailable_fallback_uses_standard_dispatch(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: True if name == "deep_ep" else find_spec(name),
    )
    args = server_args(model_path, tp_size=2)
    assert resolved_view(args).moe_a2a_backend == "none"
    assert resolved_view(args).moe_runner_backend == "triton"


def test_deepep_fallback_also_resolves_tp_spanning_ep(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: True if name == "deep_ep" else find_spec(name),
    )
    args = server_args(model_path, tp_size=2, moe_runner_backend="deep_gemm")
    assert resolved_view(args).moe_a2a_backend == "deepep"
    assert resolved_view(args).ep_size == resolved_view(args).ep_size == 2


def test_blackwell_scale_mode_is_rejected_by_public_gate(model_path, monkeypatch):
    from sglang.srt.layers import deep_gemm_wrapper

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    monkeypatch.setattr(deep_gemm_wrapper, "DEEPGEMM_BLACKWELL", True)
    with pytest.raises(ValueError, match="NCCL EP.*UE8M0"):
        server_args(model_path, enable_nccl_ep_cuda_graph=True)


@pytest.mark.parametrize("capability", [(9, 0), (12, 0)])
def test_float_scale_gpu_remains_available(model_path, monkeypatch, capability):
    from sglang.srt.layers import deep_gemm_wrapper

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(deep_gemm_wrapper, "DEEPGEMM_BLACKWELL", False)
    assert (
        server_args(model_path, enable_nccl_ep_cuda_graph=True).moe_a2a_backend
        == "nccl_ep"
    )


def test_supported_quantization_and_other_backends_are_preserved():
    from sglang.srt.layers.moe.ep_moe.layer import (
        DeepEPMoE,
        FusedMoE,
        get_moe_impl_class,
    )
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
    from sglang.srt.runtime_context import get_flags

    with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP):
        assert get_moe_impl_class(Fp8Config()) is DeepEPMoE
        assert get_moe_impl_class(W4AFp8Config()) is DeepEPMoE
    with get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE):
        assert get_moe_impl_class(None) is FusedMoE
    with get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP):
        assert get_moe_impl_class(None) is DeepEPMoE


def test_unquantized_moe_is_rejected_before_allocation():
    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.runtime_context import get_flags

    with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP):
        with pytest.raises(ValueError, match="NCCL EP.*quant"):
            get_moe_impl_class(None)


def test_fp16_parameters_are_rejected_before_group_creation():
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher

    with dispatcher_environment() as env:
        config = MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            hidden_size=2048,
            top_k=2,
            params_dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="NCCL EP.*bfloat16"):
            NcclEpDispatcher(config, env.coordinator)
        assert env.events.count("group_create") == 0


def test_fp16_eager_input_is_rejected_before_handle_creation():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment

    with dispatcher_environment() as env:
        dispatcher = env.dispatcher(layer_id=0)
        x = torch.ones(1, 2048, dtype=torch.float16, device="cuda")
        ids = torch.tensor([[0, 1]], device="cuda")
        weights = torch.tensor([[0.5, 0.5]], device="cuda")
        with pytest.raises(ValueError, match="NCCL EP.*bfloat16"):
            forward_layer(dispatcher, x, ids, weights, rank=0)
        assert env.events.count("handle_create") == 0


@pytest.mark.parametrize("graph", [False, True])
def test_eplb_is_rejected_before_native_setup(model_path, supported_gpu, graph):
    with pytest.raises(ValueError, match="(?i)eplb"):
        server_args(
            model_path, tp_size=2, enable_eplb=True, enable_nccl_ep_cuda_graph=graph
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
