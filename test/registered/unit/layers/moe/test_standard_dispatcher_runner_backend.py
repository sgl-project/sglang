"""StandardDispatcher must give local expert ids to a runner that needs them.

Fp8MoEMethod runs its fp8 experts on triton when --moe-runner-backend is
flashinfer_cutlass, which NVFP4 checkpoints with fp8 MTP experts pin on sm120.
A dispatcher that only reads the global backend keeps global expert ids for
that layer, and under EP the triton kernel indexes experts the rank does not
hold (an illegal memory access on the draft extend of
nvidia/Qwen3.8-Flash-Next-NVFP4 at --tp 2 --ep 2).
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import standard
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils.common import get_device
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")

NUM_EXPERTS = 8
NUM_LOCAL_EXPERTS = 4


@pytest.fixture
def ep2_rank0(monkeypatch):
    moe = get_flags().moe
    saved = (moe.runner_backend, moe.a2a_backend)
    moe.a2a_backend = MoeA2ABackend.NONE
    monkeypatch.setattr(
        standard,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=0),
    )
    monkeypatch.setattr(
        standard,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=False, lora_paths=None),
        raising=False,
    )
    yield moe
    moe.runner_backend, moe.a2a_backend = saved


@pytest.fixture
def cutlass_pin(ep2_rank0):
    ep2_rank0.runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS
    return ep2_rank0


def _config():
    return MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL_EXPERTS,
        num_fused_shared_experts=0,
        top_k=2,
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


def test_global_backend_still_decides_without_a_runner(cutlass_pin):
    dispatcher = standard.StandardDispatcher(_config())
    assert dispatcher.skip_local_expert_mapping


def test_triton_runner_gets_local_ids_under_cutlass_pin(cutlass_pin):
    dispatcher = standard.StandardDispatcher(
        _config(), runner_backend=MoeRunnerBackend.TRITON
    )
    assert not dispatcher.skip_local_expert_mapping

    topk_ids = torch.tensor([[0, 5], [3, 7]], dtype=torch.int32, device=get_device())
    local_ids = _dispatch_ids(dispatcher, topk_ids)
    # Rank 0 holds global experts 0-3; experts on the other rank become the
    # -1 drop sentinel instead of indexing past the local weights.
    assert local_ids.tolist() == [[0, -1], [3, -1]]


def test_runner_that_keeps_global_ids_changes_nothing(cutlass_pin):
    dispatcher = standard.StandardDispatcher(
        _config(), runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS
    )
    assert dispatcher.skip_local_expert_mapping


def test_runner_never_turns_mapping_off(ep2_rank0):
    # Under auto a built runner is never AUTO, e.g. ModelOpt NVFP4 resolving
    # to flashinfer_trtllm on sm100; the global layout must stay in charge.
    ep2_rank0.runner_backend = MoeRunnerBackend.AUTO
    dispatcher = standard.StandardDispatcher(
        _config(), runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM
    )
    assert not dispatcher.skip_local_expert_mapping


def test_aiter_flag_follows_global_backend(ep2_rank0, monkeypatch):
    # Unquantized MoE on ROCm builds a triton runner next to its AITER one;
    # the AITER expert mask must still be built from the global backend.
    ep2_rank0.runner_backend = MoeRunnerBackend.AUTO
    monkeypatch.setattr(standard, "_use_aiter", True)
    dispatcher = standard.StandardDispatcher(
        _config(), runner_backend=MoeRunnerBackend.TRITON
    )
    assert dispatcher.use_aiter_moe_runner


def test_lora_with_ep_is_rejected_on_a_remapped_layer(cutlass_pin, monkeypatch):
    monkeypatch.setattr(
        standard,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=True, lora_paths=None),
    )
    with pytest.raises(ValueError, match="LoRA"):
        standard.StandardDispatcher(_config(), runner_backend=MoeRunnerBackend.TRITON)


def test_fp8_triton_fallback_reaches_the_dispatcher(cutlass_pin):
    from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.is_fp4_expert = False
    method.create_moe_runner(SimpleNamespace(), _config())
    assert method.runner.runner_backend.is_triton()

    dispatcher = create_moe_dispatcher(_config(), method)
    assert not dispatcher.skip_local_expert_mapping


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
