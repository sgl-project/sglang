import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3LinearModel,
    KimiK3MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kimi_k3_cp_model_accepts_generic_input_embed_keyword():
    parameters = inspect.signature(KimiK3LinearModel.forward).parameters

    assert "input_embeds" in parameters
    assert "inputs_embeds" in parameters


def test_kimi_k3_wrappers_expose_cp_v2_protocol():
    assert hasattr(KimiK3LinearForCausalLM, "get_input_embeddings")

    wrapper_protocol = (
        "capture_aux_hidden_states",
        "get_context_parallel_model",
        "get_input_embeddings",
        "logits_processor",
        "pp_group",
    )
    for name in wrapper_protocol:
        assert hasattr(KimiK3ForConditionalGeneration, name), name


def _run_shared_expert_gather(buffer_rows: int):
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    group = SimpleNamespace(world_size=2)
    shared_experts = Mock(side_effect=lambda value: value + 1)
    moe = SimpleNamespace(
        _shared_experts_attn_tp_comm=True,
        shared_experts=shared_experts,
    )

    def all_gather(output, local_input):
        assert output.shape == (6, 4)
        output.copy_(torch.cat((local_input, local_input), dim=0))

    def reduce_scatter(output, gathered_input):
        output.copy_(gathered_input[:3] + gathered_input[3:])

    with (
        patch(
            "sglang.srt.models.kimi_k3.get_parallel",
            return_value=SimpleNamespace(attn_tp_group=group),
        ),
        patch(
            "sglang.srt.models.kimi_k3.get_local_dp_buffer",
            return_value=torch.empty(buffer_rows, 4),
        ),
        patch(
            "sglang.srt.models.kimi_k3.attn_tp_all_gather_into_tensor",
            side_effect=all_gather,
        ),
        patch(
            "sglang.srt.models.kimi_k3.attn_tp_reduce_scatter_tensor",
            side_effect=reduce_scatter,
        ),
    ):
        output = KimiK3MoE._forward_shared_experts(moe, hidden_states)

    assert shared_experts.call_args.args[0].shape == (6, 4)
    torch.testing.assert_close(output, 2 * (hidden_states + 1))


def test_kimi_k3_cp_shared_expert_slices_full_dp_buffer():
    _run_shared_expert_gather(buffer_rows=12)


def test_kimi_k3_cp_shared_expert_grows_undersized_buffer():
    _run_shared_expert_gather(buffer_rows=2)
