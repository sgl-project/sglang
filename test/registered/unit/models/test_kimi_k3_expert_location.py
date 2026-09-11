from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.models.kimi_k3 as kimi_k3
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_kimi_k3_text_exposes_expert_location_config():
    config = KimiLinearConfig(
        num_hidden_layers=61,
        num_experts=384,
        num_expert_group=8,
    )

    expert_config = KimiK3LinearForCausalLM.get_model_config_for_expert_location(config)

    assert expert_config.num_layers == 61
    assert expert_config.num_logical_experts == 384
    assert expert_config.num_groups == 8


def test_kimi_k3_multimodal_exposes_text_expert_location_config():
    config = KimiK3Config(
        text_config={
            "num_hidden_layers": 61,
            "num_experts": 384,
            "num_expert_group": 8,
        }
    )

    expert_config = KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
        config
    )

    assert expert_config.num_layers == 61
    assert expert_config.num_logical_experts == 384
    assert expert_config.num_groups == 8


def test_kimi_k3_dense_config_has_no_expert_location_config():
    config = KimiLinearConfig(num_experts=None)

    assert KimiK3LinearForCausalLM.get_model_config_for_expert_location(config) is None


def test_kimi_k3_allocates_redundant_physical_expert_slots():
    config = SimpleNamespace(num_experts=896)

    for redundant_experts, expected_physical_experts in [
        (0, 896),
        (32, 928),
        (64, 960),
    ]:
        with patch.object(
            kimi_k3,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(ep_num_redundant_experts=redundant_experts)
            ),
        ):
            assert (
                kimi_k3._get_num_physical_routed_experts(config)
                == expected_physical_experts
            )


def test_kimi_k3_prefers_n_routed_experts_for_physical_slots():
    config = SimpleNamespace(num_experts=896, n_routed_experts=384)

    with patch.object(
        kimi_k3,
        "get_exec",
        return_value=SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=32)),
    ):
        assert kimi_k3._get_num_physical_routed_experts(config) == 416


def test_kimi_k3_topk_receives_expert_location_dispatch_info():
    expected_output = object()
    dispatch_info = object()
    owner = SimpleNamespace(
        layer_idx=17,
        topk=Mock(return_value=expected_output),
    )
    owner._expert_location_dispatch_info = lambda: (
        KimiK3MoE._expert_location_dispatch_info(owner)
    )
    hidden_states = torch.empty(2, 4)
    router_logits = torch.empty(2, 8)

    with patch.object(
        kimi_k3.ExpertLocationDispatchInfo,
        "init_new",
        return_value=dispatch_info,
    ) as init_new:
        actual = KimiK3MoE._select_experts(owner, hidden_states, router_logits)

    assert actual is expected_output
    init_new.assert_called_once_with(layer_id=17)
    owner.topk.assert_called_once_with(
        hidden_states,
        router_logits,
        expert_location_dispatch_info=dispatch_info,
    )
