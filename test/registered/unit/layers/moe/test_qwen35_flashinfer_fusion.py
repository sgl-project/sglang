from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    FlashInferMNNVLCuteDSLARFusion,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.moe.qwen35_flashinfer_fusion import (
    Qwen35MoeFinalizeHandoff,
    is_supported_forward_mode,
    resolve_max_m,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _TestPreset:
    load_shared_expert_before_pdl: bool = False


@dataclass(frozen=True)
class _TestTarget:
    preset: object


@dataclass(frozen=True)
class _TestRoutes:
    targets: tuple[_TestTarget, ...]


@dataclass(frozen=True)
class _TestProfile:
    finalize_routes: _TestRoutes


@dataclass(frozen=True)
class _TestConfig:
    profiles: tuple[_TestProfile, ...]


@pytest.mark.parametrize(
    ("forward_mode", "expected"),
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.EXTEND, True),
        (ForwardMode.IDLE, False),
        (ForwardMode.TARGET_VERIFY, True),
        (ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_supported_forward_modes(forward_mode, expected):
    assert is_supported_forward_mode(forward_mode) is expected


@patch(
    "sglang.srt.layers.moe.qwen35_flashinfer_fusion.cutedsl_moe_max_num_tokens",
    return_value=8192,
)
def test_framework_capacity_is_maximum_of_all_sources(_cutedsl_moe_max_num_tokens):
    graph = SimpleNamespace(
        decode=SimpleNamespace(max_bs=512, bs=[1, 64, 256]),
        prefill=SimpleNamespace(max_bs=4096, bs=[1024, 2048, 4096]),
    )
    server_args = SimpleNamespace(cuda_graph_config=graph)
    runner = SimpleNamespace(server_args=server_args, max_running_requests=2048)

    assert resolve_max_m(runner) == 8192


def test_deferred_handoff_reuses_producer_storage():
    m, top_k, hidden_size = 3, 10, 16
    gemm2_out = torch.empty(m * top_k + 4, hidden_size, dtype=torch.bfloat16)
    expert_weights = torch.empty(m, top_k, dtype=torch.bfloat16)
    permuted_indices = torch.empty(m, top_k, dtype=torch.int32)
    gated_shared_output = torch.empty(m, hidden_size, dtype=torch.bfloat16)
    deferred = SimpleNamespace(
        gemm2_out=gemm2_out,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=permuted_indices,
        top_k=top_k,
    )

    handoff = Qwen35MoeFinalizeHandoff.from_flashinfer(
        deferred,
        gated_shared_output=gated_shared_output,
        m=m,
    )

    assert handoff.routed_output.data_ptr() == gemm2_out.data_ptr()
    assert handoff.expert_weights.data_ptr() == expert_weights.data_ptr()
    assert handoff.permuted_indices.data_ptr() == permuted_indices.data_ptr()
    assert handoff.gated_shared_output is gated_shared_output


def test_qwen_workspace_config_enables_only_supported_finalize_presets():
    untouched_preset = object()
    default_config = _TestConfig(
        profiles=(
            _TestProfile(
                finalize_routes=_TestRoutes(
                    targets=(
                        _TestTarget(_TestPreset()),
                        _TestTarget(untouched_preset),
                    )
                )
            ),
        )
    )

    qwen_config = _with_early_finalize_shared_load(default_config)

    assert qwen_config is not default_config
    assert (
        default_config.profiles[0]
        .finalize_routes.targets[0]
        .preset.load_shared_expert_before_pdl
        is False
    )
    assert (
        qwen_config.profiles[0]
        .finalize_routes.targets[0]
        .preset.load_shared_expert_before_pdl
        is True
    )
    assert qwen_config.profiles[0].finalize_routes.targets[1].preset is untouched_preset


def test_wrapper_calls_only_the_stable_unified_api():
    calls = []
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.hidden_size = 8
    wrapper.top_k = 2
    wrapper.max_m = 4
    wrapper.rms_epsilon = 1e-5
    wrapper.weight_bias = 0.0
    wrapper.device = torch.device("cpu")
    wrapper.workspace = object()
    wrapper.supports = lambda m: True
    wrapper._patterns = SimpleNamespace(
        kARResidualRMSNorm=1,
        kMoEFinalizeARResidualRMSNorm=7,
    )
    wrapper._allreduce_fusion = lambda **kwargs: calls.append(kwargs)

    routed_output = torch.empty(8, 8, dtype=torch.bfloat16)
    expert_weights = torch.empty(4, 2, dtype=torch.bfloat16)
    permuted_indices = torch.empty(4, 2, dtype=torch.int32)
    gated_shared_output = torch.empty(4, 8, dtype=torch.bfloat16)
    residual = torch.empty(4, 8, dtype=torch.bfloat16)
    gamma = torch.empty(8, dtype=torch.bfloat16)
    norm_output = torch.empty_like(residual)
    residual_output = torch.empty_like(residual)

    wrapper.moe_finalize_all_reduce_rms_norm(
        routed_output=routed_output,
        expert_weights=expert_weights,
        permuted_indices=permuted_indices,
        gated_shared_output=gated_shared_output,
        residual=residual,
        gamma=gamma,
        norm_output=norm_output,
        residual_output=residual_output,
    )

    assert calls[0]["launch_with_pdl"] is True
    assert "routed_scaling_factor" not in calls[0]

    wrapper.all_reduce_residual_rms_norm(
        local_contribution=residual,
        residual=residual,
        gamma=gamma,
        norm_output=norm_output,
        residual_output=residual_output,
    )

    assert calls[1]["pattern"] == 1
    assert calls[1]["launch_with_pdl"] is True
    assert "routed_scaling_factor" not in calls[1]
    assert "expanded_idx_to_permuted_idx" not in calls[1]


def test_text_entry_wrapper_delegates_pre_capture_prepare():
    calls = []
    runner = object()
    wrapper = SimpleNamespace(
        model=SimpleNamespace(
            prepare_before_cuda_graph_capture=lambda value: calls.append(value)
        )
    )

    Qwen3_5ForCausalLM.prepare_before_cuda_graph_capture(wrapper, runner)

    assert calls == [runner]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
