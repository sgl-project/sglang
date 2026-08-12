from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    FlashInferMNNVLCuteDSLARFusion,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.flashinfer_provider import _make_provider
from sglang.srt.layers.moe.qwen35_flashinfer_fusion import (
    Qwen35MoeFinalizeHandoff,
    is_supported_forward_mode,
    resolve_max_m,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-c-test-cpu")


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


_TEST_DEFAULT_CONFIG = _TestConfig(
    profiles=(
        _TestProfile(
            finalize_routes=_TestRoutes(targets=(_TestTarget(_TestPreset()),))
        ),
    )
)


@pytest.mark.parametrize(
    ("forward_mode", "expected"),
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.EXTEND, True),
        (ForwardMode.IDLE, False),
        (ForwardMode.TARGET_VERIFY, False),
    ],
)
def test_supported_forward_modes(forward_mode, expected):
    assert is_supported_forward_mode(forward_mode) is expected


def test_framework_capacity_is_maximum_of_all_sources():
    graph = SimpleNamespace(
        decode=SimpleNamespace(max_bs=512, bs=[1, 64, 256]),
        prefill=SimpleNamespace(max_bs=4096, bs=[1024, 2048, 4096]),
    )
    server_args = SimpleNamespace(
        cuda_graph_config=graph,
        cutedsl_moe_max_num_tokens=lambda: 8192,
    )
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


class _CompleteWorkspace:
    def __init__(
        self,
        tp_size,
        tp_rank,
        max_token_num,
        hidden_dim,
        dtype,
        *,
        group,
        top_k,
        rms_eps,
        routed_scaling_factor,
        weight_bias,
        include_shared_expert,
        add_residual,
        write_residual_output,
        config=_TEST_DEFAULT_CONFIG,
    ):
        pass

    def is_buffer_size_sufficient(
        self, tp_size, num_tokens, hidden_dim, dtype, use_oneshot=None
    ):
        pass

    def destroy(self):
        pass


def _complete_allreduce(
    input,
    workspace,
    pattern,
    launch_with_pdl,
    residual_in,
    residual_out,
    norm_out,
    rms_gamma,
    rms_eps,
    weight_bias,
    expanded_idx_to_permuted_idx,
    expert_scale_factor,
    shared_expert_output,
):
    pass


def _complete_comm(allreduce_fusion=_complete_allreduce):
    return SimpleNamespace(
        AllReduceFusionPattern=SimpleNamespace(
            kARResidualRMSNorm=1,
            kMoEFinalizeARResidualRMSNorm=7,
        ),
        allreduce_fusion=allreduce_fusion,
    )


def test_provider_requires_the_stable_backend_specific_abi():
    provider = _make_provider(
        _complete_comm(),
        _CompleteWorkspace,
        default_config=_TEST_DEFAULT_CONFIG,
    )
    assert provider is not None
    assert provider.workspace_type is _CompleteWorkspace
    assert provider.default_config is _TEST_DEFAULT_CONFIG

    class WorkspaceWithoutRoutedScale:
        def __init__(
            self,
            tp_size,
            tp_rank,
            max_token_num,
            hidden_dim,
            dtype,
            *,
            group,
            top_k,
            rms_eps,
            weight_bias,
            include_shared_expert,
            add_residual,
            write_residual_output,
            config=_TEST_DEFAULT_CONFIG,
        ):
            pass

        def is_buffer_size_sufficient(
            self, tp_size, num_tokens, hidden_dim, dtype, use_oneshot=None
        ):
            pass

        def destroy(self):
            pass

    assert (
        _make_provider(
            _complete_comm(),
            WorkspaceWithoutRoutedScale,
            default_config=_TEST_DEFAULT_CONFIG,
        )
        is None
    )


def test_provider_accepts_forward_compatible_kwargs_abi():
    class Workspace:
        def __init__(self, **kwargs):
            pass

        def is_buffer_size_sufficient(self, **kwargs):
            pass

        def destroy(self):
            pass

    def allreduce_fusion(**kwargs):
        pass

    assert (
        _make_provider(
            _complete_comm(allreduce_fusion),
            Workspace,
            default_config=_TEST_DEFAULT_CONFIG,
        )
        is not None
    )


def test_provider_rejects_incomplete_unified_api():
    assert (
        _make_provider(
            SimpleNamespace(
                AllReduceFusionPattern=_complete_comm().AllReduceFusionPattern
            ),
            _CompleteWorkspace,
            default_config=_TEST_DEFAULT_CONFIG,
        )
        is None
    )

    def allreduce_without_pdl(
        input,
        workspace,
        pattern,
        residual_in,
        residual_out,
        norm_out,
        rms_gamma,
        rms_eps,
        weight_bias,
        expanded_idx_to_permuted_idx,
        expert_scale_factor,
        shared_expert_output,
    ):
        pass

    assert (
        _make_provider(
            _complete_comm(allreduce_without_pdl),
            _CompleteWorkspace,
            default_config=_TEST_DEFAULT_CONFIG,
        )
        is None
    )
    assert (
        _make_provider(
            SimpleNamespace(
                AllReduceFusionPattern=SimpleNamespace(kARResidualRMSNorm=1),
                allreduce_fusion=_complete_allreduce,
            ),
            _CompleteWorkspace,
            default_config=_TEST_DEFAULT_CONFIG,
        )
        is None
    )


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
    wrapper.provider = SimpleNamespace(
        patterns=SimpleNamespace(
            kARResidualRMSNorm=1,
            kMoEFinalizeARResidualRMSNorm=7,
        ),
        allreduce_fusion=lambda **kwargs: calls.append(kwargs),
    )

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
