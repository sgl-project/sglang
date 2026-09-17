from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    SUPPORTED_TP_SIZES,
    FlashInferMNNVLCuteDSLARFusion,
    _config_for_shape,
    _ht_retarget,
    _retargeted_config,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    MoeFinalizeHandoff,
    is_supported_forward_mode,
    resolve_max_m,
)
from sglang.srt.model_executor.cuda_graph_config import (
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
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
    "sglang.srt.layers.moe.cutedsl_ar_fusion.cutedsl_moe_max_num_tokens",
    return_value=8192,
)
def test_framework_capacity_is_maximum_of_all_sources(_cutedsl_moe_max_num_tokens):
    # The graph bounds are a bag leaf, so the test states them by publishing.
    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(max_bs=512, bs=[1, 64, 256]),
                prefill=PhaseConfig(max_bs=4096, bs=[1024, 2048, 4096]),
            ),
        ),
        role="test",
    )
    assert (
        resolve_max_m(server_args=SimpleNamespace(), max_running_requests=2048) == 8192
    )


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

    handoff = MoeFinalizeHandoff.from_flashinfer(
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
    wrapper.moe_finalize_all_reduce_rms_norm(
        routed_output=routed_output,
        expert_weights=expert_weights,
        permuted_indices=permuted_indices,
        gated_shared_output=gated_shared_output,
        residual=residual,
        gamma=gamma,
    )

    assert calls[0]["launch_with_pdl"] is True
    assert "routed_scaling_factor" not in calls[0]

    wrapper.all_reduce_residual_rms_norm(
        local_contribution=residual,
        residual=residual,
        gamma=gamma,
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


def _eligible_communicator(*, successor: bool):
    """Stub whose only varying input is whether a successor exists."""
    comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
    comm.successor_absorbs_all_reduce = successor
    comm.input_layernorm = SimpleNamespace()
    return comm


def test_last_layer_prepare_attn_consumes_the_pending_all_reduce():
    """A tagged tensor must still be fused, or the reduction is silently dropped."""
    last = _eligible_communicator(successor=False)
    last.fusion_service = SimpleNamespace(
        all_reduce_residual_rms_norm=(
            lambda *, local_contribution, residual, gamma: (
                local_contribution + 1,
                residual + 1,
            )
        )
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    hidden_states = torch.zeros(8, 8)
    hidden_states._sglang_needs_allreduce_fusion = True
    residual = torch.zeros(8, 8)

    finished = []
    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "_finish_prepare_attn",
            lambda self, h, r, fb: finished.append((h, r)) or (h, r),
        ),
        patch.object(
            LayerCommunicator,
            "prepare_attn",
            lambda *a, **k: pytest.fail(
                "the last layer fell through to the unfused path, dropping the "
                "reduction its predecessor skipped"
            ),
        ),
    ):
        out_hidden, out_residual = last.prepare_attn(
            hidden_states, residual, forward_batch
        )

    assert len(finished) == 1
    assert torch.equal(out_hidden, torch.ones(8, 8))
    assert torch.equal(out_residual, torch.ones(8, 8))


def test_last_layer_still_declines_to_skip_its_own_all_reduce():
    """The other half of the split: consuming is owed to it, deferring is not."""
    last = _eligible_communicator(successor=False)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
    ):
        assert last._can_consume_post_moe_all_reduce(forward_batch, 8) is True
        assert last._can_absorb_post_moe_all_reduce(forward_batch, 8) is False


def test_hybrid_ep_tp_is_refused_like_the_base_communicator():
    """Skipping the post-experts reduction drops both legs; one fused
    collective cannot restore them."""
    comm = _eligible_communicator(successor=True)
    comm.fusion_service = SimpleNamespace(supports=lambda m: True)
    comm._context = SimpleNamespace(tp_size=4, attn_dp_size=1)
    comm.layer_scatter_modes = SimpleNamespace(mlp_mode=ScatterMode.FULL)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    def parallel(*, ep, moe_tp):
        return SimpleNamespace(
            moe_ep_size=ep,
            moe_tp_size=moe_tp,
            attn_cp_size=1,
            tp_size=4,
            attn_tp_size=4,
        )

    with (
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.is_dp_attention_enabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_attn_tp_context",
            return_value=SimpleNamespace(input_scattered=False),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
    ):
        with patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_parallel",
            return_value=parallel(ep=2, moe_tp=2),
        ):
            assert comm._common_eligible(forward_batch, 8) is False
        with patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_parallel",
            return_value=parallel(ep=1, moe_tp=4),
        ):
            assert comm._common_eligible(forward_batch, 8) is True


def test_a_replicated_shared_expert_producer_keeps_its_own_all_reduce():
    """A TP1 shared expert is added after the layer's own reduction, so handing
    that reduction onward would scale it by tp_size."""
    producer = _eligible_communicator(successor=True)
    producer.owes_local_reduction = True
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, input_ids=torch.zeros(8)
    )

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "should_defer_moe_finalize",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
        patch.object(
            LayerCommunicator,
            "should_fuse_mlp_allreduce_with_next_layer",
            return_value=False,
        ),
    ):
        # The publisher of fuse_mlp_allreduce, which is what skips the reduction.
        assert (
            producer.should_fuse_mlp_allreduce_with_next_layer(forward_batch) is False
        )
        assert producer._can_absorb_post_moe_all_reduce(forward_batch, 8) is False
        # Consuming what a predecessor skipped stays independently eligible.
        assert producer._can_consume_post_moe_all_reduce(forward_batch, 8) is True


def test_install_records_which_producers_owe_a_local_reduction():
    """Without the predicate carried onto the layer, a replicated shared expert
    silently keeps the unsafe fusion."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import install_cutedsl_fusion

    def _communicator():
        comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
        # install_cutedsl_fusion() reads both norms to check the workspace epsilon.
        comm.input_layernorm = RMSNorm(8, eps=1e-6)
        comm.post_attention_layernorm = RMSNorm(8, eps=1e-6)
        return comm

    layers = [
        SimpleNamespace(layer_communicator=_communicator(), replicated=replicated)
        for replicated in (True, False)
    ]
    install_cutedsl_fusion(
        layers,
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: not layer.replicated,
        requires_local_reduction=lambda layer: layer.replicated,
        label="test",
    )

    assert layers[0].layer_communicator.owes_local_reduction is True
    assert layers[1].layer_communicator.owes_local_reduction is False


def _call_dual_stream_op(fusion, hidden_states, *, fuse_mlp_allreduce=True):
    """Redispatching to the CUDA key runs the real schema while the stubbed MoE
    keeps tensors on CPU."""
    from sglang.srt.models.deepseek_v2 import (  # noqa: F401  (registers the op)
        dsv2_flashinfer_moe_dual_stream_graph,
    )

    op = torch.ops.sglang.dsv2_flashinfer_moe_dual_stream_graph.default
    cuda_key = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    with patch(
        "sglang.srt.models.deepseek_v2.get_tc_piecewise_forward_context",
        return_value=SimpleNamespace(moe_fusions={0: fusion}),
    ):
        return op.redispatch(cuda_key, hidden_states, 0, fuse_mlp_allreduce, False)


class _DeferRecordingMoE:
    """Returns a handoff whenever the deferral reaches it, as the real MoE does."""

    def __init__(self):
        self.seen_defer = None

    def forward_normal_dual_stream(self, hidden_states):
        from sglang.srt.runtime_context import get_forward

        self.seen_defer = get_forward().defer_moe_finalize
        if self.seen_defer:
            return object()
        return hidden_states + 1


def test_dual_stream_op_pins_the_deferral_off_under_a_deferring_caller():
    """The op's Tensor schema cannot carry a handoff; the dispatcher would raise
    "Unable to cast ... to Tensor"."""
    from sglang.srt.runtime_context import get_forward

    reset_context()
    fusion = _DeferRecordingMoE()
    hidden_states = torch.zeros(4, 8)

    with get_forward().scoped(defer_moe_finalize=True):
        out = _call_dual_stream_op(fusion, hidden_states)
        # The pin is scoped to the op; the caller's own flag survives it.
        assert get_forward().defer_moe_finalize is True

    assert fusion.seen_defer is False
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, hidden_states + 1)


def test_dual_stream_op_still_republishes_its_operand_flags():
    """Pinning the deferral must not disturb the flags republished from the
    scalar operands."""
    seen = {}

    class _FlagReader:
        def forward_normal_dual_stream(self, hidden_states):
            from sglang.srt.runtime_context import get_forward

            flags = get_forward()
            seen["fuse"] = flags.fuse_mlp_allreduce
            seen["scatter"] = flags.mlp_reduce_scatter
            return hidden_states

    reset_context()
    _call_dual_stream_op(_FlagReader(), torch.zeros(4, 8))

    assert seen["fuse"] is True
    assert seen["scatter"] is False


# ---------------------------------------------------------------------------
# Kernel-shape re-targeting
# ---------------------------------------------------------------------------


def _ht_target(routes):
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    targets = [t for t in routes.targets if t.protocol is ProtocolKind.HT]
    return targets[0].preset if targets else None


# (label, hidden_size, top_k, tp_size, HT routable), shapes read from the
# checkpoint configs. A False row is a shape whose vectors per reduction shard
# are not a warp multiple at that width, which the kernel rejects.
_SHAPES = [
    ("Qwen3.8", 8192, 10, 8, True),
    ("Qwen3.8", 8192, 10, 16, True),
    ("DeepSeek-V3", 7168, 8, 4, True),
    ("DeepSeek-V3", 7168, 8, 8, False),
    ("DeepSeek-V3", 7168, 8, 16, False),
    ("GLM-5.3", 6144, 8, 4, True),
    ("GLM-5.3", 6144, 8, 8, True),
    ("GLM-5.3", 6144, 8, 16, False),
]


@pytest.mark.parametrize(
    ("model", "hidden_size", "top_k", "tp_size", "ht_routable"), _SHAPES
)
def test_every_served_shape_builds_a_capacity_covering_profile(
    model, hidden_size, top_k, tp_size, ht_routable
):
    """A shape whose HT protocol is unroutable must fall back to the LL and BT
    routes rather than lose the profile."""
    config = _retargeted_config(tp_size, hidden_size, top_k)
    profile = config.profiles[0]

    # Raises when a route does not cover the requested workspace capacity.
    profile.validate_capacity(4096)
    for routes in (profile.finalize_routes, profile.all_reduce_routes):
        assert (_ht_target(routes) is not None) is ht_routable
        assert routes.is_unbounded


@pytest.mark.parametrize(
    ("model", "hidden_size", "top_k", "tp_size", "ht_routable"), _SHAPES
)
def test_ht_tunings_satisfy_the_kernel_constraint_expressions(
    model, hidden_size, top_k, tp_size, ht_routable
):
    """Pins the derivation against the device kernel's own validation; a wrong
    split otherwise aborts at compile time on a Blackwell node."""
    if not ht_routable:
        pytest.skip("HT is retired for this shape")
    profile = _retargeted_config(tp_size, hidden_size, top_k).profiles[0]
    preset = _ht_target(profile.finalize_routes)
    packs = hidden_size // 8

    # block_threads = consumer_threads + (2 + reduction_warps) * WARP_SIZE <= 1024
    assert preset.consumer_threads + (2 + preset.reduction_warps) * 32 <= 1024
    assert preset.consumer_threads % 32 == 0
    assert preset.reduction_warps in (1, 2, 4, 8)
    # hidden % (consumer_threads * VEC_BF16 * vectors_per_thread) == 0
    assert hidden_size % (preset.consumer_threads * 8 * preset.vectors_per_thread) == 0
    # packs_per_token must divide across tp, the consumers, and the RMS copy threads
    assert packs % tp_size == 0
    assert packs % preset.consumer_threads == 0
    assert packs % (preset.consumer_threads // preset.rms_token_groups) == 0
    # the reduction shard must divide evenly across reduction threads
    assert (packs // tp_size) % (preset.reduction_warps * 32) == 0
    if preset.rms_shard_major:
        rms_warps_per_token = (preset.consumer_threads // preset.rms_token_groups) // 32
        assert tp_size % rms_warps_per_token == 0 and tp_size >= rms_warps_per_token
    if preset.rms_pipeline_stages > 1:
        assert preset.rms_token_groups * preset.rms_pipeline_stages <= preset.stages


def test_retargeting_keeps_the_shipped_preset_schedule():
    """Only the shape-dependent fields move; the preset's pipeline depth and
    RMS schedule must survive the re-target."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import HT_FINALIZE_GB300_TP16_H8192_K10

    retargeted = _ht_retarget(
        HT_FINALIZE_GB300_TP16_H8192_K10, hidden_size=8192, tp_size=16
    )

    assert retargeted == HT_FINALIZE_GB300_TP16_H8192_K10


def test_shard_major_is_dropped_when_tp_cannot_cover_the_rms_warps():
    """The kernel raises on rms_shard_major when the RMS warps do not divide tp,
    so the re-target must downgrade it."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import HT_FINALIZE_GB300_TP16_H8192_K10

    assert HT_FINALIZE_GB300_TP16_H8192_K10.rms_shard_major is True
    # H=6144 splits into 384 consumer threads -> 6 RMS warps per token, and
    # 8 % 6 != 0.
    retargeted = _ht_retarget(
        HT_FINALIZE_GB300_TP16_H8192_K10, hidden_size=6144, tp_size=8
    )

    assert retargeted.consumer_threads == 384
    assert retargeted.rms_shard_major is False


def test_a_shipped_shape_is_served_by_the_shipped_config():
    """A shipped GB300 profile must reach the kernel untouched."""
    from flashinfer.comm.mnnvl_cutedsl import DEFAULT_CONFIG

    assert (
        _config_for_shape(DEFAULT_CONFIG, tp_size=8, hidden_size=8192, top_k=10)
        is DEFAULT_CONFIG
    )
    assert (
        _config_for_shape(DEFAULT_CONFIG, tp_size=8, hidden_size=7168, top_k=8)
        is not DEFAULT_CONFIG
    )


def test_unsupported_tp_width_is_refused_with_the_legal_set():
    """TP32 otherwise fails deep inside CuTe with "tp must be 2, 4, 8, or 16"."""
    assert 32 not in SUPPORTED_TP_SIZES
    with patch(
        "sglang.srt.layers.flashinfer_mnnvl_cutedsl.dist.get_world_size",
        return_value=32,
    ):
        with pytest.raises(ValueError, match="tp_size"):
            FlashInferMNNVLCuteDSLARFusion(
                hidden_size=8192,
                top_k=10,
                max_m=128,
                rms_epsilon=1e-6,
                weight_bias=0.0,
                process_group=object(),
                device=torch.device("cuda", 0),
            )


def test_a_draft_model_build_installs_no_fusion():
    """An EAGLE draft subclassing DeepseekV2Model reaches install_cutedsl_fusion
    with is_nextn False, so the guard must be the draft scope, not a per-model
    attribute."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import install_cutedsl_fusion
    from sglang.srt.layers.moe.utils import draft_model_build_scope

    def _layer():
        comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
        comm.input_layernorm = RMSNorm(8, eps=1e-6)
        comm.post_attention_layernorm = RMSNorm(8, eps=1e-6)
        return SimpleNamespace(layer_communicator=comm)

    kwargs = dict(
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: True,
        label="test",
    )
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")

    assert install_cutedsl_fusion([_layer(), _layer()], **kwargs) is not None
    with draft_model_build_scope():
        assert install_cutedsl_fusion([_layer(), _layer()], **kwargs) is None


def test_a_per_layer_epsilon_is_refused_at_install():
    """One workspace is compiled for one epsilon, and the kernel only checks
    the value the service passes, so disagreeing norms must raise here."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import install_cutedsl_fusion

    comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
    comm.input_layernorm = RMSNorm(8, eps=1e-6)
    comm.post_attention_layernorm = RMSNorm(8, eps=1e-5)  # disagrees
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")

    with pytest.raises(RuntimeError, match="rms_epsilon"):
        install_cutedsl_fusion(
            [SimpleNamespace(layer_communicator=comm)],
            hidden_size=8,
            top_k=2,
            rms_epsilon=1e-6,
            can_defer_finalize=lambda layer: True,
            label="test",
        )


def test_a_model_without_the_communicator_is_refused_not_silently_unfused():
    """Selecting cutedsl stands the legacy workspace down, so this would
    otherwise serve with every allreduce fusion off."""
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    # BaseRunner is abstract; call the method unbound with a duck-typed self.
    check = BaseRunner._assert_model_installs_cutedsl_fusion
    unfused = SimpleNamespace(model_runner=SimpleNamespace(model=torch.nn.Linear(2, 2)))

    with pytest.raises(ValueError, match="no CuTe DSL fusion communicator"):
        check(unfused)

    layer = torch.nn.Linear(2, 2)
    layer.layer_communicator = CuteDSLFusionLayerCommunicator.__new__(
        CuteDSLFusionLayerCommunicator
    )
    fused = SimpleNamespace(
        model_runner=SimpleNamespace(model=torch.nn.Sequential(layer))
    )

    check(fused)


def test_a_second_workspace_in_one_process_is_refused():
    """Each workspace rendezvouses its own NVLS region, so the second request
    must raise rather than allocate."""
    import sglang.srt.layers.flashinfer_mnnvl_cutedsl as mod

    saved = mod._WORKSPACE
    try:
        mod._WORKSPACE = object()
        with pytest.raises(RuntimeError, match="second MNNVL CuTe DSL"):
            mod.get_flashinfer_mnnvl_cutedsl_ar_fusion(
                hidden_size=8192, top_k=10, max_m=8, rms_epsilon=1e-6, weight_bias=0.0
            )
    finally:
        mod._WORKSPACE = saved


def test_the_handoff_constructs_inside_a_dynamo_traced_region():
    """Both producers build a handoff under fullgraph=True, and Dynamo cannot
    construct a msgspec.Struct, so migrating this container away from a frozen
    dataclass is a capture-time server failure rather than a graph break.

    Constructs the dataclass directly: from_flashinfer() would trace a stand-in
    producer instead, and fail for its own reasons.
    """

    def build(x):
        return MoeFinalizeHandoff(
            routed_output=x,
            expert_weights=x,
            permuted_indices=x,
            gated_shared_output=x,
            m=2,
        )

    handoff = torch.compile(build, fullgraph=True, backend="eager")(torch.zeros(4, 8))

    assert handoff.m == 2
    assert tuple(handoff.routed_output.shape) == (4, 8)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
