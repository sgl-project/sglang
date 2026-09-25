from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.communicator import LayerCommunicator, UnreducedOutput
from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    FlashInferMNNVLCuteDSLARFusion,
    _retargeted_config,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    MoeFinalizeHandoff,
    install_cutedsl_fusion,
    prepare_cutedsl_fusion,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_MODULE = "sglang.srt.layers.moe.cutedsl_ar_fusion"
_DECODE = SimpleNamespace(forward_mode=ForwardMode.DECODE, input_ids=torch.zeros(8))


def _communicator():
    comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
    comm.input_layernorm = RMSNorm(8, eps=1e-6)
    comm.post_attention_layernorm = RMSNorm(8, eps=1e-6)
    return comm


def _install(layers, **kwargs):
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    return install_cutedsl_fusion(
        layers,
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: False,
        label="test",
        **kwargs,
    )


@pytest.fixture
def eligible():
    """Every shared gate open, so a case varies only the predicate it names."""
    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch(
            f"{_MODULE}.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
    ):
        yield


def test_last_layer_consumes_but_does_not_skip_the_pending_all_reduce(eligible):
    """The last layer must fuse the reduction its predecessor skipped, or the
    unreduced output reaches the final norm; it must not skip its own."""
    last = _communicator()
    last.successor_absorbs_all_reduce = False
    last.fusion_service = SimpleNamespace(
        all_reduce_residual_rms_norm=lambda *, local_contribution, residual, gamma: (
            local_contribution + 1,
            residual + 1,
        )
    )
    hidden_states = UnreducedOutput(torch.zeros(8, 8))

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "_finish_prepare_attn",
            lambda self, *, hidden_states, residual, forward_batch: (
                hidden_states,
                residual,
            ),
        ),
        patch.object(
            LayerCommunicator,
            "prepare_attn",
            lambda *a, **k: pytest.fail("fell through to the unfused path"),
        ),
    ):
        out_hidden, _ = last.prepare_attn(hidden_states, torch.zeros(8, 8), _DECODE)

    assert torch.equal(out_hidden, torch.ones(8, 8))
    assert last._can_absorb_post_moe_all_reduce(_DECODE, 8) is False


def test_a_replicated_output_producer_keeps_its_own_all_reduce(eligible):
    """A TP1 shared expert (or TP1 dense MLP) is not partial; handing its layer's
    reduction onward would scale the replicated output by tp_size."""
    layers = [
        SimpleNamespace(layer_communicator=_communicator(), replicated=replicated)
        for replicated in (True, False, False)
    ]
    _install(layers, requires_local_reduction=lambda layer: layer.replicated)
    replicated, plain = (layer.layer_communicator for layer in layers[:2])

    with patch.object(
        LayerCommunicator,
        "should_fuse_mlp_allreduce_with_next_layer",
        return_value=False,
    ):
        assert replicated.should_fuse_mlp_allreduce_with_next_layer(_DECODE) is False
        assert plain.should_fuse_mlp_allreduce_with_next_layer(_DECODE) is True
    # Consuming what a predecessor skipped stays independently eligible.
    assert replicated._can_consume_post_moe_all_reduce(_DECODE, 8) is True


def test_a_service_nested_under_a_wrapper_is_prepared():
    """A VLM wrapper holds the model as a submodule and has no pre-capture hook;
    an unprepared service declines every M, leaving no fusion at all."""
    prepared = []
    layer = torch.nn.Linear(2, 2)
    layer.layer_communicator = _communicator()
    wrapper = torch.nn.Module()
    wrapper.language_model = torch.nn.Sequential(layer)
    _install([layer])
    layer.layer_communicator.fusion_service.prepare = lambda *, max_m: prepared.append(
        max_m
    )

    # The workspace M bound is the largest of every framework source.
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
    with patch(f"{_MODULE}.cutedsl_moe_max_num_tokens", return_value=8192):
        prepare_cutedsl_fusion(wrapper, max_running_requests=2048)
    assert prepared == [8192]

    with pytest.raises(ValueError, match="installed no CuTe DSL fusion service"):
        prepare_cutedsl_fusion(torch.nn.Linear(2, 2), max_running_requests=2048)


def test_wrapper_passes_no_routed_scaling_factor_to_the_kernel():
    """The routed output is already scaled; forwarding the factor would scale
    DeepSeek's routed contribution twice."""
    calls = []
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.hidden_size, wrapper.device = 8, torch.device("cpu")
    wrapper.rms_epsilon, wrapper.weight_bias = 1e-6, 0.0
    wrapper.workspace = object()
    wrapper.supports = lambda m: True
    wrapper._patterns = SimpleNamespace(
        kARResidualRMSNorm=1, kMoEFinalizeARResidualRMSNorm=7
    )
    wrapper._allreduce_fusion = lambda **kwargs: calls.append(kwargs)
    x = torch.empty(4, 8, dtype=torch.bfloat16)
    wrapper.moe_finalize_all_reduce_rms_norm(
        routed_output=torch.empty(8, 8, dtype=torch.bfloat16),
        expert_weights=torch.empty(4, 2, dtype=torch.bfloat16),
        permuted_indices=torch.empty(4, 2, dtype=torch.int32),
        gated_shared_output=x,
        residual=x,
        gamma=torch.empty(8, dtype=torch.bfloat16),
    )
    wrapper.all_reduce_residual_rms_norm(
        local_contribution=x, residual=x, gamma=torch.empty(8, dtype=torch.bfloat16)
    )

    assert [call["pattern"] for call in calls] == [7, 1]
    assert all("routed_scaling_factor" not in call for call in calls)


def test_the_handoff_views_the_producer_storage():
    """The handoff is consumed in place by the next layer's kernel; a copy would
    cost the [M*top_k, hidden] round trip the fusion exists to save."""
    m, top_k = 3, 10
    gemm2_out = torch.empty(m * top_k + 4, 16, dtype=torch.bfloat16)
    expert_weights = torch.empty(m + 1, top_k, dtype=torch.bfloat16)
    permuted_indices = torch.empty(m + 1, top_k, dtype=torch.int32)
    handoff = MoeFinalizeHandoff.from_flashinfer(
        SimpleNamespace(
            gemm2_out=gemm2_out,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=permuted_indices,
            top_k=top_k,
        ),
        gated_shared_output=torch.empty(m, 16, dtype=torch.bfloat16),
        m=m,
    )

    assert handoff.routed_output.data_ptr() == gemm2_out.data_ptr()
    assert handoff.permuted_indices.data_ptr() == permuted_indices.data_ptr()
    assert tuple(handoff.expert_weights.shape) == (m, top_k)


def test_early_shared_load_touches_only_the_finalize_routes():
    """Only a fused finalize has a completed shared-expert handoff to load early;
    the standalone all-reduce must keep the safe ordering."""
    from flashinfer.comm.mnnvl_cutedsl import DEFAULT_CONFIG
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import HTFinalizeTuning

    config = _with_early_finalize_shared_load(DEFAULT_CONFIG)

    for before, after in zip(DEFAULT_CONFIG.profiles, config.profiles):
        assert after.all_reduce_routes == before.all_reduce_routes
        for target in after.finalize_routes.targets:
            # HT has no shared-load ordering option.
            if not isinstance(target.preset, HTFinalizeTuning):
                assert target.preset.load_shared_expert_before_pdl is True


def test_dual_stream_op_pins_the_deferral_off_under_a_deferring_caller():
    """The op's Tensor schema cannot carry a handoff; the dispatcher would raise
    "Unable to cast ... to Tensor"."""
    from sglang.srt.models.deepseek_v2 import (  # noqa: F401  (registers the op)
        dsv2_flashinfer_moe_dual_stream_graph,
    )

    class _DeferRecordingMoE:
        def forward_normal_dual_stream(self, hidden_states):
            self.seen_defer = get_forward().defer_moe_finalize
            return object() if self.seen_defer else hidden_states + 1

    reset_context()
    fusion = _DeferRecordingMoE()
    op = torch.ops.sglang.dsv2_flashinfer_moe_dual_stream_graph.default
    # The CUDA key runs the real schema while the stub keeps tensors on CPU.
    cuda_key = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    with (
        get_forward().scoped(defer_moe_finalize=True),
        patch(
            "sglang.srt.models.deepseek_v2.get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(moe_fusions={0: fusion}),
        ),
    ):
        out = op.redispatch(cuda_key, torch.zeros(4, 8), 0, True, False)
        assert get_forward().defer_moe_finalize is True

    assert fusion.seen_defer is False
    assert torch.equal(out, torch.ones(4, 8))


def _flashinfer_accepts(preset, *, hidden_size, top_k, tp_size):
    """FlashInfer's own HT kernel validation, which runs before any compile."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht.device_kernel import (
        _MoeFinalizeAllReduceRMSNormHTDeviceKernel,
    )

    _MoeFinalizeAllReduceRMSNormHTDeviceKernel(
        hidden=hidden_size,
        top_k=top_k,
        tp=tp_size,
        rank=0,
        # Arbitrary grid when the preset leaves it to the device's SM count.
        active_ctas=preset.persistent_ctas or tp_size * 8,
        stages=preset.stages,
        consumer_threads=preset.consumer_threads,
        vectors_per_thread=preset.vectors_per_thread,
        reduction_warps=preset.reduction_warps,
        reduction_cta_groups=preset.reduction_cta_groups,
        rms_token_groups=preset.rms_token_groups,
        rms_pipeline_stages=preset.rms_pipeline_stages,
        rms_shard_major=preset.rms_shard_major,
        rms_epsilon=1e-6,
        routed_scaling_factor=1.0,
        weight_bias=0.0,
        include_shared_expert=True,
        add_residual=True,
        write_residual_output=True,
        enable_pdl=preset.enable_pdl,
    )


# (hidden_size, top_k, tp_size, HT routable), from the checkpoint configs of
# Qwen3.8, DeepSeek-V3 and GLM-5.3. False: the vectors per reduction shard are
# not a warp multiple at that width, which the kernel rejects.
@pytest.mark.parametrize(
    ("hidden_size", "top_k", "tp_size", "ht_routable"),
    [
        (8192, 10, 8, True),
        (8192, 10, 16, True),
        (7168, 8, 4, True),
        (7168, 8, 8, False),
        (7168, 8, 16, False),
        (6144, 8, 4, True),
        (6144, 8, 8, True),
        (6144, 8, 16, False),
    ],
)
def test_retargeted_profiles_cover_capacity_and_pass_flashinfer_validation(
    hidden_size, top_k, tp_size, ht_routable
):
    """A wrong split aborts at compile time on a Blackwell node, and an
    unroutable HT must fall back to LL and BT rather than lose the profile."""
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    profile = _retargeted_config(tp_size, hidden_size, top_k).profiles[0]
    profile.validate_capacity(4096)
    for routes, routed_top_k in (
        (profile.finalize_routes, top_k),
        (profile.all_reduce_routes, 0),
    ):
        assert routes.is_unbounded
        ht = [t.preset for t in routes.targets if t.protocol is ProtocolKind.HT]
        assert bool(ht) is ht_routable
        for preset in ht:
            _flashinfer_accepts(
                preset, hidden_size=hidden_size, top_k=routed_top_k, tp_size=tp_size
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
