from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.flashinfer_mnnvl_cutedsl import _retargeted_config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    MoeFinalizeHandoff,
    install_cutedsl_fusion,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward, get_parallel, publish, reset_context
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
    tagged tensor reaches the final norm unreduced; it must not skip its own."""
    last = _communicator()
    last.successor_absorbs_all_reduce = False
    last.fusion_service = SimpleNamespace(
        all_reduce_residual_rms_norm=lambda *, local_contribution, residual, gamma: (
            local_contribution + 1,
            residual + 1,
        )
    )
    hidden_states = torch.zeros(8, 8)
    hidden_states._sglang_needs_allreduce_fusion = True

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "_finish_prepare_attn",
            lambda self, h, r, fb: (h, r),
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


def test_a_replicated_shared_expert_producer_keeps_its_own_all_reduce(eligible):
    """A TP1 shared expert is added after the layer's own reduction; handing that
    reduction onward would scale the shared output by tp_size."""
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    layers = [
        SimpleNamespace(layer_communicator=_communicator(), replicated=replicated)
        for replicated in (True, False, False)
    ]
    install_cutedsl_fusion(
        layers,
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: False,
        requires_local_reduction=lambda layer: layer.replicated,
        label="test",
    )
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


@pytest.mark.parametrize(
    ("moe_ep", "moe_tp", "moe_dp", "all_reduce", "finalize"),
    [
        (1, 4, 1, True, True),
        # EP x MoE-TP with MoE-DP 1 merges into one TP reduction.
        (2, 2, 1, True, False),
        (2, 2, 2, False, False),
    ],
)
def test_hybrid_ep_tp_fuses_the_all_reduce_only_when_the_legs_merge(
    moe_ep, moe_tp, moe_dp, all_reduce, finalize
):
    """Mirrors the base communicator's mergeable-EP x TP rule; the deferred
    finalize keeps its EP=1 restriction."""
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    comm = _communicator()
    comm.fusion_service = SimpleNamespace(supports=lambda m: True)
    comm._context = SimpleNamespace(tp_size=8)
    comm.layer_scatter_modes = SimpleNamespace(mlp_mode=ScatterMode.FULL)

    tp_size = moe_ep * moe_tp * moe_dp
    with get_parallel().override(
        moe_ep_size=moe_ep,
        moe_tp_size=moe_tp,
        moe_dp_size=moe_dp,
        tp_size=tp_size,
        attn_tp_size=tp_size,
    ):
        assert comm._common_eligible(_DECODE, 8) is all_reduce
        assert comm._should_use_finalize(_DECODE, 8) is finalize


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
def test_retargeted_profiles_cover_capacity_and_satisfy_the_kernel(
    hidden_size, top_k, tp_size, ht_routable
):
    """A wrong split aborts at compile time on a Blackwell node, and an
    unroutable HT must fall back to LL and BT rather than lose the profile."""
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    profile = _retargeted_config(tp_size, hidden_size, top_k).profiles[0]
    profile.validate_capacity(4096)
    for routes in (profile.finalize_routes, profile.all_reduce_routes):
        assert routes.is_unbounded
        ht = [t.preset for t in routes.targets if t.protocol is ProtocolKind.HT]
        assert bool(ht) is ht_routable
        if not ht:
            continue
        preset, packs = ht[0], hidden_size // 8
        # The device kernel's own validation expressions.
        assert preset.consumer_threads + (2 + preset.reduction_warps) * 32 <= 1024
        assert preset.consumer_threads % 32 == 0
        assert preset.reduction_warps in (1, 2, 4, 8)
        assert (
            hidden_size % (preset.consumer_threads * 8 * preset.vectors_per_thread) == 0
        )
        assert packs % tp_size == 0
        assert packs % (preset.consumer_threads // preset.rms_token_groups) == 0
        assert (packs // tp_size) % (preset.reduction_warps * 32) == 0
        if preset.rms_shard_major:
            rms_warps = (preset.consumer_threads // preset.rms_token_groups) // 32
            assert tp_size % rms_warps == 0 and tp_size >= rms_warps
        if preset.rms_pipeline_stages > 1:
            assert preset.rms_token_groups * preset.rms_pipeline_stages <= preset.stages


def test_the_handoff_constructs_inside_a_dynamo_traced_region():
    """Both producers build a handoff under fullgraph=True, and Dynamo cannot
    construct a msgspec.Struct, so migrating it off a frozen dataclass fails
    server startup at capture time."""

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
