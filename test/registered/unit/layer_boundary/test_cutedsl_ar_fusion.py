import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    FlashInferMNNVLCuteDSLARFusion,
    _retargeted_config,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.layer_boundary import (
    ADD,
    NORM_QUANT_READ,
    Layout,
    StageKind,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.layer_boundary.construction import BatchVariant
from sglang.srt.layers.layer_boundary.fusions.allreduce import (
    attention_fusions,
    complete_attention_input,
    complete_ffn_input,
    ffn_fusions,
)
from sglang.srt.layers.layer_boundary.fusions.cutedsl import (
    CuteDSLFusion,
    MoeFinalizeHandoff,
    install_cutedsl_fusion,
    prepare_cutedsl_fusion,
)
from sglang.srt.layers.layer_boundary.prepare import _consumer_step, _read_input
from sglang.srt.layers.layer_boundary.residual.access import finish_layer_stack
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.boundary_fixtures import prepare_input, stub_plan, stub_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_MODULE = "sglang.srt.layers.layer_boundary.fusions.cutedsl"
_DECODE = SimpleNamespace(forward_mode=ForwardMode.DECODE, input_ids=torch.zeros(8))


def _communicator():
    comm = stub_plan()
    comm.terminal = False
    comm.fusions = CuteDSLFusion()
    comm.norm = RMSNorm(8, eps=1e-6)
    comm._attn_input_fusions = attention_fusions(comm)
    # Only the ordinary batches' attention input half, with these entries.
    comm._paths[BatchVariant.SEQUENCE_PARALLEL] = comm._paths[
        BatchVariant.INPUT_SCATTERED
    ] = comm._paths[BatchVariant.CONTEXT_PARALLEL] = None
    comm._paths[BatchVariant.ORDINARY] = SimpleNamespace(
        entry=SimpleNamespace(
            prepare=partial(
                _consumer_step,
                adds_plainly=True,
                step=partial(
                    _read_input,
                    layer_input=None,
                    enters_stack=False,
                    read=NORM_QUANT_READ,
                    update=ADD,
                ),
                carried_fusions=comm._attn_input_fusions,
            ),
            input_move=None,
            input_sum=None,
            handoff=None,
        )
    )
    return comm


def _bound(entry):
    """What an entry runs and the layer it is bound to."""
    return entry.func, entry.args


def _test_layer(**kwargs):
    comm = _communicator()
    return SimpleNamespace(
        attn_boundary=stub_stage(comm, StageKind.ATTENTION),
        ffn_boundary=stub_stage(comm, StageKind.FFN),
        **kwargs,
    )


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
        patch.object(CuteDSLFusion, "_common_eligible", return_value=True),
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
    last.fusions.install(
        SimpleNamespace(
            all_reduce_residual_rms_norm=lambda *, local_contribution, residual, gamma: (
                local_contribution + 1,
                residual + 1,
            )
        ),
        hands_off_finalize=False,
    )
    hidden_states = UnreducedOutput(torch.zeros(8, 8))

    with (
        patch_communicator(
            "reduce_output",
            lambda *a, **k: pytest.fail("fell through to the unfused path"),
        ),
    ):
        out_hidden, _ = prepare_input(
            stub_stage(last, StageKind.ATTENTION),
            hidden_states,
            torch.zeros(8, 8),
            _DECODE,
        )

    assert torch.equal(out_hidden, torch.ones(8, 8))
    assert not last.fusions.can_defer_finalize(last, _DECODE)


def test_cutedsl_entries_come_before_the_base_fused_kernel():
    comm = _communicator()
    fusion = comm.fusions
    finalize, reduce, base_input = comm._attn_input_fusions
    assert _bound(finalize) == (
        fusion._finalize_output_and_update_and_read_residual,
        (comm,),
    )
    assert _bound(reduce) == (
        fusion._reduce_output_and_update_and_read_residual,
        (comm,),
    )
    assert _bound(base_input) == (complete_attention_input, (comm,))
    base = (complete_ffn_input, (comm,))
    cutedsl = (fusion._mlp_input_reduce_output_and_update_and_read_residual, (comm,))
    with patch(
        f"{_MODULE}.get_parallel",
        return_value=SimpleNamespace(attn_tp_size=2, tp_size=2),
    ):
        comm.variants = {
            BatchVariant.ORDINARY: SimpleNamespace(
                incoming=SimpleNamespace(residual=Layout(frozenset()))
            )
        }
        fusions = ffn_fusions(comm)
        assert _bound(fusions[0].run) == cutedsl
        assert _bound(fusions[1].run) == base
        # The workspace reduces over the TP group, which is the attention-TP group here.
        assert [f.completes for f in fusions] == [SumGroup.ATTN_TP] * 2
        # A residual on each rank's slice is gathered first, which the
        # workspace does not do.
        comm.variants = {
            BatchVariant.ORDINARY: SimpleNamespace(
                incoming=SimpleNamespace(
                    residual=Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
                )
            )
        }
        assert [_bound(f.run) for f in ffn_fusions(comm)] == [base]


def test_the_fusion_runs_only_on_the_ffn_full_rows():
    """The finalize and the AR + norm sum over the whole TP group, so they need
    the batch's FFN input on the full rows, not each rank's own slice (a2a, the
    fp4 all-gather, DWDP or a dense MLP on every rank)."""
    comm = _communicator()
    fusion = comm.fusions
    fusion.service = SimpleNamespace(supports=lambda m: True)
    with (
        patch(
            f"{_MODULE}.get_parallel",
            return_value=SimpleNamespace(attn_cp_size=1, tp_size=2),
        ),
        patch(f"{_MODULE}.is_dp_attention_enabled", return_value=False),
        patch(
            f"{_MODULE}.get_attn_tp_context",
            return_value=SimpleNamespace(input_scattered=False),
        ),
        patch(
            f"{_MODULE}.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
    ):
        for sharded, eligible in (
            (frozenset(), True),
            (frozenset({TokenAxis.ATTN_TP_SCATTER}), False),
        ):
            comm._batch_steps = lambda fb, rows=Layout(sharded): SimpleNamespace(
                entry=SimpleNamespace(input_rows=rows)
            )
            assert fusion._common_eligible(comm, _DECODE, 8) is eligible


def test_finalize_handoff_is_a_producer_capability(eligible):
    comm = _communicator()
    fusion = comm.fusions
    assert not fusion.can_defer_finalize(comm, _DECODE)
    fusion.install(SimpleNamespace(), hands_off_finalize=True)
    with patch.object(CuteDSLFusion, "_should_use_finalize", return_value=True):
        assert fusion.can_defer_finalize(comm, _DECODE)


def test_install_does_not_require_a_fused_successor():
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    first, last = _communicator(), _communicator()
    ordinary = SimpleNamespace(ffn_boundary=SimpleNamespace(fusions=None))
    install_cutedsl_fusion(
        [
            SimpleNamespace(
                attn_boundary=stub_stage(first, StageKind.ATTENTION),
                ffn_boundary=stub_stage(first, StageKind.FFN),
            ),
            ordinary,
            SimpleNamespace(
                attn_boundary=stub_stage(last, StageKind.ATTENTION),
                ffn_boundary=stub_stage(last, StageKind.FFN),
            ),
        ],
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: True,
        label="test",
    )
    assert first.fusions.hands_off_finalize
    assert last.fusions.hands_off_finalize


def test_a_service_nested_under_a_wrapper_is_prepared():
    """A VLM wrapper holds the model as a submodule and has no pre-capture hook;
    an unprepared service declines every M, leaving no fusion at all."""
    prepared = []
    layer = torch.nn.Linear(2, 2)
    comm = _communicator()
    layer.attn_boundary, layer.ffn_boundary = (
        stub_stage(comm, StageKind.ATTENTION),
        stub_stage(comm, StageKind.FFN),
    )
    wrapper = torch.nn.Module()
    wrapper.language_model = torch.nn.Sequential(layer)
    _install([layer])
    layer.ffn_boundary.fusions.service.prepare = lambda *, max_m: prepared.append(max_m)

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
        reduce=lambda h: h,
    )

    assert handoff.routed_output.data_ptr() == gemm2_out.data_ptr()
    assert handoff.permuted_indices.data_ptr() == permuted_indices.data_ptr()
    assert tuple(handoff.expert_weights.shape) == (m, top_k)


def _handoff(finish):
    rows = torch.zeros(2, 8)
    return MoeFinalizeHandoff(
        routed_output=rows,
        expert_weights=rows,
        permuted_indices=rows,
        gated_shared_output=rows,
        m=2,
        finish=finish,
    )


def test_its_producer_completes_a_handoff_for_any_other_reader():
    """Only the fused kernel reads the handoff's parts; everything else gets
    the MoE's own unfused tail, run once."""
    finished = torch.full((2, 8), 3.0)
    finish = MagicMock(return_value=finished)
    assert reduce_output(_handoff(finish)) is finished
    finish.assert_called_once_with()


def test_from_flashinfer_completes_with_the_moe_s_own_sum():
    deferred = SimpleNamespace(
        gemm2_out=torch.zeros(4, 8),
        expert_weights=torch.zeros(2, 2),
        expanded_idx_to_permuted_idx=torch.zeros(2, 2, dtype=torch.int32),
        top_k=2,
    )
    shared = torch.ones(2, 8)
    with patch(
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm."
        "finalize_flashinfer_trtllm_deferred_output",
        side_effect=lambda d, s: s + 1,
    ) as finalize:
        handoff = MoeFinalizeHandoff.from_flashinfer(
            deferred, gated_shared_output=shared, m=2, reduce=lambda h: h * 10
        )
        finalize.assert_not_called()
        torch.testing.assert_close(handoff.complete(), torch.full((2, 8), 20.0))
    finalize.assert_called_once_with(deferred, shared)


def test_a_handoff_the_kernel_does_not_take_is_completed_then_normed():
    """The consumer asks only whether its own kernel takes the batch; when it
    does not, the handoff is completed and the input read as usual."""

    class AddNorm(torch.nn.Module):
        def forward(self, x, residual=None, post_residual_addition=None):
            residual.add_(x)
            return residual * 2, residual

    comm = _communicator()
    comm.norm = AddNorm()
    comm.fusions.service = SimpleNamespace(
        finalize=lambda **kw: pytest.fail("the kernel does not take this batch")
    )
    finish = MagicMock(return_value=torch.ones(2, 8))
    with (
        # A norm the kernel could fold in, on a batch it does not take.
        patch(f"{_MODULE}._fused_norm_gamma", return_value=torch.ones(8)),
        patch.object(CuteDSLFusion, "_should_use_finalize", return_value=False),
        patch.object(CuteDSLFusion, "_common_eligible", return_value=False),
    ):
        hidden, residual = prepare_input(
            stub_stage(comm, StageKind.ATTENTION),
            _handoff(finish),
            torch.full((2, 8), 2.0),
            _DECODE,
        )
    finish.assert_called_once_with()
    torch.testing.assert_close(residual.residual, torch.full((2, 8), 3.0))
    torch.testing.assert_close(hidden, torch.full((2, 8), 6.0))


def test_the_layer_stack_hands_a_handoff_only_to_a_final_norm_that_takes_it():
    for takes, completed in ((True, 0), (False, 1)):
        finish = MagicMock(return_value=torch.ones(2, 8))
        handoff = _handoff(finish)
        hidden, _ = finish_layer_stack(
            handoff, torch.zeros(2, 8), _DECODE, final_norm_takes_handoff=takes
        )
        assert finish.call_count == completed
        assert (hidden is handoff) == takes


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
    """The captured dual-stream path returns tensors without deferred handoffs."""

    class _DeferRecordingMoE:
        def forward_normal_dual_stream(self, hidden_states):
            self.seen_defer = get_forward().defer_moe_finalize
            return object() if self.seen_defer else hidden_states + 1

    reset_context()
    fusion = _DeferRecordingMoE()
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    with get_forward().scoped(defer_moe_finalize=True):
        out = DeepseekV2MoE._forward_moe_dual_stream_graph(
            fusion, torch.zeros(4, 8), True, False
        )
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


class TestDeferredLoraAllReduce(unittest.TestCase):
    def test_installed_cutedsl_provider_restores_deferred_sum(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.layers.layer_boundary import exit as exits
        from sglang.srt.layers.layer_boundary.fusions.cutedsl import CuteDSLFusion

        group = object()
        fb = SimpleNamespace(
            input_ids=torch.zeros(2), residual_stream=SimpleNamespace(residual=None)
        )
        provider = CuteDSLFusion()
        boundary = SimpleNamespace(fusions=provider)
        for eligible, a2a in ((True, False), (False, False), (True, True)):
            with (
                patch.object(exits, "_ffn_has_tokens", return_value=True),
                patch.object(
                    exits, "post_experts_sum_is_one_all_reduce", return_value=False
                ),
                patch.object(
                    exits, "get_lora", return_value=SimpleNamespace(enable_lora=True)
                ),
                patch.object(
                    exits,
                    "get_exec",
                    return_value=SimpleNamespace(
                        comm=SimpleNamespace(enable_quant_communications=False)
                    ),
                ),
                patch.object(
                    exits,
                    "get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: not a2a),
                ),
                patch.object(
                    exits, "get_parallel", return_value=SimpleNamespace(tp_group=group)
                ),
                patch.object(exits, "post_experts_reduction_group", return_value=group),
                patch.object(
                    exits, "apply_flashinfer_allreduce_fusion", return_value=False
                ),
                patch.object(
                    provider, "can_defer_all_reduce", return_value=eligible
                ) as gate,
            ):
                # Avoid platform-specific Aiter details: no residual cannot fuse.
                with patch.object(exits, "_use_aiter", False, create=True):
                    result = exits._can_defer_ffn_reduction(fb, boundary)
                self.assertEqual(result, eligible and not a2a)
                if a2a:
                    gate.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
