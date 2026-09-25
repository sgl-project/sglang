"""The boundaries of an attention and an FFN, chosen from both sides'
declarations. The FFN runs on the TP group (a dense MLP, or a MoE not
dispatched per DP shard) or on each attention-TP rank's slice of its DP shard's
rows (a MoE dispatched per DP shard).

The numeric checks run every rank of a DP x attention-TP world as a thread over
fake collectives that wait for all members of their group, build the layers'
communicators for real, and pass them scatter modes that fail on any read of a
mode field.
"""

import itertools
import threading
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.boundary_layout import (
    SumGroup,
    TokenAxis,
    decoder_layer_sides,
    input_scattered_layer_sides,
    sequence_parallel_layer_sides,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    UnreducedOutput,
    scatter_mode_layouts,
)
from sglang.srt.layers.communicator_dsa_cp import DSACPLayerCommunicator
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator
from sglang.srt.layers.moe.cutedsl_ar_fusion import CuteDSLFusionLayerCommunicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 4


def parallel_of(*, attn_dp, attn_tp, attn_cp=1, **overrides):
    fields = dict(
        tp_size=attn_dp * attn_cp * attn_tp,
        tp_rank=0,
        attn_dp_size=attn_dp,
        attn_tp_size=attn_tp,
        attn_tp_rank=0,
        attn_cp_size=attn_cp,
        attn_cp_rank=0,
        moe_dense_tp_size=None,
        enable_prefill_cp=False,
        moe_ep_size=1,
        moe_tp_size=attn_dp * attn_cp * attn_tp,
        moe_dp_size=1,
        dwdp_size=1,
        enable_dp_attention=attn_dp > 1,
        enable_attn_tp_input_scattered=False,
        tp_group=SimpleNamespace(name="tp"),
        attn_tp_group=SimpleNamespace(name="attn_tp"),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@contextmanager
def planning(parallel, *, sp=False, a2a=False):
    """What layer planning and communicator construction read, without the
    process-wide parallel state. ``parallel`` may be a callable, for a
    per-thread parallel state."""
    get_parallel = parallel if callable(parallel) else (lambda: parallel)

    def moe_cp_gathers():
        return get_parallel().attn_cp_size > get_parallel().moe_dp_size

    with (
        patch.object(comm, "get_parallel", get_parallel),
        patch.object(comm, "is_dsa_enable_prefill_cp", lambda: False),
        patch.object(comm, "is_mla_cp_enabled", lambda: False),
        # MoE-CP gathers over the whole CP group when the MoE's DP is narrower.
        patch.object(
            comm,
            "get_moe_cp_size",
            lambda: get_parallel().attn_cp_size if moe_cp_gathers() else 1,
        ),
        patch.object(comm.layernorm_sp, "layernorm_sp_enabled", lambda: sp),
        patch.object(
            comm, "get_spec", lambda: SimpleNamespace(speculative_algorithm=None)
        ),
        patch.object(
            comm,
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(is_none=lambda: not a2a),
        ),
        patch.object(comm, "is_moe_input_scattered_across_dp_ranks", lambda: a2a),
        patch.object(
            comm, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
        ),
        patch.object(comm, "is_enable_moe_cp_allgather", moe_cp_gathers),
        patch.object(comm, "get_lora", lambda: SimpleNamespace(enable_lora=False)),
    ):
        yield


class Norm:
    """add + norm stand-in: (2 * (h + r), h + r); one input: 2 * h."""

    def __call__(self, x, residual=None, post_residual_addition=None):
        assert post_residual_addition is None
        if residual is None:
            return x * 2
        s = x + residual
        return s * 2, s


class FusableNorm(Norm):
    """A norm with the fused all-reduce + add + norm method."""

    def forward_with_allreduce_fusion(self, *args, **kwargs):
        raise AssertionError("not run here")


class FactsOnly:
    """Scatter modes whose mode fields fail when read: only the layer facts
    are there."""

    def __init__(self, **facts):
        self.__dict__.update(facts)

    def __getattr__(self, name):
        raise AssertionError(f"read scatter mode {name!r}")


def layer_facts(layer_id, num_layers, *, sparse=False, previous_sparse=False):
    return FactsOnly(
        is_layer_sparse=sparse,
        is_first_layer=layer_id == 0,
        is_last_layer=layer_id == num_layers - 1,
        is_previous_layer_sparse=None if layer_id == 0 else previous_sparse,
    )


def sides_of(
    axis_sizes,
    *,
    sparse=False,
    a2a=False,
    previous_a2a=False,
    last=False,
    leaves_next=False,
    leaves_rs=False,
):
    return decoder_layer_sides(
        axis_sizes=axis_sizes,
        ffn_on_local_rows=sparse and a2a,
        previous_on_local_rows=previous_a2a,
        is_last_layer=last,
        attention_gathers_local_rows=False,
        ffn_group=SumGroup.MOE_OUTPUT if sparse else SumGroup.TP,
        leaves_for_next_layer=leaves_next,
        leaves_for_reduce_scatter=leaves_rs,
        leaves_for_reduce_scatterv=leaves_rs or sparse,
    )


def build(modes, parallel, *, cls=LayerCommunicator, sp=False, a2a=False, **kwargs):
    with planning(parallel, sp=sp, a2a=a2a):
        return cls(
            layer_scatter_modes=modes,
            input_layernorm=Norm(),
            post_attention_layernorm=Norm(),
            **kwargs,
        )


def planned_modes(
    layer_id, num_layers, *, sparse, previous_sparse, parallel, a2a=False
):
    with planning(parallel, a2a=a2a):
        return LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=num_layers,
            is_layer_sparse=sparse,
            is_previous_layer_sparse=previous_sparse,
            is_next_layer_sparse=False,
        )


class TestDeclarationsMatchScatterModes(CustomTestCase):
    """The layouts derived from the groups each side computes over are the
    layouts of the modes layer planning gives the same layers."""

    def test_every_attention_dp_and_tp(self):
        for (
            attn_dp,
            attn_tp,
            layer_id,
            sparse,
            previous_sparse,
            a2a,
        ) in itertools.product(
            (1, 2, 4, 8),
            (1, 2, 4),
            (0, 1, 3),
            (False, True),
            (False, True),
            (False, True),
        ):
            with self.subTest(
                attn_dp=attn_dp,
                attn_tp=attn_tp,
                layer_id=layer_id,
                sparse=sparse,
                previous_sparse=previous_sparse,
                a2a=a2a,
            ):
                parallel = parallel_of(attn_dp=attn_dp, attn_tp=attn_tp)
                modes = planned_modes(
                    layer_id,
                    4,
                    sparse=sparse,
                    previous_sparse=previous_sparse,
                    parallel=parallel,
                    a2a=a2a,
                )
                sizes = {
                    TokenAxis.ATTN_DP: attn_dp,
                    TokenAxis.ATTN_CP: 1,
                    TokenAxis.ATTN_TP_SCATTER: attn_tp,
                }
                layouts = scatter_mode_layouts(
                    attn_dp_size=attn_dp, attn_cp_size=1, attn_tp_size=attn_tp
                )
                sides = sides_of(
                    sizes,
                    sparse=sparse,
                    a2a=a2a,
                    previous_a2a=layer_id > 0 and previous_sparse and a2a,
                    last=layer_id == 3,
                    leaves_next=True,
                    leaves_rs=True,
                )
                self.assertEqual(sides.input_rows, layouts[modes.layer_input_mode])
                self.assertEqual(
                    sides.ffn_residual_rows, layouts[modes.middle_residual_mode]
                )
                self.assertEqual(sides.output_rows, layouts[modes.layer_output_mode])
                self.assertEqual(sides.attention.layout, layouts[modes.attn_mode])
                self.assertEqual(sides.ffn.layout, layouts[modes.mlp_mode])
                self.assertEqual(
                    sides.attention_output.layout, layouts[modes.attn_mode]
                )
                self.assertEqual(sides.ffn_output.layout, sides.ffn.layout)
                # A MoE dispatched per DP shard hands on a complete output.
                self.assertIs(sides.ffn_output.group is None, sparse and a2a)

    def test_the_attention_output_owes_the_attention_tp_sum(self):
        for attn_tp in (1, 2):
            sides = sides_of(
                {
                    TokenAxis.ATTN_DP: 2,
                    TokenAxis.ATTN_CP: 1,
                    TokenAxis.ATTN_TP_SCATTER: attn_tp,
                }
            )
            owed = attn_tp > 1
            self.assertIs(
                sides.attention_output.group, SumGroup.ATTN_TP if owed else None
            )
            self.assertIs(sides.attention_output.always_leaves, owed)
            self.assertIs(sides.ffn_output.group, SumGroup.TP)
            self.assertFalse(sides.ffn_output.always_leaves)


class TestWhichLayersUseDeclarations(CustomTestCase):
    """The layer is the unit: all three boundary sides it owns follow the
    declarations, or none do."""

    def declared(self, communicator):
        return getattr(communicator._steps.ffn_input, "func", None) in (
            comm._mlp_input_dp_partial,
            comm._mlp_input_dp_replicate,
            comm._mlp_input_without_dp,
            comm._mlp_input_scatter,
        )

    def test_a_dense_model(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                communicator = build(layer_facts(layer_id, 3), parallel)
                self.assertTrue(self.declared(communicator))
                self.assertIs(
                    communicator._steps.attention_input,
                    comm.CommunicateSimpleFn._trivial,
                )
                self.assertTrue(communicator._steps.returns_over_dp)

    def test_plain_tp(self):
        # Without attention DP nothing is gathered: the attention-TP sum, then
        # add + norm, with the fused kernel first when the norm has it.
        for norm, fuses in ((Norm, False), (FusableNorm, True)):
            with self.subTest(norm=norm.__name__):
                with planning(parallel_of(attn_dp=1, attn_tp=2)):
                    communicator = LayerCommunicator(
                        layer_scatter_modes=layer_facts(1, 3),
                        input_layernorm=norm(),
                        post_attention_layernorm=norm(),
                    )
                self.assertIs(
                    communicator._steps.ffn_input.func, comm._mlp_input_without_dp
                )
                entry = (
                    communicator._mlp_input_reduce_output_and_update_and_read_residual
                )
                self.assertEqual(
                    communicator._steps.ffn_input.keywords["fusions"],
                    (entry,) if fuses else (),
                )
                self.assertIs(
                    any(f.may_return_new_residual for f in communicator._steps.fused),
                    fuses,
                )
                self.assertFalse(communicator._steps.returns_over_dp)
                self.assertIs(
                    communicator._steps.ffn_output_move,
                    comm.CommunicateSummableTensorPairFn._trivial,
                )
        # One rank: the attention output is complete, only the norm is left.
        one_rank = parallel_of(attn_dp=1, attn_tp=1)
        single = build(layer_facts(1, 3), one_rank)
        with planning(one_rank):
            self.assertIsNotNone(single._declared_sides())
        self.assertIs(single._steps.ffn_input, comm._mlp_input_norm)

    def test_the_order_follows_the_attention_output(self):
        for attn_tp, force, order in (
            (2, False, comm._mlp_input_dp_partial),
            (2, True, comm._mlp_input_dp_replicate),
            (1, False, comm._mlp_input_dp_replicate),
        ):
            with self.subTest(attn_tp=attn_tp, force=force):
                communicator = build(
                    layer_facts(1, 3),
                    parallel_of(attn_dp=2, attn_tp=attn_tp),
                    force_layernorm_before_dp_gather=force,
                )
                self.assertIs(communicator._steps.ffn_input.func, order)

    def test_a_dense_first_layer_before_sparse_ones(self):
        # DeepSeek-V2-Lite: layer 0 is dense, the rest sparse; a dense layer
        # after a sparse one as well. Every layer takes the declared path.
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for a2a in (False, True):
            with self.subTest(a2a=a2a):
                layers = [
                    build(
                        planned_modes(
                            i,
                            4,
                            sparse=sparse,
                            previous_sparse=previous,
                            parallel=parallel,
                            a2a=a2a,
                        ),
                        parallel,
                        a2a=a2a,
                        allow_reduce_scatter=True,
                    )
                    for i, (sparse, previous) in enumerate(
                        ((False, False), (True, False), (True, True), (False, True))
                    )
                ]
                self.assertEqual([self.declared(layer) for layer in layers], [True] * 4)
                if not a2a:
                    moe = layers[1]._steps.ffn_output
                    self.assertIs(moe.group, SumGroup.MOE_OUTPUT)
                    self.assertTrue(moe.leaves_for_reduce_scatterv)
                    continue
                # The first a2a layer slices the residual it takes from a
                # dense layer; the next one takes it already sliced.
                self.assertEqual(
                    [layers[i]._steps.ffn_input.func for i in (1, 2)],
                    [comm._mlp_input_scatter] * 2,
                )
                self.assertEqual(
                    [
                        layers[i]._steps.ffn_input.keywords["scatters_residual"]
                        for i in (1, 2)
                    ],
                    [True, False],
                )
                self.assertIsNone(layers[1]._steps.ffn_output.group)
                # Their rows are gathered back for attention, and a dense layer
                # after them gathers the residual back as well.
                self.assertIs(
                    layers[2]._steps.attention_input,
                    comm.CommunicateSimpleFn._scattered_to_tp_attn_full,
                )
                self.assertIs(
                    layers[3]._steps.ffn_input.func, comm._mlp_input_dp_partial
                )
                self.assertTrue(layers[3]._steps.ffn_input.keywords["gathers_residual"])

    def test_the_last_a2a_layer_folds_the_residual_back(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        last = build(
            planned_modes(
                3, 4, sparse=True, previous_sparse=True, parallel=parallel, a2a=True
            ),
            parallel,
            a2a=True,
        )
        self.assertIs(
            last._steps.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._gather,
        )

    def test_an_attention_that_gathers_its_slice_itself(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        modes = planned_modes(
            2, 4, sparse=True, previous_sparse=True, parallel=parallel, a2a=True
        )
        with patch.object(comm, "_use_ag_after_qlora", True):
            layer = build(modes, parallel, a2a=True)
        self.assertIs(layer._steps.attention_input, comm.CommunicateSimpleFn._trivial)

    def test_layers_that_keep_the_scatter_modes(self):
        dp = parallel_of(attn_dp=2, attn_tp=2)
        cases = {
            "attention CP": (
                layer_facts(1, 3),
                parallel_of(attn_dp=2, attn_tp=1, attn_cp=2),
            ),
            "dense MLP fully DP": (
                layer_facts(1, 3),
                parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1),
            ),
        }
        for name, (facts, parallel) in cases.items():
            with self.subTest(name):
                with planning(parallel):
                    communicator = LayerCommunicator.__new__(LayerCommunicator)
                    communicator.layer_scatter_modes = facts
                    communicator.allow_deferred_ffn_reduction = True
                    communicator.allow_reduce_scatter = False
                    self.assertIsNone(communicator._declared_sides())
        # Modes given directly (Nemotron-H's stages) do not say which rows the
        # layer takes.
        direct = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        self.assertFalse(self.declared(build(direct, dp)))

    def test_subclasses_that_pick_their_own_steps(self):
        for cls in (
            MHCLayerCommunicator,
            DSACPLayerCommunicator,
            CuteDSLFusionLayerCommunicator,
        ):
            with self.subTest(cls.__name__):
                self.assertFalse(cls._takes_declared_boundaries)
        self.assertTrue(LayerCommunicator._takes_declared_boundaries)


class TestTheAttentionOutputDecidesItsSum(CustomTestCase):
    """prepare_mlp completes the attention-TP sum exactly when the attention
    output's declaration says it is owed, whatever the attention-TP size."""

    def run_steps(self, produced, *, force=False):
        sizes = {
            TokenAxis.ATTN_DP: 2,
            TokenAxis.ATTN_CP: 1,
            TokenAxis.ATTN_TP_SCATTER: 2,
        }
        sides = sides_of(sizes)
        steps, _ = comm._select_ffn_input(
            produced,
            residual=sides.input_rows,
            residual_to=sides.ffn_residual_rows,
            need=sides.ffn,
            force_layernorm_before_gather=force,
            fusions=(),
        )
        reduced = []
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=0)
        with (
            patch.object(
                comm,
                "attention_tensor_model_parallel_all_reduce",
                lambda x: reduced.append(x) or 2 * x,
            ),
            patch.object(comm, "_redistribute_input_to_dp", lambda h, fb: h),
            patch.object(
                comm, "get_parallel", lambda: parallel_of(attn_dp=2, attn_tp=2)
            ),
            patch.object(
                comm, "use_symmetric_memory", lambda g, disabled=False: nullcontext()
            ),
            patch.object(comm, "is_allocation_symmetric", lambda: False),
        ):
            hidden, residual = steps(
                torch.full((1, HIDDEN), 7.0),
                torch.full((1, HIDDEN), 3.0),
                None,
                Norm(),
                context,
            )
        return len(reduced), hidden, residual

    def test_a_complete_output_is_not_summed_again(self):
        layout = sides_of(
            {
                TokenAxis.ATTN_DP: 2,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        ).input_rows
        complete = comm.StageOutput(layout)
        for force in (False, True):
            with self.subTest(force=force):
                reductions, hidden, residual = self.run_steps(complete, force=force)
                self.assertEqual(reductions, 0)
                torch.testing.assert_close(residual, torch.full((1, HIDDEN), 10.0))
                torch.testing.assert_close(hidden, torch.full((1, HIDDEN), 20.0))

    def test_an_owed_output_is_summed_once(self):
        layout = sides_of(
            {
                TokenAxis.ATTN_DP: 2,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        ).input_rows
        owed = comm.StageOutput(layout, group=SumGroup.ATTN_TP, always_leaves=True)
        reductions, hidden, residual = self.run_steps(owed, force=True)
        self.assertEqual(reductions, 1)
        # The stand-in all-reduce doubles the one rank's value.
        torch.testing.assert_close(residual, torch.full((1, HIDDEN), 17.0))

    def test_declarations_the_steps_cannot_run_are_rejected(self):
        layout = sides_of(
            {
                TokenAxis.ATTN_DP: 2,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        ).input_rows
        for produced in (
            # Leaves the attention-TP sum only when asked, like a mixer exit.
            comm.StageOutput(layout, group=SumGroup.ATTN_TP),
            # Always leaves a sum over no group.
            comm.StageOutput(layout, always_leaves=True),
            # Owes a sum over a group the steps do not complete.
            comm.StageOutput(layout, group=SumGroup.TP, always_leaves=True),
        ):
            with (
                self.subTest(produced=produced),
                self.assertRaises(NotImplementedError),
            ):
                self.run_steps(produced)


class TestFusedKernelsTakeOnlyTheStepsTheyComplete(CustomTestCase):
    """A fused kernel replaces the attention-TP sum, the residual add and the
    norm only where those are the steps, and only if it completes the sum the
    attention output owes."""

    def fused(self, completes, may_return_new_residual):
        return comm.FusedMlpInput(
            completes=completes,
            run=lambda h, r, fb: None,
            may_return_new_residual=may_return_new_residual,
        )

    def select(self, *, attn_dp, fusions):
        sides = sides_of(
            {
                TokenAxis.ATTN_DP: attn_dp,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        )
        return comm._select_ffn_input(
            sides.attention_output,
            residual=sides.input_rows,
            residual_to=sides.ffn_residual_rows,
            need=sides.ffn,
            force_layernorm_before_gather=False,
            fusions=fusions,
        )

    def test_a_kernel_over_another_group_is_not_chosen(self):
        over_tp = self.fused(SumGroup.TP, True)
        over_attention_tp = self.fused(SumGroup.ATTN_TP, False)
        steps, chosen = self.select(attn_dp=1, fusions=(over_tp, over_attention_tp))
        self.assertIs(steps.func, comm._mlp_input_without_dp)
        self.assertEqual(steps.keywords["fusions"], (over_attention_tp.run,))
        self.assertEqual(chosen, (over_attention_tp,))

    def test_no_kernel_is_chosen_where_rows_are_gathered(self):
        steps, chosen = self.select(
            attn_dp=2, fusions=(self.fused(SumGroup.ATTN_TP, True),)
        )
        self.assertIs(steps.func, comm._mlp_input_dp_partial)
        self.assertEqual(chosen, ())

    def test_aux_capture_follows_the_chosen_kernels(self):
        # Only a chosen kernel that may hand back a new residual lets aux capture
        # keep its reference to the one prepare_mlp took.
        for kernels, expected in (
            ((self.fused(SumGroup.ATTN_TP, False),), False),
            ((self.fused(SumGroup.ATTN_TP, True),), True),
            ((self.fused(SumGroup.TP, True),), False),
        ):
            with self.subTest(kernels=kernels):

                class Declaring(LayerCommunicator):
                    def _select_mlp_input_fusions(self):
                        return kernels

                with planning(parallel_of(attn_dp=1, attn_tp=2)):
                    communicator = Declaring(
                        layer_scatter_modes=layer_facts(1, 3),
                        input_layernorm=Norm(),
                        post_attention_layernorm=Norm(),
                    )
                self.assertIs(
                    any(f.may_return_new_residual for f in communicator._steps.fused),
                    expected,
                )


class TestTheSequenceParallelRegion(CustomTestCase):
    """While a LayerNorm SP region is active, the layer runs the steps its
    region declarations choose: the linears gather and reduce-scatter
    themselves, so every boundary stays on this rank's slice. Other batches
    take the layer's ordinary declared steps."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP_SCATTER: 2}
    SP_STEPS = comm.BoundarySteps(
        attention_input=comm.CommunicateSimpleFn._trivial,
        ffn_input=comm._mlp_input_norm,
        ffn_output=sequence_parallel_layer_sides(axis_sizes=SIZES).ffn_output,
        ffn_output_move=comm.CommunicateSummableTensorPairFn._trivial,
        ffn_sum_is_movable=False,
    )

    def test_the_region_declarations_choose_local_steps(self):
        for attn_tp in (2, 4):
            with self.subTest(attn_tp=attn_tp):
                sides = sequence_parallel_layer_sides(
                    axis_sizes={
                        TokenAxis.ATTN_DP: 1,
                        TokenAxis.ATTN_CP: 1,
                        TokenAxis.ATTN_TP_SCATTER: attn_tp,
                    }
                )
                local = frozenset({TokenAxis.ATTN_TP_SCATTER})
                self.assertEqual(sides.input_rows.sharded, local)
                self.assertEqual(sides.attention.gathers_itself, local)
                self.assertEqual(sides.ffn.gathers_itself, local)
                self.assertIsNone(sides.attention_output.group)
                self.assertIsNone(sides.ffn_output.group)
                steps = comm._select_boundary_steps(sides)
                self.assertEqual(
                    (steps.attention_input, steps.ffn_input, steps.ffn_output_move),
                    (
                        self.SP_STEPS.attention_input,
                        self.SP_STEPS.ffn_input,
                        self.SP_STEPS.ffn_output_move,
                    ),
                )
                self.assertIsNone(steps.layer_input)

    def test_a_layer_under_sp_takes_both_sets_of_steps(self):
        parallel = parallel_of(attn_dp=1, attn_tp=2)
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                with planning(parallel, sp=True):
                    communicator = LayerCommunicator(
                        layer_scatter_modes=layer_facts(layer_id, 3),
                        input_layernorm=Norm(),
                        post_attention_layernorm=Norm(),
                    )
                self.assertEqual(communicator._sp_steps, self.SP_STEPS)
                # Outside the region: the attention-TP sum, then add + norm.
                self.assertIs(
                    communicator._steps.ffn_input.func, comm._mlp_input_without_dp
                )

    def test_what_the_ffn_exit_reads_comes_from_the_batch_s_steps(self):
        # Inside the region the FFN output is complete, so its sum cannot move to
        # the next layer; outside it the layer's ordinary steps say it can.
        parallel = parallel_of(attn_dp=1, attn_tp=2)
        communicator = build(layer_facts(1, 3), parallel, sp=True)
        for active in (False, True):
            with (
                self.subTest(sp_active=active),
                planning(parallel, sp=True),
                patch.object(
                    comm, "get_forward", lambda: SimpleNamespace(sp_active=active)
                ),
                patch.object(
                    comm,
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
                patch.object(comm, "is_dp_attention_enabled", lambda: False),
            ):
                self.assertIs(
                    communicator._ffn_sum_can_move_to_next_layer(None), not active
                )

    def test_without_sp_there_is_no_region(self):
        with planning(parallel_of(attn_dp=1, attn_tp=2)):
            communicator = LayerCommunicator(
                layer_scatter_modes=layer_facts(1, 3),
                input_layernorm=Norm(),
                post_attention_layernorm=Norm(),
            )
        self.assertIsNone(communicator._sp_steps)

    def test_the_same_selector_chooses_a_layer_s_ordinary_steps(self):
        # A sum the FFN exit may hand to the next layer, and a move back over
        # attention DP whose step the exit chooses per batch.
        kept = comm._select_boundary_steps(sides_of(self.SIZES, leaves_next=True))
        self.assertTrue(kept.ffn_output.leaves_for_next_layer)
        self.assertTrue(kept.ffn_sum_is_movable)
        self.assertIs(
            kept.ffn_output_move, comm.CommunicateSummableTensorPairFn._trivial
        )
        returned = comm._select_boundary_steps(
            sides_of({**self.SIZES, TokenAxis.ATTN_DP: 2})
        )
        self.assertTrue(returned.returns_over_dp)
        self.assertIsNone(returned.ffn_output_move)


class TestInputScatteredAttention(CustomTestCase):
    """On a batch whose attention input is scattered over attention TP, a layer
    runs the steps that batch's declarations choose: its input arrives as a TP
    partial that a reduce-scatter completes onto each rank's slice, and the
    residual comes back to every row inside the attention output's sum."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP_SCATTER: 2}

    def test_the_declarations_choose_the_steps(self):
        for hands_on in (False, True):
            with self.subTest(hands_on_partial=hands_on):
                sides = input_scattered_layer_sides(
                    axis_sizes=self.SIZES,
                    ffn_group=SumGroup.TP,
                    hands_on_partial=hands_on,
                )
                steps = comm._select_boundary_steps(sides)
                self.assertIs(steps.layer_input, comm.tp_reduce_scatter)
                self.assertIs(steps.attention_input, comm.CommunicateSimpleFn._trivial)
                self.assertIs(steps.ffn_input, comm._mlp_input_residual_into_sum)
                self.assertIs(
                    steps.ffn_output_move,
                    comm.CommunicateSummableTensorPairFn._trivial,
                )
                self.assertIs(steps.ffn_output.leaves_for_reduce_scatter, hands_on)
                self.assertFalse(steps.ffn_output.leaves_for_next_layer)

    def test_the_order_of_a_residual_on_the_slice_is_declared(self):
        # The same layouts take two orders: with a scattered input the residual
        # joins the attention output's sum; after a MoE on local rows it is
        # gathered first.
        sizes = self.SIZES
        attention = comm.Layout.sharded_over(axis_sizes=sizes)
        local = comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
        owed = comm.StageOutput(attention, group=SumGroup.ATTN_TP, always_leaves=True)
        for joins, step in (
            (True, comm._mlp_input_residual_into_sum),
            (False, comm._mlp_input_without_dp),
        ):
            with self.subTest(residual_joins_sum=joins):
                steps, _ = comm._select_ffn_input(
                    owed,
                    residual=local,
                    residual_to=attention,
                    need=comm.StageInput(attention),
                    force_layernorm_before_gather=False,
                    fusions=(),
                    residual_joins_sum=joins,
                )
                self.assertIs(getattr(steps, "func", steps), step)

    def test_which_layers_can_scatter_their_input(self):
        configured = dict(enable_attn_tp_input_scattered=True)
        for name, parallel, a2a, expected in (
            ("pure TP", parallel_of(attn_dp=1, attn_tp=2, **configured), False, True),
            ("not configured", parallel_of(attn_dp=1, attn_tp=2), False, False),
            (
                "attention DP",
                parallel_of(attn_dp=2, attn_tp=2, **configured),
                False,
                False,
            ),
            ("a2a", parallel_of(attn_dp=1, attn_tp=2, **configured), True, False),
        ):
            with self.subTest(name):
                communicator = build(
                    layer_facts(1, 3, sparse=a2a, previous_sparse=a2a),
                    parallel,
                    a2a=a2a,
                    allow_reduce_scatter=True,
                )
                self.assertIs(communicator._input_scattered_steps is not None, expected)

    def test_a_batch_runs_them_only_while_its_input_is_scattered(self):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        communicator = build(layer_facts(1, 3), parallel, allow_reduce_scatter=True)
        for scattered in (False, True):
            with (
                self.subTest(input_scattered=scattered),
                patch.object(
                    comm,
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=scattered),
                ),
                patch.object(
                    comm, "get_forward", lambda: SimpleNamespace(sp_active=False)
                ),
            ):
                self.assertIs(
                    communicator._batch_steps(None),
                    communicator._input_scattered_steps
                    if scattered
                    else communicator._steps,
                )

    def test_prepare_attn_completes_the_scattered_input(self):
        # The layer's input is a TP partial on every row; the reduce-scatter
        # completes it onto this rank's slice, and the residual is sliced too.
        scattered = []
        group = SimpleNamespace(
            name="tp",
            reduce_scatter_tensor=lambda out, x: (
                scattered.append(x.shape[0]) or out.copy_(x[: out.shape[0]] * 2)
            ),
        )
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True, tp_group=group
        )
        communicator = build(layer_facts(1, 3), parallel, allow_reduce_scatter=True)
        with (
            patch.object(comm, "get_parallel", lambda: parallel),
            patch.object(
                comm,
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=True),
            ),
            patch.object(comm, "get_forward", lambda: SimpleNamespace(sp_active=False)),
        ):
            hidden, residual = communicator.prepare_attn(
                torch.ones(4, HIDDEN), torch.full((4, HIDDEN), 3.0), None
            )
        self.assertEqual(scattered, [4])
        # Norm: (2 * (h + r), h + r) on the slice, with h the completed sum.
        torch.testing.assert_close(residual, torch.full((2, HIDDEN), 5.0))
        torch.testing.assert_close(hidden, torch.full((2, HIDDEN), 10.0))

    def test_the_last_layer_completes_its_own_sum(self):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        for layer_id, allow, hands_on in (
            (1, True, True),
            (2, True, False),
            (1, False, False),
        ):
            with self.subTest(layer_id=layer_id, allow_reduce_scatter=allow):
                communicator = build(
                    layer_facts(layer_id, 3), parallel, allow_reduce_scatter=allow
                )
                steps = communicator._input_scattered_steps
                self.assertIs(steps.ffn_output.leaves_for_reduce_scatter, hands_on)


class TestOneRepresentation(CustomTestCase):
    """Every layer runs from one BoundarySteps per batch, whichever entry chose
    its steps: none keeps the steps as separate attributes."""

    SEPARATE = (
        "_communicate_simple_fn",
        "_mlp_input",
        "_communicate_summable_tensor_pair_fn",
        "_ffn_output",
        "_postprocess_scatters_to_local_tokens",
        "_ffn_sum_is_movable",
        "_mlp_input_may_return_new_residual",
    )

    def test_both_entries_hold_steps_only(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        direct = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        for name, communicator in (
            ("declarations", build(layer_facts(1, 3), parallel)),
            ("scatter modes", build(direct, parallel)),
        ):
            with self.subTest(name):
                self.assertIsInstance(communicator._steps, comm.BoundarySteps)
                for attribute in self.SEPARATE:
                    self.assertFalse(hasattr(communicator, attribute), attribute)
                self.assertIs(communicator._batch_steps(None), communicator._steps)
                # Under attention DP both bring the FFN output back to this
                # rank's tokens with the step the FFN exit chooses.
                self.assertTrue(communicator._steps.returns_over_dp)

    def test_the_scatter_mode_steps_complete_a_scattered_input(self):
        # The scatter-mode path's input completion reads the batch fact itself.
        steps = build(
            LayerScatterModes(
                layer_input_mode=ScatterMode.TP_ATTN_FULL,
                attn_mode=ScatterMode.TP_ATTN_FULL,
                mlp_mode=ScatterMode.FULL,
                middle_residual_mode=ScatterMode.TP_ATTN_FULL,
                layer_output_mode=ScatterMode.TP_ATTN_FULL,
            ),
            parallel_of(attn_dp=1, attn_tp=2),
        )._steps
        self.assertIs(steps.layer_input, comm._complete_scattered_input)
        hidden, residual = torch.ones(2, HIDDEN), torch.ones(2, HIDDEN)
        with patch.object(
            comm, "get_attn_tp_context", lambda: SimpleNamespace(input_scattered=False)
        ):
            self.assertEqual(
                steps.layer_input(hidden, residual, None), (hidden, residual)
            )


class TestPrefillCP(CustomTestCase):
    """A prefill CP shards a batch's tokens over attention CP only on a CP
    extend; the layer runs its CP steps on those batches and its ordinary ones
    on every other."""

    def cp_parallel(self, **overrides):
        return parallel_of(
            attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=True, **overrides
        )

    def test_a_cp_extend_gathers_over_cp_and_takes_its_chunk_back(self):
        communicator = build(layer_facts(1, 3), self.cp_parallel())
        cp = communicator._cp_steps
        self.assertIs(cp.ffn_input.func, comm._mlp_input_gather_moe_cp)
        # Each rank completes its own chunk first: the attention-TP sum, then
        # add + norm.
        self.assertIs(cp.ffn_input.keywords["gather"].func, comm._mlp_input_without_dp)
        self.assertIs(
            cp.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
        )
        self.assertFalse(cp.returns_over_dp)
        # The sum over the gathered rows stays with the layer.
        self.assertFalse(cp.ffn_output.leaves_for_next_layer)
        self.assertFalse(cp.ffn_output.leaves_for_reduce_scatter)
        # Other batches hold every token on each CP rank: plain TP steps.
        self.assertIs(communicator._steps.ffn_input.func, comm._mlp_input_without_dp)
        self.assertIs(
            communicator._steps.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._trivial,
        )

    def test_the_fused_kernels_run_on_each_chunk(self):
        with planning(self.cp_parallel()):
            communicator = LayerCommunicator(
                layer_scatter_modes=layer_facts(1, 3),
                input_layernorm=FusableNorm(),
                post_attention_layernorm=FusableNorm(),
            )
        chunk = communicator._cp_steps.ffn_input.keywords["gather"]
        self.assertEqual(
            chunk.keywords["fusions"],
            (communicator._mlp_input_reduce_output_and_update_and_read_residual,),
        )

    def test_which_cp_the_declarations_cover(self):
        for name, parallel, declared in (
            ("prefill CP", self.cp_parallel(), True),
            (
                "CP without prefill CP",
                parallel_of(attn_dp=1, attn_tp=2, attn_cp=2),
                False,
            ),
            (
                "CP under attention DP",
                parallel_of(attn_dp=2, attn_tp=1, attn_cp=2, enable_prefill_cp=True),
                False,
            ),
            (
                "a MoE-CP group narrower than CP",
                parallel_of(
                    attn_dp=1,
                    attn_tp=1,
                    attn_cp=4,
                    enable_prefill_cp=True,
                    moe_dp_size=2,
                ),
                False,
            ),
        ):
            with self.subTest(name):
                modes = planned_modes(
                    1, 3, sparse=False, previous_sparse=False, parallel=parallel
                )
                communicator = build(modes, parallel)
                self.assertIs(communicator._cp_steps is not None, declared)

    def test_a_batch_runs_them_only_on_a_cp_extend(self):
        communicator = build(layer_facts(1, 3), self.cp_parallel())
        for rows, expected in (
            (None, communicator._steps),
            ([2, 1], communicator._cp_steps),
        ):
            with (
                self.subTest(gathered_rows=rows),
                patch.object(comm, "moe_cp_gathered_rows", lambda fb: rows),
                patch.object(
                    comm, "get_forward", lambda: SimpleNamespace(sp_active=False)
                ),
                patch.object(
                    comm,
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
            ):
                self.assertIs(communicator._batch_steps(None), expected)


# ---------------------------------------------------------------------------
# Two consecutive layers, rank by rank.


class World:
    def __init__(self, size):
        self.size = size
        self.local = threading.local()
        self.groups = []

    def state(self):
        return self.local.state

    def run(self, states, fn):
        results, errors = [None] * self.size, [None] * self.size

        def body(rank):
            self.local.state = states[rank]
            try:
                results[rank] = fn(rank)
            except BaseException as error:  # noqa: BLE001
                errors[rank] = error
                for group in self.groups:
                    group.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results, errors


class Group:
    """A process group whose collectives wait for every member thread."""

    def __init__(self, world, name, ranks):
        self.world, self.name, self.ranks = world, name, list(ranks)
        self.barrier = threading.Barrier(len(self.ranks), timeout=30)
        self.slots = {}
        world.groups.append(self)

    @property
    def world_size(self):
        return len(self.ranks)

    @property
    def rank_in_group(self):
        return self.ranks.index(self.world.state().rank)

    def _exchange(self, x):
        self.world.state().calls.append(self.name)
        self.slots[self.world.state().rank] = x.clone()
        self.barrier.wait()
        values = [self.slots[r] for r in self.ranks]
        self.barrier.wait()
        return values

    def all_reduce(self, x):
        return torch.stack(self._exchange(x)).sum(0)

    def reduce_scatter_tensor(self, output, input):
        full = torch.stack(self._exchange(input)).sum(0)
        output.copy_(full.tensor_split(self.world_size)[self.rank_in_group])

    def all_gather_into_tensor(self, output, input):
        output.copy_(torch.cat(self._exchange(input)))

    def reduce_scatterv(self, input, output=None, sizes=None):
        full = torch.stack(self._exchange(input)).sum(0)
        output.copy_(full.split(list(sizes))[self.rank_in_group])
        return output


class Flags:
    """The per-forward flags an FFN exit publishes, one set per rank."""

    def __init__(self):
        self.fuse_mlp_allreduce = False
        self.mlp_reduce_scatter = False
        self.defer_moe_finalize = False
        self.sp_active = False

    @contextmanager
    def scoped(self, **flags):
        saved = {k: getattr(self, k) for k in flags}
        self.__dict__.update(flags)
        try:
            yield
        finally:
            self.__dict__.update(saved)


def fake_dp_gather(global_tokens, local_tokens, forward_batch, *, is_partial):
    state = WORLD[0].state()
    global_tokens.zero_()
    if is_partial or state.parallel.attn_tp_rank == 0:
        global_tokens[state.offset : state.offset + state.rows] += local_tokens[
            : state.rows
        ]
    global_tokens.copy_(state.parallel.tp_group.all_reduce(global_tokens))


def fake_dp_scatter(local_tokens, global_tokens, forward_batch):
    state = WORLD[0].state()
    local_tokens.fill_(0)
    local_tokens[: state.rows] = global_tokens[state.offset : state.offset + state.rows]


def fake_dp_reduce_scatter_tensor(output, input):
    # dp_attention.dp_reduce_scatter_tensor for max-len padding.
    parallel = WORLD[0].state().parallel
    if parallel.tp_size == parallel.attn_dp_size:
        parallel.tp_group.reduce_scatter_tensor(output, input)
    else:
        chunk = input.tensor_split(parallel.tp_size)[parallel.tp_rank].clone()
        parallel.tp_group.reduce_scatter_tensor(chunk, input)
        parallel.attn_tp_group.all_gather_into_tensor(output, chunk)


WORLD = [None]


def state():
    return WORLD[0].state()


@contextmanager
def running(*, reduce_scatterv, a2a=False):
    replaced = {
        "get_forward": lambda: state().flags,
        "attention_tensor_model_parallel_all_reduce": lambda x: (
            state().parallel.attn_tp_group.all_reduce(x)
        ),
        "get_global_dp_buffer": lambda g: torch.zeros(
            state().global_rows, HIDDEN, dtype=torch.double
        ),
        "get_local_dp_buffer": lambda g: torch.zeros(
            state().local_rows, HIDDEN, dtype=torch.double
        ),
        "dp_gather_replicate": partial(fake_dp_gather, is_partial=False),
        "dp_gather_partial": partial(fake_dp_gather, is_partial=True),
        "dp_scatter": fake_dp_scatter,
        "dp_reduce_scatter_tensor": fake_dp_reduce_scatter_tensor,
        "attn_tp_reduce_scatter_tensor": lambda out, x: (
            state().parallel.attn_tp_group.reduce_scatter_tensor(out, x)
        ),
        "attn_tp_all_gather_into_tensor": lambda out, x: (
            state().parallel.attn_tp_group.all_gather_into_tensor(out, x)
        ),
        "get_dp_global_num_tokens": lambda: state().dp_rows,
        "should_use_dp_reduce_scatterv": lambda: reduce_scatterv,
        "can_use_dp_reduce_scatter": lambda: True,
        "is_dp_attention_enabled": lambda: state().parallel.attn_dp_size > 1,
        "apply_flashinfer_allreduce_fusion": lambda n: False,
        "post_experts_output_is_complete": lambda **kw: False,
        "post_experts_reduction_group": lambda: state().parallel.tp_group,
        "get_lora": lambda: SimpleNamespace(enable_lora=False),
        "get_exec": lambda: SimpleNamespace(
            comm=SimpleNamespace(enable_quant_communications=False)
        ),
        "get_attn_tp_context": lambda: SimpleNamespace(
            input_scattered=False, set_attn_inputs=lambda x: None
        ),
        "use_symmetric_memory": lambda group, disabled=False: nullcontext(),
        "is_allocation_symmetric": lambda: False,
    }
    with ExitStack() as stack:
        stack.enter_context(planning(lambda: state().parallel, a2a=a2a))
        for name, value in replaced.items():
            stack.enter_context(patch.object(comm, name, value))
        yield


# Binary fractions, so partial sums add back to the value exactly.
WEIGHTS = {1: [1.0], 2: [0.25, 0.75], 4: [0.125, 0.25, 0.375, 0.25]}


def attention(x, state):
    """The attention output: 3 * x, as this rank's attention-TP partial."""
    return 3 * x * WEIGHTS[state.parallel.attn_tp_size][state.parallel.attn_tp_rank]


def dense_mlp(x, state):
    """A dense MLP on the TP group: 5 * x, reduced unless a published flag asks
    it to leave the sum, as RowParallelLinear does."""
    partial_sum = 5 * x * WEIGHTS[state.parallel.tp_size][state.parallel.tp_rank]
    if state.flags.fuse_mlp_allreduce or state.flags.mlp_reduce_scatter:
        return partial_sum
    return state.parallel.tp_group.all_reduce(partial_sum)


def moe(x, state):
    """A MoE block not dispatched per DP shard: 5 * x, reduced over the MoE
    output's group unless a published flag asks it to leave the sum, or the
    attention-DP reduce_scatterv takes it, as the MoE blocks do."""
    partial_sum = 5 * x * WEIGHTS[state.parallel.tp_size][state.parallel.tp_rank]
    if (
        state.flags.fuse_mlp_allreduce
        or state.flags.mlp_reduce_scatter
        or comm.should_use_dp_reduce_scatterv()
    ):
        return partial_sum
    return comm.post_experts_reduction_group().all_reduce(partial_sum)


def reference(x):
    """(hidden, residual) after two layers of Norm / attention / FFN."""
    residual = x
    for _ in range(2):
        hidden = 2 * residual
        residual = 3 * hidden + residual
        ffn = 5 * (2 * residual)
        residual = ffn + residual
    # The last layer hands on its FFN output and the residual before it.
    return ffn, residual - ffn


class TestTwoLayers(CustomTestCase):
    def run_world(
        self,
        *,
        attn_dp,
        attn_tp,
        rows,
        padding,
        reduce_scatterv,
        leaves,
        sparse=(False, False),
        a2a=False,
    ):
        tp = attn_dp * attn_tp
        world = World(tp)
        WORLD[0] = world
        tp_group = Group(world, "tp", range(tp))
        attn_tp_groups = [
            tp_group
            if attn_tp == tp
            else Group(world, f"attn_tp{d}", range(d * attn_tp, (d + 1) * attn_tp))
            for d in range(attn_dp)
        ]
        max_len = padding == "max_len"
        # Max-len padding pads every DP rank to one length that attention TP tiles.
        local_len = -(-max(rows) // attn_tp) * attn_tp if max_len else None
        offsets = (
            [d * local_len for d in range(attn_dp)]
            if max_len
            else [sum(rows[:d]) for d in range(attn_dp)]
        )
        global_rows = attn_dp * local_len if max_len else sum(rows)
        generator = torch.Generator().manual_seed(0)
        embeddings = [
            torch.randint(-4, 4, (n, HIDDEN), generator=generator).double()
            for n in rows
        ]
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: False,
                is_decode_or_idle=lambda: True,
            ),
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: max_len),
            global_dp_buffer_len=global_rows,
            # Without attention DP the FFN exit counts this rank's tokens.
            input_ids=torch.zeros(rows[0]),
        )
        states = []
        for rank in range(tp):
            d, t = divmod(rank, attn_tp)
            states.append(
                SimpleNamespace(
                    rank=rank,
                    parallel=parallel_of(
                        attn_dp=attn_dp,
                        attn_tp=attn_tp,
                        tp_rank=rank,
                        attn_tp_rank=t,
                        tp_group=tp_group,
                        attn_tp_group=attn_tp_groups[d],
                    ),
                    flags=Flags(),
                    calls=[],
                    dp=d,
                    rows=rows[d],
                    offset=offsets[d],
                    local_rows=local_len or rows[d],
                    global_rows=global_rows,
                    dp_rows=[local_len] * attn_dp if max_len else list(rows),
                )
            )

        def forward(rank):
            s = states[rank]
            layers = [
                LayerCommunicator(
                    layer_scatter_modes=layer_facts(
                        i,
                        2,
                        sparse=sparse[i],
                        previous_sparse=i > 0 and sparse[i - 1],
                    ),
                    input_layernorm=Norm(),
                    post_attention_layernorm=Norm(),
                    allow_reduce_scatter=leaves,
                    allow_deferred_ffn_reduction=leaves,
                )
                for i in range(2)
            ]
            hidden = torch.zeros(s.local_rows, HIDDEN).double()
            hidden[: s.rows] = embeddings[s.dp]
            residual = None
            handed_on = []
            for layer_index, layer in enumerate(layers):
                hidden, residual = layer.prepare_attn(hidden, residual, forward_batch)
                hidden = attention(hidden, s)
                hidden, residual = layer.prepare_mlp(hidden, residual, forward_batch)
                with layer.ffn_exit(forward_batch) as ffn_exit:
                    if not sparse[layer_index]:
                        hidden = dense_mlp(hidden, s)
                    elif a2a:
                        # On this rank's slice; the combine completes the sum.
                        hidden = 5 * hidden
                    else:
                        hidden = moe(hidden, s)
                hidden, residual = ffn_exit.finish(hidden, residual)
                handed_on.append(type(hidden))
            hidden, residual = layers[-1].finish_layer_stack(
                hidden, residual, forward_batch
            )
            # The last a2a layer folds the residual into its output.
            if residual is not None:
                residual = residual[: s.rows]
            return hidden[: s.rows], residual, handed_on

        with running(reduce_scatterv=reduce_scatterv, a2a=a2a):
            results, errors = world.run(states, forward)
        for rank, error in enumerate(errors):
            if error is not None:
                raise AssertionError(f"rank {rank} raised") from error
        for rank, (hidden, residual, handed_on) in enumerate(results):
            want_hidden, want_residual = reference(embeddings[states[rank].dp])
            if residual is None:
                want_hidden, want_residual = want_hidden + want_residual, None
            torch.testing.assert_close(hidden, want_hidden, rtol=0, atol=0)
            self.assertIs(residual is None, want_residual is None)
            if residual is not None:
                torch.testing.assert_close(residual, want_residual, rtol=0, atol=0)
        # Every rank ran the same collectives.
        self.assertEqual(len({tuple(s.calls) for s in states if s.dp == 0}), 1)
        return results

    def test_rows_come_back_to_each_rank(self):
        for (attn_dp, attn_tp), rows, padding, leaves, sparse in itertools.product(
            ((2, 2), (4, 1), (2, 1), (1, 2), (1, 4)),
            ("ragged", "idle"),
            ("sum_len", "max_len"),
            (False, True),
            ((False, False), (False, True), (True, True)),
        ):
            token_rows = {
                ("ragged", 1): [3],
                ("idle", 1): [0],
                ("ragged", 2): [3, 1],
                ("idle", 2): [2, 0],
                ("ragged", 4): [2, 3, 1, 1],
                ("idle", 4): [2, 0, 3, 1],
            }[rows, attn_dp]
            for reduce_scatterv, a2a in itertools.product(
                (False, True) if attn_tp == 1 and attn_dp > 1 else (False,),
                (False, True) if any(sparse) else (False,),
            ):
                if reduce_scatterv and a2a:
                    continue
                with self.subTest(
                    attn_dp=attn_dp,
                    attn_tp=attn_tp,
                    rows=token_rows,
                    padding=padding,
                    leaves=leaves,
                    reduce_scatterv=reduce_scatterv,
                    sparse=sparse,
                    a2a=a2a,
                ):
                    results = self.run_world(
                        attn_dp=attn_dp,
                        attn_tp=attn_tp,
                        rows=token_rows,
                        padding=padding,
                        reduce_scatterv=reduce_scatterv,
                        leaves=leaves,
                        sparse=sparse,
                        a2a=a2a,
                    )
                    first_layer_hands_on = results[0][2][0]
                    # A producer that may leave its sum leaves it for the next
                    # layer's input; one that may not hands on a complete value.
                    # Without attention DP an empty batch completes its own sum.
                    # A MoE dispatched per DP shard hands on a complete value.
                    # Otherwise the reduce-scatter back to this rank's tokens
                    # can be left to the next layer; the all-reduce only
                    # without an a2a backend (the exit's existing constraint).
                    has_tokens = attn_dp > 1 or sum(token_rows) > 0
                    reduce_scatter_left = attn_dp > 1 and (
                        padding == "max_len" or reduce_scatterv
                    )
                    owes = (
                        leaves
                        and has_tokens
                        and not (sparse[0] and a2a)
                        and (reduce_scatter_left or not a2a)
                    )
                    self.assertIs(
                        first_layer_hands_on,
                        UnreducedOutput if owes else torch.Tensor,
                    )


if __name__ == "__main__":
    unittest.main()
