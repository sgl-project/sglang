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
    with (
        patch.object(comm, "get_parallel", get_parallel),
        patch.object(comm, "is_dsa_enable_prefill_cp", lambda: False),
        patch.object(comm, "is_mla_cp_enabled", lambda: False),
        patch.object(comm, "get_moe_cp_size", lambda: 1),
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
        patch.object(comm, "is_enable_moe_cp_allgather", lambda: False),
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
        return getattr(communicator._mlp_input, "func", None) in (
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
                    communicator._communicate_simple_fn,
                    comm.CommunicateSimpleFn._trivial,
                )
                self.assertTrue(communicator._postprocess_scatters_to_local_tokens)

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
                self.assertIs(communicator._mlp_input.func, comm._mlp_input_without_dp)
                entry = (
                    communicator._mlp_input_reduce_output_and_update_and_read_residual
                )
                self.assertEqual(
                    communicator._mlp_input.keywords["fusions"],
                    (entry,) if fuses else (),
                )
                self.assertIs(communicator._mlp_input_may_return_new_residual, fuses)
                self.assertFalse(communicator._postprocess_scatters_to_local_tokens)
                self.assertIs(
                    communicator._communicate_summable_tensor_pair_fn,
                    comm.CommunicateSummableTensorPairFn._trivial,
                )
        # One rank: the attention output is complete, only the norm is left.
        one_rank = parallel_of(attn_dp=1, attn_tp=1)
        single = build(layer_facts(1, 3), one_rank)
        with planning(one_rank):
            self.assertIsNotNone(single._declared_sides())
        self.assertIs(single._mlp_input, comm._mlp_input_norm)

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
                self.assertIs(communicator._mlp_input.func, order)

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
                    moe = layers[1]._ffn_output
                    self.assertIs(moe.group, SumGroup.MOE_OUTPUT)
                    self.assertTrue(moe.leaves_for_reduce_scatterv)
                    continue
                # The first a2a layer slices the residual it takes from a
                # dense layer; the next one takes it already sliced.
                self.assertEqual(
                    [layers[i]._mlp_input.func for i in (1, 2)],
                    [comm._mlp_input_scatter] * 2,
                )
                self.assertEqual(
                    [
                        layers[i]._mlp_input.keywords["scatters_residual"]
                        for i in (1, 2)
                    ],
                    [True, False],
                )
                self.assertIsNone(layers[1]._ffn_output.group)
                # Their rows are gathered back for attention, and a dense layer
                # after them gathers the residual back as well.
                self.assertIs(
                    layers[2]._communicate_simple_fn,
                    comm.CommunicateSimpleFn._scattered_to_tp_attn_full,
                )
                self.assertIs(layers[3]._mlp_input.func, comm._mlp_input_dp_partial)
                self.assertTrue(layers[3]._mlp_input.keywords["gathers_residual"])

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
            last._communicate_summable_tensor_pair_fn,
            comm.CommunicateSummableTensorPairFn._gather,
        )

    def test_an_attention_that_gathers_its_slice_itself(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        modes = planned_modes(
            2, 4, sparse=True, previous_sparse=True, parallel=parallel, a2a=True
        )
        with patch.object(comm, "_use_ag_after_qlora", True):
            layer = build(modes, parallel, a2a=True)
        self.assertIs(layer._communicate_simple_fn, comm.CommunicateSimpleFn._trivial)

    def test_layers_that_keep_the_scatter_modes(self):
        dp = parallel_of(attn_dp=2, attn_tp=2)
        cases = {
            "input-scattered attention": (
                layer_facts(1, 3),
                parallel_of(attn_dp=1, attn_tp=4, enable_attn_tp_input_scattered=True),
            ),
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
        # Modes given directly (Nemotron-H's stages, the SP sibling) do not say
        # which rows the layer takes.
        direct = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        self.assertFalse(self.declared(build(direct, dp)))
        with planning(dp, sp=True):
            communicator = LayerCommunicator.__new__(LayerCommunicator)
            communicator.layer_scatter_modes = layer_facts(1, 3)
            self.assertIsNone(communicator._declared_sides())

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
                self.assertIs(communicator._mlp_input_may_return_new_residual, expected)


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
