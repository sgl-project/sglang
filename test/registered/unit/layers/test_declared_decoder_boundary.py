"""The boundaries of an attention and an FFN, chosen from both sides'
declarations. The FFN runs on the TP group (a dense MLP, or a MoE not
dispatched per DP shard) or on each attention-TP rank's slice of its DP shard's
rows (a MoE dispatched per DP shard).

The numeric checks run every rank of a DP x attention-TP world as a thread over
fake collectives that wait for all members of their group, build the layers'
communicators for real, and pass them layer facts that fail on any read of
something else.
"""

import itertools
import threading
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerFacts,
    MHCLayerCommunicator,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
)
from sglang.srt.layers.communicator import boundary as comm_boundary
from sglang.srt.layers.communicator import (
    decoder_layer_sides,
    input_scattered_layer_sides,
)
from sglang.srt.layers.communicator import layer as comm_layer
from sglang.srt.layers.communicator import layout as comm_layout
from sglang.srt.layers.communicator import ops as comm_ops
from sglang.srt.layers.communicator import (
    sequence_parallel_layer_sides,
)
from sglang.srt.layers.communicator.residual import mhc as mhc_module
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import LoRABatchLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import COMMUNICATOR_MODULES, patch_communicator
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
        tp_group=SimpleNamespace(
            name="tp", ranks=list(range(attn_dp * attn_cp * attn_tp))
        ),
        attn_tp_group=SimpleNamespace(name="attn_tp", ranks=list(range(attn_tp))),
        attn_cp_group=SimpleNamespace(name="attn_cp", ranks=list(range(attn_cp))),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@contextmanager
def planning(parallel, *, sp=False, a2a=False, dsa_cp=False):
    """What layer planning and communicator construction read, without the
    process-wide parallel state. ``parallel`` may be a callable, for a
    per-thread parallel state. ``dsa_cp``: the prefill CP is DSA's (MLA's is
    the same to the communicator)."""
    get_parallel = parallel if callable(parallel) else (lambda: parallel)

    def moe_cp_gathers():
        return get_parallel().attn_cp_size > get_parallel().moe_dp_size

    with (
        patch_communicator("get_parallel", get_parallel),
        patch_communicator("is_dsa_enable_prefill_cp", lambda: dsa_cp),
        patch_communicator("is_mla_cp_enabled", lambda: False),
        patch_communicator(
            "get_moe_cp_size",
            lambda: get_parallel().attn_cp_size if moe_cp_gathers() else 1,
        ),
        patch.object(layernorm_sp, "layernorm_sp_enabled", lambda: sp),
        patch_communicator(
            "get_spec", lambda: SimpleNamespace(speculative_algorithm=None)
        ),
        patch_communicator(
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(is_none=lambda: not a2a),
        ),
        patch_communicator("is_moe_input_scattered_across_dp_ranks", lambda: a2a),
        patch_communicator("is_enable_moe_cp_allgather", moe_cp_gathers),
        patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=False)),
        # A MoE whose EP and TP sums merge: its output's group is the TP group.
        patch_communicator(
            "post_experts_reduction_group", lambda: get_parallel().tp_group
        ),
        # Planning asks whether a dense layer gathers for two-batch overlap.
        patch_communicator(
            "get_exec",
            lambda: SimpleNamespace(
                overlap=SimpleNamespace(enable_two_batch_overlap=False)
            ),
        ),
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
    """Layer facts that fail on any read of something else: construction reads
    only the layer facts."""

    def __init__(self, **facts):
        self.__dict__.update(facts)

    def __getattr__(self, name):
        raise AssertionError(f"read {name!r}, which is not a layer fact")


def layer_facts(
    layer_id, num_layers, *, sparse=False, previous_sparse=False, next_sparse=False
):
    return FactsOnly(
        is_layer_sparse=sparse,
        is_first_layer=layer_id == 0,
        is_last_layer=layer_id == num_layers - 1,
        is_previous_layer_sparse=None if layer_id == 0 else previous_sparse,
        is_next_layer_sparse=next_sparse,
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


def build(
    facts,
    parallel,
    *,
    cls=LayerCommunicator,
    sp=False,
    a2a=False,
    dsa_cp=False,
    **kwargs,
):
    with planning(parallel, sp=sp, a2a=a2a, dsa_cp=dsa_cp):
        return cls(
            layer_facts=facts,
            input_layernorm=Norm(),
            post_attention_layernorm=Norm(),
            **kwargs,
        )


def planned_facts(
    layer_id,
    num_layers,
    *,
    sparse,
    previous_sparse,
    parallel,
    a2a=False,
    dsa_cp=False,
):
    with planning(parallel, a2a=a2a, dsa_cp=dsa_cp):
        return LayerFacts.init_new(
            layer_id=layer_id,
            num_layers=num_layers,
            is_layer_sparse=sparse,
            is_previous_layer_sparse=previous_sparse,
            is_next_layer_sparse=False,
        )


class TestWhichLayersUseDeclarations(CustomTestCase):
    """The layer is the unit: all three boundary sides it owns follow the
    declarations, or none do."""

    def declared(self, communicator):
        return getattr(
            communicator._steps.ffn.prepare.keywords["step"], "func", None
        ) in (
            comm_ops._mlp_input_dp_partial,
            comm_ops._mlp_input_dp_replicate,
            comm_ops._mlp_input_without_dp,
            comm_ops._mlp_input_scatter,
        )

    def test_a_dense_model(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                communicator = build(layer_facts(layer_id, 3), parallel)
                self.assertTrue(self.declared(communicator))
                self.assertIsNone(communicator._steps.attention.input_move)
                self.assertTrue(communicator._steps.returns_over_dp)

    def test_plain_tp(self):
        for norm, fuses in ((Norm, False), (FusableNorm, True)):
            with self.subTest(norm=norm.__name__):
                with planning(parallel_of(attn_dp=1, attn_tp=2)):
                    communicator = LayerCommunicator(
                        layer_facts=layer_facts(1, 3),
                        input_layernorm=norm(),
                        post_attention_layernorm=norm(),
                    )
                self.assertIs(
                    communicator._steps.ffn.prepare.keywords["step"].func,
                    comm_ops._mlp_input_without_dp,
                )
                entry = (
                    communicator._mlp_input_reduce_output_and_update_and_read_residual
                )
                self.assertEqual(
                    communicator._steps.ffn.prepare.keywords["step"].keywords[
                        "fusions"
                    ],
                    (entry,) if fuses else (),
                )
                self.assertIs(
                    any(
                        f.may_return_new_residual for f in communicator._steps.ffn.fused
                    ),
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
        self.assertIs(
            single._steps.ffn.prepare.keywords["step"].func, comm_ops._read_input
        )

    def test_the_order_follows_the_attention_output(self):
        for attn_tp, force, order in (
            (2, False, comm_ops._mlp_input_dp_partial),
            (2, True, comm_ops._mlp_input_dp_replicate),
            (1, False, comm_ops._mlp_input_dp_replicate),
        ):
            with self.subTest(attn_tp=attn_tp, force=force):
                communicator = build(
                    layer_facts(1, 3),
                    parallel_of(attn_dp=2, attn_tp=attn_tp),
                    force_layernorm_before_dp_gather=force,
                )
                self.assertIs(
                    communicator._steps.ffn.prepare.keywords["step"].func, order
                )

    def test_a_dense_first_layer_before_sparse_ones(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for a2a in (False, True):
            with self.subTest(a2a=a2a):
                layers = [
                    build(
                        planned_facts(
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
                    [
                        layers[i]._steps.ffn.prepare.keywords["step"].func
                        for i in (1, 2)
                    ],
                    [comm_ops._mlp_input_scatter] * 2,
                )
                self.assertEqual(
                    [
                        layers[i]
                        ._steps.ffn.prepare.keywords["step"]
                        .keywords["scatters_residual"]
                        for i in (1, 2)
                    ],
                    [True, False],
                )
                self.assertIsNone(layers[1]._steps.ffn_output.group)
                self.assertIs(
                    layers[2]._steps.attention.input_move,
                    comm.CommunicateSimpleFn._scattered_to_tp_attn_full,
                )
                self.assertIs(
                    layers[3]._steps.ffn.prepare.keywords["step"].func,
                    comm_ops._mlp_input_dp_partial,
                )
                self.assertTrue(
                    layers[3]
                    ._steps.ffn.prepare.keywords["step"]
                    .keywords["gathers_residual"]
                )

    def test_a_dense_mlp_on_every_rank(self):
        # moe_dense_tp_size 1: each rank runs the dense MLP on its own slice, as
        # an a2a MoE does, and owes no sum.
        parallel = parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)
        no_overlap = SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=False)
        )
        for a2a in (False, True):
            with (
                self.subTest(a2a=a2a),
                patch_communicator("get_exec", lambda: no_overlap),
            ):
                layers = [
                    build(
                        planned_facts(
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
                        ((False, False), (False, False), (True, False), (True, True))
                    )
                ]
                self.assertEqual([self.declared(layer) for layer in layers], [True] * 4)
                for dense in layers[:2]:
                    self.assertIs(
                        dense._steps.ffn.prepare.keywords["step"].func,
                        comm_ops._mlp_input_scatter,
                    )
                    self.assertIsNone(dense._steps.ffn_output.group)
                for after_dense in layers[1:3]:
                    self.assertIs(
                        after_dense._steps.attention.input_move,
                        comm.CommunicateSimpleFn._scattered_to_tp_attn_full,
                    )

    def test_the_last_a2a_layer_folds_the_residual_back(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        last = build(
            planned_facts(
                3, 4, sparse=True, previous_sparse=True, parallel=parallel, a2a=True
            ),
            parallel,
            a2a=True,
        )
        self.assertIs(
            last._steps.ffn_output_move.func,
            comm.CommunicateSummableTensorPairFn._gather,
        )

    def test_an_attention_that_gathers_its_slice_itself(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        facts = planned_facts(
            2, 4, sparse=True, previous_sparse=True, parallel=parallel, a2a=True
        )
        with patch_communicator("_use_ag_after_qlora", True):
            layer = build(facts, parallel, a2a=True)
        self.assertIsNone(layer._steps.attention.input_move)

    def test_cp_without_prefill_cp(self):
        """No batch shards its tokens, so no CP steps. Under attention DP the
        reduce-scatter back runs over the TP group, which spans the CP ranks,
        so the FFN completes its own sum; without attention DP it may leave it
        and input-scattered attention runs, as without CP."""
        under_dp = build(
            layer_facts(1, 3), parallel_of(attn_dp=2, attn_tp=1, attn_cp=2)
        )
        self.assertIsNone(under_dp._cp_steps)
        self.assertFalse(under_dp._steps.ffn_output.leaves_for_next_layer)
        self.assertFalse(under_dp._steps.ffn_output.leaves_for_reduce_scatterv)
        without_dp = build(
            layer_facts(1, 3),
            parallel_of(
                attn_dp=1, attn_tp=2, attn_cp=2, enable_attn_tp_input_scattered=True
            ),
        )
        self.assertTrue(without_dp._steps.ffn_output.leaves_for_next_layer)
        self.assertIsNotNone(without_dp._input_scattered_steps)

    def test_a_layer_without_the_previous_layer_s_facts_is_refused(self):
        direct = LayerFacts()
        with self.assertRaises(NotImplementedError):
            build(direct, parallel_of(attn_dp=2, attn_tp=2))


def build_mhc(
    facts, parallel, *, a2a=False, dsa_cp=False, two_batch_overlap=False, **kwargs
):
    overlap = SimpleNamespace(
        overlap=SimpleNamespace(enable_two_batch_overlap=two_batch_overlap)
    )
    with (
        planning(parallel, a2a=a2a, dsa_cp=dsa_cp),
        patch_communicator("get_exec", lambda: overlap),
    ):
        return MHCLayerCommunicator(
            layer_facts=facts,
            input_layernorm=Norm(),
            post_attention_layernorm=Norm(),
            hc_mult=2,
            hc_attn_pre=lambda *a: None,
            hc_ffn_pre=lambda *a: None,
            **{"hc_post": lambda *a: None, **kwargs},
        )


class TestMhcOnTheDeclarations(CustomTestCase):
    """An MHC layer takes the same declarations and runs the same steps as any
    other layer; only the residual operations the steps run are MHC's."""

    def assert_step(self, step, func, communicator, **keywords):
        if step.func is comm_ops._consumer_step:
            step = step.keywords["step"]
        self.assertIs(step.func, func)
        residual = communicator._residual
        if "read" in step.keywords:
            # The FFN input: the attention output's hc_post, then the FFN's read.
            self.assertIs(step.keywords["read"], residual.ffn_read)
            self.assertIs(step.keywords["update"], residual.attention_update)
        else:
            # The FFN output's move: its hc_post.
            self.assertIs(step.keywords["update"], residual.ffn_update)
        for key, value in keywords.items():
            self.assertEqual(step.keywords[key], value, key)

    def test_each_boundary_runs_the_shared_step(self):
        for name, parallel, ffn_input, output_move in (
            (
                "attention TP 1",
                parallel_of(attn_dp=1, attn_tp=1),
                comm_ops._read_input,
                comm.CommunicateSummableTensorPairFn._trivial,
            ),
            (
                "the attention-TP sum",
                parallel_of(attn_dp=1, attn_tp=2),
                comm_ops._mlp_input_without_dp,
                comm.CommunicateSummableTensorPairFn._trivial,
            ),
            # The DP gather runs after hc_post, which is not a plain add.
            (
                "a DP gather",
                parallel_of(attn_dp=2, attn_tp=2),
                comm_ops._mlp_input_dp_replicate,
                None,
            ),
        ):
            with self.subTest(name):
                communicator = build_mhc(layer_facts(1, 3), parallel)
                self.assert_step(
                    communicator._steps.ffn.prepare, ffn_input, communicator
                )
                self.assertIs(communicator._steps.ffn_output_move, output_move)
                # The fused add + RMSNorm kernels do not write hc_post.
                self.assertEqual(communicator._steps.ffn.fused, ())
                self.assertEqual(
                    communicator._steps.attention.prepare.keywords["carried_fusions"],
                    (),
                )

    def test_the_move_back_over_dp_stays_with_the_layer(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        communicator = build_mhc(layer_facts(1, 3), parallel, allow_reduce_scatter=True)
        self.assertTrue(communicator._steps.returns_over_dp)
        self.assertFalse(communicator._steps.ffn_output.leaves_for_next_layer)
        # The move back also writes the FFN output into the streams.
        self.assertFalse(communicator._local_token_move_can_go_to_next_layer(None))
        max_len = SimpleNamespace(
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False),
        )
        with (
            planning(parallel),
            patch_communicator("get_forward", lambda: SimpleNamespace(sp_active=False)),
            patch_communicator("should_use_dp_reduce_scatterv", lambda: False),
            patch_communicator("can_use_dp_reduce_scatter", lambda: True),
        ):
            step = communicator._postprocess_dp_step(max_len)
            self.assertTrue(
                communicator._ffn_leaves_sum_to_reduce_scatter(max_len, step)
            )

    def test_the_exit_runs_the_move_back_over_dp_it_chose(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for name, reduce_scatterv, is_max_len, move in (
            ("SUM_LEN", True, False, comm_ops._reduce_and_redistribute_output_varlen),
            ("MAX_LEN", False, True, comm_ops._reduce_and_redistribute_output_max_len),
        ):
            with self.subTest(name):
                hc_post = MagicMock(side_effect=lambda h, r, h_res, h_post: h + r)
                communicator = build_mhc(
                    layer_facts(1, 3),
                    parallel,
                    allow_reduce_scatter=True,
                    hc_post=hc_post,
                )
                communicator.mhc.h_res = communicator.mhc.h_post = torch.zeros(2)
                forward_batch = SimpleNamespace(
                    dp_padding_mode=SimpleNamespace(is_max_len=lambda: is_max_len),
                    forward_mode=SimpleNamespace(
                        is_context_parallel_extend=lambda: False
                    ),
                )
                choose = MagicMock(wraps=comm_ops._reduce_and_redistribute_output_step)
                to_local_tokens = MagicMock(side_effect=lambda step, fb, h: h[:2])
                with (
                    planning(parallel),
                    patch_communicator(
                        "should_use_dp_reduce_scatterv", lambda: reduce_scatterv
                    ),
                    patch_communicator("can_use_dp_reduce_scatter", lambda: True),
                    patch_communicator("_reduce_and_redistribute_output_step", choose),
                    patch_communicator("_to_local_tokens", to_local_tokens),
                ):
                    with communicator.ffn_exit(forward_batch) as ffn_exit:
                        self.assertTrue(ffn_exit.mlp_reduce_scatter)
                    hidden, residual = ffn_exit.finish(
                        torch.ones(4, 4), torch.full((2, 4), 2.0)
                    )
                choose.assert_called_once()
                to_local_tokens.assert_called_once()
                self.assertIs(to_local_tokens.call_args.args[0], move)
                hc_post.assert_called_once()
                self.assertIsNone(residual)
                torch.testing.assert_close(hidden, torch.full((2, 4), 3.0))

    def test_a2a_layers_take_their_slice_with_the_coefficients(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        layers = [
            build_mhc(
                planned_facts(
                    i,
                    3,
                    sparse=sparse,
                    previous_sparse=previous,
                    parallel=parallel,
                    a2a=True,
                ),
                parallel,
                a2a=True,
            )
            for i, (sparse, previous) in enumerate(
                ((False, False), (True, False), (True, True))
            )
        ]
        first_a2a, last = layers[1], layers[2]
        self.assert_step(
            first_a2a._steps.ffn.prepare,
            comm_ops._mlp_input_scatter,
            first_a2a,
            scatters_residual=True,
        )
        self.assert_step(
            last._steps.ffn.prepare,
            comm_ops._mlp_input_scatter,
            last,
            scatters_residual=False,
        )
        self.assert_step(
            last._steps.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._gather,
            last,
        )
        state = first_a2a.mhc
        state.h_res, state.h_post = torch.arange(4.0), torch.arange(4.0) + 10
        context = SimpleNamespace(attn_tp_rank=1, attn_tp_size=2)
        residual = state.residual_to_attn_tp_shard(torch.arange(4.0) + 20, context)
        self.assertEqual(residual.tolist(), [22.0, 23.0])
        self.assertEqual(state.h_res.tolist(), [2.0, 3.0])
        self.assertEqual(state.h_post.tolist(), [12.0, 13.0])

    def test_an_input_scattered_batch_keeps_the_residual_on_the_slice(self):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                communicator = build_mhc(
                    layer_facts(layer_id, 3), parallel, allow_reduce_scatter=True
                )
                steps = communicator._input_scattered_steps
                self.assert_step(
                    steps.ffn.prepare,
                    comm_ops._mlp_input_on_residual_shard,
                    communicator,
                )
                self.assert_step(
                    steps.ffn_output_move,
                    comm.CommunicateSummableTensorPairFn._onto_residual_shard,
                    communicator,
                    sums=True,
                    gathers_back=layer_id == 2,
                )
                self.assertTrue(steps.ffn_output_move_completes_sum)
                completes = build_mhc(
                    layer_facts(layer_id, 3), parallel, allow_reduce_scatter=False
                )._input_scattered_steps
                self.assertFalse(completes.ffn_output_move.keywords["sums"])
                self.assertFalse(completes.ffn_output_move_completes_sum)
                # Only the first layer's input, the embedding's partial sum,
                # is completed onto the slice.
                self.assertIs(
                    steps.attention.prepare.keywords["step"].keywords["layer_input"],
                    comm.tp_reduce_scatter if layer_id == 0 else None,
                )
                self.assertIsNone(steps.attention.input_move)
                self.assertIs(
                    steps.attention.handoff, comm_ops._hand_scattered_input_to_attention
                )
                self.assertIs(
                    communicator._steps.ffn_output_move,
                    comm.CommunicateSummableTensorPairFn._trivial,
                )

    def test_two_batch_overlap_gathers_on_the_declarations(self):
        # The dense layer before a sparse one hands the split the attention's
        # rows: MHC writes its output into the streams, then gathers them.
        parallel = parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)
        communicator = build_mhc(
            layer_facts(1, 3, next_sparse=True), parallel, two_batch_overlap=True
        )
        self.assert_step(
            communicator._steps.ffn.prepare, comm_ops._mlp_input_scatter, communicator
        )
        self.assert_step(
            communicator._steps.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._gather,
            communicator,
        )

    def test_the_postprocess_writes_the_output_into_the_streams(self):
        # The last layer also contracts the streams into the hidden states.
        parallel = parallel_of(attn_dp=1, attn_tp=1)
        for layer_id, contracted in ((1, False), (2, True)):
            with self.subTest(layer_id=layer_id):
                communicator = build_mhc(
                    layer_facts(layer_id, 3),
                    parallel,
                    hc_post=lambda h, r, h_res, h_post: h + r,
                )
                communicator.mhc.h_res = communicator.mhc.h_post = torch.zeros(2)
                with (
                    planning(parallel),
                    patch_communicator(
                        "get_forward", lambda: SimpleNamespace(sp_active=False)
                    ),
                    patch.object(
                        mhc_module, "hc_contract", lambda h, hc_mult: h.sum(-1)
                    ),
                ):
                    hidden, residual = communicator.postprocess_layer(
                        torch.ones(2, 4), torch.full((2, 4), 2.0), None
                    )
                self.assertIsNone(residual)
                expected = torch.full((2, 4), 3.0)
                torch.testing.assert_close(
                    hidden, expected.sum(-1) if contracted else expected
                )
                # The write-back consumes this layer's coefficients.
                self.assertIsNone(communicator.mhc.h_res)

    def test_a_gather_over_attention_cp_is_rejected(self):
        # A MoE on the TP group under DSA prefill CP gathers its input over
        # attention CP, which MHC has not been run with.
        parallel = parallel_of(
            attn_dp=1, attn_tp=1, attn_cp=2, enable_prefill_cp=True, moe_dense_tp_size=1
        )
        facts = planned_facts(
            1, 3, sparse=True, previous_sparse=False, parallel=parallel, dsa_cp=True
        )
        with self.assertRaises(NotImplementedError):
            build_mhc(facts, parallel, dsa_cp=True, allow_reduce_scatter=True)
        # A dense layer on every rank computes on its own shard: nothing moves.
        facts = planned_facts(
            1, 3, sparse=False, previous_sparse=False, parallel=parallel, dsa_cp=True
        )
        build_mhc(facts, parallel, dsa_cp=True, allow_reduce_scatter=True)

    def test_input_scattered_attention_under_attention_cp_is_rejected(self):
        scattered = dict(enable_attn_tp_input_scattered=True)
        parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2, **scattered)
        with self.assertRaisesRegex(NotImplementedError, "input-scattered"):
            build_mhc(
                planned_facts(
                    1, 3, sparse=False, previous_sparse=False, parallel=parallel
                ),
                parallel,
            )
        parallel = parallel_of(attn_dp=1, attn_tp=2, **scattered)
        build_mhc(
            planned_facts(1, 3, sparse=False, previous_sparse=False, parallel=parallel),
            parallel,
        )

    def test_a_moe_gathered_over_moe_cp_is_rejected(self):
        # A MoE-CP gather (MoE DP narrower than CP), which MHC has not been run with.
        for prefill_cp in (True, False):
            with self.subTest(prefill_cp=prefill_cp):
                parallel = parallel_of(
                    attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=prefill_cp
                )
                facts = planned_facts(
                    1, 3, sparse=True, previous_sparse=True, parallel=parallel
                )
                with self.assertRaisesRegex(NotImplementedError, "MoE-CP group"):
                    build_mhc(facts, parallel)
        # A dense layer builds, and so does a MoE on its own CP shard (MoE DP = CP).
        parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2)
        build_mhc(
            planned_facts(1, 3, sparse=False, previous_sparse=False, parallel=parallel),
            parallel,
        )
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, attn_cp=2, moe_dp_size=2, moe_tp_size=2
        )
        build_mhc(
            planned_facts(1, 3, sparse=True, previous_sparse=True, parallel=parallel),
            parallel,
        )


class TestTwoBatchOverlap(CustomTestCase):
    """Two-batch overlap splits the attention's rows. A dense MLP on every rank
    hands the sparse layer after it those rows, and the split moves the input
    from the rows the first overlapped layer takes."""

    def layers(self, *, tbo):
        parallel = parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)
        overlap = SimpleNamespace(overlap=SimpleNamespace(enable_two_batch_overlap=tbo))
        layers = []
        with planning(parallel), patch_communicator("get_exec", lambda: overlap):
            for i, sparse in enumerate((False, False, True, True)):
                facts = LayerFacts.init_new(
                    layer_id=i,
                    num_layers=4,
                    is_layer_sparse=sparse,
                    is_previous_layer_sparse=i > 2,
                    is_next_layer_sparse=i >= 1,
                )
                layers.append(
                    LayerCommunicator(
                        layer_facts=facts,
                        input_layernorm=Norm(),
                        post_attention_layernorm=Norm(),
                    )
                )
        return layers

    def test_the_dense_layer_before_the_split_hands_on_the_attention_rows(self):
        pair = comm.CommunicateSummableTensorPairFn
        attention = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
        local = comm.Layout(frozenset({TokenAxis.ATTN_DP, TokenAxis.ATTN_TP_SCATTER}))
        for tbo, gathered in ((True, True), (False, False)):
            with self.subTest(two_batch_overlap=tbo):
                first, before, after, last = self.layers(tbo=tbo)
                for layer in (first, before, after, last):
                    self.assertIsNotNone(layer._declared)
                self.assertIs(first._steps.ffn_output_move, pair._trivial)
                if gathered:
                    self.assertIs(before._steps.ffn_output_move.func, pair._gather)
                    self.assertEqual(after.input_rows, attention)
                    self.assertIsNone(after._steps.attention.input_move)
                else:
                    self.assertIs(before._steps.ffn_output_move, pair._trivial)
                    self.assertEqual(after.input_rows, local)
                    self.assertIs(
                        after._steps.attention.input_move,
                        comm.CommunicateSimpleFn._scattered_to_tp_attn_full,
                    )

    def test_the_split_moves_from_the_first_layer_s_rows(self):
        pair = comm.CommunicateSummableTensorPairFn
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        dp = frozenset({TokenAxis.ATTN_DP})
        with planning(parallel):
            self.assertEqual(
                comm.tbo_split_moves(comm.Layout(dp)), (pair._trivial, pair._trivial)
            )
            self.assertEqual(
                comm.tbo_split_moves(comm.Layout(dp | {TokenAxis.ATTN_TP_SCATTER})),
                (pair._gather, pair._scatter),
            )
            with self.assertRaises(NotImplementedError):
                comm.tbo_split_moves(comm.Layout(frozenset()))

    def build(self, facts, parallel, *, a2a, two_batch_overlap):
        overlap = SimpleNamespace(
            overlap=SimpleNamespace(enable_two_batch_overlap=two_batch_overlap)
        )
        with (
            planning(parallel, a2a=a2a),
            patch_communicator("get_exec", lambda: overlap),
        ):
            communicator = LayerCommunicator(
                layer_facts=facts,
                input_layernorm=Norm(),
                post_attention_layernorm=Norm(),
            )
        communicator._publish_lora_layout = True
        return communicator

    def test_the_rows_are_the_mlp_mode_layout(self):
        for (
            attn_dp,
            attn_tp,
            layer_id,
            sparse,
            previous_sparse,
            a2a,
            dense_fully_dp,
            two_batch_overlap,
        ) in itertools.product(
            (1, 2, 4),
            (1, 2),
            (0, 1, 3),
            (False, True),
            (False, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            if two_batch_overlap and attn_dp == 1:
                continue
            with self.subTest(
                attn_dp=attn_dp,
                attn_tp=attn_tp,
                layer_id=layer_id,
                sparse=sparse,
                previous_sparse=previous_sparse,
                a2a=a2a,
                dense_fully_dp=dense_fully_dp,
                two_batch_overlap=two_batch_overlap,
            ):
                parallel = parallel_of(
                    attn_dp=attn_dp,
                    attn_tp=attn_tp,
                    moe_dense_tp_size=1 if dense_fully_dp else None,
                )
                facts = planned_facts(
                    layer_id,
                    4,
                    sparse=sparse,
                    previous_sparse=previous_sparse,
                    parallel=parallel,
                    a2a=a2a,
                )
                communicator = self.build(
                    facts,
                    parallel,
                    a2a=a2a,
                    two_batch_overlap=two_batch_overlap,
                )
                steps = communicator._steps
                self.assertEqual(
                    steps.ffn.input_rows, communicator._declared.ffn.layout
                )
                published = {}
                with patch_communicator(
                    "get_forward",
                    lambda: SimpleNamespace(set=published.__setitem__),
                ):
                    communicator.publish_mlp_lora_layout(steps)
                # LoRA admits attention DP only with attention TP 1, where the
                # FFN takes every row unless it runs on each rank's own: a MoE
                # dispatched by an a2a backend, or a dense MLP on every rank.
                on_local_rows = a2a if sparse else dense_fully_dp
                if attn_dp > 1 and attn_tp == 1:
                    self.assertIs(
                        published["lora_batch_layout"],
                        (
                            LoRABatchLayout.DP_LOCAL
                            if on_local_rows
                            else LoRABatchLayout.TP_GLOBAL
                        ),
                    )


class TestTheAttentionOutputDecidesItsSum(CustomTestCase):
    """prepare_mlp completes the attention-TP sum exactly when the attention
    output's declaration says it is owed, whatever the attention-TP size."""

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

    def run_steps(self, produced, *, force=False):
        sizes = {
            TokenAxis.ATTN_DP: 2,
            TokenAxis.ATTN_CP: 1,
            TokenAxis.ATTN_TP_SCATTER: 2,
        }
        sides = sides_of(sizes)
        steps, _, _ = comm_boundary._select_input_steps(
            produced,
            residual=sides.input_rows,
            residual_to=sides.ffn_residual_rows,
            need=sides.ffn,
            update=comm.ADD,
            fusions=(),
            force_layernorm_before_gather=force,
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        reduced = []
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=0)
        with (
            patch_communicator(
                "attention_tensor_model_parallel_all_reduce",
                lambda x: reduced.append(x) or 2 * x,
            ),
            patch_communicator(
                "_redistribute_input_to_dp", lambda h, fb, cp_shard_counts=None: h
            ),
            patch_communicator(
                "get_parallel", lambda: parallel_of(attn_dp=2, attn_tp=2)
            ),
            patch_communicator(
                "use_symmetric_memory", lambda g, disabled=False: nullcontext()
            ),
            patch_communicator("is_allocation_symmetric", lambda: False),
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

    def test_a_sum_left_only_for_some_batches_comes_with_the_value(self):
        # Like a mixer exit: the steps take the output as complete, and a value
        # that carries the sum is completed before them.
        layout = sides_of(
            {
                TokenAxis.ATTN_DP: 2,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        ).input_rows
        asked = comm.StageOutput(layout, group=SumGroup.ATTN_TP)
        reductions, hidden, residual = self.run_steps(asked)
        self.assertEqual(reductions, 0)
        torch.testing.assert_close(residual, torch.full((1, HIDDEN), 10.0))

    def test_declarations_the_steps_cannot_run_are_rejected(self):
        layout = sides_of(
            {
                TokenAxis.ATTN_DP: 2,
                TokenAxis.ATTN_CP: 1,
                TokenAxis.ATTN_TP_SCATTER: 2,
            }
        ).input_rows
        for produced in (
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
        steps, fused, _ = comm_boundary._select_input_steps(
            sides.attention_output,
            residual=sides.input_rows,
            residual_to=sides.ffn_residual_rows,
            need=sides.ffn,
            update=comm.ADD,
            fusions=fusions,
            force_layernorm_before_gather=False,
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        return steps, fused

    def test_a_kernel_over_another_group_is_not_chosen(self):
        over_tp = self.fused(SumGroup.TP, True)
        over_attention_tp = self.fused(SumGroup.ATTN_TP, False)
        steps, chosen = self.select(attn_dp=1, fusions=(over_tp, over_attention_tp))
        self.assertIs(steps.func, comm_ops._mlp_input_without_dp)
        self.assertEqual(steps.keywords["fusions"], (over_attention_tp.run,))
        self.assertEqual(chosen, (over_attention_tp,))

    def test_no_kernel_is_chosen_where_rows_are_gathered(self):
        steps, chosen = self.select(
            attn_dp=2, fusions=(self.fused(SumGroup.ATTN_TP, True),)
        )
        self.assertIs(steps.func, comm_ops._mlp_input_dp_partial)
        self.assertEqual(chosen, ())

    def test_aux_capture_follows_the_chosen_kernels(self):
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
                        layer_facts=layer_facts(1, 3),
                        input_layernorm=Norm(),
                        post_attention_layernorm=Norm(),
                    )
                self.assertIs(
                    any(
                        f.may_return_new_residual for f in communicator._steps.ffn.fused
                    ),
                    expected,
                )


class TestTheSequenceParallelRegion(CustomTestCase):
    """While a LayerNorm SP region is active, the layer runs the steps its
    region declarations choose: the linears gather and reduce-scatter
    themselves, so every boundary stays on this rank's slice. Other batches
    take the layer's ordinary declared steps."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP_SCATTER: 2}
    SP_STEPS = comm.BoundarySteps(
        attention=comm.StageEntry(
            prepare=comm_ops._read_input,
            input_rows=comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER})),
            handoff=comm_ops._hand_qkv_hook_its_input,
        ),
        ffn=comm.StageEntry(
            prepare=comm_ops._read_input,
            input_rows=comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER})),
        ),
        ffn_output=sequence_parallel_layer_sides(axis_sizes=SIZES).ffn_output,
        ffn_output_move=comm.CommunicateSummableTensorPairFn._trivial,
        ffn_sum_is_movable=False,
    )

    def unbound(self, steps):
        """``steps`` with the plain residual's binding taken off the FFN input
        and the attention input, whose input owes nothing in the region."""
        ffn = steps.ffn.prepare.keywords["step"]
        attention = steps.attention.prepare.keywords["step"]
        for step, read in ((ffn, comm.NORM_READ), (attention, comm.NORM_QUANT_READ)):
            self.assertIsNone(step.keywords["layer_input"])
            self.assertIs(step.keywords["read"], read)
            self.assertIs(step.keywords["update"], comm.ADD)
        replace = msgspec.structs.replace
        return replace(
            steps,
            ffn=replace(steps.ffn, prepare=ffn.func),
            attention=replace(steps.attention, prepare=attention.func),
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
                steps = self.unbound(comm_boundary._select_boundary_steps(sides))
                self.assertEqual(
                    (
                        steps.attention.input_move,
                        steps.ffn.prepare,
                        steps.ffn_output_move,
                    ),
                    (
                        self.SP_STEPS.attention.input_move,
                        self.SP_STEPS.ffn.prepare,
                        self.SP_STEPS.ffn_output_move,
                    ),
                )

    def test_a_layer_under_sp_takes_both_sets_of_steps(self):
        parallel = parallel_of(attn_dp=1, attn_tp=2)
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                with planning(parallel, sp=True):
                    communicator = LayerCommunicator(
                        layer_facts=layer_facts(layer_id, 3),
                        input_layernorm=Norm(),
                        post_attention_layernorm=Norm(),
                    )
                self.assertEqual(self.unbound(communicator._sp_steps), self.SP_STEPS)
                self.assertIs(
                    communicator._steps.ffn.prepare.keywords["step"].func,
                    comm_ops._mlp_input_without_dp,
                )

    def test_what_the_ffn_exit_reads_comes_from_the_batch_s_steps(self):
        # Inside the region the FFN output is complete: no sum to move.
        parallel = parallel_of(attn_dp=1, attn_tp=2)
        communicator = build(layer_facts(1, 3), parallel, sp=True)
        for active in (False, True):
            with (
                self.subTest(sp_active=active),
                planning(parallel, sp=True),
                patch_communicator(
                    "get_forward", lambda: SimpleNamespace(sp_active=active)
                ),
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
                patch_communicator("is_dp_attention_enabled", lambda: False),
            ):
                self.assertIs(
                    communicator._ffn_sum_can_move_to_next_layer(None), not active
                )

    def test_without_sp_there_is_no_region(self):
        with planning(parallel_of(attn_dp=1, attn_tp=2)):
            communicator = LayerCommunicator(
                layer_facts=layer_facts(1, 3),
                input_layernorm=Norm(),
                post_attention_layernorm=Norm(),
            )
        self.assertIsNone(communicator._sp_steps)

    def test_the_same_selector_chooses_a_layer_s_ordinary_steps(self):
        # Under attention DP the FFN exit chooses the move back per batch.
        kept = comm_boundary._select_boundary_steps(
            sides_of(self.SIZES, leaves_next=True)
        )
        self.assertTrue(kept.ffn_output.leaves_for_next_layer)
        self.assertTrue(kept.ffn_sum_is_movable)
        self.assertIs(
            kept.ffn_output_move, comm.CommunicateSummableTensorPairFn._trivial
        )
        returned = comm_boundary._select_boundary_steps(
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
                steps = comm_boundary._select_boundary_steps(sides)
                self.assertIs(
                    steps.attention.prepare.keywords["step"].keywords["layer_input"],
                    comm.tp_reduce_scatter,
                )
                self.assertIsNone(steps.attention.input_move)
                slice_ = comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
                self.assertEqual(steps.attention.input_rows, slice_)
                self.assertEqual(steps.ffn.input_rows, comm.Layout(frozenset()))
                self.assertIs(
                    steps.ffn.prepare.keywords["step"].func,
                    comm_ops._mlp_input_residual_into_sum,
                )
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
            (True, comm_ops._mlp_input_residual_into_sum),
            (False, comm_ops._mlp_input_without_dp),
        ):
            with self.subTest(residual_joins_sum=joins):
                steps, _, _ = comm_boundary._select_input_steps(
                    owed,
                    residual=local,
                    residual_to=attention,
                    need=comm.StageInput(attention),
                    update=comm.ADD,
                    fusions=(),
                    force_layernorm_before_gather=False,
                    residual_joins_sum=joins,
                    cp_moves=None,
                    enters_stack=False,
                )
                self.assertIs(getattr(steps, "func", steps), step)

    def test_a_complete_value_is_sliced_onto_each_rank(self):
        # An FFN on each rank's slice after an FFN, or at the start of the
        # layer stack, takes a complete input.
        sizes = self.SIZES
        attention = comm.Layout.sharded_over(axis_sizes=sizes)
        local = comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER}))
        step, _, _ = comm_boundary._select_input_steps(
            comm.StageOutput(attention),
            residual=attention,
            residual_to=local,
            need=comm.StageInput(local),
            update=comm.ADD,
            fusions=(),
            force_layernorm_before_gather=False,
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        self.assertIs(step.func, comm_ops._mlp_input_slice)
        hidden = torch.arange(4.0)[:, None].expand(4, HIDDEN).clone()
        residual = torch.ones(4, HIDDEN)
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        out, out_residual = step(hidden, residual, None, Norm(), context)
        torch.testing.assert_close(out_residual, hidden[2:] + 1)
        torch.testing.assert_close(out, 2 * (hidden[2:] + 1))
        # At the start of the layer stack the input is the residual.
        out, out_residual = step(hidden, None, None, Norm(), context)
        torch.testing.assert_close(out_residual, hidden[2:])
        torch.testing.assert_close(out, 2 * hidden[2:])

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
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=scattered),
                ),
                patch_communicator(
                    "get_forward", lambda: SimpleNamespace(sp_active=False)
                ),
            ):
                self.assertIs(
                    communicator._batch_steps(None),
                    communicator._input_scattered_steps
                    if scattered
                    else communicator._steps,
                )

    def test_prepare_attn_completes_the_scattered_input(self):
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
            patch_communicator("get_parallel", lambda: parallel),
            patch_communicator(
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=True),
            ),
            patch_communicator("get_forward", lambda: SimpleNamespace(sp_active=False)),
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

    def test_the_layer_holds_steps_only(self):
        communicator = build(layer_facts(1, 3), parallel_of(attn_dp=2, attn_tp=2))
        self.assertIsInstance(communicator._steps, comm.BoundarySteps)
        for attribute in self.SEPARATE:
            self.assertFalse(hasattr(communicator, attribute), attribute)
        self.assertIs(communicator._batch_steps(None), communicator._steps)
        self.assertTrue(communicator._steps.returns_over_dp)


class TestTheAttentionInputHalf(CustomTestCase):
    """Every batch variant's half into the attention tries the fused entries
    the layer chose, and starts the residual only on the stack's first
    layer."""

    def build_fusable(self, facts, parallel, **planning_kwargs):
        with planning(parallel, **planning_kwargs):
            return LayerCommunicator(
                layer_facts=facts,
                input_layernorm=FusableNorm(),
                post_attention_layernorm=Norm(),
            )

    def test_every_variant_takes_the_layers_fused_entries(self):
        cp = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=True)
        for name, parallel, kwargs in (
            (
                "input-scattered",
                parallel_of(attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True),
                {},
            ),
            ("LayerNorm SP", parallel_of(attn_dp=1, attn_tp=2), {"sp": True}),
            ("prefill CP", cp, {}),
        ):
            for layer_id in (0, 1):
                with self.subTest(name, layer_id=layer_id):
                    facts = planned_facts(
                        layer_id,
                        3,
                        sparse=False,
                        previous_sparse=False,
                        parallel=parallel,
                    )
                    communicator = self.build_fusable(facts, parallel, **kwargs)
                    fusions = communicator._attn_input_fusions
                    self.assertTrue(fusions)
                    variants = [
                        steps
                        for steps in (
                            communicator._steps,
                            communicator._sp_steps,
                            communicator._input_scattered_steps,
                            communicator._cp_steps,
                        )
                        if steps is not None
                    ]
                    self.assertEqual(len(variants), 2)
                    for steps in variants:
                        prepare = steps.attention.prepare
                        self.assertIs(prepare.keywords["carried_fusions"], fusions)
                        bound = prepare.keywords["step"].keywords
                        self.assertIs(bound["enters_stack"], layer_id == 0)


class TestPrefillCP(CustomTestCase):
    """A prefill CP shards a batch's tokens over attention CP only on a CP
    extend; the layer runs its CP steps on those batches and its ordinary ones
    on every other."""

    def cp_parallel(self, **overrides):
        return parallel_of(
            attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=True, **overrides
        )

    def dsa_parallel(self, **overrides):
        # DSA and MLA CP run attention TP 1 and the dense MLP on every rank.
        return parallel_of(
            attn_dp=1,
            attn_tp=1,
            attn_cp=2,
            enable_prefill_cp=True,
            moe_dense_tp_size=1,
            **overrides,
        )

    def test_a_dsa_cp_extend_leaves_its_sum_to_the_reduce_scatter(self):
        # A MoE on the TP group, as under DSA interleave CP without a2a.
        parallel = self.dsa_parallel()
        facts = planned_facts(
            1, 3, sparse=True, previous_sparse=False, parallel=parallel, dsa_cp=True
        )
        communicator = build(facts, parallel, dsa_cp=True, allow_reduce_scatter=True)
        cp, ordinary = communicator._cp_steps, communicator._steps
        self.assertIs(
            cp.ffn.prepare.keywords["step"].func,
            comm_ops._mlp_input_gather_attention_cp,
        )
        self.assertIs(
            cp.ffn.prepare.keywords["step"].keywords["gather"].func,
            comm_ops._read_input,
        )
        self.assertIs(
            cp.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._reduce_scatter_over_cp,
        )
        self.assertTrue(cp.ffn_output.leaves_for_reduce_scatter)
        self.assertFalse(cp.ffn_output.leaves_for_next_layer)
        # The dense layer before it ran on the same shard: nothing to move.
        self.assertIsNone(cp.attention.input_move)
        # Other batches hold every token on each CP rank: the MoE sums itself.
        self.assertIs(ordinary.ffn.prepare.keywords["step"].func, comm_ops._read_input)
        self.assertIs(
            ordinary.ffn_output_move, comm.CommunicateSummableTensorPairFn._trivial
        )
        with (
            patch_communicator("get_forward", lambda: SimpleNamespace(sp_active=False)),
            patch_communicator(
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=False),
            ),
        ):
            for shards, leaves in ((True, True), (False, False)):
                with (
                    self.subTest(cp_extend=shards),
                    patch_communicator("_batch_shards_over_cp", lambda fb: shards),
                ):
                    self.assertIs(
                        communicator._ffn_leaves_sum_to_reduce_scatter(
                            SimpleNamespace(
                                forward_mode=SimpleNamespace(
                                    is_context_parallel_extend=lambda: False
                                )
                            ),
                            None,
                        ),
                        leaves,
                    )

    def test_dsa_cp_asks_its_own_predicates_for_a_cp_extend(self):
        def batch(cp_extend):
            return SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_context_parallel_extend=lambda: cp_extend
                )
            )

        with planning(self.dsa_parallel(), dsa_cp=True):
            for cp_extend, active, shards in (
                (True, True, True),
                (True, False, False),
                (False, True, False),
            ):
                with (
                    self.subTest(cp_extend=cp_extend, active=active),
                    patch_communicator("dsa_use_prefill_cp", lambda fb: active),
                    patch_communicator("is_mla_cp_active", lambda fb: False),
                ):
                    self.assertIs(
                        comm_layout._batch_shards_over_cp(batch(cp_extend)), shards
                    )

    def test_dsa_cp_dense_layers_run_on_their_shard(self):
        parallel = self.dsa_parallel()
        facts = planned_facts(
            1, 3, sparse=False, previous_sparse=False, parallel=parallel, dsa_cp=True
        )
        communicator = build(facts, parallel, dsa_cp=True, allow_reduce_scatter=True)
        for steps in (communicator._steps, communicator._cp_steps):
            self.assertIs(steps.ffn.prepare.keywords["step"].func, comm_ops._read_input)
            self.assertIsNone(steps.ffn_output.group)
            self.assertIs(
                steps.ffn_output_move, comm.CommunicateSummableTensorPairFn._trivial
            )

    def test_a_cp_extend_gathers_over_cp_and_takes_its_chunk_back(self):
        communicator = build(layer_facts(1, 3), self.cp_parallel())
        cp = communicator._cp_steps
        self.assertIs(
            cp.ffn.prepare.keywords["step"].func, comm_ops._mlp_input_gather_moe_cp
        )
        # Each rank completes its own chunk before the gather.
        self.assertIs(
            cp.ffn.prepare.keywords["step"].keywords["gather"].func,
            comm_ops._mlp_input_without_dp,
        )
        self.assertIs(
            cp.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
        )
        self.assertFalse(cp.returns_over_dp)
        self.assertFalse(cp.ffn_output.leaves_for_next_layer)
        self.assertFalse(cp.ffn_output.leaves_for_reduce_scatter)
        self.assertIs(
            communicator._steps.ffn.prepare.keywords["step"].func,
            comm_ops._mlp_input_without_dp,
        )
        self.assertIs(
            communicator._steps.ffn_output_move,
            comm.CommunicateSummableTensorPairFn._trivial,
        )

    def test_a_moe_on_its_own_cp_shard_hands_on_its_sum(self):
        # MoE DP equal to CP: each CP rank's MoE runs on its own TP ranks.
        parallel = self.cp_parallel(moe_dp_size=2, moe_tp_size=2)
        communicator = build(
            layer_facts(1, 3, sparse=True, previous_sparse=True),
            parallel,
            allow_deferred_ffn_reduction=True,
        )
        for steps in (communicator._cp_steps, communicator._steps):
            # The attention's rows are the MoE's: nothing to gather or take back.
            self.assertIs(
                steps.ffn.prepare.keywords["step"].func, comm_ops._mlp_input_without_dp
            )
            self.assertIs(
                steps.ffn_output_move, comm.CommunicateSummableTensorPairFn._trivial
            )
            self.assertTrue(steps.ffn_output.leaves_for_next_layer)
        dense = build(layer_facts(1, 3), parallel, allow_deferred_ffn_reduction=True)
        self.assertIs(
            dense._cp_steps.ffn.prepare.keywords["step"].func,
            comm_ops._mlp_input_gather_moe_cp,
        )
        self.assertFalse(dense._cp_steps.ffn_output.leaves_for_next_layer)

    def test_the_fused_kernels_run_on_each_chunk(self):
        with planning(self.cp_parallel()):
            communicator = LayerCommunicator(
                layer_facts=layer_facts(1, 3),
                input_layernorm=FusableNorm(),
                post_attention_layernorm=FusableNorm(),
            )
        chunk = communicator._cp_steps.ffn.prepare.keywords["step"].keywords["gather"]
        self.assertEqual(
            chunk.keywords["fusions"],
            (communicator._mlp_input_reduce_output_and_update_and_read_residual,),
        )

    def cp_outcome(self, parallel, *, sparse=True, a2a=False, dsa_cp=False):
        """A layer under attention CP: "cp steps" for the batches that shard
        their tokens, "ordinary" when no batch does, or "refused"."""
        facts = planned_facts(
            1,
            3,
            sparse=sparse,
            previous_sparse=sparse,
            parallel=parallel,
            a2a=a2a,
            dsa_cp=dsa_cp,
        )
        try:
            layer = build(
                facts, parallel, a2a=a2a, dsa_cp=dsa_cp, allow_reduce_scatter=True
            )
        except NotImplementedError:
            return "refused"
        return "ordinary" if layer._cp_steps is None else "cp steps"

    def test_which_cp_the_steps_cover(self):
        dp_cp = dict(attn_dp=2, attn_tp=1, attn_cp=2, enable_prefill_cp=True)
        moe_dp_eq_cp = dict(attn_cp=2, enable_prefill_cp=True, moe_dp_size=2)
        dense = dict(sparse=False)
        for expected, name, parallel, kwargs in (
            ("cp steps", "DSA or MLA CP", self.dsa_parallel(), dict(dsa_cp=True)),
            (
                "cp steps",
                "DSA or MLA CP, an a2a MoE under attention DP",
                parallel_of(**dp_cp, moe_dense_tp_size=1),
                dict(dsa_cp=True, a2a=True),
            ),
            ("cp steps", "prefill CP", self.cp_parallel(), dense),
            ("cp steps", "CP under attention DP", parallel_of(**dp_cp), dense),
            (
                "cp steps",
                "a MoE under attention DP and GQA CP",
                parallel_of(**dp_cp),
                {},
            ),
            (
                "cp steps",
                "MoE DP equal to CP",
                parallel_of(attn_dp=1, attn_tp=2, **moe_dp_eq_cp),
                {},
            ),
            (
                "cp steps",
                "a dense layer beside it under attention DP",
                parallel_of(attn_dp=2, attn_tp=1, **moe_dp_eq_cp),
                dense,
            ),
            (
                "ordinary",
                "CP without prefill CP",
                parallel_of(attn_dp=1, attn_tp=2, attn_cp=2),
                {},
            ),
            (
                "ordinary",
                "an a2a MoE under attention DP, CP without prefill CP",
                parallel_of(attn_dp=2, attn_tp=1, attn_cp=2),
                dict(a2a=True),
            ),
            (
                "refused",
                "DSA or MLA CP, MoE DP equal to CP",
                self.dsa_parallel(moe_dp_size=2, moe_tp_size=1),
                dict(dsa_cp=True),
            ),
            (
                "refused",
                "an a2a MoE under attention DP and GQA CP",
                parallel_of(**dp_cp),
                dict(a2a=True),
            ),
            (
                "refused",
                "MoE DP equal to CP under attention DP",
                parallel_of(attn_dp=2, attn_tp=1, **moe_dp_eq_cp),
                {},
            ),
            (
                "refused",
                "the same without prefill CP",
                parallel_of(attn_dp=2, attn_tp=1, attn_cp=2, moe_dp_size=2),
                {},
            ),
        ):
            with self.subTest(name):
                self.assertEqual(self.cp_outcome(parallel, **kwargs), expected)

    def test_under_attention_dp_one_dp_sum_gathers_both_axes(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2, attn_cp=2, enable_prefill_cp=True)
        communicator = build(layer_facts(1, 3), parallel)
        cp, ordinary = communicator._cp_steps, communicator._steps
        self.assertIs(
            cp.ffn.prepare.keywords["step"].func, comm_ops._mlp_input_dp_partial
        )
        self.assertTrue(cp.ffn.prepare.keywords["step"].keywords["places_cp_shards"])
        self.assertIs(
            cp.ffn_output_move, comm.CommunicateSummableTensorPairFn._take_back_cp_shard
        )
        self.assertIs(
            ordinary.ffn.prepare.keywords["step"].func, comm_ops._mlp_input_dp_partial
        )
        self.assertFalse(
            ordinary.ffn.prepare.keywords["step"].keywords["places_cp_shards"]
        )
        self.assertTrue(ordinary.returns_over_dp)
        for steps in (cp, ordinary):
            self.assertFalse(steps.ffn_output.leaves_for_next_layer)
            self.assertFalse(steps.ffn_output.leaves_for_reduce_scatter)
            self.assertFalse(steps.ffn_output.leaves_for_reduce_scatterv)

    def test_a_batch_runs_them_only_on_a_cp_extend(self):
        communicator = build(layer_facts(1, 3), self.cp_parallel())

        def batch(cp_extend):
            return SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_context_parallel_extend=lambda: cp_extend
                )
            )

        def unread(fb):
            raise AssertionError("read for a batch that is not a CP extend")

        for fb, rows, expected in (
            (batch(False), unread, communicator._steps),
            (batch(True), lambda fb: None, communicator._steps),
            (batch(True), lambda fb: [2, 1], communicator._cp_steps),
        ):
            with (
                self.subTest(expected=expected is communicator._cp_steps),
                planning(self.cp_parallel()),
                patch_communicator("moe_cp_gathered_rows", rows),
                patch_communicator(
                    "get_forward", lambda: SimpleNamespace(sp_active=False)
                ),
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
            ):
                self.assertIs(communicator._batch_steps(fb), expected)


class TestBranchRows(CustomTestCase):
    """Two FFNs that branch from one input and merge again (LongCat's MoE and
    dense branch): each layer's communicator gives the rows of its FFN input,
    of its residual while the FFN runs and of what it hands on, and a complete
    value moves between them."""

    local = comm.Layout(frozenset({TokenAxis.ATTN_DP, TokenAxis.ATTN_TP_SCATTER}))
    attention = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
    full = comm.Layout(frozenset())

    def rows(self, communicator, parallel, cp_extend=False):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: cp_extend)
        )
        with (
            planning(parallel),
            patch_communicator("get_forward", lambda: SimpleNamespace(sp_active=False)),
            patch_communicator(
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=False),
            ),
            patch_communicator("moe_cp_gathered_rows", lambda fb: [2, 1]),
        ):
            return communicator._branch_rows(batch)

    def test_each_branch_declares_its_rows(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        dense = build(layer_facts(2, 6), parallel)
        self.assertEqual(
            self.rows(dense, parallel), (self.full, self.attention, self.attention)
        )
        for a2a, expected in (
            (True, (self.local, self.local, self.local)),
            (False, (self.full, self.attention, self.attention)),
        ):
            with self.subTest(a2a=a2a):
                moe = build(
                    layer_facts(1, 3, sparse=True, previous_sparse=True),
                    parallel,
                    a2a=a2a,
                )
                self.assertEqual(self.rows(moe, parallel), expected)

    def test_a_batch_on_other_steps_has_no_branch_rows(self):
        parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=True)
        communicator = build(layer_facts(1, 3), parallel)
        with self.assertRaises(NotImplementedError):
            self.rows(communicator, parallel, cp_extend=True)

    def test_branches_move_between_the_declared_rows(self):
        # An a2a MoE beside a dense FFN, under attention DP and TP.
        class Communicator(SimpleNamespace):
            branch_input = LayerCommunicator.branch_input
            branch_output = LayerCommunicator.branch_output
            merge_branch = LayerCommunicator.merge_branch

        moe = Communicator(_branch_rows=lambda fb: (self.local, self.local, self.local))
        dense = Communicator(
            _branch_rows=lambda fb: (self.full, self.attention, self.attention)
        )
        moves = []

        def recorded(value, rows, to, forward_batch):
            moves.append((value, rows, to))
            return value

        with patch_communicator("move_rows", recorded):
            dense.branch_input(moe, "h0", "residual", None)
            self.assertEqual(
                moves,
                [
                    ("h0", self.local, self.full),
                    ("residual", self.local, self.attention),
                ],
            )
            moves.clear()
            moe.branch_output("shortcut", None)
            self.assertEqual(moves, [("shortcut", self.local, self.local)])
            moves.clear()
            merged = moe.merge_branch(1, 2, "residual", dense, None)
            self.assertEqual(
                moves,
                [
                    (2, self.attention, self.local),
                    ("residual", self.attention, self.local),
                ],
            )
            # The contribution adds; the residual is the dense branch's.
            self.assertEqual(merged, (3, "residual"))

    def test_a_value_is_gathered_then_cut_in_order(self):
        # Stand-ins that change the rows distinctly, so the order of the moves shows.
        def tp_gather(h):
            return torch.cat([h, h])

        def dp_gather(h):
            return torch.cat([h, h + 100])

        def dp_scatter(h):
            return h[: len(h) // 2]

        def tp_cut(h):
            return h[len(h) // 2 :]

        value = torch.arange(4.0).view(4, 1)
        with (
            patch_communicator("_redistribute_from_attn_tp_shards", tp_gather),
            patch_communicator("_redistribute_input_to_dp", lambda h, fb: dp_gather(h)),
            patch_communicator("_to_local_tokens", lambda step, fb, h: dp_scatter(h)),
            patch_communicator(
                "get_parallel",
                lambda: SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
            ),
        ):
            for rows, to, expected in (
                (self.local, self.full, dp_gather(tp_gather(value))),
                (self.local, self.attention, tp_gather(value)),
                (self.attention, self.local, tp_cut(value)),
                (self.full, self.attention, dp_scatter(value)),
                (self.full, self.local, tp_cut(dp_scatter(value))),
                (self.attention, self.attention, value),
            ):
                with self.subTest(rows=rows, to=to):
                    self.assertTrue(
                        torch.equal(comm.move_rows(value, rows, to, None), expected)
                    )
            cp = comm.Layout(frozenset({TokenAxis.ATTN_CP}))
            for rows, to in (
                (comm.Layout(frozenset({TokenAxis.ATTN_TP_SCATTER})), self.attention),
                (cp, self.full),
            ):
                with (
                    self.subTest(rows=rows, to=to),
                    self.assertRaises(NotImplementedError),
                ):
                    comm.move_rows(value, rows, to, None)


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


def fake_dp_gather(
    global_tokens, local_tokens, forward_batch, cp_shard_counts=None, *, is_partial
):
    state = WORLD[0].state()
    global_tokens.zero_()
    if is_partial or state.parallel.attn_tp_rank == 0:
        global_tokens[state.offset : state.offset + state.rows] += local_tokens[
            : state.rows
        ]
    global_tokens.copy_(state.parallel.tp_group.all_reduce(global_tokens))


def fake_dp_scatter(local_tokens, global_tokens, forward_batch, cp_shard_counts=None):
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
        "get_local_dp_buffer": lambda g, hidden_size=None: torch.zeros(
            state().local_rows, hidden_size or HIDDEN, dtype=torch.double
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
            if any(hasattr(m, name) for m in COMMUNICATOR_MODULES):
                stack.enter_context(patch_communicator(name, value))
        # What the MoE declares its skipped reduction leaves, read in its module.
        for name, value in (
            ("get_parallel", lambda: state().parallel),
            ("get_moe_a2a_backend", comm_layer.get_moe_a2a_backend),
            (
                "post_experts_output_is_complete",
                replaced["post_experts_output_is_complete"],
            ),
            ("post_experts_reduction_group", replaced["post_experts_reduction_group"]),
            ("get_lora", replaced["get_lora"]),
            ("get_exec", replaced["get_exec"]),
        ):
            stack.enter_context(patch.object(moe_utils, name, value))
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
        or comm_ops.should_use_dp_reduce_scatterv()
    ):
        return partial_sum
    return comm_layout.post_experts_reduction_group().all_reduce(partial_sum)


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
                    layer_facts=layer_facts(
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
                    # without an a2a backend.
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


class TestTheFfnInputReduction(CustomTestCase):
    """With quantized communications a prefill reduces the FFN input quantized
    for a plain residual; MHC sums its streams in full precision."""

    def test_only_a_plain_residual_reduces_quantized(self):
        exec_ = SimpleNamespace(comm=SimpleNamespace(enable_quant_communications=True))
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False)
        )
        for adds_plainly, reduction in ((True, "quant"), (False, "full")):
            with self.subTest(adds_plainly=adds_plainly):
                update = SimpleNamespace(adds_plainly=adds_plainly)
                read = SimpleNamespace(
                    update_and_read=lambda update, h, r, norm: (h, r)
                )
                calls = []
                with (
                    patch_communicator("get_exec", lambda: exec_),
                    patch_communicator(
                        "attention_tensor_model_parallel_quant_all_reduce",
                        lambda h: calls.append("quant") or h,
                    ),
                    patch_communicator(
                        "attention_tensor_model_parallel_all_reduce",
                        lambda h: calls.append("full") or h,
                    ),
                ):
                    comm_ops._mlp_input_without_dp(
                        torch.ones(2, HIDDEN),
                        torch.ones(2, HIDDEN),
                        batch,
                        None,
                        SimpleNamespace(cache=None),
                        gathers_residual=False,
                        fusions=(),
                        read=read,
                        update=update,
                    )
                self.assertEqual(calls, [reduction])


class TestALayerThatIsOneStage(CustomTestCase):
    """A layer that is one stage reads its input and writes its output into the
    residual as its stage declares."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP_SCATTER: 2}

    class ProbeRead:
        norms_plainly = True

        def __init__(self):
            self.reads = 0

        def read(self, residual, norm, quant_format="", post_residual_addition=None):
            self.reads += 1
            return residual + 100, residual

        def update_and_read(
            self,
            update,
            hidden_states,
            residual,
            norm,
            quant_format="",
            post_residual_addition=None,
        ):
            return self.read(update.update(hidden_states, residual), norm)

    def _stage(self, read, update):
        rows = comm.Layout.sharded_over(axis_sizes=self.SIZES)
        return comm.LayerStage(
            kind=comm.StageKind.FFN,
            edges=comm.stage_edges(
                previous=None,
                stage=comm.StageDecl(
                    comm.StageInput(rows, read=read),
                    comm.StageOutput(rows, update=update),
                ),
                rows=rows,
            ),
        )

    def test_its_steps_run_the_declared_read_and_update(self):
        read, update = self.ProbeRead(), SimpleNamespace(adds_plainly=True)
        communicator = build(
            layer_facts(1, 3),
            parallel_of(attn_dp=1, attn_tp=2),
            stage=self._stage(read, update),
        )
        into, out_of = communicator.stage_edges
        self.assertIs(into.need.read, read)
        self.assertIs(out_of.produced.update, update)
        hidden, residual = torch.ones(2, HIDDEN), torch.full((2, HIDDEN), 3.0)
        out, out_residual = communicator._steps.ffn.prepare(
            hidden, residual, None, Norm(), None
        )
        self.assertEqual(read.reads, 1)
        torch.testing.assert_close(out_residual, torch.full((2, HIDDEN), 4.0))
        torch.testing.assert_close(out, torch.full((2, HIDDEN), 104.0))

    def test_it_takes_no_residual_of_its_own(self):
        own = comm.LayerResidual(
            attention_read=self.ProbeRead(),
            attention_update=comm.ADD,
            ffn_read=comm.NORM_READ,
            ffn_update=comm.ADD,
        )
        with self.assertRaises(ValueError):
            build(
                layer_facts(1, 3),
                parallel_of(attn_dp=1, attn_tp=2),
                stage=self._stage(comm.NORM_READ, comm.ADD),
                residual=own,
            )


if __name__ == "__main__":
    unittest.main()
