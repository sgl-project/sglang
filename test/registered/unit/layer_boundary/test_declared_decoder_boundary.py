"""The boundaries of an attention and an FFN, chosen from both sides'
declarations. The FFN runs on the TP group (a dense MLP, or a MoE not
dispatched per DP shard) or on each attention-TP rank's slice of its DP shard's
rows (a MoE dispatched per DP shard).

The numeric checks run every rank of a DP x attention-TP world as a thread over
fake collectives that wait for all members of their group, build the layers'
boundaries through the production factories.
"""

import itertools
import threading
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary import (
    SumGroup,
    TokenAxis,
)
from sglang.srt.layers.layer_boundary import boundary as comm_boundary
from sglang.srt.layers.layer_boundary import construction as comm_layer
from sglang.srt.layers.layer_boundary import exit as comm_exit
from sglang.srt.layers.layer_boundary import layout as comm_layout
from sglang.srt.layers.layer_boundary import ops as transport_ops
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.adapters import branch
from sglang.srt.layers.layer_boundary.construction import BatchVariant
from sglang.srt.layers.layer_boundary.fusions.allreduce import fused_ffn_input
from sglang.srt.layers.layer_boundary.ops import (
    attn_cp_reduce_scatter_output,
    attn_tp_gather_input,
    attn_tp_slice_output,
    dp_cp_take_back_output,
    keep_output,
    moe_cp_take_back_output,
    residual_slice_output,
    update_attn_tp_gather_output,
)
from sglang.srt.layers.layer_boundary.residual import mhc as mhc_module
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_READOUT,
    PLAIN_RESIDUAL_OPS,
)
from sglang.srt.layers.layer_boundary.residual.stream import OwedOutput, ResidualStream
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward
from sglang.test.boundary_fixtures import (
    finish_exit,
    make_test_stages,
    postprocess_output,
    prepare_input,
)
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
        enable_cp_tp_group_sharing=False,
        moe_ep_size=1,
        moe_tp_size=attn_dp * attn_cp * attn_tp,
        moe_dp_size=1,
        dwdp_size=1,
        attn_dp_enabled=attn_dp > 1,
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
def planning(
    parallel,
    *,
    sp=False,
    a2a=False,
    dsa_cp=False,
    boundary_reduction="rs+rsv",
):
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
                comm=SimpleNamespace(boundary_reduction=boundary_reduction),
                overlap=SimpleNamespace(enable_two_batch_overlap=False),
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


def layer_case(
    layer_id,
    num_layers,
    *,
    sparse=False,
    previous_sparse=False,
    next_layer_sparse=False,
):
    return dict(
        sparse=sparse,
        first=layer_id == 0,
        last=layer_id == num_layers - 1,
        previous_sparse=previous_sparse,
        next_layer_sparse=next_layer_sparse,
    )


def build(
    facts,
    parallel,
    *,
    sp=False,
    a2a=False,
    dsa_cp=False,
    **kwargs,
):
    with planning(
        parallel,
        sp=sp,
        a2a=a2a,
        dsa_cp=dsa_cp,
        boundary_reduction=kwargs.pop("boundary_reduction", "rs+rsv"),
    ):
        return make_test_stages(
            **facts,
            attention_norm=Norm(),
            ffn_norm=Norm(),
            **kwargs,
        )


class TestStageLayoutSelection(CustomTestCase):
    """Production factories choose the input and output paths for each stage."""

    def test_the_order_follows_the_attention_output(self):
        for attn_tp, force, order in (
            (2, False, comm_ops._dp_gather_sum_read),
            (2, True, comm_ops._reduce_update_read_dp_gather),
            (1, False, comm_ops._reduce_update_read_dp_gather),
        ):
            with self.subTest(attn_tp=attn_tp, force=force):
                communicator = build(
                    layer_case(1, 3),
                    parallel_of(attn_dp=2, attn_tp=attn_tp),
                    residual=PLAIN_RESIDUAL_OPS._replace(
                        ffn_readout=replace(NORM_READOUT, reads_before_dp_gather=force)
                    ),
                )
                self.assertIs(
                    communicator.ffn.plan.paths.get(BatchVariant.ORDINARY)
                    .entry.prepare.keywords["step"]
                    .func,
                    order,
                )

    def test_the_last_a2a_layer_folds_the_residual_back(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        last = build(
            layer_case(3, 4, sparse=True, previous_sparse=True),
            parallel,
            a2a=True,
        )
        self.assertIs(
            last.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move.func,
            update_attn_tp_gather_output,
        )

    def test_an_attention_that_gathers_its_slice_itself(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        facts = layer_case(2, 4, sparse=True, previous_sparse=True)
        with patch_communicator("_use_ag_after_qlora", True):
            layer = build(facts, parallel, a2a=True)
        self.assertIsNone(
            layer.attn.plan.paths.get(BatchVariant.ORDINARY).entry.input_move
        )

    def test_cp_without_prefill_cp(self):
        """No batch shards its tokens, so no CP steps. Under attention DP the
        reduce-scatter back runs over the TP group, which spans the CP ranks,
        so the FFN completes its own sum; without attention DP it may leave it
        and input-scattered attention runs, as without CP."""
        under_dp = build(layer_case(1, 3), parallel_of(attn_dp=2, attn_tp=1, attn_cp=2))
        self.assertIsNone(under_dp.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL))
        self.assertFalse(
            under_dp.ffn.plan.paths.get(BatchVariant.ORDINARY).output.may_defer_to_next
        )
        self.assertFalse(
            under_dp.ffn.plan.paths.get(
                BatchVariant.ORDINARY
            ).output.may_reduce_scatterv
        )
        without_dp = build(
            layer_case(1, 3),
            parallel_of(
                attn_dp=1, attn_tp=2, attn_cp=2, enable_attn_tp_input_scattered=True
            ),
        )
        self.assertTrue(
            without_dp.ffn.plan.paths.get(
                BatchVariant.ORDINARY
            ).output.may_defer_to_next
        )
        self.assertIsNotNone(
            without_dp.ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
        )


def build_mhc(
    facts, parallel, *, a2a=False, dsa_cp=False, two_batch_overlap=False, **kwargs
):
    overlap = SimpleNamespace(
        comm=SimpleNamespace(
            boundary_reduction=kwargs.pop("boundary_reduction", "rs+rsv")
        ),
        overlap=SimpleNamespace(enable_two_batch_overlap=two_batch_overlap),
    )
    with (
        planning(parallel, a2a=a2a, dsa_cp=dsa_cp),
        patch_communicator("get_exec", lambda: overlap),
    ):
        mhc = mhc_module.MHCState(
            is_last_layer=facts["last"],
            hc_mult=2,
            hc_attn_pre=lambda *a: None,
            hc_ffn_pre=lambda *a: None,
            **{"hc_post": lambda *a: None, **kwargs},
        )
        stages = make_test_stages(
            **facts,
            attention_norm=Norm(),
            ffn_norm=Norm(),
            residual=mhc.residual_ops(),
        )
        stages.mhc = mhc
        return stages


class TestMhcOnTheDeclarations(CustomTestCase):
    """An MHC layer takes the same declarations and runs the same steps as any
    other layer; only the residual operations the steps run are MHC's."""

    def assert_step(self, step, func, communicator, **keywords):
        if step.func is comm_ops._run_entry:
            step = step.keywords["step"]
        self.assertIs(step.func, func)
        declaration = communicator.ffn.declaration
        if "read" in step.keywords:
            # The FFN input: the attention output's hc_post, then the FFN's read.
            self.assertIs(step.keywords["read"], declaration.read)
            self.assertNotIn("update", step.keywords)
        else:
            # The FFN output's move: its hc_post.
            self.assertIs(step.keywords["update"], declaration.update)
        for key, value in keywords.items():
            self.assertEqual(step.keywords[key], value, key)

    def test_the_move_back_over_dp_stays_with_the_layer(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        communicator = build_mhc(layer_case(1, 3), parallel)
        self.assertTrue(
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY).returns_over_dp
        )
        self.assertFalse(
            communicator.ffn.plan.paths.get(
                BatchVariant.ORDINARY
            ).output.may_defer_to_next
        )
        # The move back also writes the FFN output into the streams.
        self.assertFalse(
            communicator.ffn.plan.output._next_input_can_scatter(
                communicator.ffn.plan.output.plan.path_for(
                    SimpleNamespace(forward_mode=ForwardMode.DECODE)
                )
            )
        )
        max_len = SimpleNamespace(
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
            forward_mode=ForwardMode.DECODE,
        )
        with (
            planning(parallel),
            patch_communicator(
                "get_forward",
                lambda: SimpleNamespace(sp_active=False, attn_input_scattered=False),
            ),
            patch_communicator("should_use_dp_reduce_scatterv", lambda: False),
            patch_communicator("can_use_dp_reduce_scatter", lambda: True),
        ):
            step = communicator.ffn.plan.output._dp_reduce_scatter_step(
                max_len, communicator.ffn.plan.output.plan.path_for(max_len)
            )
            self.assertTrue(
                communicator.ffn.plan.output._sum_in_reduce_scatter(
                    communicator.ffn.plan.output.plan.path_for(max_len), step
                )
            )

    def test_the_exit_runs_the_move_back_over_dp_it_chose(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        for name, reduce_scatterv, is_max_len, move in (
            (
                "SUM_LEN",
                True,
                False,
                transport_ops.dp_reduce_scatterv,
            ),
            (
                "MAX_LEN",
                False,
                True,
                transport_ops.dp_reduce_scatter,
            ),
        ):
            with self.subTest(name):
                hc_post = MagicMock(side_effect=lambda h, r, h_res, h_post: h + r)
                communicator = build_mhc(
                    layer_case(1, 3),
                    parallel,
                    hc_post=hc_post,
                )
                communicator.mhc.h_res = communicator.mhc.h_post = torch.zeros(2)
                forward_batch = SimpleNamespace(
                    dp_padding_mode=SimpleNamespace(is_max_len=lambda: is_max_len),
                    forward_mode=ForwardMode.DECODE,
                )
                choose = MagicMock(wraps=comm_exit._select_dp_reduce_scatter)
                to_local_tokens = MagicMock(side_effect=lambda step, fb, h: h[:2])
                with (
                    planning(parallel),
                    patch_communicator(
                        "should_use_dp_reduce_scatterv", lambda: reduce_scatterv
                    ),
                    patch_communicator("can_use_dp_reduce_scatter", lambda: True),
                    patch_communicator("_select_dp_reduce_scatter", choose),
                    patch_communicator("to_dp_local", to_local_tokens),
                ):
                    with communicator.ffn.plan.output.ffn_exit(
                        forward_batch, stream=ResidualStream()
                    ) as ffn_exit:
                        # The reduce-scatter on the way back completes the sum;
                        # the FFN never runs or skips it.
                        self.assertFalse(get_forward().mlp_reduce_scatter)
                    hidden, residual = finish_exit(
                        ffn_exit, torch.ones(4, 4), torch.full((2, 4), 2.0)
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
                layer_case(i, 3, sparse=sparse, previous_sparse=previous),
                parallel,
                a2a=True,
            )
            for i, (sparse, previous) in enumerate(
                ((False, False), (True, False), (True, True))
            )
        ]
        first_a2a, last = layers[1], layers[2]
        self.assert_step(
            first_a2a.ffn.plan.paths.get(BatchVariant.ORDINARY).entry.prepare,
            comm_ops._attn_tp_reduce_scatter_update_read,
            first_a2a,
            scatters_residual=True,
        )
        self.assert_step(
            last.ffn.plan.paths.get(BatchVariant.ORDINARY).entry.prepare,
            comm_ops._attn_tp_reduce_scatter_update_read,
            last,
            scatters_residual=False,
        )
        self.assert_step(
            last.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
            update_attn_tp_gather_output,
            last,
        )
        state = first_a2a.mhc
        state.h_res, state.h_post = torch.arange(4.0), torch.arange(4.0) + 10
        context = SimpleNamespace(attn_tp_rank=1, attn_tp_size=2)
        with patch_communicator("get_parallel", lambda: context):
            residual = state.slice_residual_attn_tp(torch.arange(4.0) + 20)
        self.assertEqual(residual.tolist(), [22.0, 23.0])
        self.assertEqual(state.h_res.tolist(), [2.0, 3.0])
        self.assertEqual(state.h_post.tolist(), [12.0, 13.0])

    def test_an_input_scattered_batch_keeps_the_residual_on_the_slice(self):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        for layer_id in range(3):
            with self.subTest(layer_id=layer_id):
                communicator = build_mhc(layer_case(layer_id, 3), parallel)
                steps = communicator.ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                self.assert_step(
                    steps.entry.prepare,
                    comm_ops._tp_reduce_scatter_update_read_gather,
                    communicator,
                )
                self.assert_step(
                    steps.output_move,
                    residual_slice_output,
                    communicator,
                    sums=True,
                    gathers_back=layer_id == 2,
                )
                self.assertTrue(steps.output_move_completes_sum)
                completes = build_mhc(
                    layer_case(layer_id, 3),
                    parallel,
                    boundary_reduction="ar",
                ).ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                self.assertFalse(completes.output_move.keywords["sums"])
                self.assertFalse(completes.output_move_completes_sum)
                # Only the first layer's input, the embedding's partial sum,
                # is completed onto the slice.
                self.assertIs(
                    communicator.attn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                    .entry.prepare.keywords["step"]
                    .keywords["pre_move"],
                    comm.tp_reduce_scatter if layer_id == 0 else None,
                )
                self.assertIsNone(
                    communicator.attn.plan.paths.get(
                        BatchVariant.INPUT_SCATTERED
                    ).entry.input_move
                )
                self.assertIs(
                    communicator.attn.plan.paths.get(
                        BatchVariant.INPUT_SCATTERED
                    ).entry.attn_input_adapter,
                    comm_ops._attn_input_scattered,
                )
                self.assertIs(
                    communicator.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
                    keep_output,
                )

    def test_two_batch_overlap_gathers_on_the_declarations(self):
        # The dense layer before a sparse one hands the split the attention's
        # rows: MHC writes its output into the streams, then gathers them.
        parallel = parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)
        communicator = build_mhc(
            layer_case(1, 3, next_layer_sparse=True), parallel, two_batch_overlap=True
        )
        self.assert_step(
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY).entry.prepare,
            comm_ops._attn_tp_reduce_scatter_update_read,
            communicator,
        )
        self.assert_step(
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
            update_attn_tp_gather_output,
            communicator,
        )

    def test_the_postprocess_writes_the_output_into_the_streams(self):
        # The last layer also contracts the streams into the hidden states.
        parallel = parallel_of(attn_dp=1, attn_tp=1)
        for layer_id, contracted in ((1, False), (2, True)):
            with self.subTest(layer_id=layer_id):
                communicator = build_mhc(
                    layer_case(layer_id, 3),
                    parallel,
                    hc_post=lambda h, r, h_res, h_post: h + r,
                )
                communicator.mhc.h_res = communicator.mhc.h_post = torch.zeros(2)
                with (
                    planning(parallel),
                    patch_communicator(
                        "get_forward",
                        lambda: SimpleNamespace(
                            sp_active=False, attn_input_scattered=False
                        ),
                    ),
                    patch.object(
                        mhc_module, "hc_contract", lambda h, hc_mult: h.sum(-1)
                    ),
                ):
                    hidden, residual = postprocess_output(
                        communicator.ffn.plan.output,
                        torch.ones(2, 4),
                        torch.full((2, 4), 2.0),
                        SimpleNamespace(forward_mode=ForwardMode.DECODE),
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
        facts = layer_case(1, 3, sparse=True, previous_sparse=False)
        with self.assertRaises(NotImplementedError):
            build_mhc(facts, parallel, dsa_cp=True)
        # A dense layer on every rank computes on its own shard: nothing moves.
        facts = layer_case(1, 3, sparse=False, previous_sparse=False)
        build_mhc(facts, parallel, dsa_cp=True)

    def test_input_scattered_attention_under_attention_cp_is_rejected(self):
        scattered = dict(enable_attn_tp_input_scattered=True)
        parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2, **scattered)
        with self.assertRaisesRegex(NotImplementedError, "input-scattered"):
            build_mhc(
                layer_case(1, 3, sparse=False, previous_sparse=False),
                parallel,
            )
        parallel = parallel_of(attn_dp=1, attn_tp=2, **scattered)
        build_mhc(
            layer_case(1, 3, sparse=False, previous_sparse=False),
            parallel,
        )

    def test_a_moe_gathered_over_moe_cp_is_rejected(self):
        # A MoE-CP gather (MoE DP narrower than CP), which MHC has not been run with.
        for prefill_cp in (True, False):
            with self.subTest(prefill_cp=prefill_cp):
                parallel = parallel_of(
                    attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=prefill_cp
                )
                facts = layer_case(1, 3, sparse=True, previous_sparse=True)
                with self.assertRaisesRegex(NotImplementedError, "MoE-CP group"):
                    build_mhc(facts, parallel)
        # A dense layer builds, and so does a MoE on its own CP shard (MoE DP = CP).
        parallel = parallel_of(attn_dp=1, attn_tp=2, attn_cp=2)
        build_mhc(
            layer_case(1, 3, sparse=False, previous_sparse=False),
            parallel,
        )
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, attn_cp=2, moe_dp_size=2, moe_tp_size=2
        )
        build_mhc(
            layer_case(1, 3, sparse=True, previous_sparse=True),
            parallel,
        )
        # CP-TP group sharing (Qwen4-Exp) runs MHC on the MoE-CP gather.
        parallel = parallel_of(
            attn_dp=1,
            attn_tp=1,
            attn_cp=2,
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
        )
        build_mhc(
            layer_case(1, 3, sparse=True, previous_sparse=True),
            parallel,
        )


class TestTwoBatchOverlap(CustomTestCase):
    """Two-batch overlap splits the attention's rows. A dense MLP on every rank
    hands the sparse layer after it those rows, and the split moves the input
    from the rows the first overlapped layer takes."""

    def layers(self, *, tbo):
        parallel = parallel_of(attn_dp=2, attn_tp=2, moe_dense_tp_size=1)
        overlap = SimpleNamespace(
            comm=SimpleNamespace(boundary_reduction="rs+rsv"),
            overlap=SimpleNamespace(enable_two_batch_overlap=tbo),
        )
        layers = []
        with planning(parallel), patch_communicator("get_exec", lambda: overlap):
            for i, sparse in enumerate((False, False, True, True)):
                facts = layer_case(
                    layer_id=i,
                    num_layers=4,
                    sparse=sparse,
                    previous_sparse=i > 2,
                    next_layer_sparse=i >= 1,
                )
                layers.append(
                    make_test_stages(**facts, attention_norm=Norm(), ffn_norm=Norm())
                )
        return layers

    def test_the_dense_layer_before_the_split_hands_on_the_attention_rows(self):
        attention = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
        local = comm.Layout(frozenset({TokenAxis.ATTN_DP, TokenAxis.ATTN_TP}))
        for tbo, gathered in ((True, True), (False, False)):
            with self.subTest(two_batch_overlap=tbo):
                first, before, after, last = self.layers(tbo=tbo)
                for layer in (first, before, after, last):
                    self.assertIsNotNone(layer.attn.plan.edges)
                self.assertIs(
                    first.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
                    keep_output,
                )
                if gathered:
                    self.assertIs(
                        before.ffn.plan.paths.get(
                            BatchVariant.ORDINARY
                        ).output_move.func,
                        update_attn_tp_gather_output,
                    )
                    self.assertEqual(after.attn.incoming_residual_rows, attention)
                    self.assertIsNone(
                        after.attn.plan.paths.get(
                            BatchVariant.ORDINARY
                        ).entry.input_move
                    )
                else:
                    self.assertIs(
                        before.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
                        keep_output,
                    )
                    self.assertEqual(after.attn.incoming_residual_rows, local)
                    self.assertIs(
                        after.attn.plan.paths.get(
                            BatchVariant.ORDINARY
                        ).entry.input_move,
                        attn_tp_gather_input,
                    )

    def test_the_split_moves_from_the_first_layer_s_rows(self):
        parallel = parallel_of(attn_dp=2, attn_tp=2)
        dp = frozenset({TokenAxis.ATTN_DP})
        with planning(parallel):
            self.assertEqual(
                comm.tbo_split_moves(comm.Layout(dp)),
                (keep_output, keep_output),
            )
            self.assertEqual(
                comm.tbo_split_moves(comm.Layout(dp | {TokenAxis.ATTN_TP})),
                (update_attn_tp_gather_output, attn_tp_slice_output),
            )
            with self.assertRaises(NotImplementedError):
                comm.tbo_split_moves(comm.Layout(frozenset()))


class TestTheAttentionOutputDecidesItsSum(CustomTestCase):
    """prepare_mlp completes the attention-TP sum exactly when the attention
    output's declaration says it is owed, whatever the attention-TP size."""

    def test_the_attention_output_owes_the_attention_tp_sum(self):
        for attn_tp in (1, 2):
            stages = build(layer_case(1, 3), parallel_of(attn_dp=2, attn_tp=attn_tp))
            produced = stages.attn.plan.paths.get(BatchVariant.ORDINARY).output
            self.assertIs(produced.group, SumGroup.ATTN_TP if attn_tp > 1 else None)
            self.assertIs(produced.always_partial, attn_tp > 1)
            self.assertIs(
                stages.ffn.plan.paths.get(BatchVariant.ORDINARY).output.group,
                SumGroup.TP,
            )
            self.assertFalse(
                stages.ffn.plan.paths.get(BatchVariant.ORDINARY).output.always_partial
            )

    def run_steps(self, produced, *, force=False):
        sizes = {
            TokenAxis.ATTN_DP: 2,
            TokenAxis.ATTN_CP: 1,
            TokenAxis.ATTN_TP: 2,
        }
        steps, _ = comm_boundary._select_entry_step(
            produced,
            residual=comm.Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes),
            residual_to=comm.Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes),
            need=comm.InputContract(
                comm.Layout(frozenset()),
                read=replace(NORM_READOUT, reads_before_dp_gather=force),
            ),
            is_plain_add=True,
            fusions=(),
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        reduced = []
        with (
            patch_communicator(
                "attention_tensor_model_parallel_all_reduce",
                lambda x: reduced.append(x) or 2 * x,
            ),
            patch_communicator("dp_gather", lambda h, fb, cp_shard_counts=None: h),
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
            )
        return len(reduced), hidden, residual

    def test_a_complete_output_is_not_summed_again(self):
        layout = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
        complete = comm.OutputContract(layout)
        for force in (False, True):
            with self.subTest(force=force):
                reductions, hidden, residual = self.run_steps(complete, force=force)
                self.assertEqual(reductions, 0)
                torch.testing.assert_close(residual, torch.full((1, HIDDEN), 10.0))
                torch.testing.assert_close(hidden, torch.full((1, HIDDEN), 20.0))

    def test_an_owed_output_is_summed_once(self):
        layout = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
        owed = comm.OutputContract(layout, group=SumGroup.ATTN_TP, always_partial=True)
        reductions, hidden, residual = self.run_steps(owed, force=True)
        self.assertEqual(reductions, 1)
        # The stand-in all-reduce doubles the one rank's value.
        torch.testing.assert_close(residual, torch.full((1, HIDDEN), 17.0))

    def test_a_sum_left_only_for_some_batches_comes_with_the_value(self):
        # Like a mixer exit: the steps take the output as complete, and a value
        # that carries the sum is completed before them.
        layout = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
        asked = comm.OutputContract(layout, group=SumGroup.ATTN_TP)
        reductions, hidden, residual = self.run_steps(asked)
        self.assertEqual(reductions, 0)
        torch.testing.assert_close(residual, torch.full((1, HIDDEN), 10.0))


class TestFusedKernelsTakeOnlyTheStepsTheyComplete(CustomTestCase):
    """A fused kernel replaces the attention-TP sum, the residual add and the
    norm only where those are the steps, and only if it completes the sum the
    attention output owes."""

    def fused(self, completes):
        return comm.FfnInputFusion(
            completes=completes,
            run=lambda h, r, fb: None,
        )

    def select(self, *, attn_dp, fusions):
        sizes = {
            TokenAxis.ATTN_DP: attn_dp,
            TokenAxis.ATTN_CP: 1,
            TokenAxis.ATTN_TP: 2,
        }
        attention = comm.Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes)
        steps, _ = comm_boundary._select_entry_step(
            comm.OutputContract(attention, group=SumGroup.ATTN_TP, always_partial=True),
            residual=comm.Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes),
            residual_to=comm.Layout.sharded_over(TokenAxis.ATTN_DP, axis_sizes=sizes),
            need=comm.InputContract(comm.Layout(frozenset())),
            is_plain_add=True,
            fusions=fusions,
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        return steps, steps.keywords.get("fusions", ())

    def test_a_kernel_over_another_group_is_not_chosen(self):
        over_tp = self.fused(SumGroup.TP)
        over_attention_tp = self.fused(SumGroup.ATTN_TP)
        steps, chosen = self.select(attn_dp=1, fusions=(over_tp, over_attention_tp))
        self.assertIs(steps.func, comm_ops._reduce_update_read)
        self.assertEqual(steps.keywords["fusions"], (over_attention_tp.run,))
        self.assertEqual(chosen, (over_attention_tp.run,))

    def test_no_kernel_is_chosen_where_rows_are_gathered(self):
        steps, chosen = self.select(attn_dp=2, fusions=(self.fused(SumGroup.ATTN_TP),))
        self.assertIs(steps.func, comm_ops._dp_gather_sum_read)
        self.assertEqual(chosen, ())


class TestTheSequenceParallelRegion(CustomTestCase):
    """While a LayerNorm SP region is active, the layer runs the steps its
    region declarations choose: the linears gather and reduce-scatter
    themselves, so every boundary stays on this rank's slice. Other batches
    take the layer's ordinary declared steps."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP: 2}

    def test_what_the_ffn_exit_reads_comes_from_the_batch_s_steps(self):
        # Inside the region the FFN output is complete: no sum to move.
        parallel = parallel_of(attn_dp=1, attn_tp=2)
        communicator = build(layer_case(1, 3), parallel, sp=True)
        for active in (False, True):
            with (
                self.subTest(sp_active=active),
                planning(parallel, sp=True),
                patch_communicator(
                    "get_forward",
                    lambda: SimpleNamespace(
                        sp_active=active, attn_input_scattered=False
                    ),
                ),
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
                patch_communicator("is_dp_attention_enabled", lambda: False),
            ):
                self.assertIs(
                    communicator.ffn.plan.output._sum_deferral_allowed(
                        communicator.ffn.plan.output.plan.path_for(
                            SimpleNamespace(forward_mode=ForwardMode.DECODE)
                        )
                    ),
                    not active,
                )


class TestInputScatteredAttention(CustomTestCase):
    """On a batch whose attention input is scattered over attention TP, a layer
    runs the steps that batch's declarations choose: its input arrives as a TP
    partial that a reduce-scatter completes onto each rank's slice, and the
    residual comes back to every row inside the attention output's sum."""

    SIZES = {TokenAxis.ATTN_DP: 1, TokenAxis.ATTN_CP: 1, TokenAxis.ATTN_TP: 2}

    def test_the_order_of_a_residual_on_the_slice_is_declared(self):
        # The same layouts take two orders: with a scattered input the residual
        # joins the attention output's sum; after a MoE on local rows it is
        # gathered first.
        sizes = self.SIZES
        attention = comm.Layout.sharded_over(axis_sizes=sizes)
        local = comm.Layout(frozenset({TokenAxis.ATTN_TP}))
        owed = comm.OutputContract(
            attention, group=SumGroup.ATTN_TP, always_partial=True
        )
        for joins, step in (
            (True, comm_ops._tp_sum_with_residual_read),
            (False, comm_ops._reduce_update_read),
        ):
            with self.subTest(residual_joins_sum=joins):
                steps, _ = comm_boundary._select_entry_step(
                    owed,
                    residual=local,
                    residual_to=attention,
                    need=comm.InputContract(attention),
                    is_plain_add=True,
                    fusions=(),
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
        local = comm.Layout(frozenset({TokenAxis.ATTN_TP}))
        step, _ = comm_boundary._select_entry_step(
            comm.OutputContract(attention),
            residual=attention,
            residual_to=local,
            need=comm.InputContract(local),
            is_plain_add=True,
            fusions=(),
            residual_joins_sum=False,
            cp_moves=None,
            enters_stack=False,
        )
        self.assertIs(step.func, comm_ops._attn_tp_slice_update_read)
        hidden = torch.arange(4.0)[:, None].expand(4, HIDDEN).clone()
        residual = torch.ones(4, HIDDEN)
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        with patch_communicator("get_parallel", lambda: context):
            out, out_residual = step(hidden, residual, None, Norm())
        torch.testing.assert_close(out_residual, hidden[2:] + 1)
        torch.testing.assert_close(out, 2 * (hidden[2:] + 1))
        # At the start of the layer stack the input is the residual.
        with patch_communicator("get_parallel", lambda: context):
            out, out_residual = step(hidden, None, None, Norm())
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
                    layer_case(1, 3, sparse=a2a, previous_sparse=a2a),
                    parallel,
                    a2a=a2a,
                )
                self.assertIs(
                    communicator.ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                    is not None,
                    expected,
                )

    def test_a_batch_runs_them_only_while_its_input_is_scattered(self):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True
        )
        communicator = build(layer_case(1, 3), parallel)
        for scattered in (False, True):
            with (
                self.subTest(input_scattered=scattered),
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=scattered),
                ),
                patch_communicator(
                    "get_forward",
                    lambda: SimpleNamespace(
                        sp_active=False, attn_input_scattered=False
                    ),
                ),
            ):
                self.assertIs(
                    communicator.ffn.plan.path_for(
                        SimpleNamespace(forward_mode=ForwardMode.DECODE)
                    ),
                    communicator.ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                    if scattered
                    else communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
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
        communicator = build(layer_case(1, 3), parallel)
        with (
            patch_communicator("get_parallel", lambda: parallel),
            patch_communicator(
                "get_attn_tp_context",
                lambda: SimpleNamespace(input_scattered=True),
            ),
            patch_communicator(
                "get_forward",
                lambda: SimpleNamespace(sp_active=False, attn_input_scattered=False),
            ),
        ):
            hidden, residual = prepare_input(
                communicator.attn,
                torch.ones(4, HIDDEN),
                torch.full((4, HIDDEN), 3.0),
                None,
            )
        self.assertEqual(scattered, [4])
        # Norm: (2 * (h + r), h + r) on the slice, with h the completed sum.
        torch.testing.assert_close(residual.residual, torch.full((2, HIDDEN), 5.0))
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
            with self.subTest(
                layer_id=layer_id, boundary_reduction="rs+rsv" if allow else "ar"
            ):
                communicator = build(
                    layer_case(layer_id, 3),
                    parallel,
                    boundary_reduction="rs+rsv" if allow else "ar",
                )
                steps = communicator.ffn.plan.paths.get(BatchVariant.INPUT_SCATTERED)
                self.assertIs(steps.output.may_reduce_scatter, hands_on)


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
        facts = layer_case(1, 3, sparse=True, previous_sparse=False)
        communicator = build(facts, parallel, dsa_cp=True)
        cp, ordinary = (
            communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL),
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
        )
        self.assertIs(
            cp.entry.prepare.keywords["step"].func,
            comm_ops._then_attn_cp_gather,
        )
        self.assertIs(
            cp.entry.prepare.keywords["step"].keywords["gather"].func,
            comm_ops._update_read,
        )
        self.assertIs(
            cp.output_move,
            attn_cp_reduce_scatter_output,
        )
        self.assertTrue(cp.output.may_reduce_scatter)
        self.assertFalse(cp.output.may_defer_to_next)
        # The dense layer before it ran on the same shard: nothing to move.
        self.assertIsNone(
            communicator.attn.plan.paths.get(
                BatchVariant.CONTEXT_PARALLEL
            ).entry.input_move
        )
        # Other batches hold every token on each CP rank: the MoE sums itself.
        self.assertIs(
            ordinary.entry.prepare.keywords["step"].func, comm_ops._update_read
        )
        self.assertIs(ordinary.output_move, keep_output)
        with (
            patch_communicator(
                "get_forward",
                lambda: SimpleNamespace(sp_active=False, attn_input_scattered=False),
            ),
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
                        communicator.ffn.plan.output._sum_in_reduce_scatter(
                            communicator.ffn.plan.output.plan.path_for(
                                SimpleNamespace(forward_mode=ForwardMode.DECODE)
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
        facts = layer_case(1, 3, sparse=False, previous_sparse=False)
        communicator = build(facts, parallel, dsa_cp=True)
        for steps in (
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
            communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL),
        ):
            self.assertIs(
                steps.entry.prepare.keywords["step"].func, comm_ops._update_read
            )
            self.assertIsNone(steps.output.group)
            self.assertIs(steps.output_move, keep_output)

    def test_a_cp_extend_gathers_over_cp_and_takes_its_chunk_back(self):
        communicator = build(layer_case(1, 3), self.cp_parallel())
        cp = communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL)
        self.assertIs(
            cp.entry.prepare.keywords["step"].func, comm_ops._then_moe_cp_gather
        )
        # Each rank completes its own chunk before the gather.
        self.assertIs(
            cp.entry.prepare.keywords["step"].keywords["gather"].func,
            comm_ops._reduce_update_read,
        )
        self.assertIs(
            cp.output_move,
            moe_cp_take_back_output,
        )
        self.assertFalse(cp.returns_over_dp)
        self.assertFalse(cp.output.may_defer_to_next)
        self.assertFalse(cp.output.may_reduce_scatter)
        self.assertIs(
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY)
            .entry.prepare.keywords["step"]
            .func,
            comm_ops._reduce_update_read,
        )
        self.assertIs(
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY).output_move,
            keep_output,
        )

    def test_a_moe_on_its_own_cp_shard_hands_on_its_sum(self):
        # MoE DP equal to CP: each CP rank's MoE runs on its own TP ranks.
        parallel = self.cp_parallel(moe_dp_size=2, moe_tp_size=2)
        communicator = build(
            layer_case(1, 3, sparse=True, previous_sparse=True),
            parallel,
        )
        for steps in (
            communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL),
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
        ):
            # The attention's rows are the MoE's: nothing to gather or take back.
            self.assertIs(
                steps.entry.prepare.keywords["step"].func,
                comm_ops._reduce_update_read,
            )
            self.assertIs(steps.output_move, keep_output)
            self.assertTrue(steps.output.may_defer_to_next)
        dense = build(layer_case(1, 3), parallel)
        self.assertIs(
            dense.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL)
            .entry.prepare.keywords["step"]
            .func,
            comm_ops._then_moe_cp_gather,
        )
        self.assertFalse(
            dense.ffn.plan.paths.get(
                BatchVariant.CONTEXT_PARALLEL
            ).output.may_defer_to_next
        )

    def test_the_fused_kernels_run_on_each_chunk(self):
        with planning(self.cp_parallel()):
            communicator = make_test_stages(
                **layer_case(1, 3), attention_norm=FusableNorm(), ffn_norm=FusableNorm()
            )
        chunk = (
            communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL)
            .entry.prepare.keywords["step"]
            .keywords["gather"]
        )
        self.assertEqual(len(chunk.keywords["fusions"]), 1)
        self.assertIs(chunk.keywords["fusions"][0].func, fused_ffn_input)
        self.assertEqual(chunk.keywords["fusions"][0].args, (communicator.ffn.plan,))

    def cp_outcome(self, parallel, *, sparse=True, a2a=False, dsa_cp=False):
        """A layer under attention CP: "cp steps" for the batches that shard
        their tokens, "ordinary" when no batch does, or "refused"."""
        facts = layer_case(1, 3, sparse=sparse, previous_sparse=sparse)
        try:
            layer = build(facts, parallel, a2a=a2a, dsa_cp=dsa_cp)
        except NotImplementedError:
            return "refused"
        return (
            "ordinary"
            if layer.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL) is None
            else "cp steps"
        )

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
        communicator = build(layer_case(1, 3), parallel)
        cp, ordinary = (
            communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL),
            communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
        )
        self.assertIs(
            cp.entry.prepare.keywords["step"].func, comm_ops._dp_gather_sum_read
        )
        self.assertTrue(cp.entry.prepare.keywords["step"].keywords["places_cp_shards"])
        self.assertIs(cp.output_move, dp_cp_take_back_output)
        self.assertIs(
            ordinary.entry.prepare.keywords["step"].func, comm_ops._dp_gather_sum_read
        )
        self.assertFalse(
            ordinary.entry.prepare.keywords["step"].keywords["places_cp_shards"]
        )
        self.assertTrue(ordinary.returns_over_dp)
        for steps in (cp, ordinary):
            self.assertFalse(steps.output.may_defer_to_next)
            self.assertFalse(steps.output.may_reduce_scatter)
            self.assertFalse(steps.output.may_reduce_scatterv)

    def test_a_batch_runs_them_only_on_a_cp_extend(self):
        communicator = build(layer_case(1, 3), self.cp_parallel())

        def batch(cp_extend):
            return SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_context_parallel_extend=lambda: cp_extend
                )
            )

        def unread(fb):
            raise AssertionError("read for a batch that is not a CP extend")

        for fb, rows, expected in (
            (
                batch(False),
                unread,
                communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
            ),
            (
                batch(True),
                lambda fb: None,
                communicator.ffn.plan.paths.get(BatchVariant.ORDINARY),
            ),
            (
                batch(True),
                lambda fb: [2, 1],
                communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL),
            ),
        ):
            with (
                self.subTest(
                    expected=expected
                    is communicator.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL)
                ),
                planning(self.cp_parallel()),
                patch_communicator("moe_cp_gathered_rows", rows),
                patch_communicator(
                    "get_forward",
                    lambda: SimpleNamespace(
                        sp_active=False, attn_input_scattered=False
                    ),
                ),
                patch_communicator(
                    "get_attn_tp_context",
                    lambda: SimpleNamespace(input_scattered=False),
                ),
            ):
                self.assertIs(communicator.ffn.plan.path_for(fb), expected)


class TestBranchRows(CustomTestCase):
    """Two FFNs that branch from one input and merge again (LongCat's MoE and
    dense branch): each layer's communicator gives the rows of its FFN input,
    of its residual while the FFN runs and of what it hands on, and a complete
    value moves between them."""

    local = comm.Layout(frozenset({TokenAxis.ATTN_DP, TokenAxis.ATTN_TP}))
    attention = comm.Layout(frozenset({TokenAxis.ATTN_DP}))
    full = comm.Layout(frozenset())

    def test_branches_move_between_the_declared_rows(self):
        # An a2a MoE beside a dense FFN, under attention DP and TP.
        class Communicator(SimpleNamespace):
            branch_input = branch.branch_input
            branch_output = branch.branch_output
            merge_branch = branch.merge_branch

        moe = Communicator(
            branch_rows=lambda fb: (self.local, self.local, self.local),
            # The a2a combine already summed the MoE output.
            path_for=lambda fb: SimpleNamespace(output=comm.OutputContract(self.local)),
        )
        dense = Communicator(
            branch_rows=lambda fb: (self.full, self.attention, self.attention)
        )
        moves = []

        def recorded(value, rows, to, forward_batch):
            moves.append((value, rows, to))
            return value

        with patch_communicator("move_rows", recorded):
            dense.branch_input(moe, "h0", ResidualStream("residual"), None)
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
            stream = ResidualStream("residual")
            hidden = stream.record(2, comm.PLAIN_ADD)
            merged = moe.merge_branch(1, hidden, stream, dense, None)
            self.assertEqual(
                moves,
                [
                    (2, self.attention, self.local),
                    ("residual", self.attention, self.local),
                ],
            )
            # The contribution adds; the residual is the dense branch's.
            self.assertEqual(merged[0], 3)
            self.assertIs(merged[1], stream)
            self.assertEqual(stream.residual, "residual")

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
            patch_communicator("attn_tp_gather", tp_gather),
            patch_communicator("dp_gather", lambda h, fb: dp_gather(h)),
            patch_communicator("to_dp_local", lambda step, fb, h: dp_scatter(h)),
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
                (comm.Layout(frozenset({TokenAxis.ATTN_TP})), self.attention),
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
def running(*, reduce_scatterv, a2a=False, use_reduce_scatter=True):
    replaced = {
        "get_forward": lambda: state().flags,
        "attention_tensor_model_parallel_all_reduce": lambda x: (
            state().parallel.attn_tp_group.all_reduce(x)
        ),
        "tensor_model_parallel_all_reduce": lambda x: (
            state().parallel.tp_group.all_reduce(x)
        ),
        "sum_post_experts_output": lambda x: (
            comm_layout.post_experts_reduction_group().all_reduce(x)
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
        "flashinfer_ar_fusion_applies": lambda n: False,
        "post_experts_output_is_complete": lambda **kw: False,
        "post_experts_reduction_group": lambda: state().parallel.tp_group,
        "get_lora": lambda: SimpleNamespace(enable_lora=False),
        "get_exec": lambda: SimpleNamespace(
            comm=SimpleNamespace(
                enable_quant_communications=False,
                boundary_reduction="rs+rsv" if use_reduce_scatter else "ar",
            )
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
    """A dense MLP on the TP group: 5 * x, as this rank's partial sum; the
    stage boundary completes it."""
    return 5 * x * WEIGHTS[state.parallel.tp_size][state.parallel.tp_rank]


def moe(x, state):
    """A MoE block not dispatched per DP shard: 5 * x, as this rank's partial
    sum over the MoE output's group; the stage boundary completes it."""
    return 5 * x * WEIGHTS[state.parallel.tp_size][state.parallel.tp_rank]


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
        use_reduce_scatter,
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
                make_test_stages(
                    **layer_case(
                        i, 2, sparse=sparse[i], previous_sparse=i > 0 and sparse[i - 1]
                    ),
                    attention_norm=Norm(),
                    ffn_norm=Norm(),
                )
                for i in range(2)
            ]
            hidden = torch.zeros(s.local_rows, HIDDEN).double()
            hidden[: s.rows] = embeddings[s.dp]
            residual = None
            handed_on = []
            for layer_index, layer in enumerate(layers):
                hidden, residual = prepare_input(
                    layer.attn, hidden, residual, forward_batch
                )
                hidden = attention(hidden, s)
                hidden = residual.record(
                    hidden,
                    comm.PLAIN_ADD,
                    declared_sum=layer.ffn.plan.path_for(
                        forward_batch
                    ).entry.declared_sum,
                )
                hidden, residual = prepare_input(
                    layer.ffn, hidden, residual, forward_batch
                )
                with layer.ffn.plan.output.ffn_exit(
                    forward_batch, stream=ResidualStream()
                ) as ffn_exit:
                    if not sparse[layer_index]:
                        hidden = dense_mlp(hidden, s)
                    elif a2a:
                        # On this rank's slice; the combine completes the sum.
                        hidden = 5 * hidden
                    else:
                        hidden = moe(hidden, s)
                hidden, residual = finish_exit(ffn_exit, hidden, residual)
                handed_on.append(type(hidden))
            hidden, residual = residual.export(hidden)
            # The last a2a layer folds the residual into its output.
            if residual is not None:
                residual = residual[: s.rows]
            return hidden[: s.rows], residual, handed_on

        with running(
            reduce_scatterv=reduce_scatterv,
            a2a=a2a,
            use_reduce_scatter=use_reduce_scatter,
        ):
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
        for (
            attn_dp,
            attn_tp,
        ), rows, padding, use_reduce_scatter, sparse in itertools.product(
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
                    use_reduce_scatter=use_reduce_scatter,
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
                        use_reduce_scatter=use_reduce_scatter,
                        sparse=sparse,
                        a2a=a2a,
                    )
                    first_layer_hands_on = results[0][2][0]
                    # Disabling optional RS still permits an ordinary AR at
                    # the next boundary. The actual owed work travels with the value.
                    # Without attention DP an empty batch completes its own sum.
                    # A MoE dispatched per DP shard hands on a complete value.
                    # Otherwise the reduce-scatter back to this rank's tokens
                    # can be left to the next layer; the all-reduce only
                    # without an a2a backend.
                    has_tokens = attn_dp > 1 or sum(token_rows) > 0
                    reduce_scatter_left = (
                        use_reduce_scatter
                        and attn_dp > 1
                        and (padding == "max_len" or reduce_scatterv)
                    )
                    owes = (
                        has_tokens
                        and not (sparse[0] and a2a)
                        and (reduce_scatter_left or not a2a)
                    )
                    self.assertIs(
                        first_layer_hands_on,
                        OwedOutput if owes else torch.Tensor,
                    )


class TestTheFfnInputReduction(CustomTestCase):
    """With quantized communications a prefill reduces the FFN input quantized
    for a plain residual; MHC sums its streams in full precision."""

    def test_only_a_plain_residual_reduces_quantized(self):
        exec_ = SimpleNamespace(
            comm=SimpleNamespace(
                enable_quant_communications=True, boundary_reduction="rs+rsv"
            )
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False)
        )
        for is_plain_add, reduction in ((True, "quant"), (False, "full")):
            with self.subTest(is_plain_add=is_plain_add):
                update = SimpleNamespace(is_plain_add=is_plain_add)
                read = SimpleNamespace(
                    update_and_read=lambda update, h, r, norm, **read_kwargs: (h, r)
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
                    comm_ops._reduce_update_read(
                        torch.ones(2, HIDDEN),
                        torch.ones(2, HIDDEN),
                        batch,
                        None,
                        gathers_residual=False,
                        fusions=(),
                        read=read,
                        update=update,
                    )
                self.assertEqual(calls, [reduction])


if __name__ == "__main__":
    unittest.main()
