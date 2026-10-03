"""Local MoE routing with CP gathering owned by compute and return by the boundary."""

import itertools
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import TokenAxis
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.factories import (
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.ops import (
    keep_output,
    moe_cp_gather,
    moe_cp_reduce_scatter_output,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_RESIDUAL_OPS
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingGroup(fixture.Group):
    def all_reduce(self, value):
        self.world.state().events.append("all_reduce")
        return super().all_reduce(value)

    def reduce_scatter_tensor(self, output, value):
        self.world.state().events.append("reduce_scatter")
        return super().reduce_scatter_tensor(output, value)

    def all_gather_into_tensor(self, output, value):
        self.world.state().events.append("gather")
        return super().all_gather_into_tensor(output, value)


def parallel_of(**overrides):
    return fixture.parallel_of(
        attn_dp=1, attn_tp=1, attn_cp=2, enable_prefill_cp=True, **overrides
    )


def build(residual=PLAIN_RESIDUAL_OPS, *, output_transform=None, with_following=False):
    norm = fixture.Norm() if residual is PLAIN_RESIDUAL_OPS else None
    stages = (
        (declare_attn(read=residual.attn_readout, update=residual.attn_update), norm),
        (
            declare_ffn(
                sparse=True,
                gathers_cp_input=True,
                read=residual.ffn_readout,
                update=residual.ffn_update,
                output_transform=output_transform,
            ),
            norm,
        ),
    )
    if with_following:
        stages += (
            (
                declare_attn(read=residual.attn_readout, update=residual.attn_update),
                norm,
            ),
        )
    return make_stages(*stages)


@contextmanager
def planning(parallel, group, *, policy="rs+rsv", a2a=False, dsa_cp=False):
    with (
        fixture.planning(parallel, boundary_reduction=policy, a2a=a2a, dsa_cp=dsa_cp),
        patch_communicator("get_moe_cp_group", lambda: group),
    ):
        yield


def batch(rows, *, cp=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_context_parallel_extend=lambda: cp, is_decode_or_idle=lambda: not cp
        ),
        attn_cp_metadata=SimpleNamespace(per_rank_actual_token=rows) if cp else None,
    )


class TestComputeOwnedCpMoe(CustomTestCase):
    def test_local_entry_and_policy_controlled_return(self):
        parallel = parallel_of()
        group = parallel.tp_group
        for policy in ("ar", "rs", "rsv", "rs+rsv"):
            with self.subTest(policy=policy), planning(parallel, group, policy=policy):
                _, ffn = build()
                cp = ffn.plan.paths[BatchVariant.CONTEXT_PARALLEL]
                ordinary = ffn.plan.paths[BatchVariant.ORDINARY]
                self.assertIn(TokenAxis.ATTN_CP, cp.entry.input_rows.sharded)
                self.assertTrue(cp.output.always_partial)
                self.assertFalse(ordinary.output.always_partial)
                self.assertEqual(
                    cp.output_move_completes_sum, policy in ("rs", "rs+rsv")
                )
                self.assertFalse(cp.output.may_defer_to_next)
                if policy in ("rs", "rs+rsv"):
                    self.assertIs(cp.output_move, moe_cp_reduce_scatter_output)
                self.assertIs(ordinary.output_move, keep_output)
                self.assertFalse(ordinary.output_move_completes_sum)
                self.assertNotIn(TokenAxis.ATTN_CP, ordinary.entry.input_rows.sharded)

    def test_only_cp_batches_offer_cp_local_input(self):
        parallel = parallel_of()
        flags = fixture.Flags()
        with (
            planning(parallel, parallel.tp_group),
            patch_communicator("get_forward", lambda: flags),
            patch_communicator(
                "get_attn_tp_context", lambda: SimpleNamespace(input_scattered=False)
            ),
        ):
            _, ffn = build()
            for cp, metadata, expected in (
                (True, True, BatchVariant.CONTEXT_PARALLEL),
                (True, False, BatchVariant.ORDINARY),
                (False, True, BatchVariant.ORDINARY),
            ):
                fb = batch([3, 1], cp=cp)
                if not metadata:
                    fb.attn_cp_metadata = None
                with self.subTest(cp=cp, metadata=metadata):
                    self.assertIs(ffn.entry(fb), ffn.plan.paths[expected].entry)
                    self.assertEqual(
                        TokenAxis.ATTN_CP in ffn.entry(fb).input_rows.sharded,
                        expected is BatchVariant.CONTEXT_PARALLEL,
                    )

    def test_rejects_incompatible_topology(self):
        with self.assertRaisesRegex(ValueError, "sparse FFN"):
            declare_ffn(gathers_cp_input=True)
        for fields, options in (
            ({"moe_ep_size": 2, "moe_tp_size": 1}, {}),
            ({"moe_dp_size": 2, "moe_tp_size": 1}, {}),
            ({"moe_tp_size": 1}, {}),
            ({"dwdp_size": 2}, {}),
            ({}, {"a2a": True}),
            ({}, {"dsa_cp": True}),
            ({"attn_cp_group": SimpleNamespace(ranks=[1, 0])}, {}),
        ):
            with self.subTest(fields=fields, options=options):
                parallel = parallel_of(**fields)
                with planning(parallel, parallel.tp_group, **options):
                    with self.assertRaises(NotImplementedError):
                        build()
        parallel = parallel_of()
        with planning(parallel, SimpleNamespace(ranks=[1, 0])):
            with self.assertRaisesRegex(NotImplementedError, "ordered"):
                build()
        with (
            planning(parallel, parallel.tp_group),
            patch_communicator(
                "post_experts_reduction_group", lambda: SimpleNamespace(ranks=[1, 0])
            ),
        ):
            with self.assertRaisesRegex(NotImplementedError, "ordered"):
                build()

    def run_world(
        self,
        rows,
        policy,
        *,
        mhc=False,
        cp=True,
        complete_output=False,
        square_output=False,
        with_following=False,
    ):
        world = fixture.World(2)
        group = RecordingGroup(world, "cp", range(2))
        states = [
            SimpleNamespace(
                rank=rank,
                calls=[],
                events=[],
                flags=fixture.Flags(),
                parallel=parallel_of(
                    tp_rank=rank,
                    attn_cp_rank=rank,
                    tp_group=group,
                    attn_cp_group=group,
                ),
            )
            for rank in range(2)
        ]

        def run_rank(rank):
            state = world.state()
            nrows = rows[rank]
            # Ordinary batches replicate the same rows on every CP rank.
            value = torch.arange(nrows * 4, dtype=torch.double).reshape(nrows, 4)
            value = value + (10 * rank if cp else 0)
            residual = value + 1
            if mhc:
                residual = torch.stack((residual, residual + 2), dim=1)

                def post(hidden, streams, h_res, h_post):
                    self.assertEqual(hidden.shape[0], nrows)
                    self.assertEqual(streams.shape[0], nrows)
                    self.assertEqual(h_res.shape[0], nrows)
                    state.events.append("post")
                    return streams + hidden[:, None, :] * h_post[:, :, None]

                def pre(streams, *unused):
                    self.assertEqual(streams.shape[0], nrows)
                    return (
                        streams.sum(1),
                        torch.ones(nrows, 2, dtype=torch.double),
                        torch.full((nrows, 2), 2.0, dtype=torch.double),
                        False,
                    )

                mhc_state = MHCState(2, pre, pre, post)
                mhc_state.h_res = torch.ones(nrows, 2, dtype=torch.double)
                mhc_state.h_post = torch.ones(nrows, 2, dtype=torch.double)
                residual_ops = mhc_state.residual_ops()
                expected_residual = residual + value[:, None, :]
                expected_input = expected_residual.sum(1)
            else:
                residual_ops = PLAIN_RESIDUAL_OPS
                expected_residual = residual + value
                expected_input = 2 * expected_residual

            def square(value):
                state.events.append("transform")
                return value.square()

            attention, ffn, *following = build(
                residual_ops,
                output_transform=OutputTransform(square) if square_output else None,
                with_following=with_following,
            )
            fb = batch(rows, cp=cp)
            fb.residual_stream = ResidualStream(residual)
            value = attention.finish(value, fb)
            local = ffn.prepare(value, fb)
            stream = fb.residual_stream
            torch.testing.assert_close(local, expected_input, rtol=0, atol=0)
            self.assertNotIn("gather", state.events, "routing must see local rows")
            routing = local.sum(-1, keepdim=True)
            with ffn.exit(fb) as exit_:
                skip = state.flags.mlp_reduce_scatter
                if cp:
                    gathered = moe_cp_gather(local, rows, 2)
                    routed = moe_cp_gather(routing, rows, 2)
                else:
                    gathered, routed = local, routing
                output = 5 * gathered + routed
                if not complete_output:
                    output *= fixture.WEIGHTS[2][rank]
                if not cp and not skip:
                    output = group.all_reduce(output)
                if cp:
                    self.assertNotIn("all_reduce", state.events)
                    self.assertNotIn("reduce_scatter", state.events)
            self.assertFalse(state.flags.mlp_reduce_scatter)
            output = (
                ffn.finish_complete_output(output, fb)
                if complete_output
                else exit_.finish(output)
            )
            expected = 5 * expected_input + expected_input.sum(-1, keepdim=True)
            if square_output:
                expected = expected.square()
            if mhc:
                expected = expected_residual + 2 * expected[:, None, :]
                self.assertIsNone(mhc_state.h_res)
                self.assertIsNone(mhc_state.h_post)
                self.assertIsNone(stream.pending)
                self.assertEqual(state.events.count("post"), 2)
                self.assertEqual(state.events[-1], "post")
            else:
                torch.testing.assert_close(stream.residual, expected_residual)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertEqual(
                skip, cp and policy in ("rs", "rs+rsv") and not square_output
            )
            self.assertEqual(state.events.count("gather"), 2 if cp else 0)
            self.assertEqual(
                state.events.count("all_reduce"),
                int(not skip and not complete_output and (not cp or max(rows) > 0)),
            )
            self.assertEqual(
                state.events.count("reduce_scatter"),
                int(skip and not complete_output and max(rows) > 0),
            )
            if square_output and not complete_output and (not cp or max(rows) > 0):
                self.assertLess(
                    state.events.index("all_reduce"), state.events.index("transform")
                )
            if following:
                events = list(state.events)
                next_input = following[0].prepare(output, fb)
                torch.testing.assert_close(
                    next_input, 2 * (expected_residual + expected), rtol=0, atol=0
                )
                self.assertEqual(
                    state.events, events, "the exit already completed the sum"
                )

        with ExitStack() as stack:
            stack.enter_context(
                planning(lambda: world.state().parallel, group, policy=policy)
            )
            for name, value in {
                "get_forward": lambda: world.state().flags,
                "get_moe_cp_rank": lambda: world.state().rank,
                "moe_cp_all_gather_into_tensor": group.all_gather_into_tensor,
                "get_attn_tp_context": lambda: SimpleNamespace(input_scattered=False),
                "is_dp_attention_enabled": lambda: False,
                "post_experts_sum_is_one_all_reduce": lambda: True,
                "use_symmetric_memory": lambda *a, **kw: nullcontext(),
                "is_allocation_symmetric": lambda: False,
            }.items():
                stack.enter_context(patch_communicator(name, value))
            _, errors = world.run(states, run_rank)
        for error in errors:
            if error is not None and not isinstance(
                error, fixture.threading.BrokenBarrierError
            ):
                raise error
        self.assertEqual(errors, [None, None])

    def test_rank_major_padded_output_and_local_residual(self):
        for rows, policy, mhc in itertools.product(
            ([3, 3], [3, 1], [0, 2], [0, 0]),
            ("ar", "rs", "rsv", "rs+rsv"),
            (False, True),
        ):
            with self.subTest(rows=rows, policy=policy, mhc=mhc):
                self.run_world(rows, policy, mhc=mhc)

    def test_ordinary_batches_preserve_allreduce(self):
        for policy, mhc in itertools.product(("ar", "rs+rsv"), (False, True)):
            with self.subTest(policy=policy, mhc=mhc):
                self.run_world([2, 2], policy, mhc=mhc, cp=False)

    def test_complete_output_skips_boundary_reduction(self):
        for policy, mhc in itertools.product(("ar", "rs"), (False, True)):
            with self.subTest(policy=policy, mhc=mhc):
                self.run_world([3, 1], policy, mhc=mhc, complete_output=True)

    def test_nonlinear_transform_follows_sum_before_mhc_update(self):
        for policy, mhc in itertools.product(("ar", "rs"), (False, True)):
            with self.subTest(policy=policy, mhc=mhc):
                self.run_world([3, 1], policy, mhc=mhc, square_output=True)

    def test_following_attention_does_not_reduce_again(self):
        for policy in ("ar", "rs"):
            with self.subTest(policy=policy):
                self.run_world([3, 1], policy, with_following=True)

    def test_rejects_inconsistent_padded_row_metadata(self):
        parallel = parallel_of()
        group = SimpleNamespace(world_size=2, rank_in_group=0)
        for rows, count in ((None, 4), ([2], 4), ([-1, 2], 4), ([2, 1], 3)):
            with self.subTest(rows=rows, count=count), planning(parallel, group):
                with self.assertRaisesRegex(ValueError, "MoE-CP output"):
                    moe_cp_reduce_scatter_output(
                        torch.ones(count, 4), None, batch(rows)
                    )


if __name__ == "__main__":
    unittest.main()
