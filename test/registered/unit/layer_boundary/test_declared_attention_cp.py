"""A MoE layer on the TP group under DSA (and MLA) prefill CP, rank by rank.

Two CP ranks build the layer's communicator for real and run prepare_mlp and
the FFN exit on a CP extend on CPU: the FFN input is gathered over attention CP
in equal shards, and the output comes back to each rank's shard. The
attention-CP all-gather and reduce-scatter are replaced by what the ranks hand
to them.
"""

import contextlib
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.cp import interleave as dsa_cp
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.boundary_fixtures import finish_exit, make_test_stages, prepare_input
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 4
CP_SIZE = 2
# Rows per CP shard: DSA and MLA CP shard a CP extend in equal lengths.
ROWS = 3
# Binary fractions, so the partial sums add back to the value exactly.
PARTIAL_WEIGHTS = [0.25, 0.75]


def layernorm(hidden_states, residual=None):
    """Norm as the identity, keeping the fused add of the two-argument form."""
    if residual is None:
        return hidden_states.clone()
    summed = hidden_states + residual
    return summed, summed.clone()


class Flags:
    def __init__(self):
        self.fuse_mlp_allreduce = False
        self.mlp_reduce_scatter = False
        self.defer_moe_finalize = False
        self.sp_active = False

    @contextlib.contextmanager
    def scoped(self, **flags):
        saved = {k: getattr(self, k) for k in flags}
        self.__dict__.update(flags)
        try:
            yield
        finally:
            self.__dict__.update(saved)


def group(name, ranks):
    return SimpleNamespace(name=name, ranks=list(ranks))


class TestAttentionCpBoundary(CustomTestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.values = [
            torch.randint(-8, 8, (ROWS, HIDDEN), generator=generator).double()
            for _ in range(CP_SIZE)
        ]
        self.residuals = [
            torch.randint(-8, 8, (ROWS, HIDDEN), generator=generator).double()
            for _ in range(CP_SIZE)
        ]

    @contextmanager
    def as_rank(self, cp, collectives):
        """The first attention-TP lane of CP rank ``cp``, with fake CP collectives.

        Other attention-TP lanes hold identical rows after attention's sum.
        """
        attn_tp_size = getattr(self, "attn_tp_size", 1)
        tp_size = CP_SIZE * attn_tp_size
        tp_rank = cp * attn_tp_size
        moe_tp_size = getattr(self, "moe_tp_size", tp_size)
        moe_start = tp_rank // moe_tp_size * moe_tp_size
        moe_group = group("moe", range(moe_start, moe_start + moe_tp_size))
        parallel = SimpleNamespace(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_dp_enabled=False,
            attn_tp_size=attn_tp_size,
            attn_tp_rank=0,
            attn_cp_size=CP_SIZE,
            attn_cp_rank=cp,
            enable_prefill_cp=True,
            moe_dense_tp_size=1 if getattr(self, "sparse", True) else None,
            moe_dp_size=1,
            moe_ep_size=1,
            moe_tp_size=moe_tp_size,
            dwdp_size=1,
            enable_attn_tp_input_scattered=False,
            tp_group=group("tp", range(tp_size)),
            attn_tp_group=group("attn_tp", range(tp_rank, tp_rank + attn_tp_size)),
            attn_cp_group=group("attn_cp", range(0, tp_size, attn_tp_size)),
        )
        flags = Flags()
        with ExitStack() as stack:
            for target, value in [
                ((comm, "get_parallel"), lambda: parallel),
                ((dsa_cp, "get_parallel"), lambda: parallel),
                ((comm, "is_dsa_enable_prefill_cp"), lambda: True),
                ((comm, "is_mla_cp_enabled"), lambda: False),
                (
                    (comm, "dsa_use_prefill_cp"),
                    lambda fb: fb.forward_mode.is_context_parallel_extend(),
                ),
                ((comm, "is_mla_cp_active"), lambda fb: False),
                ((comm, "is_moe_input_scattered_across_dp_ranks"), lambda: False),
                ((comm, "is_enable_moe_cp_allgather"), lambda: False),
                ((comm, "get_moe_cp_size"), lambda: CP_SIZE),
                ((comm, "get_moe_cp_rank"), lambda: cp),
                ((comm, "should_use_dp_reduce_scatterv"), lambda: False),
                (
                    (comm, "get_moe_a2a_backend"),
                    lambda: SimpleNamespace(is_none=lambda: True),
                ),
                (
                    (comm, "post_experts_reduction_group"),
                    lambda: moe_group,
                ),
                (
                    (comm, "get_exec"),
                    lambda: SimpleNamespace(
                        overlap=SimpleNamespace(enable_two_batch_overlap=False)
                    ),
                ),
                (
                    (comm, "get_spec"),
                    lambda: SimpleNamespace(speculative_algorithm=None),
                ),
                ((comm, "get_forward"), lambda: flags),
                (
                    (comm, "get_attn_tp_context"),
                    lambda: SimpleNamespace(input_scattered=False),
                ),
                ((layernorm_sp, "layernorm_sp_enabled"), lambda: False),
                ((comm, "use_symmetric_memory"), lambda *a, **k: nullcontext()),
                ((comm, "is_allocation_symmetric"), lambda: False),
                (
                    (dsa_cp, "use_symmetric_memory"),
                    lambda *a, **k: nullcontext(),
                ),
                ((dsa_cp, "is_allocation_symmetric"), lambda: False),
                ((dsa_cp, "attn_cp_all_gather_into_tensor"), collectives["gather"]),
                (
                    (comm, "attn_cp_reduce_scatter_tensor"),
                    collectives["reduce_scatter"],
                ),
            ]:
                stack.enter_context(
                    patch_communicator(target[1], value)
                    if target[0] is comm
                    else patch.object(*target, value)
                )
            yield SimpleNamespace(parallel=parallel, flags=flags)

    def build(self, use_reduce_scatter):
        # A MoE layer after a dense one; dense layers run on every rank here.

        with patch_communicator(
            "get_exec",
            lambda: SimpleNamespace(
                comm=SimpleNamespace(
                    boundary_reduction="rs+rsv" if use_reduce_scatter else "ar"
                ),
                overlap=SimpleNamespace(enable_two_batch_overlap=False),
            ),
        ):
            return make_test_stages(
                first=False,
                last=False,
                sparse=getattr(self, "sparse", True),
                previous_sparse=False,
                next_layer_sparse=False,
                attention_norm=layernorm,
                ffn_norm=layernorm,
            )

    def cp_extend(self):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True)
        )

    def run_ranks(self, use_reduce_scatter):
        """Gather on each rank, run an FFN that leaves a partial sum or not,
        finish the exit and return what each rank got back and published."""
        handed = {}

        def record_gather(cp):
            def gather(output, local):
                handed[cp] = local.clone()

            return gather

        def fill_gather(output, local):
            output.copy_(torch.cat([handed[cp] for cp in range(CP_SIZE)]))

        def unused(*args):
            raise AssertionError("no reduce-scatter here")

        # The all-gather: record each rank's shard, then give every rank all.
        for cp in range(CP_SIZE):
            with self.as_rank(
                cp, dict(gather=record_gather(cp), reduce_scatter=unused)
            ):
                prepare_input(
                    self.build(use_reduce_scatter).ffn,
                    self.values[cp],
                    self.residuals[cp],
                    self.cp_extend(),
                )
        gathered, residuals = {}, {}
        for cp in range(CP_SIZE):
            with self.as_rank(cp, dict(gather=fill_gather, reduce_scatter=unused)):
                gathered[cp], residuals[cp] = prepare_input(
                    self.build(use_reduce_scatter).ffn,
                    self.values[cp],
                    self.residuals[cp],
                    self.cp_extend(),
                )
        expected_rows = torch.cat(
            [self.values[cp] + self.residuals[cp] for cp in range(CP_SIZE)]
        )
        for cp in range(CP_SIZE):
            torch.testing.assert_close(gathered[cp], expected_rows, rtol=0, atol=0)

        def compute_output(cp, leaves):
            return gathered[cp] * PARTIAL_WEIGHTS[cp] if leaves else gathered[cp]

        reduced = {}

        def record_reduce_scatter(cp):
            def reduce_scatter(output, input_):
                reduced[cp] = input_.clone()

            return reduce_scatter

        def fill_reduce_scatter(cp):
            def reduce_scatter(output, input_):
                total = sum(reduced[r] for r in range(CP_SIZE))
                output.copy_(total.tensor_split(CP_SIZE)[cp])

            return reduce_scatter

        published, back = {}, {}
        for phase in ("record", "fill"):
            for cp in range(CP_SIZE):
                reduce_scatter = (
                    record_reduce_scatter(cp)
                    if phase == "record"
                    else fill_reduce_scatter(cp)
                )
                with self.as_rank(
                    cp, dict(gather=fill_gather, reduce_scatter=reduce_scatter)
                ) as rank:
                    communicator = self.build(use_reduce_scatter)
                    with communicator.ffn.plan.output.ffn_exit(
                        self.cp_extend(), stream=ResidualStream()
                    ) as exit_:
                        published[cp] = rank.flags.mlp_reduce_scatter
                        output = compute_output(cp, leaves=published[cp])
                    back[cp], _ = finish_exit(
                        exit_, output, ResidualStream(residuals[cp].residual)
                    )
        return published, back, reduced

    def test_the_reduce_scatter_completes_the_sum_the_moe_leaves(self):
        published, back, reduced = self.run_ranks(use_reduce_scatter=True)
        self.assertEqual(published, {0: True, 1: True}, "the MoE leaves its sum")
        self.assertEqual(sorted(reduced), [0, 1], "every rank joins the reduce-scatter")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_a_complete_output_is_only_taken_back(self):
        published, back, reduced = self.run_ranks(use_reduce_scatter=False)
        self.assertEqual(published, {0: False, 1: False}, "the MoE sums itself")
        self.assertEqual(reduced, {}, "nothing is summed again")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_dense_tp_returns_the_same_context_shard(self):
        self.sparse = False
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                _, back, _ = self.run_ranks(use_reduce_scatter=reduce_scatter)
                for cp in range(CP_SIZE):
                    torch.testing.assert_close(
                        back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
                    )

    def test_gather_uses_actual_rows_with_attention_dp(self):
        # CP partners belong to one DP replica. Neither DP buffer padding nor
        # expanded residual width determines the collective's output shape.
        parallel = SimpleNamespace(
            attn_dp_size=2,
            attn_tp_size=1,
            attn_cp_size=CP_SIZE,
            attn_cp_group=group("attn_cp", [2, 3]),
        )
        for shape in ((0, HIDDEN), (5, HIDDEN), (7, 4, HIDDEN)):
            with self.subTest(shape=shape):
                local = torch.arange(torch.tensor(shape).prod()).reshape(shape).double()
                expected = torch.cat((local, local + 1))

                def gather(output, input_):
                    self.assertEqual(output.shape, expected.shape)
                    torch.testing.assert_close(input_, local)
                    output.copy_(expected)

                with (
                    patch.object(dsa_cp, "get_parallel", lambda: parallel),
                    patch.object(
                        dsa_cp, "use_symmetric_memory", lambda *a, **k: nullcontext()
                    ),
                    patch.object(dsa_cp, "is_allocation_symmetric", lambda: False),
                    patch.object(dsa_cp, "attn_cp_all_gather_into_tensor", gather),
                ):
                    torch.testing.assert_close(
                        dsa_cp.attn_cp_interleave_gather(local), expected
                    )

    def test_a_sum_over_other_ranks_is_not_left_to_it(self):
        # A MoE group narrower than attention CP, as when MoE DP splits CP.
        self.moe_tp_size = 1
        published, back, reduced = self.run_ranks(use_reduce_scatter=True)
        self.assertEqual(published, {0: False, 1: False}, "the MoE sums itself")
        self.assertEqual(reduced, {}, "CP must not sum over the wrong ranks")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_attention_tp_completes_ffn_sum_before_cp_take_back(self):
        # DP1 x CP2 x attention TP2: the FFN sums over TP4, not CP2.
        self.attn_tp_size = 2
        complete_output = torch.cat(self.values)

        def unused(*args):
            raise AssertionError("a complete FFN output needs no CP collective")

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                self.sparse = sparse
                for cp in range(CP_SIZE):
                    with self.as_rank(
                        cp, dict(gather=unused, reduce_scatter=unused)
                    ) as rank:
                        stages = self.build(use_reduce_scatter=True)
                        stream = ResidualStream(self.residuals[cp])
                        with stages.ffn.plan.output.ffn_exit(
                            self.cp_extend(), stream=stream
                        ) as exit_:
                            self.assertFalse(rank.flags.mlp_reduce_scatter)
                            self.assertFalse(rank.flags.fuse_mlp_allreduce)
                            # Compute must return its completed TP4 sum.
                            output = complete_output.clone()
                        back, stream = finish_exit(exit_, output, stream)
                        torch.testing.assert_close(
                            back, self.values[cp], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            stream.residual, self.residuals[cp], rtol=0, atol=0
                        )

    def test_mhc_updates_local_streams_only_after_the_cp_sum(self):
        from sglang.srt.layers.layer_boundary.factories import (
            declare_attn,
            declare_ffn,
            make_stages,
        )
        from sglang.srt.layers.layer_boundary.residual.mhc import MHCState

        # Two streams initially contain x. Attention computes 3 * sum(streams),
        # then FFN computes 5 * sum(updated_streams): each final stream is 77*x.
        x = torch.arange(1, CP_SIZE * ROWS * HIDDEN + 1).view(-1, HIDDEN).double()
        for tensor_parallel in (False, True):
            for cp in range(CP_SIZE):
                with self.subTest(tensor_parallel=tensor_parallel, cp=cp):
                    local = x.tensor_split(CP_SIZE)[cp]
                    gathers = [2 * x, 14 * x] if tensor_parallel else [14 * x]
                    sums = [6 * x, 70 * x] if tensor_parallel else [70 * x]

                    def gather(output, input_):
                        expected = gathers.pop(0)
                        torch.testing.assert_close(
                            input_, expected.tensor_split(CP_SIZE)[cp]
                        )
                        output.copy_(expected)

                    def reduce_scatter(output, input_):
                        expected = sums.pop(0)
                        torch.testing.assert_close(
                            input_, expected * PARTIAL_WEIGHTS[cp]
                        )
                        output.copy_(expected.tensor_split(CP_SIZE)[cp])

                    def read(streams, *args):
                        self.assertEqual(streams.shape, (ROWS, 2, HIDDEN))
                        return (
                            streams.sum(1),
                            streams.clone(),
                            torch.ones(ROWS, 2),
                            True,
                        )

                    def write(hidden, residual, h_res, h_post):
                        torch.testing.assert_close(h_res, residual)
                        return residual + hidden[:, None, :] * h_post[:, :, None]

                    with self.as_rank(
                        cp, dict(gather=gather, reduce_scatter=reduce_scatter)
                    ):
                        state = MHCState(2, read, read, write)
                        residual = state.residual_ops()
                        with patch_communicator(
                            "get_exec",
                            lambda: SimpleNamespace(
                                comm=SimpleNamespace(boundary_reduction="rs+rsv"),
                                overlap=SimpleNamespace(enable_two_batch_overlap=False),
                            ),
                        ):
                            attn, ffn = make_stages(
                                (
                                    declare_attn(
                                        read=residual.attn_readout,
                                        update=residual.attn_update,
                                        tensor_parallel_over_cp=tensor_parallel,
                                    ),
                                    None,
                                ),
                                (
                                    declare_ffn(
                                        sparse=True,
                                        read=residual.ffn_readout,
                                        update=residual.ffn_update,
                                    ),
                                    None,
                                ),
                            )
                        batch = self.cp_extend()
                        streams = local[:, None, :].expand(-1, 2, -1).clone()
                        batch.residual_stream = ResidualStream(streams)
                        hidden = attn.prepare(streams, batch)
                        torch.testing.assert_close(
                            hidden, 2 * (x if tensor_parallel else local)
                        )
                        hidden = (
                            hidden * 3 * (PARTIAL_WEIGHTS[cp] if tensor_parallel else 1)
                        )
                        hidden = ffn.prepare(attn.finish(hidden, batch), batch)
                        torch.testing.assert_close(hidden, 14 * x)
                        with ffn.exit(batch) as output:
                            hidden = hidden * 5 * PARTIAL_WEIGHTS[cp]
                        hidden = output.finish(hidden)
                        torch.testing.assert_close(hidden, 77 * streams)
                        self.assertEqual(gathers, [])
                        self.assertEqual(sums, [])
                        self.assertIsNone(state.h_res)

    def test_mixer_decode_sums_cp_before_gathering_attention_dp(self):
        from sglang.srt.layers.layer_boundary import prepare
        from sglang.srt.layers.layer_boundary.boundary import bind_entry
        from sglang.srt.layers.layer_boundary.contracts import (
            EdgeContract,
            InputContract,
            OutputContract,
        )
        from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, TokenAxis
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD

        local_rows = Layout(frozenset({TokenAxis.ATTN_DP}))
        edge = EdgeContract(
            OutputContract(
                local_rows,
                group=SumGroup.ATTN_CP,
                always_partial=True,
                update=PLAIN_ADD,
            ),
            InputContract(Layout(frozenset())),
            local_rows,
            local_rows,
        )
        parallel = SimpleNamespace(
            attn_cp_group=SimpleNamespace(all_reduce=lambda value: value * 2),
            tp_group=None,
        )
        value = torch.tensor([[2.0, 3.0]])
        residual = torch.tensor([[10.0, 20.0]])
        with (
            patch_communicator("get_parallel", lambda: parallel),
            patch_communicator("use_symmetric_memory", lambda *a, **k: nullcontext()),
            patch_communicator("is_allocation_symmetric", lambda: False),
            patch.object(
                prepare,
                "dp_gather",
                lambda value, *args: torch.cat((value, value + 100)),
            ),
        ):
            entry = bind_entry(edge)
            hidden, updated = entry.prepare(
                value, residual, SimpleNamespace(), layernorm
            )
        torch.testing.assert_close(updated, torch.tensor([[14.0, 26.0]]))
        torch.testing.assert_close(hidden, torch.tensor([[14.0, 26.0], [114.0, 126.0]]))


if __name__ == "__main__":
    unittest.main()
