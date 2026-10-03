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
    def as_rank(self, cp, collectives, moe_group=None):
        """Rank ``cp`` of attention CP 2 (attention DP and TP 1, TP 2), with
        the attention-CP collectives the case gives."""
        parallel = SimpleNamespace(
            tp_size=CP_SIZE,
            tp_rank=cp,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_dp_enabled=False,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=CP_SIZE,
            attn_cp_rank=cp,
            enable_prefill_cp=True,
            moe_dense_tp_size=1,
            moe_dp_size=1,
            moe_ep_size=1,
            moe_tp_size=CP_SIZE,
            dwdp_size=1,
            enable_attn_tp_input_scattered=False,
            tp_group=group("tp", range(CP_SIZE)),
            attn_tp_group=group("attn_tp", [cp]),
            attn_cp_group=group("attn_cp", range(CP_SIZE)),
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
                    lambda: moe_group or parallel.tp_group,
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
                    (dsa_cp, "get_local_dp_buffer"),
                    lambda g: torch.empty(ROWS * CP_SIZE, HIDDEN).double(),
                ),
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
                sparse=True,
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
        """Gather on each rank, run an FFN that leaves its partial sum, finish
        the exit and return the ranks the exit summed over, what each rank got
        back, and the ranks the reduce-scatter summed over."""
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

        reduced, summed = {}, {}

        def record_sum(cp):
            def sum_output(value, *args, **kwargs):
                summed[cp] = value.clone()
                return value

            return sum_output

        def fill_sum(value, *args, **kwargs):
            return sum(summed[r] for r in range(CP_SIZE))

        def record_reduce_scatter(cp):
            def reduce_scatter(output, input_):
                reduced[cp] = input_.clone()

            return reduce_scatter

        def fill_reduce_scatter(cp):
            def reduce_scatter(output, input_):
                total = sum(reduced[r] for r in range(CP_SIZE))
                output.copy_(total.tensor_split(CP_SIZE)[cp])

            return reduce_scatter

        back = {}
        for phase in ("record", "fill"):
            for cp in range(CP_SIZE):
                reduce_scatter = (
                    record_reduce_scatter(cp)
                    if phase == "record"
                    else fill_reduce_scatter(cp)
                )
                with (
                    self.as_rank(
                        cp, dict(gather=fill_gather, reduce_scatter=reduce_scatter)
                    ) as rank,
                    patch_communicator(
                        "sum_output", record_sum(cp) if phase == "record" else fill_sum
                    ),
                ):
                    communicator = self.build(use_reduce_scatter)
                    with communicator.ffn.plan.output.ffn_exit(
                        self.cp_extend(), stream=ResidualStream()
                    ) as exit_:
                        self.assertFalse(rank.flags.mlp_reduce_scatter)
                        output = gathered[cp] * PARTIAL_WEIGHTS[cp]
                    back[cp], _ = finish_exit(
                        exit_, output, ResidualStream(residuals[cp].residual)
                    )
        return summed, back, reduced

    def test_the_reduce_scatter_completes_the_sum_the_moe_leaves(self):
        summed, back, reduced = self.run_ranks(use_reduce_scatter=True)
        self.assertEqual(summed, {}, "the exit runs no all-reduce")
        self.assertEqual(sorted(reduced), [0, 1], "every rank joins the reduce-scatter")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_without_a_reduce_scatter_the_exit_sums_then_takes_back(self):
        summed, back, reduced = self.run_ranks(use_reduce_scatter=False)
        self.assertEqual(sorted(summed), [0, 1], "the exit sums what the MoE leaves")
        self.assertEqual(reduced, {}, "the take-back sums nothing")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_a_sum_over_other_ranks_is_not_left_to_it(self):
        # A MoE group narrower than attention CP, as when MoE DP splits CP.
        def unused(*args):
            raise AssertionError("not run")

        with self.as_rank(
            0, dict(gather=unused, reduce_scatter=unused), moe_group=group("moe", [0])
        ):
            with self.assertRaises(NotImplementedError):
                self.build(use_reduce_scatter=True)


if __name__ == "__main__":
    unittest.main()
