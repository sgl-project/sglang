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

from sglang.srt.layers import communicator as comm
from sglang.srt.layers import communicator_dsa_cp as dsa_cp
from sglang.srt.layers import layernorm_sp
from sglang.test.ci.ci_register import register_cpu_ci
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
            enable_dp_attention=False,
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
                    (comm, "should_use_flashinfer_cutlass_moe_fp4_allgather"),
                    lambda: False,
                ),
                (
                    (comm, "post_experts_reduction_group"),
                    lambda: moe_group or parallel.tp_group,
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
                    (dsa_cp, "attn_cp_reduce_scatter_tensor"),
                    collectives["reduce_scatter"],
                ),
            ]:
                stack.enter_context(patch.object(*target, value))
            yield SimpleNamespace(parallel=parallel, flags=flags)

    def build(self, allow_reduce_scatter):
        # A MoE layer after a dense one; dense layers run on every rank here.
        return comm.LayerCommunicator(
            layer_scatter_modes=SimpleNamespace(
                is_first_layer=False,
                is_last_layer=False,
                is_layer_sparse=True,
                is_previous_layer_sparse=False,
            ),
            input_layernorm=layernorm,
            post_attention_layernorm=layernorm,
            allow_reduce_scatter=allow_reduce_scatter,
        )

    def cp_extend(self):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True)
        )

    def run_ranks(self, allow_reduce_scatter):
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
                self.build(allow_reduce_scatter).prepare_mlp(
                    self.values[cp], self.residuals[cp], self.cp_extend()
                )
        gathered, residuals = {}, {}
        for cp in range(CP_SIZE):
            with self.as_rank(cp, dict(gather=fill_gather, reduce_scatter=unused)):
                gathered[cp], residuals[cp] = self.build(
                    allow_reduce_scatter
                ).prepare_mlp(self.values[cp], self.residuals[cp], self.cp_extend())
        expected_rows = torch.cat(
            [self.values[cp] + self.residuals[cp] for cp in range(CP_SIZE)]
        )
        for cp in range(CP_SIZE):
            torch.testing.assert_close(gathered[cp], expected_rows, rtol=0, atol=0)

        # The FFN over every row: a partial sum per rank when it leaves the sum,
        # the complete output otherwise.
        def ffn_output(cp, leaves):
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
                    communicator = self.build(allow_reduce_scatter)
                    with communicator.ffn_exit(self.cp_extend()) as exit_:
                        published[cp] = rank.flags.mlp_reduce_scatter
                        output = ffn_output(cp, leaves=published[cp])
                    back[cp], _ = exit_.finish(output, residuals[cp])
        return published, back, reduced

    def test_the_reduce_scatter_completes_the_sum_the_moe_leaves(self):
        published, back, reduced = self.run_ranks(allow_reduce_scatter=True)
        self.assertEqual(published, {0: True, 1: True}, "the MoE leaves its sum")
        self.assertEqual(sorted(reduced), [0, 1], "every rank joins the reduce-scatter")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_a_complete_output_is_only_taken_back(self):
        published, back, reduced = self.run_ranks(allow_reduce_scatter=False)
        self.assertEqual(published, {0: False, 1: False}, "the MoE sums itself")
        self.assertEqual(reduced, {}, "nothing is summed again")
        for cp in range(CP_SIZE):
            torch.testing.assert_close(
                back[cp], self.values[cp] + self.residuals[cp], rtol=0, atol=0
            )

    def test_a_sum_over_other_ranks_is_not_left_to_it(self):
        # A MoE output owed over fewer ranks than the attention-CP group (a
        # MoE DP that splits CP): the reduce-scatter would add too much.
        def unused(*args):
            raise AssertionError("not run")

        with self.as_rank(
            0, dict(gather=unused, reduce_scatter=unused), moe_group=group("moe", [0])
        ):
            with self.assertRaises(NotImplementedError):
                self.build(allow_reduce_scatter=True)


if __name__ == "__main__":
    unittest.main()
