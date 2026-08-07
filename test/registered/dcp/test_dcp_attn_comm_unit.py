"""CPU unit test for the DCP attention communication group.

Pins the three facts ``DcpAttnComm`` centralizes, which used to be re-derived at
each call site:

1. widened head arithmetic (``num_kernel_heads`` / ``narrow_local_heads``)
2. the head-shard mapping ``o_proj`` alignment depends on (``check_layout``)
3. the ``ag_rs`` / ``a2a`` / ``fi_a2a`` dispatch selection

Also pins ``build_dcp_group_ranks`` to the layout that makes (2) hold, so a
future layout change has to update the invariant deliberately.

Usage:
    python -m pytest test_dcp_attn_comm_unit.py -v
    python test_dcp_attn_comm_unit.py
"""

import unittest

import torch

from sglang.srt.distributed.parallel_state import build_dcp_group_ranks
from sglang.srt.layers.dcp import get_dcp_attn_comm, is_lse_base_on_e
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDcpGroupLayout(CustomTestCase):
    def test_groups_are_contiguous_lowest_order_slices(self):
        tp_groups = [list(range(8))]
        self.assertEqual(
            build_dcp_group_ranks(tp_groups, 2),
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )
        self.assertEqual(
            build_dcp_group_ranks(tp_groups, 4),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        self.assertEqual(build_dcp_group_ranks(tp_groups, 8), [list(range(8))])

    def test_groups_are_built_per_tp_group(self):
        tp_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.assertEqual(
            build_dcp_group_ranks(tp_groups, 2),
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )

    def test_layout_yields_dcp_rank_equal_to_attn_tp_rank_mod_dcp_size(self):
        # The identity the post-attention merge relies on, checked directly
        # against the constructed groups rather than assumed.
        tp_size = 8
        for dcp_size in (1, 2, 4, 8):
            groups = build_dcp_group_ranks([list(range(tp_size))], dcp_size)
            for group in groups:
                for rank_in_group, tp_rank in enumerate(group):
                    self.assertEqual(
                        rank_in_group,
                        tp_rank % dcp_size,
                        f"dcp_size={dcp_size} tp_rank={tp_rank}",
                    )


class TestDcpAttnCommHeadArithmetic(CustomTestCase):
    def test_head_counts_and_slices_across_ranks(self):
        comm = get_dcp_attn_comm()
        num_local_heads = 16

        for dcp_size in (1, 2, 4, 8):
            for dcp_rank in range(dcp_size):
                with get_parallel().override(
                    dcp_enabled=dcp_size > 1,
                    attn_dcp_size=dcp_size,
                    attn_dcp_rank=dcp_rank,
                ):
                    self.assertEqual(
                        comm.num_kernel_heads(num_local_heads),
                        num_local_heads * dcp_size,
                    )
                    self.assertEqual(comm.head_shard_index, dcp_rank)
                    self.assertEqual(
                        comm.local_head_offset(num_local_heads),
                        dcp_rank * num_local_heads,
                    )

    def test_narrow_local_heads_selects_this_ranks_contribution(self):
        comm = get_dcp_attn_comm()
        num_local_heads, dcp_size = 4, 4
        # Head index encoded in the values so the selected shard is identifiable.
        widened = (
            torch.arange(num_local_heads * dcp_size, dtype=torch.float32)
            .view(1, -1, 1)
            .expand(2, -1, 8)
            .contiguous()
        )

        for dcp_rank in range(dcp_size):
            with get_parallel().override(
                dcp_enabled=True, attn_dcp_size=dcp_size, attn_dcp_rank=dcp_rank
            ):
                got = comm.narrow_local_heads(widened, num_local_heads)
                self.assertEqual(got.shape, (2, num_local_heads, 8))
                expected_heads = torch.arange(
                    dcp_rank * num_local_heads,
                    (dcp_rank + 1) * num_local_heads,
                    dtype=torch.float32,
                )
                self.assertTrue(torch.equal(got[0, :, 0], expected_heads))

    def test_disabled_dcp_is_identity(self):
        comm = get_dcp_attn_comm()
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            self.assertFalse(comm.enabled)
            self.assertEqual(comm.num_kernel_heads(16), 16)
            self.assertEqual(comm.local_head_offset(16), 0)
            # The reduction pattern is irrelevant with a single rank; report the
            # default rather than reading an unpublished config leaf.
            self.assertEqual(comm.comm_backend, "ag_rs")


class TestDcpAttnCommLayoutCheck(CustomTestCase):
    def test_check_layout_accepts_the_contiguous_layout(self):
        comm = get_dcp_attn_comm()
        # tp8 / dcp2, no DP or prefill CP: attn_tp_size 8, so each DCP group sits
        # inside the single attention TP group.
        for tp_rank in range(8):
            with get_parallel().override(
                dcp_enabled=True,
                attn_dcp_size=2,
                attn_dcp_rank=tp_rank % 2,
                attn_tp_size=8,
                attn_tp_rank=tp_rank,
                attn_dp_size=1,
                attn_cp_size=1,
                tp_size=8,
            ):
                comm.check_layout()

    def test_check_layout_rejects_dcp_wider_than_the_attention_tp_group(self):
        comm = get_dcp_attn_comm()
        # tp8 + dp4 -> attn_tp_size 2, so a dcp_size-4 group spans two attention
        # head shards: the widened head count would exceed the model's heads.
        with get_parallel().override(
            dcp_enabled=True,
            attn_dcp_size=4,
            attn_dcp_rank=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            attn_dp_size=4,
            attn_cp_size=1,
            tp_size=8,
        ):
            with self.assertRaisesRegex(ValueError, "must be a multiple of dcp_size"):
                comm.check_layout()

    def test_check_layout_rejects_a_mismatched_head_shard_index(self):
        comm = get_dcp_attn_comm()
        # A non-contiguous layout: rank 2 of 8 lands at DCP position 1 while
        # attn_tp_rank % dcp_size is 0, so the merge would return the wrong shard.
        with get_parallel().override(
            dcp_enabled=True,
            attn_dcp_size=2,
            attn_dcp_rank=1,
            attn_tp_size=8,
            attn_tp_rank=2,
            attn_dp_size=1,
            attn_cp_size=1,
            tp_size=8,
        ):
            with self.assertRaisesRegex(ValueError, "head-shard mapping"):
                comm.check_layout()

    def test_check_layout_is_a_noop_when_dcp_is_disabled(self):
        comm = get_dcp_attn_comm()
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            comm.check_layout()


class TestDcpAttnCommLseBase(CustomTestCase):
    def test_only_flashmla_reports_natural_log_lse(self):
        self.assertTrue(is_lse_base_on_e("flashmla"))
        for backend in (
            "flashinfer_mla",
            "cutedsl_mla",
            "tokenspeed_mla",
            "trtllm_mla",
            None,
        ):
            self.assertFalse(is_lse_base_on_e(backend), backend)


if __name__ == "__main__":
    unittest.main()
