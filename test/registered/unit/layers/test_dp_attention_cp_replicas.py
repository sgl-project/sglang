"""The DP gather and scatter under attention DP x CP x TP, rank by rank, with the
CP ranks of each DP group holding the same rows (decode, and the logits of
every forward).

Every rank runs dp_gather_partial / dp_gather_replicate and dp_scatter on CPU;
the TP-group all-reduce is replaced by the sum of the buffers the ranks hand to
it.
"""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 4
DP_SIZE = 2
CP_SIZE = 2
ATTN_TP_SIZE = 2
# Binary fractions, so the partial sums add back to the value exactly.
PARTIAL_WEIGHTS = [0.25, 0.75]
# Rows past a DP group's tokens hold whatever the forward left there.
PAD_VALUE = 99.0


class TestDpGatherCpReplicas(CustomTestCase):
    def test_reduce_scatter_restores_every_cp_and_tp_replica(self):
        # TP reduce-scatter leaves DP x CP x attention-TP shards. Decode needs
        # the complete DP slot on every CP and attention-TP rank, not just the
        # attention-TP gather (which is an identity for DSA CP).
        for cp_size, attn_tp_size, rows in (
            (1, 1, 3),
            (1, 2, 6),
            (2, 1, 6),
            (2, 2, 12),
            (2, 1, 0),
            (2, 2, 0),
            (2, 1, 1),
            (2, 1, 3),
            (4, 1, 1),
            (4, 1, 3),
            (2, 2, 2),
            (2, 2, 6),
            (4, 2, 2),
            (4, 2, 6),
        ):
            tp_size = DP_SIZE * cp_size * attn_tp_size
            summed = torch.arange(DP_SIZE * rows * HIDDEN).reshape(-1, HIDDEN).float()
            for rank in range(tp_size):
                dp = rank // (cp_size * attn_tp_size)
                cp = rank // attn_tp_size % cp_size
                with self.subTest(
                    cp_size=cp_size, attn_tp_size=attn_tp_size, rank=rank, rows=rows
                ):
                    calls = []
                    local_contribution = summed / tp_size
                    dp_rows = summed[dp * rows : (dp + 1) * rows]
                    cp_rows = dp_rows.tensor_split(cp_size)[cp]

                    def all_reduce(value):
                        calls.append("all_reduce")
                        torch.testing.assert_close(value, local_contribution)
                        return summed.clone()

                    def reduce_scatter(output, value):
                        calls.append("reduce_scatter")
                        torch.testing.assert_close(value, local_contribution)
                        output.copy_(summed.tensor_split(tp_size)[rank])

                    def tp_gather(output, value):
                        calls.append("attn_tp_gather")
                        self.assertEqual(output.shape[0], value.shape[0] * attn_tp_size)
                        torch.testing.assert_close(
                            value, summed.tensor_split(tp_size)[rank]
                        )
                        output.copy_(cp_rows)

                    def cp_gather(output, value):
                        calls.append("attn_cp_gather")
                        self.assertEqual(output.shape[0], value.shape[0] * cp_size)
                        torch.testing.assert_close(value, cp_rows)
                        output.copy_(dp_rows)

                    parallel = SimpleNamespace(
                        tp_size=tp_size,
                        tp_rank=rank,
                        attn_dp_size=DP_SIZE,
                        attn_dp_rank=dp,
                        attn_cp_size=cp_size,
                        attn_cp_rank=cp,
                        attn_tp_size=attn_tp_size,
                        tp_group=SimpleNamespace(
                            reduce_scatter_tensor=reduce_scatter, all_reduce=all_reduce
                        ),
                        attn_tp_group=SimpleNamespace(all_gather_into_tensor=tp_gather),
                        attn_cp_group=SimpleNamespace(all_gather_into_tensor=cp_gather),
                    )
                    output = torch.empty_like(dp_rows)
                    with (
                        patch.object(
                            dp_attention, "get_parallel", return_value=parallel
                        ),
                        patch.object(dp_attention, "_note_dp_gather_in_prefill_graph"),
                        patch.object(
                            dp_attention, "is_dp_gatherv_active", return_value=False
                        ),
                    ):
                        dp_attention.dp_reduce_scatter_tensor(
                            output, local_contribution.clone()
                        )
                    torch.testing.assert_close(output, dp_rows)
                    if rows % (cp_size * attn_tp_size):
                        self.assertEqual(calls, ["all_reduce"])
                    else:
                        self.assertEqual(calls[0], "reduce_scatter")
                        self.assertEqual(
                            calls.count("attn_cp_gather"), int(cp_size > 1)
                        )

    def run_ranks(self, token_rows, padding, is_partial):
        """Gather on every rank and check each DP slot holds its group's rows
        once, then scatter the result back and check every rank's rows."""
        if padding == DpPaddingMode.MAX_LEN:
            global_num_tokens = [max(token_rows)] * DP_SIZE
        else:
            global_num_tokens = list(token_rows)
        generator = torch.Generator().manual_seed(0)
        values = [
            torch.randint(-8, 8, (n, HIDDEN), generator=generator).double()
            for n in token_rows
        ]
        ranks = [
            (dp, cp, tp)
            for dp in range(DP_SIZE)
            for cp in range(CP_SIZE)
            for tp in range(ATTN_TP_SIZE)
        ]

        def local_rows(dp, tp):
            value = values[dp] * PARTIAL_WEIGHTS[tp] if is_partial else values[dp]
            pad = global_num_tokens[dp] - value.shape[0]
            return torch.cat([value, torch.full((pad, HIDDEN), PAD_VALUE).double()])

        @contextmanager
        def as_rank(dp, cp, tp, all_reduce):
            parallel = SimpleNamespace(
                attn_dp_rank=dp,
                attn_cp_rank=cp,
                attn_tp_rank=tp,
                attn_dp_size=DP_SIZE,
                attn_cp_size=CP_SIZE,
                attn_tp_size=ATTN_TP_SIZE,
                tp_size=DP_SIZE * CP_SIZE * ATTN_TP_SIZE,
                tp_rank=(dp * CP_SIZE + cp) * ATTN_TP_SIZE + tp,
                # No all-gather: the gather under CP is a sum.
                tp_group=SimpleNamespace(unique_name="tp"),
                attn_tp_group=SimpleNamespace(),
            )
            with ExitStack() as stack:
                for name, value in [
                    ("get_parallel", lambda: parallel),
                    ("world_dp_gather_enabled", lambda: False),
                    ("_note_dp_gather_in_prefill_graph", lambda: None),
                    ("memcpy_func", dp_attention.memcpy_cpu),
                    (
                        "memcpy_with_zero_fill_func",
                        dp_attention.memcpy_cpu_with_zero_fill,
                    ),
                    ("tensor_model_parallel_all_reduce", all_reduce),
                ]:
                    stack.enter_context(patch.object(dp_attention, name, value))
                yield SimpleNamespace(
                    global_num_tokens_cpu=list(global_num_tokens),
                    global_num_tokens_gpu=torch.tensor(
                        global_num_tokens, dtype=torch.int64
                    ),
                    dp_padding_mode=padding,
                    dp_local_start_pos=None,
                    dp_local_num_tokens=None,
                )

        gather = (
            dp_attention.dp_gather_partial
            if is_partial
            else dp_attention.dp_gather_replicate
        )
        buffer_len = sum(global_num_tokens)
        handed = {}

        def record(rank):
            def all_reduce(x):
                self.assertNotIn(rank, handed, "one all-reduce per rank")
                handed[rank] = x.clone()
                return x

            return all_reduce

        for rank in ranks:
            dp, cp, tp = rank
            with as_rank(*rank, record(rank)) as forward_batch:
                gather(
                    torch.empty(buffer_len, HIDDEN).double(),
                    local_rows(dp, tp),
                    forward_batch,
                )
        self.assertEqual(sorted(handed), ranks, "every rank joins the gather")
        summed = sum(handed.values())

        # Each DP slot holds its group's rows once; the rest of the slot is the
        # rows the forward padded the group with.
        expected = torch.zeros(buffer_len, HIDDEN).double()
        for dp in range(DP_SIZE):
            start = sum(global_num_tokens[:dp])
            expected[start : start + global_num_tokens[dp]] = sum(
                local_rows(dp, tp) for tp in range(ATTN_TP_SIZE if is_partial else 1)
            )
        torch.testing.assert_close(summed, expected, rtol=0, atol=0)

        for rank in ranks:
            dp, cp, tp = rank
            with as_rank(*rank, None) as forward_batch:
                back = torch.empty(global_num_tokens[dp], HIDDEN).double()
                dp_attention.dp_scatter(back, summed, forward_batch)
            start = sum(global_num_tokens[:dp])
            torch.testing.assert_close(
                back, expected[start : start + global_num_tokens[dp]], rtol=0, atol=0
            )

    def test_every_padding_and_order(self):
        for padding in (DpPaddingMode.SUM_LEN, DpPaddingMode.MAX_LEN):
            for is_partial in (False, True):
                with self.subTest(padding=padding.name, is_partial=is_partial):
                    self.run_ranks([3, 2], padding, is_partial)

    def test_beside_an_idle_dp_group(self):
        for is_partial in (False, True):
            with self.subTest(is_partial=is_partial):
                self.run_ranks([3, 0], DpPaddingMode.SUM_LEN, is_partial)


if __name__ == "__main__":
    unittest.main()
