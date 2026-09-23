import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.boundary_layout import GatheredRows
from sglang.srt.layers.cp.base import ContextParallelStrategy
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 4


def forward_batch(forward_mode, metadata):
    return SimpleNamespace(forward_mode=forward_mode, attn_cp_metadata=metadata)


class TestMoeCpGatheredRows(CustomTestCase):
    def test_gathered_only_on_a_context_parallel_extend(self):
        metadata = SimpleNamespace(per_rank_actual_token=[3, 2])
        for label, forward_mode, meta, moe_cp_size, gathered in [
            ("cp_extend", ForwardMode.EXTEND, metadata, 2, True),
            ("cp_mixed", ForwardMode.MIXED, metadata, 2, True),
            ("extend_without_cp_metadata", ForwardMode.EXTEND, None, 2, False),
            ("decode_with_stale_metadata", ForwardMode.DECODE, metadata, 2, False),
            ("idle", ForwardMode.IDLE, metadata, 2, False),
            ("single_rank_moe_cp_group", ForwardMode.EXTEND, metadata, 1, False),
        ]:
            with (
                self.subTest(case=label),
                patch.object(comm, "get_moe_cp_size", return_value=moe_cp_size),
            ):
                rows = comm.moe_cp_gathered_rows(forward_batch(forward_mode, meta))
                if gathered:
                    self.assertEqual(rows, GatheredRows.of([3, 2]))
                else:
                    self.assertIsNone(rows)

    def test_rows_are_padded_rank_major_chunks(self):
        rows = GatheredRows.of([3, 2, 3])
        self.assertEqual(rows.chunk, 3)
        self.assertEqual(
            [rows.rank_rows(r) for r in range(3)], [(0, 3), (3, 2), (6, 3)]
        )


class TestMoeCpGatherRoundTrip(CustomTestCase):
    """Gathering every CP rank's rows and taking back one rank's chunk returns
    that rank's rows, for the token counts the CP strategies produce."""

    def check(self, strategy, extend_seqs_len):
        cp_size = strategy.cp_size
        # Per-rank token counts cover every rank; the builder's own rank only
        # selects its attention offsets.
        with patch.object(
            ContextParallelStrategy,
            "cp_rank",
            new_callable=PropertyMock,
            return_value=0,
        ):
            metadata = strategy.build_metadata(
                sum(extend_seqs_len), extend_seqs_len, extend_seqs_len
            )
        batch = forward_batch(ForwardMode.EXTEND, metadata)
        counts = metadata.per_rank_actual_token
        local = [
            torch.arange(n * HIDDEN, dtype=torch.float32).reshape(n, HIDDEN) + 1000 * r
            for r, n in enumerate(counts)
        ]
        chunk = max(counts)
        gathered = torch.cat(
            [torch.nn.functional.pad(x, [0, 0, 0, chunk - x.shape[0]]) for x in local]
        )

        def all_gather(output, rank_input):
            self.assertEqual(rank_input.shape[0], chunk)
            output.copy_(gathered)

        context = SimpleNamespace(attn_dp_size=1)
        for rank in range(cp_size):
            with (
                patch.object(comm, "get_moe_cp_size", return_value=cp_size),
                patch.object(comm, "get_moe_cp_rank", return_value=rank),
                patch.object(comm, "moe_cp_all_gather_into_tensor", all_gather),
                patch.object(
                    comm.CommunicateWithAllReduceAndLayerNormFn,
                    "_gather_hidden_states_and_residual",
                    lambda hidden_states, residual, **kwargs: (hidden_states, residual),
                ),
            ):
                residual = torch.zeros_like(local[rank])
                hidden_states, out_residual = (
                    comm.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                        local[rank],
                        residual,
                        batch,
                        layernorm=None,
                        context=context,
                        residual_input_mode=comm.ScatterMode.TP_ATTN_FULL,
                    )
                )
                torch.testing.assert_close(hidden_states, gathered)
                self.assertIs(out_residual, residual)
                back, _ = (
                    comm.CommunicateSummableTensorPairFn._scatter_hidden_states_moe(
                        hidden_states, residual, batch, context
                    )
                )
                torch.testing.assert_close(back, local[rank])

    def test_zigzag(self):
        for cp_size, extend_seqs_len in [(2, [7, 5]), (2, [16]), (4, [9, 3, 6])]:
            with self.subTest(cp_size=cp_size, extend_seqs_len=extend_seqs_len):
                self.check(ZigzagCPStrategy(cp_size=cp_size), extend_seqs_len)

    def test_interleave(self):
        for cp_size, extend_seqs_len in [(2, [7, 5]), (2, [16]), (4, [9, 3, 6])]:
            with self.subTest(cp_size=cp_size, extend_seqs_len=extend_seqs_len):
                self.check(InterleaveCPStrategy(cp_size=cp_size), extend_seqs_len)


if __name__ == "__main__":
    unittest.main()
