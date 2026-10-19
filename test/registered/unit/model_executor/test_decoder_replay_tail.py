"""The decoder replay tail of DeepSeek-V4.1: the state's round trip through the
graph runner's static buffers, the DP bucket and layout, the prefill-graph bridge,
and the nested break context. CPU only; capture runs on GPU."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.row_bucket_graph_runner import _Graph
from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState
from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail, _flatten_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlattenState(CustomTestCase):
    def test_streams_and_pre_round_trip_and_row_caches_drop(self):
        state = HcState(
            torch.randn(3, 4, 8),
            torch.randn(3, 4),
            HcNormed(torch.randn(3, 8)),
            (torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 16)),
        )
        leaves, structure, rebuild = _flatten_state(state)
        self.assertEqual(len(leaves), 2)
        back = rebuild(leaves)
        self.assertIs(back.streams, state.streams)
        self.assertIs(back.pre, state.pre)
        self.assertIsNone(back.input)
        self.assertIsNone(back.stats)
        # The same graphs serve a state with or without the per-row caches.
        self.assertEqual(
            structure, _flatten_state(HcState(state.streams, state.pre))[1]
        )
        bare = _flatten_state(HcState(torch.randn(3, 4, 8)))
        self.assertEqual(len(bare[0]), 1)
        self.assertNotEqual(structure, bare[1])
        self.assertIsNone(bare[2](bare[0]).pre)


def _tail(max_rows=1024):
    return DecoderReplayTail(
        model=SimpleNamespace(late_layer_vl_routing=False),
        layer_ids=range(20, 40),
        row_step=128,
        max_rows=max_rows,
    )


class TestDataParallel(CustomTestCase):
    def test_every_rank_replays_the_bucket_of_the_largest_tail(self):
        tail = _tail()
        tail._break_context = lambda batch: nullcontext()
        captured = []

        def capture(rows, num_rows, static, forward_batch, body, batch_scope):
            captured.append((rows, forward_batch.forward_mode))
            return _Graph(
                SimpleNamespace(replay=MagicMock()),
                None,
                [static[0][:rows], None],
                forward_batch,
            )

        tail.runner._capture = MagicMock(side_effect=capture)
        layouts = []
        module = "sglang.srt.models.deepseek_v4_replay_tail"

        def run(batch, n, a2a_none, **kwargs):
            backend = SimpleNamespace(is_none=lambda: a2a_none)
            with patch(f"{module}.get_moe_a2a_backend", return_value=backend):
                return tail.run(
                    state=HcState(torch.randn(n, 4)),
                    positions=torch.arange(n),
                    input_ids=torch.arange(n),
                    input_ids_global=None,
                    forward_batch=batch,
                    capture_dspark=False,
                    **kwargs,
                )

        with (
            patch(
                f"{module}.enter_late_layer_dp",
                side_effect=lambda batch, counts: layouts.append(counts),
            ),
            patch(f"{module}.exit_late_layer_dp"),
        ):
            # An idle rank under a TP MoE: no rows of its own, an extend graph of
            # the largest rank's bucket, outputs cut to its own padded rows, and
            # a DP layout of that bucket on every rank.
            idle = SimpleNamespace(
                global_num_token_non_padded_cpu=None,
                num_token_non_padded=None,
                forward_mode=ForwardMode.IDLE,
            )
            residual, _, _ = run(idle, 0, True, dp_rows=[300, 0, 128, 1], out_rows=1)
            self.assertEqual(residual.shape[0], 1)
            self.assertEqual(idle.forward_mode, ForwardMode.IDLE)
            # A MoE routing each rank's own rows (MegaMoE) captures its own
            # graphs and sees only the local bucket.
            busy = SimpleNamespace(
                global_num_token_non_padded_cpu=None,
                num_token_non_padded=None,
                forward_mode=ForwardMode.EXTEND,
            )
            residual, _, _ = run(busy, 200, False, dp_rows=[200, 300])
            self.assertEqual(residual.shape[0], 200)
        buckets = range(1024, 383, -128)
        self.assertEqual(captured, [(r, ForwardMode.EXTEND) for r in buckets] * 2)
        self.assertEqual(layouts, [[384] * 4, [384]])


class TestBridge(CustomTestCase):
    def test_views_share_one_buffer_and_zero_the_rows_past_the_tail(self):
        tail = _tail()
        residual = torch.randn(5, 4, 8)
        first = tail.bridge(16, [residual, None, torch.randn(5, 8)])
        self.assertIsNone(first[1])
        torch.testing.assert_close(first[0][:5], residual)
        self.assertEqual(int(first[0][5:].abs().sum()), 0)
        # A smaller captured shape reuses the same buffer through a view.
        smaller = tail.bridge(8, [torch.randn(3, 4, 8), None, torch.randn(3, 8)])
        self.assertEqual(smaller[0].data_ptr(), first[0].data_ptr())
        self.assertEqual(smaller[0].shape[0], 8)
        # A larger one takes a new buffer and keeps the old one for its graphs.
        larger = tail.bridge(32, [torch.randn(3, 4, 8), None, torch.randn(3, 8)])
        self.assertNotEqual(larger[0].data_ptr(), first[0].data_ptr())
        (sets,) = tail._bridge.values()
        self.assertEqual(len(sets), 2)


class TestBreakContext(CustomTestCase):
    def test_nested_break_swaps_only_the_outer_batch(self):
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            context_manager,
        )

        tail = _tail()
        outer_batch, tail_batch = object(), object()
        with context_manager.set_tc_piecewise_forward_context(
            outer_batch, [], None, [], []
        ):
            outer = context_manager.get_tc_piecewise_forward_context()
            with tail._break_context(tail_batch):
                self.assertIs(context_manager.get_tc_piecewise_forward_context(), outer)
                self.assertIs(outer.forward_batch, tail_batch)
            self.assertIs(outer.forward_batch, outer_batch)


if __name__ == "__main__":
    unittest.main()
