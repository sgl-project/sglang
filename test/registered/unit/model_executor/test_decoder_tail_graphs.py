"""Decoder tail graphs of DeepSeek-V4.1's replay tail: bucketing, the state's
static-buffer round trip, and the replay refill. CPU only; capture runs on GPU."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState
from sglang.srt.models.deepseek_v4_tail_graphs import (
    DecoderTailGraphs,
    _flatten_state,
    _TailGraph,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _graphs(max_rows=1024):
    return DecoderTailGraphs(
        model=None, layer_ids=range(20, 40), row_step=128, max_rows=max_rows
    )


class TestBucketRows(CustomTestCase):
    def test_multiples_of_128_up_to_max(self):
        graphs = _graphs(max_rows=1000)
        self.assertEqual(graphs.max_rows, 896)
        self.assertIsNone(graphs.bucket_rows(0))
        self.assertEqual(graphs.bucket_rows(1), 128)
        self.assertEqual(graphs.bucket_rows(128), 128)
        self.assertEqual(graphs.bucket_rows(129), 256)
        self.assertEqual(graphs.bucket_rows(896), 896)
        self.assertIsNone(graphs.bucket_rows(897))


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


class TestReplayRefill(CustomTestCase):
    def test_steps_refill_shared_static_rows_without_recapture(self):
        graphs = _graphs()
        graphs._break_context = lambda batch: nullcontext()
        captured = {}

        def capture(rows, num_rows, static, rebuild, forward_batch, capture_dspark):
            inputs = [buf[:rows] for buf in static]
            graph = _TailGraph(
                SimpleNamespace(replay=MagicMock()),
                torch.zeros((), dtype=torch.int32),
                inputs[0] * 2,
                None,
                [inputs[0].sum(-1)] if capture_dspark else [],
                forward_batch,
            )
            captured[(rows, capture_dspark)] = graph
            return graph

        graphs._capture = MagicMock(side_effect=capture)
        batch = SimpleNamespace(
            global_num_token_non_padded_cpu=None, forward_mode=ForwardMode.EXTEND
        )

        def step(n, seed, capture_dspark=False):
            torch.manual_seed(seed)
            state = HcState(torch.randn(n, 4))
            residual, pre, aux = graphs.run(
                state=state,
                positions=torch.arange(n),
                input_ids=torch.arange(n) + 7,
                input_ids_global=torch.arange(n) + 7,
                forward_batch=batch,
                capture_dspark=capture_dspark,
            )
            return state, residual, pre, aux

        step(1000, 0)
        self.assertEqual(sorted(captured), [(1024, False)])
        # A key's first smaller bucket captures every bucket from the largest down.
        step(100, 4)
        buckets = list(range(128, 1025, 128))
        self.assertEqual(sorted(captured), [(r, False) for r in buckets])
        state, residual, pre, aux = step(90, 1)
        self.assertEqual(graphs._capture.call_count, len(buckets))
        graph = captured[(128, False)]
        self.assertEqual(graph.graph.replay.call_count, 2)
        (static,) = graphs._static.values()
        self.assertEqual(static[0].shape[0], 1024)
        torch.testing.assert_close(static[0][:90], state.streams)
        torch.testing.assert_close(static[1][:90], torch.arange(90))
        # MoE top-k must not route the pad rows of this step.
        self.assertEqual(int(graph.num_token_non_padded), 90)
        self.assertEqual(residual.shape[0], 90)
        self.assertIsNone(pre)
        self.assertEqual(aux, [])
        # A DSpark step captures its own graphs over the same static buffers,
        # and the auxiliary states are cut to the live rows.
        step(300, 2)
        _, _, _, aux = step(200, 3, capture_dspark=True)
        self.assertEqual(
            sorted(captured),
            sorted([(r, False) for r in buckets] + [(r, True) for r in buckets[1:]]),
        )
        self.assertEqual(len(graphs._static), 1)
        self.assertEqual(aux[0].shape[0], 200)
        # The caller's batch is never edited; the breaks read a copy.
        self.assertIsNone(batch.global_num_token_non_padded_cpu)


if __name__ == "__main__":
    unittest.main()
