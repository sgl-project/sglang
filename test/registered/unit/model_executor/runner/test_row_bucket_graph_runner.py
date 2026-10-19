"""RowBucketGraphRunner: bucketing, largest-first capture, the replay refill of the
shared static buffers, and the hooks for a body that holds a collective. CPU
only; capture runs on GPU."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
    _Graph,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(max_rows=1024):
    return RowBucketGraphRunner(row_step=128, max_rows=max_rows, label="test")


class TestBucketRows(CustomTestCase):
    def test_multiples_of_the_step_up_to_max(self):
        runner = _runner(max_rows=1000)
        self.assertEqual(runner.max_rows, 896)
        self.assertIsNone(runner.bucket_rows(0))
        self.assertEqual(runner.bucket_rows(1), 128)
        self.assertEqual(runner.bucket_rows(128), 128)
        self.assertEqual(runner.bucket_rows(129), 256)
        self.assertEqual(runner.bucket_rows(896), 896)
        self.assertIsNone(runner.bucket_rows(897))


class TestReplayRefill(CustomTestCase):
    def test_steps_refill_shared_static_rows_without_recapture(self):
        runner = _runner()
        captured = {}

        def capture(rows, num_rows, static, forward_batch, body, batch_scope):
            # A real capture records the body over the bucket's static views.
            graph = _Graph(
                SimpleNamespace(replay=MagicMock()),
                torch.zeros((), dtype=torch.int32),
                body([buf[:rows] for buf in static], forward_batch),
                forward_batch,
            )
            captured[rows, body.key] = graph
            return graph

        runner._capture = MagicMock(side_effect=capture)
        batch = SimpleNamespace(global_num_token_non_padded_cpu=None)

        def step(n, seed, key="plain"):
            torch.manual_seed(seed)
            x = torch.randn(n, 4)

            def body(inputs, _batch):
                return [inputs[0] * 2, None] + ([inputs[1]] if key == "aux" else [])

            body.key = key
            out = runner.run(
                key=(key,),
                live=[x, torch.arange(n)],
                forward_batch=batch,
                body=body,
                context=lambda _batch: nullcontext(),
            )
            return x, out

        step(1000, 0)
        self.assertEqual(sorted(captured), [(1024, "plain")])
        # A key's first smaller bucket captures every bucket from the largest down.
        step(100, 4)
        buckets = list(range(128, 1025, 128))
        self.assertEqual(sorted(captured), [(r, "plain") for r in buckets])
        x, out = step(90, 1)
        self.assertEqual(runner._capture.call_count, len(buckets))
        graph = captured[128, "plain"]
        self.assertEqual(graph.graph.replay.call_count, 2)
        (static,) = runner._static.values()
        self.assertEqual(static[0].shape[0], 1024)
        torch.testing.assert_close(static[0][:90], x)
        torch.testing.assert_close(static[1][:90], torch.arange(90))
        # MoE top-k must not route the pad rows of this step.
        self.assertEqual(int(graph.num_token_non_padded), 90)
        self.assertEqual(out[0].shape[0], 90)
        self.assertIsNone(out[1])
        # Another key captures its own graphs over the same static buffers, and
        # every output is cut to the live rows.
        step(300, 2)
        _, out = step(200, 3, key="aux")
        self.assertEqual(
            sorted(captured),
            sorted([(r, "plain") for r in buckets] + [(r, "aux") for r in buckets[1:]]),
        )
        self.assertEqual(len(runner._static), 1)
        self.assertEqual(out[2].shape[0], 200)
        # The caller's batch is never edited; the breaks read a copy.
        self.assertIsNone(batch.global_num_token_non_padded_cpu)


class TestCollectiveHooks(CustomTestCase):
    def test_bucket_from_out_rows_and_batch_scope(self):
        runner = _runner()
        captured = []

        def capture(rows, num_rows, static, forward_batch, body, batch_scope):
            captured.append(rows)
            return _Graph(
                SimpleNamespace(replay=MagicMock()),
                None,
                body([buf[:rows] for buf in static], forward_batch),
                forward_batch,
            )

        runner._capture = MagicMock(side_effect=capture)
        scopes = []

        @contextmanager
        def scope(batch, inputs):
            scopes.append((batch.global_num_token_non_padded_cpu, inputs[0].shape[0]))
            yield

        out = runner.run(
            key=(),
            live=[torch.randn(0, 4)],
            forward_batch=SimpleNamespace(global_num_token_non_padded_cpu=None),
            body=lambda inputs, _batch: [inputs[0] + 1],
            context=lambda _batch: nullcontext(),
            batch_scope=scope,
            bucket_from=300,
            out_rows=1,
        )
        # A rank with no rows of its own replays the bucket of another rank's
        # rows, its outputs cut to the rows it asked for.
        self.assertEqual(captured, [1024, 896, 768, 640, 512, 384])
        self.assertEqual(scopes, [(0, 384)])
        self.assertEqual(out[0].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
