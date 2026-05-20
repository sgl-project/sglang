"""Unit tests for /v1/loads _compute_aggregate.

Narrow scope: lock in the semantic of new aggregate keys added by this PR
(total_used_tokens vs total_tokens). Trivial helpers (dict filtering,
zero-init branch) are not covered — they would just restate Python.
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.v1_loads import _compute_aggregate
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    QueueMetrics,
)
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    LoadSnapshotReader,
    LoadSnapshotWriter,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load(
    *,
    dp_rank=0,
    running=0,
    waiting=0,
    used=0,
    total=0,
    token_usage=0.0,
    throughput=0.0,
    utilization=0.0,
):
    return {
        "dp_rank": dp_rank,
        "num_running_reqs": running,
        "num_waiting_reqs": waiting,
        "num_used_tokens": used,
        "num_total_tokens": total,
        "token_usage": token_usage,
        "gen_throughput": throughput,
        "utilization": utilization,
    }


def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(self, reader, dp_size: int):
        self.load_snapshot_reader = reader
        self.server_args = SimpleNamespace(dp_size=dp_size)

    def auto_create_handle_loop(self):
        pass


class TestComputeAggregate(CustomTestCase):
    def test_multi_dp_rank_sums(self):
        agg = _compute_aggregate(
            [
                _load(dp_rank=0, running=3, waiting=1, used=50, total=70),
                _load(dp_rank=1, running=5, waiting=2, used=80, total=100),
                _load(dp_rank=2, running=0, waiting=4, used=0, total=40),
            ]
        )
        self.assertEqual(agg["total_running_reqs"], 8)
        self.assertEqual(agg["total_waiting_reqs"], 7)
        self.assertEqual(agg["total_reqs"], 15)
        self.assertEqual(agg["total_used_tokens"], 130)
        self.assertEqual(agg["total_tokens"], 210)

    def test_averages_over_dp_count(self):
        agg = _compute_aggregate(
            [
                _load(token_usage=0.6, throughput=100.0, utilization=0.5),
                _load(token_usage=0.8, throughput=200.0, utilization=0.7),
            ]
        )
        self.assertAlmostEqual(agg["avg_token_usage"], 0.7)
        self.assertAlmostEqual(agg["avg_throughput"], 150.0)
        self.assertAlmostEqual(agg["avg_utilization"], 0.6)

    def test_total_tokens_differs_from_total_used_tokens(self):
        agg = _compute_aggregate([_load(used=10, total=30), _load(used=20, total=45)])
        self.assertEqual(agg["total_used_tokens"], 30)
        self.assertEqual(agg["total_tokens"], 75)


class TestGetLoads(CustomTestCase):
    def test_reads_snapshot_and_filters_sections(self):
        path = _temp_path()
        writer = LoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = LoadSnapshotReader(path, dp_size=1)
        try:
            initial_load = reader.read(0)
            self.assertIsNotNone(initial_load)
            self.assertEqual(initial_load.num_total_tokens, 0)

            writer.write(
                LoadSnapshot.from_metrics(
                    dp_rank=0,
                    timestamp=1.25,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_used_tokens=128,
                    num_total_tokens=256,
                    max_total_num_tokens=4096,
                    token_usage=0.125,
                    gen_throughput=99.5,
                    cache_hit_rate=0.75,
                    utilization=0.5,
                    max_running_requests=128,
                    disaggregation=DisaggregationMetrics(
                        mode="decode",
                        decode_transfer_queue_reqs=4,
                    ),
                    queues=QueueMetrics(
                        waiting=2,
                        grammar=1,
                        paused=0,
                        retracted=3,
                    ),
                )
            )

            manager = _FakeTokenizerManager(reader, dp_size=1)
            loads = asyncio.run(manager.get_loads(include=["core"], dp_rank=0))

            self.assertEqual(len(loads), 1)
            self.assertEqual(loads[0].num_total_tokens, 256)
            self.assertFalse(loads[0].has_disaggregation)
            self.assertFalse(loads[0].has_queues)
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
