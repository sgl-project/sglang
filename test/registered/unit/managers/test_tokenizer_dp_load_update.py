"""Unit tests for TokenizerManager DP load update coalescing."""

import asyncio
import dataclasses
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GetLoadsReqOutput, WatchLoadUpdateReq
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


_BASE_LOAD = GetLoadsReqOutput(
    dp_rank=0,
    timestamp=0.0,
    num_running_reqs=0,
    num_waiting_reqs=0,
    num_used_tokens=0,
    num_total_tokens=0,
    max_total_num_tokens=4096,
    token_usage=0.0,
    gen_throughput=0.0,
    cache_hit_rate=0.0,
    utilization=0.0,
    max_running_requests=128,
)


def _load(**overrides) -> GetLoadsReqOutput:
    return dataclasses.replace(_BASE_LOAD, **overrides)


class _Sender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


def _tokenizer_manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.send_to_scheduler = _Sender()
    manager.dp_load_update_coalesce_interval = 0.0
    manager.latest_dp_load_updates = {}
    manager.dp_load_update_flush_task = None
    manager.asyncio_tasks = set()
    return manager


async def _enqueue_and_run_flush(manager: TokenizerManager, loads):
    for load in loads:
        manager.enqueue_dp_load_update(load)
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestTokenizerDPLoadUpdateCoalescing(CustomTestCase):
    def test_coalesces_to_latest_load_per_dp_rank(self):
        manager = _tokenizer_manager()

        asyncio.run(
            _enqueue_and_run_flush(
                manager,
                [
                    _load(dp_rank=0, timestamp=1.0, num_total_tokens=10),
                    _load(dp_rank=0, timestamp=2.0, num_total_tokens=20),
                    _load(dp_rank=1, timestamp=1.5, num_total_tokens=30),
                ],
            )
        )

        self.assertEqual(len(manager.send_to_scheduler.sent), 1)
        req = manager.send_to_scheduler.sent[0]
        self.assertIsInstance(req, WatchLoadUpdateReq)
        self.assertEqual([load.dp_rank for load in req.loads], [0, 1])
        self.assertEqual([load.timestamp for load in req.loads], [2.0, 1.5])
        self.assertEqual([load.num_total_tokens for load in req.loads], [20, 30])

    def test_coalesces_by_arrival_order_not_timestamp(self):
        manager = _tokenizer_manager()

        asyncio.run(
            _enqueue_and_run_flush(
                manager,
                [
                    _load(dp_rank=0, timestamp=2.0, num_total_tokens=20),
                    _load(dp_rank=0, timestamp=1.0, num_total_tokens=10),
                ],
            )
        )

        req = manager.send_to_scheduler.sent[0]
        self.assertEqual(len(req.loads), 1)
        self.assertEqual(req.loads[0].timestamp, 1.0)
        self.assertEqual(req.loads[0].num_total_tokens, 10)


if __name__ == "__main__":
    unittest.main()
