"""Unit tests for max_queued_requests backpressure: Scheduler._enforce_queue_cap,
HTTP 503 propagation through process_generation_batch, and ServerArgs validation.
"""

import types
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class _GenReq:
    def __init__(self, name: str, is_warmup: bool = False):
        self.name = name
        self.is_warmup = is_warmup


class _ControlReq:
    def __init__(self, name: str):
        self.name = name


class _DummyScheduler:
    def __init__(self, max_queued_requests, queue_depth: int = 0):
        self.server_args = types.SimpleNamespace(
            max_queued_requests=max_queued_requests
        )
        self.waiting_queue: deque = deque(
            (None, _GenReq(f"prefilled-{i}"), 0.0) for i in range(queue_depth)
        )
        self.rejected: list[tuple[OutputBatch, bytes | None]] = []

    @staticmethod
    def _first_generation_req(req_or_group):
        if isinstance(req_or_group, _GenReq):
            return req_or_group
        if isinstance(req_or_group, list) and req_or_group:
            first = req_or_group[0]
            if isinstance(first, _GenReq):
                return first
        return None

    @classmethod
    def _is_warmup_item(cls, req_or_group):
        req = cls._first_generation_req(req_or_group)
        return req.is_warmup if req is not None else False

    def return_result(self, output_batch, identity=None, is_warmup=False):
        self.rejected.append((output_batch, identity))


def _enforce(dummy: _DummyScheduler, new_reqs):
    return Scheduler._enforce_queue_cap(dummy, new_reqs)


class TestEnforceQueueCap(unittest.TestCase):
    def test_unlimited_admits_all(self):
        dummy = _DummyScheduler(max_queued_requests=None)
        new_reqs = [(b"a", _GenReq("r1")), (b"b", _GenReq("r2"))]
        admitted = _enforce(dummy, new_reqs)
        self.assertEqual(admitted, new_reqs)
        self.assertEqual(dummy.rejected, [])

    def test_cap_admits_up_to_limit_then_rejects(self):
        dummy = _DummyScheduler(max_queued_requests=2)
        new_reqs = [
            (b"a", _GenReq("r1")),
            (b"b", _GenReq("r2")),
            (b"c", _GenReq("r3")),
        ]
        admitted = _enforce(dummy, new_reqs)

        self.assertEqual([item[0] for item in admitted], [b"a", b"b"])
        self.assertEqual(len(dummy.rejected), 1)
        rejected_batch, rejected_id = dummy.rejected[0]
        self.assertEqual(rejected_id, b"c")
        self.assertEqual(rejected_batch.status_code, 503)
        self.assertEqual(rejected_batch.error, "The request queue is full.")

    def test_cap_counts_existing_queue_depth(self):
        dummy = _DummyScheduler(max_queued_requests=2, queue_depth=2)
        new_reqs = [(b"x", _GenReq("r1"))]
        admitted = _enforce(dummy, new_reqs)

        self.assertEqual(admitted, [])
        self.assertEqual(len(dummy.rejected), 1)
        self.assertEqual(dummy.rejected[0][0].status_code, 503)

    def test_control_reqs_bypass_cap(self):
        dummy = _DummyScheduler(max_queued_requests=1, queue_depth=1)
        shutdown = _ControlReq("shutdown")
        admitted = _enforce(dummy, [(b"s", shutdown)])
        self.assertEqual(admitted, [(b"s", shutdown)])
        self.assertEqual(dummy.rejected, [])

    def test_warmup_reqs_bypass_cap(self):
        dummy = _DummyScheduler(max_queued_requests=1, queue_depth=1)
        warmup_req = _GenReq("warmup", is_warmup=True)
        admitted = _enforce(dummy, [(b"w", warmup_req)])
        self.assertEqual(admitted, [(b"w", warmup_req)])
        self.assertEqual(dummy.rejected, [])

    def test_mixed_batch_only_generation_overflow_rejected(self):
        dummy = _DummyScheduler(max_queued_requests=1)
        gen1 = _GenReq("r1")
        ctrl = _ControlReq("set_lora")
        gen2 = _GenReq("r2")
        new_reqs = [(b"a", gen1), (b"b", ctrl), (b"c", gen2)]

        admitted = _enforce(dummy, new_reqs)

        self.assertEqual(admitted, [(b"a", gen1), (b"b", ctrl)])
        self.assertEqual(len(dummy.rejected), 1)
        self.assertEqual(dummy.rejected[0][1], b"c")
        self.assertEqual(dummy.rejected[0][0].status_code, 503)

    def test_grouped_list_req_counts_as_one(self):
        dummy = _DummyScheduler(max_queued_requests=1)
        group = [_GenReq("r1a"), _GenReq("r1b")]
        new_reqs = [(b"a", group), (b"b", _GenReq("r2"))]

        admitted = _enforce(dummy, new_reqs)

        self.assertEqual(admitted, [(b"a", group)])
        self.assertEqual(len(dummy.rejected), 1)
        self.assertEqual(dummy.rejected[0][1], b"b")


class TestProcessGenerationBatchPropagatesStatusCode(unittest.IsolatedAsyncioTestCase):
    async def test_status_code_raises_http_exception(self):
        scheduler_client = SimpleNamespace(
            forward=AsyncMock(
                return_value=OutputBatch(
                    error="The request queue is full.",
                    status_code=503,
                )
            )
        )
        batch = SimpleNamespace(
            trace_ctx=SimpleNamespace(trace_req_finish=lambda: None),
            prompt="test",
        )

        with self.assertRaises(HTTPException) as ctx:
            await process_generation_batch(scheduler_client, batch)

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "The request queue is full.")

    async def test_no_status_code_falls_through_to_no_output_check(self):
        scheduler_client = SimpleNamespace(
            forward=AsyncMock(
                return_value=OutputBatch(error="pipeline crashed", status_code=None)
            )
        )
        batch = SimpleNamespace(
            trace_ctx=SimpleNamespace(trace_req_finish=lambda: None),
            prompt="test",
        )

        with self.assertRaises(RuntimeError):
            await process_generation_batch(scheduler_client, batch)


class TestValidateMaxQueuedRequests(unittest.TestCase):
    @staticmethod
    def _fake(**overrides):
        defaults = dict(
            batching_mode="dynamic",
            batching_max_size=1,
            batching_delay_ms=0.0,
            max_queued_requests=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_none_is_valid(self):
        ServerArgs._validate_batching(self._fake(max_queued_requests=None))

    def test_one_is_valid(self):
        ServerArgs._validate_batching(self._fake(max_queued_requests=1))

    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            ServerArgs._validate_batching(self._fake(max_queued_requests=0))

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            ServerArgs._validate_batching(self._fake(max_queued_requests=-5))


if __name__ == "__main__":
    unittest.main()
