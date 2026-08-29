import asyncio
import sys

import pytest

from sglang.srt.disaggregation.encoder.runtime import (
    EncoderScheduler,
    PendingRequest,
    _resolve_encoder_batch_policy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _pending(modality: str = "image") -> PendingRequest:
    return PendingRequest(
        {"req_id": f"{modality}-request", "modality": modality},
        asyncio.get_running_loop(),
    )


def test_collect_batch_yields_for_concurrent_image_request_without_fixed_wait():
    # The end-to-end coalescing test cannot replace this case: asyncio.gather
    # enqueues both requests within one event-loop turn, so it passes even with
    # the yield removed. Only a second request enqueued from a separate task
    # observes whether _collect_batch yields at all.
    async def run_test():
        scheduler = EncoderScheduler(
            encoder=None,
            send_sockets=[],
            max_batch_size=8,
            coalesce_same_turn=True,
        )
        first = _pending()
        second = _pending()
        await scheduler.pending_queue.put(first)

        async def enqueue_after_worker_yields():
            await scheduler.pending_queue.put(second)

        producer = asyncio.create_task(enqueue_after_worker_yields())
        batch = await scheduler._collect_batch()
        await producer

        assert batch == [first, second]

    asyncio.run(run_test())


def test_collect_batch_respects_max_batch_size():
    async def run_test():
        scheduler = EncoderScheduler(
            encoder=None,
            send_sockets=[],
            max_batch_size=2,
            coalesce_same_turn=True,
        )
        requests = [_pending() for _ in range(3)]
        for request in requests:
            await scheduler.pending_queue.put(request)

        assert await scheduler._collect_batch() == requests[:2]
        assert scheduler.pending_queue.get_nowait() is requests[2]

    asyncio.run(run_test())


def test_scheduler_coalesces_concurrent_submissions():
    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.batches = []

        async def batch_encode(self, requests, _modality):
            self.batches.append([request["req_id"] for request in requests])
            return [(1, 2, 3, None, None) for _ in requests]

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=8,
            coalesce_same_turn=True,
        )
        scheduler.start()
        try:
            requests = [
                {
                    "req_id": f"image-{index}",
                    "modality": "image",
                    "mm_items": [object()],
                    "num_parts": 1,
                    "part_idx": 0,
                }
                for index in range(2)
            ]
            results = await asyncio.gather(
                *(scheduler.submit(request) for request in requests)
            )
        finally:
            await scheduler.stop()

        assert encoder.batches == [["image-0", "image-1"]]
        assert results == [(1, 2, 3, None, None)] * 2

    asyncio.run(run_test())


def test_cancelled_queued_request_is_retired():
    class FakeEncoder:
        def __init__(self):
            self.released = []

        async def release_request(self, req_id):
            self.released.append(req_id)

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=1,
        )
        task = asyncio.create_task(scheduler.submit({"req_id": "cancelled"}))
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pending = scheduler.pending_queue.get_nowait()
        assert pending.future.cancelled()
        assert encoder.released == ["cancelled"]

    asyncio.run(run_test())


def test_cancelled_active_request_is_retired_without_stopping_encode():
    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.encode_started = asyncio.Event()
            self.finish_encode = asyncio.Event()
            self.released = asyncio.Event()

        async def batch_encode(self, requests, _modality):
            self.encode_started.set()
            await self.finish_encode.wait()
            return [(1, 2, 3, None, None) for _ in requests]

        async def release_request(self, _req_id):
            self.released.set()

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=1,
        )
        scheduler.start()
        request = {
            "req_id": "cancelled",
            "modality": "image",
            "mm_items": [object()],
            "num_parts": 1,
            "part_idx": 0,
        }
        task = asyncio.create_task(scheduler.submit(request))
        await encoder.encode_started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert encoder.released.is_set()
        assert not scheduler._worker_task.done()
        encoder.finish_encode.set()
        await scheduler.stop()

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("model_type", "configured", "explicit", "expected"),
    [
        ("kimi_k3", 8, False, (2, True)),
        ("kimi_k3", 8, True, (8, True)),
        ("kimi_k3", 1, False, (1, True)),
        ("qwen3_vl", 8, False, (8, False)),
    ],
)
def test_resolve_encoder_batch_policy(model_type, configured, explicit, expected):
    assert _resolve_encoder_batch_policy(model_type, configured, explicit) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
