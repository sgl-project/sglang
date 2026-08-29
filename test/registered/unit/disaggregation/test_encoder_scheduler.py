import asyncio
import sys

import pytest

from sglang.srt.disaggregation.encoder.runtime import (
    EncoderScheduler,
    PendingRequest,
    _resolve_encoder_batch_policy,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _pending(modality: str = "image") -> PendingRequest:
    return PendingRequest(
        {"req_id": f"{modality}-request", "modality": modality},
        asyncio.get_running_loop(),
    )


def _encoder_pending(req_id: str) -> PendingRequest:
    return PendingRequest(
        {
            "req_id": req_id,
            "modality": "image",
            "mm_items": [object()],
            "num_parts": 1,
            "part_idx": 0,
        },
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


def test_scheduler_isolates_client_error_from_encoder_batch():
    success = (1, 2, 3, None, None)
    bad_request = (0, 0, 0, "invalid media", 400)

    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.batches = []
            self.individual_requests = []

        async def batch_encode(self, requests, _modality):
            self.batches.append([request["req_id"] for request in requests])
            return [bad_request] * len(requests)

        async def encode(self, *, req_id, **_kwargs):
            self.individual_requests.append(req_id)
            return bad_request if req_id == "bad" else success

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=2,
        )
        group = [_encoder_pending(req_id) for req_id in ("bad", "good")]

        await scheduler._dispatch_group(group, Modality.IMAGE)

        assert encoder.batches == [["bad", "good"]]
        assert encoder.individual_requests == ["bad", "good"]
        assert [pending.future.result() for pending in group] == [
            bad_request,
            success,
        ]

    asyncio.run(run_test())


def test_scheduler_does_not_retry_internal_batch_error():
    internal_error = (0, 0, 0, "encoder failure", 500)

    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()

        async def batch_encode(self, requests, _modality):
            return [internal_error] * len(requests)

        async def encode(self, **_kwargs):
            raise AssertionError("internal failures must not be retried")

    async def run_test():
        scheduler = EncoderScheduler(
            encoder=FakeEncoder(),
            send_sockets=[],
            max_batch_size=2,
        )
        group = [_encoder_pending(f"request-{index}") for index in range(2)]

        await scheduler._dispatch_group(group, Modality.IMAGE)

        assert [pending.future.result() for pending in group] == [
            internal_error,
            internal_error,
        ]

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
