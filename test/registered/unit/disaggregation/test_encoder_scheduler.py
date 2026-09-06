import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.disaggregation.encoder.runtime import (
    EncoderScheduler,
    PendingRequest,
    _resolve_encoder_batch_policy,
    validate_encode_request,
)
from sglang.srt.managers.schedule_batch import Modality
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


def test_scheduler_isolates_bad_request_from_failed_fused_batch():
    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.batches = []

        async def batch_encode(self, requests, _modality):
            req_ids = [request["req_id"] for request in requests]
            self.batches.append(req_ids)
            if len(requests) > 1 or req_ids == ["bad"]:
                return [(0, 0, 0, "bad image", 400) for _ in requests]
            return [(1, 2, 3, None, None)]

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=8,
            coalesce_same_turn=True,
        )
        collector = SimpleNamespace(observe_queue_wait=Mock())
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.server_module.encoder_metrics_collector",
            collector,
        ):
            scheduler.start()
            try:
                requests = [
                    {
                        "req_id": req_id,
                        "modality": "image",
                        "mm_items": [object()],
                        "num_parts": 1,
                        "part_idx": 0,
                    }
                    for req_id in ("bad", "good")
                ]
                results = await asyncio.gather(
                    *(scheduler.submit(request) for request in requests)
                )
            finally:
                await scheduler.stop()

        assert encoder.batches == [["bad", "good"], ["bad"], ["good"]]
        assert results == [(0, 0, 0, "bad image", 400), (1, 2, 3, None, None)]
        assert collector.observe_queue_wait.call_count == len(requests)

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"req_id": ""}, "missing or invalid req_id"),
        ({"modality": "text"}, "unsupported modality"),
        ({"mm_items": []}, "missing or empty mm_items"),
        ({"num_parts": 0}, "num_parts must be a positive integer"),
        ({"part_idx": 1}, "part_idx must be in [0, 1)"),
    ],
)
def test_validate_encode_request_rejects_invalid_fields(update, expected):
    request = {
        "req_id": "request",
        "modality": "image",
        "mm_items": [object()],
        "num_parts": 1,
        "part_idx": 0,
    }
    request.update(update)

    assert expected in validate_encode_request(request)


def test_video_request_is_validated_before_tp_broadcast():
    class FakeSocket:
        pass

    class FakeEncoder:
        async def encode(self, **_kwargs):
            raise AssertionError("invalid request must not reach the encoder")

    async def run_test():
        scheduler = EncoderScheduler(
            encoder=FakeEncoder(),
            send_sockets=[FakeSocket()],
            max_batch_size=1,
        )
        pending = PendingRequest(
            {
                "req_id": "bad-video",
                "modality": "video",
                "mm_items": [object()],
                "num_parts": 1,
                "part_idx": 1,
            },
            asyncio.get_running_loop(),
        )

        await scheduler._dispatch_per_request([pending], Modality.VIDEO)

        with pytest.raises(Exception, match="part_idx must be in"):
            pending.future.result()

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
