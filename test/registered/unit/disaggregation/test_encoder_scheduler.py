import asyncio
import sys
from http import HTTPStatus
from unittest.mock import AsyncMock, patch

import pytest

from sglang.srt.disaggregation.encoder.runtime import (
    DPDispatcher,
    EncoderScheduler,
    PendingRequest,
    _resolve_encoder_batch_policy,
    execute_encode_pipeline,
)
from sglang.srt.disaggregation.encoder.server import MMError
from sglang.srt.environ import envs
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


def test_scheduler_rejects_when_pending_limit_is_full():
    async def run_test():
        with envs.SGLANG_ENCODER_MAX_PENDING_REQUESTS.override(1):
            scheduler = EncoderScheduler(
                AsyncMock(), [], max_batch_size=1, request_timeout=1.0
            )
        scheduler.pending_queue.put_nowait(_pending())

        with pytest.raises(MMError, match="pending request limit") as exc_info:
            await asyncio.wait_for(
                scheduler.submit({"req_id": "overflow", "modality": "image"}),
                timeout=5,
            )

        assert exc_info.value.code == HTTPStatus.SERVICE_UNAVAILABLE
        assert scheduler.pending_queue.qsize() == 1

    asyncio.run(run_test())


def test_video_pipeline_uses_bounded_scheduler():
    async def run_test():
        encoder = AsyncMock()
        encoder.transfer_backend = "mooncake"
        scheduler = AsyncMock()
        scheduler.submit.side_effect = asyncio.CancelledError
        request = {"req_id": "pipeline-video", "modality": "video"}
        with pytest.raises(asyncio.CancelledError):
            await execute_encode_pipeline(encoder, scheduler, request)
        scheduler.submit.assert_awaited_once_with(request)
        encoder.encode.assert_not_awaited()

    asyncio.run(run_test())


def test_dp_dispatcher_enforces_capacity_and_skips_full_ranks():
    async def run_test():
        with envs.SGLANG_ENCODER_MAX_PENDING_REQUESTS.override(1):
            dispatcher = DPDispatcher(2, [object(), object()], None, [])
        loop = asyncio.get_running_loop()
        dispatcher.pending_futures[0]["rank-0"] = loop.create_future()
        dispatcher.pending_futures[1]["rank-1"] = loop.create_future()

        with pytest.raises(MMError, match="pending request limit") as exc_info:
            await dispatcher.dispatch({"req_id": "overflow", "modality": "image"})

        assert exc_info.value.code == HTTPStatus.SERVICE_UNAVAILABLE
        assert dispatcher.pending_counts == [1, 1]

        original = dispatcher.pending_futures[0]["rank-0"]
        with pytest.raises(MMError) as exc_info:
            await dispatcher.dispatch({"req_id": "rank-0"})
        assert exc_info.value.code == HTTPStatus.CONFLICT
        assert dispatcher.pending_futures[0]["rank-0"] is original

        dispatcher.pending_futures[1].pop("rank-1")

        with patch(
            "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
            new_callable=AsyncMock,
        ):
            task = asyncio.create_task(
                dispatcher.dispatch({"req_id": "new", "modality": "image"})
            )
            await asyncio.sleep(0)
            assert dispatcher.req_id_to_rank["new"] == 1
            future = dispatcher.pending_futures[1].pop("new")
            future.set_result({"content": "ok"})
            assert await task == {"content": "ok"}

        with pytest.raises(MMError) as exc_info:
            await dispatcher.dispatch({"req_id": "new"})
        assert exc_info.value.code == HTTPStatus.CONFLICT
        assert dispatcher.pending_counts == [1, 0]

    asyncio.run(run_test())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
