import asyncio
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.encoder.runtime import (
    EncoderScheduler,
    PendingRequest,
    _resolve_encoder_batch_policy,
)
from sglang.srt.disaggregation.encoder.server import (
    PreprocessWorker,
    _handle_encoder_worker_request,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakePreprocessWorker:
    def __init__(self, events=None, enabled=True):
        self.events = events
        self.enabled = enabled
        self.jobs = {}

    def can_overlap(self, _modality):
        return self.enabled

    async def submit(self, batch_id, requests, modality):
        req_id = requests[0]["req_id"]
        if self.events is not None:
            self.events.append(("prepare", req_id))
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, req_id)
        self.jobs[batch_id] = (requests, modality, future)

    async def wait_ready(self, batch_id):
        await self.jobs[batch_id][2]

    def take(self, batch_id):
        return self.jobs.pop(batch_id)

    async def cancel(self, batch_id):
        job = self.jobs.pop(batch_id, None)
        if job is not None:
            job[2].cancel()


def _pending(
    modality: str = "image", suffix: str = "request", item_count: int = 1
) -> PendingRequest:
    return PendingRequest(
        {
            "req_id": f"{modality}-{suffix}",
            "modality": modality,
            "mm_items": [object()] * item_count,
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
        first = _pending("image", "first")
        second = _pending("image", "second")
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
        assert await scheduler._collect_batch() == requests[2:]

    asyncio.run(run_test())


def test_collect_batch_respects_flattened_mm_item_budget_without_reordering():
    async def run_test():
        scheduler = EncoderScheduler(
            encoder=None,
            send_sockets=[],
            max_batch_size=8,
        )
        requests = [
            _pending(suffix=str(index), item_count=item_count)
            for index, item_count in enumerate((3, 4, 2, 6))
        ]
        for request in requests:
            await scheduler.pending_queue.put(request)

        assert await scheduler._collect_batch() == requests[:2]
        assert await scheduler._collect_batch() == requests[2:]

    asyncio.run(run_test())


def test_collect_batch_admits_oversized_single_request_without_starvation():
    async def run_test():
        scheduler = EncoderScheduler(
            encoder=None,
            send_sockets=[],
            max_batch_size=8,
        )
        oversized = _pending(suffix="oversized", item_count=9)
        following = _pending(suffix="following", item_count=1)
        await scheduler.pending_queue.put(oversized)
        await scheduler.pending_queue.put(following)

        assert await scheduler._collect_batch() == [oversized]
        assert await scheduler._collect_batch() == [following]

    asyncio.run(run_test())


def test_collect_batch_merges_new_arrivals_with_same_modality_staging():
    async def run_test():
        scheduler = EncoderScheduler(
            encoder=None,
            send_sockets=[],
            max_batch_size=2,
        )
        image_requests = [_pending("image", str(index)) for index in range(2)]
        audio_requests = [_pending("audio", str(index)) for index in range(2)]

        # Simulate a mixed backlog accumulated while the worker was busy.
        for request in (image_requests[0], audio_requests[0], image_requests[1]):
            await scheduler.pending_queue.put(request)

        assert await scheduler._collect_batch() == image_requests

        # The first audio request remains staged until its modality actually
        # gets the lease, so it can merge with an arrival from the meantime.
        await scheduler.pending_queue.put(audio_requests[1])
        assert await scheduler._collect_batch() == audio_requests

    asyncio.run(run_test())


def test_scheduler_coalesces_concurrent_submissions():
    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.batches = []
            self.preprocess_worker = FakePreprocessWorker(enabled=False)

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


def test_scheduler_overlaps_preprocess_and_fences_execution_on_delivery():
    class FakeEncoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.events = []
            self.preprocess_worker = FakePreprocessWorker(self.events)
            self.delivery_gates = {
                "image-0": asyncio.Event(),
                "image-1": asyncio.Event(),
                "image-2": asyncio.Event(),
            }

        async def encode_preprocessed_batch(self, batch_id):
            requests, _modality, handle = self.preprocess_worker.take(batch_id)
            await handle
            req_id = requests[0]["req_id"]
            self.events.append(("execute", req_id))
            return [(1, 2, 3, None, None)]

        async def wait_for_batch_delivery(self, req_ids):
            req_id = req_ids[0]
            self.events.append(("delivery_wait", req_id))
            await self.delivery_gates[req_id].wait()
            self.events.append(("delivery_done", req_id))

        async def release_request(self, _req_id):
            pass

    async def wait_until(predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("scheduler did not reach the expected state")

    async def run_test():
        encoder = FakeEncoder()
        scheduler = EncoderScheduler(
            encoder=encoder,
            send_sockets=[],
            max_batch_size=1,
        )
        scheduler.start()
        requests = [
            {
                "req_id": f"image-{index}",
                "modality": "image",
                "mm_items": [object()],
                "num_parts": 1,
                "part_idx": 0,
            }
            for index in range(3)
        ]
        tasks = [asyncio.create_task(scheduler.submit(req)) for req in requests]
        try:
            await wait_until(lambda: ("delivery_wait", "image-0") in encoder.events)
            assert ("prepare", "image-1") in encoder.events
            assert encoder.events.index(("prepare", "image-1")) < encoder.events.index(
                ("execute", "image-0")
            )
            assert ("prepare", "image-2") not in encoder.events
            assert ("execute", "image-1") not in encoder.events

            encoder.delivery_gates["image-0"].set()
            await wait_until(lambda: ("delivery_wait", "image-1") in encoder.events)
            assert encoder.events.index(
                ("delivery_done", "image-0")
            ) < encoder.events.index(("execute", "image-1"))
            assert encoder.events.index(("prepare", "image-2")) < encoder.events.index(
                ("execute", "image-1")
            )
            assert ("execute", "image-2") not in encoder.events

            encoder.delivery_gates["image-1"].set()
            await wait_until(lambda: ("delivery_wait", "image-2") in encoder.events)
            assert encoder.events.index(
                ("delivery_done", "image-1")
            ) < encoder.events.index(("execute", "image-2"))

            encoder.delivery_gates["image-2"].set()
            assert await asyncio.gather(*tasks) == [(1, 2, 3, None, None)] * 3
        finally:
            for gate in encoder.delivery_gates.values():
                gate.set()
            await scheduler.stop()

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("model_type", "model_hook", "use_gpu", "modality", "expected"),
    [
        ("qwen3_vl", None, False, Modality.IMAGE, True),
        ("kimi_k3", object(), False, Modality.IMAGE, True),
        ("mimo_v2", object(), False, Modality.IMAGE, False),
        ("qwen3_vl", None, True, Modality.IMAGE, False),
        ("qwen3_vl", None, False, Modality.AUDIO, False),
    ],
)
def test_preprocess_overlap_is_limited_to_cpu_image_safe_processors(
    model_type, model_hook, use_gpu, modality, expected
):
    worker = PreprocessWorker(
        SimpleNamespace(
            model_type=model_type,
            preprocessor=SimpleNamespace(
                _model_preprocessor=model_hook,
                use_image_processor_gpu=use_gpu,
            ),
        )
    )
    try:
        assert worker.can_overlap(modality) is expected
    finally:
        worker.shutdown()


def test_tp_worker_prepare_execute_protocol_reuses_local_context_future():
    class FakeEncoder:
        def __init__(self):
            self.events = []
            self.preprocess_worker = FakePreprocessWorker()

        async def batch_encode(
            self,
            requests,
            modality,
            *,
            preprocess_handle,
        ):
            self.events.append(("execute", requests, modality, await preprocess_handle))

        async def encode_preprocessed_batch(self, batch_id):
            requests, modality, handle = self.preprocess_worker.take(batch_id)
            await self.batch_encode(requests, modality, preprocess_handle=handle)

    async def run_test():
        encoder = FakeEncoder()
        requests = [{"req_id": "image-0"}]
        await _handle_encoder_worker_request(
            encoder,
            {
                "type": "prepare_batch",
                "batch_id": 7,
                "modality": "image",
                "requests": requests,
            },
        )
        assert 7 in encoder.preprocess_worker.jobs

        await _handle_encoder_worker_request(
            encoder,
            {"type": "execute_batch", "batch_id": 7},
        )

        assert encoder.preprocess_worker.jobs == {}
        assert encoder.events == [
            ("execute", requests, Modality.IMAGE, "image-0"),
        ]

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("model_type", "configured", "explicit", "expected"),
    [
        ("kimi_k3", 8, False, (2, True)),
        ("kimi_k3", 8, True, (8, True)),
        ("kimi_k3", 1, False, (1, True)),
        ("qwen3_vl", 8, False, (8, True)),
    ],
)
def test_resolve_encoder_batch_policy(model_type, configured, explicit, expected):
    assert _resolve_encoder_batch_policy(model_type, configured, explicit) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
