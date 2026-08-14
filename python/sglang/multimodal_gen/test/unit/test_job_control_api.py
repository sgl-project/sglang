import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import msgspec
from fastapi import HTTPException
from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _process_image_batch,
    _request_fingerprint,
    cancel_job,
    job_status,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _dispatch_job_async,
    _require_trackable_video_batch,
    delete_video,
    download_video_content,
)
from sglang.multimodal_gen.runtime.managers.job_registry import (
    CancelReq,
    RequestCancelledError,
    RequestConflictError,
    RequestOverloadedError,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient


class TestImageJobIdentity(unittest.TestCase):
    def test_request_id_is_bounded(self):
        request = ImageGenerationsRequest(prompt="p", request_id="a" * 128)
        self.assertEqual(len(request.request_id), 128)
        with self.assertRaises(ValidationError):
            ImageGenerationsRequest(prompt="p", request_id="a" * 129)
        for invalid in ("", "bad/id", " leading"):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                ImageGenerationsRequest(prompt="p", request_id=invalid)

    def test_fingerprint_is_canonical_and_parameter_sensitive(self):
        first = ImageGenerationsRequest(
            prompt="p", request_id="id", diffusers_kwargs={"b": 2, "a": 1}
        )
        retry = ImageGenerationsRequest(
            prompt="p", request_id="id", diffusers_kwargs={"a": 1, "b": 2}
        )
        changed = ImageGenerationsRequest(prompt="different", request_id="id")
        self.assertEqual(_request_fingerprint(first), _request_fingerprint(retry))
        self.assertNotEqual(_request_fingerprint(first), _request_fingerprint(changed))

    def test_grouped_video_is_rejected_only_when_control_is_enabled(self):
        _require_trackable_video_batch(SimpleNamespace(job_control_enabled=True), None)
        with self.assertRaises(HTTPException) as raised:
            _require_trackable_video_batch(
                SimpleNamespace(job_control_enabled=True), [object(), object()]
            )
        self.assertEqual(raised.exception.status_code, 400)
        _require_trackable_video_batch(
            SimpleNamespace(job_control_enabled=False), [object(), object()]
        )


class TestImageJobRoutes(unittest.IsolatedAsyncioTestCase):
    async def test_precancel_overload_maps_to_429(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.async_scheduler_client.job_control",
                new=AsyncMock(
                    return_value={
                        "status": "unknown",
                        "cancelled": False,
                        "overloaded": True,
                    }
                ),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await cancel_job("valid-id")
        self.assertEqual(raised.exception.status_code, 429)

    async def test_side_channel_uses_structured_msgpack(self):
        class Socket:
            def setsockopt(self, *_args):
                pass

            def connect(self, endpoint):
                self.endpoint = endpoint

            async def send(self, payload):
                self.payload = payload

            async def recv(self):
                return msgspec.msgpack.encode(
                    {"request_id": "valid-id", "status": "unknown"}
                )

            def close(self):
                pass

        socket = Socket()
        client = AsyncSchedulerClient()
        client.context = SimpleNamespace(socket=lambda _kind: socket)
        client.server_args = SimpleNamespace(
            scheduler_cancel_endpoint="tcp://127.0.0.1:5601"
        )
        reply = await client.job_control(CancelReq(request_id="valid-id"))
        self.assertEqual(reply["status"], "unknown")
        self.assertEqual(
            msgspec.msgpack.decode(socket.payload),
            {"operation": "cancel", "request_id": "valid-id"},
        )

    async def test_job_control_failures_have_http_statuses(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.async_scheduler_client.job_control",
                new=AsyncMock(side_effect=TimeoutError),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await job_status("valid-id")
        self.assertEqual(raised.exception.status_code, 504)

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.async_scheduler_client.job_control",
                new=AsyncMock(return_value={"error": "broken"}),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await cancel_job("valid-id")
        self.assertEqual(raised.exception.status_code, 503)


class TestVideoDelete(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await VIDEO_STORE.upsert(
            "video-id",
            {
                "id": "video-id",
                "status": "running",
                "size": "",
                "seconds": "4",
                "quality": "standard",
            },
        )

    async def asyncTearDown(self):
        await VIDEO_STORE.pop("video-id")

    async def test_disabled_control_returns_503_without_removing_job(self):
        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
            return_value=SimpleNamespace(job_control_enabled=False),
        ):
            with self.assertRaises(HTTPException) as raised:
                await delete_video("video-id")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIsNotNone(await VIDEO_STORE.get("video-id"))

    async def test_negative_ack_returns_409_without_mutating_job(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.scheduler_client.async_scheduler_client.job_control",
                new=AsyncMock(return_value={"status": "completed", "cancelled": False}),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await delete_video("video-id")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual((await VIDEO_STORE.get("video-id"))["status"], "running")

    async def test_terminal_store_job_is_not_relabelled(self):
        await VIDEO_STORE.update_fields("video-id", {"status": "completed"})
        with self.assertRaises(HTTPException) as raised:
            await delete_video("video-id")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual((await VIDEO_STORE.get("video-id"))["status"], "completed")

    async def test_positive_ack_marks_cancelling_without_removing_job(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.scheduler_client.async_scheduler_client.job_control",
                new=AsyncMock(return_value={"status": "running", "cancelled": True}),
            ),
        ):
            response = await delete_video("video-id")
        self.assertEqual(response.status, "cancelling")
        self.assertEqual((await VIDEO_STORE.get("video-id"))["status"], "cancelling")

    async def test_completion_race_is_not_regressed_to_cancelling(self):
        async def complete_before_ack(_request):
            await VIDEO_STORE.update_fields("video-id", {"status": "completed"})
            return {"status": "running", "cancelled": True}

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.scheduler_client.async_scheduler_client.job_control",
                new=AsyncMock(side_effect=complete_before_ack),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await delete_video("video-id")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual((await VIDEO_STORE.get("video-id"))["status"], "completed")

    async def test_timeout_is_reported_and_job_is_retained(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.get_global_server_args",
                return_value=SimpleNamespace(job_control_enabled=True),
            ),
            patch(
                "sglang.multimodal_gen.runtime.scheduler_client.async_scheduler_client.job_control",
                new=AsyncMock(side_effect=TimeoutError),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await delete_video("video-id")
        self.assertEqual(raised.exception.status_code, 504)
        self.assertIsNotNone(await VIDEO_STORE.get("video-id"))

    async def test_cancelled_dispatch_clears_media_locations(self):
        await VIDEO_STORE.update_fields(
            "video-id",
            {
                "url": "https://stale.invalid/video.mp4",
                "file_path": "/tmp/stale.mp4",
                "file_paths": ["/tmp/stale.mp4"],
                "num_outputs": 1,
            },
        )
        batch = SimpleNamespace(
            sampling_params=SimpleNamespace(cleanup_video_request=lambda _batch: None)
        )
        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.process_generation_batch",
            new=AsyncMock(side_effect=RequestCancelledError("cancelled")),
        ):
            await _dispatch_job_async("video-id", batch)
        job = await VIDEO_STORE.get("video-id")
        self.assertEqual(job["status"], "cancelled")
        for field in ("url", "file_path", "file_paths", "num_outputs"):
            self.assertIsNone(job[field])

    async def test_cancelled_content_is_not_reported_as_in_progress(self):
        await VIDEO_STORE.update_fields("video-id", {"status": "cancelled"})
        with self.assertRaises(HTTPException) as raised:
            await download_video_content("video-id")
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(
            raised.exception.detail,
            "Video content is unavailable for a cancelled generation",
        )


class TestTypedSchedulerErrors(unittest.IsolatedAsyncioTestCase):
    async def test_typed_scheduler_conflict_is_preserved(self):
        client = SimpleNamespace(
            forward=AsyncMock(
                return_value=OutputBatch(
                    error="request_id was reused",
                    idempotency_conflict=True,
                )
            )
        )
        batch = SimpleNamespace(trace_ctx=None, prompt="prompt")
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.trace_req",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.log_generation_timer",
                return_value=nullcontext(),
            ),
        ):
            with self.assertRaises(RequestConflictError):
                await process_generation_batch(client, batch)

    async def test_typed_scheduler_overload_is_preserved(self):
        client = SimpleNamespace(
            forward=AsyncMock(
                return_value=OutputBatch(
                    error="job-control admission capacity is exhausted",
                    overloaded=True,
                )
            )
        )
        batch = SimpleNamespace(trace_ctx=None, prompt="prompt")
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.trace_req",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.log_generation_timer",
                return_value=nullcontext(),
            ),
        ):
            with self.assertRaises(RequestOverloadedError):
                await process_generation_batch(client, batch)

    async def test_image_routes_map_scheduler_overload_to_429(self):
        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.process_generation_batch",
            new=AsyncMock(side_effect=RequestOverloadedError("at capacity")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await _process_image_batch(object())
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.detail, "at capacity")


if __name__ == "__main__":
    unittest.main()
