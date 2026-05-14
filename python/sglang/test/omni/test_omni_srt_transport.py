# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.omni.core.protocol import (
    OmniContextBundle,
    OmniContextRef,
    OmniOutputSegment,
    OmniResponse,
)
from sglang.omni.entrypoints.http_api import create_srt_omni_router
from sglang.omni.runtime.srt_scheduler_state import OmniSchedulerState
from sglang.omni.runtime.srt_transport import handle_omni_generate_with_omni_coordinator
from sglang.srt.managers.io_struct import (
    OmniGenerateReqInput,
    OmniGenerateReqOutput,
    OmniGenerateStreamOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class FakeCoordinator:
    def __init__(self):
        self.requests = []
        self.contexts = []
        self.ar_backend = SimpleNamespace(released=[])
        self.ar_backend.release = self.ar_backend.released.append

    def generate_with_context(
        self,
        request,
        *,
        context=None,
        release_context=True,
        stop_after_generation_limit=False,
        stream_sink=None,
    ):
        self.requests.append(request)
        if context is None:
            context = OmniContextBundle(
                full=OmniContextRef(context_id="ctx0", session_id="s0")
            )
        self.contexts.append(context)
        response = OmniResponse(
            segments=(
                OmniOutputSegment(type="text", text="ok"),
                OmniOutputSegment(type="image", image=Image.new("RGB", (1, 1))),
            ),
            context=context.full,
            stats={"num_segments": 2},
        )
        if release_context:
            self.ar_backend.release(context)
        return response, context


class FakeSender:
    def __init__(self):
        self.obj = None

    def send_pyobj(self, obj):
        self.obj = obj


class FakeRawRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class FakeTokenizerManager:
    def __init__(self):
        self.obj = None

    async def omni_generate(self, obj, request):
        self.obj = obj
        return OmniGenerateReqOutput(
            rid=obj.rid,
            success=True,
            payload={"segments": [{"type": "text", "text": "ok"}]},
        )


class TestOmniSRTTransport(unittest.TestCase):
    def test_scheduler_transport_serializes_images_and_reuses_session(self):
        coordinator = FakeCoordinator()
        scheduler = SimpleNamespace(
            omni_scheduler_state=OmniSchedulerState(
                coordinator=coordinator,
                coordinator_model_key="sensenova-u1",
            ),
            server_args=SimpleNamespace(),
        )

        response = handle_omni_generate_with_omni_coordinator(
            scheduler=scheduler,
            payload={
                "model": "sensenova-u1",
                "messages": [{"type": "text", "text": "draw"}],
            },
        )

        self.assertEqual("draw", coordinator.requests[0].messages[0].text)
        self.assertEqual("ok", response["segments"][0]["text"])
        self.assertIn("b64_json", response["segments"][1]["image"])
        self.assertEqual(1, len(coordinator.ar_backend.released))

        first = handle_omni_generate_with_omni_coordinator(
            scheduler=scheduler,
            payload={
                "model": "sensenova-u1",
                "keep_session": True,
                "messages": [{"type": "text", "text": "draw"}],
            },
        )
        second = handle_omni_generate_with_omni_coordinator(
            scheduler=scheduler,
            payload={
                "model": "sensenova-u1",
                "session_id": first["session"]["id"],
                "messages": [{"type": "text", "text": "again"}],
            },
        )

        self.assertEqual("s0", first["session"]["id"])
        self.assertEqual(2, second["session"]["turns"])
        self.assertIs(coordinator.contexts[1], coordinator.contexts[2])
        self.assertEqual(1, len(coordinator.ar_backend.released))

        closed = handle_omni_generate_with_omni_coordinator(
            scheduler=scheduler,
            payload={"action": "close_session", "session_id": "s0"},
        )

        self.assertFalse(closed["session"]["alive"])
        self.assertEqual(2, len(coordinator.ar_backend.released))

    def test_tokenizer_manager_omni_generate_waits_for_scheduler_response(self):
        asyncio.run(self._run_tokenizer_manager_omni_generate())

    def test_tokenizer_manager_omni_generate_stream_yields_scheduler_events(self):
        asyncio.run(self._run_tokenizer_manager_omni_generate_stream())

    def test_srt_http_route_forwards_payload_to_tokenizer_manager(self):
        asyncio.run(self._run_srt_http_route())

    async def _run_tokenizer_manager_omni_generate(self):
        sender = FakeSender()
        tokenizer_manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            omni_futures={},
            omni_stream_queues={},
            send_to_scheduler=sender,
        )

        task = asyncio.create_task(
            TokenizerManager.omni_generate(
                tokenizer_manager,
                OmniGenerateReqInput(payload={"messages": [{"type": "text"}]}),
            )
        )
        await asyncio.sleep(0)

        self.assertIsNotNone(sender.obj.rid)
        TokenizerManager._handle_omni_generate_req_output(
            tokenizer_manager,
            OmniGenerateReqOutput(
                rid=sender.obj.rid,
                success=True,
                payload={"segments": []},
            ),
        )

        response = await task
        self.assertTrue(response.success)
        self.assertEqual({"segments": []}, response.payload)

    async def _run_tokenizer_manager_omni_generate_stream(self):
        sender = FakeSender()
        tokenizer_manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            omni_futures={},
            omni_stream_queues={},
            send_to_scheduler=sender,
        )

        async def collect():
            return [
                event
                async for event in TokenizerManager.omni_generate_stream(
                    tokenizer_manager,
                    OmniGenerateReqInput(payload={"messages": [{"type": "text"}]}),
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0)

        self.assertIsNotNone(sender.obj.rid)
        self.assertTrue(sender.obj.stream)
        TokenizerManager._handle_omni_generate_stream_output(
            tokenizer_manager,
            OmniGenerateStreamOutput(
                rid=sender.obj.rid,
                event={
                    "type": "text_delta",
                    "segment_id": "seg-0",
                    "delta": "ok",
                },
            ),
        )
        TokenizerManager._handle_omni_generate_req_output(
            tokenizer_manager,
            OmniGenerateReqOutput(rid=sender.obj.rid, success=True, payload={}),
        )

        self.assertEqual(
            [{"type": "text_delta", "segment_id": "seg-0", "delta": "ok"}],
            await task,
        )

    async def _run_srt_http_route(self):
        tokenizer_manager = FakeTokenizerManager()
        router = create_srt_omni_router(
            get_tokenizer_manager=lambda: tokenizer_manager,
            validate_json_request=lambda raw_request: None,
        )

        endpoint = next(
            route.endpoint
            for route in router.routes
            if route.path == "/v1/omni/generate"
        )
        response = await endpoint(
            FakeRawRequest(
                {
                    "model": "sensenova-u1",
                    "messages": [{"type": "text", "text": "draw"}],
                }
            )
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("draw", tokenizer_manager.obj.payload["messages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
