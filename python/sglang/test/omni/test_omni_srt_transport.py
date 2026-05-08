# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.omni.protocol import OmniOutputSegment, OmniResponse
from sglang.omni.srt_transport import handle_omni_generate_from_scheduler
from sglang.srt.managers.io_struct import OmniGenerateReqInput, OmniGenerateReqOutput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class FakeOrchestrator:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return OmniResponse(
            segments=(
                OmniOutputSegment(type="text", text="ok"),
                OmniOutputSegment(type="image", image=Image.new("RGB", (1, 1))),
            ),
            stats={"num_segments": 2},
        )


class FakeSender:
    def __init__(self):
        self.obj = None

    def send_pyobj(self, obj):
        self.obj = obj


class TestOmniSRTTransport(unittest.TestCase):
    def test_scheduler_transport_uses_cached_orchestrator_and_serializes_images(self):
        orchestrator = FakeOrchestrator()
        scheduler = SimpleNamespace(
            _omni_orchestrators={"sensenova-u1": orchestrator},
            server_args=SimpleNamespace(),
        )

        response = handle_omni_generate_from_scheduler(
            scheduler=scheduler,
            payload={
                "model": "sensenova-u1",
                "messages": [{"type": "text", "text": "draw"}],
            },
        )

        self.assertEqual("draw", orchestrator.requests[0].messages[0].text)
        self.assertEqual("ok", response["segments"][0]["text"])
        self.assertIn("b64_json", response["segments"][1]["image"])

    def test_scheduler_transport_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported omni model"):
            handle_omni_generate_from_scheduler(
                scheduler=SimpleNamespace(server_args=SimpleNamespace()),
                payload={
                    "model": "other-model",
                    "messages": [{"type": "text", "text": "draw"}],
                },
            )

    def test_tokenizer_manager_omni_generate_waits_for_scheduler_response(self):
        asyncio.run(self._run_tokenizer_manager_omni_generate())

    async def _run_tokenizer_manager_omni_generate(self):
        sender = FakeSender()
        tokenizer_manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            omni_futures={},
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


if __name__ == "__main__":
    unittest.main()
