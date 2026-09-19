import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.cache import MultimodalPreprocessCache
from sglang.srt.multimodal.video_cache import validate_video_cache_request
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import VideoData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestVideoCacheProtocol(CustomTestCase):
    @staticmethod
    def _chat_request(**video_fields):
        return ChatCompletionRequest.model_validate(
            {
                "model": "qwen2.5-vl",
                "cache_salt": "tenant-a",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this video."},
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": "https://example.com/clip.mp4",
                                    **video_fields,
                                },
                            },
                        ],
                    }
                ],
            }
        )

    def test_native_json_normalization_preserves_caller_identity(self):
        payload = {
            "text": "Describe <|vision_start|><|video_pad|><|vision_end|>",
            "cache_salt": "tenant-a",
            "video_data": [
                {
                    "url": "https://example.com/clip.mp4",
                    "cache_id": "clip-v1",
                    "preprocess_kwargs": {"fps": 2.0, "max_frames": 8},
                }
            ],
        }
        request = GenerateReqInput(**json.loads(json.dumps(payload)))
        request.normalize_batch_and_arguments()
        self.assertEqual(request.video_data, payload["video_data"])
        self.assertEqual(request.cache_salt, "tenant-a")

    def test_jinja_chat_conversion_preserves_identity_and_sampling(self):
        request = self._chat_request(cache_id="clip-v1", fps=2.0, max_frames=8)
        videos = []
        message = process_content_for_template_format(
            request.messages[0].model_dump(), "openai", [], videos, [], []
        )
        self.assertEqual(
            videos,
            [
                VideoData(
                    url="https://example.com/clip.mp4",
                    cache_id="clip-v1",
                    preprocess_kwargs={"fps": 2.0, "max_frames": 8},
                )
            ],
        )
        self.assertEqual(message["content"][-1], {"type": "video"})
        self.assertEqual(request.cache_salt, "tenant-a")

    def test_named_chat_template_preserves_identity_and_sampling(self):
        request = self._chat_request(cache_id="clip-v1", fps=2.0, max_frames=8)
        conversation = generate_chat_conv(request, "qwen2-vl")
        self.assertEqual(
            conversation.video_data,
            [
                VideoData(
                    url="https://example.com/clip.mp4",
                    cache_id="clip-v1",
                    preprocess_kwargs={"fps": 2.0, "max_frames": 8},
                )
            ],
        )

    def test_no_identity_keeps_existing_chat_video_representation(self):
        request = self._chat_request()
        videos = []
        process_content_for_template_format(
            request.messages[0].model_dump(), "openai", [], videos, [], []
        )
        self.assertEqual(videos, ["https://example.com/clip.mp4"])
        self.assertEqual(generate_chat_conv(request, "qwen2-vl").video_data, videos)

    def test_invalid_identity_is_rejected_by_chat_schema(self):
        for identity in ("", "x" * 257, 42):
            with self.subTest(identity=identity), self.assertRaises(ValidationError):
                self._chat_request(cache_id=identity)

    def test_text_only_template_cannot_silently_discard_cache_id(self):
        request = self._chat_request(cache_id="clip-v1")
        with self.assertRaisesRegex(ValueError, "cache_id"):
            process_content_for_template_format(
                request.messages[0].model_dump(), "string", [], [], [], []
            )


class TestVideoCacheDispatchValidation(CustomTestCase):
    def setUp(self):
        publish(
            ServerArgs(
                model_path="dummy",
                trust_mm_cache_ids=True,
                mm_process_config={},
                encoder_transfer_backend="zmq_to_scheduler",
            ),
            role="tokenizer",
        )
        self.addCleanup(reset_context)

    def test_epd_rejects_ids_before_dispatch_but_keeps_untagged_dispatch(self):
        # Avoid opening manager sockets; invoke the real dispatch method with
        # just its dependencies, so a late validation would call the receiver.
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.mm_processor = SimpleNamespace(model_type="qwen2_5_vl")
        manager.enable_trace = False
        manager.encoder_dispatch_ready = {}
        manager.mm_receiver = Mock()
        manager.mm_receiver.send_encode_request.return_value = None
        request = GenerateReqInput(
            text="Describe video",
            video_data=[VideoData("https://example.com/clip.mp4", cache_id="clip")],
            cache_salt="tenant-a",
        )
        with self.assertRaisesRegex(ValueError, "local multimodal processor"):
            manager._handle_epd_disaggregation_encode_request(request)
        manager.mm_receiver.send_encode_request.assert_not_called()
        request.video_data = ["https://example.com/clip.mp4"]
        manager._handle_epd_disaggregation_encode_request(request)
        self.assertTrue(request.need_wait_for_mm_inputs)
        manager.mm_receiver.send_encode_request.assert_called_once()

    def test_streaming_handler_rejects_invalid_id_request_before_http_200(self):
        from sglang.srt.entrypoints import http_server

        processor = SimpleNamespace(
            model_type="qwen2_5_vl",
            supports_video_cache_ids=True,
            mm_preprocess_cache=MultimodalPreprocessCache(1024),
        )
        reached_media = []

        async def generate(request, raw_request):
            validate_video_cache_request(request, processor=processor, enabled=False)
            reached_media.append(True)
            yield {"text": "unreachable"}

        manager = SimpleNamespace(generate_request=generate, create_abort_task=Mock())
        request = GenerateReqInput(
            text="Describe video",
            stream=True,
            video_data=[VideoData("https://example.com/clip.mp4", cache_id="clip")],
            cache_salt="tenant-a",
        )
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
        ):
            response = asyncio.run(http_server.generate_request(request, None))
        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "trust-mm-cache-ids", json.loads(response.body)["error"]["message"]
        )
        self.assertFalse(reached_media)
        manager.create_abort_task.assert_not_called()

    def test_streaming_first_chunk_is_preserved_and_no_id_stays_lazy(self):
        from sglang.srt.entrypoints import http_server

        async def run():
            for cache_id in (None, "clip"):
                with self.subTest(cache_id=cache_id):
                    events = []

                    async def generate(request, raw_request):
                        events.append("started")
                        yield {"text": "first"}
                        yield {"text": "second"}

                    manager = SimpleNamespace(
                        generate_request=generate,
                        create_abort_task=Mock(return_value=None),
                    )
                    request = GenerateReqInput(
                        text="Describe video",
                        stream=True,
                        video_data=[VideoData("clip.mp4", cache_id=cache_id)],
                    )
                    with patch.object(
                        http_server,
                        "_global_state",
                        SimpleNamespace(tokenizer_manager=manager),
                    ):
                        response = await http_server.generate_request(request, None)
                        self.assertEqual(
                            events, [] if cache_id is None else ["started"]
                        )
                        chunks = [chunk async for chunk in response.body_iterator]
                    self.assertEqual(response.status_code, 200)
                    payloads = [
                        json.loads(chunk.removeprefix(b"data: ").strip())
                        for chunk in chunks[:-1]
                    ]
                    self.assertEqual(payloads, [{"text": "first"}, {"text": "second"}])
                    self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
