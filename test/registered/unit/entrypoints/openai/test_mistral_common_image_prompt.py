import asyncio
import base64
import http.server
import io
import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    TokenizeRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers.mistral_utils import wrap_as_pixtral
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Public v15 tekken with the same vision settings as the Pixtral-style
# checkpoints: 1540px images, 14px patches, 2x2 merge, image ids 10/12/13.
TEKKEN_REPO = "mistralai/Leanstral-1.5-119B-A6B"
TEKKEN_REVISION = "81592da95d94ab0439bfce16df1d55b402e598b6"
# 1618x1148 downscales to 1540x1092.6: mistral-common rounds up a patch row
# that the HF image processor's floor would drop.
IMAGE_SIZES = [(100, 100), (1618, 1148), (1540, 1540)]

# The download is released by the event loop; this only unblocks a stalled loop.
RELEASE_WATCHDOG_SECONDS = 10


def _png(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buf, format="PNG")
    return buf.getvalue()


def _image_url(width, height):
    return "data:image/png;base64," + base64.b64encode(_png(width, height)).decode()


class _ImageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.hits.append(self.path)
        if self.path == "/held.png":
            self.server.download_entered.set()
            self.server.release_download.wait()
        if self.path in ("/image.png", "/held.png"):
            self._send(200, _png(64, 32))
        elif self.path == "/redirect.png":
            self.send_response(302)
            port = self.server.server_port
            self.send_header("Location", f"http://localhost:{port}/image.png")
            self.end_headers()
        elif self.path == "/oversized.png":
            self._send(200, b"x" * (1024 * 1024 + 1))
        else:
            self._send(404, b"")

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class TestMistralCommonImagePrompt(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download

        cls._dir = tempfile.TemporaryDirectory()
        cls.tekken_path = os.path.join(cls._dir.name, "tekken.json")
        shutil.copy(
            hf_hub_download(TEKKEN_REPO, "tekken.json", revision=TEKKEN_REVISION),
            cls.tekken_path,
        )
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
        cls.server.hits = []
        cls.server.download_entered = threading.Event()
        cls.server.release_download = threading.Event()
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._dir.cleanup()

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            mm_process_config={},
            allowed_media_domains=["127.0.0.1"],
            media_url_max_file_size_mb=1,
        )
        publish(server_args, role="tokenizer")

        tokenizer = get_tokenizer(self._dir.name, tokenizer_mode="mistral")
        hf_config = SimpleNamespace(
            architectures=["PixtralForConditionalGeneration"],
            model_type="pixtral",
            image_token_index=10,
            vision_config=SimpleNamespace(
                image_size=1540, patch_size=14, spatial_merge_size=2
            ),
        )
        self.processor = PixtralProcessor(
            hf_config, server_args, wrap_as_pixtral(tokenizer, hf_config), "default"
        )

        tokenizer_manager = Mock(
            tokenizer=tokenizer,
            model_config=Mock(
                is_multimodal=True, hf_config=hf_config, context_len=131072
            ),
            served_model_name="x",
            mm_processor=self.processor,
        )
        template_manager = Mock(
            chat_template_name=None,
            jinja_template_content_format="openai",
            completion_template_name=None,
            reasoning_config=None,
            force_reasoning=False,
            jinja_template_may_reorder_tool_results=False,
        )
        self.chat = OpenAIServingChat(tokenizer_manager, template_manager)
        self.tokenize = OpenAIServingTokenize(tokenizer_manager, template_manager)

    def _convert(self, messages):
        request = ChatCompletionRequest(
            model="x", messages=messages, reasoning_effort="none"
        )
        adapted, _ = self.chat._convert_to_internal_request(request)
        return adapted

    async def _load(self, adapted):
        return await self.processor.process_mm_data_async(
            image_data=adapted.image_data,
            input_text=adapted.input_ids or adapted.text,
            request_obj=adapted,
        )

    def _process(self, messages):
        return asyncio.run(self._load(self._convert(messages)))

    @staticmethod
    def _image_message(url):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]

    def _reference_ids(self, messages):
        from mistral_common.protocol.instruct.request import (
            ChatCompletionRequest as MistralChatCompletionRequest,
        )
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        tokenizer = MistralTokenizer.from_file(self.tekken_path)
        # Tokens do not depend on the pixel transform, which needs opencv.
        with patch(
            "mistral_common.tokens.tokenizers.image.transform_image",
            return_value=np.zeros((3, 1, 1), dtype=np.float32),
        ):
            return tokenizer.encode_chat_completion(
                MistralChatCompletionRequest.from_openai(
                    messages, reasoning_effort="none"
                )
            ).tokens

    def test_image_prompt_matches_mistral_common(self):
        """Bug regression: image requests sent the decoded prompt text to the
        processor, so control tokens became text and the image moved after it."""
        for width, height in IMAGE_SIZES:
            with self.subTest(size=(width, height)):
                messages = self._image_message(_image_url(width, height))
                self.assertEqual(
                    self._process(messages).input_ids, self._reference_ids(messages)
                )

    def test_multi_image_offsets_cover_each_grid(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_url(56, 28)}},
                    {"type": "text", "text": "Compare"},
                    {"type": "image_url", "image_url": {"url": _image_url(28, 84)}},
                ],
            }
        ]
        output = self._process(messages)
        self.assertEqual(output.input_ids, self._reference_ids(messages))
        image_positions = [i for i, t in enumerate(output.input_ids) if t == 10]
        covered = sorted(
            p
            for item in output.mm_items
            for start, end in item.offsets
            for p in range(start, end + 1)
        )
        self.assertEqual(covered, image_positions)

    def _tokenize(self, messages):
        request = TokenizeRequest(messages=messages, reasoning_effort="none")
        return asyncio.run(
            self.tokenize._handle_non_streaming_request(request, request, None)
        )

    def test_tokenize_counts_expanded_image_prompt(self):
        """Bug regression: chat /tokenize returned one [IMG] per image instead of
        its grid, undercounting the prompt the engine runs."""
        for width, height in IMAGE_SIZES:
            with self.subTest(size=(width, height)):
                messages = self._image_message(_image_url(width, height))
                response = self._tokenize(messages)
                expected = self._reference_ids(messages)
                self.assertEqual(response.tokens, expected)
                self.assertEqual(response.count, len(expected))

        self.server.hits.clear()
        response = self._tokenize(
            self._image_message(f"http://localhost:{self.port}/image.png")
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not allowed", response.body)
        self.assertEqual(self.server.hits, [])

    def test_remote_images_load_under_media_url_policy(self):
        """Bug regression: rendering the chat fetched remote images itself, before
        the media URL allowlist, redirect checks and size limit applied."""
        base = f"http://127.0.0.1:{self.port}"
        cases = [
            (f"http://localhost:{self.port}/image.png", "not allowed", []),
            (f"{base}/redirect.png", "not allowed", ["/redirect.png"]),
            (f"{base}/oversized.png", "byte download limit", ["/oversized.png"]),
        ]
        for url, error, hits in cases:
            with self.subTest(url=url):
                self.server.hits.clear()
                adapted = self._convert(self._image_message(url))
                self.assertEqual(self.server.hits, [])
                with self.assertRaisesRegex(ValueError, error):
                    asyncio.run(self._load(adapted))
                self.assertEqual(self.server.hits, hits)

        self.server.hits.clear()
        output = self._process(self._image_message(f"{base}/image.png"))
        self.assertEqual(self.server.hits, ["/image.png"])
        self.assertEqual(
            output.input_ids,
            self._reference_ids(self._image_message(_image_url(64, 32))),
        )

    def test_image_download_does_not_block_event_loop(self):
        """Bug regression: rendering the chat downloaded images synchronously on
        the serving event loop, stalling every other request meanwhile."""
        messages = self._image_message(f"http://127.0.0.1:{self.port}/held.png")
        entered, release = self.server.download_entered, self.server.release_download
        entered.clear()
        release.clear()
        released_by = []
        lock = threading.Lock()

        def release_once(who):
            with lock:
                if not release.is_set():
                    released_by.append(who)
                    release.set()

        async def run():
            load = asyncio.create_task(self._load(self._convert(messages)))
            try:
                while not entered.is_set() and not load.done():
                    await asyncio.sleep(0.01)
                pending_at_release = not load.done()
                release_once("event loop")
                await load
                return pending_at_release
            finally:
                if not load.done():
                    load.cancel()
                    await asyncio.gather(load, return_exceptions=True)

        watchdog = threading.Timer(
            RELEASE_WATCHDOG_SECONDS, release_once, args=("watchdog",)
        )
        watchdog.start()
        try:
            pending_at_release = asyncio.run(run())
        finally:
            watchdog.cancel()
            release.set()
        self.assertEqual(released_by, ["event loop"])
        self.assertTrue(pending_at_release)

    def test_invalid_images_are_client_errors(self):
        """Bug regression: an undecodable image raised from chat rendering as a
        server error instead of the media loader's client error."""
        corrupt = base64.b64encode(_png(64, 32)[:40]).decode()
        sources = [
            "data:image/png;base64,bm90YW5pbWFnZQ==",
            "data:image/png;base64," + corrupt,
            os.path.join(self._dir.name, "missing.png"),
            f"http://127.0.0.1:{self.port}/missing.png",
        ]
        for url in sources:
            with self.subTest(url=url[:48]):
                adapted = self._convert(self._image_message(url))
                with self.assertRaises(ValueError):
                    asyncio.run(self._load(adapted))


if __name__ == "__main__":
    unittest.main()
