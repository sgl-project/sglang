"""Unsupported Ollama image input must be rejected explicitly, not silently dropped.

The Ollama-compatible endpoints declare ``images`` fields but are text-only:
a request carrying images must fail with an Ollama-style error before any
template or inference work. Only tokenization and inference are doubles here;
routes, request parsing, schemas and OllamaServing are real.
"""

import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.testclient import TestClient  # noqa: E402

from sglang.srt.entrypoints.http_server import app  # noqa: E402
from sglang.srt.entrypoints.ollama.serving import OllamaServing  # noqa: E402
from sglang.srt.multimodal.media_io import load_image  # noqa: E402
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    loaded, _ = load_image(encoded, gpu_image_decode=False)
    assert loaded.size == (32, 32)
    assert loaded.getpixel((0, 0)) == color
    return encoded


RED, BLUE = png((255, 0, 0)), png((0, 0, 255))


class Boundary:
    served_model_name = "fixture"

    def __init__(self, multimodal=True):
        self.requests = []
        self.messages = []
        self.model_config = SimpleNamespace(is_multimodal=multimodal)
        self.tokenizer = SimpleNamespace(apply_chat_template=self.template)

    def template(self, messages, **kwargs):
        self.messages.append(messages)
        return [1, 2, 3]

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        yield {
            "text": "inference boundary reached",
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
            },
        }


class TestImagesRejection(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def _post(self, endpoint, boundary, payload):
        with patch.dict(app.state._state, ollama_serving=OllamaServing(boundary)):
            return TestClient(app).post("/api/" + endpoint, json=payload)

    def test_images_are_rejected_with_error(self):
        cases = [
            (
                "chat",
                "latest",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Describe the colors",
                            "images": [RED, BLUE],
                        }
                    ]
                },
            ),
            (
                "chat",
                "history",
                {
                    "messages": [
                        {"role": "user", "content": "First", "images": [RED]},
                        {"role": "assistant", "content": "Red"},
                        {"role": "user", "content": "Compare", "images": [BLUE]},
                    ]
                },
            ),
            (
                "chat",
                "history_only",
                {
                    "messages": [
                        {"role": "user", "content": "First", "images": [RED]},
                        {"role": "assistant", "content": "Red"},
                        {"role": "user", "content": "Follow up without images"},
                    ]
                },
            ),
            (
                "generate",
                "prompt",
                {"prompt": "Describe the colors", "images": [RED, BLUE]},
            ),
            ("generate", "image_only", {"prompt": "", "images": [RED, BLUE]}),
        ]
        for multimodal in (True, False):
            for stream in (False, True):
                for endpoint, case, fields in cases:
                    with self.subTest(
                        multimodal=multimodal,
                        stream=stream,
                        endpoint=endpoint,
                        case=case,
                    ):
                        boundary = Boundary(multimodal)
                        response = self._post(
                            endpoint,
                            boundary,
                            {"model": "fixture", "stream": stream, **fields},
                        )
                        self.assertEqual(response.status_code, 400, response.text)
                        body = response.json()
                        self.assertIn("error", body)
                        self.assertIn("image", body["error"].lower())
                        self.assertEqual(boundary.requests, [])
                        self.assertEqual(boundary.messages, [])

    def test_absent_null_empty_images_are_text_requests(self):
        for endpoint in ("chat", "generate"):
            for name, fields in (
                ("absent", {}),
                ("null", {"images": None}),
                ("empty", {"images": []}),
            ):
                with self.subTest(endpoint=endpoint, case=name):
                    boundary = Boundary()
                    payload = {
                        "model": "fixture",
                        "stream": False,
                        "options": {"temperature": 0.7},
                    }
                    if endpoint == "chat":
                        payload["messages"] = [
                            {"role": "user", "content": "Hello", **fields}
                        ]
                    else:
                        payload.update(prompt="Hello", **fields)
                    response = self._post(endpoint, boundary, payload)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(boundary.requests), 1)
                    self.assertIsNone(boundary.requests[0].image_data)
                    self.assertEqual(
                        boundary.requests[0].sampling_params["temperature"], 0.7
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
