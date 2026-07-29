"""
Tests for Anthropic-compatible image input via the /v1/messages endpoint.

python3 anthorpic_api/test/manual/vlm/test_anthropic_vision.py
"""

import json
import unittest

import pybase64
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"


def _fetch_image_base64(url: str) -> str:
    """Download an image and return its base64-encoded content."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pybase64.b64encode(resp.content).decode("utf-8")


class TestAnthropicVision(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--enable-multimodal",
                "--cuda-graph-max-bs=4",
            ],
        )
        cls.messages_url = cls.base_url + "/v1/messages"
        # Pre-fetch the image as base64 once for all tests
        cls.image_base64 = _fetch_image_base64(IMAGE_MAN_IRONING_URL)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _make_request(self, payload, stream=False):
        """Send a request to the /v1/messages endpoint."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return requests.post(
            self.messages_url,
            headers=headers,
            json=payload,
            stream=stream,
        )

    def _parse_sse_events(self, response):
        """Parse SSE events from a streaming response."""
        events = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
        return events

    def _verify_ironing_image_content(self, text):
        """Verify the response text describes the man-ironing-on-SUV image."""
        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ["man", "person", "driver", "someone"]),
            f"Expected mention of a person, got: {text}",
        )
        self.assertTrue(
            any(
                w in text_lower
                for w in ["cab", "taxi", "suv", "vehicle", "car", "trunk", "back"]
            ),
            f"Expected mention of a vehicle, got: {text}",
        )
        self.assertTrue(
            any(
                w in text_lower
                for w in ["iron", "hang", "cloth", "holding", "laundry", "shirt"]
            ),
            f"Expected mention of ironing/clothes, got: {text}",
        )

    # ---- Base64 image tests ----

    def test_single_image_base64(self):
        """Test sending a single base64 image in Anthropic format."""
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                }
            ],
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertTrue(len(body["content"]) > 0)
        self.assertEqual(body["content"][0]["type"], "text")
        text = body["content"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0, "Response text should not be empty")

        # Verify response describes the image content
        self._verify_ironing_image_content(text)

        # Verify usage
        self.assertIn("usage", body)
        self.assertGreater(body["usage"]["input_tokens"], 0)
        self.assertGreater(body["usage"]["output_tokens"], 0)

        # Verify id format
        self.assertTrue(
            body["id"].startswith("msg_"),
            f"ID should start with 'msg_', got: {body['id']}",
        )

    def test_single_image_url(self):
        """Test sending an image via URL (converted to data URI internally)."""
        # Anthropic format uses source.type="base64", but we test the data URI path
        # by pre-encoding the URL image as base64
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What objects do you see in this image?",
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)
        text = body["content"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

        # Verify response describes the image content
        self._verify_ironing_image_content(text)

    def test_image_with_text_blocks(self):
        """Test image combined with multiple text content blocks."""
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Look at this image carefully.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe what you see in one sentence.",
                        },
                    ],
                }
            ],
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)
        self.assertEqual(body["content"][0]["type"], "text")
        text = body["content"][0]["text"]
        self.assertTrue(len(text) > 0)

        # Verify response describes the image content
        self._verify_ironing_image_content(text)

    # ---- Streaming with image ----

    def test_image_stream(self):
        """Test streaming response with image input."""
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image briefly.",
                        },
                    ],
                }
            ],
        }
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

        events = self._parse_sse_events(resp)
        event_types = [e["type"] for e in events]

        # Verify event sequence
        self.assertIn("message_start", event_types)
        self.assertIn("message_stop", event_types)
        self.assertEqual(events[0]["type"], "message_start")

        # Verify we got content
        content_deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.assertTrue(len(content_deltas) > 0, "Expected content_block_delta events")

        # Reconstruct text
        full_text = "".join(
            e["delta"]["text"]
            for e in content_deltas
            if e["delta"].get("type") == "text_delta"
        )
        self.assertTrue(len(full_text) > 0, "Streamed text should not be empty")

        # Verify streamed response describes the image content
        self._verify_ironing_image_content(full_text)

        # Verify message_delta has stop_reason
        message_deltas = [e for e in events if e["type"] == "message_delta"]
        self.assertTrue(len(message_deltas) > 0)
        self.assertIn("stop_reason", message_deltas[-1]["delta"])

    # ---- Multi-image tests ----

    def test_multi_image(self):
        """Test sending multiple images in a single message."""
        logo_base64 = _fetch_image_base64(IMAGE_SGL_LOGO_URL)

        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": logo_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "How many images do you see? Describe each briefly.",
                        },
                    ],
                }
            ],
        }
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)
        text = body["content"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    # ---- Multi-turn with image ----

    def test_multi_turn_with_image(self):
        """Test multi-turn conversation with image context."""
        # First turn: send image
        payload = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                    ],
                },
            ],
            "temperature": 0,
        }
        resp1 = self._make_request(payload)
        self.assertEqual(resp1.status_code, 200, f"Response: {resp1.text}")
        body1 = resp1.json()
        first_response_text = body1["content"][0]["text"]

        # Verify first turn describes the image
        self._verify_ironing_image_content(first_response_text)

        # Second turn: ask follow-up without re-sending image
        payload2 = {
            "model": self.model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": first_response_text,
                },
                {
                    "role": "user",
                    "content": "Can you describe the colors you see?",
                },
            ],
            "temperature": 0,
        }
        resp2 = self._make_request(payload2)
        self.assertEqual(resp2.status_code, 200, f"Response: {resp2.text}")

        body2 = resp2.json()
        self.assertEqual(body2["type"], "message")
        self.assertTrue(len(body2["content"]) > 0)
        self.assertEqual(body2["content"][0]["type"], "text")
        self.assertTrue(len(body2["content"][0]["text"]) > 0)


if __name__ == "__main__":
    unittest.main()
