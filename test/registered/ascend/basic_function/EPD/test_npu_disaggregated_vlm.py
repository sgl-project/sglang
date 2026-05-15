import os
import unittest

import requests

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


class TestDisaggregatedVLM(TestDisaggregationBase):
    __test__ = False
    encoder_transfer_backend: str = None
    """Verify encoder-only + language-only configuration."""

    @classmethod
    def setUpClass(cls):
        if cls is TestDisaggregatedVLM:
            raise unittest.SkipTest("Base class, skipping setup")

        super().setUpClass()
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

        # SGLANG_MM_SKIP_COMPUTE_HASH: Ascend NPU backend does not support
        # _local_scalar_dense_npu for UInt64, which is used in multimodal hash
        # computation. This env var replaces hash with a random UUID instead.
        os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

        cls.start_encoder()
        cls.start_language()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

    @classmethod
    def start_encoder(cls):
        encoder_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
        ]

        cls.process_prefill = popen_launch_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
        )

    @classmethod
    def start_language(cls):
        language_args = [
            "--language-only",
            "--encoder-urls",
            cls.prefill_url,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.process_decode = popen_launch_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
        )

    def test_encoder_only_health(self):
        response = requests.get(f"{self.prefill_url}/health", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            "Encoder-only server failed /health check; visual encoder model "
            "may not have loaded correctly on NPU 0-1.",
        )

    def test_language_only_health_generate(self):
        response = requests.get(f"{self.decode_url}/health_generate", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            "Language-only server failed /health_generate; model may not "
            "have initialized correctly.",
        )

    def test_language_only_text_generation(self):
        """Verify the language-only server correctly handles a text-only inference request.

        A language-only VLM server must process text prompts without a visual encoder.
        The expected answer ('Paris') is stable and unambiguous, making it a reliable
        correctness signal. Temperature=0 ensures deterministic output.

        """
        response = requests.post(
            f"{self.decode_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        generated_text = response.json().get("text", "")
        self.assertIn(
            "Paris",
            generated_text,
            f"Language-only server returned unexpected output: '{generated_text}'. "
            "Expected 'Paris' for the capital of France.",
        )

    def test_encoder_processes_image_via_language_server(self):
        """
        Verify end-to-end image processing through encoder + language servers.

        Sends a single-image multimodal request to the language server.
        The language server forwards the image to the encoder server via
        zmq_to_scheduler, receives the embedding, runs language model inference,
        and returns a text response.

        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        }
        response = requests.post(
            f"{self.decode_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Image request through encoder+language servers failed with "
            f"status {response.status_code}. "
            f"Response body: {response.text[:300]}",
        )
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self.assertGreater(
            len(content),
            0,
            "Language server returned empty content for image request; "
            "encoder embedding may not have been received correctly.",
        )

    def test_encoder_processes_multi_images_via_language_server(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {"type": "text", "text": "Describe these two images briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        response = requests.post(
            f"{self.decode_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(response.status_code, 200)
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self.assertGreater(len(content), 0)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_MM_SKIP_COMPUTE_HASH", None)
        super().tearDownClass()


class TestDisaggregatedVLM_ZMQ_Tokenizer(TestDisaggregatedVLM):
    encoder_transfer_backend = "zmq_to_tokenizer"


if __name__ == "__main__":
    unittest.main()
