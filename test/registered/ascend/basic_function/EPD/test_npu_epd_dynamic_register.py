"""Test EPD (Encoder Processing Disaggregation) dynamic registration on NPU.

[Test Category] EPD
[Test Target] --encoder-bootstrap-port; --encoder-register-urls;
--language-only; --encoder-only; --base-gpu-id; --tp-size

Fused workflow: start language-only + encoder-only servers, verify health,
registration, end-to-end VLM request, encoder offline, and re-registration.
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}

ENCODER_BOOTSTRAP_PORT = 8997
LANGUAGE_SERVER_PORT = 30000
LANGUAGE_SERVER_URL = f"http://127.0.0.1:{LANGUAGE_SERVER_PORT}"
ENCODER_SERVER_PORT = 30010
ENCODER_SERVER_URL = f"http://127.0.0.1:{ENCODER_SERVER_PORT}"

# Test image (1x1 red pixel PNG in base64)
TEST_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8/5+h"
    "gQABAAEAf9zB9QAAAABJRU5ErkJggg=="
)


class TestNPUEPDDynamicRegister(CustomTestCase):
    """Verify EPD dynamic registration and encoder lifecycle in one workflow."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.language_url = LANGUAGE_SERVER_URL
        cls.encoder_url = ENCODER_SERVER_URL

        # Step 1: Start language-only server with --encoder-bootstrap-port
        logger.info("=== Starting language-only server with EPD bootstrap ===")
        logger.info("Model: %s", cls.model)
        logger.info("Bootstrap port: %d", ENCODER_BOOTSTRAP_PORT)

        language_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "0",
            "--language-only",
            "--encoder-bootstrap-port",
            str(ENCODER_BOOTSTRAP_PORT),
            "--host",
            "127.0.0.1",
            "--port",
            str(LANGUAGE_SERVER_PORT),
        ]

        cls.language_process = popen_launch_server(
            cls.model,
            cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
            env=NPU_ENV,
        )
        logger.info("Language-only server started.")

        # Step 2: Start encoder-only server with --encoder-register-urls
        logger.info("=== Starting encoder-only server with dynamic registration ===")

        encoder_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--encoder-only",
            "--encoder-register-urls",
            f"http://127.0.0.1:{ENCODER_BOOTSTRAP_PORT}",
            "--host",
            "127.0.0.1",
            "--port",
            str(ENCODER_SERVER_PORT),
        ]

        cls.encoder_process = popen_launch_server(
            cls.model,
            cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
            env=NPU_ENV,
        )
        logger.info("Encoder-only server started and registered.")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "encoder_process"):
            kill_process_tree(cls.encoder_process.pid)
        if hasattr(cls, "language_process"):
            kill_process_tree(cls.language_process.pid)

    def _wait_for_health(self, url, timeout=60):
        """Wait for server to be healthy."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(url + "/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _send_vlm_request(self):
        """Send a VLM request and return the response content."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TEST_IMAGE_BASE64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in one word.",
                        },
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        }

        resp = requests.post(
            self.language_url + "/v1/chat/completions",
            json=payload,
            timeout=180,
        )
        self.assertEqual(resp.status_code, 200, f"VLM request failed: {resp.text}")
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)
        return content

    def test_epd_dynamic_register(self):
        """Verify EPD lifecycle: health, registration, VLM, offline, re-register."""
        # 1. Verify both servers healthy
        self.assertTrue(
            self._wait_for_health(self.language_url),
            "Language server should be healthy",
        )
        self.assertTrue(
            self._wait_for_health(self.encoder_url),
            "Encoder server should be healthy",
        )
        logger.info("Both servers are healthy.")

        # 2. Verify encoder registered via /server_info
        resp = requests.get(self.encoder_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        resp = requests.get(self.language_url + "/server_info", timeout=30)
        self.assertEqual(resp.status_code, 200)
        info = resp.json()
        self.assertTrue(
            info.get("language_only"),
            f"language_only should be True, got: {info.get('language_only')}",
        )
        bootstrap_port = info.get("encoder_bootstrap_port")
        self.assertIsNotNone(
            bootstrap_port,
            "encoder_bootstrap_port should be set in server_info",
        )
        logger.info(
            "Language server /server_info: language_only=True, "
            "encoder_bootstrap_port=%s",
            bootstrap_port,
        )

        # 3. VLM request (end-to-end proof of registration)
        content = self._send_vlm_request()
        logger.info("VLM response: %s", content[:200])

        # 4. Stop encoder, verify it goes offline
        logger.info("Stopping encoder server...")
        kill_process_tree(self.__class__.encoder_process.pid)

        time.sleep(15)

        encoder_healthy = False
        try:
            resp = requests.get(self.encoder_url + "/health", timeout=5)
            encoder_healthy = resp.status_code == 200
        except Exception:
            pass

        self.assertFalse(
            encoder_healthy,
            "Encoder /health should fail after the encoder is stopped",
        )
        logger.info("Encoder is offline (encoder /health no longer responds).")

        # 5. Restart encoder, verify re-registration via VLM request
        logger.info("Restarting encoder server...")

        encoder_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--encoder-only",
            "--encoder-register-urls",
            f"http://127.0.0.1:{ENCODER_BOOTSTRAP_PORT}",
            "--host",
            "127.0.0.1",
            "--port",
            str(ENCODER_SERVER_PORT),
        ]

        self.__class__.encoder_process = popen_launch_server(
            self.model,
            self.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
            env=NPU_ENV,
        )
        logger.info("Encoder server restarted.")

        self.assertTrue(
            self._wait_for_health(self.encoder_url),
            "Encoder server should be healthy after restart",
        )

        content = self._send_vlm_request()
        logger.info("VLM response after re-register: %s", content[:200])
        logger.info("Encoder successfully re-registered (verified end-to-end).")


if __name__ == "__main__":
    unittest.main()
