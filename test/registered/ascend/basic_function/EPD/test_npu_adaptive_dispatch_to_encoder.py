import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=800, suite="nightly-8-npu-a3", nightly=True)


_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


class TestAdaptiveDispatchToEncoder(CustomTestCase):
    """Testcase: Verify --enable-adaptive-dispatch-to-encoder on Ascend NPU.

    --enable-adaptive-dispatch-to-encoder changes the routing policy:
    - Single-image requests  --> processed locally (no encoder server needed)
    - Multi-image requests   --> dispatched to remote encoder server(s)

    This test starts a language-only server WITHOUT any --encoder-urls.
    This is deliberate: if adaptive dispatch is working correctly, single-image
    requests never attempt to reach an encoder server, so the absence of encoder
    URLs is harmless.  If adaptive dispatch is NOT working, the request will try
    to reach a non-existent encoder and fail -- which is exactly what we want
    to detect.

    [Test Category] Parameter
    [Test Target] --enable-adaptive-dispatch-to-encoder; --language-only
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        # SGLANG_MM_SKIP_COMPUTE_HASH must be set for Ascend NPU:
        # the NPU backend does not support _local_scalar_dense for UInt64,
        # Setting this variable replaces hash computation with a random UUID.
        env = os.environ.copy()
        env["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--language-only",
                "--enable-adaptive-dispatch-to-encoder",
                "--encoder-urls",
                "http://127.0.0.1:9999",
                "--tp-size",
                "2",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_flag_accepted_by_server(self):
        """Verify --enable-adaptive-dispatch-to-encoder is stored in server config."""
        response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
        self.assertEqual(response.status_code, 200)
        info = response.json()
        self.assertTrue(
            info.get("enable_adaptive_dispatch_to_encoder"),
            f"Expected enable_adaptive_dispatch_to_encoder=True in server info, "
            f"got: {info.get('enable_adaptive_dispatch_to_encoder')!r}. "
            f"Full server info: {info}",
        )

    def test_single_image_processed_locally(self):
        """Verify a single-image request is processed locally without an encoder server.

        Assertion rationale:- HTTP 200 means the language server processed the image locally.
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
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Single-image request failed with status {response.status_code}. "
            "Possible causes: (1) adaptive dispatch not routing single-image locally, "
            f"Response body: {response.text[:300]}",
        )


class TestAdaptiveDispatchToEncoderMultiImage(CustomTestCase):
    """Test multi-image request with adaptive dispatch: should be forwarded to encoder server.

    This test starts both an encoder-only server and a language-only server with
    --enable-adaptive-dispatch-to-encoder. For a multi-image request (two images),
    the adaptive dispatch should forward the encoding task to the remote encoder
    server, not process locally. The test verifies that the request succeeds
    (HTTP 200) and returns non-empty content.
    """

    @classmethod
    def setUpClass(cls):
        # Use different ports to avoid conflict with single-image test
        cls.encoder_port = 31100
        cls.language_port = 21100
        cls.encoder_url = f"http://127.0.0.1:{cls.encoder_port}"
        cls.language_url = f"http://127.0.0.1:{cls.language_port}"
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

        env = os.environ.copy()
        env["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

        # Start encoder-only server (with zmq_to_scheduler backend)
        encoder_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
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
        cls.encoder_process = popen_launch_server(
            cls.model,
            cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=encoder_args,
        )

        # Start language-only server with adaptive dispatch and encoder URLs
        language_args = [
            "--language-only",
            "--enable-adaptive-dispatch-to-encoder",
            "--encoder-urls",
            cls.encoder_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.language_process = popen_launch_server(
            cls.model,
            cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=language_args,
        )

        # Wait for both servers to be ready
        cls.wait_server_ready(cls.encoder_url + "/health")
        cls.wait_server_ready(cls.language_url + "/health")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.encoder_process.pid)
        kill_process_tree(cls.language_process.pid)

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start = time.time()
        while True:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - start > timeout:
                raise RuntimeError(f"Server {url} not ready")
            time.sleep(1)

    def test_multi_image_forwarded_to_encoder(self):
        """Send a request with two identical images. Adaptive dispatch should
        forward to encoder server. Expect HTTP 200 and non-empty content."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
                        {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
                        {"type": "text", "text": "Describe these two images briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        response = requests.post(
            f"{self.language_url}/v1/chat/completions",
            json=payload,
            timeout=180,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Multi-image request failed with status {response.status_code}. "
            f"Response: {response.text[:300]}",
        )
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        self.assertGreater(len(content), 0, "Response content is empty")


if __name__ == "__main__":
    unittest.main()
