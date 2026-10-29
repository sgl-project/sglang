"""Single-NPU image association and cache regression for graph-backed batching."""

import base64
import io
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuViTBatching(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_TEST_MODEL", QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        )
        cls.url = os.environ.get("SGLANG_TEST_BASE_URL", DEFAULT_URL_FOR_TEST)
        cls.process = None
        if "SGLANG_TEST_BASE_URL" not in os.environ:
            cls.process = popen_launch_server(
                cls.model,
                cls.url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    "ascend",
                    "--dtype",
                    "bfloat16",
                    "--context-length",
                    "8192",
                    "--chunked-prefill-size",
                    "512",
                    "--max-running-requests",
                    "8",
                    "--mem-fraction-static",
                    "0.7",
                    "--disable-cuda-graph",
                ],
                env={
                    "ASCEND_USE_FIA": "1",
                    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
                },
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def ask(self, spec):
        colors, size, prefix = spec
        content = []
        for color in colors:
            image = Image.new("RGB", size, color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode()
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": prefix
                + "Name the main color of each image in order. Only give the English color names.",
            }
        )
        response = requests.post(
            self.url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 16,
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].lower()
        position = 0
        for color in colors:
            found = answer.find(color, position)
            self.assertGreaterEqual(found, 0, (spec, answer))
            position = found + len(color)

    def test_mixed_images_and_cache_reuse(self):
        # Different sizes, multi-image requests, and a >512-token image.
        specs = [
            (["red"], (448, 448), ""),
            (["blue"], (448, 448), ""),
            (["green", "yellow"], (336, 448), ""),
            (["yellow"], (896, 896), ""),
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(self.ask, specs))
            # Different text prefixes exercise reuse of image embeddings rather
            # than relying only on an identical full-prompt radix-cache hit.
            list(
                executor.map(
                    self.ask,
                    [(c, s, "Look carefully. ") for c, s, _ in reversed(specs)],
                )
            )


if __name__ == "__main__":
    unittest.main()
