"""NPU CI test for sleep-mode comm cleanup with a ViT NPU graph (Qwen3-VL).

With SGLANG_VIT_ENABLE_CUDA_GRAPH=1 the vision tower's blocks (which contain
TP all-gathers) are captured into ViT NPU graphs held by ViTNpuGraphRunner
inside the model — outside every model-runner-level graph. A comm sleep must
drop those graphs and their class-level pool; the wake recaptures lazily on
the next image request. A stale ViT graph would replay TP collectives on the
destroyed communicator: hang (request timeout) or wrong color answer.

The image is a solid-red PNG generated in-process (no dataset dependency);
asking for its color gives a deterministic functional check of the ViT path.

Usage:
python3 -m unittest test_sleep_comm_cleanup_vit
"""

import base64
import os
import struct
import unittest
import zlib

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=1200,
    suite="full-2-npu-a3",
    nightly=True,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

_MIB = 1 << 20
FULL_RELEASE_MIN_DELTA_MB = 500


def _solid_red_png_b64(width=256, height=256) -> str:
    """A solid-red RGB PNG built with stdlib only."""

    def chunk(tag, data):
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes((200, 30, 30)) * width for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _device_used_mb():
    free, total = torch.npu.mem_get_info()
    return (total - free) / _MIB


def _post(base_url, endpoint, payload=None):
    return requests.post(f"{base_url}{endpoint}", json=payload or {}, timeout=300)


def _ask_image_color(test_case, base_url):
    response = _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image? Answer with one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_solid_red_png_b64()}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
        },
    )
    test_case.assertEqual(response.status_code, 200, response.text)
    content = response.json()["choices"][0]["message"]["content"]
    test_case.assertIn("red", content.lower())
    return content


class TestSleepCommCleanupVit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--enable-multimodal",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--enable-memory-saver",
                "--enable-weights-cpu-backup",
                "--disable-radix-cache",
            ],
            # Route ViT execution through ViTNpuGraphRunner.
            env={"SGLANG_VIT_ENABLE_CUDA_GRAPH": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_1_comm_only_sleep_wake_with_vit_graph(self):
        # Baseline: captures the ViT graph for this shape.
        _ask_image_color(self, self.base_url)

        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)

        # ViT graph lazily recaptured on the rebuilt comms.
        _ask_image_color(self, self.base_url)

    def test_2_full_sleep_wake_with_vit_graph(self):
        _ask_image_color(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            FULL_RELEASE_MIN_DELTA_MB,
            f"full sleep should free memory "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _post(self.base_url, "/resume_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)

        _ask_image_color(self, self.base_url)


if __name__ == "__main__":
    unittest.main()
