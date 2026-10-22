"""GLM-5.3-Flash interleave CP4 + native MTP on four Blackwell GPUs."""

import base64
import io
import unittest

import requests
from PIL import Image

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")


class TestGLM53FlashB200ContextParallel(
    GSM8KMixin,
    CustomTestCase,
):
    gsm8k_score_threshold = 0.93
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    server_args = [
        "--tp-size",
        "4",
        "--dsa-prefill-backend",
        "trtllm",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--attn-cp-size",
        "4",
        # KDA partitions heads over TP4, sharing the CP group. EP1 keeps every MoE
        # on TP4; dense FFNs also use TP4 rather than per-rank computation.
        "--ep-size",
        "1",
        "--moe-dense-tp-size",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model("zai-org/GLM-5.3-Flash")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.server_args,
        )

    @classmethod
    def tearDownClass(cls):
        if process := getattr(cls, "process", None):
            terminate_and_kill_process_tree(process)
            _wait_for_gpu_idle_in_ci(timeout=120)

    def test_gsm8k(self):
        super().test_gsm8k()
        # Require the metric as well as the floor: an accuracy pass alone must
        # not hide disabled MTP or a draft whose tokens are mostly rejected.
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        self.assertGreater(
            response.json()["internal_states"][0]["avg_spec_accept_length"], 4.0
        )

    def test_image_input(self):
        # Color is supplied only by the image: the prompt cannot substitute for
        # the encoder. Exercise two images to catch missing/cached embeddings.
        for color in ("red", "blue"):
            with self.subTest(color=color):
                png = io.BytesIO()
                Image.new("RGB", (256, 256), color=color).save(png, format="PNG")
                image_url = (
                    "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
                )
                response = requests.post(
                    self.base_url + "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                    {
                                        "type": "text",
                                        "text": "What is the dominant color of this image? Answer with one English color word.",
                                    },
                                ],
                            }
                        ],
                        "temperature": 0,
                        # The checkpoint template always opens a thinking block.
                        # Allow it to finish before checking the final answer.
                        "max_tokens": 1024,
                    },
                    timeout=180,
                )
                self.assertEqual(response.status_code, 200, response.text)
                answer = response.json()["choices"][0]["message"]["content"]
                self.assertIn(color, answer.lower(), response.text)


if __name__ == "__main__":
    unittest.main()
