"""B200 per-commit coverage for the GLM-5.3-Flash serving recipes.

Runs the Low Latency, DFlash2, High Throughput, and Prefill CP + MTP recipes on
four B200 GPUs. All recipes must retain GSM8K accuracy; CP + MTP also checks
speculative acceptance, and Low Latency checks single-request decode performance.
"""

import base64
import io
import unittest

import requests
from PIL import Image

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=3000, stage="base-c", runner_config="4-gpu-b200")

MODEL_PATH = "zai-org/GLM-5.3-Flash"
DFLASH2_DRAFT_MODEL_PATH = "incoai/GLM-5.3-Flash-DFlash2"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120

COMMON_SERVER_ARGS = [
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
]


def _stop_server(process):
    if process:
        terminate_and_kill_process_tree(process)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class _GLM53FlashB200Base(CustomTestCase):
    server_args: list[str]

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=cls.server_args,
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


class TestGLM53FlashB200LowLatency(
    SpecDecodingMixin,
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    gsm8k_score_threshold = 0.93
    # Match the established DSA+MTP accuracy workload. The generic 200-question,
    # 5-shot defaults leave a single question worth 0.5 percentage points and
    # make this tight quality floor unnecessarily sensitive to kernel numerics.
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    accept_length_thres = 4.0
    bs_1_speed_thres = 250
    server_args = [
        *COMMON_SERVER_ARGS,
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]


class TestGLM53FlashB200HighThroughput(
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    gsm8k_score_threshold = 0.93
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    server_args = [
        *COMMON_SERVER_ARGS,
        "--attn-dp-size",
        "4",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--mm-enable-dp-encoder",
    ]


class TestGLM53FlashB200DFlash2(
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    gsm8k_score_threshold = 0.93
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    server_args = [
        *COMMON_SERVER_ARGS,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DFLASH2_DRAFT_MODEL_PATH,
        "--speculative-draft-attention-backend",
        "fa4",
    ]


class TestGLM53FlashB200ContextParallel(
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    gsm8k_score_threshold = 0.93
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    server_args = [
        *TestGLM53FlashB200LowLatency.server_args,
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
