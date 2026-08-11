"""B300 per-commit CI coverage for Kimi-K3 serving recipes.

Runs the Low Latency DSPARK recipe and the Balanced DCP/HiCache recipe on
eight B300 GPUs. Each server must preserve basic model quality on GSM8K, and
the Low Latency recipe must also preserve single-request decode performance.
"""

import base64
import hashlib
import io
import unittest

import openai
import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = (
    "/data/radixark/model-cache/hub/models--moonshotai--Kimi-K3/"
    "snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


def _stop_server(process):
    if process:
        kill_process_tree(process.pid)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


def _metric_sum(text, name, **required_labels):
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        labels, value = line.rsplit(" ", 1)
        if all(f'{key}="{label}"' in labels for key, label in required_labels.items()):
            total += float(value)
    return total


def _png_data_url():
    output = io.BytesIO()
    Image.new("RGB", (256, 192), (35, 90, 180)).save(output, format="PNG")
    payload = output.getvalue()
    return (
        "data:image/png;base64," + base64.b64encode(payload).decode(),
        "sha256:" + hashlib.sha256(payload).hexdigest(),
    )


class TestKimiK3B300LowLatency(GSM8KMixin, SpecDecodingMixin, CustomTestCase):
    """TP8 Low Latency recipe with DSPARK linear ReplaySSM speculation."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    # Gated on GSM8K rather than on test_bs_1_speed below: a 200-question
    # average holds steady when a numerics change moves where the single
    # greedy prompt hits EOS.
    gsm8k_accept_length_thres = 4.5
    # Both scale with how far that one greedy prompt runs, and speed is
    # end-to-end, so launch and TTFT are amortized over the output -- it sits
    # well below the steady decode rate the server logs. Coarse guards only.
    accept_length_thres = 4.0
    bs_1_speed_thres = 300

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--mem-fraction-static",
                "0.85",
                "--weight-loader-prefetch-checkpoints",
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "0.86",
                "--enable-metrics",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--speculative-dspark-block-size",
                "7",
                "--enable-linear-replayssm-spec",
            ],
        )

    def test_agent_turn_reuses_image_embedding(self):
        image_url, content_hash = _png_data_url()
        image_part = {
            "type": "image_url",
            "image_url": {"url": image_url, "content_hash": content_hash},
        }
        first_user = {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
        client = openai.Client(api_key="EMPTY", base_url=self.base_url + "/v1")
        first = client.chat.completions.create(
            model="default",
            messages=[first_user],
            temperature=0,
            max_tokens=16,
        )
        before_hot = requests.get(self.base_url + "/metrics", timeout=30).text

        second = client.chat.completions.create(
            model="default",
            messages=[
                first_user,
                {"role": "assistant", "content": "I see the image."},
                {"role": "user", "content": "What is its main color?"},
            ],
            temperature=0,
            max_tokens=16,
        )
        after_hot = requests.get(self.base_url + "/metrics", timeout=30).text

        self.assertTrue(first.choices)
        self.assertTrue(second.choices)
        for stage in ("processor", "transport", "vit"):
            self.assertEqual(
                _metric_sum(
                    after_hot,
                    "sglang:mm_cache_skipped_stages_total",
                    stage=stage,
                )
                - _metric_sum(
                    before_hot,
                    "sglang:mm_cache_skipped_stages_total",
                    stage=stage,
                ),
                1,
            )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


class TestKimiK3B300Balanced(GSM8KMixin, CustomTestCase):
    """TP8/DCP8 Balanced recipe with hierarchical cache."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--dcp-size",
                "8",
                "--mem-fraction-static",
                "0.85",
                "--weight-loader-prefetch-checkpoints",
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "7.21",
                "--enable-hierarchical-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


if __name__ == "__main__":
    unittest.main()
