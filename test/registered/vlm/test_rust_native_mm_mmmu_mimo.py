"""MMMU accuracy gate for the Rust server's native MiMo-V2.5 multimodal path.

The CPU parity units (``test/registered/unit/multimodal/rust/mimo``) pin the
Rust pixel/layout math against the Python ``MiMoV2Processor``; this test pins
that the whole native path (``SGLANG_RUST_SERVER=1``: rust ingress →
mm workers → sidecar drain → TP scheduler) produces *equally good* model
inputs on the real checkpoint. A systematic preprocessing skew (wrong resample
kernel, normalization scale, patch layout, position contract) still yields
fluent text but drops MMMU accuracy below the gate.

The rust server does not serve ``/v1/chat/completions`` yet, so the eval
drives the native ``/generate`` endpoint with hand-rendered MiMo chat prompts
(same ``<|im_start|>``/vision-block format as Qwen, plus an empty
``<think></think>`` block — the template's ``enable_thinking=false``
rendering — so answers fit the token budget).
"""

import importlib.util
import os
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.simple_eval_common import MessageList, SamplerBase
from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    dump_metric,
    popen_launch_server,
)

register_cuda_ci(est_time=2400, stage="base-c", runner_config="4-gpu-b200")

MODEL = "XiaomiMiMo/MiMo-V2.5"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

NUM_EXAMPLES = 100
# Calibrated 2026-07-28 on B300 (TP4, triton attention, temperature 0): the
# native rust MM path scores 0.58 on this fixed 100-sample subset, matching
# the Python tokenizer-manager reference (0.56, same sampler and samples);
# gate leaves headroom for batching nondeterminism.
MMMU_ACCURACY_THRESHOLD = 0.50


def chat_prompt(segments):
    return (
        "<|im_start|>user\n"
        + "".join(segments)
        + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )


class MimoGenerateVisionSampler(SamplerBase):
    """Drive ``/generate`` with MiMo chat prompts and ``image_data``.

    ``MMMUVLMEval`` emits OpenAI-style messages whose content mixes ``text``
    and ``image_url`` parts. The rust server has no chat-completions route, so
    this sampler renders the MiMo chat format by hand — each image part
    becomes a vision block at its original position — and ships the images
    through ``image_data``.
    """

    def __init__(self, base_url: str, max_tokens: int = 1024):
        self.generate_url = base_url + "/generate"
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> str:
        segments = []
        images = []
        for message in message_list:
            content = message["content"]
            parts = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
            )
            for part in parts:
                if part["type"] == "image_url":
                    images.append(part["image_url"]["url"])
                    segments.append(VISION_BLOCK)
                else:
                    segments.append(part["text"])
        payload = {
            "text": chat_prompt(segments),
            "image_data": images,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_tokens,
            },
        }
        # Retry transient failures, but fail loudly when they persist: silently
        # returning "" would degrade the score and blur the accuracy gate.
        for attempt in range(3):
            try:
                response = requests.post(self.generate_url, json=payload, timeout=600)
                response.raise_for_status()
                return response.json()["text"]
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)


@unittest.skipIf(
    importlib.util.find_spec("sglang.srt.server._core") is None,
    "sglang-server rust extension not installed (e.g. AMD suite)",
)
class TestRustNativeMmMMMUMimo(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Capture the server log so the test can pin that the native MM
        # pipeline is active.
        cls.log_dir = tempfile.TemporaryDirectory()
        cls.server_logs = tuple(
            open(os.path.join(cls.log_dir.name, name), "w")
            for name in ("stdout.log", "stderr.log")
        )
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # MiMo-V2.5's fused qkv weights are TP=4-interleaved, so plain
            # --tp 4 is the smallest launch the rust server serves (the DP
            # -attention layout the python-server e2e uses is out of scope
            # here). Triton attention is pinned because the Blackwell default
            # (trtllm_mha) fails cuda-graph capture on this model's hybrid-SWA
            # KV layout; fa3 (the H200 CI choice) does not build on Blackwell.
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--enable-multimodal",
                "--mem-fraction-static",
                "0.65",
                "--attention-backend",
                "triton",
            ],
            env={**os.environ, "SGLANG_RUST_SERVER": "1"},
            return_stdout_stderr=cls.server_logs,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "server_logs"):
            for f in cls.server_logs:
                f.close()
        if hasattr(cls, "log_dir"):
            cls.log_dir.cleanup()

    def _read_server_log(self):
        text = []
        for f in self.server_logs:
            with open(f.name) as reader:
                text.append(reader.read())
        return "\n".join(text)

    def test_mmmu_accuracy(self):
        # Guard the path under test: if the family ever drops out of
        # NativeMmHost's spec resolution, launch fails and this assertion
        # names the missing log line explicitly.
        self.assertIn(
            "native MM pipeline enabled (family=mimo_v2)",
            self._read_server_log(),
            "rust server did not enable the native mimo_v2 MM pipeline for "
            f"{MODEL}; this test must exercise the native path",
        )

        eval_obj = MMMUVLMEval(num_examples=NUM_EXAMPLES, num_threads=32)
        sampler = MimoGenerateVisionSampler(base_url=DEFAULT_URL_FOR_TEST)
        result = eval_obj(sampler)
        print(f"MMMU metrics: {result.metrics}")
        dump_metric(
            "mmmu_score",
            result.score,
            labels={"model": MODEL, "eval": "mmmu", "api": "generate-rust-native-mm"},
        )
        self.assertGreaterEqual(
            result.score,
            MMMU_ACCURACY_THRESHOLD,
            f"Rust native MM path scored {result.score:.4f} on MMMU, below the "
            f"{MMMU_ACCURACY_THRESHOLD:.2f} gate",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
