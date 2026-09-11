"""DSV4-Flash 8-GPU PD-Multiplexing concurrency test (TP8, no spec decoding).

Covers the PDMux-differentiating paths the serial sanity cannot reach:

- P/D true concurrency: a streaming decode request is confirmed running before
  long prompts are injected, so running_batch and split_prefill_batch are
  non-empty together and the shared green-context SM groups get selected.
- Cross-segment split prefill: split_forward_token_budget=8192 with ~4k-token
  prompts forces the layer-split state (hidden/residual + mHC carries) to
  survive many segments on real tensors.
- Dual NCCL communicators in flight + per-call sampler TP-group resolution
  (SYNC_TOKEN_IDS_ACROSS_TP=1 puts an all-reduce on every decode step and on
  the split-prefill final sample).
- Per-SM-group decode graph replay: manual division thresholds spread 1-8
  concurrent decodes across different shared groups.

Assertion -> path map:
  (i)   streaming prefix matches serial baseline  -> concurrent prefill does
        not corrupt decode (SM/KV isolation, communicator ordering)
  (ii)  needle answers correct per request        -> cross-segment numerics +
        cross-request KV isolation (unique code per request)
  (iii) all requests finish within the deadline   -> no dual-communicator or
        prefill-handshake deadlock
  (iv)  post-storm sanity answer                  -> finalize merge into a
        non-empty running batch (GPU-level)

Model path can be overridden for offline containers:
  SGLANG_TEST_DSV4_FLASH_MODEL_PATH=/model python3 <this file>
"""

import concurrent.futures
import json
import os
import tempfile
import threading
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = os.environ.get(
    "SGLANG_TEST_DSV4_FLASH_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)

DSV4_FLASH_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    # Pure-TP (no deepep A2A) resolves the auto MoE runner to Triton, which
    # cannot execute mxfp4-packed experts; dequant them to FP8 at load. No-op
    # for checkpoints whose experts are already FP8.
    "SGLANG_DSV4_FP4_DEQUANT": "1",
    # The sampler reads this unprefixed name (sampler.py SYNC_TOKEN_IDS_ACROSS_TP):
    # forces a TP all-reduce with per-call group resolution on every decode step
    # and on the split-prefill final sample.
    "SYNC_TOKEN_IDS_ACROSS_TP": "1",
}
if not os.path.isdir(DSV4_FLASH_MODEL_PATH):
    # Local checkpoints let model_config auto-detect the routed-expert layout
    # (mxfp4-packed vs converted FP8) from the safetensors header. For the HF
    # slug the header may not be cached yet, so pin the FP8 layout the
    # sgl-project/DeepSeek-V4-Flash-FP8 repo uses.
    DSV4_FLASH_ENV["SGLANG_DSV4_FP4_EXPERTS"] = "0"

# Sized for H20 (78 SMs); prefill counts are multiples of 8 for Hopper
# green-context granularity. On larger GPUs the remaining SMs stay unused.
# Thresholds 1..8 spread 1-8 concurrent decodes across different shared groups.
# budget 8192 with ~4k-token prompts => forward_count=2 => >20 segments.
PDMUX_YAML = """\
sm_group_num: 8
split_forward_token_budget: 8192
manual_divisions:
  - [56, 22, 1]
  - [48, 30, 2]
  - [40, 38, 3]
  - [32, 46, 4]
  - [24, 54, 6]
  - [16, 62, 8]
"""

STREAM_PROMPT = "Write a detailed explanation of how photosynthesis works."
FILLER = "The quick brown fox jumps over the lazy dog near the river bank. "
NEEDLE_CODES = ["7391", "2846", "5017", "9463"]


def _needle_prompt(code: str) -> str:
    # ~4000 tokens of filler between the needle and the question so the
    # split prefill spans many segments and long-range attention is exercised.
    filler = FILLER * 320
    return (
        f"Remember this secret code: {code}.\n{filler}\n"
        "What is the secret code mentioned at the beginning? "
        "Answer with the digits only:"
    )


class TestDSV4FlashTP8PDMuxConcurrent(CustomTestCase):
    """TP8, PD-Multiplexing enabled, concurrent P/D load."""

    @classmethod
    def setUpClass(cls):
        config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        config.write(PDMUX_YAML)
        config.close()
        cls.pdmux_config_path = config.name
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DSV4_FLASH_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-pdmux",
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "-1",
                "--pdmux-config-path",
                cls.pdmux_config_path,
                "--max-running-requests",
                "8",
                "--mem-fraction-static",
                "0.85",
            ],
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.unlink(cls.pdmux_config_path)

    def _generate(self, text: str, max_new_tokens: int) -> str:
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["text"]

    def test_concurrent_prefill_does_not_corrupt_decode(self):
        # Serial baseline on the same server (the serial pdmux path is covered
        # by the sanity test); used for the prefix comparison in (i).
        baseline_text = self._generate(STREAM_PROMPT, 256)

        first_token_seen = threading.Event()
        stream_state = {"text": "", "error": None}

        def stream_worker():
            try:
                with requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": STREAM_PROMPT,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 256,
                        },
                        "stream": True,
                    },
                    stream=True,
                    timeout=600,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            break
                        stream_state["text"] = json.loads(payload)["text"]
                        first_token_seen.set()
            except Exception as e:  # surfaced via assertion below
                stream_state["error"] = e
                first_token_seen.set()

        stream_thread = threading.Thread(target=stream_worker)
        stream_thread.start()
        self.assertTrue(
            first_token_seen.wait(timeout=180), "decode stream never started"
        )
        self.assertIsNone(stream_state["error"], f"stream failed: {stream_state['error']}")

        # Inject long prompts while the stream is decoding: running_batch and
        # split_prefill_batch are now concurrently non-empty.
        with concurrent.futures.ThreadPoolExecutor(len(NEEDLE_CODES)) as pool:
            futures = [
                pool.submit(self._generate, _needle_prompt(code), 32)
                for code in NEEDLE_CODES
            ]
            done, not_done = concurrent.futures.wait(futures, timeout=300)
            # (iii) deadlock detection
            self.assertFalse(
                not_done, "long-prompt requests timed out (possible deadlock)"
            )

        stream_thread.join(timeout=300)
        self.assertFalse(stream_thread.is_alive(), "stream request never finished")
        self.assertIsNone(stream_state["error"], f"stream failed: {stream_state['error']}")

        # (ii) cross-segment numerics + cross-request KV isolation.
        for code, future in zip(NEEDLE_CODES, futures):
            self.assertIn(
                code,
                future.result(),
                f"needle {code} lost across split-prefill segments",
            )

        # (i) prefix comparison only: once the long prompts merge into decode,
        # the batch size changes and greedy outputs may legitimately diverge;
        # the early tokens are decoded while the long prompts are still
        # prefilling on the other SM partition.
        prefix_len = 200
        self.assertEqual(
            baseline_text[:prefix_len],
            stream_state["text"][:prefix_len],
            "concurrent prefill corrupted decode output",
        )

        # (iv) post-storm sanity: finalize merged into a non-empty running
        # batch during the storm; the server must still be healthy.
        after_text = self._generate("The capital of France is", 32)
        self.assertIn("Paris", after_text)


if __name__ == "__main__":
    unittest.main()
