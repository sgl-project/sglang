"""Mixed chunk prefill x speculative decoding, overlap scheduler.

One cell per supported algorithm (EAGLE3, DFLASH, DSPARK). Inside a mixed
step every running request degrades to a 1-token extend of its pending
bonus token and drafting resumes the next decode step; under overlap the
tail state is late-bound at forward entry. Regression guards for the
bring-up failure modes: tail rows dropped from attention metadata (the
spec seq_lens convention zeroed the tail's qo len - a hard crash on
flashinfer, silent kv-span truncation elsewhere), unwritten relay rows
read as tail inputs, and stale schedule-time tail state under overlap.
Chunked prefill is set small so eval prompts span multiple chunks and
mixing actually engages. The DSPARK cell also runs with
SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION=1, where running requests draft and
verify inside the mixed step instead.
"""

import json
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


class TestEagle3MixedChunk(
    Eagle3Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
):
    disable_overlap = False
    # Small chunks so eval prompts span several of them and mixing engages
    # (the Eagle3Base preset raises the fixture default to 1024).
    chunked_prefill_size = 128
    extra_args = ("--enable-mixed-chunk",)


class TestDFlashMixedChunk(GSM8KMixin, CustomTestCase):
    model = DEFAULT_TARGET_MODEL_DFLASH

    gsm8k_num_questions = 200
    # Observed 0.755-0.78 across local runs; accept length is the tighter guard.
    gsm8k_accuracy_thres = 0.70
    gsm8k_accept_length_thres = 2.8

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The dflash draft config derives a shorter context than the target.
        with envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    DEFAULT_DRAFT_MODEL_DFLASH,
                    "--enable-mixed-chunk",
                    "--chunked-prefill-size",
                    "128",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


DSPARK_TARGET_MODEL = "Qwen/Qwen3-14B"
DSPARK_DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"


class TestDSparkMixedChunk(GSM8KMixin, CustomTestCase):
    model = DSPARK_TARGET_MODEL
    extra_args = ()

    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 2.0

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "trtllm_mha" if is_sm100_supported() else "fa3",
                "--speculative-draft-attention-backend",
                "fa4" if is_sm100_supported() else "fa3",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                "128",
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--cuda-graph-backend-prefill=disabled",
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


class TestDSparkVerifyMergedMixedChunk(TestDSparkMixedChunk):
    extra_args = ("--enable-metrics",)

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION.override(True):
            super().setUpClass()

    def _generate(self, text, *, return_sampling_mask, max_new_tokens, started=None):
        """Streams the request, setting `started` on its first output chunk.
        Returns the time its final chunk arrived."""
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0.8,
                "top_k": 8,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_sampling_mask": return_sampling_mask,
            "stream": True,
        }
        if return_sampling_mask:
            payload["sampling_logprobs_mode"] = "support"
        with requests.post(
            f"{self.base_url}/generate", json=payload, stream=True, timeout=120
        ) as response:
            # Not assertEqual(..., response.text): that reads the whole stream.
            if response.status_code != 200:
                self.fail(response.text)
            for line in response.iter_lines():
                if line.startswith(b"data: {"):
                    last = line
                    if started is not None:
                        started.set()
        finished_at = time.monotonic()
        output = json.loads(last[len(b"data: ") :])
        self.assertNotIn("error", output)
        self.assertEqual(len(output["output_ids"]), max_new_tokens)
        if return_sampling_mask:
            masks = output["meta_info"]["output_token_sampling_mask"]
            self.assertEqual(len(masks), max_new_tokens)
        return finished_at

    def _merged_steps(self):
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        match = re.search(
            r"^sglang:mixed_verify_steps_total(?:\{[^}]*\})? (\S+)$", metrics, re.M
        )
        return float(match.group(1)) if match else 0.0

    def test_sampling_mask_with_chunked_prefill(self):
        # A multi-chunk prefill arrives while a request decodes. Mask requests,
        # on either side, must keep the step unmerged and get their masks; the
        # no-mask case is the control that the same overlap does merge.
        cases = ((False, False), (True, False), (False, True), (True, True))
        for i, (prefill_mask, running_mask) in enumerate(cases):
            with self.subTest(prefill_mask=prefill_mask, running_mask=running_mask):
                merged_before = self._merged_steps()
                decoding = threading.Event()
                with ThreadPoolExecutor(2) as pool:
                    running = pool.submit(
                        self._generate,
                        f"{i}. Count from one to two thousand:",
                        return_sampling_mask=running_mask,
                        max_new_tokens=1024,
                        started=decoding,
                    )
                    self.assertTrue(decoding.wait(timeout=120))
                    # A distinct first token keeps the radix cache from
                    # shortening the prefill to one chunk.
                    prefill = pool.submit(
                        self._generate,
                        f"{i}. " + "The quick brown fox jumps over the lazy dog. " * 60,
                        return_sampling_mask=prefill_mask,
                        max_new_tokens=8,
                    )
                    prefill_done = prefill.result()
                    running_done = running.result()
                # The running request was in the running batch for every
                # prefill chunk.
                self.assertGreater(running_done, prefill_done)
                merged = self._merged_steps() - merged_before
                if prefill_mask or running_mask:
                    self.assertEqual(merged, 0)
                else:
                    self.assertGreater(merged, 0)

    def test_zz_verify_merged_steps_ran(self):
        # Runs after the eval, whose concurrent requests mix prefill chunks
        # with running (verifying) requests.
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        self.assertRegex(metrics, r"sglang:mixed_verify_steps_total(\{[^}]*\})? [1-9]")


if __name__ == "__main__":
    unittest.main()
