"""End-to-end test for --speculative-draft-kv-ratio.

Pairs a linear/full-attention target with an all-SWA DFLASH draft:
``Qwen/Qwen3.8-27B`` and ``z-lab/Qwen3.8-27B-DFlash2`` (5 ``sliding_attention``
layers, ``sliding_window`` 2048, block size 8). Below 1.0 the draft's KV holds
a fraction of the target's tokens in a sliding-window pool instead of one slot
per target token, and a radix prefix hit still reuses the whole cached prefix.
"""

import os
import re
import tempfile
import unittest
from contextlib import ExitStack

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=500, stage="nightly", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3.8-27B"
DRAFT_MODEL = "z-lab/Qwen3.8-27B-DFlash2"
# The checkpoint's sliding_window of 2048, less the current token.
DRAFT_WINDOW = 2047
DRAFT_BLOCK_SIZE = 8
DRAFT_KV_RATIO = 0.25
MAX_RUNNING_REQUESTS = 8
# The target resumes a prefix hit at its last linear-attention checkpoint,
# which is chunk aligned; allow one chunk of slack before the prompt end.
CHECKPOINT_SLACK = 128

# Several draft windows long, so a repeated request is a prefix hit whose
# draft window lies well inside cached territory.
_LONG_PROMPT = " ".join(
    f"Entry {i}: the quick brown fox number {i} jumps over the lazy dog number {i + 1}."
    for i in range(700)
)
_ESSAY_QUESTION = (
    "\n\nWrite a detailed essay on how foxes and dogs are portrayed in folklore, "
    "with examples from at least three cultures."
)


class TestDFlashDraftKVRatio(CustomTestCase, GSM8KMixin):
    model = TARGET_MODEL
    draft_model = DRAFT_MODEL
    page_size = 1
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 3.0
    gsm8k_num_questions = 200

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_dir = tempfile.mkdtemp(prefix="dflash_draft_kv_ratio_")
        cls.stdout = open(os.path.join(cls.log_dir, "server_stdout.log"), "w")
        cls.stderr = open(os.path.join(cls.log_dir, "server_stderr.log"), "w")
        launch_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-draft-tokens",
            str(DRAFT_BLOCK_SIZE),
            "--speculative-draft-kv-ratio",
            str(DRAFT_KV_RATIO),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--page-size",
            str(cls.page_size),
            "--mem-fraction-static",
            "0.8",
        ]
        with ExitStack() as stack:
            for env, value in (
                (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
                (envs.SGLANG_ENABLE_ASYNC_ASSERT, True),
            ):
                stack.enter_context(env.override(value))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800),
                other_args=launch_args,
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        for f in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if f is not None:
                f.close()

    def _server_log(self) -> str:
        self.stdout.flush()
        self.stderr.flush()
        text = ""
        for name in ("server_stdout.log", "server_stderr.log"):
            with open(os.path.join(self.log_dir, name)) as f:
                text += f.read()
        return text

    def _generate(self, prompt: str, max_new_tokens: int) -> dict:
        res = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
        )
        res.raise_for_status()
        return res.json()

    def _flush_cache(self):
        requests.post(self.base_url + "/flush_cache").raise_for_status()

    def _pool_sizing(self) -> tuple:
        """(window, ratio, draft tokens, request cap) from the startup log."""
        sizing = re.search(
            r"DFLASH draft KV pool: window=(\d+), ratio=([0-9.]+), tokens=(\d+) "
            r"\(request cap (\d+)",
            self._server_log(),
        )
        self.assertIsNotNone(sizing, "draft KV pool sizing not logged")
        window, ratio, tokens, cap = sizing.groups()
        return int(window), float(ratio), int(tokens), int(cap)

    def test_draft_pool_is_ratio_of_target(self):
        window, ratio, draft_tokens, cap = self._pool_sizing()
        self.assertEqual(window, DRAFT_WINDOW)
        self.assertEqual(ratio, DRAFT_KV_RATIO)
        self.assertGreaterEqual(draft_tokens, cap)
        log = self._server_log()
        target_tokens = int(re.search(r"max_total_num_tokens=(\d+)", log).group(1))
        self.assertAlmostEqual(
            draft_tokens / target_tokens,
            DRAFT_KV_RATIO,
            delta=self.page_size / target_tokens,
        )
        # The draft pool is built with that many slots.
        pools = re.findall(r"SWAKVPool .*swa size: (\d+), full size: (\d+)", log)
        self.assertIn(draft_tokens, [int(swa) for swa, _ in pools])

    def test_prefix_hit_reuses_whole_prompt(self):
        """A repeated long prompt is served almost entirely from the cache."""
        self._flush_cache()
        cold = self._generate(_LONG_PROMPT, 8)
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        self.assertGreater(prompt_tokens, 2 * DRAFT_WINDOW)
        warm = self._generate(_LONG_PROMPT, 8)
        cached_tokens = warm["meta_info"]["cached_tokens"]
        print(
            f"prefix hit: prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
        )
        self.assertGreater(cached_tokens, prompt_tokens - DRAFT_WINDOW)
        self.assertGreaterEqual(
            cached_tokens, prompt_tokens - CHECKPOINT_SLACK - self.page_size
        )

    def test_greedy_determinism_across_hit(self):
        self._flush_cache()
        prompt = _LONG_PROMPT + _ESSAY_QUESTION
        cold = self._generate(prompt, 256)
        warm = self._generate(prompt, 256)
        self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
        self.assertEqual(cold["text"], warm["text"])
        self.assertIsNone(self.process.poll())

    def test_evicted_window_falls_back(self):
        """More cached prefixes than the draft pool holds: the oldest loses its
        window, and a hit on it re-prefills instead of reading freed slots."""
        self._flush_cache()
        _, _, draft_tokens, cap = self._pool_sizing()
        # One more than the pool holds, at the window each cached prefix keeps.
        num_prompts = (draft_tokens - cap) // DRAFT_WINDOW + 2
        prompts = [
            _LONG_PROMPT.replace("quick", f"quick-{i}") + _ESSAY_QUESTION
            for i in range(num_prompts)
        ]
        cold = self._generate(prompts[0], 64)
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        for prompt in prompts[1:]:
            self._generate(prompt, 8)
        warm = self._generate(prompts[0], 64)
        cached_tokens = warm["meta_info"]["cached_tokens"]
        print(
            f"after {num_prompts} prefixes: prompt_tokens={prompt_tokens}, "
            f"cached_tokens={cached_tokens}"
        )
        self.assertLessEqual(cached_tokens, prompt_tokens - DRAFT_WINDOW)
        self.assertEqual(cold["text"], warm["text"])
        self.assertIsNone(self.process.poll())


class TestDFlashDraftKVRatioPaged(TestDFlashDraftKVRatio):
    """The same pool at page size 64, where the allocator pages both sides and
    the draft backend builds its page table from translated slots."""

    page_size = 64

    def test_gsm8k(self):
        self.skipTest("accuracy does not depend on the page size")


if __name__ == "__main__":
    unittest.main()
