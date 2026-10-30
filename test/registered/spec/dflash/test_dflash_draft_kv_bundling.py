"""DFlash draft KV bundled with Mamba checkpoints, with a write-back host tier."""

import os
import re
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="nightly", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3.8-27B"
DRAFT_MODEL = "z-lab/Qwen3.8-27B-DFlash2"
# The checkpoint's sliding_window of 2048, less the current token.
DRAFT_WINDOW = 2047
PAGE_SIZE = 64
# Few enough Mamba slots that a few dozen cached prompts overflow them.
MAMBA_SLOTS = 64


def _prompt(entries: int) -> str:
    return " ".join(
        f"Entry {i}: the quick brown fox number {i} jumps over the lazy dog number {i + 1}."
        for i in range(entries)
    )


_QUESTION = (
    "\n\nWrite a detailed essay on how foxes and dogs are portrayed in folklore, "
    "with examples from at least three cultures."
)


class TestDFlashDraftKVBundling(CustomTestCase, GSM8KMixin):
    model = TARGET_MODEL
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 3.0
    gsm8k_num_questions = 200

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_dir = tempfile.mkdtemp(prefix="dflash_draft_kv_bundling_")
        cls.stdout = open(os.path.join(cls.log_dir, "server_stdout.log"), "w")
        cls.stderr = open(os.path.join(cls.log_dir, "server_stderr.log"), "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800),
            other_args=[
                "--trust-remote-code",
                "--speculative-algorithm=DFLASH",
                f"--speculative-draft-model-path={DRAFT_MODEL}",
                "--speculative-num-draft-tokens=8",
                f"--speculative-draft-window-size={DRAFT_WINDOW}",
                "--speculative-draft-kv-bundling",
                "--max-running-requests=8",
                f"--max-mamba-cache-size={MAMBA_SLOTS}",
                f"--page-size={PAGE_SIZE}",
                "--mem-fraction-static=0.8",
                "--enable-hierarchical-cache",
                "--hicache-write-policy=write_back",
                # Auto-sizing needs a visible cgroup memory limit.
                "--hicache-size=48",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        cls.stdout.close()
        cls.stderr.close()

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

    def test_pools_hold_one_window_per_mamba_slot(self):
        log = self._server_log()
        window = -(-DRAFT_WINDOW // PAGE_SIZE) * PAGE_SIZE
        slots, running, per_request, cap, cap_one, tokens = map(
            int,
            re.search(
                r"mamba slots=(\d+), running requests=(\d+) x (\d+) slots, "
                r"request cap=(\d+), one-request cap=(\d+), draft pool=(\d+) tokens",
                log,
            ).groups(),
        )
        self.assertEqual(slots, MAMBA_SLOTS)
        expected = slots * window + max(
            0,
            cap - running * per_request * window,
            cap_one - per_request * window,
        )
        self.assertEqual(tokens, -(-expected // PAGE_SIZE) * PAGE_SIZE)
        host_tokens, host_slots = map(
            int, re.search(r"host pool: (\d+) tokens \((\d+) host Mamba", log).groups()
        )
        self.assertLessEqual(host_tokens, host_slots * window + PAGE_SIZE)

    def test_hit_keeps_greedy_output_and_acceptance(self):
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        prompt = _prompt(700) + _QUESTION
        cold = self._generate(prompt, 256)
        warm = self._generate(prompt, 256)
        meta = warm["meta_info"]
        self.assertGreater(meta["cached_tokens"], meta["prompt_tokens"] - DRAFT_WINDOW)
        self.assertEqual(cold["text"], warm["text"])
        self.assertGreaterEqual(
            meta["spec_accept_length"], 0.9 * cold["meta_info"]["spec_accept_length"]
        )

    def test_evicted_bundle_is_restored_from_host(self):
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        # Longer than one draft window and shorter than one prefill chunk.
        prompts = [
            _prompt(95).replace("quick", f"quick-{i}") + _QUESTION
            for i in range(MAMBA_SLOTS + 8)
        ]
        self._generate(prompts[0], 64)
        resident = self._generate(prompts[0], 64)
        for prompt in prompts[1:]:
            self._generate(prompt, 8)

        warm = self._generate(prompts[0], 64)
        meta = warm["meta_info"]
        self.assertGreater((meta.get("cached_tokens_details") or {}).get("host", 0), 0)
        self.assertEqual(meta["cached_tokens"], resident["meta_info"]["cached_tokens"])
        self.assertEqual(warm["text"], resident["text"])
        self.assertGreaterEqual(
            meta["spec_accept_length"],
            0.8 * resident["meta_info"]["spec_accept_length"],
        )
        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
