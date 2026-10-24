"""--speculative-token-map under TP>1: hot rows must assemble from the
vocab-sharded target lm_head instead of indexing it with global ids.

Historically the draft bound ``head.data[hot_token_id]`` on each rank's
~vocab/tp shard, crashing init at TP>1 with a device-side assert
(vectorized_gather_kernel, #42397). TP=1 worked only for unquantized heads;
quantized (block-scale) heads silently bound garbage packed rows.

The token map here is generated on the fly, spanning both TP shards evenly
so the cross-rank assembly is exercised. Correctness is asserted as healthy
greedy generation with the speculative path active (the hot subset is not
frequency-curated, so only accept-length sanity is claimed, not parity).
"""

import os
import tempfile
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

MODEL = "Qwen/Qwen3.5-9B"
NUM_HOT = 16384
SERVER_LAUNCH_TIMEOUT = 600

_TOKEN_MAP_PATH = None


def _write_spanning_token_map():
    """A deterministic hot set spread evenly over [0, vocab) so every TP
    shard contributes rows to the assembled head."""
    global _TOKEN_MAP_PATH
    if _TOKEN_MAP_PATH is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        vocab = getattr(config, "vocab_size", None) or config.text_config.vocab_size
        hot = torch.linspace(0, vocab - 1, NUM_HOT).long()
        assert hot.unique().numel() == NUM_HOT
        fd, _TOKEN_MAP_PATH = tempfile.mkstemp(suffix=".pt")
        with os.fdopen(fd, "w") as _:
            pass
        torch.save(hot.tolist(), _TOKEN_MAP_PATH)
    return _TOKEN_MAP_PATH


def _launch_args(tp):
    return [
        "--trust-remote-code",
        "--tp",
        str(tp),
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-token-map",
        _write_spanning_token_map(),
        "--mem-fraction-static",
        "0.8",
        "--disable-radix-cache",
    ]


class _TokenMapServerTestBase(CustomTestCase):
    TP = 2

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=_launch_args(cls.TP),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    PROMPTS = (
        "The capital of France is",
        "Write a python function to reverse a string:",
        "1+1=2, 2+2=4, 4+4=",
    )

    def _greedy(self, prompt, max_new_tokens=48):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        return response.json()

    def test_greedy_generation_is_healthy(self):
        for prompt in self.PROMPTS:
            with self.subTest(prompt=prompt):
                r = self._greedy(prompt)
                self.assertIn("text", r, r)
                self.assertTrue(r["text"].strip(), r)
                # Deterministic sampling: a repeated greedy call must match.
                self.assertEqual(r["text"], self._greedy(prompt)["text"])

    def test_speculative_path_is_active(self):
        total_completion = 0
        total_verify = 0
        for prompt in self.PROMPTS:
            meta = self._greedy(prompt)["meta_info"]
            total_completion += meta["completion_tokens"]
            total_verify += meta.get("spec_verify_ct", 0)
        self.assertGreater(total_verify, 0, "spec verify never ran")
        acc_length = total_completion / total_verify
        print(f"{acc_length=:.4f}")
        # The map is synthetic (not frequency-curated), so most drafts fall
        # outside the hot set; only sanity is claimed: something got accepted.
        self.assertGreater(acc_length, 1.0)


class TestSpecTokenMapTP2(_TokenMapServerTestBase):
    """The regression: TP=2 + token map used to crash in init_lm_head."""

    TP = 2


class TestSpecTokenMapTP1(_TokenMapServerTestBase):
    """TP=1 control: the fast path (single-shard row select) keeps working."""

    TP = 1


if __name__ == "__main__":
    unittest.main()
