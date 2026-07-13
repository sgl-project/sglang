"""LoRA under the breakable (BCG) prefill CUDA graph.

For each LoRA kernel backend (csgmv and triton), launches the same
LoRA-enabled server twice — once with --cuda-graph-backend-prefill=breakable
and once with =disabled — and compares per-token prompt logprobs for:
  * single LoRA requests across prompt lengths spanning several token buckets,
  * base-model (no-adapter) requests served by the same LoRA-enabled server,
  * one multi-request batch mixing LoRA and base requests (bs > 1 replay).

Two guards against passing vacuously:
  * scrapes the breakable server's logs for a "Prefill batch ... cuda graph:
    True" line — the scheduler reports can_run_cuda_graph per batch, which is
    True only when PrefillCudaGraphRunner.execute actually served the batch;
  * asserts the adapter materially changes prompt logprobs while the prefill
    graph is enabled, which catches a graph captured without LoRA applied.
"""

import os
import re
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-large")

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_ADAPTER = "nissenj/Qwen3-4B-lora-v2"
LORA_NAME = "lora0"

# Max absolute per-token prompt logprob difference tolerated between the
# graph-enabled and graph-disabled runs (bf16; graph replay pads the token
# axis up to the captured bucket, which can perturb GEMM tiling slightly).
GRAPH_VS_EAGER_TOLERANCE = 1e-1
# The adapter must move at least one prompt logprob by this much, otherwise
# LoRA was not applied under the graph at all.
LORA_EFFECT_THRESHOLD = 5e-2

# Prompt lengths chosen to land in different captured token buckets.
PROMPTS = [
    "What is the capital of France?",
    "List three benefits of regular exercise. " * 8,
    "The quick brown fox jumps over the lazy dog. " * 40,
]

MIXED_BATCH_PROMPTS = [
    "Summarize the plot of Romeo and Juliet.",
    "Explain how photosynthesis works. " * 6,
    "Write a haiku about the sea.",
    "Describe the water cycle in detail. " * 12,
]
MIXED_BATCH_LORA_PATHS = [LORA_NAME, None, LORA_NAME, None]

PREFILL_GRAPH_REPLAY_PATTERN = re.compile(r"Prefill batch.*cuda graph: True")


def _generate(text, lora_path):
    resp = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "text": text,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "lora_path": lora_path,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _prompt_logprobs(out):
    # First entry has no logprob (no preceding context).
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def _max_abs_diff(a, b):
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    assert ta.shape == tb.shape, f"token count mismatch: {ta.shape} vs {tb.shape}"
    return (ta - tb).abs().max().item()


class BCGLoRAServerMixin:
    """Launches breakable + disabled servers for one LoRA kernel backend and
    collects prompt logprobs from both. Concrete classes set lora_backend."""

    lora_backend: str

    @classmethod
    def _server_args(cls, prefill_backend):
        return [
            "--enable-lora",
            "--lora-paths",
            f"{LORA_NAME}={LORA_ADAPTER}",
            "--max-loras-per-batch",
            "2",
            "--lora-backend",
            cls.lora_backend,
            "--cuda-graph-backend-prefill",
            prefill_backend,
            "--cuda-graph-max-bs-prefill",
            "1024",
            "--cuda-graph-max-bs-decode",
            "8",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.8",
            "--random-seed",
            "42",
        ]

    @classmethod
    def _collect(cls):
        return {
            "lora": [_prompt_logprobs(_generate(p, LORA_NAME)) for p in PROMPTS],
            "base": [_prompt_logprobs(_generate(p, None)) for p in PROMPTS],
            "mixed": [
                _prompt_logprobs(out)
                for out in _generate(MIXED_BATCH_PROMPTS, MIXED_BATCH_LORA_PATHS)
            ],
        }

    @classmethod
    def setUpClass(cls):
        cls.stdout_path = f"bcg_lora_{cls.lora_backend}_server.out"
        cls.stderr_path = f"bcg_lora_{cls.lora_backend}_server.err"
        cls.stdout = open(cls.stdout_path, "w")
        cls.stderr = open(cls.stderr_path, "w")
        process = popen_launch_server(
            BASE_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._server_args("breakable"),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        try:
            cls.with_graph = cls._collect()
        finally:
            kill_process_tree(process.pid)
            cls.stdout.close()
            cls.stderr.close()

        process = popen_launch_server(
            BASE_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._server_args("disabled"),
        )
        try:
            cls.without_graph = cls._collect()
        finally:
            kill_process_tree(process.pid)

    @classmethod
    def tearDownClass(cls):
        for attr in ("stdout", "stderr"):
            f = getattr(cls, attr, None)
            if f is not None and not f.closed:
                f.close()
        for attr in ("stdout_path", "stderr_path"):
            path = getattr(cls, attr, None)
            if path and os.path.exists(path):
                os.remove(path)

    def test_prefill_graph_replay_logged(self):
        # can_run_cuda_graph is True in the scheduler's per-batch prefill log
        # only when PrefillCudaGraphRunner.execute served the batch; without
        # this, an init-time or per-batch fallback to eager prefill would make
        # the equivalence tests below pass vacuously.
        logs = ""
        for path in (self.stdout_path, self.stderr_path):
            with open(path) as f:
                logs += f.read()
        self.assertRegex(
            logs,
            PREFILL_GRAPH_REPLAY_PATTERN,
            "No prefill batch was served by the prefill CUDA graph; the "
            "breakable-vs-disabled comparison is vacuous.",
        )

    def test_lora_prefill_logprobs_match_eager(self):
        for i, prompt in enumerate(PROMPTS):
            diff = _max_abs_diff(
                self.with_graph["lora"][i], self.without_graph["lora"][i]
            )
            print(f"[lora] prompt {i} ({len(prompt)} chars): max_abs_diff={diff:.5f}")
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"LoRA prompt logprobs diverge between breakable prefill graph "
                f"and eager prefill for prompt {i}",
            )

    def test_base_prefill_logprobs_match_eager(self):
        for i, prompt in enumerate(PROMPTS):
            diff = _max_abs_diff(
                self.with_graph["base"][i], self.without_graph["base"][i]
            )
            print(f"[base] prompt {i} ({len(prompt)} chars): max_abs_diff={diff:.5f}")
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"Base-model prompt logprobs diverge between breakable prefill "
                f"graph and eager prefill for prompt {i}",
            )

    def test_mixed_batch_matches_eager(self):
        for i in range(len(MIXED_BATCH_PROMPTS)):
            diff = _max_abs_diff(
                self.with_graph["mixed"][i], self.without_graph["mixed"][i]
            )
            print(
                f"[mixed] req {i} (lora={MIXED_BATCH_LORA_PATHS[i]}): "
                f"max_abs_diff={diff:.5f}"
            )
            self.assertLess(
                diff,
                GRAPH_VS_EAGER_TOLERANCE,
                f"Mixed LoRA/base batch logprobs diverge between breakable "
                f"prefill graph and eager prefill for request {i}",
            )

    def test_lora_is_applied_under_graph(self):
        # If the captured graph dropped the LoRA kernels (or read stale
        # metadata), LoRA and base logprobs would coincide.
        for i in range(len(PROMPTS)):
            diff = _max_abs_diff(self.with_graph["lora"][i], self.with_graph["base"][i])
            print(f"[effect] prompt {i}: lora-vs-base max_abs_diff={diff:.5f}")
            self.assertGreater(
                diff,
                LORA_EFFECT_THRESHOLD,
                f"Adapter did not change prompt logprobs under the breakable "
                f"prefill graph for prompt {i}; LoRA was likely not applied.",
            )


class TestBCGLoRACsgmv(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "csgmv"


class TestBCGLoRATriton(BCGLoRAServerMixin, CustomTestCase):
    lora_backend = "triton"


if __name__ == "__main__":
    unittest.main()
