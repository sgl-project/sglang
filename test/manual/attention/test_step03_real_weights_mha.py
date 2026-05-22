"""
Step-03 E2E tests with real MHA model weights.

Uses meta-llama/Llama-3.2-1B-Instruct (DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
and meta-llama/Llama-3.2-1B (base) as available.  Tests two distinct things:

1. Accuracy — each (backend, runner) combination produces correct answers
   to simple factual questions via /v1/chat/completions.

2. Cross-runner output consistency — triton eager / full-CG / PCG must
   produce *identical* greedy outputs for the same raw-completion prompts.
   This is the key invariant step-03 must preserve: the init-API unification
   must not perturb attention outputs.

Run on the cluster (mounts sgl-workspace as the code source):
  python test_step03_real_weights_mha.py                    # all tests
  python test_step03_real_weights_mha.py TestTritonConsistency
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Dict

import requests
import torch

sys.path.insert(0, str(Path(__file__).parent))
from step03_test_utils import _model_exists  # type: ignore[attr-defined]

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_CUDA = torch.cuda.is_available()

# Allow overriding the model via env var (e.g. SMALL_MODEL=meta-llama/Llama-3.1-8B-Instruct
# when Llama-3.2-1B is not accessible but the 8B model is in the HF cache).
_SMALL_MODEL_OVERRIDE = os.environ.get("SMALL_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.7",
    "--max-running-requests",
    "8",
    "--tp-size",
    "1",
]

# Simple questions whose answers are unambiguous even for a 1B model.
_CHAT_QA_PAIRS = [
    ("What is the capital of France? Answer in one word.", "Paris"),
    ("What is 2 plus 3? Answer with just the number.", "5"),
    ("What color is the sky on a clear day? Answer in one word.", "blue"),
]

# Raw-completion prompts for deterministic output comparison.
# These work well with base models; instruct models also handle them.
_COMPLETION_PROMPTS = [
    "The capital of France is",
    "The sum of 2 and 3 is",
    "Water freezes at 0 degrees",
]


def _chat_generate(
    base_url: str, model: str, question: str, max_tokens: int = 16
) -> str:
    """Send a chat message and return the assistant reply (stripped)."""
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _raw_generate(base_url: str, prompt: str, max_new_tokens: int = 16) -> str:
    """Raw text completion via /generate (greedy, deterministic)."""
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, list):
        body = body[0]
    return body["text"]


# ---------------------------------------------------------------------------
# Per-(backend, runner) accuracy tests
# ---------------------------------------------------------------------------


class _MHAAccuracyBase(CustomTestCase):
    """Launch a real-weights server, verify factual accuracy and output quality."""

    model: str = _SMALL_MODEL_OVERRIDE
    runner_args: list = []

    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("CUDA required")
        if not _model_exists(cls.model):
            raise unittest.SkipTest(f"model not in cache: {cls.model}")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_BASE_SERVER_ARGS + cls.runner_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ---- accuracy ----

    def test_factual_accuracy_chat(self):
        """Each question should produce the expected answer."""
        for question, expected in _CHAT_QA_PAIRS:
            with self.subTest(question=question):
                answer = _chat_generate(self.base_url, self.model, question)
                self.assertIn(
                    expected.lower(),
                    answer.lower(),
                    f"Expected '{expected}' in answer '{answer}'",
                )

    def test_outputs_non_empty(self):
        """All raw completions must be non-empty strings."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                text = _raw_generate(self.base_url, prompt)
                self.assertIsInstance(text, str)
                self.assertGreater(
                    len(text.strip()), 0, f"Empty output for prompt: {prompt!r}"
                )

    def test_no_nan_in_output(self):
        """No NaN markers in any output."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                text = _raw_generate(self.base_url, prompt)
                self.assertNotIn(
                    "nan", text.lower(), f"NaN marker in output for: {prompt!r}"
                )

    # ---- prefill path ----

    def test_long_prefill_then_decode(self):
        """Send a ~200-token context then decode; verify coherent output."""
        context = (
            "Paris is the capital of France. The Eiffel Tower is located in Paris. "
            "Paris is known for its art, culture, and cuisine. "
        ) * 5
        question = context + " In one word: what is the capital of France?"
        answer = _chat_generate(self.base_url, self.model, question, max_tokens=8)
        self.assertIn("paris", answer.lower())

    # ---- decode batch ----

    def test_sequential_decode_requests(self):
        """Multiple sequential decode requests all succeed."""
        for prompt in _COMPLETION_PROMPTS:
            text = _raw_generate(self.base_url, prompt)
            self.assertGreater(len(text.strip()), 0)


# ----- Triton -----


class TestTritonEagerAccuracy(_MHAAccuracyBase):
    """Triton + eager (no CUDA graph)."""

    runner_args = ["--attention-backend", "triton", "--disable-cuda-graph"]


class TestTritonCudaGraphAccuracy(_MHAAccuracyBase):
    """Triton + full CUDA graph decode."""

    runner_args = [
        "--attention-backend",
        "triton",
        "--disable-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


class TestTritonPCGAccuracy(_MHAAccuracyBase):
    """Triton + piecewise CUDA graph (PCG)."""

    runner_args = [
        "--attention-backend",
        "triton",
        "--enforce-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


# ----- FlashInfer -----


class TestFlashInferEagerAccuracy(_MHAAccuracyBase):
    """FlashInfer + eager."""

    runner_args = ["--attention-backend", "flashinfer", "--disable-cuda-graph"]


class TestFlashInferCudaGraphAccuracy(_MHAAccuracyBase):
    """FlashInfer + full CUDA graph."""

    runner_args = [
        "--attention-backend",
        "flashinfer",
        "--disable-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


class TestFlashInferPCGAccuracy(_MHAAccuracyBase):
    """FlashInfer + PCG."""

    runner_args = [
        "--attention-backend",
        "flashinfer",
        "--enforce-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


# ----- TRTLLM MHA -----


class TestTRTLLMMHAEagerAccuracy(_MHAAccuracyBase):
    """TRTLLM MHA + eager."""

    runner_args = ["--attention-backend", "trtllm_mha", "--disable-cuda-graph"]


class TestTRTLLMMHACudaGraphAccuracy(_MHAAccuracyBase):
    """TRTLLM MHA + full CUDA graph."""

    runner_args = [
        "--attention-backend",
        "trtllm_mha",
        "--disable-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


class TestTRTLLMMHAPCGAccuracy(_MHAAccuracyBase):
    """TRTLLM MHA + PCG."""

    runner_args = [
        "--attention-backend",
        "trtllm_mha",
        "--enforce-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


# ---------------------------------------------------------------------------
# Cross-runner consistency — the core step-03 invariant
# ---------------------------------------------------------------------------


class TestTritonConsistency(CustomTestCase):
    """
    Verify that triton eager / full-CG / PCG produce *identical* greedy outputs.

    This is the key invariant for step-03: unifying init_forward_metadata
    across capture / replay / eager into a single init_forward_data call
    must not change model outputs.  Any divergence here is a step-03 bug.

    Implementation: launches three servers sequentially (same model, different
    runner flags), runs the same greedy-decode prompts through each, and asserts
    text equality.  Requires ~10–15 min on Llama-3.2-1B-Instruct.
    """

    model = _SMALL_MODEL_OVERRIDE

    _RUNNERS = {
        "eager": ["--attention-backend", "triton", "--disable-cuda-graph"],
        "full_cg": [
            "--attention-backend",
            "triton",
            "--disable-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
        "pcg": [
            "--attention-backend",
            "triton",
            "--enforce-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
    }

    @classmethod
    def _collect(cls, base_url: str) -> Dict[str, str]:
        return {
            p: _raw_generate(base_url, p, max_new_tokens=16)
            for p in _COMPLETION_PROMPTS
        }

    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("CUDA required")
        if not _model_exists(cls.model):
            raise unittest.SkipTest(f"model not in cache: {cls.model}")

        cls.outputs: Dict[str, Dict[str, str]] = {}
        base_port = 30200

        for i, (name, extra_args) in enumerate(cls._RUNNERS.items()):
            base_url = f"http://127.0.0.1:{base_port + i}"
            process = popen_launch_server(
                cls.model,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_BASE_SERVER_ARGS + extra_args,
            )
            try:
                cls.outputs[name] = cls._collect(base_url)
            finally:
                kill_process_tree(process.pid)

    def test_eager_vs_full_cg_identical(self):
        """Full CUDA-graph outputs must be token-for-token identical to eager."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                eager_out = self.outputs["eager"][prompt]
                cg_out = self.outputs["full_cg"][prompt]
                self.assertEqual(
                    eager_out,
                    cg_out,
                    f"eager≠full_cg for prompt {prompt!r}\n"
                    f"  eager : {eager_out!r}\n"
                    f"  full_cg: {cg_out!r}",
                )

    def test_eager_vs_pcg_identical(self):
        """PCG outputs must be token-for-token identical to eager."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                eager_out = self.outputs["eager"][prompt]
                pcg_out = self.outputs["pcg"][prompt]
                self.assertEqual(
                    eager_out,
                    pcg_out,
                    f"eager≠PCG for prompt {prompt!r}\n"
                    f"  eager: {eager_out!r}\n"
                    f"  pcg  : {pcg_out!r}",
                )

    def test_full_cg_vs_pcg_identical(self):
        """Full-CG and PCG outputs must match (both should equal eager)."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.outputs["full_cg"][prompt],
                    self.outputs["pcg"][prompt],
                )


class TestFlashInferConsistency(TestTritonConsistency):
    """Same cross-runner consistency check for flashinfer backend."""

    _RUNNERS = {
        "eager": ["--attention-backend", "flashinfer", "--disable-cuda-graph"],
        "full_cg": [
            "--attention-backend",
            "flashinfer",
            "--disable-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
        "pcg": [
            "--attention-backend",
            "flashinfer",
            "--enforce-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
    }


class TestTRTLLMMHAConsistency(TestTritonConsistency):
    """Same cross-runner consistency check for TRTLLM MHA backend."""

    _RUNNERS = {
        "eager": ["--attention-backend", "trtllm_mha", "--disable-cuda-graph"],
        "full_cg": [
            "--attention-backend",
            "trtllm_mha",
            "--disable-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
        "pcg": [
            "--attention-backend",
            "trtllm_mha",
            "--enforce-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            "8",
        ],
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
