"""
Step-03 E2E tests with real MLA model weights.

Uses lmsys/sglang-ci-dsv3-test (DEFAULT_MODEL_NAME_FOR_TEST_MLA), a tiny
DeepSeek-V3-style MLA model built for CI.  Tests:

1. Accuracy / quality — each MLA backend produces non-empty, sensible outputs.
2. Cross-backend consistency — flashinfer_mla and trtllm_mla must produce
   identical greedy outputs for the same prompts.  Different MLA backends
   should be numerically equivalent (same attention algorithm, different kernel).
3. Cross-runner consistency — each backend's eager vs. full-CG runner must
   give identical greedy outputs (the step-03 invariant for MLA).

Run on the cluster:
  python test_step03_real_weights_mla.py
  python test_step03_real_weights_mla.py TestFlashInferMLAConsistency
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict

import requests
import torch

sys.path.insert(0, str(Path(__file__).parent))
from step03_test_utils import _model_exists  # type: ignore[attr-defined]

from sglang.srt.layers import dp_attention as _dp_attn

# TP=1 for all MLA unit-level checks
_dp_attn.get_attention_tp_size = lambda: 1

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Helpers
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

# Questions suitable for a tiny (CI) DSV3 model
_CHAT_QA_PAIRS = [
    ("What is 1 + 1? Answer with just the number.", "2"),
    ("What color is the sky? One word only.", "blue"),
    ("Is water wet? Answer yes or no.", "yes"),
]

_COMPLETION_PROMPTS = [
    "The sum of 1 and 1 is",
    "The sky is",
    "Water is",
]


def _chat_generate(
    base_url: str, model: str, question: str, max_tokens: int = 8
) -> str:
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
# Per-backend accuracy tests
# ---------------------------------------------------------------------------


class _MLAAccuracyBase(CustomTestCase):
    model: str = DEFAULT_MODEL_NAME_FOR_TEST_MLA
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

    def test_outputs_non_empty(self):
        """All completions must produce non-empty text."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                text = _raw_generate(self.base_url, prompt)
                self.assertGreater(
                    len(text.strip()), 0, f"Empty output for: {prompt!r}"
                )

    def test_no_nan_in_output(self):
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                text = _raw_generate(self.base_url, prompt)
                self.assertNotIn("nan", text.lower())

    def test_simple_accuracy(self):
        """Model should get basic facts right even as a tiny CI model."""
        for question, expected in _CHAT_QA_PAIRS:
            with self.subTest(question=question):
                answer = _chat_generate(self.base_url, self.model, question)
                self.assertIn(
                    expected.lower(),
                    answer.lower(),
                    f"Expected '{expected}' in '{answer}'",
                )

    def test_prefill_then_decode(self):
        """Non-trivial prefill followed by decode must succeed."""
        context = "The answer to 1 + 1 is 2. " * 20
        text = _raw_generate(self.base_url, context, max_new_tokens=8)
        self.assertGreater(len(text.strip()), 0)


# ----- FlashInfer MLA -----


class TestFlashInferMLAEagerAccuracy(_MLAAccuracyBase):
    """FlashInfer MLA + eager."""

    runner_args = ["--attention-backend", "flashinfer", "--disable-cuda-graph"]


class TestFlashInferMLACudaGraphAccuracy(_MLAAccuracyBase):
    """FlashInfer MLA + full CUDA graph."""

    runner_args = [
        "--attention-backend",
        "flashinfer",
        "--disable-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


class TestFlashInferMLAPCGAccuracy(_MLAAccuracyBase):
    """FlashInfer MLA + PCG — exercises extend-mode capture for MLA."""

    runner_args = [
        "--attention-backend",
        "flashinfer",
        "--enforce-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


# ----- TRTLLM MLA -----


class TestTRTLLMMLAEagerAccuracy(_MLAAccuracyBase):
    """TRTLLM MLA + eager."""

    runner_args = ["--attention-backend", "trtllm", "--disable-cuda-graph"]


class TestTRTLLMMLACudaGraphAccuracy(_MLAAccuracyBase):
    """TRTLLM MLA + full CUDA graph."""

    runner_args = [
        "--attention-backend",
        "trtllm",
        "--disable-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


class TestTRTLLMMLAPCGAccuracy(_MLAAccuracyBase):
    """TRTLLM MLA + PCG."""

    runner_args = [
        "--attention-backend",
        "trtllm",
        "--enforce-piecewise-cuda-graph",
        "--cuda-graph-max-bs",
        "8",
    ]


# ---------------------------------------------------------------------------
# Cross-backend consistency: flashinfer_mla vs trtllm_mla
# ---------------------------------------------------------------------------


class TestMLABackendConsistency(CustomTestCase):
    """
    Verify that flashinfer_mla and trtllm_mla produce identical greedy outputs.
    Both backends implement the same MLA attention algorithm; any divergence
    indicates a kernel correctness issue.

    Launches two servers (flashinfer_mla eager, trtllm_mla eager) sequentially
    and compares raw-completion outputs.
    """

    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA

    _BACKENDS = {
        "flashinfer_mla": ["--attention-backend", "flashinfer", "--disable-cuda-graph"],
        "trtllm_mla": ["--attention-backend", "trtllm", "--disable-cuda-graph"],
    }

    @classmethod
    def _collect(cls, base_url: str) -> Dict[str, str]:
        return {p: _raw_generate(base_url, p) for p in _COMPLETION_PROMPTS}

    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("CUDA required")
        if not _model_exists(cls.model):
            raise unittest.SkipTest(f"model not in cache: {cls.model}")

        cls.outputs: Dict[str, Dict[str, str]] = {}
        base_port = 30300

        for i, (name, extra_args) in enumerate(cls._BACKENDS.items()):
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

    def test_flashinfer_vs_trtllm_identical(self):
        """flashinfer_mla and trtllm_mla must produce identical greedy outputs."""
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                fi = self.outputs["flashinfer_mla"][prompt]
                tr = self.outputs["trtllm_mla"][prompt]
                self.assertEqual(
                    fi,
                    tr,
                    f"flashinfer_mla≠trtllm_mla for {prompt!r}\n"
                    f"  flashinfer: {fi!r}\n"
                    f"  trtllm    : {tr!r}",
                )


# ---------------------------------------------------------------------------
# Cross-runner consistency per MLA backend — the step-03 invariant
# ---------------------------------------------------------------------------


class _MLAConsistencyBase(CustomTestCase):
    """
    Base: verify that eager / full-CG / PCG produce identical greedy outputs
    for a given MLA backend.
    """

    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
    backend_flag: str = "flashinfer"

    @property
    def _runners(self) -> Dict[str, list]:
        b = self.backend_flag
        return {
            "eager": [f"--attention-backend", b, "--disable-cuda-graph"],
            "full_cg": [
                f"--attention-backend",
                b,
                "--disable-piecewise-cuda-graph",
                "--cuda-graph-max-bs",
                "8",
            ],
            "pcg": [
                f"--attention-backend",
                b,
                "--enforce-piecewise-cuda-graph",
                "--cuda-graph-max-bs",
                "8",
            ],
        }

    @classmethod
    def _collect(cls, base_url: str) -> Dict[str, str]:
        return {p: _raw_generate(base_url, p) for p in _COMPLETION_PROMPTS}

    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("CUDA required")
        if not _model_exists(cls.model):
            raise unittest.SkipTest(f"model not in cache: {cls.model}")

        # Build the runner map using the class attribute
        b = cls.backend_flag
        runners = {
            "eager": ["--attention-backend", b, "--disable-cuda-graph"],
            "full_cg": [
                "--attention-backend",
                b,
                "--disable-piecewise-cuda-graph",
                "--cuda-graph-max-bs",
                "8",
            ],
            "pcg": [
                "--attention-backend",
                b,
                "--enforce-piecewise-cuda-graph",
                "--cuda-graph-max-bs",
                "8",
            ],
        }

        cls.outputs: Dict[str, Dict[str, str]] = {}
        base_port = 30400

        for i, (name, extra_args) in enumerate(runners.items()):
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
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.outputs["eager"][prompt],
                    self.outputs["full_cg"][prompt],
                    f"[{self.backend_flag}] eager≠full_cg for {prompt!r}",
                )

    def test_eager_vs_pcg_identical(self):
        for prompt in _COMPLETION_PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    self.outputs["eager"][prompt],
                    self.outputs["pcg"][prompt],
                    f"[{self.backend_flag}] eager≠PCG for {prompt!r}",
                )


class TestFlashInferMLAConsistency(_MLAConsistencyBase):
    """FlashInfer MLA: eager / full-CG / PCG must produce identical outputs."""

    backend_flag = "flashinfer"


class TestTRTLLMMLAConsistency(_MLAConsistencyBase):
    """TRTLLM MLA: eager / full-CG / PCG must produce identical outputs."""

    backend_flag = "trtllm"


if __name__ == "__main__":
    unittest.main(verbosity=2)
