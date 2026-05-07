"""Integration tests for TLI (Token-Level Intersection) speculative decoding.

TLI is a lossless speculative decoding algorithm for heterogeneous vocabularies
described in "Accelerating LLM Inference with Lossless Speculative Decoding
Algorithms for Heterogeneous Vocabularies" (Timor et al., ICML 2025,
https://arxiv.org/abs/2502.05202).

Two test groups:
- Same-tokenizer (Llama-3.1-8B + Llama-3.2-1B): high vocab overlap, verifies
  correctness and acceptance-length parity with STANDALONE.
- Cross-family (Llama-3.1-8B + Qwen2.5-0.5B): different tokenizer families,
  verifies TLI's unique lossless cross-family capability.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI,
    DEFAULT_DRAFT_MODEL_TLI,
    DEFAULT_TARGET_MODEL_TLI,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, suite="stage-b-test-1-gpu-large")

GSM_DATASET_PATH = None

# ---------------------------------------------------------------------------
# Shared server-arg templates
# ---------------------------------------------------------------------------

_COMMON_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "TLI",
    "--speculative-num-steps",
    "4",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--mem-fraction-static",
    "0.7",
]

_SAME_TOK_ARGS = _COMMON_ARGS + [
    "--speculative-draft-model-path",
    DEFAULT_DRAFT_MODEL_TLI,
]

_CROSS_FAMILY_ARGS = _COMMON_ARGS + [
    "--speculative-draft-model-path",
    DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI,
]


# ---------------------------------------------------------------------------
# Base class — same-tokenizer pair (Llama-3.1-8B + Llama-3.2-1B)
# ---------------------------------------------------------------------------


class TestTLISameTokenizerBase(CustomTestCase):
    """TLI with a same-tokenizer draft model (≈100% vocab overlap)."""

    model = DEFAULT_TARGET_MODEL_TLI
    draft_model = DEFAULT_DRAFT_MODEL_TLI
    base_url = DEFAULT_URL_FOR_TEST
    # Thresholds: accuracy matches baseline; acceptance length is slightly
    # lower than STANDALONE (3.6) to account for the intersection constraint.
    accuracy_threshold = 0.69
    spec_decode_threshold = 3.0

    @classmethod
    def get_server_args(cls):
        return _SAME_TOK_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


# ---------------------------------------------------------------------------
# Base class — cross-family pair (Llama-3.1-8B + Qwen2.5-0.5B)
# ---------------------------------------------------------------------------


class TestTLICrossFamilyBase(CustomTestCase):
    """TLI with a cross-family draft model (~85% vocab overlap).

    Exercises TLI's unique capability: lossless speculative decoding when the
    draft and target models have different tokenizer families.
    """

    model = DEFAULT_TARGET_MODEL_TLI
    draft_model = DEFAULT_CROSS_FAMILY_DRAFT_MODEL_TLI
    base_url = DEFAULT_URL_FOR_TEST
    # Cross-family: same accuracy target (lossless), but lower acceptance length
    # due to weaker draft-model alignment with the target distribution.
    accuracy_threshold = 0.69
    spec_decode_threshold = 2.5

    @classmethod
    def get_server_args(cls):
        return _CROSS_FAMILY_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


# ---------------------------------------------------------------------------
# Concrete classes — same-tokenizer, different attention backends
# ---------------------------------------------------------------------------


class TestTLISameTokenizerTriton(TestTLISameTokenizerBase):
    @classmethod
    def get_server_args(cls):
        return _SAME_TOK_ARGS + ["--attention-backend", "triton"]


class TestTLISameTokenizerFlashinfer(TestTLISameTokenizerBase):
    @classmethod
    def get_server_args(cls):
        return _SAME_TOK_ARGS + ["--attention-backend", "flashinfer"]


# ---------------------------------------------------------------------------
# Concrete classes — cross-family, different attention backends
# ---------------------------------------------------------------------------


class TestTLICrossFamilyTriton(TestTLICrossFamilyBase):
    @classmethod
    def get_server_args(cls):
        return _CROSS_FAMILY_ARGS + ["--attention-backend", "triton"]


class TestTLICrossFamilyFlashinfer(TestTLICrossFamilyBase):
    @classmethod
    def get_server_args(cls):
        return _CROSS_FAMILY_ARGS + ["--attention-backend", "flashinfer"]


if __name__ == "__main__":
    unittest.main(verbosity=3)
