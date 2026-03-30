"""E2E accuracy tests for DCP (Decode Context Parallelism).

Launches DeepSeek-V2 with DCP enabled (AG+RS and A2A backends),
runs few-shot GSM8K evaluation, and verifies accuracy meets threshold.

Requires 8 GPUs.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=600, suite="stage-c-test-8-gpu-h200")

DCP_MODEL = "deepseek-ai/DeepSeek-V2"
GSM8K_ACCURACY_THRESHOLD = 0.78
GSM8K_NUM_QUESTIONS = 200

COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--dcp-size",
    "8",
    "--disable-radix-cache",
    "--enable-symm-mem",
    "--chunked-prefill-size",
    "32768",
    "--max-running-requests",
    "512",
]

COMMON_ENV = {
    "SGLANG_DCP_SYMM_ONLY": "true",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}


def _run_gsm8k(test_case, label):
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=GSM8K_NUM_QUESTIONS,
        max_new_tokens=512,
        parallel=64,
        host="http://127.0.0.1",
        port=int(test_case.base_url.split(":")[-1]),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    print(f"{metrics=}")

    if is_in_ci():
        write_github_step_summary(
            f"### test_gsm8k ({label})\n" f'{metrics["accuracy"]=:.3f}\n'
        )
    test_case.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


class TestDCPAccuracyFlashInferAGRS(CustomTestCase):
    """DCP with FlashInfer backend and AG+RS communication."""

    @classmethod
    def setUpClass(cls):
        cls.model = DCP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = COMMON_SERVER_ARGS + [
            "--dcp-comm-backend",
            "ag_rs",
            "--attention-backend",
            "flashinfer",
            "--mem-fraction-static",
            "0.85",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=COMMON_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self, "dcp-flashinfer-ag_rs")


class TestDCPAccuracyFlashInferA2A(CustomTestCase):
    """DCP with FlashInfer backend and A2A communication."""

    @classmethod
    def setUpClass(cls):
        cls.model = DCP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = COMMON_SERVER_ARGS + [
            "--dcp-comm-backend",
            "a2a",
            "--attention-backend",
            "flashinfer",
            "--mem-fraction-static",
            "0.85",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=COMMON_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self, "dcp-flashinfer-a2a")


class TestDCPAccuracyFA3AGRS(CustomTestCase):
    """DCP with FA3 backend and AG+RS communication."""

    @classmethod
    def setUpClass(cls):
        cls.model = DCP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = COMMON_SERVER_ARGS + [
            "--dcp-comm-backend",
            "ag_rs",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.83",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=COMMON_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self, "dcp-fa3-ag_rs")


class TestDCPAccuracyFA3A2A(CustomTestCase):
    """DCP with FA3 backend and A2A communication."""

    @classmethod
    def setUpClass(cls):
        cls.model = DCP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = COMMON_SERVER_ARGS + [
            "--dcp-comm-backend",
            "a2a",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.83",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=COMMON_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _run_gsm8k(self, "dcp-fa3-a2a")


if __name__ == "__main__":
    unittest.main()
