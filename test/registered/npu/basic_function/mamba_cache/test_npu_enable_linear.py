"""
Test module for linear attention kernel verification and ReplaySSM feature validation on Ascend NPU.

This test case launches an SGLang serving instance with specified parameters,
and verifies that --linear-attn-verify-backend and --enable-linear-replayssm-spec
take effect by serving real inference requests and checking characteristic log
outputs during server startup.
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_6_35B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)


class TestLinearAttentionAndReplaySSM(CustomTestCase):
    """Testcase: Verify --linear-attn-verify-backend and --enable-linear-replayssm-spec flags.

    [Test Category] Parameter
    [Test Target] --linear-attn-verify-backend; --enable-linear-replayssm-spec
    """

    model = QWEN3_6_35B_A3B_WEIGHTS_PATH
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "2",
                "--max-running-requests",
                "12",
                "--mem-fraction-static",
                "0.85",
                "--cuda-graph-bs-decode",
                "2",
                "4",
                "8",
                "--dtype",
                "bfloat16",
                "--linear-attn-verify-backend",
                "flashinfer",
                "--enable-linear-replayssm-spec",
            ],
            env={
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "HCCL_BUFFSIZE": "1600",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "HCCL_SOCKET_IFNAME": "lo",
                "GLOO_SOCKET_IFNAME": "lo",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
                "SGLANG_DEEPEP_EF16_DISPATCH": "1",
                "ENABLE_ASCEND_MOE_NZ": "1",
            },
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)
        if getattr(cls, "out_log_file", None):
            cls.out_log_file.close()
        if getattr(cls, "err_log_file", None):
            cls.err_log_file.close()
        for path in (cls.OUT_LOG_PATH, cls.ERR_LOG_PATH):
            if os.path.exists(path):
                os.remove(path)

    def _read_full_logs(self) -> str:
        """Read the full server logs written so far."""
        with open(self.OUT_LOG_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_and_check(self):
        """Send a real inference request to prove the feature path works."""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

    def test_linear_attn_verify_backend_flashinfer(self):
        """Verify --linear-attn-verify-backend flashinfer takes effect."""
        self._generate_and_check()
        logs = self._read_full_logs()
        expected = "Linear attention kernel backend: decode=triton, prefill=triton, verify=flashinfer"
        self.assertIn(
            expected,
            logs,
            f"Expected log not found: '{expected}'\n"
            f"Last 2000 chars of logs:\n{logs[-2000:]}",
        )

    def test_enable_linear_replayssm_spec(self):
        """Verify --enable-linear-replayssm-spec takes effect."""
        self._generate_and_check()
        logs = self._read_full_logs()
        expected = "GDN ReplaySSM ring buffers allocated"
        self.assertIn(
            expected,
            logs,
            f"Expected log not found: '{expected}'\n"
            f"Last 2000 chars of logs:\n{logs[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
