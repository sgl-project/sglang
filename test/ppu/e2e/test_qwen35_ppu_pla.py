"""E2E: Qwen3.5-397B-A17B-W8A8-INT8 on PPU with pla kernel.

This test verifies that the pla (PPU FLA) adaptation works correctly by:
1. Launching a Qwen3.5-397B-A17B-W8A8-INT8 server on PPU (SGLANG_SAIL_FLA_CUDA auto-enabled).
2. Sending a generation request to exercise the GDN linear attention path.
3. Verifying the server responds correctly without crashes.

Qwen3.5-397B-A17B-W8A8-INT8 uses GDN (Gated Delta Network) linear attention, which triggers
the pla.decode kernel path:
  - fused_sigmoid_gating_delta_rule_forward_k_last (decode)
  - fused_sigmoid_gating_delta_rule_forward_k_last_packed (packed decode)

Without correct pla adaptation, the server would crash with:
  ModuleNotFoundError: No module named 'fla'
or
  AttributeError: module 'pla' has no attribute '...'
"""

import io
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/Qwen3.5-397B-A17B-W8A8-INT8",
)
SERVER_LAUNCH_TIMEOUT = 3600


class TestQwen35PpuPla(CustomTestCase):
    """E2E: Qwen3.5-397B-A17B W8A8-INT8 on PPU verifying pla kernel adaptation."""

    @classmethod
    def setUpClass(cls):
        cls.stdout_cap = io.StringIO()
        cls.stderr_cap = io.StringIO()
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            return_stdout_stderr=(cls.stdout_cap, cls.stderr_cap),
            other_args=[
                "--tp-size",
                "16",
                "--quantization",
                "w8a8_int8",
                "--trust-remote-code",
                "--disable-custom-all-reduce",
                "--mem-fraction-static",
                "0.85",
                "--watchdog-timeout",
                "24000",
                "--dist-timeout",
                "24000",
                "--cuda-graph-max-bs",
                "32",
                "--max-running-requests",
                "32",
                "--disable-radix-cache",
                "--mamba-scheduler-strategy",
                "extra_buffer_lazy",
                "--disable-shared-experts-fusion",
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generate_simple_request(self):
        """Send a generation request to verify pla kernel works end-to-end."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        ret = response.json()
        self.assertIn("text", ret)
        output_text = ret["text"]
        print(f"Server response: {output_text}")
        self.assertTrue(
            len(output_text) > 0,
            f"Expected non-empty output, but got: {output_text}",
        )

    def test_pla_kernel_used(self):
        """Verify that the pla kernel was actually invoked by the server.

        Checks captured server output for the pla log message printed by
        fused_sigmoid_gating_recurrent.py and fused_recurrent.py.
        """
        stderr_output = self.stderr_cap.getvalue()
        stdout_output = self.stdout_cap.getvalue()
        combined = stderr_output + stdout_output

        self.assertIn(
            "USE PPU SAIL CUDA FLA kernel",
            combined,
            "pla kernel was NOT used by the server! "
            "Check that SGLANG_SAIL_FLA_CUDA is enabled and pla is installed.",
        )


if __name__ == "__main__":
    unittest.main()
