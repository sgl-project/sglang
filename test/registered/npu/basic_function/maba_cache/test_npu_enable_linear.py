"""
Test module for linear attention kernel verification and ReplaySSM feature validation on Ascend NPU.

This test case launches an SGLang serving instance with specified parameters,
and verifies that --linear-attn-verify-backend and --enable-linear-replayssm-spec
take effect by checking characteristic log outputs during server startup.
"""

import os
import subprocess
import tempfile
import unittest
from urllib.parse import urlparse

from sglang.test.ascend.test_ascend_utils import QWEN3_6_35B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _create_clean_subprocess_env,
    _wait_for_server_health,
    kill_process_tree,
)

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)


def _load_ascend_env(base_env: dict) -> dict:
    """Source Ascend toolkit environment scripts and merge them into the base environment."""
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; "
        "source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null; "
        'python3 -c "import os, json; print(json.dumps(dict(os.environ)))"'
    )
    try:
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=base_env,
        )
        if result.returncode == 0:
            last_line = result.stdout.strip().split("\n")[-1]
            ascend_env = __import__("json").loads(last_line)
            base_env.update(ascend_env)
    except Exception:
        pass
    return base_env


class TestLinearAttentionAndReplaySSM(CustomTestCase):
    """Testcase: Verify --linear-attn-verify-backend and --enable-linear-replayssm-spec flags.

    [Test Category] Parameter
    [Test Target] --linear-attn-verify-backend, --enable-linear-replayssm-spec
    """

    @classmethod
    def setUpClass(cls):
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.host = parsed_url.hostname
        cls.port = parsed_url.port
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Initialize clean subprocess environment (aligned with reference framework spec)
        env = _create_clean_subprocess_env(os.environ.copy())

        # Load Ascend toolkit environment
        env = _load_ascend_env(env)

        # Apply business-specific environment variables
        env.update(
            {
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
            }
        )

        cls.env = env

        # Create temporary log file (aligned with reference log capture approach)
        cls.log_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".sglang.log", delete=False
        )
        cls.log_file_path = cls.log_file.name

        # Construct server startup arguments
        server_args = [
            "--model-path",
            QWEN3_6_35B_A3B_WEIGHTS_PATH,
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--tp-size",
            "2",
            "--trust-remote-code",
            "--host",
            cls.host,
            "--max-running-requests",
            "12",
            "--mem-fraction-static",
            "0.85",
            "--port",
            str(cls.port),
            "--cuda-graph-bs",
            "2",
            "4",
            "8",
            "--dtype",
            "bfloat16",
            "--linear-attn-verify-backend",
            "flashinfer",
            "--enable-linear-replayssm-spec",
        ]

        # Start server process
        command = ["python3", "-m", "sglang.launch_server"] + server_args
        cls.process = subprocess.Popen(
            command,
            stdout=cls.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=cls.env,
        )

        # Wait for server health check to pass (aligned with reference startup wait logic)
        try:
            _wait_for_server_health(
                cls.process,
                cls.base_url,
                None,
                DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
        except Exception:
            # Output tail of logs on startup failure for troubleshooting
            cls.log_file.flush()
            with open(cls.log_file_path, "r") as f:
                logs = f.read()
            cls.tearDownClass()
            raise RuntimeError(
                f"Server failed to start within {DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH}s.\n"
                f"Last 5000 chars of logs:\n{logs[-5000:]}"
            )

    @classmethod
    def tearDownClass(cls):
        # Destroy server process tree (aligned with reference framework)
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        # Close and clean up temporary log file
        if hasattr(cls, "log_file") and cls.log_file:
            cls.log_file.close()
        if hasattr(cls, "log_file_path") and os.path.exists(cls.log_file_path):
            os.unlink(cls.log_file_path)

    def _read_full_logs(self) -> str:
        """Read the full server startup logs."""
        with open(self.log_file_path, "r") as f:
            return f.read()

    def test_linear_attn_verify_backend_flashinfer(self):
        """Verify --linear-attn-verify-backend flashinfer takes effect."""
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
