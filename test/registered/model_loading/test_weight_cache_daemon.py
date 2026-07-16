import os
import subprocess
import sys
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
register_cuda_ci(est_time=120, stage="extra-a", runner_config="2-gpu-large")

# Capture the client server's logs so test_loaded_via_ipc can assert the IPC
# load path actually ran (and did not silently fall back to disk).
STDOUT_FILENAME = "/tmp/test_weight_cache_daemon_stdout.log"
STDERR_FILENAME = "/tmp/test_weight_cache_daemon_stderr.log"

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
]


class TestWeightCacheDaemonTP2(CustomTestCase):
    """E2E test: start weight cache daemons, then launch server in client mode with TP2."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tp_size = 2

        # Clean up stale ready/socket files from previous runs
        for rank in range(cls.tp_size):
            for suffix in (".ready", ".sock"):
                path = f"/tmp/sglang_weight_cache_rank{rank}{suffix}"
                if os.path.exists(path):
                    os.unlink(path)

        # Step 1: Launch weight cache daemons (blocks until all ranks are ready,
        # then monitors child processes)
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(cls.tp_size),
            ]
        )

        # Step 2: Wait for all daemon ready files
        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start = time.time()
        for rank in range(cls.tp_size):
            ready_path = f"/tmp/sglang_weight_cache_rank{rank}.ready"
            while not os.path.exists(ready_path):
                if time.time() - start > timeout:
                    kill_process_tree(cls.daemon_process.pid)
                    raise TimeoutError(
                        f"Weight cache daemon rank {rank} not ready within {timeout}s"
                    )
                if cls.daemon_process.poll() is not None:
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely "
                        f"with code {cls.daemon_process.returncode}"
                    )
                time.sleep(2)

        # Step 3: Launch server in client mode — loads weights via IPC from daemons
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "daemon_process") and cls.daemon_process:
            kill_process_tree(cls.daemon_process.pid)
        for stream in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        for rank in range(getattr(cls, "tp_size", 2)):
            for suffix in (".ready", ".sock"):
                path = f"/tmp/sglang_weight_cache_rank{rank}{suffix}"
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def test_generate(self):
        for prompt in PROMPTS:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            text = data["choices"][0]["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Empty output for prompt: {prompt}")

    def test_chat(self):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 2+3?"}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_loaded_via_ipc(self):
        """Assert the server actually loaded weights over IPC.

        Without this, the test would still pass if the IPC path silently
        regressed to disk loading (the daemon would just sit unused), because
        generation output looks identical either way. The daemon-side loader
        logs "[IpcModelLoader] Loaded model via IPC" on every rank, so its
        presence in the captured server logs is our proof the IPC path ran.
        """
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
        logs = ""
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as f:
                    logs += f.read()
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )


if __name__ == "__main__":
    unittest.main()
