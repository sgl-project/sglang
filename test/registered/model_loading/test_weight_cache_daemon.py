import os
import subprocess
import sys
import time
import unittest

import requests
import torch

from sglang.srt.platforms import current_platform
from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# A ~1B model keeps the daemon->client IPC handoff cheap to exercise on every
# PR (fast download + load) while still covering the real block-load path; the
# test asserts the IPC path ran, not any particular model's quality.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# This file runs in two suites. The TP=2 class needs the 2-GPU runner (extra-a);
# the TP=1 smoke class is always-on (base-b / 1-gpu-small) so the daemon->client
# IPC handoff is exercised on every PR. Since the CI runner executes the whole
# file per suite, TestWeightCacheDaemonTP2 self-skips when fewer than 2 GPUs are
# visible (i.e. on the 1-gpu runner). Each class also has a vmm_fd variant, which
# brings up one more fixture.
register_cuda_ci(est_time=780, stage="extra-a", runner_config="2-gpu-large")
register_cuda_ci(est_time=142, stage="base-b", runner_config="1-gpu-small")

# Capture the client server's logs so test_loaded_via_ipc can assert the IPC
# load path actually ran (and did not silently fall back to disk).
STDOUT_FILENAME = "/tmp/test_weight_cache_daemon_stdout.log"
STDERR_FILENAME = "/tmp/test_weight_cache_daemon_stderr.log"
SMOKE_STDOUT_FILENAME = "/tmp/test_weight_cache_daemon_smoke_stdout.log"
SMOKE_STDERR_FILENAME = "/tmp/test_weight_cache_daemon_smoke_stderr.log"

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
]

# --weight-cache-transport vmm_fd, for the daemons and the servers of the vmm_fd
# variants at the end of this file.
VMM_FD_ARGS = ["--weight-cache-transport", "vmm_fd"]

# Greedy completions of PROMPTS from every fixture that came up in this process,
# keyed by its configuration. A vmm_fd variant compares its outputs with those
# of the torch_ipc fixture of the same configuration.
_GREEDY_OUTPUTS = {}


def _gpu_uuids(tp_size: int) -> list:
    # Single-node, default base_gpu_id/gpu_id_step: rank i runs on physical GPU i.
    return [current_platform.get_device_uuid(i) for i in range(tp_size)]


def _config_key(model, tp_size, daemon_args, server_args) -> tuple:
    return (model, tp_size, tuple(daemon_args), tuple(server_args))


def _greedy_completions(base_url: str, model: str) -> list:
    texts = []
    for prompt in PROMPTS:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 32, "temperature": 0},
        )
        resp.raise_for_status()
        texts.append(resp.json()["choices"][0]["text"])
    return texts


def _read_logs(paths) -> str:
    logs = ""
    for path in paths:
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                logs += f.read()
    return logs


@unittest.skipIf(
    torch.cuda.device_count() < 2,
    "TP=2 weight cache daemon test requires >=2 GPUs (skipped on the 1-gpu runner)",
)
class TestWeightCacheDaemonTP2(CustomTestCase):
    """E2E test: start weight cache daemons, then launch server in client mode with TP2."""

    model_override = None
    daemon_args = []
    server_args = []
    tp_size = 2
    log_files = (STDOUT_FILENAME, STDERR_FILENAME)

    @classmethod
    def setUpClass(cls):
        cls.model = cls.model_override or DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.gpu_uuids = _gpu_uuids(cls.tp_size)

        # Clean up stale ready/socket files from previous runs
        for device_uuid in cls.gpu_uuids:
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
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
                *cls.daemon_args,
            ]
        )

        # Step 2: Wait for all daemon ready files
        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start = time.time()
        for device_uuid in cls.gpu_uuids:
            ready_path = get_ready_path(device_uuid)
            while not os.path.exists(ready_path):
                if time.time() - start > timeout:
                    kill_process_tree(cls.daemon_process.pid)
                    raise TimeoutError(
                        f"Weight cache daemon for GPU {device_uuid} not ready "
                        f"within {timeout}s"
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
                *cls.server_args,
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        cls.greedy_outputs = _greedy_completions(cls.base_url, cls.model)
        _GREEDY_OUTPUTS[
            _config_key(cls.model, cls.tp_size, cls.daemon_args, cls.server_args)
        ] = cls.greedy_outputs

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
        for device_uuid in getattr(cls, "gpu_uuids", ()):
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
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
        generation output looks identical either way. The engine-side loader
        logs "[WeightCacheModelLoader] Loaded model via IPC" on every rank, so
        its presence in the captured server logs is our proof the IPC path ran.
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


class TestWeightCacheDaemonTP1Smoke(CustomTestCase):
    """Always-on TP=1 smoke: start a single weight cache daemon, launch a server
    in client mode, and confirm it loads weights via IPC and generates.

    This is the fast single-GPU sanity check (small model) that runs on every PR
    in the base-b / 1-gpu-small suite; the heavier TP=2 case above only runs on
    the 2-GPU runner.
    """

    daemon_args = []
    server_args = []
    tp_size = 1
    log_files = (SMOKE_STDOUT_FILENAME, SMOKE_STDERR_FILENAME)

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.gpu_uuids = _gpu_uuids(cls.tp_size)

        # Clean up stale ready/socket files from previous runs.
        for device_uuid in cls.gpu_uuids:
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
                if os.path.exists(path):
                    os.unlink(path)

        # Step 1: Launch the weight cache daemon (blocks until the rank is
        # ready, then monitors the child process).
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(cls.tp_size),
                *cls.daemon_args,
            ]
        )

        # Step 2: Wait for the daemon ready file.
        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start = time.time()
        for device_uuid in cls.gpu_uuids:
            ready_path = get_ready_path(device_uuid)
            while not os.path.exists(ready_path):
                if time.time() - start > timeout:
                    kill_process_tree(cls.daemon_process.pid)
                    raise TimeoutError(
                        f"Weight cache daemon for GPU {device_uuid} not ready "
                        f"within {timeout}s"
                    )
                if cls.daemon_process.poll() is not None:
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely "
                        f"with code {cls.daemon_process.returncode}"
                    )
                time.sleep(2)

        # Step 3: Launch server in client mode — loads weights via IPC.
        cls.stdout = open(SMOKE_STDOUT_FILENAME, "w")
        cls.stderr = open(SMOKE_STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
                *cls.server_args,
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        cls.greedy_outputs = _greedy_completions(cls.base_url, cls.model)
        _GREEDY_OUTPUTS[
            _config_key(cls.model, cls.tp_size, cls.daemon_args, cls.server_args)
        ] = cls.greedy_outputs

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
        for path in (SMOKE_STDOUT_FILENAME, SMOKE_STDERR_FILENAME):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        for device_uuid in getattr(cls, "gpu_uuids", ()):
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def test_generate(self):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        text = data["choices"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0, "Empty generation output")

    def test_loaded_via_ipc(self):
        """Assert the server actually loaded weights over IPC (see the TP=2
        variant for why this guard matters)."""
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
        logs = ""
        for path in (SMOKE_STDOUT_FILENAME, SMOKE_STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as f:
                    logs += f.read()
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )


class TestWeightCacheDaemonQwen3MoeDP(TestWeightCacheDaemonTP2):
    """Qwen3 with static attention DP through the existing IPC fixture."""

    daemon_args = [
        "--attn-dp-size",
        "2",
        "--ep-size",
        "1",
        "--enable-dp-lm-head",
        "--random-seed",
        "42",
    ]
    server_args = daemon_args[:]


class TestWeightCacheDaemonQwen3MoeEP(TestWeightCacheDaemonTP2):
    """Qwen3 MoE with static expert parallelism through the existing IPC fixture."""

    model_override = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
    daemon_args = [
        "--dp",
        "1",
        "--ep-size",
        "2",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "2",
        "--random-seed",
        "42",
    ]
    server_args = daemon_args[:]


class TestWeightCacheDaemonQwen3MoeAttnCP(TestWeightCacheDaemonTP2):
    """Qwen3 MoE with attention CP wider than MoE-DP.

    The MoE-DP communicator aliases the attention-CP group here, so it is wider
    than the configured `moe_dp_size`; the engine and the daemon must still
    state the same width in the config fingerprint.
    """

    model_override = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
    daemon_args = [
        "--attn-cp-size",
        "2",
        "--enable-prefill-cp",
        "--cp-strategy",
        "zigzag",
        "--cuda-graph-backend-prefill=disabled",
        "--random-seed",
        "42",
    ]
    server_args = daemon_args[:]


def _moe_runner_args() -> list:
    """On SM100, SGLang picks the flashinfer_trtllm MoE runner for BF16 Qwen3-MoE.
    Its weight post-processing reshapes the expert weights, which an engine that
    loads them over IPC (meta-initialized, no post-processing) cannot reproduce,
    with torch_ipc and vmm_fd alike. There, the MoE vmm_fd variants pin the
    triton runner on the daemons and the servers of both transports, so they
    compare the transports only."""
    return ["--moe-runner-backend", "triton"] if is_sm100_supported() else []


class _VmmFdVariant:
    """The parent fixture with ``--weight-cache-transport vmm_fd`` on the daemon
    and the server.

    The greedy outputs must equal those of the torch_ipc fixture with the same
    configuration. When that fixture has not come up in this process (the
    parent class did not run first, or ``extra_args()`` adds arguments), it is
    launched here first to record them. The last test kills the daemon: the
    engine's own vmm_fd mappings keep its weights valid, so it keeps serving the
    same outputs.
    """

    @classmethod
    def extra_args(cls) -> list:
        """Arguments added to the daemon and the server of both transports."""
        return []

    @classmethod
    def setUpClass(cls):
        model = getattr(cls, "model_override", None) or DEFAULT_MODEL
        daemon_args = [*cls.daemon_args, *cls.extra_args()]
        server_args = [*cls.server_args, *cls.extra_args()]
        key = _config_key(model, cls.tp_size, daemon_args, server_args)
        if key not in _GREEDY_OUTPUTS:
            cls.daemon_args, cls.server_args = daemon_args, server_args
            super().setUpClass()
            super().tearDownClass()
        cls.reference_outputs = _GREEDY_OUTPUTS[key]
        cls.daemon_args = daemon_args + VMM_FD_ARGS
        cls.server_args = server_args + VMM_FD_ARGS
        super().setUpClass()

    def test_loaded_via_vmm_fd(self):
        logs = _read_logs(self.log_files)
        self.assertEqual(
            logs.count("vmm_fd: mapped"),
            self.tp_size,
            "Expected every engine rank to map its daemon's vmm_fd allocations.",
        )

    def test_matches_torch_ipc(self):
        self.assertEqual(self.greedy_outputs, self.reference_outputs)

    def test_serves_after_daemon_exit(self):
        # Runs after the other tests of the class (tests run in name order).
        kill_process_tree(self.daemon_process.pid)
        deadline = time.time() + 60
        logs = ""
        while time.time() < deadline:
            logs = _read_logs(self.log_files)
            if logs.count("serving continues") >= self.tp_size:
                break
            time.sleep(1)
        self.assertEqual(
            logs.count("serving continues"),
            self.tp_size,
            "Expected every engine rank to log that it keeps serving after the "
            "daemon exited.",
        )
        self.assertEqual(
            _greedy_completions(self.base_url, self.model), self.greedy_outputs
        )


class TestWeightCacheDaemonTP1SmokeVmmFd(_VmmFdVariant, TestWeightCacheDaemonTP1Smoke):
    """TP=1 smoke over the vmm_fd transport."""


class TestWeightCacheDaemonTP2VmmFd(_VmmFdVariant, TestWeightCacheDaemonTP2):
    """TP=2 over the vmm_fd transport."""


class TestWeightCacheDaemonQwen3MoeDPVmmFd(
    _VmmFdVariant, TestWeightCacheDaemonQwen3MoeDP
):
    """Static attention DP over the vmm_fd transport."""


class TestWeightCacheDaemonQwen3MoeEPVmmFd(
    _VmmFdVariant, TestWeightCacheDaemonQwen3MoeEP
):
    """Qwen3 MoE expert parallelism + EPLB over the vmm_fd transport."""

    @classmethod
    def extra_args(cls) -> list:
        return _moe_runner_args()


class TestWeightCacheDaemonQwen3MoeAttnCPVmmFd(
    _VmmFdVariant, TestWeightCacheDaemonQwen3MoeAttnCP
):
    """Qwen3 MoE attention CP over the vmm_fd transport."""

    @classmethod
    def extra_args(cls) -> list:
        return _moe_runner_args()


if __name__ == "__main__":
    unittest.main()
