# SPDX-License-Identifier: Apache-2.0
"""The weight cache daemon serving a real engine on Intel XPU.

One daemon and three servers on one card: a disk-loading server for reference,
a client-mode server that takes its weights from the daemon, and a second
client-mode server that attaches while the daemon stays up.

What this adds over the transport tests, which share synthetic tensors between
two bare processes:

- the daemon starts on XPU at all, and ``--weight-cache-transport auto``
  resolves to vmm_fd there, since XPU serves no torch_ipc;
- a full model's parameters and buffers survive the layout, including whatever
  post-processing the loader did, so greedy generation matches the disk-loading
  server token for token;
- the KV pool is sized against memory the mapped weights are charged to, so a
  daemon-loaded engine never claims a larger pool than a disk-loading one on
  the same card;
- an engine can attach after another has come and gone.
"""

import os
import subprocess
import sys
import time
import unittest

import requests
import torch

from sglang.srt.platforms import current_platform
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=300, suite="extra-test-1-gpu-xpu")

MODEL = "Qwen/Qwen3-0.6B"
GPU_ID = 0
# Low enough that the daemon's copy of the weights and an engine's KV pool fit
# on one card next to each other.
MEM_FRACTION_STATIC = "0.5"
PROMPTS = (
    "The capital of France is",
    "In one sentence, a transistor is",
    "2 + 2 =",
)


def _greedy(base_url: str) -> list:
    outputs = []
    for prompt in PROMPTS:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 24,
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        outputs.append(response.json()["choices"][0]["text"])
    return outputs


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as handle:
        return handle.read()


class _Server:
    """A launched server and the log it wrote, killed on close()."""

    def __init__(self, name: str, extra_args: list):
        self.stdout_path = f"/tmp/wcd_xpu_{name}.out"
        self.stderr_path = f"/tmp/wcd_xpu_{name}.err"
        self.stdout = open(self.stdout_path, "w")
        self.stderr = open(self.stderr_path, "w")
        self.base_url = f"http://127.0.0.1:{find_available_port(31200)}"
        self.process = popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="xpu",
            other_args=[
                "--device",
                "xpu",
                "--base-gpu-id",
                str(GPU_ID),
                "--tp",
                "1",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                *extra_args,
            ],
            return_stdout_stderr=(self.stdout, self.stderr),
        )

    @property
    def logs(self) -> str:
        for stream in (self.stdout, self.stderr):
            if not stream.closed:
                try:
                    stream.flush()
                except OSError:
                    pass
        return _read(self.stdout_path) + _read(self.stderr_path)

    def max_total_num_tokens(self) -> int:
        marker = "max_total_num_tokens="
        logs = self.logs
        index = logs.rindex(marker) + len(marker)
        return int(logs[index:].split(",")[0].strip())

    def close(self):
        if self.process is not None:
            terminate_and_kill_process_tree(self.process)
            self.process = None
        for stream in (self.stdout, self.stderr):
            try:
                stream.close()
            except OSError:
                pass


@unittest.skipUnless(torch.xpu.is_available(), "no XPU device")
class TestXpuWeightCacheDaemonE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_uuid = current_platform.get_device_uuid(GPU_ID)
        cls.ready_path = get_ready_path(cls.device_uuid)
        cls.socket_path = get_socket_path(cls.device_uuid)
        for path in (cls.ready_path, cls.socket_path):
            if os.path.exists(path):
                os.unlink(path)

        # Reference first, so the card is otherwise idle for it.
        cls.disk_server = _Server("disk", [])
        cls.disk_outputs = _greedy(cls.disk_server.base_url)
        cls.disk_tokens = cls.disk_server.max_total_num_tokens()
        cls.disk_server.close()
        cls.disk_server = None

        cls.daemon_log = "/tmp/wcd_xpu_daemon.log"
        with open(cls.daemon_log, "w") as log:
            cls.daemon = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.weight_cache.daemon",
                    "--model-path",
                    MODEL,
                    "--device",
                    "xpu",
                    "--tp-size",
                    "1",
                    "--gpu-id",
                    str(GPU_ID),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while not os.path.exists(cls.ready_path):
            if time.time() > deadline:
                raise TimeoutError(
                    f"daemon not ready in time:\n{_read(cls.daemon_log)[-4000:]}"
                )
            if cls.daemon.poll() is not None:
                raise RuntimeError(
                    f"daemon exited with {cls.daemon.returncode}:\n"
                    f"{_read(cls.daemon_log)[-4000:]}"
                )
            time.sleep(2)

        cls.client = _Server("client", ["--weight-cache-mode", "client"])
        cls.client_outputs = _greedy(cls.client.base_url)
        # Snapshotted here, so that a case which retires this server (the
        # restart one, whichever order the cases run in) does not take the
        # evidence with it.
        cls.client_logs = cls.client.logs
        cls.client_tokens = cls.client.max_total_num_tokens()

    @classmethod
    def tearDownClass(cls):
        for server in (getattr(cls, "client", None), getattr(cls, "disk_server", None)):
            if server is not None:
                server.close()
        daemon = getattr(cls, "daemon", None)
        if daemon is not None and daemon.poll() is None:
            terminate_and_kill_process_tree(daemon)
        for path in (
            getattr(cls, "ready_path", None),
            getattr(cls, "socket_path", None),
        ):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def test_the_daemon_serves_vmm_fd_without_being_asked_for_it(self):
        """XPU serves no torch_ipc, so 'auto' must not resolve to it."""
        logs = _read(self.daemon_log)
        self.assertIn("Using transport backend: vmm_fd", logs)
        self.assertIn("[weight_cache] vmm_fd: packed", logs)

    def test_the_engine_loaded_its_weights_over_the_transport(self):
        logs = self.client_logs
        self.assertIn(
            "[weight_cache] vmm_fd: mapped",
            logs,
            "the engine did not map the daemon's allocations",
        )
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "the engine fell back to a disk load instead of reusing the weights",
        )

    def test_generation_matches_a_disk_loaded_server(self):
        """Every parameter, buffer and stride has to survive the layout for
        greedy decoding to land on the same tokens."""
        self.assertEqual(self.client_outputs, self.disk_outputs)
        self.assertTrue(all(output.strip() for output in self.client_outputs))

    def test_the_kv_pool_is_charged_for_the_mapped_weights(self):
        """The mapped weights are outside the torch allocator, so an engine
        that sized its pool from allocator bookkeeping would claim a *larger*
        pool than the disk-loading server did on an idle card -- and then run
        out of device memory."""
        daemon_logs = _read(self.daemon_log)
        self.assertIn("preloaded_weights_bytes=", daemon_logs)
        reported = int(
            daemon_logs.rsplit("preloaded_weights_bytes=", 1)[1].split(")")[0]
        )
        self.assertGreater(reported, 0)
        self.assertLessEqual(
            self.client_tokens,
            self.disk_tokens,
            "the daemon-loaded engine claimed a bigger KV pool than the "
            "disk-loading one, so the weights it mapped were not charged to it",
        )

    def test_a_second_engine_attaches_while_the_daemon_stays_up(self):
        """An engine restart is the point of the daemon: the next one must get
        its own fds and mapping from a daemon that never reloaded."""
        self.assertIsNone(self.daemon.poll(), "the daemon exited")
        self.client.close()
        again = _Server("client2", ["--weight-cache-mode", "client"])
        try:
            self.assertEqual(_greedy(again.base_url), self.disk_outputs)
            self.assertIn("[weight_cache] vmm_fd: mapped", again.logs)
            self.assertIn("Loaded model via IPC", again.logs)
        finally:
            again.close()
        self.assertIsNone(self.daemon.poll(), "the daemon exited while serving")
        self.assertEqual(
            _read(self.daemon_log).count("[weight_cache] vmm_fd: packed"),
            1,
            "the daemon packed the weights more than once",
        )


if __name__ == "__main__":
    unittest.main()
