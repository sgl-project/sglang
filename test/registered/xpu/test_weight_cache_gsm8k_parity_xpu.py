# SPDX-License-Identifier: Apache-2.0
"""GSM8K through a daemon-loaded engine on Intel XPU, against a disk-loaded one.

The end of the Phase 5 correctness gate. The tensor manifest test next door
proves the two models hold the same bytes in the same layout; this proves the
served model still answers, at the accuracy the checkpoint is supposed to reach,
when its weights came over Level Zero vmm_fd instead of off disk.

What is asserted and what is only reported, deliberately:

- Each side must clear ``gsm8k_score_threshold``. That is a property of the
  checkpoint and the repository's usual accuracy contract.
- The daemon-loaded engine must have actually mapped the daemon's weights.
  Without that the two runs are the same run and the comparison is vacuous.
- The two scores' difference is measured and printed next to the spread of two
  runs against one unchanged server, which is the eval's own run-to-run noise
  at this sampling. It is NOT asserted against a tolerance: the roadmap defines
  none, and inventing one would either pass everything or flake. Byte parity is
  what establishes equality; this establishes that the bytes still serve.

Sampling follows Qwen3's non-thinking recommendation rather than sgl-eval's
defaults, and is identical on both sides, as is the checkpoint, tokenizer,
template, example count and generation cap.
"""

import os
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.platforms import current_platform
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=700, suite="extra-test-1-gpu-xpu")

MODEL = "Qwen/Qwen3-0.6B"
GPU_ID = 0
# Low enough that the daemon's copy of the weights and the engine's KV pool fit
# on one card next to each other.
MEM_FRACTION_STATIC = "0.55"

# The cookbook's Battlemage deployment flags, minus the XPU graph backends:
# capture happens on both sides identically, so it cannot move the comparison,
# and it doubles a test that already starts three servers.
# docs/cookbook/autoregressive/Qwen/Qwen3.mdx
COOKBOOK_BMG_FLAGS = [
    "--attention-backend",
    "intel_xpu",
    "--page-size",
    "64",
]

NUM_EXAMPLES = 200
NUM_THREADS = 128
# Qwen3 non-thinking recommended sampling. sgl-eval would otherwise apply its
# own defaults, so these are passed explicitly on both sides.
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 2048
# Pinned, and sent to the server, so the only thing that could move the score
# between the two runs is the weights. Without it the eval samples freely and a
# few points of run-to-run noise swamp what this test is looking for.
SEED = 0
# Measured on Arc Pro B60 at exactly these settings: 0.660 and 0.655
# disk-loaded, 0.660 daemon-loaded. The binomial standard error at 200 examples
# is about 0.033, so this floor sits roughly three standard errors below the
# lowest observation. It is a floor on the checkpoint, not a parity tolerance:
# the parity claim rests on the byte-level manifest gate, not here.
SCORE_THRESHOLD = 0.55


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as handle:
        return handle.read()


class _Server:
    def __init__(self, name: str, workdir: str, extra_args: list):
        self.log_path = os.path.join(workdir, f"{name}.log")
        self.log = open(self.log_path, "w")
        self.base_url = f"http://127.0.0.1:{find_available_port(31800)}"
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
                *COOKBOOK_BMG_FLAGS,
                *extra_args,
            ],
            return_stdout_stderr=(self.log, self.log),
        )

    @property
    def logs(self) -> str:
        if not self.log.closed:
            try:
                self.log.flush()
            except OSError:
                pass
        return _read(self.log_path)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            # Graceful first. This class runs three servers in sequence on one
            # card, and a SIGKILLed predecessor can still be holding device
            # memory when the next one sizes its KV pool.
            terminate_and_kill_process_tree(self.process)
        self.process = None
        if not self.log.closed:
            self.log.close()


def _gsm8k(base_url: str) -> float:
    """GSM8K through sgl-eval against a running server, at fixed sampling."""
    metrics = run_sgl_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=base_url,
            model=MODEL,
            num_examples=NUM_EXAMPLES,
            num_threads=NUM_THREADS,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            chat_template_kwargs=None,
        )
    )
    return float(metrics["score"])


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestWeightCacheGsm8kParityXpu(CustomTestCase):
    disk = None
    client = None
    daemon = None
    uuid = ""

    @classmethod
    def setUpClass(cls):
        try:
            import sgl_eval  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "sgl-eval not installed; pip install 'sglang[test]'"
            )

        import tempfile

        cls.workdir = tempfile.mkdtemp(prefix="wcd_gsm8k_parity_xpu_")
        cls.uuid = current_platform.get_device_uuid(GPU_ID)
        cls._clear_daemon_files()
        try:
            # Disk first, and twice, so the eval's own spread on an unchanged
            # server is known before the daemon-loaded score is compared to it.
            cls.disk = _Server(
                "server_disk", cls.workdir, ["--weight-cache-mode", "off"]
            )
            cls.disk_score = _gsm8k(cls.disk.base_url)
            cls.disk_score_repeat = _gsm8k(cls.disk.base_url)
            cls.disk_logs = cls.disk.logs
            cls.disk.close()
            cls.disk = None

            cls._start_daemon()
            cls.client = _Server(
                "server_client", cls.workdir, ["--weight-cache-mode", "client"]
            )
            cls.client_logs = cls.client.logs
            cls.daemon_score = _gsm8k(cls.client.base_url)
        except BaseException:
            cls._shutdown()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._shutdown()

    @classmethod
    def _shutdown(cls):
        for server in (cls.disk, cls.client):
            if server is not None:
                server.close()
        cls.disk = cls.client = None
        if cls.daemon is not None and cls.daemon.poll() is None:
            terminate_and_kill_process_tree(cls.daemon)
        cls.daemon = None
        cls._clear_daemon_files()

    @classmethod
    def _clear_daemon_files(cls):
        if not cls.uuid:
            return
        for path in (get_ready_path(cls.uuid), get_socket_path(cls.uuid)):
            if os.path.exists(path):
                os.unlink(path)

    @classmethod
    def _start_daemon(cls):
        log_path = os.path.join(cls.workdir, "daemon.log")
        cls.daemon_log_path = log_path
        with open(log_path, "w") as log:
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
                    "--base-gpu-id",
                    str(GPU_ID),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        ready = get_ready_path(cls.uuid)
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while not os.path.exists(ready):
            if cls.daemon.poll() is not None:
                raise RuntimeError(
                    f"daemon exited with {cls.daemon.returncode}:\n"
                    f"{_read(log_path)[-4000:]}"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"daemon not ready in time:\n{_read(log_path)[-4000:]}"
                )
            time.sleep(2)

    def test_the_daemon_loaded_engine_really_used_the_daemon(self):
        # The comparison below means nothing if this engine disk-loaded too.
        self.assertIn("Loaded model via IPC", self.client_logs)
        self.assertNotIn("falling back to disk load", self.client_logs)
        self.assertIn("transport=vmm_fd", self.client_logs)

    def test_the_disk_reference_did_not_reach_a_daemon(self):
        self.assertNotIn("Loaded model via IPC", self.disk_logs)

    def test_disk_loaded_accuracy_clears_the_threshold(self):
        self.assertGreaterEqual(
            self.disk_score,
            SCORE_THRESHOLD,
            f"disk GSM8K {self.disk_score:.4f} (repeat "
            f"{self.disk_score_repeat:.4f}) below {SCORE_THRESHOLD}",
        )

    def test_daemon_loaded_accuracy_clears_the_threshold(self):
        self.assertGreaterEqual(
            self.daemon_score,
            SCORE_THRESHOLD,
            f"daemon GSM8K {self.daemon_score:.4f} below {SCORE_THRESHOLD}; "
            f"disk scored {self.disk_score:.4f}",
        )

    def test_report_the_measured_difference(self):
        """Print the numbers next to the eval's own spread, and assert only that
        all three runs produced a real score -- a graded run that silently
        scored nothing would otherwise look like agreement."""
        scores = {
            "disk": self.disk_score,
            "disk_repeat": self.disk_score_repeat,
            "daemon": self.daemon_score,
        }
        delta = abs(self.daemon_score - self.disk_score)
        same_server_spread = abs(self.disk_score_repeat - self.disk_score)
        print(
            f"gsm8k parity: disk={self.disk_score:.4f} "
            f"disk_repeat={self.disk_score_repeat:.4f} "
            f"daemon={self.daemon_score:.4f} "
            f"delta={delta:.4f} same_server_spread={same_server_spread:.4f} "
            f"num_examples={NUM_EXAMPLES} temperature={TEMPERATURE} "
            f"top_p={TOP_P} max_tokens={MAX_TOKENS} seed={SEED}"
        )
        for label, score in scores.items():
            with self.subTest(run=label):
                self.assertTrue(0.0 < score <= 1.0, f"{label} scored {score!r}")


if __name__ == "__main__":
    unittest.main()
