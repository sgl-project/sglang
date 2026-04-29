"""Shared fixture for DeepSeek-V4 cookbook launch-command tests.

Each sibling test_*.py declares one launch combination from the
"Basic Configuration" chapter of:
    https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4

It launches the server with that exact config, runs the AIME25
evaluation by invoking the corresponding subcommands of
``scripts/bench_gpqa_aime.py`` (run-aime25 + regrade-aime25), and
asserts the recovered metrics file appeared.

These are MANUAL tests (not CI). They expect the documented
container layout: nemo-skills venv at /sgl-workspace/ns-venv and
log dir at /sgl-workspace/logs. ``setup-ns`` is invoked once per
process if the venv is missing.

Knobs (env vars):
    DSV4_AIME25_NUM_REPEATS       (default 16)
    DSV4_AIME25_TEMPERATURE       (default 1.0)
    DSV4_AIME25_MAX_TOKENS        (default 400000)
    DSV4_AIME25_MAX_CONCURRENCY   (default 512)
    DSV4_AIME25_TIMEOUT_SEC       (default 21600)
    DSV4_AIME25_SCORE_THRESHOLD   (default 0.0; >0 to enforce)
    DSV4_AIME25_SKIP_SETUP_NS     (default 0; 1 to skip setup-ns even if venv is missing)
    DSV4_BENCH_SCRIPT             (override path to bench_gpqa_aime.py)
"""

import glob
import json
import os
import subprocess
import time
import unittest
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_gpqa_aime.py"
BENCH_SCRIPT = Path(os.environ.get("DSV4_BENCH_SCRIPT", str(DEFAULT_BENCH_SCRIPT)))

LOG_DIR = "/sgl-workspace/logs"
NS_VENV = "/sgl-workspace/ns-venv"

AIME25_NUM_REPEATS = int(os.environ.get("DSV4_AIME25_NUM_REPEATS", "16"))
AIME25_TEMPERATURE = float(os.environ.get("DSV4_AIME25_TEMPERATURE", "1.0"))
AIME25_MAX_TOKENS = int(os.environ.get("DSV4_AIME25_MAX_TOKENS", "400000"))
AIME25_MAX_CONCURRENCY = int(os.environ.get("DSV4_AIME25_MAX_CONCURRENCY", "512"))
AIME25_TIMEOUT_SEC = int(os.environ.get("DSV4_AIME25_TIMEOUT_SEC", "21600"))
AIME25_SCORE_THRESHOLD = float(os.environ.get("DSV4_AIME25_SCORE_THRESHOLD", "0.0"))
AIME25_SKIP_SETUP_NS = os.environ.get("DSV4_AIME25_SKIP_SETUP_NS", "0") == "1"


class Dsv4Aime25TestBase(CustomTestCase):
    """Subclass and set MODEL / OTHER_ARGS / EXTRA_ENV per cookbook cell."""

    MODEL: ClassVar[str] = ""
    OTHER_ARGS: ClassVar[List[str]] = []
    EXTRA_ENV: ClassVar[Dict[str, str]] = {}

    @classmethod
    def setUpClass(cls):
        if cls is Dsv4Aime25TestBase:
            raise unittest.SkipTest("base class; subclass to run")
        if not cls.MODEL or not cls.OTHER_ARGS:
            raise unittest.SkipTest(f"{cls.__name__}: MODEL and OTHER_ARGS must be set")
        cls.base_url = DEFAULT_URL_FOR_TEST
        env: Optional[Dict[str, str]] = dict(cls.EXTRA_ENV) if cls.EXTRA_ENV else None
        cls.process = popen_launch_server(
            cls.MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(cls.OTHER_ARGS),
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_aime25(self):
        if not BENCH_SCRIPT.exists():
            self.skipTest(f"bench script not found at {BENCH_SCRIPT}")

        bench_env = self._bench_env()
        self._setup_ns_if_needed(bench_env)

        before = self._snapshot_log_folders()
        self._run_bench(
            [
                "run-aime25",
                "--num-repeats",
                str(AIME25_NUM_REPEATS),
                "--temperature",
                str(AIME25_TEMPERATURE),
                "--max-tokens",
                str(AIME25_MAX_TOKENS),
                "--max-concurrency",
                str(AIME25_MAX_CONCURRENCY),
            ],
            bench_env,
        )
        log_folder = self._wait_for_new_log_folder(before)
        print(f"[{type(self).__name__}] AIME25 log folder: {log_folder}", flush=True)

        metrics_path = self._wait_for_metrics(log_folder)
        self._run_bench(["regrade-aime25", log_folder], bench_env)

        with open(metrics_path) as f:
            metrics = json.load(f)
        print(
            f"[{type(self).__name__}] AIME25 metrics: "
            f"{json.dumps(metrics, indent=2)}",
            flush=True,
        )

        score = self._extract_score(metrics)
        if AIME25_SCORE_THRESHOLD > 0:
            self.assertGreater(score, AIME25_SCORE_THRESHOLD)

    def _bench_env(self):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        env = os.environ.copy()
        env["HOST"] = host
        env["PORT"] = port
        return env

    def _run_bench(self, args, env):
        cmd = ["python", str(BENCH_SCRIPT), *args]
        print(f"[{type(self).__name__}] + {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, env=env)

    def _setup_ns_if_needed(self, env):
        if AIME25_SKIP_SETUP_NS or Path(NS_VENV).exists():
            return
        self._run_bench(["setup-ns"], env)

    @staticmethod
    def _snapshot_log_folders():
        return set(glob.glob(f"{LOG_DIR}/aime25_logs/*"))

    def _wait_for_new_log_folder(self, before):
        deadline = time.time() + 120
        while time.time() < deadline:
            new = sorted(self._snapshot_log_folders() - before)
            if new:
                return new[-1]
            time.sleep(2)
        self.fail("run-aime25 did not produce a new log folder within 120s")

    def _wait_for_metrics(self, folder):
        metrics_path = Path(folder) / "eval-results" / "aime25" / "metrics.json"
        deadline = time.time() + AIME25_TIMEOUT_SEC
        while time.time() < deadline:
            if metrics_path.exists():
                return metrics_path
            time.sleep(30)
        self.fail(
            f"AIME25 eval did not finish in {AIME25_TIMEOUT_SEC}s "
            f"({metrics_path} missing)"
        )

    @staticmethod
    def _extract_score(metrics):
        def walk(o):
            if isinstance(o, dict):
                if "symbolic_correct" in o:
                    return o["symbolic_correct"]
                for v in o.values():
                    s = walk(v)
                    if s is not None:
                        return s
            return None

        return walk(metrics) or 0.0
