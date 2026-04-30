"""Shared fixture for DeepSeek-V4 cookbook launch-command tests.

Each sibling ``test_<hardware>_<model_size>.py`` declares ONE
``hardware x model_size`` cell from the cookbook (e.g. B200 x Flash)
and contains one ``CustomTestCase`` subclass per recipe
(Low-Latency / Balanced / Max-Throughput / CP, where supported).

Each subclass launches the server with the cookbook's exact flags and
runs the AIME25 evaluation by shelling out to ``sgl-eval run aime25``
(https://github.com/sgl-project/sgl-eval). sgl-eval bundles the AIME25
dataset and uses ``math_verify`` for grading, so no nemo-skills setup
is required and there is no separate ``regrade`` pass — the relaxed
extractor is built in.

Cookbook reference:
    https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4

These are MANUAL tests (not CI). ``sgl-eval`` must be on PATH.

Per-variant defaults (set on the Flash/Pro intermediate base classes):
    Flash recipes -> AIME25 score threshold 0.93
    Pro   recipes -> AIME25 score threshold 0.95

AIME25 knobs (env vars):
    DSV4_AIME25_NUM_REPEATS       (default 16  -> --n-repeats)
    DSV4_AIME25_TEMPERATURE       (default 1.0 -> --temperature)
    DSV4_AIME25_TOP_P             (default 1.0 -> --top-p)
    DSV4_AIME25_MAX_TOKENS        (default 65536 -> --max-tokens)
    DSV4_AIME25_NUM_THREADS       (default 512 -> --num-threads)
    DSV4_AIME25_OUT_DIR           (default /tmp/sgl-eval-out -> --out-dir)
    DSV4_AIME25_SCORE_METRIC      (default "pass@1"; key in sgl-eval JSON to assert on)
    DSV4_AIME25_SCORE_THRESHOLD   (default 0; >0 overrides the per-variant default)
    DSV4_SGL_EVAL_BIN             (default "sgl-eval"; override path to the CLI)

Multi-node knobs (only consumed by multi-node test classes; if either
is unset, those classes ``SkipTest``):
    DSV4_NODE_RANK                (per-node rank for --node-rank)
    DSV4_DIST_INIT_ADDR           (e.g. 10.0.0.1:20000 for --dist-init-addr)
"""

import json
import os
import shutil
import subprocess
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

SGL_EVAL_BIN = os.environ.get("DSV4_SGL_EVAL_BIN", "sgl-eval")

AIME25_NUM_REPEATS = int(os.environ.get("DSV4_AIME25_NUM_REPEATS", "16"))
AIME25_TEMPERATURE = float(os.environ.get("DSV4_AIME25_TEMPERATURE", "1.0"))
AIME25_TOP_P = float(os.environ.get("DSV4_AIME25_TOP_P", "1.0"))
AIME25_MAX_TOKENS = int(os.environ.get("DSV4_AIME25_MAX_TOKENS", "65536"))
AIME25_NUM_THREADS = int(os.environ.get("DSV4_AIME25_NUM_THREADS", "512"))
AIME25_OUT_DIR = os.environ.get("DSV4_AIME25_OUT_DIR", "/tmp/sgl-eval-out")
AIME25_SCORE_METRIC = os.environ.get("DSV4_AIME25_SCORE_METRIC", "pass@1")
AIME25_SCORE_THRESHOLD = float(os.environ.get("DSV4_AIME25_SCORE_THRESHOLD", "0.0"))

# DeepEP "large SMS" config — appears as `--deepep-config '{...}'` in every
# DeepEP recipe except multi-node ones (where it is gated off in the JSX).
DEEPEP_LARGE_SMS_CONFIG = (
    '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
)


def multinode_args(nnodes: int) -> List[str]:
    """Return CLI args for a multi-node launch, or skip the test.

    Reads DSV4_NODE_RANK and DSV4_DIST_INIT_ADDR from the env. Raises
    ``unittest.SkipTest`` when either is missing — call from inside
    ``setUpClass`` so the whole class skips cleanly.
    """
    rank = os.environ.get("DSV4_NODE_RANK")
    addr = os.environ.get("DSV4_DIST_INIT_ADDR")
    if rank is None or addr is None:
        raise unittest.SkipTest(
            "multi-node test requires DSV4_NODE_RANK and DSV4_DIST_INIT_ADDR"
        )
    return [
        "--nnodes",
        str(nnodes),
        "--node-rank",
        rank,
        "--dist-init-addr",
        addr,
    ]


class Dsv4Aime25TestBase(CustomTestCase):
    """Subclass via ``Dsv4FlashAime25TestBase`` or ``Dsv4ProAime25TestBase``,
    not directly. Per-recipe subclasses set MODEL / OTHER_ARGS / EXTRA_ENV.

    SCORE_THRESHOLD is set by the Flash/Pro intermediate base classes:
    Flash 0.93, Pro 0.95.
    """

    MODEL: ClassVar[str] = ""
    OTHER_ARGS: ClassVar[List[str]] = []
    EXTRA_ENV: ClassVar[Dict[str, str]] = {}

    SCORE_THRESHOLD: ClassVar[float] = 0.0

    _BASE_CLASSES: ClassVar[set] = set()

    @classmethod
    def setUpClass(cls):
        if cls in cls._BASE_CLASSES:
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
        if shutil.which(SGL_EVAL_BIN) is None:
            self.skipTest(f"{SGL_EVAL_BIN!r} not found on PATH")

        out_dir = Path(AIME25_OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        before = set(out_dir.glob("sgl_eval_aime25_*.json"))

        cmd = [
            SGL_EVAL_BIN,
            "run",
            "aime25",
            "--base-url",
            f"{self.base_url}/v1",
            "--n-repeats",
            str(AIME25_NUM_REPEATS),
            "--temperature",
            str(AIME25_TEMPERATURE),
            "--top-p",
            str(AIME25_TOP_P),
            "--max-tokens",
            str(AIME25_MAX_TOKENS),
            "--num-threads",
            str(AIME25_NUM_THREADS),
            "--out-dir",
            str(out_dir),
        ]
        print(f"[{type(self).__name__}] + {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        new = sorted(set(out_dir.glob("sgl_eval_aime25_*.json")) - before)
        if not new:
            self.fail(f"sgl-eval produced no new results JSON in {out_dir}")
        result_path = new[-1]
        with open(result_path) as f:
            result = json.load(f)
        print(
            f"[{type(self).__name__}] sgl-eval result ({result_path.name}): "
            f"{json.dumps(result, indent=2)}",
            flush=True,
        )

        score = self._extract_score(result, AIME25_SCORE_METRIC)
        threshold = (
            AIME25_SCORE_THRESHOLD
            if AIME25_SCORE_THRESHOLD > 0
            else self.SCORE_THRESHOLD
        )
        if threshold > 0:
            self.assertGreaterEqual(
                score,
                threshold,
                f"{AIME25_SCORE_METRIC}={score} below threshold {threshold}",
            )

    @staticmethod
    def _extract_score(result, metric):
        """Find ``metric`` (e.g. "pass@1") anywhere in the sgl-eval JSON tree."""

        def walk(o):
            if isinstance(o, dict):
                if metric in o and isinstance(o[metric], (int, float)):
                    return float(o[metric])
                for v in o.values():
                    s = walk(v)
                    if s is not None:
                        return s
            elif isinstance(o, list):
                for v in o:
                    s = walk(v)
                    if s is not None:
                        return s
            return None

        score = walk(result)
        if score is None:
            raise AssertionError(f"metric {metric!r} not found in sgl-eval result JSON")
        return score


class Dsv4FlashAime25TestBase(Dsv4Aime25TestBase):
    """Base for DeepSeek-V4-Flash recipes: AIME25 threshold 0.93."""

    SCORE_THRESHOLD = 0.93


class Dsv4ProAime25TestBase(Dsv4Aime25TestBase):
    """Base for DeepSeek-V4-Pro recipes: AIME25 threshold 0.95."""

    SCORE_THRESHOLD = 0.95


Dsv4Aime25TestBase._BASE_CLASSES = {
    Dsv4Aime25TestBase,
    Dsv4FlashAime25TestBase,
    Dsv4ProAime25TestBase,
}
