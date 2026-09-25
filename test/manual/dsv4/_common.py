"""Shared fixture for DeepSeek-V4 cookbook launch-command tests.

Each sibling ``test_<hardware>_<model_size>.py`` declares ONE
``hardware x model_size`` cell from the cookbook (e.g. B200 x Flash)
and contains one ``CustomTestCase`` subclass per recipe
(Low-Latency / Balanced / Max-Throughput / CP, where supported).

Each subclass launches the server with the cookbook's exact flags and
runs two sgl-eval evaluations (https://github.com/sgl-project/sgl-eval):
- ``test_smoke_gsm8k`` — short, cheap GSM8K pass to verify the server
  can produce coherent math answers at all (sanity gate).
- ``test_aime25`` — full AIME25 accuracy run (heavy; 16 repeats default).

Cookbook reference:
    https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4

These are MANUAL tests (not CI). The ``sgl-eval`` package must be installed.

Per-variant defaults (set on the Flash/Pro intermediate base classes):
    Flash recipes -> AIME25 score threshold 0.93
    Pro   recipes -> AIME25 score threshold 0.95
GSM8K sanity threshold (0.93) is shared across Flash and Pro.

AIME25 knobs (env vars):
    DSV4_AIME25_NUM_REPEATS       (default 16    -> --n-repeats)
    DSV4_AIME25_TEMPERATURE       (default 1.0   -> --temperature)
    DSV4_AIME25_TOP_P             (default 1.0   -> --top-p)
    DSV4_AIME25_MAX_TOKENS        (default 65536 -> --max-tokens)
    DSV4_AIME25_NUM_THREADS       (default 512   -> --num-threads)
    DSV4_AIME25_SCORE_METRIC      (default "score"; sgl-eval JSON key under "aggregate")
    DSV4_AIME25_SCORE_THRESHOLD   (default 0; >0 overrides per-variant default)

GSM8K sanity knobs (env vars):
    DSV4_GSM8K_NUM_EXAMPLES       (default 50    -> --num-examples)
    DSV4_GSM8K_N_REPEATS          (default 1     -> --n-repeats)
    DSV4_GSM8K_TEMPERATURE        (default 0.6   -> --temperature)
    DSV4_GSM8K_TOP_P              (default 0.95  -> --top-p)
    DSV4_GSM8K_MAX_TOKENS         (default 8192  -> --max-tokens)
    DSV4_GSM8K_NUM_THREADS        (default 64    -> --num-threads)
    DSV4_GSM8K_SCORE_METRIC       (default "score"; sgl-eval JSON key under "aggregate")
    DSV4_GSM8K_SCORE_THRESHOLD    (default 0.93; set to 0 to skip the assertion)

Shared knobs:
    DSV4_SGL_EVAL_OUT_DIR         (default /tmp/sgl-eval-out)
    DSV4_SERVER_LAUNCH_TIMEOUT    (default 3600s; the sglang 600s default is
                                   too short for DSV4 model load + DeepGEMM
                                   warmup. 1800s is also tight for the heavier
                                   recipes (DP-attn + DeepEP); 3600s is the
                                   safe default. Bump again for first-run
                                   model downloads if needed.)

Multi-node knobs (only consumed by multi-node test classes; if either
is unset, those classes ``SkipTest``):
    DSV4_NODE_RANK                (per-node rank for --node-rank)
    DSV4_DIST_INIT_ADDR           (e.g. 10.0.0.1:20000 for --dist-init-addr)

Always-on env (set by the base class for every recipe; per-recipe EXTRA_ENV
wins on key conflict):
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1   skip the slow DeepGEMM warmup grid
"""

import os
import unittest
from types import SimpleNamespace
from typing import ClassVar, Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

SGL_EVAL_OUT_DIR = os.environ.get("DSV4_SGL_EVAL_OUT_DIR", "/tmp/sgl-eval-out")

# DSV4 server launch needs more than the 600s sglang default: model load alone
# can take 5+ min and DeepGEMM warmup another ~5 min. First-run model download
# adds ~10-30 min on top. 1800s covers steady-state; bump via env for downloads.
SERVER_LAUNCH_TIMEOUT = int(os.environ.get("DSV4_SERVER_LAUNCH_TIMEOUT", "3600"))

# Defaults applied to every recipe's EXTRA_ENV. Per-recipe EXTRA_ENV wins on key
# conflict.
BASE_ENV: Dict[str, str] = {
    # Skip the slow exhaustive DeepGEMM warmup grid; covers the shapes DSV4
    # actually hits and shaves several minutes off server startup.
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
}

AIME25_NUM_REPEATS = int(os.environ.get("DSV4_AIME25_NUM_REPEATS", "16"))
AIME25_TEMPERATURE = float(os.environ.get("DSV4_AIME25_TEMPERATURE", "1.0"))
AIME25_TOP_P = float(os.environ.get("DSV4_AIME25_TOP_P", "1.0"))
AIME25_MAX_TOKENS = int(os.environ.get("DSV4_AIME25_MAX_TOKENS", "65536"))
AIME25_NUM_THREADS = int(os.environ.get("DSV4_AIME25_NUM_THREADS", "512"))
AIME25_SCORE_METRIC = os.environ.get("DSV4_AIME25_SCORE_METRIC", "score")
AIME25_SCORE_THRESHOLD = float(os.environ.get("DSV4_AIME25_SCORE_THRESHOLD", "0.0"))

GSM8K_NUM_EXAMPLES = int(os.environ.get("DSV4_GSM8K_NUM_EXAMPLES", "50"))
GSM8K_N_REPEATS = int(os.environ.get("DSV4_GSM8K_N_REPEATS", "1"))
GSM8K_TEMPERATURE = float(os.environ.get("DSV4_GSM8K_TEMPERATURE", "0.6"))
GSM8K_TOP_P = float(os.environ.get("DSV4_GSM8K_TOP_P", "0.95"))
GSM8K_MAX_TOKENS = int(os.environ.get("DSV4_GSM8K_MAX_TOKENS", "8192"))
GSM8K_NUM_THREADS = int(os.environ.get("DSV4_GSM8K_NUM_THREADS", "64"))
GSM8K_SCORE_METRIC = os.environ.get("DSV4_GSM8K_SCORE_METRIC", "score")
GSM8K_SCORE_THRESHOLD = float(os.environ.get("DSV4_GSM8K_SCORE_THRESHOLD", "0.93"))

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


class DSV4Aime25TestBase(CustomTestCase):
    """Subclass via ``DSV4FlashAime25TestBase`` or ``DSV4ProAime25TestBase``,
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
        env: Optional[Dict[str, str]] = {**BASE_ENV, **(cls.EXTRA_ENV or {})}
        cls.process = popen_launch_server(
            cls.MODEL,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=list(cls.OTHER_ARGS),
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_smoke_gsm8k(self):
        """Quick GSM8K pass to verify the server is producing math answers."""
        self._run_sgl_eval(
            eval_name="gsm8k",
            n_repeats=GSM8K_N_REPEATS,
            temperature=GSM8K_TEMPERATURE,
            top_p=GSM8K_TOP_P,
            max_tokens=GSM8K_MAX_TOKENS,
            num_threads=GSM8K_NUM_THREADS,
            num_examples=GSM8K_NUM_EXAMPLES,
            metric=GSM8K_SCORE_METRIC,
            threshold=GSM8K_SCORE_THRESHOLD,
        )

    def test_aime25(self):
        """Full AIME25 accuracy run; threshold gated by Flash vs Pro base."""
        threshold = (
            AIME25_SCORE_THRESHOLD
            if AIME25_SCORE_THRESHOLD > 0
            else self.SCORE_THRESHOLD
        )
        self._run_sgl_eval(
            eval_name="aime25",
            n_repeats=AIME25_NUM_REPEATS,
            temperature=AIME25_TEMPERATURE,
            top_p=AIME25_TOP_P,
            max_tokens=AIME25_MAX_TOKENS,
            num_threads=AIME25_NUM_THREADS,
            num_examples=None,
            metric=AIME25_SCORE_METRIC,
            threshold=threshold,
        )

    def _run_sgl_eval(
        self,
        eval_name,
        n_repeats,
        temperature,
        top_p,
        max_tokens,
        num_threads,
        num_examples,
        metric,
        threshold,
    ):
        args = SimpleNamespace(
            eval_name=eval_name,
            base_url=self.base_url,
            repeat=n_repeats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            num_threads=num_threads,
            num_examples=num_examples,
            sgl_eval_out_dir=SGL_EVAL_OUT_DIR,
        )
        score = run_sgl_eval(args)[metric]
        if threshold > 0:
            self.assertGreaterEqual(
                score,
                threshold,
                f"{eval_name} {metric}={score} below threshold {threshold}",
            )


class DSV4FlashAime25TestBase(DSV4Aime25TestBase):
    """Base for DeepSeek-V4-Flash recipes: AIME25 threshold 0.93."""

    SCORE_THRESHOLD = 0.93


class DSV4ProAime25TestBase(DSV4Aime25TestBase):
    """Base for DeepSeek-V4-Pro recipes: AIME25 threshold 0.95."""

    SCORE_THRESHOLD = 0.95


DSV4Aime25TestBase._BASE_CLASSES = {
    DSV4Aime25TestBase,
    DSV4FlashAime25TestBase,
    DSV4ProAime25TestBase,
}
