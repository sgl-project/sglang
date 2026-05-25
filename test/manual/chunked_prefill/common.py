"""Shared scaffolding for chunked-prefill refactor manual accuracy fixtures.

Background: this suite is a per-feature manual test net for the chunked-prefill
scheduler refactor (see ``user-written/instructions.md`` and
``2026-05-25-testing-strategy-overview.md`` in the project notes). Each fixture
launches a server with one chunked-relevant feature flag set, runs the
mixed-prefix GSM8K eval, and asserts a deliberately loose score floor. The
primary safety net is KV canary (when its PR lands); this layer is a tertiary
backstop and is intentionally not registered with CI.

Why mixed-prefix gsm8k: standard gsm8k shares one few-shot prefix across all
questions, so under radix cache only the first few concurrent requests
actually exercise the chunked-prefill code path. ``gsm8k_mixed`` routes each
question deterministically through four prefix modes (standard / cluster /
random-sample / zero-shot) so the chunked path is hit on diverse content
within a single run. See ``simple_eval_gsm8k_mixed.py``.

Why a single loose threshold (0.50): mixed-prefix scores are lower and less
calibrated than standard gsm8k. Per-mode thresholds would need empirical
calibration data we don't have yet. A single conservative gate catches
catastrophic regressions (output garbage, server hung) without false alarms.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import ClassVar, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# === Knobs shared across every fixture in this suite =========================

# Force a small chunked-prefill size so even short prompts split. The value
# matches the convention in `test_pp_single_node.py` etc. Override per-fixture
# only if a feature requires a specific size.
DEFAULT_CHUNKED_PREFILL_SIZE: int = 256

# Number of test questions per run. With 4-way mixed prefix this is 25 per
# mode — enough to flush radix in mode 1/2 reliably while keeping the wall
# clock under ~5 minutes for default-config fixtures.
DEFAULT_NUM_EXAMPLES: int = 100

# Default few-shot count. Yields ~1500-2000 token prompts which already chunk
# under DEFAULT_CHUNKED_PREFILL_SIZE.
DEFAULT_NUM_SHOTS: int = 10

# For features whose semantics depend on prompt length exceeding a structural
# boundary (SWA window size, HiSparse top_k, ...). Yields ~3000-4000 token
# prompts.
LONG_PROMPT_NUM_SHOTS: int = 24

# Deliberately loose. See module docstring.
SCORE_THRESHOLD: float = 0.50

# Concurrency. 128 matches GSM8KMixin defaults in this repo.
DEFAULT_NUM_THREADS: int = 128

# Per-question generation budget.
DEFAULT_MAX_TOKENS: int = 512

# Seed for the mixed eval's per-question random sampling (mode 2). Fixed so
# runs are comparable across chunk_size sweeps.
DEFAULT_SEED: int = 42

# Once the KV canary PR lands, set this to the appropriate flag(s), e.g.
# ``["--enable-kv-canary"]``. Every fixture in this suite appends it to the
# server's other_args verbatim, so flipping this single constant turns the
# whole suite into canary-gated.
KV_CANARY_ARGS: List[str] = []


# === Base class ==============================================================


class ChunkedRefactorTestBase(CustomTestCase):
    """Base for one feature's chunked-prefill manual accuracy fixture.

    Subclasses must define:

      - ``model``: HF model id or local path
      - ``feature_args``: list of server args specific to the feature under
        test (e.g. ``["--tp-size", "2", "--pp-size", "2"]``). Do NOT include
        ``--chunked-prefill-size`` here — the base class adds it from
        ``chunked_prefill_size`` (so chunk_size overrides remain explicit and
        diff-able).

    Subclasses may override:

      - ``chunked_prefill_size`` (default ``DEFAULT_CHUNKED_PREFILL_SIZE``)
      - ``num_shots`` (default ``DEFAULT_NUM_SHOTS``; SWA / HiSparse should
        use ``LONG_PROMPT_NUM_SHOTS``)
      - ``num_examples`` (default ``DEFAULT_NUM_EXAMPLES``)
      - ``score_threshold`` (default ``SCORE_THRESHOLD``)

    This base is **intentionally not registered with any CI runner**. Run it
    by hand from ``test/manual/chunked_prefill/``.
    """

    # Subclass contract.
    model: ClassVar[str] = DEFAULT_MODEL_NAME_FOR_TEST
    feature_args: ClassVar[List[str]] = []

    # Knobs (override per fixture as needed).
    chunked_prefill_size: ClassVar[int] = DEFAULT_CHUNKED_PREFILL_SIZE
    num_shots: ClassVar[int] = DEFAULT_NUM_SHOTS
    num_examples: ClassVar[int] = DEFAULT_NUM_EXAMPLES
    num_threads: ClassVar[int] = DEFAULT_NUM_THREADS
    max_tokens: ClassVar[int] = DEFAULT_MAX_TOKENS
    score_threshold: ClassVar[float] = SCORE_THRESHOLD
    seed: ClassVar[int] = DEFAULT_SEED

    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    launch_timeout: ClassVar[int] = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    process: ClassVar[Optional[object]] = None

    @classmethod
    def build_other_args(cls) -> List[str]:
        """Compose the full ``other_args`` for ``popen_launch_server``.

        Order:
          1. ``--chunked-prefill-size <cls.chunked_prefill_size>``
          2. subclass ``feature_args``
          3. ``KV_CANARY_ARGS`` (so flipping the global constant covers all
             fixtures uniformly)
        """
        return (
            [
                "--chunked-prefill-size",
                str(cls.chunked_prefill_size),
            ]
            + list(cls.feature_args)
            + list(KV_CANARY_ARGS)
        )

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.launch_timeout,
            other_args=cls.build_other_args(),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _run_gsm8k_mixed(self) -> dict:
        """Run mixed-prefix GSM8K against the running server and return metrics."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k_mixed",
            api="completion",
            max_tokens=self.max_tokens,
            num_examples=self.num_examples,
            num_threads=self.num_threads,
            num_shots=self.num_shots,
            gsm8k_mixed_seed=self.seed,
            temperature=0.0,
        )
        tic = time.perf_counter()
        metrics = run_eval(args)
        elapsed = time.perf_counter() - tic

        # Always log per-mode breakdown — useful when debugging which prefix
        # pattern caused a regression.
        print(
            f"[{type(self).__name__}] gsm8k_mixed score={metrics.get('score'):.4f}",
            f"score_standard={metrics.get('score_standard', float('nan')):.4f}",
            f"score_cluster={metrics.get('score_cluster', float('nan')):.4f}",
            f"score_random={metrics.get('score_random', float('nan')):.4f}",
            f"score_zero_shot={metrics.get('score_zero_shot', float('nan')):.4f}",
            f"elapsed={elapsed:.1f}s",
            sep=" | ",
        )

        # Optional JSON dump for run_all.sh aggregation.
        results_dir = os.environ.get("CHUNKED_PREFILL_RESULTS_DIR")
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            import json

            payload = {
                "fixture": type(self).__name__,
                "elapsed_sec": elapsed,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            with open(
                os.path.join(results_dir, f"{type(self).__name__}.json"), "w"
            ) as f:
                json.dump(payload, f, indent=2)

        return metrics

    def test_gsm8k_mixed_chunked(self):
        """The single test method every fixture inherits.

        Asserts overall mixed-prefix score is at least ``SCORE_THRESHOLD``.
        Per-mode scores are logged but not gated — under canary the score is
        a backstop, not a primary detector.
        """
        metrics = self._run_gsm8k_mixed()
        score = metrics.get("score")
        self.assertIsNotNone(score, "run_eval returned no score")
        self.assertGreaterEqual(
            score,
            self.score_threshold,
            f"Mixed-prefix gsm8k score {score:.4f} below floor "
            f"{self.score_threshold:.2f}. Likely catastrophic regression — "
            f"per-mode breakdown printed above.",
        )
