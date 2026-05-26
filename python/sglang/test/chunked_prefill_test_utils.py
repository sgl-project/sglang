from __future__ import annotations

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

DEFAULT_CHUNKED_PREFILL_SIZE: int = 256
DEFAULT_NUM_EXAMPLES: int = 100
DEFAULT_NUM_SHOTS: int = 10
LONG_PROMPT_NUM_SHOTS: int = 24
SCORE_THRESHOLD: float = 0.50
DEFAULT_NUM_THREADS: int = 128
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_SEED: int = 42

KV_CANARY_ARGS: List[str] = ["--enable-kv-canary"]


class ChunkedRefactorTestBase(CustomTestCase):
    model: ClassVar[str] = DEFAULT_MODEL_NAME_FOR_TEST
    feature_args: ClassVar[List[str]] = []

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
        return (
            ["--chunked-prefill-size", str(cls.chunked_prefill_size)]
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

        print(
            f"[{type(self).__name__}] gsm8k_mixed score={metrics.get('score', float('nan')):.4f}",
            f"score_standard={metrics.get('score_standard', float('nan')):.4f}",
            f"score_cluster={metrics.get('score_cluster', float('nan')):.4f}",
            f"score_random={metrics.get('score_random', float('nan')):.4f}",
            f"score_zero_shot={metrics.get('score_zero_shot', float('nan')):.4f}",
            f"elapsed={elapsed:.1f}s",
            sep=" | ",
        )
        return metrics

    def test_gsm8k_mixed_chunked(self):
        metrics = self._run_gsm8k_mixed()
        score = metrics.get("score")
        self.assertIsNotNone(score, "run_eval returned no score")
        self.assertGreaterEqual(score, self.score_threshold)
