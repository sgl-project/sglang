from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

DEFAULT_MODEL: str = "Qwen/Qwen3-0.6B"

DEFAULT_CHUNKED_PREFILL_SIZE: int = 256
DEFAULT_NUM_EXAMPLES: int = 100
DEFAULT_NUM_SHOTS: int = 10
LONG_PROMPT_NUM_SHOTS: int = 24
DEFAULT_NUM_THREADS: int = 128
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_SEED: int = 42

# The canary CLI is --kv-canary <mode> (+ companions); --enable-kv-canary does
# not exist and made every launched server exit at argparse.
KV_CANARY_ARGS: List[str] = [
    "--kv-canary",
    "raise",
    "--kv-canary-real-data",
    "partial",
    "--kv-canary-sweep-interval",
    "100",
]


@dataclass
class ChunkedSimpleTester:
    feature_args: List[str]
    chunked_prefill_size: int
    num_shots: int
    num_examples: int
    num_threads: int
    max_tokens: int
    gsm8k_threshold: float
    seed: int

    def build_prefill_side_args(self) -> List[str]:
        return (
            ["--chunked-prefill-size", str(self.chunked_prefill_size)]
            + list(self.feature_args)
            + list(KV_CANARY_ARGS)
        )

    def build_decode_side_args(self) -> List[str]:
        return list(KV_CANARY_ARGS)

    def run_eval(self, base_url: str, model: str, fixture_name: str) -> dict:
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mixed_prefix_gsm8k",
            api="chat_completion",
            max_tokens=self.max_tokens,
            num_examples=self.num_examples,
            num_threads=self.num_threads,
            num_shots=self.num_shots,
            mixed_prefix_gsm8k_secondary_pool_size=15,
            mixed_prefix_gsm8k_seed=self.seed,
            gsm8k_data_path=None,
            temperature=0.0,
        )
        tic = time.perf_counter()
        metrics = run_eval(args)
        metrics["elapsed_sec"] = time.perf_counter() - tic
        print(f"[{fixture_name}] {metrics}")
        return metrics

    def assert_score(self, testcase, metrics: dict, fixture_name: str) -> None:
        score = metrics.get("score")
        testcase.assertIsNotNone(score, "run_eval returned no score")
        passed = score >= self.gsm8k_threshold
        status = "PASS" if passed else "FAIL"
        print(
            f"[{fixture_name}] gsm8k score={score:.4f} "
            f"threshold={self.gsm8k_threshold:.4f} -> {status}"
        )
        testcase.assertGreaterEqual(score, self.gsm8k_threshold)


def _build_simple_tester(cls) -> ChunkedSimpleTester:
    return ChunkedSimpleTester(
        feature_args=cls.feature_args,
        chunked_prefill_size=cls.chunked_prefill_size,
        num_shots=cls.num_shots,
        num_examples=cls.num_examples,
        num_threads=cls.num_threads,
        max_tokens=cls.max_tokens,
        gsm8k_threshold=cls.gsm8k_threshold,
        seed=cls.seed,
    )


class ChunkedTestBase(CustomTestCase):
    # base class: not collectible as a test on its own
    __test__ = False
    model: ClassVar[str] = DEFAULT_MODEL
    feature_args: ClassVar[List[str]] = []

    chunked_prefill_size: ClassVar[int] = DEFAULT_CHUNKED_PREFILL_SIZE
    num_shots: ClassVar[int] = DEFAULT_NUM_SHOTS
    num_examples: ClassVar[int] = DEFAULT_NUM_EXAMPLES
    num_threads: ClassVar[int] = DEFAULT_NUM_THREADS
    max_tokens: ClassVar[int] = DEFAULT_MAX_TOKENS
    gsm8k_threshold: ClassVar[float]
    seed: ClassVar[int] = DEFAULT_SEED

    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    launch_timeout: ClassVar[int] = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    process: ClassVar[Optional[object]] = None
    _simple_tester: ClassVar[Optional[ChunkedSimpleTester]] = None

    @classmethod
    def setUpClass(cls):
        cls._simple_tester = _build_simple_tester(cls)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.launch_timeout,
            other_args=cls._simple_tester.build_prefill_side_args(),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_mixed_prefix_gsm8k_chunked(self):
        fixture_name = type(self).__name__
        metrics = self._simple_tester.run_eval(self.base_url, self.model, fixture_name)
        self._simple_tester.assert_score(self, metrics, fixture_name)


class ChunkedTestPDBase(PDDisaggregationServerBase):
    # base class: not collectible as a test on its own
    __test__ = False
    model: ClassVar[str] = DEFAULT_MODEL
    feature_args: ClassVar[List[str]] = []
    # Extra args for the decode server only (e.g. matching its TP to the prefill
    # side so the transferred KV layout lines up).
    decode_feature_args: ClassVar[List[str]] = []

    chunked_prefill_size: ClassVar[int] = DEFAULT_CHUNKED_PREFILL_SIZE
    num_shots: ClassVar[int] = DEFAULT_NUM_SHOTS
    num_examples: ClassVar[int] = DEFAULT_NUM_EXAMPLES
    num_threads: ClassVar[int] = DEFAULT_NUM_THREADS
    max_tokens: ClassVar[int] = DEFAULT_MAX_TOKENS
    gsm8k_threshold: ClassVar[float]
    seed: ClassVar[int] = DEFAULT_SEED

    _simple_tester: ClassVar[Optional[ChunkedSimpleTester]] = None

    @classmethod
    def setUpClass(cls):
        cls._simple_tester = _build_simple_tester(cls)
        cls.extra_prefill_args = cls._simple_tester.build_prefill_side_args()
        cls.extra_decode_args = cls._simple_tester.build_decode_side_args() + list(
            cls.decode_feature_args
        )
        PDDisaggregationServerBase.setUpClass()
        cls.model = try_cached_model(cls.model)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        PDDisaggregationServerBase.tearDownClass()

    def test_mixed_prefix_gsm8k_chunked(self):
        fixture_name = type(self).__name__
        metrics = self._simple_tester.run_eval(self.base_url, self.model, fixture_name)
        self._simple_tester.assert_score(self, metrics, fixture_name)
