from __future__ import annotations

import time
from types import SimpleNamespace
from typing import ClassVar, List, Optional

import requests

from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

DEFAULT_MODEL: str = "Qwen/Qwen3-0.6B"

DEFAULT_CHUNKED_PREFILL_SIZE: int = 256
DEFAULT_NUM_EXAMPLES: int = 100
DEFAULT_PREFIX_REPETITIONS: int = 320
DEFAULT_NUM_THREADS: int = 128
DEFAULT_MAX_TOKENS: int = 512

KV_CANARY_ARGS: List[str] = [
    "--kv-canary",
    "raise",
    "--kv-canary-real-data",
    "partial",
    "--kv-canary-sweep-interval",
    "100",
    "--cuda-graph-backend-prefill=disabled",
]


class ChunkedGsm8kMixin:
    __test__ = False
    use_kv_canary: ClassVar[bool] = True
    model: ClassVar[str] = DEFAULT_MODEL
    feature_args: ClassVar[List[str]] = []

    chunked_prefill_size: ClassVar[int] = DEFAULT_CHUNKED_PREFILL_SIZE
    prefix_repetitions: ClassVar[int] = DEFAULT_PREFIX_REPETITIONS
    num_examples: ClassVar[int] = DEFAULT_NUM_EXAMPLES
    num_threads: ClassVar[int] = DEFAULT_NUM_THREADS
    max_tokens: ClassVar[int] = DEFAULT_MAX_TOKENS
    gsm8k_threshold: ClassVar[float]

    def build_prefill_side_args(self) -> List[str]:
        canary = list(KV_CANARY_ARGS) if self.use_kv_canary else []
        return (
            ["--chunked-prefill-size", str(self.chunked_prefill_size)]
            + list(self.feature_args)
            + canary
        )

    def test_mixed_prefix_chunked(self):
        base_url = self.base_url.removesuffix("/v1").rstrip("/")
        prefix = (
            "The following is a synthetic cache stress document. "
            * self.prefix_repetitions
        )
        prompts = [
            prefix
            + (f"Section {i % 3}: additional context. " * 64)
            + f"\nWrite the number {i} and stop."
            for i in range(8)
        ]
        sampling_params = {"temperature": 0, "max_new_tokens": 16}

        def generate(text):
            response = requests.post(
                base_url + "/generate",
                json={"text": text, "sampling_params": sampling_params},
                timeout=300,
            )
            response.raise_for_status()
            return response.json()

        expected = []
        for prompt in prompts:
            requests.get(base_url + "/flush_cache", timeout=30).raise_for_status()
            expected.append(generate(prompt)["text"])
        requests.get(base_url + "/flush_cache", timeout=30).raise_for_status()
        for _ in range(2):
            results = generate(prompts)
            self.assertEqual([result["text"] for result in results], expected)
            for result in results:
                self.assertGreater(
                    result["meta_info"]["prompt_tokens"], self.chunked_prefill_size
                )

    def test_gsm8k_chunked(self):
        fixture_name = type(self).__name__

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=self.max_tokens,
            num_examples=self.num_examples,
            num_threads=self.num_threads,
            temperature=0.0,
        )
        tic = time.perf_counter()
        metrics = run_eval(args)
        metrics["elapsed_sec"] = time.perf_counter() - tic
        print(f"[{fixture_name}] {metrics} threshold={self.gsm8k_threshold:.4f}")

        score = metrics.get("score")
        self.assertIsNotNone(score, "run_eval returned no score")
        self.assertGreaterEqual(score, self.gsm8k_threshold)


class ChunkedTestBase(ChunkedGsm8kMixin, CustomTestCase):
    __test__ = False

    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    launch_timeout: ClassVar[int] = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    process: ClassVar[Optional[object]] = None

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.launch_timeout,
            other_args=cls("test_gsm8k_chunked").build_prefill_side_args(),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)


class ChunkedTestPDBase(ChunkedGsm8kMixin, PDDisaggregationServerBase):
    __test__ = False
    decode_feature_args: ClassVar[List[str]] = []

    @classmethod
    def setUpClass(cls):
        cls.extra_prefill_args = cls("test_gsm8k_chunked").build_prefill_side_args()
        canary = list(KV_CANARY_ARGS) if cls.use_kv_canary else []
        cls.extra_decode_args = canary + list(cls.decode_feature_args)
        PDDisaggregationServerBase.setUpClass()
        cls.model = try_cached_model(cls.model)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        PDDisaggregationServerBase.tearDownClass()
