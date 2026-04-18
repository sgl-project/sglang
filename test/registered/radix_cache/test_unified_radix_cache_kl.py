import random
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def _random_suffixes(n, length, seed):
    """Generate n random token-id lists of the given length."""
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


MAMBA_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAMBA_CHUNK_SIZE = 64
MAMBA_TRACK_INTERVAL = 128

SWA_MODEL = "openai/gpt-oss-20b"
FULL_MODEL = "Qwen/Qwen3-32B"

register_cuda_ci(est_time=594, suite="stage-c-test-4-gpu-h100")


class UnifiedRadixTreeTestMixin:
    """Mixin: gsm8k、mmlu and multi-turn KL tests with multi-branch interleaving."""

    kl_threshold: float = 0.003
    max_new_tokens: int = 512
    num_groups: int = 3
    branches_per_group: int = 3
    prefix_len: int = 512
    prefill_cache_assert = None
    decode_cache_assert = None

    gsm8k_threshold: float = 0.93
    mmlu_threshold: float = 0.8
    num_gsm8k_questions: int = 200

    def test_gsm8k(self):
        """Few-shot GSM8K math reasoning accuracy."""
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=10,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=16000,
            parallel=128,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)

    def test_mmlu(self):
        """Simple-evals MMLU multi-task accuracy."""
        from sglang.test.run_eval import run_eval as run_simple_eval

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_simple_eval(args)
        print(
            f"[{self.__class__.__name__}] MMLU score: {metrics['score']:.3f} "
            f"(threshold: {self.mmlu_threshold})"
        )
        self.assertGreaterEqual(metrics["score"], self.mmlu_threshold)

    def test_multiturn_logprobs_match(self):
        """Helper 1: 3-turn, no explicit cache seeding."""
        ids = self.input_ids[:4]
        n = len(ids)
        t2 = _random_suffixes(n, 512, seed=100)
        t3 = _random_suffixes(n, 256, seed=200)
        test_input_output_logprobs_match_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            ids,
            turn_suffixes=[t2, t3],
            assert_decode_cached_tokens=self.decode_cache_assert,
            max_new_tokens=self.max_new_tokens,
        )

    def test_multiturn_prefill_cache_hit_branching(self):
        """Helper 2: prefill hit + 2 decode-hit turns, multi-branch interleaved."""
        num_groups = self.num_groups
        branches = self.branches_per_group
        n = num_groups * branches
        rng = random.Random(456)
        prefix_ids, full_ids = [], []
        for g in range(num_groups):
            prefix = self.input_ids[g][: self.prefix_len]
            for b in range(branches):
                suffix = [rng.randint(1, 30000) for _ in range(256 + b * 64)]
                prefix_ids.append(list(prefix))
                full_ids.append(prefix + suffix)

        t2 = _random_suffixes(n, 512, seed=789)
        t3 = _random_suffixes(n, 256, seed=890)
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            prefix_input_ids=prefix_ids,
            full_input_ids=full_ids,
            turn_suffixes=[t2, t3],
            assert_prefill_cached_tokens=self.prefill_cache_assert,
            assert_decode_cached_tokens=self.decode_cache_assert,
            branches_per_group=branches,
            max_new_tokens=self.max_new_tokens,
        )

    def test_multiturn_decode_cache_hit_branching(self):
        """Helper 3: 3-turn decode hit, multi-branch interleaved."""
        num_groups = self.num_groups
        branches = self.branches_per_group
        n = num_groups * branches
        first_turn = []
        for g in range(num_groups):
            base = self.input_ids[g][: self.prefix_len]
            for _ in range(branches):
                first_turn.append(list(base))

        t2 = _random_suffixes(n, 512, seed=300)
        t3 = _random_suffixes(n, 256, seed=400)
        test_input_output_logprobs_match_decode_cache_hit_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            first_turn,
            turn_suffixes=[t2, t3],
            assert_decode_cached_tokens=self.decode_cache_assert,
            branches_per_group=branches,
            max_new_tokens=self.max_new_tokens,
        )


class TestUnifiedFullRadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Full attention."""

    kl_threshold = 0.0025

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.80",
                "--page-size",
                "64",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestUnifiedMambaRadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Mamba hybrid + UnifiedRadixCache."""

    kl_threshold = 0.003
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=MAMBA_CHUNK_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=MAMBA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = MAMBA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "2048",
                "--mem-fraction-static",
                "0.85",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(MAMBA_TRACK_INTERVAL),
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestUnifiedSWARadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """SWA hybrid + UnifiedRadixCache."""

    kl_threshold = 0.03
    gsm8k_threshold = 0.75
    mmlu_threshold = 0.75

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--disable-piecewise-cuda-graph",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
