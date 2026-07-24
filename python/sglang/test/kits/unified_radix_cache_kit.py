import random
import torch
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)


def _random_suffixes(n, length, seed):
    """Generate n random token-id lists of the given length."""
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class UnifiedRadixTreeTestMixin:
    """Mixin: gsm8k, mmlu and multi-turn KL tests with multi-branch interleaving."""

    kl_threshold: float = 0.003
    max_new_tokens: int = 512
    num_groups: int = 3
    branches_per_group: int = 3
    prefix_len: int = 512
    prefill_cache_assert = None
    decode_cache_assert = None
    sampling_temperature: float = 1
    decode_hit_request_batch_size: int | None = None
    decode_hit_inter_batch_delay_s: float = 0

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
            sampling_temperature=self.sampling_temperature,
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
            sampling_temperature=self.sampling_temperature,
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
            sampling_temperature=self.sampling_temperature,
            request_batch_size=self.decode_hit_request_batch_size,
            inter_batch_delay_s=self.decode_hit_inter_batch_delay_s,
        )
