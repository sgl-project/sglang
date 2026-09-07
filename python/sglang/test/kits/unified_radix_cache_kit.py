import random
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.test.kl_multiturn_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)


def _random_suffixes(n, length, seed):
    """Generate n random token-id lists of the given length."""
    rng = random.Random(seed)
    return [[rng.randint(1, 30000) for _ in range(length)] for _ in range(n)]


class UnifiedRadixTreeTestMixin:
    """Mixin: gsm8k and multi-turn KL tests with multi-branch interleaving."""

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


class AccuracyTwoPassMixin:
    """Mixin: run an eval twice with flush in between, verify accuracy diff.

    Subclass must provide:
      - self.base_url
      - self.model (for logging)
    """

    gsm8k_threshold: float = 0.90
    num_gsm8k_questions: int = 200
    gsm8k_parallel: int = 40

    max_accuracy_diff: float = 0.02

    l3_prefetch_page_size: int = 64
    l3_prefetch_prompt_pages: int = 16
    # Max tokens that may stay uncached on a full-prompt re-request; the bound
    # depends on model architecture. Defaults to page_size; subclasses override.
    l3_prefetch_max_uncached_tokens: int = None

    def _run_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=10,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=16000,
            parallel=self.gsm8k_parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        return metrics["accuracy"]

    def _flush_cache(self):
        response = requests.post(
            self.base_url + "/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        response.raise_for_status()

    def _two_pass(self, name: str, run_fn, threshold: float):
        # First pass
        acc1 = run_fn()
        print(f"[{self.__class__.__name__}] {name} pass 1 accuracy: {acc1:.3f}")
        self.assertGreaterEqual(
            acc1,
            threshold,
            f"{name} pass 1 accuracy {acc1:.3f} < threshold {threshold}",
        )

        # Flush cache
        self._flush_cache()

        # Second pass
        acc2 = run_fn()
        print(f"[{self.__class__.__name__}] {name} pass 2 accuracy: {acc2:.3f}")
        self.assertGreaterEqual(
            acc2,
            threshold,
            f"{name} pass 2 accuracy {acc2:.3f} < threshold {threshold}",
        )

        # Verify diff (only fail when 2nd pass regressed)
        if acc1 > acc2:
            diff = abs(acc1 - acc2)
            print(
                f"[{self.__class__.__name__}] {name} accuracy diff: {diff:.3f} "
                f"(max allowed: {self.max_accuracy_diff})"
            )
            self.assertLessEqual(
                diff,
                # Allow floating-point noise at the exact threshold (e.g. 0.98 - 0.96).
                self.max_accuracy_diff + 1e-12,
                f"{name} accuracy diff {diff:.3f} exceeds max {self.max_accuracy_diff} "
                f"(pass1={acc1:.3f}, pass2={acc2:.3f})",
            )

    def test_gsm8k_two_passes(self):
        """Run GSM8K twice with flush in between, verify accuracy diff <= max_accuracy_diff."""
        self._two_pass("GSM8K", self._run_gsm8k, self.gsm8k_threshold)

    def test_l3_prefetch_full_prefix_hit_after_flush(self):
        from sglang.test.kl_test_utils import _flush_cache, _generate

        page = int(self.l3_prefetch_page_size)
        n_tokens = page * int(self.l3_prefetch_prompt_pages)
        max_uncached = int(
            self.l3_prefetch_max_uncached_tokens
            if self.l3_prefetch_max_uncached_tokens is not None
            else page
        )

        rng = random.Random(987)
        input_ids = [rng.randint(1, 30000) for _ in range(n_tokens)]

        _generate(self.base_url, [input_ids], max_new_tokens=4)
        _flush_cache(self.base_url)
        results = _generate(self.base_url, [input_ids], max_new_tokens=4)
        cached = int(results[0]["meta_info"]["cached_tokens"])

        expected_min = n_tokens - max_uncached
        self.assertGreaterEqual(
            cached,
            expected_min,
            f"cached_tokens={cached} < {expected_min} (= input_len - {max_uncached})",
        )
