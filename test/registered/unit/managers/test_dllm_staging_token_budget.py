"""Regression tests for the dLLM staging token-budget fallback cap.

Bug: ``PrefillAdder._get_dllm_remain_tokens`` falls back to ``rem_dllm_tokens``
when ``rem_total_tokens`` is exhausted (it charges ``max_new_tokens`` upfront
per staging row every round, so long-output workloads drive it negative long
before the KV pool fills). Uncapped, the fallback can grant a staging row more
than ``block_size`` tokens, producing a variable-length row that crashes the
denoise algorithms' uniform-block reshape (``view(B, block_size)``).
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import PrefillAdder

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_adder(
    *, rem_dllm_tokens: int, dllm_block_size: int, available_tokens: int
) -> PrefillAdder:
    """Build a PrefillAdder with only the state `_get_dllm_remain_tokens` reads.

    `rem_total_tokens` is a property over the allocator/tree-cache sizes minus
    `rem_total_token_offset`; drive it via `available_tokens`.
    """
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.rem_dllm_tokens = rem_dllm_tokens
    adder.dllm_block_size = dllm_block_size
    adder.is_all_swa = False
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: available_tokens
    )
    adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
    adder.rem_total_token_offset = 0
    return adder


class TestDllmStagingTokenBudgetFallbackCap(CustomTestCase):
    BLOCK_SIZE = 32

    def test_exhausted_budget_fallback_is_capped_at_one_block(self):
        # rem_total_tokens <= 0 takes the fallback branch. The grant must be
        # capped at one block: anything larger produces a variable-length
        # staging row and breaks the uniform view(B, block_size) reshape.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)

    def test_fallback_refuses_a_sub_block_dllm_budget(self):
        # A partial grant is as unusable as an oversized one: the denoise
        # reshape needs whole block_size rows, so a budget below one block
        # must yield 0 (NO_TOKEN) rather than a ragged row.
        adder = _make_adder(
            rem_dllm_tokens=16,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_fallback_refuses_a_zero_dllm_budget(self):
        adder = _make_adder(
            rem_dllm_tokens=0,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_fallback_grants_exactly_one_block_at_the_boundary(self):
        # Exactly one block is available: grant it whole, don't round down.
        adder = _make_adder(
            rem_dllm_tokens=self.BLOCK_SIZE,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)

    def test_normal_budget_path_is_unaffected(self):
        # Positive rem_total_tokens: min(rem_dllm, block, total), unchanged
        # by the cap.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=20,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 20)


if __name__ == "__main__":
    unittest.main()
