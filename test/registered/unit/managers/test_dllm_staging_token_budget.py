"""Regression tests for the dLLM staging token-budget fallback.

Bug 1: ``PrefillAdder._get_dllm_remain_tokens`` falls back to
``rem_dllm_tokens`` when ``rem_total_tokens`` is exhausted (it charges
``max_new_tokens`` upfront per staging row every round, so long-output
workloads drive it negative long before the KV pool fills). Uncapped, the
fallback can grant a staging row more than ``block_size`` tokens, producing a
variable-length row that crashes the denoise algorithms' uniform-block reshape
(``view(B, block_size)``).

Bug 2: the fallback must only override an exhausted ``rem_total_tokens``, not
real pool exhaustion. A fresh block still charges
``ceil_paged(block_size) + page_size`` against ``cur_rem_tokens`` (see
``_update_prefill_budget``), so granting one at zero current capacity drives
the budget negative and ``alloc_for_extend`` raises on the real allocation.
Rows reusing retained FDFO KV (``needs_fresh_kv=False``) allocate zero fresh
tokens and must stay admittable regardless of current capacity.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import PrefillAdder

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_adder(
    *,
    rem_dllm_tokens: int,
    dllm_block_size: int,
    available_tokens: int,
    rem_total_token_offset: int = 0,
    page_size: int = 1,
) -> PrefillAdder:
    """Build a PrefillAdder with only the state `_get_dllm_remain_tokens` reads.

    `rem_total_tokens` and `cur_rem_tokens` are properties over the same
    allocator/tree-cache sizes minus their respective offsets. Mirror the real
    __init__ invariant (`rem_total_token_offset >= cur_rem_token_offset`, since
    rem_total additionally accrues max_new_tokens and the running-batch
    reservation): keep `cur_rem_token_offset` at 0, drive `cur_rem_tokens` via
    `available_tokens` and `rem_total_tokens` via `rem_total_token_offset`.
    """
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.rem_dllm_tokens = rem_dllm_tokens
    adder.dllm_block_size = dllm_block_size
    adder.page_size = page_size
    adder.is_all_swa = False
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: available_tokens
    )
    adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
    adder.rem_total_token_offset = rem_total_token_offset
    adder.cur_rem_token_offset = 0
    return adder


class TestDllmStagingTokenBudgetFallbackCap(CustomTestCase):
    BLOCK_SIZE = 32

    def test_exhausted_budget_fallback_is_capped_at_one_block(self):
        # rem_total_tokens <= 0 (upfront max_new_tokens charges) with a healthy
        # pool takes the fallback branch. The grant must be capped at one
        # block: anything larger produces a variable-length staging row and
        # breaks the uniform view(B, block_size) reshape.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
            rem_total_token_offset=4196,  # rem_total_tokens == -100
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)

    def test_fallback_refuses_a_sub_block_dllm_budget(self):
        # A partial grant is as unusable as an oversized one: the denoise
        # reshape needs whole block_size rows, so a budget below one block
        # must yield 0 (NO_TOKEN) rather than a ragged row.
        adder = _make_adder(
            rem_dllm_tokens=16,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
            rem_total_token_offset=4196,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_fallback_refuses_a_zero_dllm_budget(self):
        adder = _make_adder(
            rem_dllm_tokens=0,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
            rem_total_token_offset=4196,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_fallback_grants_exactly_one_block_at_the_boundary(self):
        # Exactly one block of dLLM budget: grant it whole, don't round down.
        adder = _make_adder(
            rem_dllm_tokens=self.BLOCK_SIZE,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
            rem_total_token_offset=4196,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)

    def test_fallback_refuses_a_fresh_block_at_zero_current_capacity(self):
        # The pool is really exhausted (cur_rem_tokens == 0): a fresh block
        # would drive cur_rem_tokens negative and alloc_for_extend would raise
        # on the real allocation. Refuse.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=0,
            rem_total_token_offset=100,  # rem_total_tokens == -100
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_fallback_gates_on_the_full_admission_charge(self):
        # _update_prefill_budget debits ceil_paged(extend) + page_size, so the
        # gate must cover ceil_paged(block_size) + page_size — a bare
        # block_size pool would still go negative by the page overhead.
        for page_size in (1, 4, 16):
            charge = -(-self.BLOCK_SIZE // page_size) * page_size + page_size
            for available, expected in (
                (charge - 1, 0),
                (charge, self.BLOCK_SIZE),
            ):
                adder = _make_adder(
                    rem_dllm_tokens=4096,
                    dllm_block_size=self.BLOCK_SIZE,
                    available_tokens=available,
                    rem_total_token_offset=available + 100,
                    page_size=page_size,
                )
                self.assertEqual(
                    adder._get_dllm_remain_tokens(),
                    expected,
                    f"page_size={page_size} available={available}",
                )

    def test_fallback_page_rounds_an_unaligned_block(self):
        # block_size not a page multiple: the admission charge pages the
        # extend up (ceil_paged(24) == 32 at page_size 16), so the gate is
        # 32 + 16 == 48, not 24 + 16.
        block, page = 24, 16
        for available, expected in ((47, 0), (48, block)):
            adder = _make_adder(
                rem_dllm_tokens=4096,
                dllm_block_size=block,
                available_tokens=available,
                rem_total_token_offset=available + 100,
                page_size=page,
            )
            self.assertEqual(adder._get_dllm_remain_tokens(), expected)

    def test_fallback_reuse_rows_bypass_the_current_capacity_gate(self):
        # Retained incomplete-block rows take the reuse_kv path in
        # alloc_for_extend and allocate zero fresh tokens: they must stay
        # admittable at zero (or negative) current capacity, or their
        # retained KV is stranded.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(
            adder._get_dllm_remain_tokens(needs_fresh_kv=False), self.BLOCK_SIZE
        )
        # The same state refuses a fresh-KV row.
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_normal_path_rounds_sub_block_budget_to_zero(self):
        # Positive rem_total_tokens below one block: round to 0 (NO_TOKEN),
        # not to a ragged 20-token row — and not to the whole-block fallback,
        # since a small positive budget is genuine and a full block would
        # overshoot it.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=20,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_normal_path_grants_one_full_block(self):
        # Positive rem_total_tokens at or above one block: grant exactly one
        # block (the min already caps at block_size). No cur_rem_tokens gate
        # needed: cur_rem_tokens >= rem_total_tokens by construction.
        for available in (self.BLOCK_SIZE, 100):
            adder = _make_adder(
                rem_dllm_tokens=4096,
                dllm_block_size=self.BLOCK_SIZE,
                available_tokens=available,
            )
            self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()
