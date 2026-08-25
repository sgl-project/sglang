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

Hardenings on top:

- The reuse exemption mirrors ``alloc_for_extend``'s ``reuse_kv`` predicate
  (``req_pool_idx is not None and bool(dllm_incomplete_ids)``): a row with
  incomplete ids but a freed req slot re-allocates a full fresh block, so it
  must NOT bypass the capacity gate.
- The reuse grant is uniform across budget regimes (only the dLLM concurrency
  budget applies), fixing an inversion where a reuse row was refused at
  small-positive ``rem_total_tokens`` yet admitted at negative.
- The fallback gate charges ``fresh_extra_charge`` (the mamba gap reserve that
  ``_update_prefill_budget`` also debits) on top of
  ``ceil_paged(block_size) + page_size``.
- ``process_dllm_staging_reqs`` treats NO_TOKEN as a per-row refusal: a
  fresh-needing row at the head no longer starves zero-cost reuse rows behind
  it, and one budget-exhausting admission per round no longer serializes the
  rest of the queue.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

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

    def test_reuse_rows_get_a_block_at_small_positive_rem_total(self):
        # Monotonicity: a reuse row allocates zero fresh tokens, so a genuine
        # small-positive rem_total_tokens must not refuse it. Previously it
        # was refused here (sub-block budget rounds to 0 on the normal path)
        # yet admitted at NEGATIVE rem_total via the fallback exemption — a
        # logical inversion.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=20,
        )
        self.assertEqual(
            adder._get_dllm_remain_tokens(needs_fresh_kv=False), self.BLOCK_SIZE
        )
        # The same state still rounds a fresh row to zero.
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_reuse_rows_still_respect_the_dllm_budget(self):
        # The reuse exemption only bypasses the token budgets; the dLLM
        # concurrency budget still applies in every rem_total regime.
        for available in (-100, 20, 4096):
            adder = _make_adder(
                rem_dllm_tokens=16,  # < block
                dllm_block_size=self.BLOCK_SIZE,
                available_tokens=available,
            )
            self.assertEqual(
                adder._get_dllm_remain_tokens(needs_fresh_kv=False),
                0,
                f"available={available}",
            )

    def test_fallback_gate_includes_the_fresh_extra_charge(self):
        # On a unified Mamba pool the real admission debit also includes the
        # mamba gap reserve (`_update_prefill_budget(...,
        # mamba_gap_reserve=...)`); the gate receives it as
        # `fresh_extra_charge` and must refuse when the pool covers the block
        # charge but not the extra.
        extra = 8
        charge = self.BLOCK_SIZE + 1 + extra  # ceil_paged(32) + page 1 + extra
        for available, expected in ((charge - 1, 0), (charge, self.BLOCK_SIZE)):
            adder = _make_adder(
                rem_dllm_tokens=4096,
                dllm_block_size=self.BLOCK_SIZE,
                available_tokens=available,
                rem_total_token_offset=available + 100,
            )
            self.assertEqual(
                adder._get_dllm_remain_tokens(fresh_extra_charge=extra),
                expected,
                f"available={available}",
            )


class _StubReq:
    """Just enough of Req for `add_dllm_staging_req`."""

    def __init__(self, *, rid, fill_len, incomplete_len, req_pool_idx):
        self.rid = rid
        self.full_untruncated_fill_ids = list(range(fill_len))
        self.prefix_indices = []
        self.dllm_incomplete_ids = list(range(incomplete_len))
        self.req_pool_idx = req_pool_idx
        self.retracted_stain = False
        self.mamba_pool_idx = 0  # irrelevant: _mamba_slot_cost == 0 below
        self.sampling_params = SimpleNamespace(max_new_tokens=64)
        self.extend_range = None

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end, length=end - start)


def _arm_admission(adder: PrefillAdder) -> PrefillAdder:
    """Add the state `add_dllm_staging_req` needs beyond the budget getter."""
    adder.dllm_config = SimpleNamespace()
    adder.can_run_list = []
    adder._mamba_slot_cost = 0  # non-Mamba: gap reserve is 0
    adder.rem_input_tokens = 1 << 30
    adder.log_hit_tokens = 0
    adder.log_input_tokens = 0
    return adder


class TestDllmStagingReuseExemptionPredicate(CustomTestCase):
    """`add_dllm_staging_req` must mirror `alloc_for_extend`'s reuse_kv
    predicate: exempt iff req_pool_idx is not None AND incomplete ids exist."""

    BLOCK_SIZE = 32

    def _exhausted_adder(self):
        # Real pool exhaustion: cur_rem_tokens == 0, rem_total_tokens < 0.
        return _arm_admission(
            _make_adder(
                rem_dllm_tokens=4096,
                dllm_block_size=self.BLOCK_SIZE,
                available_tokens=0,
                rem_total_token_offset=100,
            )
        )

    def test_incomplete_ids_with_a_freed_slot_is_not_exempt(self):
        # incomplete ids but req_pool_idx None: alloc_for_extend computes
        # reuse_kv=False and allocates a full fresh block, so the gate must
        # treat the row as fresh and refuse it at pool exhaustion.
        adder = self._exhausted_adder()
        req = _StubReq(rid="freed", fill_len=8, incomplete_len=8, req_pool_idx=None)
        self.assertEqual(adder.add_dllm_staging_req(req), AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])

    def test_retained_slot_with_incomplete_ids_is_exempt(self):
        # Both predicate halves hold: the true reuse row is admitted at pool
        # exhaustion. The tail recheck may return NO_TOKEN (no budget for a
        # NEXT row), but this row is already in can_run_list.
        adder = self._exhausted_adder()
        req = _StubReq(rid="reuse", fill_len=8, incomplete_len=8, req_pool_idx=7)
        adder.add_dllm_staging_req(req)
        self.assertEqual(adder.can_run_list, [req])

    def test_retained_slot_without_incomplete_ids_is_not_exempt(self):
        adder = self._exhausted_adder()
        req = _StubReq(rid="fresh", fill_len=8, incomplete_len=0, req_pool_idx=7)
        self.assertEqual(adder.add_dllm_staging_req(req), AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])


class _ScriptedAdder:
    """Adder stub replaying fixed per-rid results for the staging loop."""

    def __init__(self, results):
        self.results = dict(results)
        self.calls = []

    def add_dllm_staging_req(self, req):
        self.calls.append(req.rid)
        return self.results[req.rid]


class TestProcessDllmStagingReqsPerRowRefusal(CustomTestCase):
    """NO_TOKEN must refuse one row, not abort the round: rows behind a
    refused fresh-needing head (e.g. zero-cost reuse rows) still get their own
    admission check, and a post-admission NO_TOKEN (the tail recheck in
    `add_dllm_staging_req`) doesn't cut off the rest of the queue either.
    `process_dllm_staging_reqs` never touches `self`, so call it unbound."""

    @staticmethod
    def _run(adder, rids):
        reqs = [SimpleNamespace(rid=rid) for rid in rids]
        return SchedulerDllmMixin.process_dllm_staging_reqs(None, adder, reqs)

    def test_refused_head_does_not_starve_rows_behind_it(self):
        adder = _ScriptedAdder(
            {
                "fresh-head": AddReqResult.NO_TOKEN,
                "reuse-1": AddReqResult.CONTINUE,
                "reuse-2": AddReqResult.CONTINUE,
            }
        )
        result = self._run(adder, ["fresh-head", "reuse-1", "reuse-2"])
        self.assertEqual(adder.calls, ["fresh-head", "reuse-1", "reuse-2"])
        # Any refusal still reports NO_TOKEN so incoming reqs don't jump
        # ahead of starved staging rows (_process_batch_by_phase).
        self.assertEqual(result, AddReqResult.NO_TOKEN)

    def test_post_admission_no_token_does_not_serialize_the_queue(self):
        # The tail recheck in add_dllm_staging_req returns NO_TOKEN after a
        # SUCCESSFUL admission when the fresh budget for a next row is gone;
        # zero-cost rows behind it must still be offered admission (each does
        # its own reuse-aware check) instead of one row per round.
        adder = _ScriptedAdder(
            {
                "admitted-exhausts": AddReqResult.NO_TOKEN,
                "reuse-1": AddReqResult.CONTINUE,
            }
        )
        self._run(adder, ["admitted-exhausts", "reuse-1"])
        self.assertEqual(adder.calls, ["admitted-exhausts", "reuse-1"])

    def test_all_admitted_returns_continue(self):
        adder = _ScriptedAdder({"a": AddReqResult.CONTINUE, "b": AddReqResult.CONTINUE})
        result = self._run(adder, ["a", "b"])
        self.assertEqual(result, AddReqResult.CONTINUE)


if __name__ == "__main__":
    unittest.main()
