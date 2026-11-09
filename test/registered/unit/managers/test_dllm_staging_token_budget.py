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
``ceil_paged(block_size) + per_req_token_overhead`` against ``cur_rem_tokens``
(``_update_prefill_budget`` reserves the page-ceiled extend plus the
per-request allocation rounding through ``PrefillBudget.reserve``), so
granting one at zero current capacity drives the budget negative and
``alloc_for_extend`` raises on the real allocation. Rows reusing retained
FDFO KV (``reuses_kv=True``) allocate zero fresh tokens and must stay
admittable regardless of current capacity.

Hardenings on top:

- The reuse exemption mirrors ``alloc_for_extend``'s ``reuse_kv`` predicate
  (``kv.holds_kv and bool(dllm_incomplete_ids)``): a row with incomplete ids
  but a freed req slot re-allocates a full fresh block, so it must NOT bypass
  the capacity gate.
- The reuse grant is uniform across budget regimes (only the dLLM concurrency
  budget applies), fixing an inversion where a reuse row was refused at
  small-positive ``rem_total_tokens`` yet admitted at negative.
- ``process_dllm_staging_reqs`` treats NO_TOKEN as a per-row refusal: a
  fresh-needing row at the head no longer starves zero-cost reuse rows behind
  it, and one budget-exhausting admission per round no longer serializes the
  rest of the queue.
- ``_maybe_break_dllm_staging_stall`` breaks the livelock the capacity gate
  can otherwise create: retained rows that all need a fresh block can hold the
  whole pool, so every round refuses every row, admits nothing, and nothing
  ever resolves or frees KV. A zero-admission round with a capacity refusal
  (and no in-flight batch) retracts one retained row — picked by the standard
  decode-retraction order — to fund the next round.

The capped fallback and whole-block rounding only apply to the classic dLLM
staging path; separate-context-encoding models keep upstream's fallback, whose
``cur_rem_tokens - page_size`` cap already protects the pool.
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import (
    _leave_owner_list,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget

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
    allocator/tree-cache sizes minus the memory_budget's respective offsets.
    Mirror the real __init__ invariant (`total_offset >= current_offset`, since
    the total additionally accrues max_new_tokens and the running-batch
    reservation): keep `current_offset` at 0, drive `cur_rem_tokens` via
    `available_tokens` and `rem_total_tokens` via `rem_total_token_offset`.
    Use the real `PrefillBudget` so the tests exercise the production offset
    arithmetic, not a re-implementation of it.
    """
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.rem_dllm_tokens = rem_dllm_tokens
    adder.dllm_block_size = dllm_block_size
    adder.page_size = page_size
    # No CP sharding and no KV sharding in these tests: the logical page is
    # the page, and the per-request allocation-rounding overhead is one
    # logical page (see PrefillAdder.__init__).
    adder.logical_page_size = page_size
    adder.per_req_token_overhead = page_size
    adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=False)
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: available_tokens, page_size=page_size
    )
    adder.tree_cache = SimpleNamespace(
        evictable_size=lambda: 0, supports_mamba=lambda: False
    )
    adder.memory_budget = PrefillBudget(
        adder.token_to_kv_pool_allocator, adder.tree_cache
    )
    adder.memory_budget.total_offset = rem_total_token_offset
    return adder


class TestDllmStagingTokenBudgetFallbackCap(CustomTestCase):
    BLOCK_SIZE = 32

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

    def test_fallback_grants_exactly_one_block_at_the_boundary(self):
        # Exactly one block of dLLM budget: grant it whole, don't round down.
        adder = _make_adder(
            rem_dllm_tokens=self.BLOCK_SIZE,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
            rem_total_token_offset=4196,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(), self.BLOCK_SIZE)

    def test_fallback_page_rounds_an_unaligned_block(self):
        # The full-charge boundary: _update_prefill_budget debits
        # ceil_paged(extend) plus one per_req_token_overhead (one logical page
        # without KV sharding), so with block_size not a page multiple the
        # gate is ceil_paged(24) + 16 == 48 at page_size 16, not 24 + 16. One
        # token below refuses; the boundary grants exactly one block.
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
        # retained KV is stranded. The only case that grants a reuse row at
        # NEGATIVE pool headroom — exactly the state the exemption exists
        # for; the other reuse cases run with pool headroom to spare.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=-100,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(reuses_kv=True), self.BLOCK_SIZE)
        # The same state refuses a fresh-KV row.
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_normal_path_grants_one_full_block(self):
        # Positive rem_total_tokens at or above one block: grant exactly one
        # block (the min already caps at block_size). No cur_rem_tokens gate
        # needed: cur_rem_tokens >= rem_total_tokens by construction. The
        # only case where a genuine positive budget GRANTS through the normal
        # path; every other case reaches it just to refuse.
        adder = _make_adder(
            rem_dllm_tokens=4096,
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=100,
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
        self.assertEqual(adder._get_dllm_remain_tokens(reuses_kv=True), self.BLOCK_SIZE)
        # The same state still rounds a fresh row to zero.
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

    def test_reuse_rows_still_respect_the_dllm_budget(self):
        # The reuse exemption only bypasses the token budgets; the dLLM
        # concurrency budget still applies. A healthy pool makes the refusal
        # attributable to the dLLM budget alone (the exemption ignores the
        # token budgets outright, so exercising more budget regimes here
        # walks the same line).
        adder = _make_adder(
            rem_dllm_tokens=16,  # < block
            dllm_block_size=self.BLOCK_SIZE,
            available_tokens=4096,
        )
        self.assertEqual(adder._get_dllm_remain_tokens(reuses_kv=True), 0)


class _StubReq:
    """Just enough of Req for `add_dllm_staging_req`."""

    def __init__(self, *, rid, fill_len, incomplete_len, req_pool_idx):
        self.rid = rid
        self.full_untruncated_fill_ids = list(range(fill_len))
        self.prefix_len = 0
        self.dllm_incomplete_ids = list(range(incomplete_len))
        # holds_kv mirrors ReqKvInfo's property (req_pool_idx is not None);
        # holds_mamba is irrelevant here (the plain PrefillBudget has no
        # state slots).
        self.kv = SimpleNamespace(
            req_pool_idx=req_pool_idx,
            holds_kv=req_pool_idx is not None,
            holds_mamba=True,
        )
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(max_new_tokens=64)
        self.extend_end = None

    def is_dllm_prefill(self):
        return False

    @property
    def extend_len(self):
        return self.extend_end - self.prefix_len


def _arm_admission(adder: PrefillAdder) -> PrefillAdder:
    """Add the state `add_dllm_staging_req` needs beyond the budget getter."""
    adder.can_run_list = []
    adder.exact_chunk_fill = False
    adder.rem_input_tokens = 1 << 30
    adder.rem_chunk_tokens = None
    adder.log_hit_tokens = 0
    adder.log_input_tokens = 0
    adder.log_replay_tokens = 0
    adder.reprocessed_log_hit_tokens = 0
    adder.reprocessed_log_input_tokens = 0
    return adder


class TestDllmStagingReuseExemptionPredicate(CustomTestCase):
    """`add_dllm_staging_req` must mirror `alloc_for_extend`'s reuse_kv
    predicate: exempt iff kv.holds_kv AND incomplete ids exist."""

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


class TestDllmStagingGateChargeEqualsDebit(CustomTestCase):
    """The fallback gate's block_charge must equal what the admission really
    debits from cur_rem_tokens (`_update_prefill_budget` through the real
    `PrefillBudget.reserve`): admitted at a pool of exactly block_charge, a
    fresh row must leave cur_rem_tokens at exactly 0 — over-charging would
    refuse rows that fit, under-charging admits rows alloc_for_extend
    cannot serve."""

    def test_admission_at_the_gate_boundary_drains_the_pool_exactly(self):
        for block, page in ((32, 1), (32, 16), (24, 16)):
            charge = -(-block // page) * page + page
            adder = _arm_admission(
                _make_adder(
                    rem_dllm_tokens=4096,
                    dllm_block_size=block,
                    available_tokens=charge,
                    rem_total_token_offset=charge + 100,
                    page_size=page,
                )
            )
            req = _StubReq(
                rid="fresh", fill_len=block, incomplete_len=0, req_pool_idx=7
            )
            adder.add_dllm_staging_req(req)
            self.assertEqual(adder.can_run_list, [req], (block, page))
            self.assertEqual(int(adder.cur_rem_tokens), 0, (block, page))


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
    admission check. A post-admission NO_TOKEN (the tail recheck in
    `add_dllm_staging_req`) is the same return value to this loop, so the
    refused-head case covers it. `process_dllm_staging_reqs` never touches
    `self`, so call it unbound."""

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

    def test_all_admitted_returns_continue(self):
        adder = _ScriptedAdder({"a": AddReqResult.CONTINUE, "b": AddReqResult.CONTINUE})
        result = self._run(adder, ["a", "b"])
        self.assertEqual(result, AddReqResult.CONTINUE)

    def test_batch_by_phase_propagates_the_staging_refusal(self):
        # The refusal signal feeds the stall breaker: a refused staging pass
        # must come back as NO_TOKEN from _process_batch_by_phase (and skip
        # the incoming reqs, which would jump ahead of starved staging rows).
        incoming_calls = []
        stub = SimpleNamespace(
            process_dllm_staging_reqs=lambda adder, reqs: AddReqResult.NO_TOKEN,
            process_dllm_incoming_reqs=lambda *a, **k: incoming_calls.append(a),
        )
        staged = SimpleNamespace(dllm_phase=DllmReqPhase.STAGING_DECODE)
        incoming = SimpleNamespace(dllm_phase=DllmReqPhase.INCOMING_DECODE)
        result = SchedulerDllmMixin._process_batch_by_phase(
            stub,
            None,
            [staged, incoming],
            DllmReqPhase.STAGING_DECODE,
            DllmReqPhase.INCOMING_DECODE,
            running_batch=None,
        )
        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(incoming_calls, [])

    def test_batch_by_phase_propagates_an_incoming_refusal(self):
        # Retained rows can also starve incoming rows (the pool they hold
        # refuses every admission while this branch keeps winning the
        # prefill-over-decode priority), so an incoming NO_TOKEN must reach
        # the stall breaker too.
        stub = SimpleNamespace(
            process_dllm_staging_reqs=lambda adder, reqs: AddReqResult.CONTINUE,
            process_dllm_incoming_reqs=(
                lambda adder, reqs, running_batch: AddReqResult.NO_TOKEN
            ),
        )
        staged = SimpleNamespace(dllm_phase=DllmReqPhase.STAGING_DECODE)
        incoming = SimpleNamespace(dllm_phase=DllmReqPhase.INCOMING_DECODE)
        result = SchedulerDllmMixin._process_batch_by_phase(
            stub,
            None,
            [staged, incoming],
            DllmReqPhase.STAGING_DECODE,
            DllmReqPhase.INCOMING_DECODE,
            running_batch=None,
        )
        self.assertEqual(result, AddReqResult.NO_TOKEN)


class _StubStallReq:
    """Just enough of Req for `_maybe_break_dllm_staging_stall`."""

    owner_list = None

    def __init__(self, *, rid, holds_kv, origin_len=64, output_len=0, priority=None):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=holds_kv)
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.priority = priority
        self.dllm_config = SimpleNamespace(
            requires_separate_context_encoding=False, block_size=32
        )
        self.dllm_incomplete_ids = array("q", [7] * 4)
        self.dllm_algo_state = object()
        self.dllm_phase = DllmReqPhase.STAGING_DECODE
        self.reset_calls = 0
        self.to_finish = None

    def reset_for_retract(self):
        # Req.reset_for_retract ends by resetting the dLLM progress.
        self.reset_calls += 1
        ReqDllmMixin.reset_dllm_progress(self)

    def request_finish(self, reason):
        self.to_finish = reason


def _dllm_waiting_queue(reqs):
    queue = IndexedReqList(ListKind.DLLM)
    move(reqs, queue)
    return queue


class _StubRetirer:
    def __init__(self, *, resumable):
        self.resumable = resumable
        self.retracted = []
        self.aborted = []

    def retract(self, req, *, offload_kv):
        self.retracted.append(req)
        return self.resumable

    def abort_retracted(self, req, *, reason):
        # The real retirer takes the request out of the list it is in.
        _leave_owner_list(req)
        self.aborted.append((req, reason))


class TestDllmStagingStallBreaker(CustomTestCase):
    """The capacity gate can livelock a small pool: retained rows that all
    need a fresh block hold every KV slot, every round refuses every row, and
    with nothing admitted nothing ever resolves or frees KV. A zero-admission
    round with a capacity refusal must retract one retained row. The method
    only reads scheduler attributes, so call it unbound on a stub."""

    BLOCK_SIZE = 32

    def setUp(self):
        from sglang.srt.runtime_context import get_context

        # Victim selection reads get_schedule().retraction_policy and the
        # unresumable branch words its abort by disaggregation mode.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _scheduler(reqs, *, resumable=True):
        return SimpleNamespace(
            dllm_manager=SimpleNamespace(waiting_queue=_dllm_waiting_queue(reqs)),
            tree_cache=SimpleNamespace(supports_prefix_sharing=lambda: False),
            request_retirer=_StubRetirer(resumable=resumable),
            dllm_config=SimpleNamespace(
                requires_separate_context_encoding=False, block_size=32
            ),
        )

    @staticmethod
    def _fire(scheduler, *, staging_result=AddReqResult.NO_TOKEN, running_reqs=()):
        SchedulerDllmMixin._maybe_break_dllm_staging_stall(
            scheduler, staging_result, SimpleNamespace(reqs=list(running_reqs))
        )

    def test_stuck_round_retracts_the_newest_retained_row_on_ties(self):
        # Oldest first in the waiting queue; the fresh row holds no KV and is
        # not a candidate. With equal committed work the retraction order is
        # stable, so the tie falls to the newest retained row.
        old = _StubStallReq(rid="old", holds_kv=True)
        new = _StubStallReq(rid="new", holds_kv=True)
        fresh = _StubStallReq(rid="fresh", holds_kv=False)
        scheduler = self._scheduler([old, fresh, new])

        self._fire(scheduler)

        self.assertEqual(scheduler.request_retirer.retracted, [new])
        self.assertEqual(new.reset_calls, 1)
        # The unresolved block cannot survive without its retained KV.
        self.assertEqual(len(new.dllm_incomplete_ids), 0)
        self.assertIsNone(new.dllm_algo_state)
        # The victim rejoins as incoming and stays queued for re-prefill.
        self.assertEqual(new.dllm_phase, DllmReqPhase.INCOMING_PREFILL)
        self.assertIn(new, list(scheduler.dllm_manager.waiting_queue))
        # No one else was touched.
        self.assertEqual(old.reset_calls, 0)
        self.assertEqual(fresh.reset_calls, 0)

    def test_victim_is_the_least_committed_retained_row(self):
        # The breaker reuses ScheduleBatch._get_decode_retraction_order: the
        # default length policy retracts the fewest committed output tokens
        # first (the least work to recompute), not simply the newest row.
        old_bare = _StubStallReq(rid="old-bare", holds_kv=True, output_len=0)
        new_rich = _StubStallReq(rid="new-rich", holds_kv=True, output_len=16)
        scheduler = self._scheduler([old_bare, new_rich])

        self._fire(scheduler)

        self.assertEqual(scheduler.request_retirer.retracted, [old_bare])
        self.assertEqual(new_rich.reset_calls, 0)

    def test_priority_retraction_policy_picks_the_lowest_priority_victim(self):
        from sglang.srt.runtime_context import get_context

        # Under retraction_policy="priority" the shared order ranks by
        # request priority before committed length, so the breaker cannot
        # invert priority scheduling by retracting a high-priority row.
        override = get_context().override_server_args(
            enable_priority_scheduling=True, retraction_policy="priority"
        )
        override.install()
        self.addCleanup(override.restore)

        important = _StubStallReq(
            rid="important", holds_kv=True, output_len=0, priority=10
        )
        expendable = _StubStallReq(
            rid="expendable", holds_kv=True, output_len=16, priority=1
        )
        scheduler = self._scheduler([important, expendable])

        self._fire(scheduler)

        self.assertEqual(scheduler.request_retirer.retracted, [expendable])
        self.assertEqual(important.reset_calls, 0)

    def test_short_prompt_victim_rejoins_as_incoming_decode(self):
        # Mirrors Req.reset_dllm_progress: a committed length shorter than one
        # block restarts in the decode phase on the classic path.
        victim = _StubStallReq(rid="short", holds_kv=True, origin_len=8)
        scheduler = self._scheduler([victim])
        self._fire(scheduler)
        self.assertEqual(victim.dllm_phase, DllmReqPhase.INCOMING_DECODE)

    def test_no_fire_without_a_capacity_refusal(self):
        # A round can admit nothing simply because nothing was staged or the
        # batch was full (OTHER); only a NO_TOKEN refusal means the gate is
        # starving rows.
        retained = _StubStallReq(rid="retained", holds_kv=True)
        for result in (AddReqResult.CONTINUE, AddReqResult.OTHER):
            scheduler = self._scheduler([retained])
            self._fire(scheduler, staging_result=result)
            self.assertEqual(scheduler.request_retirer.retracted, [], result)

    def test_no_fire_while_a_running_batch_is_in_flight(self):
        # In-flight work can still finish and free KV, so the stall is not
        # proven; never retract while progress is possible.
        retained = _StubStallReq(rid="retained", holds_kv=True)
        scheduler = self._scheduler([retained])
        self._fire(scheduler, running_reqs=[object()])
        self.assertEqual(scheduler.request_retirer.retracted, [])

    def test_no_fire_without_a_retained_row(self):
        # Nothing to free: the refusal is not caused by retained staging KV.
        scheduler = self._scheduler([_StubStallReq(rid="fresh", holds_kv=False)])
        self._fire(scheduler)
        self.assertEqual(scheduler.request_retirer.retracted, [])
        self.assertEqual(scheduler.request_retirer.aborted, [])

    def test_unresumable_victim_is_aborted_and_dequeued(self):
        victim = _StubStallReq(rid="unresumable", holds_kv=True)
        scheduler = self._scheduler([victim], resumable=False)

        self._fire(scheduler)

        self.assertIsNotNone(victim.to_finish)
        self.assertEqual(
            scheduler.request_retirer.aborted, [(victim, victim.to_finish)]
        )
        self.assertNotIn(victim, list(scheduler.dllm_manager.waiting_queue))

    def test_retraction_funds_the_next_rounds_admission(self):
        # The loop the breaker exists to close, at unit scope: the stalled
        # round's gate refuses a fresh block at zero pool headroom; the
        # breaker retracts the retained row; with that row's tokens back in
        # the pool the same gate grants, so the next round can admit.
        block_charge = self.BLOCK_SIZE + 1  # ceil_paged(block) + overhead, page 1
        pool = {"free": 0}

        def gate():
            adder = _make_adder(
                rem_dllm_tokens=4096,
                dllm_block_size=self.BLOCK_SIZE,
                available_tokens=pool["free"],
                rem_total_token_offset=pool["free"] + 100,
            )
            return adder._get_dllm_remain_tokens()

        self.assertEqual(gate(), 0)

        victim = _StubStallReq(rid="held", holds_kv=True)
        scheduler = self._scheduler([victim])
        retirer = scheduler.request_retirer
        real_retract = retirer.retract

        def retract_frees_the_row(req, *, offload_kv):
            pool["free"] += block_charge
            return real_retract(req, offload_kv=offload_kv)

        retirer.retract = retract_frees_the_row
        self._fire(scheduler)

        self.assertEqual(retirer.retracted, [victim])
        self.assertEqual(gate(), self.BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()
