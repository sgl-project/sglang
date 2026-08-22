"""Unit tests for the HiCache HiSparse backing's admission accounting.

The coordinator hands the radix tree the KV of a request that is still decoding,
so admission has to be rationed before the prefill forward runs -- and the
rationing is what decides whether the two tiers can keep one home for every
admitted token. The arithmetic is small and entirely integer, which is why it
lives apart from the coordinator and is tested here without a GPU.

What the cases below pin down is the reasoning, not the formulas:

- a candidate predicted to run standard must not be charged the device buffer it
  will not take, and one that cannot fit the pool at all must be budgeted as
  standard even if every quota is free (otherwise the adder reserves more than
  rem_total_tokens forever and the system livelocks on an idle pool);
- "only a quota blocks it" and "it gains nothing from admission" are different
  answers with opposite scheduler actions -- queue versus run now;
- shared prefixes bill once, and an in-flight candidate bills its full footprint
  until admission resolves, which is the pacing margin that keeps write-back
  able to drain.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.hisparse_hicache_admission import AdmissionLedger
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE_SIZE = 64
TEMP_SLOTS = 4096
CHUNK_TOKENS = 4096
# Big enough that _infeasible never fires unless a case asks for it.
DEVICE_POOL = 1_000_000
HOST_TOKENS = 500_000


def _node(node_id: int, tokens: int, *, on_host: bool = False):
    """A tree node stub with just the fields the ledger reads."""
    values = list(range(tokens))
    data = SimpleNamespace(
        value=None if on_host else values,
        host_value=values if on_host else None,
    )
    return SimpleNamespace(id=node_id, component_data={ComponentType.FULL: data})


def _ledger(
    *,
    device_pool_tokens=DEVICE_POOL,
    host_tokens=HOST_TOKENS,
    chunk_tokens=CHUNK_TOKENS,
):
    ledger = AdmissionLedger(
        device_pool_tokens=device_pool_tokens,
        temp_slot_tokens=TEMP_SLOTS,
        page_size=PAGE_SIZE,
        chunk_tokens=chunk_tokens,
    )
    ledger.set_host_capacity(host_tokens)
    return ledger


def _budget(ledger, *, expanded_pages_left=10_000, tree_evictable_tokens=0):
    return ledger.make_budget(
        expanded_pages_left=expanded_pages_left,
        tree_evictable_tokens=tree_evictable_tokens,
    )


def _req(rid="r0"):
    return SimpleNamespace(rid=rid)


class TestFutureTokenReservation(CustomTestCase):
    def test_admissible_candidate_reserves_the_device_buffer(self):
        budget = _budget(_ledger())
        reserved = budget.future_tokens(30_000, 512, commit=False)
        # The temp buffer comes out of the regular pool and is held for the
        # request's lifetime, so the adder has to see it now.
        self.assertEqual(reserved, TEMP_SLOTS + 512)

    def test_short_prefix_is_budgeted_as_standard(self):
        # Below the temp buffer, admission would be a net capacity LOSS, so the
        # coordinator declines -- and charging the buffer here would reserve
        # device tokens the request never takes.
        budget = _budget(_ledger())
        self.assertEqual(budget.future_tokens(TEMP_SLOTS - 1, 512, commit=False), 512)

    def test_sub_page_prompt_is_budgeted_as_standard(self):
        budget = _budget(_ledger())
        self.assertEqual(budget.future_tokens(PAGE_SIZE - 1, 512, commit=False), 512)

    def test_pool_infeasible_candidate_is_budgeted_as_standard(self):
        """A reservation larger than the whole pool must not be made.

        The entry-side max_new clamp only guarantees the STANDARD budget fits, so
        adding the temp buffer on top can push the adder's total past
        rem_total_tokens on an *idle* system -- which never clears, because
        nothing is running that could free anything.
        """
        budget = _budget(_ledger(device_pool_tokens=40_000))
        self.assertEqual(budget.future_tokens(30_000, 8_000, commit=False), 8_000)

    def test_commit_depletes_the_round_for_later_candidates(self):
        # One page of expanded region left: the second candidate in the same
        # round must see it gone, or the adder over-promises within one pass.
        budget = _budget(_ledger(), expanded_pages_left=TEMP_SLOTS // PAGE_SIZE)
        first = budget.future_tokens(TEMP_SLOTS, 512, commit=True, req=_req("a"))
        second = budget.future_tokens(TEMP_SLOTS, 512, commit=False, req=_req("b"))
        self.assertEqual(first, TEMP_SLOTS + 512)
        self.assertEqual(second, 512)

    def test_probe_does_not_deplete(self):
        budget = _budget(_ledger(), expanded_pages_left=TEMP_SLOTS // PAGE_SIZE)
        budget.future_tokens(TEMP_SLOTS, 512, commit=False, req=_req("a"))
        self.assertEqual(
            budget.future_tokens(TEMP_SLOTS, 512, commit=False, req=_req("b")),
            TEMP_SLOTS + 512,
        )

    def test_commit_publishes_the_claim_to_the_next_round(self):
        """Without this, a round snapshots headroom an earlier round promised.

        The coordinator only records the real reservation after the prefill
        forward, so the pending claim is the only thing standing between two
        rounds and a double promise.
        """
        ledger = _ledger(host_tokens=TEMP_SLOTS * 2)
        before = ledger.reservable_left()
        _budget(ledger).future_tokens(TEMP_SLOTS, 512, commit=True, req=_req("a"))
        self.assertEqual(ledger.reservable_left(), before - TEMP_SLOTS)


class TestAdmissionExhausted(CustomTestCase):
    """Queue the candidate, or let it run standard? Opposite actions.

    A candidate blocked only by a spent quota should wait: the quota frees when an
    admitted request finishes. One that gains nothing from admission should run
    now -- reporting it as exhausted would keep it queued behind a condition that
    never changes.
    """

    def test_exhausted_when_only_the_expanded_region_is_spent(self):
        budget = _budget(_ledger(), expanded_pages_left=0)
        self.assertTrue(budget.admission_exhausted(30_000, 512))

    def test_exhausted_when_the_two_tiers_are_already_promised(self):
        # Both tiers count toward one ceiling (a prefix token needs one home), so
        # what spends it is what admitted requests already hold -- here one huge
        # claimed node.
        ledger = _ledger()
        ledger.claim_node(_node(1, ledger.reservable_left() - 1_000))
        self.assertTrue(_budget(ledger).admission_exhausted(30_000, 512))

    def test_not_exhausted_for_a_short_prefix(self):
        budget = _budget(_ledger(), expanded_pages_left=0)
        self.assertFalse(budget.admission_exhausted(TEMP_SLOTS - 1, 512))

    def test_not_exhausted_for_a_pool_infeasible_candidate(self):
        budget = _budget(_ledger(device_pool_tokens=40_000), expanded_pages_left=0)
        self.assertFalse(budget.admission_exhausted(30_000, 8_000))

    def test_not_exhausted_when_quotas_are_free(self):
        self.assertFalse(_budget(_ledger()).admission_exhausted(30_000, 512))

    def test_agrees_with_future_tokens(self):
        """The gate and the reservation must read the same quotas.

        Disagreement is a livelock either way: queueing candidates future_tokens
        would admit, or admitting candidates the gate wants queued.
        """
        for expanded, promised in ((0, 0), (10_000, HOST_TOKENS), (10_000, 0)):
            with self.subTest(expanded=expanded, promised=promised):
                ledger = _ledger()
                if promised:
                    ledger.claim_node(_node(1, promised))
                budget = _budget(ledger, expanded_pages_left=expanded)
                admitted = budget.future_tokens(30_000, 512, commit=False) != 512
                self.assertEqual(admitted, not budget.admission_exhausted(30_000, 512))


class TestPendingClaims(CustomTestCase):
    def test_pending_is_idempotent_per_request(self):
        # Chunked prefill re-budgets the same request in every round it spans; a
        # second charge would shrink headroom for a request already counted.
        ledger = _ledger()
        before = ledger.reservable_left()
        ledger.note_pending("a", 10_000)
        ledger.note_pending("a", 10_000)
        self.assertEqual(ledger.reservable_left(), before - 10_000)

    def test_dropping_an_unknown_request_is_a_no_op(self):
        # Every finish path drops, including requests that were never budgeted.
        ledger = _ledger()
        before = ledger.reservable_left()
        ledger.drop_pending("never-seen")
        self.assertEqual(ledger.reservable_left(), before)

    def test_drop_releases_exactly_once(self):
        ledger = _ledger()
        before = ledger.reservable_left()
        ledger.note_pending("a", 10_000)
        ledger.drop_pending("a")
        ledger.drop_pending("a")
        self.assertEqual(ledger.reservable_left(), before)

    def test_admission_supersedes_the_in_flight_charge(self):
        # The pending charge and the per-node claims stand for the SAME tokens,
        # so an admitted request must carry one or the other, never both. It did
        # carry both: with the real 60K-pool / 30.8K-prompt geometry that made
        # the ceiling run out one request early -- the third candidate was
        # reported as admission_exhausted (29552 left of the 30784 it needed),
        # which stops the adder outright instead of falling back to standard, and
        # per-rank concurrency sat at two with six requests queued.
        tree_len = 30_784
        ledger = _ledger(device_pool_tokens=59_968, host_tokens=120_000)
        for idx, rid in enumerate(("a", "b")):
            ledger.note_pending(rid, tree_len)
            ledger.claim_node(_node(idx, tree_len))
            ledger.activate(idx, tree_len, rid=rid, decode_reserve=512)
        self.assertFalse(_budget(ledger).admission_exhausted(tree_len, 512))
        device_reserve = (
            3 * TEMP_SLOTS  # two admitted plus the next admission
            + 2 * 512  # their output tails
            + CHUNK_TOKENS  # whatever is being prefilled
            + 3 * PAGE_SIZE  # alignment
        )
        self.assertEqual(
            ledger.reservable_left(),
            120_000 + (59_968 - device_reserve) - 2 * tree_len,
        )


class TestNodeClaims(CustomTestCase):
    def test_a_shared_node_bills_once(self):
        # Two requests on the same prefix: the data needs one home, not two.
        ledger = _ledger()
        before = ledger.reservable_left()
        node = _node(1, 1_000)
        ledger.claim_node(node)
        ledger.claim_node(node)
        self.assertEqual(ledger.reservable_left(), before - 1_000)

    def test_the_last_release_gives_the_tokens_back(self):
        ledger = _ledger()
        before = ledger.reservable_left()
        node = _node(1, 1_000)
        ledger.claim_node(node)
        ledger.claim_node(node)
        ledger.release_node(node)
        self.assertEqual(ledger.reservable_left(), before - 1_000)
        ledger.release_node(node)
        self.assertEqual(ledger.reservable_left(), before)

    def test_a_host_only_node_is_measured_by_its_host_copy(self):
        # A node the tree already demoted has no device value; billing it zero
        # would let a re-hit prefix be admitted for free -- exactly the pattern
        # that saturated host with locks until backups failed.
        ledger = _ledger()
        before = ledger.reservable_left()
        ledger.claim_node(_node(1, 1_000, on_host=True))
        self.assertEqual(ledger.reservable_left(), before - 1_000)

    def test_reservable_is_unbounded_without_a_host_tier(self):
        ledger = AdmissionLedger(
            device_pool_tokens=DEVICE_POOL,
            temp_slot_tokens=TEMP_SLOTS,
            page_size=PAGE_SIZE,
            chunk_tokens=CHUNK_TOKENS,
        )
        self.assertGreater(ledger.reservable_left(), 1 << 50)


class TestDeviceReserve(CustomTestCase):
    """What the device tier withholds from the ceiling, and what it scales with.

    This used to be a flat 25% of the pool. The fraction was the wrong shape, not
    just the wrong value: everything it stood in for scales with concurrency,
    chunk size and output length, and none of those scale with the pool. Both
    cases below are red under any pool-proportional reserve.
    """

    def test_a_bigger_pool_is_usable_in_full(self):
        # Same concurrency, same chunk, same outputs -> the same reserve, so the
        # extra pool is available down to the last token. A 25% fraction would
        # keep back a quarter of the increase.
        base = _ledger(device_pool_tokens=60_000)
        big = _ledger(device_pool_tokens=120_000)
        self.assertEqual(big.reservable_left() - base.reservable_left(), 60_000)

    def test_a_long_output_request_withholds_its_own_tail(self):
        # Output tokens are device-only until write-back takes them, so a request
        # asking for 8k of output withholds 8k -- where a pool fraction withholds
        # the same amount whether the request emits one token or ten thousand.
        short, long = _ledger(), _ledger()
        short.activate(1, 30_000, rid="r1", decode_reserve=0)
        long.activate(1, 30_000, rid="r1", decode_reserve=8_192)
        self.assertEqual(short.reservable_left() - long.reservable_left(), 8_192)


class TestDeviceEvictableOverhang(CustomTestCase):
    """Device tokens counted evictable that eviction cannot actually reclaim.

    Under write_back a demotion needs host space, and the coordinator vetoes the
    copy-less drop for data backing a live request, so those tokens stay on
    device. Budgeting against them over-admits until prefill allocation
    hard-fails, which is not a recoverable condition.
    """

    def test_zero_without_active_requests(self):
        # The veto only protects live requests; every other node is droppable.
        ledger = _ledger()
        self.assertEqual(ledger.device_evictable_overhang(1_000_000), 0)

    def test_zero_while_host_can_absorb_the_backing_set(self):
        ledger = _ledger()
        ledger.activate(1, 30_000, rid="r1", decode_reserve=0)
        self.assertEqual(ledger.device_evictable_overhang(30_000), 0)

    def test_positive_once_host_cannot_absorb(self):
        ledger = _ledger(host_tokens=20_000)
        ledger.activate(1, 100_000, rid="r1", decode_reserve=0)
        # 100k backed by 20k of host, minus the concurrency slack.
        self.assertGreater(ledger.device_evictable_overhang(100_000), 0)

    def test_clamped_by_the_tree_evictable_size(self):
        # Shared prefixes double-count in the per-request sum, so the tree's own
        # number is the ceiling; without the clamp the adder throttles on tokens
        # the tree never held.
        ledger = _ledger(host_tokens=1)
        ledger.activate(1, 100_000, rid="r1", decode_reserve=0)
        ledger.activate(2, 100_000, rid="r2", decode_reserve=0)
        self.assertLessEqual(ledger.device_evictable_overhang(50_000), 50_000)

    def test_evicted_positions_shrink_the_blocked_set(self):
        # A position already on host is no longer blocking device eviction.
        ledger = _ledger(host_tokens=20_000)
        ledger.activate(1, 100_000, rid="r1", decode_reserve=0)
        blocked = ledger.device_evictable_overhang(100_000)
        ledger.note_evicted_positions(1, 50_000)
        self.assertLess(ledger.device_evictable_overhang(100_000), blocked)

    def test_deactivate_releases_the_whole_record(self):
        # Everything a request holds goes back in one step. While these were
        # parallel dicts every exit path had to pop each one by hand, and a
        # forgotten pop is a permanent quota leak that surfaces only as
        # unexplained admission pressure, nowhere near the path that leaked it.
        ledger = _ledger()
        before = ledger.reservable_left()
        ledger.activate(1, 30_000, rid="r1", decode_reserve=8_192)
        ledger.note_evicted_positions(1, 1_000)
        ledger.deactivate(1)
        self.assertEqual(ledger.reservable_left(), before)
        self.assertEqual(ledger.device_evictable_overhang(100_000), 0)

    def test_deactivate_clears_the_request(self):
        ledger = _ledger(host_tokens=1)
        ledger.activate(1, 100_000, rid="r1", decode_reserve=0)
        ledger.deactivate(1)
        self.assertEqual(ledger.device_evictable_overhang(100_000), 0)


if __name__ == "__main__":
    unittest.main()
