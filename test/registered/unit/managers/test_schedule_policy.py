import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    CacheAwarePolicy,
    PrefillAdder,
    SchedulePolicy,
    _PrefillAdmission,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicyHRRN(CustomTestCase):
    def test_calc_priority_hrrn(self):
        """HRRN sorts by response ratio (waited_tokens / uncached_tokens).

        With three fresh reqs (waited_tokens = 0 for all), the ratio is 0 / uncached for each,
        and stable sort keeps original order among equal keys -- effectively pure SUF via the
        rid tie-breaker is avoided, so we set a non-zero arrival snapshot on some reqs to exercise
        the aging half of the formula.
        """
        tree_cache = RadixCache.create_simulated()

        # r_short: small uncached, just arrived (waited=0)
        # r_long:  large uncached, just arrived (waited=0)
        # r_aged:  medium uncached, arrived long ago (waited>>0)
        r_short = _make_req("short", "a", [1])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])

        # Fresh reqs arrived with the counter still at 0.
        r_short.arrival_processed_tokens = 0
        r_long.arrival_processed_tokens = 0
        r_aged.arrival_processed_tokens = 0

        waiting_queue = [r_long, r_aged, r_short]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=1000 -> waited_tokens is 1000 for every req here.
        # Ratios: short = 1000/1 = 1000; aged = 1000/3 ~= 333; long = 1000/10 = 100.
        # Highest ratio first -> short, aged, long.
        waiting_queue = policy.calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "short")
        self.assertEqual(waiting_queue[1].rid, "aged")
        self.assertEqual(waiting_queue[2].rid, "long")

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that has waited enough should overtake a
        just-arrived short request."""
        tree_cache = RadixCache.create_simulated()

        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])

        # long_old arrived at counter=0 and has been waiting; short_new
        # just arrived (its snapshot equals the current counter).
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000

        waiting_queue = [r_short_new, r_long_old]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=100000 -> long_old.waited = 100000, ratio = 100000/100 = 1000.
        #                            short_new.waited = 0, ratio = 0.
        # long_old should now be first.
        waiting_queue = policy.calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual(waiting_queue[0].rid, "long_old")
        self.assertEqual(waiting_queue[1].rid, "short_new")

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cached prefix length shortens uncached, so reqs with the same input
        length can sort differently by HRRN. Also verifies rid tie-break for
        equal-ratio reqs.

        Uses _sort_by_hrrn directly to bypass the prefix-match pass inside
        calc_priority, which would overwrite num_matched_prefix_tokens.
        """
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))

        # r_more_cached: 90 cached -> uncached = 10 -> ratio = 1000 / 10 = 100.
        # r_less_cached:  0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10.
        # r_tie:          0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10 (ties with r_less_cached; rid "b" < "c").
        r_more_cached.num_matched_prefix_tokens = 90
        r_less_cached.num_matched_prefix_tokens = 0
        r_tie.num_matched_prefix_tokens = 0
        r_more_cached.arrival_processed_tokens = 0
        r_less_cached.arrival_processed_tokens = 0
        r_tie.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "a")
        self.assertEqual(waiting_queue[1].rid, "b")
        self.assertEqual(waiting_queue[2].rid, "c")


class TestShortestPrefillFirst(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_len = cached
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(self.policy, "_compute_prefix_matches", return_value=set()):
            queue = self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value={duplicate.rid}
        ):
            queue = self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 3840)], 4096, 256
            ),
            256,
        )

    def test_no_reservation_when_request_cannot_fit_or_is_not_shorter(self):
        continuation = self.make_req("continuation", 8192)
        for waiting, budget in [
            ([], 4096),
            ([self.make_req("same", 8192)], 4096),
            ([self.make_req("too-large", 4096)], 4096),
            ([self.make_req("short", 1)], 256),
        ]:
            with self.subTest(budget=budget, waiting=[req.rid for req in waiting]):
                self.assertIsNone(
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256
                    )
                )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy.policy = CacheAwarePolicy.HRRN
        self.assertIsNone(
            self.policy.shortest_prefill_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
            )
        )


class TestHiCacheLoadBackPromise(CustomTestCase):
    """The load-back integrity check must compare against what the load
    actually promised to copy, not the whole host hit.

    Regression for #50892: on a joint-window (hybrid-SWA) model, a host hit
    can include FULL rows that are still device-resident (only blocked by a
    host-only window/state entry). Those rows need no H2D, so a legitimate
    load-back lands fewer rows than ``req.host_hit_length`` — and the old
    check aborted the whole scheduler over exactly that difference."""

    # (host_hit, promised H2D rows, rows the cache reports as newly landed)
    def _run(self, *, joint_window, host_hit, transfer_rows, loaded_rows):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.prefill_max_requests = None
        adder.can_run_list = []
        adder.tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        adder.tree_cache.buffer_pipeline = None
        adder.tree_cache.linker = None
        adder.tree_cache.tree_core = SimpleNamespace(
            build_hicache_transfers=Mock(
                return_value=[SimpleNamespace(host_indices=list(range(transfer_rows)))]
            )
        )
        adder.tree_cache.init_load_back = Mock(return_value=(loaded_rows, 7))
        adder.tree_cache.load_back_is_cache_owned = Mock(return_value=False)
        adder.memory_budget = SimpleNamespace(
            load_back_needs_full_tokens=joint_window,
            state_slots_fit=lambda _: True,
            prepare_load_back=lambda **_: True,
            load_back_scope=lambda **_: nullcontext(),
        )
        adder.prefill_delayer_single_pass = None
        adder.kv_shard_scratch_spec = None
        adder.per_req_token_overhead = 0
        adder.dllm_config = None
        adder.page_size = 1
        adder.rem_chunk_tokens = None

        req = Mock(spec=Req)
        req.rid = "partial-load-back"
        req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
        req.output_ids = []
        req.full_untruncated_fill_ids = list(range(64))
        req.prefix_len = 0
        req.host_hit_length = host_hit
        req.window_host_hit_lengths = {}
        req.last_node = 3
        req.best_match_node = 7
        req.needs_host_load_back.return_value = True
        req.finished.return_value = False

        admission = _PrefillAdmission(
            prefix_len=0,
            extend_len=64,
            max_new_tokens=8,
            is_chunked=False,
            replay_len=0,
        )
        select_admission = Mock(return_value=admission)
        commit = Mock()
        with (
            patch.object(adder, "_state_slots_for_req", return_value=0),
            patch.object(adder, "_request_slot_reserve_for_req", return_value=0),
            patch.object(adder, "_lock_node", lambda *_, **__: nullcontext()),
            patch.object(adder, "_select_prefill_admission", select_admission),
            patch.object(adder, "_kv_shard_reserve_scratch", return_value=True),
            patch.object(adder, "_commit_prefill_admission", commit),
            patch.object(adder, "budget_state", return_value=AddReqResult.CONTINUE),
        ):
            verdict = adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )
        return verdict, req, select_admission

    def test_partial_load_back_with_resident_full_rows_is_not_an_abort(self):
        """Joint window: half the host hit's FULL rows are already on the
        device. The transfer spec (and the reserved budget) covers only the
        missing rows, and the load lands exactly those."""
        verdict, req, _ = self._run(
            joint_window=True, host_hit=64, transfer_rows=32, loaded_rows=32
        )
        self.assertIs(verdict, AddReqResult.CONTINUE)
        self.assertEqual(req.host_loaded_length, 32)

    def test_incomplete_load_back_still_aborts(self):
        """A load that lands fewer rows than its transfer spec promised is a
        genuine partial commit and must keep aborting."""
        with self.assertRaises(RuntimeError) as ctx:
            self._run(joint_window=True, host_hit=64, transfer_rows=32, loaded_rows=16)
        self.assertIn("must commit all promised FULL tokens", str(ctx.exception))
        self.assertIn("promised=32", str(ctx.exception))

    def test_non_hybrid_promise_stays_the_whole_host_hit(self):
        """Without the joint-window branch the promise remains the whole host
        hit, so a short landing still aborts (original protection)."""
        with self.assertRaises(RuntimeError) as ctx:
            self._run(joint_window=False, host_hit=64, transfer_rows=64, loaded_rows=32)
        self.assertIn("promised=64", str(ctx.exception))

    def test_fully_resident_hit_needs_no_admission_recompute(self):
        """Nothing to copy because the whole hit is device-resident: the load
        lands zero rows without any promised transfer, so the request keeps
        its host hit instead of being recomputed from scratch."""
        verdict, req, select_admission = self._run(
            joint_window=True, host_hit=64, transfer_rows=0, loaded_rows=0
        )
        self.assertIs(verdict, AddReqResult.CONTINUE)
        # Only the initial admission selection ran; the elif recompute branch
        # (host_hit_length=0) was for "promised rows landed none".
        self.assertEqual(select_admission.call_count, 1)


if __name__ == "__main__":
    unittest.main()
