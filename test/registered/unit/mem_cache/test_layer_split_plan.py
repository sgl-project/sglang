"""Shared transfer-plan invariants, independent of Prefetch/backup direction."""

import dataclasses
import unittest

from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.srt.mem_cache.layer_split.layer_split_config import (
    StagingBufferConfig,
)
from sglang.srt.mem_cache.layer_split.layer_split_plan import (
    EXCHANGE_COMPONENTS,
    PageTransferPlan,
    PageWindow,
    build_exchange_rounds,
    build_transfer_plan,
    page_owner,
    rotation_base,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


CustomTestCase = unittest.TestCase

PAGE_SIZE = 64
SHARD_SIZE = 8
# One 8K window at page size 64.
PAGES_PER_WINDOW = 128
STAGING_BUFFER_CONFIG = StagingBufferConfig().with_host_layout(
    page_size=PAGE_SIZE, shard_size=SHARD_SIZE
)
_EMPTY_WINDOW = PageWindow(index=0, page_start=0, page_count=0)


def make_hashes(count: int):
    return [f"page{i:04d}" for i in range(count)]


def make_plan(
    page_count,
    *,
    op_id=1,
    window_base=0,
    request_id="req",
    shard_size=SHARD_SIZE,
):
    return build_transfer_plan(
        op_id=op_id,
        request_id=request_id,
        page_hashes=make_hashes(page_count),
        staging_buffer_config=STAGING_BUFFER_CONFIG.with_host_layout(
            page_size=PAGE_SIZE, shard_size=shard_size
        ),
        window_base=window_base,
    )


class TestStagingWindowResolution(CustomTestCase):
    def test_one_window_capacity_is_shared_by_both_directions(self):
        self.assertEqual(STAGING_BUFFER_CONFIG.pages_per_window, PAGES_PER_WINDOW)
        self.assertEqual(STAGING_BUFFER_CONFIG.pages_per_rank_per_window, 16)

    def test_uneven_owners_round_up_the_fixed_capacity(self):
        config = StagingBufferConfig(window_size=640, page_size=64, shard_size=3)
        self.assertEqual(config.pages_per_window, 10)
        self.assertEqual(config.pages_per_rank_per_window, 4)

    def test_window_must_be_page_aligned(self):
        with self.assertRaises(ValueError):
            StagingBufferConfig(window_size=100, page_size=64, shard_size=3)


class TestPageOwner(CustomTestCase):
    def test_owner_is_pure_and_rotates_with_window_base(self):
        self.assertEqual(
            [page_owner(i, 0, SHARD_SIZE) for i in range(8)], list(range(8))
        )
        # A different base shifts every assignment by the same amount.
        self.assertEqual([page_owner(i, 3, SHARD_SIZE) for i in range(4)], [3, 4, 5, 6])

    def test_owner_wraps_at_shard_size(self):
        self.assertEqual(page_owner(SHARD_SIZE, 0, SHARD_SIZE), 0)
        self.assertEqual(page_owner(SHARD_SIZE + 1, 0, SHARD_SIZE), 1)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            page_owner(0, 0, 0)
        with self.assertRaises(ValueError):
            page_owner(-1, 0, SHARD_SIZE)


class TestPlanConstruction(CustomTestCase):
    def test_preserves_the_entire_admitted_page_list(self):
        plan = make_plan(40)
        self.assertEqual(plan.page_count, 40)
        self.assertEqual(plan.page_hashes, tuple(make_hashes(40)))

    def test_empty_page_list_yields_empty_plan(self):
        plan = make_plan(0)
        self.assertEqual(plan.windows(), [])
        self.assertEqual(plan.owner_page_counts(), [0] * SHARD_SIZE)

    def test_page_list_is_an_immutable_snapshot(self):
        hashes = make_hashes(3)
        plan = build_transfer_plan(
            op_id=1,
            request_id="r",
            page_hashes=hashes,
            staging_buffer_config=STAGING_BUFFER_CONFIG,
        )
        hashes[0] = "changed"
        self.assertEqual(plan.page_hashes, tuple(make_hashes(3)))

    def test_window_base_defaults_to_the_request_rotation(self):
        plan = build_transfer_plan(
            op_id=5,
            request_id="req",
            page_hashes=make_hashes(8),
            staging_buffer_config=STAGING_BUFFER_CONFIG,
        )
        self.assertEqual(plan.window_base, rotation_base("req"))
        self.assertEqual(plan.page_owners[0], rotation_base("req") % SHARD_SIZE)

    def test_empty_plan_still_derives_a_base(self):
        plan = build_transfer_plan(
            op_id=1,
            request_id="req",
            page_hashes=[],
            staging_buffer_config=STAGING_BUFFER_CONFIG,
        )
        self.assertEqual(plan.page_owners, ())
        self.assertEqual(plan.window_base, rotation_base("req"))

    def test_plan_rejects_mismatched_owner_length(self):
        with self.assertRaises(ValueError):
            PageTransferPlan(
                op_id=1,
                request_id="req",
                page_hashes=("a", "b"),
                page_owners=(0,),
                shard_size=SHARD_SIZE,
                pages_per_window=PAGES_PER_WINDOW,
                window_base=0,
            )


class TestPlanDeterminism(CustomTestCase):
    def _plan(self, *, op_id, request_id="req-42", page_count=300):
        # No window_base, so this exercises the production default.
        return build_transfer_plan(
            op_id=op_id,
            request_id=request_id,
            page_hashes=make_hashes(page_count),
            staging_buffer_config=STAGING_BUFFER_CONFIG,
        )

    # Fields every rank must agree on for the owner table to stay unbroadcast.
    # op_id is excluded on purpose: it is each rank's own submission counter.
    AGREED_FIELDS = (
        "page_hashes",
        "page_owners",
        "window_base",
        "shard_size",
        "pages_per_window",
    )

    def test_every_rank_derives_the_identical_ownership(self):
        # Each rank runs its own counter, so the realistic case is that they
        # already disagree on op_id -- including wildly, after one rank skipped
        # an operation its peers did not. A divergent owner table would mean
        # pages fetched twice or not at all, and exchange split sizes that stop
        # matching, which hangs the collective rather than failing cleanly.
        plans = [self._plan(op_id=op_id) for op_id in (1, 2, 3, 4, 5, 6, 7, 999999)]
        first = plans[0]
        for other in plans[1:]:
            for field in self.AGREED_FIELDS:
                self.assertEqual(
                    getattr(first, field), getattr(other, field), msg=field
                )

    def test_op_id_is_the_only_field_allowed_to_differ(self):
        # Guards the class docstring's exception: if a second rank-local field
        # ever creeps into the plan, this fails and forces the question.
        a, b = self._plan(op_id=1), self._plan(op_id=2)
        differing = {
            f.name
            for f in dataclasses.fields(a)
            if getattr(a, f.name) != getattr(b, f.name)
        }
        self.assertEqual(differing, {"op_id"})

    def test_rotation_still_varies_between_requests(self):
        # Rank-invariance must not cost the rotation that keeps a partial
        # window's uneven tail off the same ranks on every request.
        tail_owners = {
            self._plan(op_id=1, request_id=f"req-{i}", page_count=3).page_owners[0]
            for i in range(200)
        }
        self.assertGreater(len(tail_owners), 1)

    def test_rotation_base_is_stable_across_processes(self):
        # Pinned to constants on purpose. The builtin hash() is salted per
        # process, so swapping crc32 for it would silently break agreement
        # between the eight workers while every single-process test still passed.
        self.assertEqual(rotation_base("req-x"), 1645208069)
        self.assertEqual(rotation_base(""), 0)


class TestWindowSplit(CustomTestCase):
    def test_full_windows_are_exactly_page_aligned(self):
        plan = make_plan(2 * PAGES_PER_WINDOW)
        windows = plan.windows()
        self.assertEqual([w.page_count for w in windows], [PAGES_PER_WINDOW] * 2)
        self.assertEqual([w.page_start for w in windows], [0, PAGES_PER_WINDOW])
        self.assertEqual(windows[-1].page_end, plan.page_count)

    def test_trailing_partial_window_is_kept(self):
        plan = make_plan(PAGES_PER_WINDOW + 30)
        windows = plan.windows()
        self.assertEqual([w.page_count for w in windows], [PAGES_PER_WINDOW, 30])

    def test_windows_tile_pages_without_gap_or_overlap(self):
        plan = make_plan(PAGES_PER_WINDOW * 3 + 7)
        covered = [i for w in plan.windows() for i in w.ordinals()]
        self.assertEqual(covered, list(range(plan.page_count)))


class TestOwnership(CustomTestCase):
    def test_full_window_is_perfectly_balanced(self):
        # 128 pages over 8 ranks: every rank owns exactly 16, which is what the
        # per-rank staging sizing assumes.
        plan = make_plan(PAGES_PER_WINDOW)
        self.assertEqual(
            plan.owner_page_counts(), [PAGES_PER_WINDOW // SHARD_SIZE] * SHARD_SIZE
        )

    def test_every_page_has_exactly_one_owner(self):
        plan = make_plan(PAGES_PER_WINDOW + 5)
        owned = [
            i
            for window in plan.windows()
            for rank in range(SHARD_SIZE)
            for i in plan.owned_ordinals(rank, window)
        ]
        self.assertEqual(sorted(owned), list(range(plan.page_count)))

    def test_partial_window_imbalance_is_at_most_one_page(self):
        plan = make_plan(30)
        counts = plan.owner_page_counts()
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 30)

    def test_owned_ordinals_can_be_scoped_to_a_window(self):
        plan = make_plan(2 * PAGES_PER_WINDOW)
        second = plan.windows()[1]
        scoped = plan.owned_ordinals(0, second)
        self.assertTrue(all(i in second.ordinals() for i in scoped))
        self.assertEqual(len(scoped), PAGES_PER_WINDOW // SHARD_SIZE)
        self.assertEqual(
            scoped,
            [i for i in second.ordinals() if plan.page_owners[i] == 0],
        )

    def test_owned_ordinals_are_ascending(self):
        plan = make_plan(PAGES_PER_WINDOW + 5)
        for window in plan.windows():
            for rank in range(SHARD_SIZE):
                ordinals = plan.owned_ordinals(rank, window)
                self.assertEqual(ordinals, sorted(ordinals))

    def test_rotation_moves_the_uneven_tail_between_ranks(self):
        # The tail of a partial window must not always burden the same ranks.
        tail_owner_zero = make_plan(1, window_base=0).page_owners[0]
        tail_owner_one = make_plan(1, window_base=1).page_owners[0]
        self.assertNotEqual(tail_owner_zero, tail_owner_one)


class TestConsistencyWithLayerShard(CustomTestCase):
    def test_owner_layer_ranges_tile_all_layers(self):
        # The plan decides *which rank fetches a page*; the layer shard helper
        # decides *which layers each rank keeps*. Both must agree that the 78
        # GLM-5.2 layers are covered exactly once, or the exchange would drop or
        # duplicate a layer slice.
        layer_num = 78
        ranges = [
            get_layer_shard_range(r, SHARD_SIZE, layer_num) for r in range(SHARD_SIZE)
        ]
        covered = [layer for start, end in ranges for layer in range(start, end)]
        self.assertEqual(covered, list(range(layer_num)))
        self.assertEqual([end - start for start, end in ranges], [10] * 6 + [9] * 2)


class TestExchangeRounds(CustomTestCase):
    def test_full_window_rounds_are_all_balanced(self):
        plan = make_plan(PAGES_PER_WINDOW)
        rounds = build_exchange_rounds(plan, plan.windows()[0])
        self.assertEqual(len(rounds), PAGES_PER_WINDOW // SHARD_SIZE)
        # Every rank contributes in every round, so each collective is balanced.
        self.assertTrue(all(r.is_full() for r in rounds))

    def test_rounds_cover_pages_once_with_the_correct_owners(self):
        for count in (1, 3, 8, 17, 30, PAGES_PER_WINDOW, PAGES_PER_WINDOW + 21):
            with self.subTest(page_count=count):
                plan = make_plan(count)
                for window in plan.windows():
                    rounds = build_exchange_rounds(plan, window)
                    busiest_owner = max(
                        len(plan.owned_ordinals(rank, window))
                        for rank in range(SHARD_SIZE)
                    )
                    self.assertEqual(plan.rounds(window), busiest_owner)
                    self.assertEqual(len(rounds), busiest_owner)
                    carried = [page for r in rounds for page in r.pages()]
                    self.assertEqual(sorted(carried), list(window.ordinals()))
                    for r in rounds:
                        self.assertEqual(len(r.contributions), SHARD_SIZE)
                        for rank, ordinal in enumerate(r.contributions):
                            if ordinal is not None:
                                self.assertEqual(plan.page_owners[ordinal], rank)

    def test_partial_window_lets_short_owners_idle(self):
        # 30 pages over 8 ranks -> 4 rounds; ranks owning only 3 idle in the last.
        plan = make_plan(30)
        rounds = build_exchange_rounds(plan, plan.windows()[0])
        self.assertEqual(len(rounds), 4)
        self.assertFalse(rounds[-1].is_full())
        self.assertEqual(len(rounds[-1].contributor_ranks()), 30 % SHARD_SIZE)

    def test_schedule_is_identical_on_every_rank(self):
        # Nothing rank-local feeds the schedule, so all ranks submit the same
        # rounds in the same order -- the property that keeps the collective
        # from mismatching.
        plan = make_plan(200, op_id=11)
        window = plan.windows()[0]
        schedules = [build_exchange_rounds(plan, window) for _ in range(SHARD_SIZE)]
        for other in schedules[1:]:
            self.assertEqual(schedules[0], other)

    def test_empty_window_has_no_rounds(self):
        plan = make_plan(0)
        self.assertEqual(build_exchange_rounds(plan, _EMPTY_WINDOW), [])

    def test_owner_pages_follow_round_order(self):
        plan = make_plan(PAGES_PER_WINDOW)
        window = plan.windows()[0]
        rounds = build_exchange_rounds(plan, window)
        for rank in range(SHARD_SIZE):
            from_rounds = [
                r.contributions[rank]
                for r in rounds
                if r.contributions[rank] is not None
            ]
            self.assertEqual(from_rounds, plan.owned_ordinals(rank, window))


class TestRoundEdges(CustomTestCase):
    def test_idle_owner_is_not_always_last(self):
        positions = set()
        for request_id in (f"req{i}" for i in range(32)):
            plan = make_plan(
                3,
                request_id=request_id,
                shard_size=4,
                window_base=rotation_base(request_id),
            )
            contributions = build_exchange_rounds(plan, plan.windows()[0])[
                0
            ].contributions
            self.assertEqual(sum(page is None for page in contributions), 1)
            positions.add(
                tuple(i for i, page in enumerate(contributions) if page is None)
            )
        self.assertGreater(len(positions), 1)

    def test_component_order_is_shared_by_both_directions(self):
        self.assertEqual(EXCHANGE_COMPONENTS, ("target", "indexer"))
