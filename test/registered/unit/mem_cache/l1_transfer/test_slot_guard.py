"""CPU-only physical page lifetime tests, independent of the tree and allocator."""

import random
import threading
import unittest
from dataclasses import replace
from typing import NamedTuple
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TransferId(NamedTuple):
    """Stand-in for the wire TransferId: any hashable with target_incarnation."""

    target_incarnation: str
    rid: str
    attempt: int


def make_guard(**limits):
    defaults = {
        "max_remote_pinned_pages": 8,
        "max_active_leases": 8,
        "max_pending_reclaim_pages": 8,
        "max_pin_pages": 128,
        "max_tombstones": 16,
    }
    defaults.update(limits)
    return s.SlotGuard(1, "source", **defaults)


def identity(attempt=0):
    return s.LeaseIdentity(
        "source", "target", TransferId("target", "r", attempt), f"nonce-{attempt}"
    )


def published(guard, *page_ids):
    refs = guard.register_allocation(page_ids)
    guard.publish(refs)
    return refs


def epoch(guard, number=1, protected=()):
    guard.begin_epoch(number)
    guard.install_epoch(number, protected)


class SlotGuardTest(CustomTestCase):
    def test_free_and_pin_orders_never_reuse_a_pinned_page(self):
        # Free first: retirement closes the page to new Pins.
        guard = make_guard()
        refs = published(guard, 1)
        self.assertEqual(guard.on_free(refs).pending, refs)
        with self.assertRaises(s.PageUnavailable):
            guard.try_pin(identity(), refs)
        self.assertEqual(guard.snapshot().held_pages, ())
        # Pin first: the free defers the page until its reader releases it.
        guard = make_guard()
        refs = published(guard, 1)
        guard.try_pin(identity(), refs)
        self.assertEqual(guard.on_free(refs).deferred, refs)
        self.assertEqual(guard.snapshot().free_pages, ())
        self.assertTrue(guard.release(identity(), refs))
        self.assertEqual(guard.snapshot().pending_pages, refs)
        epoch(guard)
        self.assertEqual(guard.drain_reclaim(1), refs)
        self.assertEqual(guard.drain_reclaim(1), ())

    def test_rejected_lease_operations_change_no_state(self):
        def pin(guard, refs):
            guard.try_pin(identity(), refs[:2])

        def pin_and_release(guard, refs):
            pin(guard, refs)
            guard.release(identity(), refs[:2])

        def shared_reader(guard, refs):
            guard.try_pin(identity(), refs[:1])
            guard.try_pin(identity(1), refs[:1])

        def retire_two(guard, refs):
            guard.on_free(refs[:2])

        def nothing(guard, refs):
            pass

        # (case, guard limits, prepare, rejected operation, exact error)
        cases = (
            ("retired_page_in_batch", {}, lambda g, r: g.on_free(r[1:2]),
                lambda g, r: g.try_pin(identity(), r[:2]), s.PageUnavailable),
            ("duplicate_pages", {}, nothing,
                lambda g, r: g.try_pin(identity(), (r[0], r[0])), ValueError),
            ("empty_batch", {}, nothing,
                lambda g, r: g.try_pin(identity(), ()), ValueError),
            ("mutable_batch", {}, nothing,
                lambda g, r: g.try_pin(identity(), list(r)), ValueError),
            ("foreign_source", {}, nothing,
                lambda g, r: g.try_pin(replace(identity(), source_incarnation="x"), r),
                ValueError),
            ("replay_changes_pages", {}, pin,
                lambda g, r: g.try_pin(identity(), r[:1]), s.ReplayConflict),
            ("transfer_rebound_to_new_nonce", {}, pin,
                lambda g, r: g.try_pin(replace(identity(), nonce="x"), r[:2]),
                s.ReplayConflict),
            ("nonce_rebound_to_new_transfer", {}, pin,
                lambda g, r: g.try_pin(replace(identity(1), nonce="nonce-0"), r[:2]),
                s.ReplayConflict),
            ("release_changes_pages", {}, pin,
                lambda g, r: g.release(identity(), r[1::-1]), s.ReplayConflict),
            ("unknown_lease", {}, nothing,
                lambda g, r: g.release(identity(9), r[:1]), s.SlotGuardError),
            ("released_lease_revived", {}, pin_and_release,
                lambda g, r: g.try_pin(identity(), r[:2]), s.ReplayConflict),
            ("active_lease_quota", {"max_active_leases": 1}, pin,
                lambda g, r: g.try_pin(identity(1), r[2:]), s.QuotaExceeded),
            ("tombstone_quota", {"max_tombstones": 1}, pin_and_release,
                lambda g, r: g.try_pin(identity(1), r[2:]), s.QuotaExceeded),
            ("unique_page_quota", {"max_remote_pinned_pages": 1}, shared_reader,
                lambda g, r: g.try_pin(identity(2), r[1:2]), s.QuotaExceeded),
            ("reclaim_overflow", {"max_pending_reclaim_pages": 1}, retire_two,
                lambda g, r: g.try_pin(identity(), r[2:]), s.QuotaExceeded),
        )  # fmt: skip
        for name, limits, prepare, operation, error in cases:
            with self.subTest(name):
                guard = make_guard(**limits)
                refs = published(guard, 1, 2, 3)
                prepare(guard, refs)
                before = (guard.snapshot(), guard.owner_counts())
                with self.assertRaises(error) as raised:
                    operation(guard, refs)
                self.assertIs(type(raised.exception), error)
                self.assertEqual((guard.snapshot(), guard.owner_counts()), before)

    def test_replay_is_idempotent_and_service_thread_release_only_queues(self):
        guard = make_guard(max_remote_pinned_pages=2, max_pending_reclaim_pages=1)
        refs = published(guard, 1, 2, 3)
        grant = guard.try_pin(identity(), refs[:2])
        self.assertIs(guard.try_pin(identity(), refs[:2]), grant)
        guard.try_pin(identity(1), refs[:1])
        self.assertEqual(guard.owner_counts().held_pages, 2)
        guard.on_free(refs)
        releases = ((identity(), refs[:2]), (identity(1), refs[:1]))
        results = []
        thread = threading.Thread(
            target=lambda: results.extend(
                guard.release(lease, pages) for lease, pages in releases * 2
            )
        )
        thread.start()
        thread.join(2)
        self.assertEqual(results, [True, True, False, False])
        snapshot = guard.snapshot()
        self.assertEqual(snapshot.held_pages, ())
        self.assertEqual(snapshot.free_pages, ())
        self.assertEqual(snapshot.pending_pages, refs[2:])
        self.assertEqual(snapshot.reclaim_overflow_pages, refs[:2])
        self.assertTrue(snapshot.backpressured)
        epoch(guard)
        self.assertEqual(guard.drain_reclaim(1), refs)
        self.assertFalse(guard.snapshot().backpressured)

    def test_generation_aba_and_delayed_publish_cannot_revive_old_page(self):
        guard = make_guard()
        old = published(guard, 1)
        guard.on_free(old)
        with self.assertRaises(s.PageUnavailable):
            guard.publish(old)
        epoch(guard)
        guard.drain_reclaim(1)
        new = guard.register_allocation((1,))
        self.assertEqual(new[0].generation, old[0].generation + 1)
        with self.assertRaises(s.PageUnavailable):
            guard.publish(old)
        with self.assertRaises(s.PageUnavailable):
            guard.try_pin(identity(), old)
        guard.publish(new)
        guard.finish_epoch(1)
        self.assertEqual(guard.try_pin(identity(), new).pages, new)

    def test_pending_limit_retains_overflow_without_unsafe_free(self):
        guard = make_guard(max_pending_reclaim_pages=1)
        refs = published(guard, 1, 2, 3)
        decision = guard.on_free(refs)
        self.assertEqual(decision.pending, refs[:1])
        self.assertEqual(decision.deferred, refs[1:])
        self.assertTrue(decision.backpressured)
        snapshot = guard.snapshot()
        self.assertEqual(snapshot.free_pages, ())
        self.assertEqual(snapshot.local_owned_pages, ())
        self.assertEqual(snapshot.reclaim_overflow_pages, refs[1:])
        with self.assertRaises(s.PageUnavailable):
            guard.register_allocation((2,))
        with self.assertRaises(s.UnsafeReset):
            guard.reset(2)
        epoch(guard)
        self.assertEqual(guard.drain_reclaim(1), refs)
        guard.finish_epoch(1)
        guard.reset(2)
        self.assertEqual(guard.register_allocation((1,))[0].pool_epoch, 2)

    def test_freeze_to_install_window_keeps_frozen_refs_protected(self):
        guard = make_guard()
        refs = published(guard, 1, 2, 3)
        grant = guard.try_pin(identity(), refs[:1])
        self.assertEqual(guard.on_free(refs[:1]).deferred, refs[:1])
        self.assertEqual(guard.begin_epoch(1).held_pages, refs[:1])
        self.assertIs(guard.try_pin(identity(), refs[:1]), grant)
        with self.assertRaises(s.PinsFrozen):
            guard.try_pin(identity(1), refs[1:2])
        # A release after the freeze still needs the committed union to cover it.
        self.assertTrue(guard.release(identity(), refs[:1]))
        with self.assertRaises(s.EpochError):
            guard.install_epoch(1, ())
        guard.install_epoch(1, refs[:1])
        with self.assertRaises(s.EpochError):
            guard.install_epoch(1, refs[:1])
        self.assertEqual(guard.on_free(refs[1:]).immediately_reusable, refs[1:])
        self.assertEqual(guard.drain_reclaim(1), ())
        guard.finish_epoch(1)
        epoch(guard, 2)
        self.assertEqual(guard.drain_reclaim(2), refs[:1])

    def test_unpublished_pages_double_free_and_epoch_mismatch_fail_closed(self):
        guard = make_guard()
        refs = guard.register_allocation((1,))
        with self.assertRaises(s.PageUnavailable):
            guard.try_pin(identity(), refs)
        with self.assertRaises(s.PageUnavailable):
            guard.publish((s.PageRef(1, 2, 1),))
        guard.on_free(refs)
        with self.assertRaises(s.PageUnavailable):
            guard.on_free(refs)
        with self.assertRaises(s.EpochError):
            guard.drain_reclaim(1)
        epoch(guard)
        with self.assertRaises(s.EpochError):
            guard.drain_reclaim(2)
        self.assertEqual(guard.drain_reclaim(1), refs)
        with self.assertRaises(s.PageUnavailable):
            guard.drain_reclaim(1, refs)

    def test_epoch_entries_reject_a_missing_epoch_id(self):
        """Without a committed epoch no entry may reclaim a retired page;
        None must not match an idle ledger's absent epochs."""
        guard = make_guard()
        refs = published(guard, 1)
        guard.on_free(refs)
        before = (guard.snapshot(), guard.owner_counts())
        operations = {
            "drain": lambda: guard.drain_reclaim(None),
            "install": lambda: guard.install_epoch(None, ()),
            "finish": lambda: guard.finish_epoch(None),
            "begin": lambda: guard.begin_epoch(None),
        }
        for name, operation in operations.items():
            with self.subTest(name):
                with self.assertRaises(s.EpochError):
                    operation()
                self.assertEqual((guard.snapshot(), guard.owner_counts()), before)
        epoch(guard)
        self.assertEqual(guard.drain_reclaim(1), refs)

    def test_registration_counts_each_page_once_within_max_pages(self):
        """A rejected registration batch changes nothing: no page is counted
        twice and the registration domain stays within max_pages."""
        guard = make_guard(max_pages=2)
        guard.register_allocation((1,))
        before = (guard.snapshot(), guard.owner_counts())
        batches = {"duplicate": (2, 2), "over_max_pages": (2, 3), "non_int": ("2",)}
        for name, pages in batches.items():
            for check in (guard.allocation_preflight, guard.register_allocation):
                with self.subTest(name, check=check.__name__):
                    with self.assertRaises(ValueError):
                        check(pages)
                    self.assertEqual((guard.snapshot(), guard.owner_counts()), before)

    def test_lease_identity_bounds_the_transfer_id_it_retains(self):
        """Replay history retains each transfer id, so its fields are bounded."""
        transfers = {
            "unbounded_rid": TransferId("target", "r" * 4097, 0),
            "unbounded_attempt": TransferId("target", "r", 1 << 64),
        }
        for name, transfer in transfers.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    s.LeaseIdentity("source", "target", transfer, "nonce")

    def test_owner_operations_reject_background_threads(self):
        guard = make_guard()
        refs = published(guard, 1)
        operations = (
            lambda: guard.register_allocation((2,)),
            lambda: guard.on_free(refs),
            lambda: guard.begin_epoch(1),
            lambda: guard.reset(2),
        )
        rejected = []

        def worker():
            for operation in operations:
                try:
                    operation()
                except s.OwnerThreadViolation:
                    rejected.append(operation)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertEqual(len(rejected), len(operations))
        self.assertEqual(guard.snapshot().published_pages, refs)

    def test_batches_validate_every_page_before_mutation(self):
        guard = make_guard(max_pin_pages=1)
        refs = guard.register_allocation((1, 2))
        with self.assertRaises(ValueError):
            guard.publish(refs)
        with self.assertRaises(ValueError):
            guard.try_pin(identity(), refs)
        with self.assertRaises(s.PageUnavailable):
            guard.on_free((refs[0], replace(refs[1], generation=2)))
        snapshot = guard.snapshot()
        self.assertEqual(snapshot.local_owned_pages, refs)
        self.assertEqual(snapshot.published_pages, ())

    def test_epoch_paths_never_enumerate_the_page_pool(self):
        class NoFullScan(dict):
            def values(self):
                raise AssertionError("epoch enumerated the physical pool")

            items = __iter__ = values

        for held_count in (0, 2):
            with self.subTest(held_count=held_count):
                guard = make_guard(max_pages=1024)
                refs = guard.register_allocation(tuple(range(1024)))
                guard.publish(refs[:6])
                if held_count:
                    guard.try_pin(identity(), refs[:held_count])
                    guard.on_free(refs[:held_count])
                guard.on_free(refs[2:4])
                guard._pages = NoFullScan(guard._pages)
                with patch.object(
                    guard, "snapshot", side_effect=AssertionError("debug census")
                ):
                    snapshot = guard.begin_epoch(1)
                    self.assertEqual(snapshot.held_pages, refs[:held_count])
                    self.assertEqual(snapshot.held_local_owned_pages, ())
                    self.assertEqual(snapshot.reclaim_candidate_pages, refs[2:4])
                    self.assertEqual(snapshot.registered_page_count, 1024)
                    self.assertEqual(snapshot.deferred_page_count, held_count)
                    guard.install_epoch(1, snapshot.held_pages)
                    self.assertEqual(guard.drain_reclaim(1), refs[2:4])
                    if held_count:
                        guard.release(identity(), refs[:held_count])
                    self.assertEqual(guard.drain_reclaim(1), ())
                    freed = guard.on_free(refs[5:6]).immediately_reusable
                    self.assertEqual(freed, refs[5:6])
                    guard.on_free(guard.register_allocation((5,)))
                    guard.finish_epoch(1)
                    epoch(guard, 2)
                    self.assertEqual(guard.drain_reclaim(2), refs[:held_count])
                    guard.finish_epoch(2)

    def test_sparse_epoch_counts_match_full_census_over_seeded_commands(self):
        # A small pending queue makes overflow and backpressure reachable.
        guard = make_guard(
            max_pages=16, max_tombstones=512, max_pending_reclaim_pages=2
        )
        published(guard, *range(16))
        rng = random.Random(1773)

        def expected_epoch_fields(census):
            retired = set(census.deferred_pages + census.pending_pages)
            return {
                "held_pages": census.held_pages,
                "held_local_owned_pages": census.held_local_owned_pages,
                "reclaim_candidate_pages": tuple(
                    sorted(retired - set(census.held_pages))
                ),
                "local_owned_page_count": len(census.local_owned_pages),
                "deferred_page_count": len(census.deferred_pages),
                "pending_reclaim_page_count": len(census.pending_pages),
                "free_page_count": len(census.free_pages),
                "reclaim_overflow_page_count": len(census.reclaim_overflow_pages),
                "active_lease_count": len(census.active_leases),
                "backpressured": census.backpressured,
            }

        for step in range(400):
            census = guard.snapshot()
            operation = rng.randrange(5)
            with self.subTest(step=step, operation=operation):
                try:
                    if operation == 0 and census.local_owned_pages:
                        guard.on_free((rng.choice(census.local_owned_pages),))
                    elif operation == 1 and census.published_pages:
                        page = rng.choice(census.published_pages)
                        guard.try_pin(identity(step), (page,))
                    elif operation == 2 and census.active_leases:
                        grant = rng.choice(census.active_leases)
                        guard.release(grant.identity, grant.pages)
                    elif operation == 3 and census.free_pages:
                        page_id = rng.choice(census.free_pages).page_id
                        guard.publish(guard.register_allocation((page_id,)))
                    elif operation == 4:
                        snapshot = guard.begin_epoch(step)
                        guard.install_epoch(step, snapshot.held_pages)
                        guard.drain_reclaim(step)
                        guard.finish_epoch(step)
                        expected = expected_epoch_fields(census)
                        observed = {name: vars(snapshot)[name] for name in expected}
                        self.assertEqual(observed, expected)
                except s.QuotaExceeded:
                    pass
        for grant in guard.snapshot().active_leases:
            guard.release(grant.identity, grant.pages)
        local = guard.snapshot().local_owned_pages
        if local:
            guard.on_free(local)
        epoch(guard, 400)
        guard.drain_reclaim(400)
        guard.finish_epoch(400)
        self.assertEqual(len(guard.snapshot().free_pages), 16)
        guard.reset(2)


if __name__ == "__main__":
    unittest.main()
