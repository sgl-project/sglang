"""CPU-only physical page lifetime tests, independent of the tree and allocator."""

import random
import threading
import unittest
from dataclasses import replace

from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.ledger import (
    TransferId,
    census,
    guard,
    lease_identity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CREDIT = 8


def published(g, *page_ids):
    refs = g.register_allocation(page_ids)
    g.publish(refs)
    return refs


def epoch(g, number=1, protected=()):
    """Begin and install an epoch; returns the pages it reclaimed."""
    g.begin_epoch(number)
    return g.install_epoch(number, protected)


def finish(g, number):
    g.finish_epoch(number, verified_headroom_pages=CREDIT)


class SlotGuardTest(CustomTestCase):
    def test_free_and_pin_orders_never_reuse_a_pinned_page(self):
        # Free first: retirement closes the page to new Pins.
        g = guard(credit=CREDIT)
        refs = published(g, 1)
        self.assertEqual(g.on_free(refs), ())
        with self.assertRaises(s.PageUnavailable):
            g.try_pin(lease_identity(), refs)
        self.assertEqual(census(g).held_pages, ())
        # Pin first: the free defers the page until its reader releases it.
        g = guard(credit=CREDIT)
        refs = published(g, 1)
        g.try_pin(lease_identity(), refs)
        self.assertEqual(g.on_free(refs), ())
        self.assertEqual(census(g).deferred_pages, refs)
        g.release(lease_identity(), refs)
        self.assertEqual(census(g).pending_pages, refs)
        self.assertEqual(epoch(g), refs)

    def test_rejected_lease_operations_change_no_state(self):
        def pin(g, refs):
            g.try_pin(lease_identity(), refs[:2])

        def shared_reader(g, refs):
            g.try_pin(lease_identity(), refs[:1])
            g.try_pin(lease_identity(1), refs[:1])

        def retire_two(g, refs):
            g.on_free(refs[:2])

        def nothing(g, refs):
            pass

        # (case, guard limits and credit, prepare, rejected operation, exact error)
        cases = (
            ("retired_page_in_batch", {}, lambda g, r: g.on_free(r[1:2]),
                lambda g, r: g.try_pin(lease_identity(), r[:2]), s.PageUnavailable),
            ("duplicate_pages", {}, nothing,
                lambda g, r: g.try_pin(lease_identity(), (r[0], r[0])), ValueError),
            ("foreign_source", {}, nothing,
                lambda g, r: g.try_pin(
                    replace(lease_identity(), source_incarnation="x"), r),
                ValueError),
            ("lease_already_active", {}, pin,
                lambda g, r: g.try_pin(lease_identity(), r[2:]), s.ReplayConflict),
            ("release_changes_pages", {}, pin,
                lambda g, r: g.release(lease_identity(), r[1::-1]), s.SlotGuardError),
            ("unknown_lease", {}, nothing,
                lambda g, r: g.release(lease_identity(9), r[:1]), s.SlotGuardError),
            ("active_lease_quota", {"max_active_leases": 1}, pin,
                lambda g, r: g.try_pin(lease_identity(1), r[2:]), s.QuotaExceeded),
            ("unique_page_quota", {"max_remote_pinned_pages": 1}, shared_reader,
                lambda g, r: g.try_pin(lease_identity(2), r[1:2]), s.QuotaExceeded),
            ("headroom_credit", {"credit": 1}, nothing,
                lambda g, r: g.try_pin(lease_identity(), r[:2]), s.QuotaExceeded),
            ("reclaim_overflow", {"max_pending_reclaim_pages": 1}, retire_two,
                lambda g, r: g.try_pin(lease_identity(), r[2:]), s.QuotaExceeded),
        )  # fmt: skip
        for name, limits, prepare, operation, error in cases:
            with self.subTest(name):
                g = guard(**{"credit": CREDIT, **limits})
                refs = published(g, 1, 2, 3)
                prepare(g, refs)
                before = (census(g), g.owner_counts())
                with self.assertRaises(error) as raised:
                    operation(g, refs)
                self.assertIs(type(raised.exception), error)
                self.assertEqual((census(g), g.owner_counts()), before)

    def test_service_thread_release_only_queues_and_overflow_backpressures(self):
        g = guard(credit=CREDIT, max_remote_pinned_pages=2, max_pending_reclaim_pages=1)
        refs = published(g, 1, 2, 3)
        g.try_pin(lease_identity(), refs[:2])
        g.try_pin(lease_identity(1), refs[:1])
        self.assertEqual(g.owner_counts().held_pages, 2)
        g.on_free(refs)
        releases = ((lease_identity(), refs[:2]), (lease_identity(1), refs[:1]))
        thread = threading.Thread(
            target=lambda: [g.release(lease, pages) for lease, pages in releases]
        )
        thread.start()
        thread.join(2)
        pages = census(g)
        self.assertEqual(pages.held_pages, ())
        self.assertEqual(pages.free_pages, ())
        self.assertEqual(pages.pending_pages, refs[2:])
        # The full pending queue keeps the released pages as overflow.
        self.assertEqual(pages.deferred_pages, refs[:2])
        self.assertEqual(g.owner_counts().reclaim_overflow_pages, 2)
        self.assertEqual(epoch(g), refs)
        self.assertEqual(g.owner_counts().reclaim_overflow_pages, 0)

    def test_generation_aba_and_delayed_publish_cannot_revive_old_page(self):
        g = guard(credit=CREDIT)
        old = published(g, 1)
        g.on_free(old)
        with self.assertRaises(s.PageUnavailable):
            g.publish(old)
        epoch(g)
        new = g.register_allocation((1,))
        self.assertEqual(new[0].generation, old[0].generation + 1)
        with self.assertRaises(s.PageUnavailable):
            g.publish(old)
        with self.assertRaises(s.PageUnavailable):
            g.try_pin(lease_identity(), old)
        g.publish(new)
        finish(g, 1)
        g.try_pin(lease_identity(), new)
        self.assertEqual(census(g).held_pages, new)

    def test_pending_limit_retains_overflow_without_unsafe_free(self):
        g = guard(max_pending_reclaim_pages=1)
        refs = published(g, 1, 2, 3)
        self.assertEqual(g.on_free(refs), ())
        pages = census(g)
        self.assertEqual(pages.pending_pages, refs[:1])
        self.assertEqual(pages.deferred_pages, refs[1:])
        self.assertEqual(pages.free_pages, ())
        self.assertEqual(pages.local_owned_pages, ())
        self.assertEqual(g.owner_counts().reclaim_overflow_pages, 2)
        with self.assertRaises(s.PageUnavailable):
            g.register_allocation((2,))
        with self.assertRaises(s.UnsafeReset):
            g.reset(2)
        self.assertEqual(epoch(g), refs)
        finish(g, 1)
        g.reset(2)
        self.assertEqual(g.register_allocation((1,))[0].pool_epoch, 2)

    def test_freeze_to_install_window_keeps_frozen_refs_protected(self):
        g = guard(credit=CREDIT)
        refs = published(g, 1, 2, 3)
        g.try_pin(lease_identity(), refs[:1])
        self.assertEqual(g.on_free(refs[:1]), ())
        self.assertEqual(g.begin_epoch(1).held_pages, refs[:1])
        with self.assertRaises(s.PinsFrozen):
            g.try_pin(lease_identity(1), refs[1:2])
        # A release after the freeze still needs the committed union to cover it.
        g.release(lease_identity(), refs[:1])
        with self.assertRaises(s.EpochError):
            g.install_epoch(1, ())
        self.assertEqual(g.install_epoch(1, refs[:1]), ())
        with self.assertRaises(s.EpochError):
            g.install_epoch(1, refs[:1])
        self.assertEqual(g.on_free(refs[1:]), refs[1:])
        finish(g, 1)
        self.assertEqual(epoch(g, 2), refs[:1])

    def test_unpublished_pages_double_free_and_epoch_mismatch_fail_closed(self):
        g = guard(credit=CREDIT)
        refs = g.register_allocation((1,))
        with self.assertRaises(s.PageUnavailable):
            g.try_pin(lease_identity(), refs)
        with self.assertRaises(s.PageUnavailable):
            g.publish((s.PageRef(1, 2, 1),))
        with self.assertRaises(s.PageUnavailable):
            g.on_free((replace(refs[0], generation=2),))
        self.assertEqual(census(g).local_owned_pages, refs)
        g.on_free(refs)
        with self.assertRaises(s.PageUnavailable):
            g.on_free(refs)
        g.begin_epoch(1)
        with self.assertRaises(s.EpochError):
            g.install_epoch(2, ())
        self.assertEqual(g.install_epoch(1, ()), refs)
        finish(g, 1)
        g.begin_epoch(2)
        with self.assertRaises(s.PageUnavailable):
            g.install_epoch(2, (), refs)

    def test_epoch_entries_reject_a_missing_epoch_id(self):
        """Without a committed epoch no entry may reclaim a retired page;
        None must not match an idle ledger's absent epochs."""
        g = guard()
        refs = published(g, 1)
        g.on_free(refs)
        before = (census(g), g.owner_counts())
        operations = {
            "install": lambda: g.install_epoch(None, ()),
            "finish": lambda: finish(g, None),
            "begin": lambda: g.begin_epoch(None),
        }
        for name, operation in operations.items():
            with self.subTest(name):
                with self.assertRaises(s.EpochError):
                    operation()
                self.assertEqual((census(g), g.owner_counts()), before)
        self.assertEqual(epoch(g), refs)

    def test_lease_identity_bounds_the_transfer_id(self):
        """Pin requests decode the lease identity from the control wire, so an
        unbounded transfer id is rejected when the identity is built."""
        transfers = {
            "unbounded_rid": TransferId("target", "r" * 4097, 0),
            "unbounded_attempt": TransferId("target", "r", 1 << 64),
        }
        for name, transfer in transfers.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    s.LeaseIdentity("source", "target", transfer, "nonce")

    def test_owner_operations_reject_background_threads(self):
        g = guard()
        refs = published(g, 1)
        operations = (
            lambda: g.register_allocation((2,)),
            lambda: g.on_free(refs),
            lambda: g.begin_epoch(1),
            lambda: g.reset(2),
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
        self.assertEqual(census(g).published_pages, refs)

    def test_epoch_paths_never_enumerate_the_page_pool(self):
        class NoFullScan(dict):
            def values(self):
                raise AssertionError("epoch enumerated the physical pool")

            items = __iter__ = values

        for held_count in (0, 2):
            with self.subTest(held_count=held_count):
                g = guard(credit=CREDIT, max_pages=1024)
                refs = g.register_allocation(tuple(range(1024)))
                g.publish(refs[:6])
                if held_count:
                    g.try_pin(lease_identity(), refs[:held_count])
                    g.on_free(refs[:held_count])
                g.on_free(refs[2:4])
                g._pages = NoFullScan(g._pages)
                snapshot = g.begin_epoch(1)
                self.assertEqual(snapshot.held_pages, refs[:held_count])
                self.assertEqual(snapshot.reclaim_candidate_pages, refs[2:4])
                self.assertEqual(g.install_epoch(1, snapshot.held_pages), refs[2:4])
                if held_count:
                    g.release(lease_identity(), refs[:held_count])
                self.assertEqual(g.on_free(refs[5:6]), refs[5:6])
                g.on_free(g.register_allocation((5,)))
                finish(g, 1)
                self.assertEqual(epoch(g, 2), refs[:held_count])
                finish(g, 2)

    def test_sparse_epoch_sets_match_full_census_over_seeded_commands(self):
        """The held and reclaim-candidate sets an epoch freezes, maintained
        incrementally, equal a full census after any command sequence."""
        # A small pending queue makes overflow and backpressure reachable.
        g = guard(credit=CREDIT, max_pages=16, max_pending_reclaim_pages=2)
        published(g, *range(16))
        rng = random.Random(1773)
        for step in range(1, 401):
            pages = census(g)
            operation = rng.randrange(5)
            with self.subTest(step=step, operation=operation):
                try:
                    if operation == 0 and pages.local_owned_pages:
                        g.on_free((rng.choice(pages.local_owned_pages),))
                    elif operation == 1 and pages.published_pages:
                        page = rng.choice(pages.published_pages)
                        g.try_pin(lease_identity(step), (page,))
                    elif operation == 2 and pages.active_leases:
                        g.release(*rng.choice(pages.active_leases))
                    elif operation == 3 and pages.free_pages:
                        page_id = rng.choice(pages.free_pages).page_id
                        g.publish(g.register_allocation((page_id,)))
                    elif operation == 4:
                        snapshot = g.begin_epoch(step)
                        g.install_epoch(step, snapshot.held_pages)
                        finish(g, step)
                        retired = set(pages.deferred_pages + pages.pending_pages)
                        self.assertEqual(snapshot.held_pages, pages.held_pages)
                        self.assertEqual(
                            snapshot.reclaim_candidate_pages,
                            tuple(sorted(retired - set(pages.held_pages))),
                        )
                except s.QuotaExceeded:
                    pass
        for lease, refs in census(g).active_leases:
            g.release(lease, refs)
        local = census(g).local_owned_pages
        if local:
            g.on_free(local)
        epoch(g, 401)
        finish(g, 401)
        self.assertEqual(len(census(g).free_pages), 16)
        g.reset(2)


if __name__ == "__main__":
    unittest.main()
