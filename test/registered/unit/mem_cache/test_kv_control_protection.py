# SPDX-License-Identifier: Apache-2.0
"""Native host eviction exclusion, shared budgets and split-aware releases.

This file owns protection lifecycle assertions; existing transfer fixtures only
provide real CPU pools, controlled DMA and paired host registration cleanup.
"""

import threading
import unittest
from unittest.mock import patch

import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as transfer_fixtures
import torch
from sglang.srt.managers.kv_control.contracts import ControlError, Status
from sglang.srt.managers.kv_control.protection import (
    HostPrefix,
    ProtectionKey,
    ProtectionRegistry,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
FULL = ComponentType.FULL


def claim(caller="A", task="task", incarnation=1, name="pin"):
    return ProtectionKey(
        caller_id=caller, task_id=task, task_incarnation=incarnation, claim_id=name
    )


def prefixes(*values):
    return tuple(HostPrefix(token_ids=tuple(tokens)) for tokens in values)


def registry(owner, **limits):
    case, cache, service, _ = owner
    result = ProtectionRegistry(
        cache,
        engine_id="engine",
        engine_epoch=service.control.registry.engine_epoch,
        namespace="ns",
        clock=case.clock,
        **limits,
    )
    case.addCleanup(result.invalidate)
    return result


def host_copy(owner, tokens=(1, 2, 3, 4), *, message="save"):
    _, cache, service, dma = owner
    node = transfer_fixtures.seed(cache, tokens)
    result = transfer_fixtures.transfer(service, tokens=tokens, message=message)
    service.progress()
    assert transfer_fixtures.current(service, result).status == Status.RUNNING
    dma.finish()
    service.progress()
    assert transfer_fixtures.current(service, result).status == Status.COMPLETED
    return node


class TestKVControlProtection(cache_fixtures.KVCacheTestCase):
    def error(self, code, operation, **kwargs):
        with self.assertRaises(ControlError) as caught:
            operation(**kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_native_host_pressure_preserves_bytes_until_independent_claim_release(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, _, _ = owner
            held = host_copy(owner)
            ordinary = host_copy(owner, (8, 9, 10, 11), message="ordinary")
            pins = registry(owner)
            accepted = pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
            )
            host = cache.host_pool_group.anchor_entry.host_pool
            cd = held.component_data[FULL]
            saved = cd.host_value.clone()
            self.assertEqual(accepted.state, "active")
            self.assertEqual(cd.host_lock_ref, 1)
            self.assertEqual(cache.evict_host(host.size), 4)
            self.assertIsNone(ordinary.component_data[FULL].host_value)
            self.assertTrue(torch.equal(cd.host_value, saved))
            self.assertTrue(torch.all(host.k_buffer[0][saved] == 7))
            # The transfer operation has completed and released its own holds.
            self.assertTrue(pins.has_background_work)
            self.assertEqual(pins.next_deadline, 110)
            pins.release(key=claim(), revision=2)
            self.assertEqual(cd.host_lock_ref, 0)
            self.assertEqual(cache.evict_host(host.size), 4)
            self.assertIsNone(cd.host_value)
            self.assertFalse(pins.has_background_work)
            cache.sanity_check()

    def test_overlapping_callers_and_multi_branch_scope_charge_union_after_splits(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, _, _ = owner
            leaf = host_copy(owner)
            pins = registry(owner, max_protected_tokens=6)
            pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=5
            )
            branch = host_copy(owner, (1, 2, 5, 6), message="branch")
            parent = leaf.parent
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
            shared = pins.acquire(
                key=claim("B"),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4), (1, 2, 5, 6)),
                ttl_seconds=20,
            )
            self.assertEqual(shared.physical_tokens, 6)
            self.assertEqual(pins.protected_tokens, 6)
            self.assertEqual(pins.active_claims, 2)
            # An overlapping branch/ancestor is charged once within a claim too.
            pins.acquire(
                key=claim("C"), revision=1, prefixes=prefixes((1, 2)), ttl_seconds=10
            )
            self.assertEqual(pins.protected_tokens, 6)
            self.clock.now = 106
            self.assertEqual([s.key for s in pins.expire()], [claim()])
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 2)
            self.assertEqual(cache.evict_host(100), 0)
            pins.release(key=claim("B"), revision=2)
            self.assertEqual(pins.protected_tokens, 2)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(branch.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 1)
            pins.release(key=claim("C"), revision=2)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(cache.evict_host(100), 6)
            cache.sanity_check()

    def test_revision_ordering_idempotence_and_release_before_late_renew(self):
        with transfer_fixtures.hicache(self) as owner:
            node = host_copy(owner)
            pins = registry(owner)
            request = {
                "key": claim(),
                "revision": 1,
                "prefixes": prefixes((1, 2, 3, 4)),
                "ttl_seconds": 10,
            }
            initial = pins.acquire(**request)
            self.clock.now = 103
            self.assertEqual(pins.acquire(**request), initial)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            self.error("conflict", pins.acquire, **{**request, "ttl_seconds": 20})
            self.error(
                "stale_revision", pins.renew, key=claim(), revision=0, ttl_seconds=20
            )
            renewed = pins.renew(key=claim(), revision=2, ttl_seconds=5)
            self.assertEqual(renewed.expires_at, 108)
            self.clock.now = 104
            self.assertEqual(
                pins.renew(key=claim(), revision=2, ttl_seconds=5), renewed
            )
            # Release arrives ahead of a queued older renew. Neither it nor a
            # later higher-revision renew may recreate the ended claim.
            released = pins.release(key=claim(), revision=4)
            self.assertEqual(pins.release(key=claim(), revision=4), released)
            self.error(
                "stale_revision", pins.renew, key=claim(), revision=3, ttl_seconds=10
            )
            self.error("conflict", pins.renew, key=claim(), revision=4, ttl_seconds=10)
            self.error("released", pins.renew, key=claim(), revision=5, ttl_seconds=10)
            self.error("released", pins.acquire, **{**request, "revision": 5})
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            # A release without a preceding acquire must also fence late work.
            absent = claim(name="not-yet-seen")
            pins.release(key=absent, revision=5)
            self.error("stale_revision", pins.acquire, **{**request, "key": absent})
            self.error(
                "released", pins.acquire, **{**request, "key": absent, "revision": 6}
            )
            owner[1].sanity_check()

    def test_finite_ttl_expiry_batches_and_no_access_renewal(self):
        with transfer_fixtures.hicache(self) as owner:
            node = host_copy(owner)
            pins = registry(owner, max_ttl_seconds=20)
            request = {
                "key": claim(),
                "revision": 1,
                "prefixes": prefixes((1, 2, 3, 4)),
            }
            for ttl in (float("nan"), float("inf"), -1, 0, 21, True):
                with self.subTest(ttl=ttl):
                    self.error("invalid", pins.acquire, **request, ttl_seconds=ttl)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            for caller in ("A", "B", "C"):
                pins.acquire(**{**request, "key": claim(caller)}, ttl_seconds=2)
            self.clock.now = 101
            self.assertEqual(pins.get(claim()).expires_at, 102)
            cache_fixtures.match_len(owner[1], (1, 2, 3, 4))
            self.clock.now = 102
            # Query may discover expiry before the idle service tick. The
            # terminal change must still reach the background publisher.
            self.assertEqual(pins.get(claim()).state, "expired")
            self.assertEqual(len(pins.expire(limit=1)), 1)
            self.assertTrue(pins.has_background_work)
            self.assertEqual(len(pins.expire(limit=1)), 1)
            self.assertEqual(len(pins.expire(limit=1)), 1)
            self.assertFalse(pins.has_background_work)
            self.assertIsNone(pins.next_deadline)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.error("expired", pins.renew, key=claim(), revision=2, ttl_seconds=10)
            self.assertEqual(owner[1].evict_host(100), 4)
            owner[1].sanity_check()

    def test_task_cleanup_preserves_other_callers_siblings_and_incarnations(self):
        with transfer_fixtures.hicache(self) as owner:
            node = host_copy(owner)
            pins = registry(owner)
            keys = (claim(), claim("B"), claim(task="sibling"), claim(incarnation=2))
            for key in keys:
                pins.acquire(
                    key=key, revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
                )
            removed = pins.cleanup_task(
                caller_id="A", task_id="task", task_incarnation=1
            )
            self.assertEqual([s.key for s in removed], [claim()])
            self.assertEqual(node.component_data[FULL].host_lock_ref, 3)
            self.assertEqual(pins.protected_tokens, 4)
            self.error("released", pins.renew, key=claim(), revision=2, ttl_seconds=10)
            self.assertEqual(
                pins.cleanup_task(caller_id="A", task_id="task", task_incarnation=1), ()
            )
            for key in keys[1:]:
                pins.release(key=key, revision=2)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            owner[1].sanity_check()

    def test_capacity_rejects_before_holds_and_keeps_terminal_revision_records(self):
        with transfer_fixtures.hicache(self) as owner:
            held = host_copy(owner)
            extra = host_copy(owner, (8, 9), message="extra")
            pins = registry(owner, max_protected_tokens=4, max_claims=2, max_records=3)
            scope = prefixes((1, 2, 3, 4))
            pins.acquire(key=claim(), revision=1, prefixes=scope, ttl_seconds=10)
            self.error(
                "capacity",
                pins.acquire,
                key=claim("B"),
                revision=1,
                prefixes=prefixes((8, 9)),
                ttl_seconds=10,
            )
            self.assertEqual(extra.component_data[FULL].host_lock_ref, 0)
            pins.acquire(key=claim("B"), revision=1, prefixes=scope, ttl_seconds=10)
            self.error(
                "capacity",
                pins.acquire,
                key=claim("C"),
                revision=1,
                prefixes=scope,
                ttl_seconds=10,
            )
            # Failed revision updates must leave the old scope/deadline intact.
            self.error(
                "capacity",
                pins.acquire,
                key=claim(),
                revision=2,
                prefixes=prefixes((8, 9)),
                ttl_seconds=10,
            )
            self.assertEqual(pins.get(claim()).revision, 1)
            self.assertEqual(held.component_data[FULL].host_lock_ref, 2)
            pins.release(key=claim(), revision=2)
            pins.release(key=claim("B"), revision=2)
            pins.release(key=claim("C"), revision=1)
            self.error(
                "capacity",
                pins.acquire,
                key=claim("D"),
                revision=1,
                prefixes=scope,
                ttl_seconds=10,
            )
            self.error("released", pins.renew, key=claim(), revision=3, ttl_seconds=10)
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(held.component_data[FULL].host_lock_ref, 0)
            owner[1].sanity_check()

    def test_replacement_scope_releases_old_split_receipt_without_losing_others(self):
        with transfer_fixtures.hicache(self) as owner:
            leaf = host_copy(owner)
            pins = registry(owner, max_protected_tokens=6)
            pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
            )
            branch = host_copy(owner, (1, 2, 5, 6), message="branch")
            pins.acquire(
                key=claim("B"), revision=1, prefixes=prefixes((1, 2)), ttl_seconds=10
            )
            pins.acquire(
                key=claim(), revision=2, prefixes=prefixes((1, 2, 5, 6)), ttl_seconds=20
            )
            self.assertEqual(pins.protected_tokens, 4)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(branch.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(leaf.parent.component_data[FULL].host_lock_ref, 2)
            pins.invalidate()
            for node in (leaf, branch, leaf.parent):
                self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            owner[1].sanity_check()

    def test_rejects_unmaterialized_missing_and_incomplete_prefixes_without_inserting(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, _, _ = owner
            node = host_copy(owner)
            pins = registry(owner, max_scope_tokens=8)
            for scope, code in (
                (prefixes((1, 2)), "unsupported"),
                (prefixes((1, 2, 3, 5)), "missing"),
                (prefixes((1, 2, 3, 4, 5)), "missing"),
                (prefixes(tuple(range(9))), "capacity"),
                (
                    (HostPrefix(token_ids=(1, 2, 3, 4), cache_salt="another"),),
                    "missing",
                ),
            ):
                with self.subTest(scope=scope):
                    self.error(
                        code,
                        pins.acquire,
                        key=claim(),
                        revision=1,
                        prefixes=scope,
                        ttl_seconds=10,
                    )
            self.assertIs(node.parent, cache.tree_core.root_node)
            # An inserted branch shares a host prefix, but its new suffix has
            # no host replica. Partial residency cannot activate protection.
            other = transfer_fixtures.seed(cache, (1, 2, 5, 6))
            self.assertIsNone(other.component_data[FULL].host_value)
            self.error(
                "missing",
                pins.acquire,
                key=claim(),
                revision=1,
                prefixes=prefixes((1, 2, 5, 6)),
                ttl_seconds=10,
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(pins.protected_tokens, 0)
            cache.sanity_check()

    def test_clear_invalidation_and_reset_never_replay_receipts_against_new_tree(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            pins = registry(owner)
            pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
            )
            changes = pins.invalidate(reason="namespace_changed")
            self.assertEqual(
                [(s.state, s.reason) for s in changes],
                [("invalidated", "namespace_changed")],
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.error(
                "invalidated",
                pins.acquire,
                key=claim("B"),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4)),
                ttl_seconds=10,
            )
            # A fresh registry is normally installed with a fresh epoch by the
            # service. Here direct construction isolates the reset guard.
            fresh = registry(owner)
            fresh.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
            )
            cache.reset()
            with patch.object(
                cache, "dec_host_lock_ref", side_effect=AssertionError("stale receipt")
            ):
                self.assertEqual(fresh.get(claim()).state, "invalidated")
                self.assertEqual(fresh.get(claim()).reason, "cache_reset")
                self.assertEqual(fresh.protected_tokens, 0)
                self.assertTrue(fresh.has_background_work)
                self.assertEqual(fresh.next_deadline, self.clock.now)
                self.assertEqual([s.state for s in fresh.expire()], ["invalidated"])
                self.assertFalse(fresh.has_background_work)
                fresh.invalidate()
            self.assertFalse(service.has_pending)

    def test_pending_host_backup_cannot_be_claimed_before_publication(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, dma = owner
            node = transfer_fixtures.seed(cache)
            result = transfer_fixtures.transfer(service)
            service.progress()
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.assertIsNotNone(node.write_through_pending_id)
            pins = registry(owner)
            self.error(
                "incomplete",
                pins.acquire,
                key=claim(),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4)),
                ttl_seconds=10,
            )
            self.assertEqual(pins.protected_tokens, 0)
            # The native transfer still owns its single temporary host hold.
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            self.clock.now = 105
            dma.finish()
            service.progress()
            self.assertEqual(
                transfer_fixtures.current(service, result).status, Status.COMPLETED
            )
            active = pins.acquire(
                key=claim(),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4)),
                ttl_seconds=10,
            )
            self.assertEqual(active.activated_at, 105)
            self.assertEqual(active.expires_at, 115)
            pins.release(key=claim(), revision=2)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_failed_native_acquisition_rolls_back_partial_holds_before_acceptance(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, _, _ = owner
            leaf = host_copy(owner)
            host_copy(owner, (1, 2, 5, 6), message="branch")
            pins = registry(owner)
            native_hold = cache.inc_host_lock_ref
            count = 0

            def fail_second(node_id):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("injected acquisition failure")
                return native_hold(node_id)

            request = {
                "key": claim(),
                "revision": 1,
                "prefixes": prefixes((1, 2, 3, 4)),
                "ttl_seconds": 10,
            }
            with (
                patch.object(cache, "inc_host_lock_ref", side_effect=fail_second),
                self.assertRaisesRegex(RuntimeError, "acquisition failure"),
            ):
                pins.acquire(**request)
            self.assertIsNone(pins.get(claim()))
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(leaf.parent.component_data[FULL].host_lock_ref, 0)

            def timed_hold(node_id):
                receipt = native_hold(node_id)
                self.clock.now += 1
                return receipt

            with patch.object(cache, "inc_host_lock_ref", side_effect=timed_hold):
                active = pins.acquire(**request)
            self.assertEqual(active.activated_at, 102)
            self.assertEqual(active.expires_at, 112)
            pins.release(key=claim(), revision=2)
            cache.sanity_check()

    def test_owner_thread_rejection_does_not_modify_native_holds(self):
        with transfer_fixtures.hicache(self) as owner:
            node = host_copy(owner)
            pins = registry(owner)
            pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=10
            )
            outcomes = []

            def wrong_owner():
                try:
                    pins.release(key=claim(), revision=2)
                except ControlError as exc:
                    outcomes.append(exc.code)

            thread = threading.Thread(target=wrong_owner)
            thread.start()
            thread.join()
            self.assertEqual(outcomes, ["wrong_owner"])
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            pins.release(key=claim(), revision=2)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            owner[1].sanity_check()


if __name__ == "__main__":
    unittest.main()
