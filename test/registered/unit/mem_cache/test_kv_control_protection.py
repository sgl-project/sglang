# SPDX-License-Identifier: Apache-2.0
"""Native host eviction exclusion, shared budgets and split-aware releases.

This file owns protection lifecycle assertions; existing transfer fixtures only
provide real CPU pools, controlled DMA and paired host registration cleanup.
"""

import asyncio
import threading
import unittest
from array import array
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import test_kv_control_service as service_fixtures
import test_kv_control_transfers as transfer_fixtures
import torch
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ControlError,
    SessionTarget,
    Status,
)
from sglang.srt.managers.kv_control.http import install_routes
from sglang.srt.managers.kv_control.protection import (
    HostPrefix,
    ProtectionKey,
    ProtectionRegistry,
)
from sglang.srt.managers.kv_control.service import ServiceConfig, dispatch
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.control_protection import ProtectionLimits
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
    selection = msgspec.structs.replace(
        cache_fixtures.target(tokens), namespace=service.config.namespace
    )
    result = transfer_fixtures.transfer(
        service, tokens=tokens, message=message, selection=selection
    )
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


def pin_command(
    service,
    *,
    caller="A",
    message="pin",
    mode="acquire",
    revision=1,
    claim_id="pin",
    ttl=10,
    owner_task=None,
    selection=None,
    salt=None,
):
    caps = service.capabilities()["public_protocol"]["prefix"]
    payload = {
        "mode": mode,
        "target": selection
        or {
            "kind": "prefix",
            "compatibility": {
                "namespace": caps["namespace"],
                "cache_salt": salt,
                "layout_id": caps["layout_id"],
                "page_size": caps["page_size"],
                "components": list(caps["components"]),
                "hash_scheme": caps["hash_scheme"],
            },
            "start_token": 0,
            "end_token": 4,
            "token_ids": [1, 2, 3, 4],
        },
        "location": {"tier": "host"},
        "claim_id": claim_id,
        "revision": revision,
        "owner_task": msgspec.to_builtins(owner_task) if owner_task else None,
    }
    if mode != "release":
        payload["ttl_seconds"] = ttl
    return {
        "kind": "submit",
        "context": msgspec.to_builtins(service.control.context(caller)),
        "envelope": {
            "protocol_version": "0.1",
            "message_id": message,
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "sglang.pin",
                    "action_version": "1.0",
                    "payload": payload,
                }
            ],
        },
    }


def run_pin(service, **kwargs):
    reply = dispatch(
        service,
        msgspec.json.decode(msgspec.json.encode(pin_command(service, **kwargs))),
    )
    assert reply["success"], reply
    return reply["data"]["results"][0]


def query_claim(service, *, caller="A", owner_task=None, claim_id="pin"):
    return dispatch(
        service,
        {
            "kind": "claim",
            "context": msgspec.to_builtins(service.control.context(caller)),
            "claim_id": claim_id,
            "owner_task": msgspec.to_builtins(owner_task) if owner_task else None,
        },
    )


def pin_hints(service, **kwargs):
    encoded = msgspec.json.encode(pin_command(service, **kwargs)["envelope"])
    return msgspec.json.decode(encoded, type=KvHintsEnvelope)


class TestKVControlPinRuntime(cache_fixtures.KVCacheTestCase):
    def test_attached_explicit_prefix_receipt_survives_abort_and_supports_unbound_followup(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            req = service_fixtures.serving_request()
            hints = pin_hints(service)
            original = msgspec.json.encode(hints)
            service.attach(req, hints)
            context = req.kv_control_context
            result = service.control.registry.query(context, "pin").results[0]
            self.assertEqual(result.status, Status.COMPLETED)
            claim_id = result.public_result.claim.claim_id
            self.assertNotEqual(claim_id, "pin")
            self.assertEqual(result.public_result.claim.caller_id, "request")
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_id)["data"][
                    "remaining_ttl"
                ],
                10,
            )
            self.assertEqual(
                query_claim(service, caller="request")["error"]["code"], "missing"
            )
            self.assertEqual(
                query_claim(service, claim_id=claim_id)["error"]["code"], "missing"
            )
            self.clock.now = 103
            replay = service.control.submit(hints, context)
            self.assertTrue(replay.replayed)
            self.assertEqual(replay.results[0], result)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(msgspec.json.encode(hints), original)
            # An explicit accepted claim has its own lifetime; aborting the
            # inference request does not undo that already completed action.
            service.finish_request(req, aborted=True)
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_id)["data"][
                    "remaining_ttl"
                ],
                7,
            )
            missing_selector = msgspec.to_builtins(
                SessionTarget(namespace="ns", session_id="removed")
            )
            renewed = run_pin(
                service,
                caller="request",
                claim_id=claim_id,
                mode="renew",
                revision=2,
                message="renew",
                selection=missing_selector,
            )
            self.assertEqual(renewed["public_result"]["claim"]["remaining_ttl"], 10)
            released = run_pin(
                service,
                caller="request",
                claim_id=claim_id,
                mode="release",
                revision=3,
                message="release",
                selection=missing_selector,
            )
            self.assertEqual(released["public_result"]["claim"]["state"], "released")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_request_cached_sampling_fanout_waits_for_checkpoint_and_has_separate_claims(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            hints = pin_hints(service, selection="request_cached")
            independent = run_pin(
                service, selection="request_cached", message="unbound"
            )
            self.assertEqual(independent["update"]["code"], "invalid_payload")
            original = msgspec.json.encode(hints)
            reqs = [
                service_fixtures.serving_request(),
                service_fixtures.serving_request(),
            ]
            for req in reqs:
                service.attach(req, hints)
            contexts = [req.kv_control_context for req in reqs]
            self.assertNotEqual(contexts[0].binding_id, contexts[1].binding_id)
            for context in contexts:
                queued = service.control.registry.query(context, "pin").results[0]
                self.assertEqual(queued.status, Status.QUEUED)
                self.assertIsNone(queued.public_result.claim)
                self.assertFalse(queued.public_result.accepted_scope)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            service.finish_request(reqs[0], checkpointed=True, committed_len=4)
            self.assertEqual(
                service.control.registry.query(contexts[0], "pin").results[0].status,
                Status.QUEUED,
            )
            self.assertTrue(service.control.submit(hints, contexts[0]).replayed)
            self.clock.now = 107
            for req in reqs:
                req.finished_reason = FINISH_LENGTH(1)
                req.refresh_fill_ids()
                service.finish_request(req, checkpointed=True, committed_len=4)
            completed = [
                service.control.registry.query(context, "pin").results[0]
                for context in contexts
            ]
            claim_ids = [result.public_result.claim.claim_id for result in completed]
            self.assertNotEqual(*claim_ids)
            self.assertTrue(
                all(result.status == Status.COMPLETED for result in completed)
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 2)
            self.assertEqual(
                service.control.executor.protection.registry.protected_tokens, 4
            )
            self.assertEqual(service.next_deadline, 117)
            self.assertFalse(service.has_pending)
            self.assertEqual(msgspec.json.encode(hints), original)
            # History discovery keeps each sampling incarnation identifiable.
            lookup = dispatch(
                service,
                {
                    "kind": "lookup",
                    "context": msgspec.to_builtins(service.control.context("request")),
                    "message_id": "pin",
                },
            )
            self.assertEqual(
                {
                    r["results"][0]["public_result"]["claim"]["claim_id"]
                    for r in lookup["data"]
                },
                set(claim_ids),
            )
            self.clock.now = 109
            self.assertEqual(
                service.control.submit(hints, contexts[0]).results[0], completed[0]
            )
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_ids[0])["data"][
                    "remaining_ttl"
                ],
                8,
            )
            missing = msgspec.to_builtins(
                SessionTarget(namespace="ns", session_id="removed")
            )
            attached_renew = pin_hints(
                service,
                mode="renew",
                revision=2,
                message="renew",
                ttl=20,
                selection=missing,
            )
            renewed = service.control.submit(attached_renew, contexts[0]).results[0]
            self.assertEqual(renewed.public_result.claim.claim_id, claim_ids[0])
            self.assertEqual(renewed.public_result.claim.remaining_ttl, 20)
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_ids[1])["data"][
                    "remaining_ttl"
                ],
                8,
            )
            self.clock.now = 110
            self.assertEqual(
                service.control.submit(attached_renew, contexts[0]).results[0], renewed
            )
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_ids[0])["data"][
                    "remaining_ttl"
                ],
                19,
            )
            attached_release = pin_hints(
                service,
                mode="release",
                revision=3,
                message="release",
                selection=missing,
            )
            released = service.control.submit(attached_release, contexts[0]).results[0]
            self.assertEqual(released.public_result.claim.claim_id, claim_ids[0])
            self.assertEqual(
                query_claim(service, caller="request", claim_id=claim_ids[1])["data"][
                    "state"
                ],
                "active",
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(cache.evict_host(100), 0)
            run_pin(
                service,
                caller="request",
                claim_id=claim_ids[1],
                mode="release",
                revision=2,
                message="other-release",
                selection=missing,
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(cache.evict_host(100), 4)
            cache.sanity_check()

    def test_request_cached_abort_failed_checkpoint_and_absent_host_never_activate(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            transfer_fixtures.seed(cache)
            hints = pin_hints(service, selection="request_cached")
            reqs = [service_fixtures.serving_request() for _ in range(3)]
            for req in reqs:
                service.attach(req, hints)
            contexts = [req.kv_control_context for req in reqs]
            service.finish_request(reqs[0], aborted=True)
            for req in reqs[1:]:
                req.finished_reason = FINISH_LENGTH(1)
                req.refresh_fill_ids()
            service.finish_request(reqs[1], checkpointed=False, committed_len=4)
            service.finish_request(reqs[2], checkpointed=True, committed_len=4)
            results = [
                service.control.registry.query(context, "pin").results[0]
                for context in contexts
            ]
            self.assertEqual(
                [result.status for result in results],
                [Status.CANCELLED, Status.CANCELLED, Status.MISSING],
            )
            self.assertTrue(results[2].public_result.missing_scope)
            self.assertTrue(
                all(result.public_result.claim is None for result in results)
            )
            self.assertEqual(
                service.control.executor.protection.registry.active_claims, 0
            )
            self.assertFalse(service.has_pending)
            # Late completion/replay of the cancelled binding cannot activate it.
            service.control.request_event(
                contexts[0], "success", committed=cache_fixtures.target()
            )
            self.assertTrue(service.control.submit(hints, contexts[0]).replayed)
            self.assertEqual(
                service.control.registry.query(contexts[0], "pin").results[0].status,
                Status.CANCELLED,
            )
            self.assertFalse(service.has_background_work)
            cache.sanity_check()

    def test_request_cached_native_checkpoint_installs_independent_host_hold_before_unlock(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner, tokens=(1, 2, 3, 4, 5))
            req = service_fixtures.serving_request()
            req.output_ids = array("q", [5])
            req.finished_reason = FINISH_LENGTH(1)
            cache.req_to_token_pool.alloc([req])
            slots = cache.token_to_kv_pool_allocator.alloc(5)
            cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 5
            service.attach(req, pin_hints(service, selection="request_cached"))
            context = req.kv_control_context
            observed = []
            pin = service.control.executor.pin

            def inspect(**kwargs):
                observed.append(
                    (
                        req.kv.holds_kv,
                        req.lock is not None,
                        tuple(kwargs["scope"].branches[0].content.token_ids),
                    )
                )
                return pin(**kwargs)

            with patch.object(service.control.executor, "pin", side_effect=inspect):
                release_kv_cache(req, cache, checkpoint=True)
            self.assertEqual(observed, [(True, True, (1, 2, 3, 4, 5))])
            result = service.control.registry.query(context, "pin").results[0]
            self.assertEqual(result.status, Status.COMPLETED)
            self.assertEqual(
                result.public_result.claim.actual_scope[0].coverage.end_token, 5
            )
            self.assertFalse(req.kv.holds_kv)
            self.assertIsNone(req.lock)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(cache.evict_host(100), 0)
            run_pin(
                service,
                caller="request",
                claim_id=result.public_result.claim.claim_id,
                mode="release",
                revision=2,
                message="release",
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_http_pin_lifetime_query_and_same_message_replay(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)

            class Manager:
                async def kv_control(self, command):
                    return dispatch(service, command)

            app = FastAPI()
            install_routes(app, lambda: Manager())

            async def exercise():
                async with AsyncClient(
                    transport=ASGITransport(app), base_url="http://owner"
                ) as client:
                    cap = (await client.get("/v1/kv/capabilities")).json()["data"][
                        "public_protocol"
                    ]
                    self.assertIn("sglang.pin", cap["actions"])
                    self.assertEqual(
                        cap["pin_protections"][0]["condition"], "eviction_exclusion"
                    )
                    request = pin_command(service)
                    reply = (await client.post("/v1/kv/control", json=request)).json()[
                        "data"
                    ]
                    result = reply["results"][0]
                    self.assertEqual(result["update"]["status"], "completed")
                    self.assertEqual(
                        result["public_result"]["claim"]["state"], "active"
                    )
                    self.assertTrue(result["public_result"]["accepted_scope"])
                    self.assertFalse(service.has_pending)
                    self.assertTrue(service.has_background_work)
                    self.clock.now = 103
                    replay = (await client.post("/v1/kv/control", json=request)).json()[
                        "data"
                    ]
                    self.assertTrue(replay["replayed"])
                    self.assertEqual(replay["results"][0], result)
                    query = {
                        "kind": "claim",
                        "context": request["context"],
                        "claim_id": "pin",
                    }
                    state = (await client.post("/v1/kv/control", json=query)).json()[
                        "data"
                    ]
                    self.assertEqual(state["remaining_ttl"], 7)
                    self.assertEqual(
                        state["effective_protection"], "eviction_exclusion"
                    )
                    self.assertEqual(
                        state["actual_scope"][0]["coverage"]["end_token"], 4
                    )
                    renewed = (
                        await client.post(
                            "/v1/kv/control",
                            json=pin_command(
                                service,
                                mode="renew",
                                revision=2,
                                message="renew",
                                ttl=20,
                            ),
                        )
                    ).json()["data"]["results"][0]
                    self.assertEqual(
                        renewed["public_result"]["claim"]["remaining_ttl"], 20
                    )
                    self.assertEqual(service.next_deadline, 123)
                    released = (
                        await client.post(
                            "/v1/kv/control",
                            json=pin_command(
                                service, mode="release", revision=3, message="release"
                            ),
                        )
                    ).json()["data"]["results"][0]
                    self.assertEqual(
                        released["public_result"]["claim"]["state"], "released"
                    )
                    terminal = (await client.post("/v1/kv/control", json=query)).json()[
                        "data"
                    ]
                    self.assertTrue(terminal["tombstone"])
                    self.assertIsNone(terminal["remaining_ttl"])
                    self.assertIsNone(terminal["effective_protection"])
                    self.assertEqual(
                        run_pin(service, mode="renew", revision=2, message="late")[
                            "update"
                        ]["code"],
                        "stale_revision",
                    )
                    self.assertEqual(
                        run_pin(service, mode="renew", revision=4, message="revive")[
                            "update"
                        ]["code"],
                        "released",
                    )

            asyncio.run(exercise())
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_runtime_shared_eviction_capacity_and_ttl_are_independent(self):
        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            protection_limits=ProtectionLimits(max_protected_tokens=4),
        )
        with transfer_fixtures.hicache(self, config=config) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            ordinary = host_copy(owner, (8, 9, 10, 11), message="other")
            self.assertEqual(run_pin(service)["update"]["status"], "completed")
            self.assertEqual(
                run_pin(service, caller="B", message="b", ttl=20)["update"]["status"],
                "completed",
            )
            denied = run_pin(
                service,
                caller="C",
                message="c",
                selection=msgspec.to_builtins(cache_fixtures.target((8, 9, 10, 11))),
            )
            self.assertEqual(denied["update"]["status"], "deferred")
            self.assertFalse(denied["public_result"]["accepted_scope"])
            self.assertEqual(ordinary.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(cache.evict_host(100), 4)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.clock.now = 111
            service.progress()
            self.assertEqual(query_claim(service)["data"]["state"], "expired")
            self.assertEqual(
                query_claim(service, caller="B")["data"]["remaining_ttl"], 9
            )
            self.assertEqual(cache.evict_host(100), 0)
            run_pin(service, caller="B", mode="release", revision=2, message="done")
            self.assertEqual(cache.evict_host(100), 4)
            cache.sanity_check()

    def test_task_ownership_is_explicit_and_cleanup_only_releases_finishing_caller(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            for name in ("selected", "owner", "sibling"):
                service.control.index.associate(
                    Associate(
                        session_id=name,
                        continuation_id="b",
                        expected_revision=0,
                        target=cache_fixtures.target(),
                    )
                )
            task = service.control.index.task_identity("owner")
            sibling = service.control.index.task_identity("sibling")
            selection = msgspec.to_builtins(
                SessionTarget(namespace="ns", session_id="selected")
            )
            run_pin(service, owner_task=task, selection=selection)
            run_pin(service, caller="B", owner_task=task, selection=selection)
            run_pin(service, owner_task=sibling, claim_id="sibling", message="sibling")
            run_pin(service, claim_id="unowned", message="unowned")

            def finish(name):
                return self.submit(
                    service.control,
                    "finish_task",
                    {
                        "session_id": name,
                        "expected_revision": service.control.index.session_revision(
                            name
                        ),
                    },
                    caller="A",
                )

            self.assertEqual(finish("selected").status, Status.COMPLETED)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 4)
            renewed = run_pin(
                service,
                claim_id="unowned",
                message="renew-without-session",
                mode="renew",
                revision=2,
                selection=selection,
            )
            self.assertEqual(renewed["public_result"]["claim"]["state"], "active")
            self.assertEqual(finish("owner").status, Status.COMPLETED)
            self.assertEqual(
                query_claim(service, owner_task=task)["data"]["state"], "released"
            )
            self.assertEqual(
                query_claim(service, caller="B", owner_task=task)["data"]["state"],
                "active",
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 3)
            late = run_pin(service, owner_task=task, claim_id="new", message="late")
            self.assertEqual(late["update"]["code"], "stale_revision")
            # Released task/session selectors no longer resolve; release still
            # finds the independent key and retires B's protection.
            run_pin(
                service,
                caller="B",
                owner_task=task,
                selection=selection,
                mode="release",
                revision=2,
                message="b-release",
            )
            run_pin(
                service,
                owner_task=sibling,
                claim_id="sibling",
                mode="release",
                revision=2,
                message="sibling-release",
            )
            run_pin(
                service,
                claim_id="unowned",
                mode="release",
                revision=3,
                message="unowned-release",
            )
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_idle_expiry_publishes_after_operation_completion_without_early_scans(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            run_pin(service, ttl=2)
            protection = service.control.executor.protection
            with patch.object(
                protection.registry,
                "expire",
                side_effect=AssertionError("early full scan"),
            ):
                for _ in range(10):
                    service.progress()
            self.clock.now = 102
            service.progress()
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            events = dispatch(
                service,
                {
                    "kind": "claim_events",
                    "context": msgspec.to_builtins(service.control.context("A")),
                },
            )
            self.assertEqual(
                [e["claim"]["state"] for e in events["data"]["items"]],
                ["active", "expired"],
            )
            self.assertFalse(service.has_background_work)
            self.assertIsNone(service.next_deadline)
            self.assertEqual(query_claim(service)["data"]["state"], "expired")
            cache.sanity_check()

    def test_reset_and_weights_invalidate_before_host_slots_or_namespace_change(self):
        with transfer_fixtures.hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            run_pin(service)
            old_epoch = service.control.registry.engine_epoch
            request_lock = cache.lock(node.id)
            try:
                with self.assertRaises(ControlError):
                    service.weights.begin("v2")
                self.assertEqual(query_claim(service)["data"]["state"], "active")
                self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            finally:
                cache.unlock(request_lock)
            service.weights.begin("v2")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertNotEqual(service.control.registry.engine_epoch, old_epoch)
            self.assertEqual(query_claim(service)["error"]["code"], "missing")
            service.weights.finish(True, synchronized=True)
            node = host_copy(owner, message="new-weights")
            run_pin(service, message="new-pin")
            previous = service.control.executor.protection
            cache.reset()
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(
                previous.get(caller_id="A", claim_id="pin", owner_task=None).state,
                "invalidated",
            )
            service.reset_after_cache_flush()
            self.assertFalse(service.has_background_work)
            self.assertEqual(query_claim(service)["error"]["code"], "missing")

    def test_raw_salt_is_encoded_once_and_storage_detach_preserves_l2_claims(self):
        with transfer_fixtures.hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            node = cache_fixtures.insert(
                cache, (1, 2, 3, 4), salt=service.cache_salt("raw")
            )
            copied = transfer_fixtures.transfer(
                service, selection=cache_fixtures.target(cache_salt="raw")
            )
            service.progress()
            dma.finish()
            service.progress()
            self.assertEqual(
                transfer_fixtures.current(service, copied).status, Status.COMPLETED
            )
            pinned = run_pin(service, salt="raw")
            self.assertEqual(pinned["public_result"]["claim"]["state"], "active")
            ok, reason = cache.detach_storage_backend()
            self.assertTrue(ok, reason)
            service.capabilities()
            self.assertEqual(query_claim(service)["data"]["state"], "active")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            run_pin(service, mode="release", revision=2, message="release", salt="raw")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_idle_sleep_timeout_stops_at_protection_deadline(self):
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler import Scheduler

        with transfer_fixtures.hicache(self) as owner:
            _, _, service, _ = owner
            host_copy(owner)
            run_pin(service, ttl=0.25)
            sleeps = []
            scheduler = SimpleNamespace(
                kv_control_service=service,
                idle_sleeper=SimpleNamespace(
                    maybe_sleep=lambda **kwargs: sleeps.append(kwargs["timeout_ms"])
                ),
            )
            with patch(
                "sglang.srt.managers.scheduler.time.monotonic", return_value=100
            ):
                Scheduler.maybe_sleep_on_idle(scheduler)
            self.assertEqual(sleeps, [250])
            self.assertFalse(service.has_pending)
            self.clock.now = 101
            service.progress()
            self.assertEqual(query_claim(service)["data"]["state"], "expired")

    def test_bounded_claim_events_signal_gap_and_filter_other_callers(self):
        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            protection_limits=ProtectionLimits(max_events=2),
        )
        with transfer_fixtures.hicache(self, config=config) as owner:
            _, _, service, _ = owner
            host_copy(owner)
            run_pin(service)
            run_pin(service, mode="renew", revision=2, message="renew")
            run_pin(service, caller="B", message="b")
            reply = dispatch(
                service,
                {
                    "kind": "claim_events",
                    "context": msgspec.to_builtins(service.control.context("A")),
                },
            )["data"]
            self.assertTrue(reply["gap"])
            self.assertEqual([e["claim"]["caller_id"] for e in reply["items"]], ["A"])
            self.assertEqual(reply["next_sequence"], 3)
            self.assertEqual(query_claim(service)["data"]["state"], "active")
            run_pin(service, mode="release", revision=3, message="release")
            run_pin(
                service, caller="B", mode="release", revision=2, message="b-release"
            )

    def test_missing_incomplete_and_l3_rejection_never_report_active_scope(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, dma = owner
            missing = run_pin(service)
            self.assertEqual(missing["update"]["status"], "missing")
            self.assertTrue(missing["public_result"]["missing_scope"])
            self.assertIsNone(missing["public_result"]["claim"])
            node = transfer_fixtures.seed(cache)
            result = transfer_fixtures.transfer(service)
            service.progress()
            incomplete = run_pin(service, message="pending")
            self.assertEqual(incomplete["update"]["code"], "incomplete")
            self.assertFalse(incomplete["public_result"]["accepted_scope"])
            dma.finish()
            service.progress()
            self.assertEqual(
                transfer_fixtures.current(service, result).status, Status.COMPLETED
            )
            request = pin_command(service, message="l3")
            request["envelope"]["actions"][0]["payload"]["location"] = {
                "tier": "storage",
                "backend": "mooncake",
            }
            denied = dispatch(service, request)["data"]["results"][0]
            self.assertEqual(denied["update"]["status"], "unsupported")
            self.assertIsNone(denied["public_result"]["claim"])
            legacy = pin_command(service, message="legacy")
            legacy["envelope"]["actions"][0]["action_type"] = "sglang.set_priority"
            denied = dispatch(service, legacy)["data"]["results"][0]
            self.assertEqual(denied["update"]["status"], "unsupported")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
