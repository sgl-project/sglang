# SPDX-License-Identifier: Apache-2.0
"""HTTP control replay and local cache ordering using native cache fixtures."""

import threading
import unittest
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import test_kv_control_service as service_fixtures
import test_kv_control_transfers as transfer_fixtures
from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.managers.kv_control.feedback import FeedbackLimits
from sglang.srt.managers.kv_control.service import ServiceConfig, dispatch
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def events(service, *, caller="A", cursor=None, limit=256):
    reply = dispatch(
        service,
        {
            "kind": "events",
            "context": msgspec.to_builtins(service.control.context(caller)),
            "cursor": cursor,
            "limit": limit,
        },
    )
    assert reply["success"], reply
    return reply["data"]


class TestKVControlFeedback(cache_fixtures.KVCacheTestCase):
    def test_pin_completed_receipt_and_idle_expiry_share_general_feedback(self):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, dma = owner
            node = transfer_fixtures.seed(cache)
            transfer_fixtures.backup_prefix(cache, dma, node)
            done = self.submit(
                service.control,
                "pin",
                {
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                    "mode": "acquire",
                    "location": {"tier": "host"},
                    "claim_id": "keep",
                    "revision": 1,
                    "ttl_seconds": 1,
                },
                caller="A",
                message="pin",
            )
            self.assertEqual(done.status, Status.COMPLETED)
            page = events(service)
            terminal = [
                e
                for e in page["items"]
                if e["kind"] == "operation"
                and e["payload"]["action_type"] == "sglang.pin"
                and e["payload"]["update"]["status"] == "completed"
            ]
            self.assertEqual(len(terminal), 1)
            self.assertEqual(
                terminal[0]["payload"]["public_result"]["claim"]["state"], "active"
            )
            self.assertFalse(service.has_pending)
            self.clock.now += 2
            expired = events(service, cursor=page["next_cursor"])
            self.assertEqual(
                [
                    e["payload"]["state"]
                    for e in expired["items"]
                    if e["kind"] == "claim"
                ],
                ["expired"],
            )
            self.assertFalse(service.has_background_work)
            cache.sanity_check()

    def test_partial_demote_feedback_reports_actual_effects(
        self,
    ):
        with transfer_fixtures.hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            ancestor = transfer_fixtures.seed(cache, (1, 2))
            leaf = transfer_fixtures.seed(cache)
            transfer_fixtures.backup_prefix(cache, dma, leaf, evict=True)
            lock = cache.lock(ancestor.id)
            try:
                result = transfer_fixtures.public_transfer(
                    service, "demote", "host", "storage", "partial", caller="A"
                )
                done = transfer_fixtures.wait(service, result)
            finally:
                cache.unlock(lock)
            self.assertEqual(done.status, Status.PARTIAL)
            page = events(service)
            partial = [
                e["payload"]
                for e in page["items"]
                if e["kind"] == "operation"
                and e["payload"]["update"]["status"] == "partial"
            ]
            self.assertEqual(
                partial[-1]["public_result"]["preserved_scope"],
                msgspec.json.decode(
                    msgspec.json.encode(done.public_result.preserved_scope)
                ),
            )
            self.assertEqual(
                partial[-1]["public_result"]["reclaimed_scope"],
                msgspec.json.decode(
                    msgspec.json.encode(done.public_result.reclaimed_scope)
                ),
            )
            self.assertEqual(
                partial[-1]["cache_watermark"],
                msgspec.to_builtins(done.cache_watermark),
            )
            cache.sanity_check()

    def test_owner_watermark_orders_real_ready_remove_and_scoped_reconciliation(self):
        with transfer_fixtures.hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            self.assertFalse(cache.tree_core.kv_events.enabled)
            before = service.cache_watermark()
            node = transfer_fixtures.seed(cache)
            self.assertGreater(service.cache_watermark().version, before.version)
            transfer_fixtures.persist_prefix(cache, service, dma, node)
            self.assertEqual(cache.evict_host(4), 4)
            result = transfer_fixtures.transfer(service)
            complete = transfer_fixtures.wait(service, result)
            node = transfer_fixtures.prefix_path(cache, service)[-1][0]
            self.assertEqual(complete.status, Status.COMPLETED)
            ready = complete.cache_watermark
            page = events(service)
            outcomes = [e for e in page["items"] if e["kind"] == "operation"]
            self.assertEqual(outcomes[-1]["payload"]["update"]["status"], "completed")
            self.assertEqual(
                outcomes[-1]["payload"]["cache_watermark"], msgspec.to_builtins(ready)
            )
            self.assertTrue(
                all(
                    a["sequence"] < b["sequence"]
                    for a, b in zip(page["items"], page["items"][1:])
                )
            )
            self.assertEqual(cache.evict_host(100), 4)
            self.assertIsNone(
                node.component_data[cache_fixtures.ComponentType.FULL].host_value
            )
            self.assertGreater(service.cache_watermark().version, ready.version)
            scoped = dispatch(
                service,
                {
                    "kind": "residency",
                    "context": msgspec.to_builtins(service.control.context("A")),
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                    "location": {"tier": "host"},
                },
            )["data"]
            self.assertFalse(scoped["items"])
            self.assertTrue(scoped["missing"])
            self.assertEqual(scoped["coverage_semantics"], "requested_scope_only")
            self.assertGreater(scoped["cache_watermark"]["version"], ready.version)
            # The historical result is unchanged even when its replica is gone.
            self.assertEqual(service.control.registry.get(result.ref), complete)
            self.assertEqual(cache.take_events(), [])
            cache.sanity_check()

    def test_cancellation_progress_and_caller_filtered_replay_have_distinct_sequences(
        self,
    ):
        with transfer_fixtures.hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = transfer_fixtures.seed(cache)
            transfer_fixtures.backup_prefix(cache, dma, node, evict=True)
            entered, release = threading.Event(), threading.Event()
            controller = cache.cache_controller
            page_set = controller.page_set_func

            def blocked(*args, **kwargs):
                saved = page_set(*args, **kwargs)
                entered.set()
                assert release.wait(5)
                return saved

            with patch.object(controller, "page_set_func", blocked):
                a = transfer_fixtures.public_transfer(
                    service, "demote", "host", "storage", "a", caller="A"
                )
                try:
                    self.assertTrue(entered.wait(5))
                    # Same physical work, separately owned receipt.
                    b = transfer_fixtures.public_transfer(
                        service, "demote", "host", "storage", "b", caller="B"
                    )
                    service.control.cancel(b.ref)
                    cancelled = service.control.cancel(a.ref)
                    self.assertEqual(cancelled.status, Status.CANCELLING)
                    before = events(service)
                finally:
                    release.set()
                transfer_fixtures.wait(service, a)
            after = events(service, cursor=before["next_cursor"])
            self.assertTrue(
                all(
                    e["caller_id"] in (None, "A")
                    for e in before["items"] + after["items"]
                )
            )
            states = [
                e["payload"]["update"]["status"]
                for e in before["items"] + after["items"]
                if e["kind"] == "operation"
            ]
            self.assertIn("received", states)
            self.assertIn("running", states)
            self.assertIn("cancelling", states)
            self.assertEqual(states[-1], "cancelled")
            final = service.control.registry.get(a.ref)
            self.assertEqual(
                final.update.target_outcomes[0].completed_steps, ("destination_ready",)
            )
            replay = events(service, cursor=before["next_cursor"])
            self.assertEqual(replay["items"], after["items"])
            self.assertEqual(replay["latest_sequence"], after["latest_sequence"])
            self.assertNotEqual(
                after["source_epoch"], service.control.registry.engine_epoch
            )
            cache.sanity_check()

    def test_retention_and_oversized_events_report_gap_without_fabricating_outcome(
        self,
    ):
        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            feedback_limits=FeedbackLimits(
                max_events=2, max_bytes=4096, max_event_bytes=2048
            ),
        )
        with transfer_fixtures.hicache(self, config=config, storage=True) as owner:
            _, _, service, _ = owner
            start = events(service)["next_cursor"]
            for n in range(4):
                service.feedback.append(
                    "association", "A", {"revision": n, "association_state": "unknown"}
                )
            page = events(service, cursor=start, limit=1)
            self.assertTrue(page["gap"])
            self.assertEqual(len(page["items"]), 1)
            self.assertEqual(page["items"][0]["payload"]["revision"], 2)
            following = events(service, cursor=page["next_cursor"])
            self.assertFalse(following["gap"])
            self.assertEqual(following["items"][0]["payload"]["revision"], 3)
            service.feedback.append("operation", "A", {"reason": "x" * 3000})
            lost = events(service, cursor=following["next_cursor"])
            self.assertTrue(lost["gap"])
            self.assertFalse(lost["items"])
            self.assertEqual(lost["recovery"], "query_known_scopes")

    def test_pure_hit_request_attribution_is_explicit_and_sampling_incarnations_do_not_merge(
        self,
    ):
        with service_fixtures.service_owner(self) as owner:
            cache, service = owner
            cache_fixtures.insert(cache, (1, 2, 3, 4))
            initial = service.cache_watermark()
            reqs = [service_fixtures.serving_request() for _ in range(2)]
            for req in reqs:
                service.prepare_request(req)
                # Existing request accounting is the producer boundary. No
                # source tier is inferred from transfer/route expectations.
                req.cached_tokens = 4
                req.cached_tokens_device = 4
                req._cache_breakdown_computed = True
                req.refresh_fill_ids()
            contexts = [req.kv_control_feedback_context for req in reqs]
            self.assertNotEqual(contexts[0].binding_id, contexts[1].binding_id)
            association = {
                "request_id": "application-r",
                "session_id": "router-session",
                "session_incarnation": "router-incarnation",
                "branch_id": "branch-a",
                "branch_revision": 7,
            }
            command = {
                "kind": "bind_request",
                "context": msgspec.to_builtins(service.control.context("request")),
                "binding_id": contexts[0].binding_id,
                "association": association,
            }
            self.assertTrue(dispatch(service, command)["success"])
            self.assertTrue(dispatch(service, command)["success"])
            foreign = {
                **command,
                "context": msgspec.to_builtins(service.control.context("other")),
            }
            self.assertEqual(dispatch(service, foreign)["error"]["code"], "missing")
            for req in reqs:
                req.finished_reason = FINISH_LENGTH(1)
                service.finish_request(req, committed_len=4, checkpointed=True)
            page = events(service, caller="request")
            uses = [e["payload"] for e in page["items"] if e["kind"] == "request_use"]
            self.assertEqual(len(uses), 2)
            self.assertEqual(uses[0]["association"], association)
            self.assertEqual(uses[1]["association_state"], "unknown")
            self.assertIsNone(uses[1]["association"])
            self.assertEqual([u["device_tokens"] for u in uses], [4, 4])
            self.assertTrue(all(u["tier_accounting"] == "measured" for u in uses))
            unknown = service_fixtures.serving_request()
            service.prepare_request(unknown)
            unknown.cached_tokens = 2
            unknown.finished_reason = FINISH_LENGTH(1)
            service.finish_request(unknown, aborted=True)
            unknown_use = [
                e["payload"]
                for e in events(service, caller="request")["items"]
                if e["kind"] == "request_use"
            ][-1]
            self.assertEqual(unknown_use["tier_accounting"], "unknown")
            self.assertIsNone(unknown_use["device_tokens"])
            self.assertIsNone(unknown_use["committed_prefix"])
            self.assertEqual(service.cache_watermark(), initial)
            self.assertEqual(cache.take_events(), [])

    def test_flush_restart_and_storage_detach_invalidate_only_owner_observations(self):
        with transfer_fixtures.hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            old_context = msgspec.to_builtins(service.control.context("A"))
            cursor = events(service)["next_cursor"]
            before = service.cache_watermark()
            ok, _ = cache.detach_storage_backend()
            self.assertTrue(ok)
            service.progress()
            page = events(service, cursor=cursor)
            self.assertEqual(page["items"][-1]["payload"]["tiers"], ["storage"])
            cache.reset()
            self.assertGreater(service.cache_watermark().generation, before.generation)
            service.reset_after_cache_flush()
            self.assertEqual(
                dispatch(service, {"kind": "events", "context": old_context})["error"][
                    "code"
                ],
                "stale_epoch",
            )
            restarted = events(service, cursor=cursor)
            self.assertTrue(restarted["gap"])
            self.assertNotEqual(restarted["source_epoch"], cursor["source_epoch"])
            self.assertEqual(
                restarted["cache_watermark"]["source_epoch"], before.source_epoch
            )


if __name__ == "__main__":
    unittest.main()
