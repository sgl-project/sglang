# SPDX-License-Identifier: Apache-2.0
"""Control replay and local cache ordering, with real cache/DMA/socket fixtures.

The separate file owns publisher/replay teardown and feedback capacity fixtures;
it reuses the native cache and request lifecycle fixtures without changing them.
"""

import threading
import time
import unittest
import uuid
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import test_kv_control_service as service_fixtures
import test_kv_control_transfers as transfer_fixtures
import zmq
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    KVEventBatch,
    WatermarkedKVEventBatch,
    ZmqEventPublisher,
)
from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.managers.kv_control.feedback import (
    ControlEventBatch,
    FeedbackLimits,
    FeedbackPublisherConfig,
)
from sglang.srt.managers.kv_control.service import ServiceConfig, dispatch
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.utils.network import get_free_port
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

    def test_local_association_and_partial_demote_preserve_coordinates_and_actual_effects(
        self,
    ):
        with transfer_fixtures.hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            ancestor = transfer_fixtures.seed(cache, (1, 2))
            leaf = transfer_fixtures.seed(cache)
            transfer_fixtures.backup_prefix(cache, dma, leaf, evict=True)
            associated = self.submit(
                service.control,
                "associate",
                {
                    "session_id": "task",
                    "continuation_id": "branch",
                    "expected_revision": 0,
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                },
                caller="A",
                message="associate",
            )
            self.assertEqual(associated.status, Status.COMPLETED)
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
            local = [
                e["payload"]["continuation"]
                for e in page["items"]
                if e["kind"] == "association" and "continuation" in e["payload"]
            ]
            self.assertEqual(local[0]["task"]["session_id"], "task")
            self.assertEqual(local[0]["continuation_id"], "branch")
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

    def test_location_publisher_attaches_same_owner_cache_watermark_without_control_events(
        self,
    ):
        with transfer_fixtures.hicache(self) as owner:
            _, cache, service, _ = owner
            cache.tree_core.kv_events.enabled = True
            cache.tree_core.kv_events.record_all_cleared()
            batches = []

            class Sink:
                def publish(self, batch):
                    batches.append(batch)

            publisher = SchedulerKvEventsPublisher.__new__(SchedulerKvEventsPublisher)
            publisher.enable_kv_cache_events = True
            publisher.tree_cache = cache
            publisher.kv_event_publisher = Sink()
            publisher.publish_kv_events()
            self.assertIsInstance(batches[0], WatermarkedKVEventBatch)
            self.assertEqual(
                batches[0].cache_watermark,
                msgspec.to_builtins(service.cache_watermark()),
            )
            legacy = msgspec.msgpack.decode(
                msgspec.msgpack.encode(batches[0]), type=KVEventBatch
            )
            self.assertTrue(
                all(isinstance(event, AllBlocksCleared) for event in legacy.events)
            )

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
            self.assertIsNone(service.control.index.task_identity("router-session"))

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

    def test_optional_publisher_uses_own_socket_replay_and_nonblocking_bounded_outbox(
        self,
    ):
        channel = "inproc://control-" + uuid.uuid4().hex
        replay_endpoint = f"tcp://127.0.0.1:{get_free_port()}"
        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            feedback_limits=FeedbackLimits(max_events=4),
            feedback_publisher=FeedbackPublisherConfig(
                endpoint=channel,
                replay_endpoint=replay_endpoint,
                buffer_steps=2,
                max_queue_size=1,
            ),
        )
        old_channel = "inproc://legacy-" + uuid.uuid4().hex
        legacy = ZmqEventPublisher(attn_dp_rank=0, endpoint=old_channel)
        context = zmq.Context.instance()
        sub, old_sub, dealer = (
            context.socket(zmq.SUB),
            context.socket(zmq.SUB),
            context.socket(zmq.DEALER),
        )
        for socket in (sub, old_sub):
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.RCVTIMEO, 3000)
        dealer.setsockopt(zmq.RCVTIMEO, 3000)
        old_sub.connect(old_channel)
        publisher = None
        try:
            with transfer_fixtures.hicache(self, config=config, storage=True) as owner:
                _, cache, service, dma = owner
                publisher = service.feedback.publisher
                sub.connect(channel)
                dealer.connect(replay_endpoint)
                time.sleep(0.05)  # ZMQ subscription handshake, external boundary.
                node = transfer_fixtures.seed(cache)
                transfer_fixtures.backup_prefix(cache, dma, node, evict=True)
                operation = transfer_fixtures.public_transfer(
                    service, "demote", "host", "storage", "save", caller="A"
                )
                with patch.object(publisher, "try_publish", return_value=False):
                    transfer_fixtures.wait(service, operation)
                self.assertTrue(service.has_background_work)
                self.assertGreater(service.next_deadline, self.clock.now)
                self.clock.now += 0.02
                service.progress()
                publisher._event_queue.join()
                frames = sub.recv_multipart()
                batch = msgspec.msgpack.decode(frames[2], type=ControlEventBatch)
                self.assertEqual(frames[0], b"sglang.control.v1")
                self.assertEqual(
                    batch.source_epoch,
                    service.capabilities()["feedback"]["source_epoch"],
                )
                self.assertEqual(
                    service.capabilities()["feedback"]["endpoint"], channel
                )
                self.assertEqual(
                    service.capabilities()["feedback"]["replay_endpoint"],
                    replay_endpoint,
                )
                self.assertEqual(
                    batch.events[-1].payload["update"]["status"], "completed"
                )
                self.assertEqual(
                    transfer_fixtures.current(service, operation).status,
                    Status.COMPLETED,
                )
                self.assertFalse(old_sub.poll(50))
                legacy.publish(KVEventBatch(ts=0, events=[AllBlocksCleared()]))
                self.assertEqual(
                    msgspec.msgpack.decode(
                        old_sub.recv_multipart()[2], type=KVEventBatch
                    ).events,
                    [AllBlocksCleared()],
                )
                for revision in range(3):
                    service.feedback.append("association", "A", {"revision": revision})
                    service.progress()
                    publisher._event_queue.join()
                dealer.send_multipart([b"", (0).to_bytes(8, "big")])
                replies = [dealer.recv_multipart() for _ in range(3)]
                self.assertEqual(
                    [int.from_bytes(r[2], "big", signed=True) for r in replies],
                    [2, 3, -1],
                )
                replayed = [
                    msgspec.msgpack.decode(r[3], type=ControlEventBatch)
                    for r in replies[:2]
                ]
                self.assertEqual(
                    [b.events[-1].payload["revision"] for b in replayed], [1, 2]
                )
                self.assertTrue(events(service)["gap"])
            self.assertFalse(publisher._thread.is_alive())
        finally:
            sub.close(linger=0)
            old_sub.close(linger=0)
            dealer.close(linger=0)
            legacy.shutdown()


if __name__ == "__main__":
    unittest.main()
