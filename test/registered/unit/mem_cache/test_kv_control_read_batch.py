# SPDX-License-Identifier: Apache-2.0
"""Bounded read batching preserves existing owner, context and query semantics."""

import asyncio
import threading
import unittest
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import test_kv_control_service as service_fixtures
import test_kv_control_transfers as transfer_fixtures
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sglang.srt.managers.io_struct import (
    KVControlReqInput,
    KVControlReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.managers.kv_control.http import install_routes
from sglang.srt.managers.kv_control.service import dispatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def context(service, caller="A", **changes):
    return {**msgspec.to_builtins(service.control.context(caller)), **changes}


def batch(service, queries, **changes):
    return {
        "kind": "read_batch",
        "context": context(service),
        "queries": queries,
        **changes,
    }


def wire(value):
    return msgspec.json.decode(msgspec.json.encode(value))


class TestKVControlReadBatch(cache_fixtures.KVCacheTestCase):
    def test_schema_is_strict_bounded_read_only_and_non_recursive(self):
        with service_fixtures.service_owner(self) as (_, service):
            cap = {"kind": "capabilities"}
            bad = [
                batch(service, []),
                batch(service, [cap] * 9),
                batch(service, [cap], version="2.0"),
                batch(service, [cap], version=1.0),
                batch(service, [cap], atomic=True),
                {"kind": "read_batch", "queries": [cap]},
                batch(service, [{"kind": "capabilities", "extra": True}]),
                batch(service, [{"kind": "capabilities", "context": context(service)}]),
            ]
            for kind in (
                "submit",
                "cancel",
                "ack",
                "lookup",
                "read_batch",
                "capacity",
                "storage_targets",
                "associations",
                "claim_events",
                "bind_request",
                "future",
            ):
                bad.append(batch(service, [{"kind": kind}]))
            before = service.control.registry.capacity()
            with patch.object(service, "execute", wraps=service.execute) as execute:
                for body in bad:
                    with self.subTest(body=body):
                        reply = dispatch(service, body)
                        self.assertFalse(reply["success"])
                        self.assertEqual(reply["error"]["code"], "invalid_payload")
                execute.assert_not_called()
            for count in (1, 8):
                reply = dispatch(service, batch(service, [cap] * count))
                self.assertTrue(reply["success"], reply)
                self.assertEqual(reply["data"]["version"], "1.0")
                self.assertEqual(len(reply["data"]["results"]), count)
            self.assertEqual(service.control.registry.capacity(), before)
            self.assertEqual(
                service.capabilities()["read_batch"],
                {
                    "version": "1.0",
                    "max_queries": 8,
                    "commands": (
                        "capabilities",
                        "query",
                        "residency",
                        "claim",
                        "events",
                    ),
                },
            )

    def test_all_child_contexts_are_checked_before_any_query_executes(self):
        with service_fixtures.service_owner(self) as (_, service):
            for change in (
                {"caller_id": "B"},
                {"engine_epoch": "stale"},
                {"binding_id": "sample"},
                {"engine_id": "other"},
                {"namespace": "other"},
            ):
                with (
                    self.subTest(change=change),
                    patch.object(service, "progress") as progress,
                ):
                    reply = dispatch(
                        service,
                        batch(
                            service,
                            [
                                {"kind": "events", "context": context(service)},
                                {
                                    "kind": "query",
                                    "context": context(service, **change),
                                    "message_id": "missing",
                                },
                            ],
                        ),
                    )
                    self.assertEqual(reply["error"]["code"], "invalid_payload")
                    progress.assert_not_called()

    def test_owner_and_top_context_are_validated_even_for_capabilities_only(self):
        with service_fixtures.service_owner(self) as (_, service):
            for change, code in (
                ({"engine_id": "other"}, "wrong_engine"),
                ({"engine_epoch": "stale"}, "stale_epoch"),
                ({"namespace": "other"}, "incompatible_namespace"),
            ):
                with (
                    self.subTest(change=change),
                    patch.object(service, "capabilities") as read,
                ):
                    reply = dispatch(
                        service,
                        batch(
                            service,
                            [{"kind": "capabilities"}],
                            context=context(service, **change),
                        ),
                    )
                    self.assertEqual(reply["error"]["code"], code)
                    read.assert_not_called()
            replies = []
            thread = threading.Thread(
                target=lambda: replies.append(
                    dispatch(service, batch(service, [{"kind": "capabilities"}]))
                )
            )
            thread.start()
            thread.join()
            self.assertEqual(replies[0]["error"]["code"], "wrong_owner")

    def test_http_and_msgpack_preserve_order_and_existing_data_formats(self):
        with transfer_fixtures.hicache(self) as (_, cache, service, dma):
            transfer_fixtures.seed(cache)
            saved = transfer_fixtures.transfer(service)
            transfer_fixtures.stage(service, dma)
            self.assertEqual(
                transfer_fixtures.wait(service, saved).status, Status.COMPLETED
            )
            pinned = self.submit(
                service.control,
                "pin",
                {
                    "target": wire(cache_fixtures.target()),
                    "mode": "acquire",
                    "location": {"tier": "host"},
                    "claim_id": "keep",
                    "revision": 1,
                    "ttl_seconds": 10,
                },
                caller="A",
                message="pin",
            )
            self.assertEqual(pinned.status, Status.COMPLETED)
            service.progress()
            ctx = context(service)
            queries = [
                {"kind": "capabilities"},
                {"kind": "query", "context": ctx, "message_id": "pin"},
                {
                    "kind": "residency",
                    "context": ctx,
                    "target": msgspec.to_builtins(cache_fixtures.target()),
                    "location": {"tier": "host"},
                },
                {"kind": "claim", "context": ctx, "claim_id": "keep"},
                {"kind": "events", "context": ctx},
            ]
            expected = [wire(dispatch(service, query)["data"]) for query in queries]
            before = service.control.registry.capacity()

            class Manager:
                async def kv_control(self, command):
                    wire = msgpack_decode(
                        msgpack_encode(
                            KVControlReqInput(request_id="batch", command=command)
                        )
                    )
                    reply = KVControlReqOutput(
                        request_id=wire.request_id,
                        result=dispatch(service, wire.command),
                    )
                    return msgpack_decode(msgpack_encode(reply)).result

            app = FastAPI()
            install_routes(app, Manager)

            async def call():
                async with AsyncClient(
                    transport=ASGITransport(app), base_url="http://owner"
                ) as client:
                    response = await client.post(
                        "/v1/kv/control", json=batch(service, queries, version="1.0")
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        response.json()["data"], {"version": "1.0", "results": expected}
                    )
                    bad = await client.post(
                        "/v1/kv/control",
                        json=batch(
                            service,
                            [
                                {
                                    "kind": "query",
                                    "context": ctx,
                                    "message_id": "missing",
                                }
                            ],
                        ),
                    )
                    self.assertEqual(bad.status_code, 404)
                    self.assertEqual(bad.json()["error"]["code"], "missing")
                    self.assertNotIn("data", bad.json())
                    other = context(service, "B")
                    private = await client.post(
                        "/v1/kv/control",
                        json=batch(
                            service,
                            [{"kind": "claim", "context": other, "claim_id": "keep"}],
                            context=other,
                        ),
                    )
                    self.assertEqual(private.status_code, 404)
                    self.assertEqual(private.json()["error"]["code"], "missing")

            asyncio.run(call())
            self.assertEqual(service.control.registry.capacity(), before)
            self.clock.now += 11
            service.progress()
            self.assertEqual(
                service.control.executor.protection.registry.protected_tokens, 0
            )

    def test_child_errors_and_caller_binding_isolation_match_single_queries(self):
        with transfer_fixtures.hicache(self) as (_, cache, service, dma):
            transfer_fixtures.seed(cache)
            saved = transfer_fixtures.transfer(service)
            transfer_fixtures.stage(service, dma)
            self.assertEqual(
                transfer_fixtures.wait(service, saved).status, Status.COMPLETED
            )
            private = service_fixtures.envelope(message="private")
            submitted = dispatch(
                service,
                {
                    "kind": "submit",
                    "context": context(service),
                    "envelope": wire(private),
                },
            )
            self.assertTrue(submitted["success"], submitted)
            before = service.control.registry.capacity()
            for ctx, query in (
                (context(service), {"kind": "query", "message_id": "missing"}),
                (context(service), {"kind": "claim", "claim_id": "missing"}),
                (context(service, "B"), {"kind": "query", "message_id": "private"}),
                (
                    context(service, binding_id="sample"),
                    {"kind": "query", "message_id": "private"},
                ),
                (
                    context(service, binding_id="sample"),
                    {"kind": "claim", "claim_id": "missing"},
                ),
            ):
                query = {**query, "context": ctx}
                with self.subTest(query=query):
                    single = dispatch(service, query)
                    combined = dispatch(
                        service,
                        batch(service, [{"kind": "capabilities"}, query], context=ctx),
                    )
                    self.assertFalse(single["success"])
                    self.assertEqual(combined["error"], single["error"])
            self.assertEqual(service.control.registry.capacity(), before)

    def test_events_progress_can_advance_generation_between_batch_results(self):
        with service_fixtures.service_owner(self) as (cache, service):
            before = service.cache_watermark()
            progress = service.progress

            def reset_during_progress():
                cache.reset()
                progress()

            with patch.object(service, "progress", side_effect=reset_during_progress):
                reply = dispatch(
                    service,
                    batch(
                        service,
                        [
                            {"kind": "capabilities"},
                            {"kind": "events", "context": context(service)},
                            {"kind": "capabilities"},
                        ],
                    ),
                )
            self.assertTrue(reply["success"], reply)
            first, events, last = reply["data"]["results"]
            self.assertEqual(
                first["feedback"]["cache_watermark"]["generation"], before.generation
            )
            self.assertGreater(
                events["cache_watermark"]["generation"], before.generation
            )
            self.assertEqual(
                last["feedback"]["cache_watermark"], events["cache_watermark"]
            )
            service.reset_after_cache_flush()


if __name__ == "__main__":
    unittest.main()
