"""CPU-only admission and real HTTP streaming tests; no GPU dependency."""

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from multidict import CIMultiDict

SOURCE = (
    Path(__file__).resolve().parents[1] / "src/sglang_router/pd_session_admission.py"
)
spec = importlib.util.spec_from_file_location("pd_session_admission_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def inventory():
    return {
        "workers": [
            {
                "url": "http://ctx:30000@0",
                "id": "ctx0",
                "worker_type": "prefill",
                "is_healthy": True,
            },
            {
                "url": "http://ctx:30000@1",
                "id": "ctx1",
                "worker_type": "prefill",
                "is_healthy": True,
            },
            {
                "url": "http://gen:30000@0",
                "id": "gen0",
                "worker_type": "decode",
                "is_healthy": True,
            },
        ]
    }


class AdmissionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.ranks = ["http://ctx@0", "http://ctx@1"]
        self.policy = module.SessionAdmission(self.ranks)

    def test_new_sessions_balance_bytes_but_existing_owner_stays(self):
        a = self.policy.reserve("a", 100, self.ranks)
        b = self.policy.reserve("b", 10, self.ranks)
        c = self.policy.reserve("a", 1000, self.ranks)
        self.assertNotEqual(a.owner, b.owner)
        self.assertEqual(a.owner, c.owner)
        for reservation in [a, b, c]:
            self.policy.release(reservation)
        self.assertEqual(sum(self.policy.bytes.values()), 0)
        self.assertEqual(self.policy.reserve("a", 1, self.ranks).owner, a.owner)

    def test_request_count_and_assigned_sessions_break_ties(self):
        a = self.policy.reserve("a", 1, self.ranks)
        self.policy.release(a)
        b = self.policy.reserve("b", 1, self.ranks)
        self.assertNotEqual(a.owner, b.owner)

    def test_unhealthy_owner_never_migrates(self):
        a = self.policy.reserve("a", 1, self.ranks)
        healthy = [r for r in self.ranks if r != a.owner]
        with self.assertRaises(OverflowError):
            self.policy.reserve("a", 1, healthy)
        self.assertEqual(self.policy.owners["a"], a.owner)
        self.assertEqual(self.policy.reserve("b", 1, healthy).owner, healthy[0])

    def test_capacity_does_not_evict_existing_affinity(self):
        policy = module.SessionAdmission(self.ranks, max_sessions=1)
        a = policy.reserve("a", 1, self.ranks)
        policy.release(a)
        with self.assertRaises(OverflowError):
            policy.reserve("b", 1, self.ranks)
        self.assertEqual(policy.reserve("a", 1, self.ranks).owner, a.owner)

    def test_double_release_and_invalid_weight_rejected(self):
        a = self.policy.reserve("a", 1, self.ranks)
        self.policy.release(a)
        with self.assertRaises(ValueError):
            self.policy.release(a)
        for weight in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                self.policy.reserve("b", weight, self.ranks)

    def test_inventory_order_can_change_but_membership_must_match(self):
        a, b = inventory(), inventory()
        b["workers"].reverse()
        self.assertEqual(len(module.inventory_ranks([a, b], 2, 1)), 2)
        b["workers"][0]["url"] = "http://another-gen@0"
        with self.assertRaises(ValueError):
            module.inventory_ranks([a, b], 2, 1)

    def test_duplicate_and_unhealthy_inventory_rejected(self):
        a = inventory()
        a["workers"][1]["url"] = a["workers"][0]["url"]
        with self.assertRaises(ValueError):
            module.inventory_ranks([a], 2, 1)
        a = inventory()
        a["workers"][0]["is_healthy"] = False
        with self.assertRaises(ValueError):
            module.inventory_ranks([a], 2, 1)

    def test_hop_headers_stripped_without_losing_duplicate_end_to_end_headers(self):
        headers = CIMultiDict(
            [
                ("Connection", "X-Hop"),
                ("Connection", "Keep-Alive"),
                ("X-Hop", "secret"),
                ("Set-Cookie", "a"),
                ("Set-Cookie", "b"),
            ]
        )
        filtered = module.forwarded_headers(headers)
        self.assertNotIn("X-Hop", filtered)
        self.assertNotIn("Connection", filtered)
        self.assertEqual(filtered.getall("Set-Cookie"), ["a", "b"])


class AdmissionHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.current = inventory()
        self.requests = []
        self.allow_tail = asyncio.Event()
        self.finished = asyncio.Event()
        self.mode = "stream"
        upstream = web.Application()

        async def workers(request):
            return web.json_response(self.current)

        async def stream(request):
            payload = await request.read()
            self.requests.append((request.headers.copy(), payload))
            if self.mode == "error":
                return web.Response(status=503, body=b"unavailable")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"content":"first"}\n\n')
            try:
                await self.allow_tail.wait()
                await response.write(b'data: {"content":"last"}\n\ndata: [DONE]\n\n')
                await response.write_eof()
            except (ConnectionResetError, asyncio.CancelledError):
                pass  # The disconnect test intentionally closes its downstream client.
            finally:
                self.finished.set()
            return response

        upstream.router.add_get("/workers", workers)
        upstream.router.add_post("/v1/chat/completions", stream)
        self.native = TestServer(upstream)
        await self.native.start_server()
        proxy = module.make_app(
            [str(self.native.make_url("/"))],
            [inventory()],
            2,
            1,
            inventory_guard_interval=0.02,
        )
        self.edge = TestServer(proxy, handler_cancellation=True)
        self.client = TestClient(self.edge)
        await self.client.start_server()

    async def asyncTearDown(self):
        self.allow_tail.set()
        await self.client.close()
        await self.native.close()

    async def state(self):
        async with self.client.get("/ctx-routing-state") as response:
            return await response.json()

    async def wait_state(self, predicate):
        for _ in range(100):
            state = await self.state()
            if predicate(state):
                return state
            await asyncio.sleep(0.01)
        self.fail("Admission state did not reach the expected condition")

    async def test_streams_first_chunk_before_upstream_finishes_and_preserves_payload(
        self,
    ):
        response = await self.client.post(
            "/v1/chat/completions",
            data=b'{"stream":true}',
            headers={module.SESSION_HEADER: "session-a"},
        )
        first = await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=1)
        self.assertEqual(first, b'data: {"content":"first"}\n\n')
        self.assertFalse(self.finished.is_set())
        self.assertEqual(self.requests[0][1], b'{"stream":true}')
        self.assertEqual(
            self.requests[0][0][module.TARGET_HEADER], "http://ctx:30000@0"
        )
        state = await self.state()
        self.assertEqual(
            sum(r["inflight_requests"] for r in state["ranks"].values()), 1
        )
        self.allow_tail.set()
        tail = await response.read()
        self.assertTrue(tail.endswith(b"data: [DONE]\n\n"))
        await self.wait_state(
            lambda s: sum(r["inflight_requests"] for r in s["ranks"].values()) == 0
        )

    async def test_missing_or_injected_routing_headers_rejected(self):
        for headers in [
            {},
            {module.SESSION_HEADER: "a", module.TARGET_HEADER: "http://ctx:30000@1"},
        ]:
            async with self.client.post(
                "/v1/chat/completions", data=b"{}", headers=headers
            ) as response:
                self.assertEqual(response.status, 400)
        self.assertEqual(len(self.requests), 0)

    async def test_chunked_upload_is_not_silently_underweighted(self):
        async def chunks():
            yield b"{}"

        async with self.client.post(
            "/v1/chat/completions", data=chunks(), headers={module.SESSION_HEADER: "a"}
        ) as response:
            self.assertEqual(response.status, 411)
        self.assertEqual(len(self.requests), 0)

    async def test_upstream_error_is_forwarded_once_and_reservation_released(self):
        self.mode = "error"
        async with self.client.post(
            "/v1/chat/completions", data=b"{}", headers={module.SESSION_HEADER: "a"}
        ) as response:
            self.assertEqual(response.status, 503)
            self.assertEqual(await response.read(), b"unavailable")
        self.assertEqual(len(self.requests), 1)
        state = await self.state()
        self.assertEqual(
            sum(r["inflight_requests"] for r in state["ranks"].values()), 0
        )

    async def test_membership_change_latches_closed_without_reassigning(self):
        self.current["workers"][0]["id"] = "replacement-process"
        await self.wait_state(lambda s: not s["inventory_valid"])
        async with self.client.post(
            "/v1/chat/completions", data=b"{}", headers={module.SESSION_HEADER: "a"}
        ) as response:
            self.assertEqual(response.status, 503)
        self.assertEqual(len(self.requests), 0)

    async def test_disconnect_releases_reservation(self):
        response = await self.client.post(
            "/v1/chat/completions", data=b"{}", headers={module.SESSION_HEADER: "a"}
        )
        await response.content.readuntil(b"\n\n")
        response.close()
        await self.wait_state(
            lambda s: sum(r["inflight_requests"] for r in s["ranks"].values()) == 0
        )
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
