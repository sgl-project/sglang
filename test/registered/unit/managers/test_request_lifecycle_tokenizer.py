import asyncio
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import httpx
from fastapi import FastAPI, Request

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers import tokenizer_control_mixin
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    GenerateReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionParams,
    SessionRoutingReqInput,
    SessionRoutingReqOutput,
)
from sglang.srt.managers.request_lifecycle import RequestLifecycle
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLifecycleTokenizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.enable_trace = False
        self.manager.disaggregation_mode = DisaggregationMode.NULL
        self.manager.rid_to_state = {}
        self.manager.request_lifecycle = RequestLifecycle()
        self.manager._lifecycle_tasks = {}
        self.attempt = uuid.uuid4().hex
        self.manager.request_lifecycle.claim(self.attempt, "null")

    async def test_open_uses_proxy_fence_but_ignores_untrusted_body_incarnation(self):
        manager = self.manager
        manager.session_futures = {}
        manager.auto_create_handle_loop = Mock()
        manager._dispatch_to_scheduler = lambda obj: (
            manager._handle_open_session_req_output(
                OpenSessionReqOutput(session_id=obj.session_id, success=True)
            )
        )
        chosen = uuid.uuid4().hex
        for request in [
            SimpleNamespace(state=SimpleNamespace(native_session_incarnation=chosen)),
            None,
        ]:
            obj = OpenSessionReqInput(
                capacity_of_str_len=0, session_id="s", session_incarnation="body-value"
            )
            self.assertEqual(await manager.open_session(obj, request), "s")
            self.assertEqual(len(obj.session_incarnation), 32)
            if request is not None:
                self.assertEqual(obj.session_incarnation, chosen)
            else:
                self.assertNotIn(obj.session_incarnation, (chosen, "body-value"))

    def test_stale_session_close_is_rejected_before_releasing_radix_state(self):
        controller, cache = MagicMock(), Mock()
        controller.get.return_value = SimpleNamespace(incarnation="current")
        controller.__contains__.return_value = True
        scheduler = SimpleNamespace(
            session_controller=controller,
            tree_cache=cache,
            enable_session_radix_cache=True,
        )
        request = CloseSessionReqInput(session_id="s", session_incarnation="stale")
        Scheduler.close_session(scheduler, request)
        cache.release_radix_session.assert_not_called()
        controller.close.assert_not_called()
        request.session_incarnation = "current"
        Scheduler.close_session(scheduler, request)
        cache.release_radix_session.assert_called_once_with("s")
        controller.close.assert_called_once_with(request)

    async def test_session_routing_cancelled_query_cannot_capture_a_later_reply(self):
        manager = self.manager
        manager.session_routing_futures = {}
        manager.elastic_worker_count = 2
        manager.auto_create_handle_loop = Mock()
        manager._dispatch_to_scheduler = Mock()

        def query():
            return SessionRoutingReqInput(
                session_params=SessionParams(id="s"), dp_rank=1
            )

        with patch.object(
            tokenizer_control_mixin,
            "get_parallel",
            return_value=SimpleNamespace(enable_dp_attention=False),
        ):
            first = asyncio.create_task(manager.session_routing(query()))
            await asyncio.sleep(0)
            first_id = manager._dispatch_to_scheduler.call_args.args[0].query_id
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(manager.session_routing_futures, {})
            second = asyncio.create_task(manager.session_routing(query()))
            await asyncio.sleep(0)
            second_id = manager._dispatch_to_scheduler.call_args.args[0].query_id
            manager._handle_session_routing_output(
                SessionRoutingReqOutput(query_id=first_id, dp_rank=1)
            )
            self.assertFalse(second.done())
            reply = SessionRoutingReqOutput(
                query_id=second_id, dp_rank=1, session_incarnation="new"
            )
            manager._handle_session_routing_output(reply)
            self.assertIs(await second, reply)
            self.assertEqual(manager.session_routing_futures, {})

    async def test_invalid_json_is_terminal_without_entering_generation(self):
        app = FastAPI()

        async def endpoint(obj: GenerateReqInput, request: Request):
            self.fail("invalid JSON must not reach generation")

        app.router.add_api_route(
            "/generate",
            endpoint,
            methods=["POST"],
            route_class_override=http_server.NativeGenerateRoute,
        )
        self.manager.auto_create_handle_loop = Mock()
        attempt = uuid.uuid4().hex
        headers = {
            "content-type": "application/json",
            "x-sglang-attempt-id": attempt,
            "x-sglang-worker-incarnation": self.manager.request_lifecycle.incarnation,
        }
        with patch.object(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=self.manager),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/generate", headers=headers, content=b"{")
        self.assertEqual(response.status_code, 422)
        snapshot = self.manager.request_lifecycle.snapshot(attempt)
        self.assertTrue(snapshot["terminal"])
        self.assertEqual(snapshot["children"], [])

    async def test_response_generator_can_outlive_accounting_acknowledgement(self):
        obj = GenerateReqInput(text="prompt", stream=True)
        obj._lifecycle_attempt_id = self.attempt
        self.manager.request_lifecycle.seal(self.attempt)

        async def generate(obj, request):
            yield {"text": "native response"}

        self.manager.generate_request = generate
        with patch.object(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=self.manager),
        ):
            generator = http_server._generate_with_lifecycle(obj, None)
            self.assertEqual(await generator.__anext__(), {"text": "native response"})
            self.manager.request_lifecycle.acknowledge(self.attempt)
            await generator.aclose()
        self.assertEqual(self.manager._lifecycle_tasks, {})

    def test_exact_child_cancellation_keeps_native_prefix_compatibility(self):
        a = SimpleNamespace(rid="client", lifecycle_id="first")
        b = SimpleNamespace(rid="client-1", lifecycle_id="second")
        reused = SimpleNamespace(rid="client", lifecycle_id="third")
        exact = AbortReq(lifecycle_id="first")
        self.assertEqual(
            [exact.matches(req) for req in (a, b, reused)], [True, False, False]
        )
        legacy = AbortReq(rid="client")
        self.assertTrue(all(legacy.matches(req) for req in (a, b, reused)))

    async def test_batch_sampling_tracks_only_executed_children(self):
        for mode, rooms, expected in (
            (DisaggregationMode.NULL, None, [None] * 8),
            (DisaggregationMode.PREFILL, 100, [100, 102, 104, 101, 103, 105]),
            (DisaggregationMode.DECODE, 100, [100, 102, 104, 101, 103, 105]),
            (
                DisaggregationMode.DECODE,
                2**63,
                [2**63 + offset for offset in (0, 2, 4, 1, 3, 5)],
            ),
            (DisaggregationMode.PREFILL, [100, 200], [100, 101, 102, 200, 201, 202]),
            (DisaggregationMode.DECODE, [100, 200], [100, 101, 102, 200, 201, 202]),
        ):
            with self.subTest(mode=mode, rooms=rooms):
                self.setUp()
                self.manager.disaggregation_mode = mode
                await self._check_sampling_children(rooms, expected)

    async def _check_sampling_children(self, rooms, expected):
        manager = self.manager
        obj = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params={"n": 3, "max_new_tokens": 1},
            rid="native",
            bootstrap_room=rooms,
        )
        obj._lifecycle_attempt_id = self.attempt
        obj.normalize_batch_and_arguments()
        manager._init_req_state(obj)
        self.assertEqual(len(manager.rid_to_state), 2)
        self.assertEqual(
            manager.request_lifecycle.snapshot(self.attempt)["children"], []
        )

        async def tokenize(item):
            return SimpleNamespace(
                rid=item.rid,
                input_ids=item.input_ids,
                sampling_params=SimpleNamespace(max_new_tokens=1),
                mm_inputs=None,
                stream=False,
                bootstrap_room=item.bootstrap_room,
            )

        dispatched_rooms = []

        async def send(item):
            dispatched_rooms.append(item.bootstrap_room)
            manager._prepare_lifecycle_dispatch(item)
            manager.request_lifecycle.scheduler_event(item.lifecycle_id, 0, "prefill")
            manager.request_lifecycle.scheduler_event(item.lifecycle_id, 0, "terminal")

        async def response(item, request):
            manager.rid_to_state.pop(item.rid)
            yield {"meta_info": {"id": item.rid}}

        manager._tokenize_one_request = tokenize
        manager._send_one_request = send
        manager._wait_one_response = response
        generator = manager._handle_batch_request(obj)
        result = await generator.__anext__()
        await generator.aclose()
        self.assertEqual(len(result), 6)
        self.assertEqual(dispatched_rooms, expected)
        self.assertEqual(manager.rid_to_state, {})
        snapshot = manager.request_lifecycle.snapshot(self.attempt)
        self.assertTrue(snapshot["terminal"])
        self.assertEqual(len(snapshot["children"]), len(expected))
        self.assertEqual(
            sum(child["kind"] == "warmup" for child in snapshot["children"]),
            len(expected) - 6,
        )

    async def test_invalid_parallel_rooms_rejected_before_dispatch(self):
        self.manager.disaggregation_mode = DisaggregationMode.PREFILL
        self.manager._send_one_request = Mock(side_effect=AssertionError("dispatched"))
        for rooms, message in (([100, 101], "overlap"), (2**64 - 2, "uint64")):
            with self.subTest(rooms=rooms):
                obj = GenerateReqInput(
                    input_ids=[[1, 2], [3, 4]],
                    sampling_params={"n": 3},
                    bootstrap_room=rooms,
                )
                obj.normalize_batch_and_arguments()
                generator = self.manager._handle_batch_request(obj)
                with self.assertRaisesRegex(ValueError, message):
                    await generator.__anext__()
                await generator.aclose()

    def test_duplicate_batch_rejection_is_atomic(self):
        obj = GenerateReqInput(input_ids=[[1], [2]], rid=["a", "b"])
        obj.normalize_batch_and_arguments()
        obj._lifecycle_attempt_id = self.attempt
        old_state = object()
        self.manager.rid_to_state["b"] = old_state
        with self.assertRaises(ValueError):
            self.manager._init_req_state(obj)
        self.assertEqual(self.manager.rid_to_state, {"b": old_state})
        self.assertEqual(
            self.manager.request_lifecycle.snapshot(self.attempt)["children"], []
        )


if __name__ == "__main__":
    unittest.main()
