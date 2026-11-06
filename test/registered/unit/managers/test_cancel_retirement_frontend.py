"""Exercise cancellation receipt HTTP/IPC consumers without launching a model.

Most cases inject scheduler replies; the integration case uses the actual
scheduler, retirer and CPU row/token pools. Model/device startup is bypassed;
the route, cancellation methods, registry and IPC codecs are real. These tests
do not establish GPU completion.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import zmq
import zmq.asyncio
from fastapi import FastAPI

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers.cancellation_retirement import CancellationRetirementRegistry
from sglang.srt.managers.io_struct import (
    AbortReq,
    CancellationRetirementOutput,
    CancellationRetirementReq,
    GenerateReqInput,
    async_sock_recv,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import create_context, use_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestCancelRetirementFrontend(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.args = ServerArgs(model_path="dummy", device="cpu")
        self.scope = use_context(create_context(self.args, role="tokenizer"))
        self.scope.__enter__()
        self.feature = envs.SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS.override(True)
        self.feature.__enter__()

        self.context = zmq.asyncio.Context()
        self.scheduler_input = self.context.socket(zmq.PULL)
        self.scheduler_input.bind("inproc://cancel-retirement-frontend")
        self.sender = zmq.Socket(self.context, zmq.PUSH)
        self.sender.connect("inproc://cancel-retirement-frontend")

        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.server_args = self.args
        self.tm.init_running_status()
        self.tm.tokenizer_ipc_name = None
        self.tm.send_to_scheduler = self.sender
        self.tm.enable_metrics = False
        self.tm.enable_lora = False
        self.tm.incremental_streaming_output = False
        self.tm.disaggregation_mode = DisaggregationMode.NULL
        self.tm._config_updates = []
        self.tm.init_request_dispatcher()

        self.prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=self.tm))
        app = FastAPI()
        app.add_api_route("/abort_request", http_server.abort_request, methods=["POST"])
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.tasks = []
        self.extra_sockets = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.aclose()
        http_server._global_state = self.prior_state
        self.sender.close(linger=0)
        self.scheduler_input.close(linger=0)
        for socket in self.extra_sockets:
            socket.close(linger=0)
        self.context.term()
        self.feature.__exit__(None, None, None)
        self.scope.__exit__(None, None, None)

    def _state(self, rid="request", instance="instance-a", **request_fields):
        obj = GenerateReqInput(rid=rid, text="hello", **request_fields)
        obj.normalize_batch_and_arguments()
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=APIServerReqTimeStats(),
            dispatched=True,
            request_instance_id=instance,
        )
        self.tm.rid_to_state[rid] = state
        return state

    async def _cancel(self, cancellation_id="cancel", rid="request", timeout=0):
        return await self.client.post(
            "/abort_request",
            json={
                "rid": rid,
                "wait_for_retirement": True,
                "cancellation_id": cancellation_id,
                "timeout_seconds": timeout,
            },
        )

    async def _next_request(self):
        return await asyncio.wait_for(async_sock_recv(self.scheduler_input), 1)

    async def _no_request(self):
        with self.assertRaises(zmq.Again):
            await self.scheduler_input.recv(flags=zmq.NOBLOCK)

    def _receipt(self, request, *, instance=None, status="retired"):
        output = CancellationRetirementOutput(
            rid=request.rid,
            cancellation_id=request.cancellation_id,
            request_instance_id=instance or request.request_instance_id,
            status=status,
        )
        output = msgpack_decode(msgpack_encode(output))
        self.assertIsInstance(output, CancellationRetirementOutput)
        self.tm._result_dispatcher(output)

    async def test_default_abort_keeps_empty_200_and_legacy_dispatch(self):
        self._state()
        with envs.SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS.override(False):
            response = await self.client.post("/abort_request", json={"rid": "request"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")
        request = await self._next_request()
        self.assertIsInstance(request, AbortReq)
        self.assertEqual(request.rid, "request")
        self.assertFalse(request.wait_for_retirement)

    async def test_timeout_retry_waits_for_receipt_not_ordinary_terminal(self):
        self._state()
        pending = await self._cancel()
        self.assertEqual(pending.status_code, 202)
        self.assertEqual(pending.json()["status"], "pending")
        self.assertEqual(pending.json()["completion_scope"], "scheduler_owned_request")
        request = await self._next_request()
        self.assertIsInstance(request, CancellationRetirementReq)

        self.tm._handle_abort_req(AbortReq(rid="request"))
        self.assertNotIn("request", self.tm.rid_to_state)
        retry = await self._cancel()
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(retry.json()["status"], "pending")
        await self._no_request()

        self._receipt(request)
        retired = await self._cancel()
        self.assertEqual(retired.status_code, 200)
        self.assertEqual(retired.json()["status"], "retired")
        await self._no_request()

    async def test_reused_rid_does_not_rebind_retry_or_accept_old_receipt(self):
        self._state()
        self.assertEqual((await self._cancel("old-operation")).status_code, 202)
        old_request = await self._next_request()

        new_state = self._state(instance="instance-b")
        self.assertEqual((await self._cancel("new-operation")).status_code, 202)
        new_request = await self._next_request()
        self.assertNotEqual(
            old_request.request_instance_id, new_request.request_instance_id
        )

        self._receipt(new_request, instance=old_request.request_instance_id)
        self._receipt(old_request)
        self.assertEqual((await self._cancel("old-operation")).status_code, 200)
        self.assertEqual((await self._cancel("new-operation")).status_code, 202)
        self.assertIs(self.tm.rid_to_state["request"], new_state)
        await self._no_request()

        self._receipt(new_request)
        self.assertEqual((await self._cancel("new-operation")).status_code, 200)
        conflict = await self._cancel("old-operation", rid="different-request")
        self.assertEqual(conflict.status_code, 400)
        self.assertIn("another request", conflict.text)
        await self._no_request()

    async def test_disconnected_http_waiter_does_not_cancel_shared_receipt(self):
        self._state()
        waiter = asyncio.create_task(self._cancel(timeout=30))
        self.tasks.append(waiter)
        request = await self._next_request()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        self._receipt(request)
        retry = await self._cancel()
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["status"], "retired")
        await self._no_request()

    async def test_negative_results_remain_bound_after_rid_reuse(self):
        cases = (
            ("unknown", {}, 409),
            ("multimodal", {"image_data": "unused-image"}, 422),
            ("parallel", {"sampling_params": {"n": 2}}, 422),
        )
        for operation, fields, expected in cases:
            with self.subTest(operation=operation):
                rid = operation
                if fields:
                    self._state(rid=rid, **fields)
                first = await self._cancel(operation, rid=rid)
                self.assertEqual(first.status_code, expected, first.text)
                replacement = self._state(rid=rid, instance="replacement")
                retry = await self._cancel(operation, rid=rid)
                self.assertEqual(retry.status_code, expected, retry.text)
                self.assertEqual(retry.json(), first.json())
                self.assertIs(self.tm.rid_to_state[rid], replacement)
                await self._no_request()

        rid = "scheduler-rejected"
        self._state(rid=rid)
        self.assertEqual((await self._cancel(rid=rid)).status_code, 202)
        request = await self._next_request()
        self._receipt(request, status="unsupported")
        rejected = await self._cancel(rid=rid)
        self.assertEqual(rejected.status_code, 422)
        legacy = await self.client.post("/abort_request", json={"rid": rid})
        self.assertEqual(legacy.status_code, 200)
        dispatched = await self._next_request()
        self.assertIsInstance(dispatched, AbortReq)
        self.assertEqual(dispatched.rid, rid)

    async def test_wait_requires_enabled_feature_exact_rid_and_bounded_timeout(self):
        self._state()
        with envs.SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS.override(False):
            disabled = await self._cancel()
        self.assertEqual(disabled.status_code, 422)
        self.assertEqual(disabled.json()["status"], "unsupported")
        for fields in ({"abort_all": True}, {"rid": ""}, {"timeout_seconds": 31}):
            with self.subTest(fields=fields):
                response = await self.client.post(
                    "/abort_request",
                    json={
                        "rid": "request",
                        "wait_for_retirement": True,
                        "cancellation_id": "invalid",
                        **fields,
                    },
                )
                self.assertEqual(response.status_code, 400, response.text)
        await self._no_request()

    async def test_capacity_rejection_keeps_the_existing_waiter(self):
        self.tm._cancel_retirement_registry = CancellationRetirementRegistry(capacity=1)
        self._state()
        self.assertEqual((await self._cancel()).status_code, 202)
        request = await self._next_request()
        self._state(rid="other")
        full = await self._cancel("other-operation", rid="other")
        self.assertEqual(full.status_code, 400)
        self.assertIn("capacity", full.text.lower())
        await self._no_request()

        self._receipt(request)
        self.assertEqual((await self._cancel()).status_code, 200)

    async def test_expiry_returns_unknown_to_a_waiting_http_caller(self):
        self.tm._cancel_retirement_registry = CancellationRetirementRegistry(
            capacity=1, ttl_seconds=0
        )
        self._state()
        waiter = asyncio.create_task(self._cancel(timeout=30))
        self.tasks.append(waiter)
        await self._next_request()
        other = await self._cancel("other-operation", rid="missing")
        self.assertEqual(other.status_code, 409)
        expired = await waiter
        self.assertEqual(expired.status_code, 409)
        self.assertEqual(expired.json()["status"], "unknown")
        self.assertIn("expired", expired.json()["reason"].lower())

    async def test_late_reply_expires_without_another_registry_operation(self):
        registry = CancellationRetirementRegistry()
        clock = SimpleNamespace(now=10.0)
        with patch(
            "sglang.srt.managers.cancellation_retirement.time",
            SimpleNamespace(monotonic=lambda: clock.now),
        ):
            receipt = registry.create("late-cancel", "request", "instance-a")
            waiter = asyncio.create_task(registry.wait(receipt, timeout_seconds=30))
            self.tasks.append(waiter)
            await asyncio.sleep(0)
            clock.now += registry.ttl_seconds
            # No lookup/create triggers expiry before this late scheduler reply.
            registry.resolve(
                CancellationRetirementOutput(
                    rid="request",
                    cancellation_id="late-cancel",
                    request_instance_id="instance-a",
                    status="retired",
                )
            )
            result = await asyncio.wait_for(waiter, 1)
            self.assertEqual(result.status, "unknown")
            self.assertIn("expired", result.reason.lower())
            self.assertIsNone(registry.lookup("late-cancel", "request"))

    async def test_http_retirement_follows_real_scheduler_cpu_resource_release(self):
        from test_scheduler_cancellation_retirement import (
            allocate_cpu_req,
            make_cpu_scheduler,
        )

        state = self._state()
        replies = self.context.socket(zmq.PULL)
        replies.bind("inproc://cancel-retirement-output")
        output_sender = zmq.Socket(self.context, zmq.PUSH)
        output_sender.connect("inproc://cancel-retirement-output")
        self.extra_sockets.extend((replies, output_sender))

        scheduler = make_cpu_scheduler(None)
        scheduler.ipc_channels.send_to_tokenizer = SenderWrapper(output_sender)
        target = allocate_cpu_req(
            scheduler, rid="request", instance=state.request_instance_id
        )
        neighbor = allocate_cpu_req(scheduler, rid="neighbor", instance="neighbor")
        scheduler.waiting_queue.join([target, neighbor])
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 6)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 26)

        waiter = asyncio.create_task(self._cancel(timeout=30))
        self.tasks.append(waiter)
        request = await self._next_request()
        self.assertIsInstance(request, CancellationRetirementReq)
        self.assertTrue(target.kv.holds_kv)
        self.assertFalse(waiter.done())
        scheduler.handle_cancellation_retirement(request)

        terminal = await asyncio.wait_for(async_sock_recv(replies), 1)
        self.assertIsInstance(terminal, AbortReq)
        self.tm._result_dispatcher(terminal)
        self.assertNotIn("request", self.tm.rid_to_state)
        self.assertFalse(waiter.done())

        receipt = await asyncio.wait_for(async_sock_recv(replies), 1)
        self.assertIsInstance(receipt, CancellationRetirementOutput)
        self.assertEqual(receipt.status, "retired")
        self.assertFalse(target.kv.holds_kv)
        self.assertTrue(target.kv.is_kv_released)
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 7)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 29)
        self.assertTrue(neighbor.kv.holds_kv)
        self.assertFalse(neighbor.finished())
        self.assertEqual(list(scheduler.waiting_queue), [neighbor])

        self.tm._result_dispatcher(receipt)
        response = await asyncio.wait_for(waiter, 1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "retired")


if __name__ == "__main__":
    unittest.main()
