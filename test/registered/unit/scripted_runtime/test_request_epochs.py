from __future__ import annotations

import unittest
from array import array
from concurrent.futures import Future
from types import SimpleNamespace

import aiohttp
import orjson

from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRequestEpochs(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=1),
            waiting_queue=[],
            running_batch=None,
            last_batch=None,
            chunked_req=None,
        )
        self.hook = object.__new__(ScriptedSchedulerHook)
        self.hook.scheduler = self.scheduler
        self.hook._is_driver = True
        self.hook._batch_log = []
        self.ctx = ScriptedContext(
            scheduler_hook=self.hook,
            tokenizer_recv_proxy=None,
            http_poster=None,
        )

    def _register(self, *, rid="reused"):
        return self.ctx._register_request(rid=rid, post_future=Future())

    def _req(self, *, rid="reused"):
        return Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [1] * 16),
            sampling_params=SamplingParams(max_new_tokens=4),
        )

    def _record(self, *, reqs, chunked=None, mode=ForwardMode.EXTEND):
        self.scheduler.chunked_req = chunked
        batch = SimpleNamespace(
            reqs=reqs,
            forward_iter=len(self.hook._batch_log),
            forward_mode=mode,
        )
        self.hook.on_run_batch(batch)
        self.scheduler.last_batch = batch

    def test_reused_rid_does_not_resolve_to_finished_predecessor(self):
        """A new request must not inherit a completed request's state before admission."""
        first = self._register()
        old_req = self._req()
        self._record(reqs=[old_req], chunked=old_req)
        old_req.finished_reason = FINISH_LENGTH(length=4)
        first._epoch.post_future.set_result(None)

        second = self._register()
        self.assertFalse(second.finished)
        self.assertIsNone(second.req)
        self.assertEqual(second.status, "unknown")
        self.assertFalse(second.is_chunking)
        self.assertEqual(second.remaining_prompt_tokens, 0)
        self.assertEqual(second.chunks_done, 0)
        self.assertTrue(first.finished)
        self.assertIs(first.req, old_req)

        new_req = self._req()
        self.scheduler.waiting_queue = [new_req]
        self.assertIs(second.req, new_req)
        self.assertEqual(second.status, "waiting")
        self.assertEqual(second.remaining_prompt_tokens, 16)
        self.assertIs(self.ctx.find_req_by_rid(second.rid), new_req)
        self.assertIs(first.req, old_req)

        self.scheduler.chunked_req = None
        self.scheduler.last_batch = None
        self.assertIsNone(first.req)
        self.assertEqual(first.status, "finished")
        self.assertFalse(second.finished)

    def test_chunk_and_park_counts_follow_request_identity(self):
        """Reused IDs must not combine chunk counts or attribute stale parks to a new request."""
        first = self._register()
        old_req = self._req()
        self._record(reqs=[old_req], chunked=old_req)
        self._record(reqs=[], chunked=old_req)
        self._record(reqs=[old_req], chunked=old_req)
        self._record(reqs=[old_req])
        old_req.finished_reason = FINISH_LENGTH(length=4)
        first._epoch.post_future.set_result(None)

        second = self._register()
        new_req = self._req()
        self._record(reqs=[new_req], chunked=old_req)
        self.assertEqual(second.chunks_done, 0)
        self.assertEqual(self.ctx.chunked_parks(second.rid), 0)
        self._record(reqs=[new_req], chunked=new_req)
        self._record(reqs=[new_req])

        self.assertEqual(first.chunks_done, 3)
        self.assertEqual(second.chunks_done, 2)
        self.assertEqual(self.ctx.chunks_done(second.rid), 2)
        self.assertEqual(self.ctx.chunked_parks(second.rid), 0)
        self.assertTrue(first.finished)
        self.assertFalse(second.finished)

    def test_unobserved_completed_request_is_recovered_from_batch_records(self):
        """A short request can finish before its handle is queried for the first time."""
        handle = self._register()
        req = self._req()
        self._record(reqs=[req])
        req.finished_reason = FINISH_LENGTH(length=4)
        self.scheduler.last_batch = None

        self.assertTrue(handle.finished)
        self.assertEqual(handle.status, "finished")
        self.assertIsNone(handle.req)

    def test_completion_without_forward_batch_uses_its_own_response(self):
        """An abort before the first forward can finish without leaving a batch record."""
        first = self._register()
        self.assertFalse(first.finished)
        first._epoch.post_future.set_result(None)
        self.assertTrue(first.finished)

        second = self._register()
        self.assertTrue(first.finished)
        self.assertFalse(second.finished)
        self.assertEqual(second.status, "unknown")

    def test_http_failure_after_arrival_is_not_reported_as_completion(self):
        """A failed response must not make an unfinished request appear successful."""
        handle = self._register()
        req = self._req()
        self._record(reqs=[req])
        handle._epoch.post_future.set_exception(RuntimeError("stream failed"))

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            _ = handle.finished
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            _ = handle.status

    def test_completed_request_surfaces_late_response_failure(self):
        """A scheduler finish must not hide a response failure that arrives afterward."""
        errors = (
            RuntimeError("late stream failure"),
            aiohttp.ClientResponseError(
                request_info=SimpleNamespace(real_url="http://localhost/generate"),
                history=(),
                status=503,
                message="backend unavailable after finish",
            ),
        )
        for index, error in enumerate(errors):
            handle = self._register(rid=f"late-error-{index}")
            req = self._req(rid=handle.rid)
            self._record(reqs=[req])
            req.finished_reason = FINISH_LENGTH(length=4)
            self.assertTrue(handle.finished)
            handle._epoch.post_future.set_exception(error)

            for name, query in (
                ("handle.finished", lambda: handle.finished),
                ("handle.status", lambda: handle.status),
                ("context.is_finished", lambda: self.ctx.is_finished(handle.rid)),
            ):
                with self.subTest(error=error, query=name):
                    with self.assertRaises(type(error)) as raised:
                        query()
                    self.assertIs(raised.exception, error)

    def test_expected_abort_response_is_terminal_in_last_batch(self):
        """A requested abort remains terminal while its finished Req is still in the last batch."""
        for message in ("Aborted", "Abort in waiting queue"):
            with self.subTest(message=message):
                handle = self._register()
                req = self._req()
                self._record(reqs=[req])
                req.finished_reason = FINISH_ABORT()
                handle._epoch.abort_requested = True
                handle._epoch.post_future.set_exception(
                    aiohttp.ClientResponseError(
                        request_info=None,
                        history=(),
                        status=400,
                        message=orjson.dumps({"error": {"message": message}}).decode(),
                    )
                )

                self.assertIs(handle.req, req)
                self.assertTrue(handle.finished)
                self.assertEqual(handle.status, "finished")
                self.assertTrue(self.ctx.is_finished(handle.rid))

    def test_abort_response_requires_exact_message_and_terminal_request(self):
        """An abort marker must not suppress unrelated failures or errors for a still-live request."""
        for index, (finished, requested, message) in enumerate(
            (
                (False, True, "Aborted"),
                (True, True, "Aborted because of invalid sampling parameters"),
                (True, False, "Abort in waiting queue"),
            )
        ):
            with self.subTest(finished=finished, requested=requested, message=message):
                handle = self._register(rid=f"abort-error-{index}")
                req = self._req(rid=handle.rid)
                self._record(reqs=[req])
                if finished:
                    req.finished_reason = FINISH_LENGTH(length=4)
                handle._epoch.abort_requested = requested
                error = aiohttp.ClientResponseError(
                    request_info=SimpleNamespace(real_url="http://localhost/generate"),
                    history=(),
                    status=400,
                    message=orjson.dumps({"error": {"message": message}}).decode(),
                )
                handle._epoch.post_future.set_exception(error)

                with self.assertRaises(aiohttp.ClientResponseError) as raised:
                    _ = handle.finished
                self.assertIs(raised.exception, error)

    def test_abort_response_before_first_forward_is_terminal(self):
        """An expected abort response completes an unobserved request, but other errors propagate."""
        for status in (400, 503):
            with self.subTest(status=status):
                handle = self._register(rid=f"aborted-{status}")
                handle._epoch.abort_requested = True
                handle._epoch.post_future.set_exception(
                    aiohttp.ClientResponseError(
                        request_info=None,
                        history=(),
                        status=status,
                        message='{"error":{"message":"Abort in waiting queue"}}',
                    )
                )
                if status == 400:
                    self.assertTrue(handle.finished)
                    self.assertEqual(handle.status, "finished")
                else:
                    with self.assertRaises(aiohttp.ClientResponseError):
                        _ = handle.finished

    def test_reset_discards_request_and_batch_history(self):
        """Separate scripts can reuse an ID without inheriting earlier completion or chunks."""
        first = self._register()
        old_req = self._req()
        self._record(reqs=[old_req], chunked=old_req)
        old_req.finished_reason = FINISH_LENGTH(length=4)
        first._epoch.post_future.set_result(None)
        self.scheduler.chunked_req = None
        self.scheduler.last_batch = None
        self.ctx._reset_request_tracking()

        second = self._register()
        self.assertFalse(second.finished)
        self.assertEqual(second.chunks_done, 0)
        self.assertEqual(self.ctx.chunked_parks(second.rid), 0)
        self.assertIsNone(second.req)
        self.assertTrue(first.finished)

    def test_tracking_reset_is_atomic_when_a_later_response_failed(self):
        """A failed response must leave every request epoch intact when tracking reset is rejected."""
        first = self._register(rid="first")
        first._epoch.post_future.set_result(None)
        later = self._register(rid="later")
        error = RuntimeError("later response failed")
        later._epoch.post_future.set_exception(error)

        with self.assertRaises(RuntimeError) as raised:
            self.ctx._reset_request_tracking()
        self.assertIs(raised.exception, error)
        self.assertIs(self.ctx._request_epochs[first.rid], first._epoch)
        self.assertIs(self.ctx._request_epochs[later.rid], later._epoch)
        self.assertFalse(first._epoch.closed)
        self.assertFalse(later._epoch.closed)

    def test_observed_waiting_abort_can_complete_after_removal(self):
        """An observed waiting request can be removed by abort without setting its finish reason."""
        handle = self._register()
        req = self._req()
        self.scheduler.waiting_queue = [req]
        self.assertIs(handle.req, req)
        self.scheduler.waiting_queue.clear()
        handle._epoch.abort_requested = True
        handle._epoch.post_future.set_exception(
            aiohttp.ClientResponseError(
                request_info=None,
                history=(),
                status=400,
                message='{"error":{"message":"Abort in waiting queue"}}',
            )
        )

        self.assertFalse(req.finished())
        self.assertTrue(handle.finished)
        self.assertEqual(handle.status, "finished")
        self.assertIsNone(handle.req)

    def test_unknown_handle_does_not_resolve_an_unrelated_request(self):
        """Unknown IDs must not inherit the status of another admitted request."""
        self._register()
        req = self._req()
        self._record(reqs=[req], chunked=req)
        handle = ScriptedReqHandle(rid="unknown", context=self.ctx)

        self.assertIsNone(handle.req)
        self.assertFalse(handle.finished)
        self.assertEqual(handle.status, "unknown")
        self.assertEqual(handle.chunks_done, 0)
        self.assertFalse(handle.is_chunking)


if __name__ == "__main__":
    unittest.main()
