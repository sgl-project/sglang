from __future__ import annotations

import asyncio
import unittest
from collections import deque
from concurrent.futures import Future
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import msgspec
import orjson
import zmq

from sglang.srt.managers.io_struct import (
    AbortReq,
    FlushCacheReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime import scheduler_hook
from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
from sglang.test.scripted_runtime.context import ScriptedContext, http_post
from sglang.test.scripted_runtime.tokenizer_recv_proxy import ScriptedTokenizerRecvProxy
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _SocketBoundary:
    def __init__(self):
        self.messages = Queue()

    def recv_pyobj(self, flags=0):
        try:
            return self.messages.get_nowait()
        except Empty:
            raise zmq.ZMQError(zmq.EAGAIN) from None


class _HttpReply(msgspec.Struct, kw_only=True):
    messages: tuple = ()
    error: Exception | None = None
    hold_response: bool = False


class _HttpBoundary:
    def __init__(self, *, socket):
        self.socket = socket
        self.replies = deque()
        self.requests = []
        self.pending_responses = []
        self.scheduler_step = 0

    async def post(self, url, json):
        self.requests.append((url, json, self.scheduler_step))
        reply = self.replies.popleft()
        for message in reply.messages:
            self.socket.messages.put(message)
        if reply.hold_response:
            completed = asyncio.Event()
            self.pending_responses.append(completed)
            await completed.wait()
        if reply.error is not None:
            raise reply.error

    async def finish_responses(self):
        for completed in self.pending_responses:
            completed.set()
        await asyncio.sleep(0)


def _http_error(*, status=400, message="Duplicate request ID detected: reused"):
    return aiohttp.ClientResponseError(
        request_info=SimpleNamespace(real_url="http://localhost/generate"),
        history=(),
        status=status,
        message=orjson.dumps({"error": {"message": message}}).decode(),
    )


def _tokenized_request(*, rid):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=None,
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


class TestRequestLifecycle(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.socket = _SocketBoundary()
        self.http = _HttpBoundary(socket=self.socket)
        self.poster = BackgroundHttpPoster()
        self.addCleanup(self._close_poster)
        self.poster.post = self.http.post
        self.proxy = ScriptedTokenizerRecvProxy(underlying=self.socket)
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(host="localhost", port=30000),
            ps=SimpleNamespace(pp_size=1),
            waiting_queue=[],
            running_batch=None,
            last_batch=None,
            chunked_req=None,
        )
        self.ctx = ScriptedContext(
            scheduler_hook=SimpleNamespace(
                scheduler=scheduler, _is_driver=True, _batch_log=[]
            ),
            tokenizer_recv_proxy=self.proxy,
            http_poster=self.poster,
        )
        self.timeout_patch = patch.dict(
            http_post._http_post_and_await_recv_msg.__kwdefaults__, timeout_s=0.08
        )
        self.timeout_patch.start()
        self.addCleanup(self.timeout_patch.stop)

    def _close_poster(self):
        try:
            self.poster.submit_coro(self.http.finish_responses()).result(timeout=2)
        finally:
            self.poster.close()

    def _register(self, *, rid="reused"):
        return self.ctx._register_request(rid=rid, post_future=Future())

    def _prepare_reset(self):
        self.ctx.scheduler._engine_paused = False
        self.ctx.scheduler.tree_cache = SimpleNamespace()
        self.ctx.scheduler.is_fully_idle = lambda: True
        self.http.replies.append(_HttpReply(messages=(AbortReq(abort_all=True),)))

    def test_sync_reuse_requires_the_previous_http_response_to_close(self):
        """Synchronous reuse must reject an open previous response without posting or replacing its state."""
        previous = self._register()
        self.http.replies.append(
            _HttpReply(messages=(_tokenized_request(rid="reused"),), hold_response=True)
        )
        with self.assertRaisesRegex(ValueError, "start_req_with_retry"):
            self.ctx.start_req(rid="reused", prompt_len=16)
        self.assertEqual(self.http.requests, [])
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)

        previous._epoch.post_future.set_result(None)
        current = self.ctx.start_req(rid="reused", prompt_len=16)
        self.assertEqual(len(self.http.requests), 1)
        self.assertIs(self.ctx._request_epochs["reused"], current._epoch)
        self.assertIsNot(current._epoch, previous._epoch)
        self.assertFalse(current.finished)

    def test_reset_waits_for_http_cleanup_before_flushing(self):
        """Scheduler idleness must not let reset flush while an old HTTP handler can still affect reuse."""
        previous = self._register()
        self._prepare_reset()
        self.http.replies.append(_HttpReply(messages=(FlushCacheReqInput(),)))
        reset = scheduler_hook._reset_engine_state(self.ctx)
        self.assertIsNone(next(reset))
        self.assertEqual(len(self.http.requests), 1)
        self.assertIsNone(next(reset))
        self.assertEqual(len(self.http.requests), 1)
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)

        previous._epoch.post_future.set_result(None)
        self.assertIsNone(next(reset))
        self.assertEqual(
            [url.rsplit("/", 1)[1] for url, _, _ in self.http.requests],
            ["abort_request", "flush_cache"],
        )
        with self.assertRaises(StopIteration):
            next(reset)

    def test_reset_http_cleanup_budget_does_not_flush_on_timeout(self):
        """A stuck HTTP response must fail reset without flushing or discarding the prior request."""
        previous = self._register()
        self._prepare_reset()
        with patch.object(scheduler_hook, "RESET_DRAIN_MAX_STEPS", 2):
            reset = scheduler_hook._reset_engine_state(self.ctx)
            self.assertIsNone(next(reset))
            for _ in range(2):
                self.assertIsNone(next(reset))
            with self.assertRaisesRegex(RuntimeError, "HTTP"):
                next(reset)
        self.assertEqual(len(self.http.requests), 1)
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)
        self.assertFalse(previous._epoch.post_future.done())

    def test_reset_accepts_http_completion_on_last_budgeted_step(self):
        """HTTP cleanup completing on the final allowed scheduler step must still permit reset."""
        previous = self._register()
        self._prepare_reset()
        self.http.replies.append(_HttpReply(messages=(FlushCacheReqInput(),)))
        with patch.object(scheduler_hook, "RESET_DRAIN_MAX_STEPS", 2):
            reset = scheduler_hook._reset_engine_state(self.ctx)
            self.assertIsNone(next(reset))
            for _ in range(2):
                self.assertIsNone(next(reset))
            previous._epoch.post_future.set_result(None)
            self.assertIsNone(next(reset))
            with self.assertRaises(StopIteration):
                next(reset)
        self.assertEqual(
            [url.rsplit("/", 1)[1] for url, _, _ in self.http.requests],
            ["abort_request", "flush_cache"],
        )

    def test_retry_yields_before_resubmission_and_commits_only_success(self):
        """A duplicate response must leave the previous request intact until a later step accepts reuse."""
        previous = self._register()
        previous._epoch.post_future.set_result(None)
        self.http.replies.extend(
            (
                _HttpReply(error=_http_error()),
                _HttpReply(
                    messages=(_tokenized_request(rid="reused"),), hold_response=True
                ),
            )
        )
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        self.assertIsNone(next(retry))
        self.assertEqual(len(self.http.requests), 1)
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)

        self.http.scheduler_step += 1
        with self.assertRaises(StopIteration) as completed:
            next(retry)
        current = completed.exception.value
        self.assertIs(self.ctx._request_epochs["reused"], current._epoch)
        self.assertIsNot(current._epoch, previous._epoch)
        self.assertFalse(current.finished)
        self.assertTrue(previous.finished)
        self.assertEqual([step for _, _, step in self.http.requests], [0, 1])
        self.assertTrue(all(not body["stream"] for _, body, _ in self.http.requests))

    def test_reuse_waits_for_previous_http_response_before_posting(self):
        """A reused ID must not be posted while the old HTTP response can still clean it up."""
        previous = self._register()
        self.http.replies.append(
            _HttpReply(messages=(_tokenized_request(rid="reused"),), hold_response=True)
        )
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        self.assertIsNone(next(retry))
        self.assertEqual(self.http.requests, [])
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)

        previous._epoch.post_future.set_result(None)
        self.http.scheduler_step += 1
        with self.assertRaises(StopIteration) as completed:
            next(retry)
        current = completed.exception.value
        self.assertIs(self.ctx._request_epochs["reused"], current._epoch)
        self.assertIsNot(current._epoch, previous._epoch)
        self.assertEqual([step for _, _, step in self.http.requests], [1])
        self.assertFalse(current.finished)

    def test_pending_http_response_exhausts_budget_without_posting(self):
        """A response that never closes must exhaust the reuse budget without issuing a new request."""
        previous = self._register()
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        for _ in range(2):
            self.assertIsNone(next(retry))
        with self.assertRaisesRegex(
            TimeoutError, "reused.*open HTTP response after 2 scheduler steps"
        ):
            next(retry)
        self.assertEqual(self.http.requests, [])
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)
        self.assertFalse(previous._epoch.post_future.done())

    def test_http_response_wait_and_duplicate_retry_share_one_budget(self):
        """Waiting for HTTP cleanup must consume the same step budget as duplicate retries."""
        previous = self._register()
        errors = [_http_error(), _http_error()]
        self.http.replies.extend(_HttpReply(error=error) for error in errors)
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        self.assertIsNone(next(retry))
        self.assertEqual(self.http.requests, [])
        previous._epoch.post_future.set_result(None)
        self.http.scheduler_step += 1

        self.assertIsNone(next(retry))
        self.http.scheduler_step += 1
        with self.assertRaisesRegex(TimeoutError, "after 2 scheduler steps") as raised:
            next(retry)
        self.assertIs(raised.exception.__cause__, errors[-1])
        self.assertEqual([step for _, _, step in self.http.requests], [1, 2])
        self.assertIs(self.ctx._request_epochs["reused"], previous._epoch)

    def test_retry_budget_counts_yields_and_preserves_duplicate_cause(self):
        """Persistent duplicates must terminate after the requested number of scheduler steps."""
        errors = [_http_error() for _ in range(3)]
        self.http.replies.extend(_HttpReply(error=error) for error in errors)
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        for _ in range(2):
            self.assertIsNone(next(retry))
            self.http.scheduler_step += 1
        with self.assertRaisesRegex(TimeoutError, "after 2 scheduler steps") as raised:
            next(retry)
        self.assertIs(raised.exception.__cause__, errors[-1])
        self.assertEqual([step for _, _, step in self.http.requests], [0, 1, 2])
        self.assertNotIn("reused", self.ctx._request_epochs)

    def test_retry_rejects_invalid_step_budgets_before_posting(self):
        """Invalid retry budgets must fail before issuing an HTTP request."""
        for max_steps in (0, -1, True, 1.0):
            with self.subTest(max_steps=max_steps):
                retry = self.ctx.start_req_with_retry(
                    rid="reused", prompt_len=16, max_steps=max_steps
                )
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    next(retry)
        self.assertEqual(self.http.requests, [])

    def test_non_duplicate_errors_are_never_retried(self):
        """Validation, malformed responses, and transport failures must escape without a retry step."""
        malformed = _http_error()
        malformed.message = "not JSON"
        wrong_shape = _http_error()
        wrong_shape.message = '["Duplicate request ID detected: reused"]'
        errors = (
            _http_error(message="Duplicate request ID detected: another"),
            _http_error(message="Duplicate request ID detected: reused (other error)"),
            _http_error(message="Invalid sampling parameters"),
            _http_error(status=500),
            malformed,
            wrong_shape,
            aiohttp.ClientConnectionError("connection refused"),
            TimeoutError("HTTP request timed out"),
        )
        for error in errors:
            with self.subTest(error=error):
                self.http.replies.append(_HttpReply(error=error))
                before = len(self.http.requests)
                retry = self.ctx.start_req_with_retry(
                    rid="reused", prompt_len=16, max_steps=2
                )
                with self.assertRaises(type(error)) as raised:
                    next(retry)
                self.assertEqual(str(raised.exception), str(error))
                self.assertEqual(len(self.http.requests), before + 1)
                self.assertNotIn("reused", self.ctx._request_epochs)

    def test_arrival_timeout_is_not_retried(self):
        """An ambiguous request arrival timeout must not issue another generation request."""
        self.http.replies.append(_HttpReply(hold_response=True))
        retry = self.ctx.start_req_with_retry(rid="reused", prompt_len=16, max_steps=2)
        with self.assertRaisesRegex(TimeoutError, "no request with rid"):
            next(retry)
        self.assertEqual(len(self.http.requests), 1)
        self.assertNotIn("reused", self.ctx._request_epochs)

    def test_abort_message_cannot_count_as_generate_arrival(self):
        """An abort carrying the same ID must not satisfy a new generation's arrival wait."""
        self.http.replies.append(_HttpReply(messages=(AbortReq(rid="reused"),)))
        with self.assertRaisesRegex(TimeoutError, "no request with rid"):
            self.ctx.start_req(rid="reused", prompt_len=16)
        self.assertNotIn("reused", self.ctx._request_epochs)
        self.assertIsInstance(self.proxy.recv_pyobj(), AbortReq)

    def test_repeated_abort_accepts_ack_without_a_second_scheduler_message(self):
        """Idempotent abort acknowledgements must not wait for an IPC message the server omits."""
        handle = self._register()
        self.http.replies.extend(
            (_HttpReply(messages=(AbortReq(rid=handle.rid),)), _HttpReply())
        )
        self.ctx.abort(handle)
        self.ctx.abort(handle)

        self.assertEqual(len(self.http.requests), 2)
        for url, body, _ in self.http.requests:
            self.assertTrue(url.endswith("/abort_request"))
            self.assertEqual(body, {"rid": handle.rid, "abort_all": False})
        self.assertEqual(self.proxy.recv_pyobj().rid, handle.rid)
        with self.assertRaises(zmq.ZMQError):
            self.proxy.recv_pyobj(flags=zmq.NOBLOCK)
        self.assertTrue(handle._epoch.abort_requested)
        self.assertFalse(handle.finished)

    def test_first_abort_requires_matching_request_message(self):
        """Unrelated single-request and abort-all messages must not acknowledge a live target's abort."""
        for message in (AbortReq(rid="other"), AbortReq(rid="reused", abort_all=True)):
            with self.subTest(message=message):
                handle = self._register()
                self.http.replies.append(_HttpReply(messages=(message,)))
                with self.assertRaisesRegex(TimeoutError, "no abort request"):
                    self.ctx.abort(handle)
                self.assertFalse(handle._epoch.abort_requested)
                self.assertIs(self.proxy.recv_pyobj(), message)

    def test_finished_request_abort_still_requires_successful_http_ack(self):
        """Completion permits a missing abort echo only after the abort endpoint acknowledges success."""
        for reply in (
            _HttpReply(error=_http_error(status=500, message="abort failed")),
            _HttpReply(hold_response=True),
        ):
            with self.subTest(reply=reply):
                handle = self._register()
                handle._epoch.post_future.set_result(None)
                self.http.replies.append(reply)
                expected_error = (
                    aiohttp.ClientResponseError
                    if reply.error is not None
                    else TimeoutError
                )
                with self.assertRaises(expected_error):
                    self.ctx.abort(handle)
                self.assertFalse(handle._epoch.abort_requested)

    def test_finished_request_abort_accepts_success_without_echo(self):
        """Aborting an already completed request must succeed when the endpoint sends no IPC message."""
        handle = self._register()
        handle._epoch.post_future.set_result(None)
        self.http.replies.append(_HttpReply())
        self.ctx.abort(handle)
        self.assertTrue(handle._epoch.abort_requested)
        self.assertEqual(len(self.http.requests), 1)
        with self.assertRaises(zmq.ZMQError):
            self.proxy.recv_pyobj(flags=zmq.NOBLOCK)

    def test_stale_handle_abort_cannot_target_reused_id(self):
        """An old handle must not send an abort that could kill the current request with its ID."""
        old = self._register()
        old._epoch.post_future.set_result(None)
        current = self._register()
        self.ctx.abort(old)
        self.assertEqual(self.http.requests, [])
        self.assertFalse(current.finished)

        self.http.replies.append(_HttpReply(messages=(AbortReq(rid=current.rid),)))
        self.ctx.abort(current)
        self.assertEqual(len(self.http.requests), 1)
        self.assertTrue(current._epoch.abort_requested)
        self.assertFalse(old._epoch.abort_requested)

    def test_abort_does_not_hide_other_generation_errors(self):
        """Requesting abort must not turn an unrelated generation failure into normal completion."""
        malformed = _http_error()
        malformed.message = "not JSON"
        errors = (
            _http_error(message="Invalid sampling parameters"),
            _http_error(status=500, message="Aborted"),
            malformed,
            aiohttp.ClientConnectionError("connection reset"),
        )
        for error in errors:
            with self.subTest(error=error):
                handle = self._register()
                handle._epoch.abort_requested = True
                handle._epoch.post_future.set_exception(error)
                with self.assertRaises(type(error)) as raised:
                    _ = handle.finished
                self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
