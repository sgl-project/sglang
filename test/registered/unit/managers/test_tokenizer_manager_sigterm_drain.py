"""SIGTERM drain of the default HTTP server and requests that arrive late.

Once the drain has seen no request left and sent ShutdownReq, no scheduler
answers again. A request that reaches the TokenizerManager after that point,
for example one sent on a kept-alive connection while the drain blocks the
event loop waiting for the schedulers to exit, must still get an answer, and
the server process must still exit on its own.

The server cases fork a process that runs the real uvicorn.Server, built as the
default single-tokenizer branch of http_server builds it, on the real
http_server.app, with the real TokenizerManager drain (SignalHandler,
sigterm_watchdog) and request path (generate_request, _init_req_state,
_wait_one_response). Stand-ins replace only the scheduler side: tokenization,
the two sends to the scheduler, the detokenizer receive loop, and the scheduler
processes the drain waits for. The parent is the client and sends SIGTERM.
"""

import asyncio
import http.client
import json
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import uvicorn
from fastapi import HTTPException

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.tokenizer_manager as tokenizer_manager_module  # noqa: E402
from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput, ShutdownReq  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ServerStatus,
    SignalHandler,
    TokenizerManager,
)
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# Longer than the 5 s default, so an idle pooled connection stays open while a
# case waits for the drain, which checks every 5 s.
_KEEP_ALIVE_S = 60
_DRAIN_TIMEOUT_S = 30
_REPLY_TIMEOUT_S = 10
_EXIT_TIMEOUT_S = 10
_HOLDER_RID = "holder"
_EVENTS = (
    "drain_blocked",
    "scheduler_may_exit",
    "drain_waiting",
    "holder_sent",
    "release_holder",
)
_CONNECTION_ERRORS = ("ConnectionResetError", "RemoteDisconnected", "BrokenPipeError")


class _FakeScheduler:
    """The scheduler and detokenizer processes and the sockets to them.

    Before ShutdownReq it finishes each request the way _handle_batch_output
    finishes a non-streaming one; after ShutdownReq it answers nothing, and it
    counts as a live scheduler process until the case sets scheduler_may_exit.
    """

    def __init__(self, tm, events):
        self.tm = tm
        self.events = events
        self.stopping = False
        self.received = []
        self.tasks = set()

    async def send_request(self, tokenized_obj):  # replaces _send_one_request
        rid = tokenized_obj.rid
        self.received.append(rid)
        if self.stopping:
            return
        if rid == _HOLDER_RID:
            self.events["holder_sent"].set()
            task = asyncio.get_running_loop().create_task(self._finish_on_release(rid))
            self.tasks.add(task)
        else:
            asyncio.get_running_loop().call_soon(self._finish, rid)

    def dispatch(self, obj):  # replaces _dispatch_to_scheduler
        if isinstance(obj, ShutdownReq):
            self.stopping = True

    def live_processes(self):  # replaces collect_scheduler_processes
        if self.stopping:
            self.events["drain_blocked"].set()
            if self.events["scheduler_may_exit"].is_set():
                return []
        return [SimpleNamespace(pid=0)]

    async def _finish_on_release(self, rid):
        while not self.events["release_holder"].is_set():
            await asyncio.sleep(0.05)
        self._finish(rid)

    def _finish(self, rid):
        state = self.tm.rid_to_state.pop(rid, None)
        if state is None:
            return
        state.out_list.append(
            {
                "text": "",
                "output_ids": [0],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "length", "length": 1},
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                },
            }
        )
        state.finished = True
        state.event.set()


async def _tokenize_one_request(obj):
    # Stand-in for the tokenizer: the requests already carry input_ids.
    return obj


async def _receive_loop():
    # Stand-in for handle_loop: outputs come from _FakeScheduler instead.
    await asyncio.Event().wait()


def _make_tokenizer_manager(events):
    """A TokenizerManager without __init__ (model, sockets, subprocesses)."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.elastic_worker_count = 1
    tm.init_running_status()
    tm.init_weight_update()
    tm.server_status = ServerStatus.Up
    tm.signal_handler_class = SignalHandler
    tm.disaggregation_mode = DisaggregationMode.NULL
    tm.enable_priority_scheduling = False
    tm.default_priority_value = None
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.tokenizer = None
    tm.request_logger = Mock()
    tm.request_metrics_exporter_manager = SimpleNamespace(
        exporter_enabled=lambda: False
    )
    scheduler = _FakeScheduler(tm, events)
    tm._tokenize_one_request = _tokenize_one_request
    tm._send_one_request = scheduler.send_request
    tm._dispatch_to_scheduler = scheduler.dispatch
    tm.handle_loop = _receive_loop
    return tm, scheduler


class _DrainLogWatch(logging.Handler):
    """Sets an event when a drain check reports requests still in flight."""

    _PREFIX = "Gracefully exiting... Remaining number of requests "

    def __init__(self, event):
        super().__init__(level=logging.INFO)
        self.event = event

    def emit(self, record):
        message = record.getMessage()
        if message.startswith(self._PREFIX):
            if int(message[len(self._PREFIX) :].split(".")[0]) > 0:
                self.event.set()


def _serve(port, events):
    """Server process: the default single-tokenizer HTTP server."""
    tm, scheduler = _make_tokenizer_manager(events)
    tokenizer_manager_module.collect_scheduler_processes = scheduler.live_processes
    tm_logger = logging.getLogger(tokenizer_manager_module.__name__)
    tm_logger.setLevel(logging.INFO)
    tm_logger.addHandler(_DrainLogWatch(events["drain_waiting"]))
    http_server.set_global_state(
        http_server._GlobalState(
            tokenizer_manager=tm, template_manager=None, scheduler_info={}
        )
    )
    # The lifespan (warmup, gRPC, sidecar) is left out; the drain does not use it.
    server = uvicorn.Server(
        uvicorn.Config(
            http_server.app,
            host="127.0.0.1",
            port=port,
            timeout_keep_alive=_KEEP_ALIVE_S,
            loop="uvloop",
            lifespan="off",
        )
    )
    tm.set_server_stop_hook(lambda: setattr(server, "should_exit", True))
    server.run()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _ServerProcess:
    def __init__(self, case):
        ctx = mp.get_context("fork")
        self.events = {name: ctx.Event() for name in _EVENTS}
        self.port = _free_port()
        self.process = ctx.Process(target=_serve, args=(self.port, self.events))
        self.process.start()
        case.addCleanup(self.kill)
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                return
            except OSError:
                if time.monotonic() > deadline or not self.process.is_alive():
                    raise
                time.sleep(0.05)

    def connect(self):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=_REPLY_TIMEOUT_S
        )
        conn.connect()
        return conn

    def sigterm(self):
        os.kill(self.process.pid, signal.SIGTERM)

    def exit_code(self, timeout):
        self.process.join(timeout)
        return self.process.exitcode

    def kill(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(5)


def _send(conn, rid):
    body = json.dumps(
        {"rid": rid, "input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 1}}
    )
    conn.request(
        "POST", "/generate", body=body, headers={"Content-Type": "application/json"}
    )


def _reply(conn):
    """The reply's status, the error that ended the connection, or None on timeout."""
    try:
        response = conn.getresponse()
        response.read()
        return response.status
    except TimeoutError:
        return None
    except (ConnectionError, http.client.HTTPException) as e:
        return type(e).__name__


class TestSigtermDrainServer(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _start(self):
        server = _ServerProcess(self)
        conn = server.connect()
        self.addCleanup(conn.close)
        # The first request installs the SIGTERM handler and starts the drain watchdog.
        for i in range(2):
            _send(conn, f"warmup-{i}")
            self.assertEqual(_reply(conn), HTTPStatus.OK)
        return server, conn

    def test_late_request_on_kept_alive_connection_is_answered(self):
        """A request written on an established connection while the drain waits
        for the schedulers after ShutdownReq is refused, and the server exits."""
        server, conn = self._start()
        local_addr = conn.sock.getsockname()
        server.sigterm()
        self.assertTrue(server.events["drain_blocked"].wait(_DRAIN_TIMEOUT_S))
        # The event loop is blocked: the request waits unread in the socket.
        _send(conn, "late")
        self.assertEqual(conn.sock.getsockname(), local_addr)
        server.events["scheduler_may_exit"].set()
        reply = _reply(conn)
        exit_code = server.exit_code(_EXIT_TIMEOUT_S)
        # (None, None): no reply at all, and the server process still running.
        self.assertEqual((reply, exit_code), (HTTPStatus.SERVICE_UNAVAILABLE, 0))

    def test_server_exits_when_nothing_arrives_after_sigterm(self):
        server, _ = self._start()
        server.events["scheduler_may_exit"].set()
        server.sigterm()
        self.assertEqual(server.exit_code(_DRAIN_TIMEOUT_S), 0)

    def test_late_request_on_new_connection_is_not_admitted(self):
        """A connection opened while the drain blocks the loop is closed by the
        shutdown or refused; it never hangs the server."""
        server, _ = self._start()
        server.sigterm()
        self.assertTrue(server.events["drain_blocked"].wait(_DRAIN_TIMEOUT_S))
        late = server.connect()
        self.addCleanup(late.close)
        _send(late, "late-new")
        server.events["scheduler_may_exit"].set()
        outcome = _reply(late)
        self.assertIn(outcome, _CONNECTION_ERRORS + (HTTPStatus.SERVICE_UNAVAILABLE,))
        self.assertEqual(server.exit_code(_EXIT_TIMEOUT_S), 0)

    def test_request_during_drain_is_served(self):
        """While a request is still in flight the drain keeps admitting and
        serving requests, then exits once the last one finished."""
        server, conn = self._start()
        holder = server.connect()
        self.addCleanup(holder.close)
        _send(holder, _HOLDER_RID)
        self.assertTrue(server.events["holder_sent"].wait(_REPLY_TIMEOUT_S))
        server.events["scheduler_may_exit"].set()
        server.sigterm()
        self.assertTrue(server.events["drain_waiting"].wait(_DRAIN_TIMEOUT_S))
        _send(conn, "during-drain")
        self.assertEqual(_reply(conn), HTTPStatus.OK)
        server.events["release_holder"].set()
        self.assertEqual(_reply(holder), HTTPStatus.OK)
        self.assertEqual(server.exit_code(_DRAIN_TIMEOUT_S), 0)


class TestGenerateAfterShutdownReq(CustomTestCase):
    def test_request_reaching_admission_after_shutdown_req_is_refused(self):
        """Whatever delayed a request on its way in (a blocked loop, an executor,
        the HTTP layer), once the drain has sent ShutdownReq the request is
        refused before it is registered or sent to a scheduler."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        events = {name: threading.Event() for name in _EVENTS}
        events["scheduler_may_exit"].set()

        async def drive():
            tm, scheduler = _make_tokenizer_manager(events)
            # No signal handlers or watchdog task in the test process.
            tm.event_loop = asyncio.get_running_loop()
            stop_hook = Mock()
            tm.set_server_stop_hook(stop_hook)
            tm.gracefully_exit = True
            with (
                patch.object(
                    tokenizer_manager_module,
                    "collect_scheduler_processes",
                    scheduler.live_processes,
                ),
                patch.object(tokenizer_manager_module, "kill_process_tree"),
            ):
                await tm.sigterm_watchdog()
            stop_hook.assert_called_once()

            obj = GenerateReqInput(
                rid="late", input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1}
            )
            task = asyncio.ensure_future(tm.generate_request(obj).__anext__())
            await asyncio.wait({task}, timeout=_REPLY_TIMEOUT_S)
            registered, sent = list(tm.rid_to_state), list(scheduler.received)
            status = None
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            elif isinstance(task.exception(), HTTPException):
                status = task.exception().status_code
            else:
                task.result()
            # (None, ["late"], ["late"]): registered, sent to the stopped
            # scheduler, and never answered.
            self.assertEqual(
                (status, registered, sent), (HTTPStatus.SERVICE_UNAVAILABLE, [], [])
            )

        asyncio.run(drive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
