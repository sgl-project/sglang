"""Regression tests for issue #31766 outbound ZMQ FD exhaustion."""

import enum
import functools
import importlib.util
import json
import os
import resource
import struct
import subprocess
import sys
import threading
import types
import unittest
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zmq
from zmq.utils.monitor import recv_monitor_message

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONN_PATH = _REPO_ROOT / "python/sglang/srt/disaggregation/common/conn.py"
_ENVIRON_PATH = _REPO_ROOT / "python/sglang/srt/environ.py"
_MOONCAKE_PATH = _REPO_ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"


class _Dummy:
    pass


class _DisaggregationMode(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class _KVPoll(enum.IntEnum):
    WaitingForInput = 0
    Bootstrapping = 1
    Transferring = 2
    Success = 3
    Failed = 4


def _module(name, **attributes):
    module = types.ModuleType(name)
    vars(module).update(attributes)
    return module


def _common_import_stubs():
    return {
        "sglang.srt.disaggregation.base.conn": _module(
            "sglang.srt.disaggregation.base.conn",
            BaseKVBootstrapServer=_Dummy,
            BaseKVManager=_Dummy,
            BaseKVReceiver=_Dummy,
            BaseKVSender=_Dummy,
            KVArgs=_Dummy,
            KVPoll=_KVPoll,
            KVTransferMetric=_Dummy,
            StateType=_Dummy,
        ),
        "sglang.srt.disaggregation.utils": _module(
            "sglang.srt.disaggregation.utils",
            DisaggregationMode=_DisaggregationMode,
            filter_kv_indices_for_cp_rank=lambda *args, **kwargs: None,
        ),
        "sglang.srt.distributed": _module(
            "sglang.srt.distributed",
            get_pp_group=lambda: None,
            get_world_group=lambda: None,
        ),
        "sglang.srt.environ": _module(
            "sglang.srt.environ",
            envs=_Dummy(),
        ),
        "sglang.srt.layers.dp_attention": _module(
            "sglang.srt.layers.dp_attention",
            get_attention_dp_rank=lambda: 0,
            get_attention_dp_size=lambda: 1,
        ),
        "sglang.srt.runtime_context": _module(
            "sglang.srt.runtime_context",
            get_parallel=lambda: None,
        ),
        "sglang.srt.server_args": _module(
            "sglang.srt.server_args",
            ServerArgs=_Dummy,
        ),
        "sglang.srt.utils.network": _module(
            "sglang.srt.utils.network",
            NetworkAddress=_Dummy,
            get_local_ip_auto=lambda: "127.0.0.1",
            get_zmq_socket_on_host=lambda *args, **kwargs: None,
        ),
    }


def _load_source(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
    return module


@functools.lru_cache(maxsize=1)
def _load_common_conn():
    return _load_source(
        "_issue31766_common_conn",
        _CONN_PATH,
        _common_import_stubs(),
    )


class _TraceStages:
    def __getattr__(self, _name):
        return SimpleNamespace(stage_name="", level=0)


class _NetworkAddress:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.is_ipv6 = False

    def to_tcp(self):
        return f"tcp://{self.host}:{self.port}"

    def to_host_port_str(self):
        return f"{self.host}:{self.port}"


@functools.lru_cache(maxsize=1)
def _load_mooncake_conn(common_conn):
    def noop_decorator(*args, **kwargs):
        return lambda function: function

    common_utils = _module(
        "sglang.srt.disaggregation.common.utils",
        AuxDataCodec=_Dummy,
        FastQueue=_Dummy,
        TransferKVChunk=_Dummy,
        group_concurrent_contiguous=lambda *args, **kwargs: [],
        pack_int_lists=lambda *args, **kwargs: b"",
        unpack_int_lists=lambda *args, **kwargs: [],
    )
    stubs = {
        **_common_import_stubs(),
        "sglang.srt.disaggregation.common.conn": common_conn,
        "sglang.srt.disaggregation.common.staging_handler": _module(
            "sglang.srt.disaggregation.common.staging_handler",
            DecodeStagingContext=_Dummy,
            PrefillStagingContext=_Dummy,
            StagingRegisterInfo=_Dummy,
            StagingTransferInfo=_Dummy,
        ),
        "sglang.srt.disaggregation.common.utils": common_utils,
        "sglang.srt.disaggregation.mooncake.utils": _module(
            "sglang.srt.disaggregation.mooncake.utils",
            check_mooncake_custom_mem_pool_enabled=lambda: (False, None),
        ),
        "sglang.srt.disaggregation.utils": _module(
            "sglang.srt.disaggregation.utils",
            DisaggregationMode=_DisaggregationMode,
            compute_mamba_state_slice_blocks=lambda *args, **kwargs: [],
        ),
        "sglang.srt.distributed.parallel_state": _module(
            "sglang.srt.distributed.parallel_state",
            get_mooncake_transfer_engine=lambda: None,
        ),
        "sglang.srt.observability.mooncake_trace": _module(
            "sglang.srt.observability.mooncake_trace",
            MooncakeRequestStage=_TraceStages(),
            mooncake_trace_func=noop_decorator,
            mooncake_trace_slice=lambda *args, **kwargs: None,
        ),
        "sglang.srt.observability.trace": _module(
            "sglang.srt.observability.trace",
            TraceNullContext=_Dummy,
            TraceReqContext=_Dummy,
            trace_set_thread_info=lambda *args, **kwargs: None,
        ),
        "sglang.srt.utils.network": _module(
            "sglang.srt.utils.network",
            NetworkAddress=_NetworkAddress,
        ),
    }
    return _load_source(
        "_issue31766_mooncake_conn",
        _MOONCAKE_PATH,
        stubs,
    )


@functools.lru_cache(maxsize=1)
def _load_environ():
    return _load_source("_issue31766_environ", _ENVIRON_PATH, {})


def _new_manager(common_conn, capacity=8, context=None):
    manager = common_conn.CommonKVManager.__new__(common_conn.CommonKVManager)
    manager._zmq_ctx = context or zmq.Context()
    manager._init_socket_cache(capacity=capacity)
    return manager


def _thread_cache(manager):
    return manager._get_thread_socket_cache()


def _close_owned_entries(manager):
    cache = _thread_cache(manager)
    for endpoint, entry in list(cache.items()):
        manager._retire_socket_entry(cache, endpoint, entry)


def _close_manager(manager):
    _close_owned_entries(manager)
    manager._zmq_ctx.term()


def _recv(socket, timeout_ms=3000):
    if not socket.poll(timeout=timeout_ms, flags=zmq.POLLIN):
        raise AssertionError(f"timed out waiting {timeout_ms} ms for a ZMQ message")
    return socket.recv_multipart(flags=zmq.NOBLOCK)


class _CountingContext:
    def __init__(self):
        self.context = zmq.Context()
        self.socket_calls = defaultdict(int)

    def socket(self, socket_type):
        self.socket_calls[socket_type] += 1
        return self.context.socket(socket_type)

    def get(self, option):
        return self.context.get(option)

    def set(self, option, value):
        return self.context.set(option, value)

    def term(self):
        self.context.term()


class _StopAfterServerSocket(Exception):
    pass


class _InitOrderContext:
    def __init__(self, events):
        self.events = events
        self.events.append("context created")

    def get(self, option):
        if option == zmq.SOCKET_LIMIT:
            return 100
        if option == zmq.MAX_SOCKETS:
            return 1
        raise AssertionError(f"unexpected context option: {option}")

    def set(self, option, value):
        if option != zmq.MAX_SOCKETS or value != 5:
            raise AssertionError(f"unexpected context configuration: {option}={value}")
        self.events.append("cache capacity/MAX_SOCKETS configured")

    def socket(self, socket_type):
        if socket_type != zmq.PULL:
            raise AssertionError(f"unexpected first socket type: {socket_type}")
        self.events.append("first server socket created")
        raise _StopAfterServerSocket()


class _FailingPairContext(_CountingContext):
    def __init__(self):
        super().__init__()
        self.push_socket = None

    def socket(self, socket_type):
        self.socket_calls[socket_type] += 1
        if socket_type == zmq.PAIR:
            raise zmq.ZMQError("injected PAIR creation failure")
        socket = self.context.socket(socket_type)
        if socket_type == zmq.PUSH:
            self.push_socket = socket
        return socket


class _NeverTerminalMonitor:
    closed = False

    def recv_multipart(self, _flags):
        raise zmq.Again()

    def close(self, linger=0):
        self.closed = True


class _TerminalMonitor:
    def __init__(self, event):
        self.closed = False
        self.message = [struct.pack("=HI", event, 0), b""]

    def poll(self, timeout, flags):
        return zmq.POLLIN if self.message is not None else 0

    def recv_multipart(self, _flags=0):
        message = self.message
        self.message = None
        return message

    def close(self, linger=0):
        self.closed = True


class _NoopSocket:
    def __init__(self):
        self.closed = False

    def disable_monitor(self):
        pass

    def close(self, linger=0):
        self.closed = True


class _FailOncePush(_NoopSocket):
    def __init__(self, events, fail_close_once):
        super().__init__()
        self.events = events
        self.fail_close_once = fail_close_once
        self.close_calls = 0

    def disable_monitor(self):
        self.events.append("disable_monitor")

    def close(self, linger=0):
        self.events.append("close_PUSH")
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError("injected PUSH close failure")
        self.closed = True


class _FailOnceTerminalMonitor(_TerminalMonitor):
    def __init__(self, event, events, fail_close_once):
        super().__init__(event)
        self.events = events
        self.fail_close_once = fail_close_once
        self.close_calls = 0

    def close(self, linger=0):
        self.events.append("close_PAIR")
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError("injected PAIR close failure")
        self.closed = True


class _CreationSocket:
    def __init__(
        self,
        socket_type,
        events,
        fail_connect=False,
        fail_close_once=False,
    ):
        self.socket_type = socket_type
        self.events = events
        self.fail_connect = fail_connect
        self.fail_close_once = fail_close_once
        self.close_calls = 0
        self.closed = False

    def setsockopt(self, *args):
        pass

    def monitor(self, *args):
        self.events.append("monitor_enabled")

    def disable_monitor(self):
        self.events.append("disable_monitor")

    def connect(self, endpoint):
        if self.socket_type == zmq.PUSH and self.fail_connect:
            self.events.append("endpoint_connect_failed")
            raise zmq.ZMQError("injected endpoint connect failure")

    def recv_multipart(self, _flags):
        raise zmq.Again()

    def close(self, linger=0):
        socket_name = "PUSH" if self.socket_type == zmq.PUSH else "PAIR"
        self.events.append(f"close_{socket_name}")
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError(f"injected {socket_name} close failure")
        self.closed = True


class _RollbackContext:
    def __init__(self):
        self.events = []
        self.fail_next_creation = True
        self.current_creation_fails = False
        self.sockets = []

    def get(self, option):
        if option == zmq.SOCKET_LIMIT:
            return 100
        if option == zmq.MAX_SOCKETS:
            return 100
        raise AssertionError(f"unexpected context option: {option}")

    def set(self, option, value):
        raise AssertionError("the fake context already has sufficient capacity")

    def socket(self, socket_type):
        if socket_type == zmq.PUSH:
            self.current_creation_fails = self.fail_next_creation
            self.fail_next_creation = False
            socket = _CreationSocket(
                socket_type,
                self.events,
                fail_connect=self.current_creation_fails,
            )
        else:
            socket = _CreationSocket(
                socket_type,
                self.events,
                fail_close_once=self.current_creation_fails,
            )
        self.sockets.append(socket)
        return socket

    def term(self):
        pass


class _SequenceQueue:
    def __init__(self, *items):
        self.items = iter(items)

    def get(self):
        try:
            return next(self.items)
        except StopIteration:
            raise SystemExit()


class _RecordingSocket:
    def __init__(self, messages):
        self.messages = messages

    def send_multipart(self, parts, *args, **kwargs):
        self.messages.append((parts, args, kwargs))


def _fd_child():
    common_conn = _load_common_conn()
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 256 if soft_limit == resource.RLIM_INFINITY else min(soft_limit, 256)
    if hard_limit != resource.RLIM_INFINITY:
        target = min(target, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard_limit))

    manager = _new_manager(common_conn, capacity=8)
    initial_fds = len(os.listdir("/proc/self/fd"))
    delivered = 0
    seen = set()
    error = None
    try:
        for index in range(80):
            receiver = manager._zmq_ctx.socket(zmq.PULL)
            try:
                while True:
                    port = receiver.bind_to_random_port("tcp://127.0.0.1")
                    endpoint = f"tcp://127.0.0.1:{port}"
                    if endpoint not in seen:
                        seen.add(endpoint)
                        break
                    receiver.unbind(endpoint)
                payload = f"probe-{index}".encode()
                manager._connect(endpoint).send_multipart([payload])
                assert _recv(receiver) == [payload]
                delivered += 1
                monitor = _thread_cache(manager)[endpoint].monitor
                receiver.close(linger=0)
                receiver = None
                if not monitor.poll(timeout=3000, flags=zmq.POLLIN):
                    raise TimeoutError(f"missing terminal event for {endpoint}")
            finally:
                if receiver is not None:
                    receiver.close(linger=0)
    except Exception as exception:
        error = [type(exception).__name__, str(exception)]

    result = {
        "error": error,
        "delivered": delivered,
        "initial_fds": initial_fds,
        "final_fds": len(os.listdir("/proc/self/fd")),
        "cached": manager._socket_cache_size,
        "limit": target,
    }
    _close_manager(manager)
    return result


class TestCommonKVManagerSocketCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common_conn = _load_common_conn()

    def test_socket_limit_is_configured_before_first_server_socket(self):
        events = []
        context = _InitOrderContext(events)
        args = SimpleNamespace(
            kv_item_lens=[],
            state_item_lens=[],
            is_hybrid_mla_backend=False,
            system_dp_rank=0,
            pp_rank=0,
        )
        server_args = SimpleNamespace(
            host="127.0.0.1",
            disaggregation_bootstrap_port=8999,
            dist_init_addr="127.0.0.1:8998",
            enable_dp_attention=False,
            dp_size=1,
            pp_size=1,
            enable_dsa_cache_layer_split=False,
        )
        parallel = SimpleNamespace(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
        )

        def create_first_server_socket(zmq_context, socket_type, host):
            self.assertIs(zmq_context, context)
            self.assertEqual(host, "127.0.0.1")
            zmq_context.socket(socket_type)

        with (
            patch.object(self.common_conn.zmq, "Context", return_value=context),
            patch.object(self.common_conn, "get_parallel", return_value=parallel),
            patch.object(
                self.common_conn,
                "get_zmq_socket_on_host",
                side_effect=create_first_server_socket,
            ),
            patch.object(
                self.common_conn.envs,
                "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER",
                SimpleNamespace(get=lambda: False),
                create=True,
            ),
            patch.object(
                self.common_conn.envs,
                "SGLANG_DISAGGREGATION_MAX_CACHED_ZMQ_ENDPOINTS",
                SimpleNamespace(get=lambda: 2),
                create=True,
            ),
            self.assertRaises(_StopAfterServerSocket),
        ):
            self.common_conn.CommonKVManager(
                args,
                _DisaggregationMode.DECODE,
                server_args,
            )

        self.assertEqual(
            events,
            [
                "context created",
                "cache capacity/MAX_SOCKETS configured",
                "first server socket created",
            ],
        )

    def test_real_context_honors_raised_pre_socket_limit(self):
        context = zmq.Context()
        sockets = []
        try:
            if context.get(zmq.SOCKET_LIMIT) < 4:
                self.skipTest("libzmq socket limit is too low for the probe")
            try:
                context.set(zmq.MAX_SOCKETS, 1)
                context.set(zmq.MAX_SOCKETS, 4)
            except zmq.ZMQError as error:
                self.skipTest(f"libzmq does not allow the controlled probe: {error}")
            sockets = [context.socket(zmq.PAIR) for _ in range(4)]
            self.assertEqual(len(sockets), 4)
        finally:
            for socket in sockets:
                socket.close(linger=0)
            context.term()

    def test_real_pyzmq_disable_monitor_api_and_cleanup(self):
        initial_fds = len(os.listdir("/proc/self/fd"))
        initial_threads = len(threading.enumerate())
        context = zmq.Context()
        push = context.socket(zmq.PUSH)
        pair = context.socket(zmq.PAIR)
        monitor_endpoint = f"inproc://disable-monitor-{id(push)}"
        push.monitor(monitor_endpoint, zmq.EVENT_ALL)
        pair.connect(monitor_endpoint)

        # Socket.disable_monitor() is documented in PyZMQ >= 14.4.
        push.disable_monitor()
        pair.close(linger=0)
        push.close(linger=0)
        self.assertTrue(pair.closed)
        self.assertTrue(push.closed)
        context.term()

        self.assertLessEqual(len(os.listdir("/proc/self/fd")), initial_fds)
        self.assertEqual(len(threading.enumerate()), initial_threads)

    def test_unique_endpoint_churn_keeps_real_fds_bounded(self):
        process = subprocess.run(
            [sys.executable, __file__, "--fd-child"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = json.loads(process.stdout)
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["delivered"], 80, result)
        self.assertLessEqual(result["cached"], 1, result)
        self.assertLessEqual(result["final_fds"] - result["initial_fds"], 40, result)

    def test_stable_endpoint_reuses_raw_socket_and_send_semantics(self):
        manager = _new_manager(self.common_conn)
        receiver = manager._zmq_ctx.socket(zmq.PULL)
        port = receiver.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        try:
            socket = manager._connect(endpoint)
            result = socket.send_multipart([b"first", b"part"], flags=0, copy=True)
            self.assertIsNone(result)
            self.assertEqual(_recv(receiver), [b"first", b"part"])
            self.assertIs(manager._connect(endpoint), socket)
            self.assertEqual(manager._socket_cache_size, 1)
        finally:
            receiver.close(linger=0)
            _close_manager(manager)

    def test_capacity_fails_before_new_fds_and_preserves_existing_endpoint(self):
        context = _CountingContext()
        manager = _new_manager(self.common_conn, capacity=1, context=context)
        receiver_a = manager._zmq_ctx.socket(zmq.PULL)
        receiver_b = manager._zmq_ctx.socket(zmq.PULL)
        port_a = receiver_a.bind_to_random_port("tcp://127.0.0.1")
        port_b = receiver_b.bind_to_random_port("tcp://127.0.0.1")
        endpoint_a = f"tcp://127.0.0.1:{port_a}"
        endpoint_b = f"tcp://127.0.0.1:{port_b}"
        try:
            manager._connect(endpoint_a).send_multipart([b"warmup"])
            self.assertEqual(_recv(receiver_a), [b"warmup"])
            before_calls = dict(context.socket_calls)
            before_fds = len(os.listdir("/proc/self/fd"))

            with self.assertRaises(self.common_conn.SocketCacheCapacityError) as caught:
                manager._connect(endpoint_b)

            message = str(caught.exception)
            self.assertIn(f"endpoint={endpoint_b}", message)
            self.assertIn("limit=1", message)
            self.assertIn("cached=1", message)
            self.assertIn("SGLANG_DISAGGREGATION_MAX_CACHED_ZMQ_ENDPOINTS", message)
            self.assertEqual(dict(context.socket_calls), before_calls)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)
            self.assertEqual(manager._socket_cache_size, 1)

            manager._connect(endpoint_a).send_multipart([b"still-live"])
            self.assertEqual(_recv(receiver_a), [b"still-live"])
        finally:
            receiver_a.close(linger=0)
            receiver_b.close(linger=0)
            _close_manager(manager)

    def test_disconnected_and_closed_entries_release_capacity_and_can_send(self):
        for expected_event in (zmq.EVENT_DISCONNECTED, zmq.EVENT_CLOSED):
            with self.subTest(event=expected_event):
                manager = _new_manager(self.common_conn, capacity=1)
                receiver = None
                replacement = None
                cache = _thread_cache(manager)
                try:
                    if expected_event == zmq.EVENT_DISCONNECTED:
                        receiver = manager._zmq_ctx.socket(zmq.PULL)
                        port = receiver.bind_to_random_port("tcp://127.0.0.1")
                        old_endpoint = f"tcp://127.0.0.1:{port}"
                        manager._connect(old_endpoint).send_multipart([b"old"])
                        self.assertEqual(_recv(receiver), [b"old"])
                        old_entry = cache[old_endpoint]
                        receiver.close(linger=0)
                        receiver = None
                    else:
                        old_endpoint = "tcp://127.0.0.1:closed"
                        old_entry = self.common_conn._SocketCacheEntry(
                            _NoopSocket(), _TerminalMonitor(expected_event)
                        )
                        cache[old_endpoint] = old_entry
                        manager._socket_cache_size = 1

                    self.assertTrue(
                        old_entry.monitor.poll(timeout=3000, flags=zmq.POLLIN),
                        "missing terminal event",
                    )
                    event = recv_monitor_message(old_entry.monitor, flags=zmq.NOBLOCK)
                    self.assertEqual(event["event"], expected_event)

                    manager._retire_socket_entry(cache, old_endpoint, old_entry)
                    self.assertEqual(manager._socket_cache_size, 0)

                    replacement = manager._zmq_ctx.socket(zmq.PULL)
                    new_port = replacement.bind_to_random_port("tcp://127.0.0.1")
                    new_endpoint = f"tcp://127.0.0.1:{new_port}"
                    manager._connect(new_endpoint).send_multipart([b"new"])

                    self.assertEqual(_recv(replacement), [b"new"])
                    self.assertTrue(old_entry.socket.closed)
                    self.assertNotIn(old_endpoint, cache)
                    self.assertIn(new_endpoint, cache)
                    self.assertEqual(manager._socket_cache_size, 1)
                finally:
                    if receiver is not None:
                        receiver.close(linger=0)
                    if replacement is not None:
                        replacement.close(linger=0)
                    _close_manager(manager)

    def test_nonterminal_attempt_retains_slot_with_operator_guidance(self):
        manager = _new_manager(self.common_conn, capacity=1)
        cache = _thread_cache(manager)
        endpoint = "tcp://127.0.0.1:unreachable"
        entry = self.common_conn._SocketCacheEntry(
            _NoopSocket(), _NeverTerminalMonitor()
        )
        cache[endpoint] = entry
        manager._socket_cache_size = 1
        try:
            with self.assertRaises(self.common_conn.SocketCacheCapacityError) as caught:
                manager._connect("tcp://127.0.0.1:next")
            self.assertIs(cache[endpoint], entry)
            self.assertIn("drain and restart this prefill", str(caught.exception))
            self.assertIn("Increase", str(caught.exception))
        finally:
            _close_manager(manager)

    def test_pair_creation_failure_rolls_back_push_and_capacity(self):
        context = _FailingPairContext()
        manager = _new_manager(self.common_conn, context=context)
        with self.assertRaisesRegex(zmq.ZMQError, "injected"):
            manager._connect("tcp://127.0.0.1:20000")
        self.assertEqual(manager._socket_cache_size, 0)
        self.assertEqual(_thread_cache(manager), {})
        self.assertTrue(context.push_socket.closed)
        manager._zmq_ctx.term()

    def _assert_terminal_cleanup_retries(self, fail_target):
        manager = _new_manager(self.common_conn, capacity=1)
        cache = _thread_cache(manager)
        endpoint = "tcp://127.0.0.1:20001"
        events = []
        push = _FailOncePush(events, fail_close_once=fail_target == "PUSH")
        monitor = _FailOnceTerminalMonitor(
            zmq.EVENT_CLOSED,
            events,
            fail_close_once=fail_target == "PAIR",
        )
        old_entry = self.common_conn._SocketCacheEntry(push, monitor, endpoint)
        replacement = self.common_conn._SocketCacheEntry(
            _NoopSocket(),
            _NeverTerminalMonitor(),
            endpoint,
        )
        cache[endpoint] = old_entry
        manager._socket_cache_size = 1

        try:
            with (
                patch.object(
                    manager,
                    "_release_socket_slot",
                    wraps=manager._release_socket_slot,
                ) as release_slot,
                patch.object(
                    manager,
                    "_create_socket_entry",
                    return_value=replacement,
                ) as create_entry,
            ):
                with self.assertRaises(
                    self.common_conn.SocketCacheCleanupError
                ) as caught:
                    manager._connect(endpoint)

                self.assertIn("capacity slot remains reserved", str(caught.exception))
                self.assertIn("drain and restart", str(caught.exception))
                self.assertTrue(old_entry.retiring)
                self.assertTrue(old_entry.cleanup_pending)
                self.assertIs(cache[endpoint], old_entry)
                self.assertEqual(manager._socket_cache_size, 1)
                self.assertEqual(release_slot.call_count, 0)
                self.assertEqual(create_entry.call_count, 0)
                self.assertEqual(
                    events[:3],
                    ["disable_monitor", "close_PAIR", "close_PUSH"],
                )

                returned = manager._connect(endpoint)
                self.assertIs(returned, replacement.socket)
                self.assertIs(cache[endpoint], replacement)
                self.assertIsNot(cache[endpoint], old_entry)
                self.assertTrue(old_entry.socket.closed)
                self.assertTrue(old_entry.monitor.closed)
                self.assertTrue(old_entry.slot_released)
                self.assertEqual(release_slot.call_count, 1)
                self.assertEqual(create_entry.call_count, 1)
                self.assertEqual(manager._socket_cache_size, 1)

                manager._retry_pending_socket_cleanup(cache)
                manager._retry_pending_socket_cleanup(cache)
                self.assertEqual(release_slot.call_count, 1)
                self.assertEqual(manager._socket_cache_size, 1)

                manager._retire_socket_entry(cache, endpoint, replacement)
                manager._retire_socket_entry(cache, endpoint, replacement)
                self.assertEqual(release_slot.call_count, 2)
                self.assertEqual(manager._socket_cache_size, 0)
        finally:
            manager._zmq_ctx.term()

    def test_monitor_cleanup_failure_retries_without_reusing_closed_socket(self):
        self._assert_terminal_cleanup_retries("PAIR")

    def test_push_cleanup_failure_retries_without_reusing_closed_socket(self):
        self._assert_terminal_cleanup_retries("PUSH")

    def test_creation_rollback_cleanup_retries_and_releases_once(self):
        context = _RollbackContext()
        manager = _new_manager(self.common_conn, capacity=1, context=context)
        first_endpoint = "tcp://127.0.0.1:20002"
        second_endpoint = "tcp://127.0.0.1:20003"
        cache = _thread_cache(manager)

        with patch.object(
            manager,
            "_release_socket_slot",
            wraps=manager._release_socket_slot,
        ) as release_slot:
            with self.assertRaises(self.common_conn.SocketCacheCleanupError) as caught:
                manager._connect(first_endpoint)

            pending = manager._get_thread_pending_socket_cleanup()
            self.assertEqual(len(pending), 1)
            partial_entry = pending[0]
            self.assertTrue(partial_entry.retiring)
            self.assertTrue(partial_entry.cleanup_pending)
            self.assertTrue(partial_entry.socket.closed)
            self.assertFalse(partial_entry.monitor.closed)
            self.assertEqual(manager._socket_cache_size, 1)
            self.assertEqual(release_slot.call_count, 0)
            self.assertEqual(cache, {})
            self.assertIn("capacity slot remains reserved", str(caught.exception))

            replacement_socket = manager._connect(second_endpoint)
            self.assertIs(replacement_socket, cache[second_endpoint].socket)
            self.assertEqual(pending, [])
            self.assertTrue(partial_entry.monitor.closed)
            self.assertTrue(partial_entry.slot_released)
            self.assertEqual(release_slot.call_count, 1)
            self.assertEqual(manager._socket_cache_size, 1)

            manager._retry_pending_socket_cleanup(cache)
            manager._retry_pending_socket_cleanup(cache)
            self.assertEqual(release_slot.call_count, 1)
            self.assertEqual(manager._socket_cache_size, 1)

            manager._retire_socket_entry(
                cache,
                second_endpoint,
                cache[second_endpoint],
            )
            self.assertEqual(release_slot.call_count, 2)
            self.assertEqual(manager._socket_cache_size, 0)
        manager._zmq_ctx.term()

    def test_each_sender_thread_owns_socket_and_multipart_stays_intact(self):
        manager = _new_manager(self.common_conn, capacity=8)
        receiver = manager._zmq_ctx.socket(zmq.PULL)
        port = receiver.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        release = threading.Event()
        ready = threading.Barrier(5)
        errors = []
        ownership = []

        def send(worker):
            try:
                socket = manager._connect(endpoint)
                entry = _thread_cache(manager)[endpoint]
                ownership.append(
                    (threading.get_ident(), entry.owner_thread_id, id(socket))
                )
                ready.wait(timeout=5)
                for sequence in range(5):
                    socket.send_multipart(
                        [
                            str(worker).encode(),
                            str(sequence).encode(),
                            f"{worker}:{sequence}".encode(),
                        ]
                    )
                release.wait(timeout=5)
            except Exception as exception:
                errors.append(exception)
            finally:
                _close_owned_entries(manager)

        threads = [threading.Thread(target=send, args=(worker,)) for worker in range(4)]
        try:
            for thread in threads:
                thread.start()
            ready.wait(timeout=5)
            received = {tuple(_recv(receiver)) for _ in range(20)}
            expected = {
                (
                    str(worker).encode(),
                    str(sequence).encode(),
                    f"{worker}:{sequence}".encode(),
                )
                for worker in range(4)
                for sequence in range(5)
            }
            self.assertEqual(received, expected)
            self.assertEqual(manager._socket_cache_size, 4)
            self.assertTrue(all(owner == creator for creator, owner, _ in ownership))
            self.assertEqual(len({socket_id for _, _, socket_id in ownership}), 4)
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=5)
            receiver.close(linger=0)
            manager._zmq_ctx.term()
        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(manager._socket_cache_size, 0)

    def test_environment_default_invalid_and_nonpositive_values(self):
        environ = _load_environ()
        field = environ.Envs.SGLANG_DISAGGREGATION_MAX_CACHED_ZMQ_ENDPOINTS
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(field.name, None)
            self.assertIsNone(field.get())
            os.environ[field.name] = "not-an-int"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertIsNone(field.get())
            self.assertIn("using default", str(caught[0].message))

        context = zmq.Context()
        manager = self.common_conn.CommonKVManager.__new__(
            self.common_conn.CommonKVManager
        )
        manager._zmq_ctx = context
        field_stub = SimpleNamespace(get=lambda: None)
        soft_limit = 65535
        with (
            patch.object(
                self.common_conn.envs,
                "SGLANG_DISAGGREGATION_MAX_CACHED_ZMQ_ENDPOINTS",
                field_stub,
                create=True,
            ),
            patch.object(
                self.common_conn.resource,
                "getrlimit",
                return_value=(soft_limit, soft_limit),
            ),
        ):
            manager._init_socket_cache()
        expected = min(
            soft_limit
            // (
                self.common_conn._OUTBOUND_CACHE_RESOURCE_DIVISOR
                * self.common_conn._ESTIMATED_FDS_PER_CACHE_ENTRY
            ),
            (
                context.get(zmq.SOCKET_LIMIT)
                - self.common_conn._RESERVED_MANAGER_CONTEXT_SOCKETS
            )
            // (
                self.common_conn._OUTBOUND_CACHE_RESOURCE_DIVISOR
                * self.common_conn._ZMQ_SOCKETS_PER_CACHE_ENTRY
            ),
        )
        self.assertEqual(manager._socket_cache_capacity, expected)
        self.assertGreaterEqual(expected, 2 * (4 + 1) * 64)
        self.assertGreaterEqual(
            context.get(zmq.MAX_SOCKETS),
            expected * self.common_conn._ZMQ_SOCKETS_PER_CACHE_ENTRY + 1,
        )
        context.term()

        for value in (0, -1):
            manager = self.common_conn.CommonKVManager.__new__(
                self.common_conn.CommonKVManager
            )
            with self.assertRaisesRegex(ValueError, f"got {value}"):
                manager._init_socket_cache(capacity=value)


class TestMooncakeCapacityRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common_conn = _load_common_conn()
        cls.mooncake = _load_mooncake_conn(cls.common_conn)

    def _assert_socket_cache_failure_recovers(self, error_type):
        manager = self.mooncake.MooncakeKVManager.__new__(
            self.mooncake.MooncakeKVManager
        )
        statuses = {
            1: _KVPoll.WaitingForInput,
            2: _KVPoll.WaitingForInput,
        }
        failures = {}
        messages = []
        manager.request_status = statuses
        manager.check_status = lambda room: statuses[room]
        manager.update_status = lambda room, status: statuses.__setitem__(room, status)
        manager.record_failure = lambda room, reason: failures.__setitem__(room, reason)
        manager.transfer_infos = {
            room: {
                f"session-{room}": SimpleNamespace(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=9000 + room,
                    mooncake_session_id=f"session-{room}",
                    dst_kv_indices=[],
                    required_dst_info_num=1,
                    is_dummy=False,
                )
            }
            for room in statuses
        }
        manager.req_to_decode_prefix_len = {1: 0, 2: 0}
        manager.decode_kv_args_table = {
            f"session-{room}": SimpleNamespace(dst_attn_tp_size=1, dst_aux_ptrs=[])
            for room in statuses
        }
        manager.enable_trace = False
        manager.enable_staging = False
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 1
        manager.attn_tp_rank = 0
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.send_aux = lambda *args, **kwargs: 0

        def connect(endpoint, is_ipv6=False):
            if endpoint.endswith(":9001"):
                raise error_type(
                    "endpoint=tcp://127.0.0.1:9001, limit=1, cached=1; increase limit"
                )
            return _RecordingSocket(messages)

        manager._connect = connect
        chunks = [
            SimpleNamespace(
                room=room,
                prefill_kv_indices=[],
                index_slice=slice(0, 0),
                is_last_chunk=True,
                prefill_aux_index=0,
                state_indices=[],
            )
            for room in statuses
        ]

        with self.assertRaises(SystemExit):
            manager.transfer_worker(_SequenceQueue(*chunks), executor=None)

        self.assertEqual(statuses[1], _KVPoll.Failed)
        self.assertEqual(statuses[2], _KVPoll.Success)
        self.assertEqual(set(failures), {1})
        self.assertNotIn(1, manager.transfer_infos)
        self.assertNotIn(2, manager.transfer_infos)
        self.assertNotIn(1, manager.req_to_decode_prefix_len)
        self.assertNotIn(2, manager.req_to_decode_prefix_len)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0][0][0], b"2")

    def test_capacity_failure_cleans_only_room_and_worker_processes_next_item(self):
        self._assert_socket_cache_failure_recovers(
            self.common_conn.SocketCacheCapacityError
        )

    def test_cleanup_failure_cleans_only_room_and_worker_processes_next_item(self):
        self._assert_socket_cache_failure_recovers(
            self.common_conn.SocketCacheCleanupError
        )


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--fd-child":
        print(json.dumps(_fd_child()))
    else:
        unittest.main()
