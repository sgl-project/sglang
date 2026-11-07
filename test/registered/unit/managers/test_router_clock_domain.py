import ipaddress
import multiprocessing as mp
import os
import pickle
import socket
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import zmq
from prometheus_client import CollectorRegistry, Histogram

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers import data_parallel_controller as dp_module
from sglang.srt.managers import delivery_clock, io_struct
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter
from sglang.srt.managers.scheduler_components import ipc_channels
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels

register_cpu_ci(est_time=35, suite="base-a-test-cpu")

_DOMAIN = bytes(range(24))
_OTHER_BOOT_DOMAIN = bytes(range(24, 48))
_OTHER_NAMESPACE_DOMAIN = bytes(range(48, 72))
_SEND_NS = 1_000_000_000
_METRIC = "sglang:detokenizer_router_delivery_seconds"


def _test_port_args(detokenizer_endpoint, *, local_endpoint=None):
    return SimpleNamespace(
        detokenizer_router_local_endpoint=local_endpoint,
        scheduler_input_ipc_name="ipc:///scheduler",
        rpc_ipc_name="ipc:///rpc",
        tokenizer_ipc_name="ipc:///tokenizer",
        detokenizer_ipc_name=detokenizer_endpoint,
        metrics_ipc_name="ipc:///metrics",
    )


@contextmanager
def _socket_pair():
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as sender, context.socket(zmq.PAIR) as receiver:
            for sock in (sender, receiver):
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.setsockopt(zmq.SNDTIMEO, 2000)
            sender.bind("inproc://clock-domain")
            receiver.connect("inproc://clock-domain")
            yield sender, receiver


class _RouterFinished(Exception):
    pass


@contextmanager
def _router_for_socket(receiver, domain):
    registry = CollectorRegistry()

    def registered_histogram(*args, **kwargs):
        return Histogram(*args, registry=registry, **kwargs)

    with (
        patch("prometheus_client.Histogram", side_effect=registered_histogram),
        patch("sglang.srt.managers.multi_tokenizer_mixin.zmq.Context"),
        patch(
            "sglang.srt.managers.multi_tokenizer_mixin.get_zmq_socket",
            return_value=receiver,
        ),
        patch(
            "sglang.srt.managers.multi_tokenizer_mixin.read_clock_domain",
            return_value=domain,
        ),
    ):
        router = MultiDetokenizerRouter(
            ["worker-a", "worker-b"],
            SimpleNamespace(detokenizer_ipc_name="unused"),
            enable_metrics=True,
        )
        yield router, registry


def _collect_routed_messages(router, count):
    routed = []

    def record(_endpoint, message):
        routed.append(message)
        if len(routed) == count:
            raise _RouterFinished

    router._send = record
    return routed


def _spawn_real_tcp_router(address, worker_endpoints, report, received):
    router = None
    failure = None
    result = None
    try:
        domain = delivery_clock.read_clock_domain()
        if domain is None:
            raise AssertionError("Spawned router could not identify its clock domain")
        registry = CollectorRegistry()

        def registered_histogram(*args, **kwargs):
            return Histogram(*args, registry=registry, **kwargs)

        # Keep the production collector; only its process-local registry differs.
        with patch("prometheus_client.Histogram", side_effect=registered_histogram):
            router = MultiDetokenizerRouter(
                worker_endpoints,
                SimpleNamespace(detokenizer_ipc_name="tcp://" + address + ":*"),
                enable_metrics=True,
            )
        if router.delivery_clock_domain != domain:
            raise AssertionError("Router did not read the child's actual clock domain")
        router.recv_from_scheduler.setsockopt(zmq.RCVTIMEO, 10000)
        endpoint = router.recv_from_scheduler.getsockopt_string(zmq.LAST_ENDPOINT)
        original_send = router._send

        def send_then_stop(target, message):
            original_send(target, message)
            if not received.wait(10):
                raise TimeoutError("Parent did not acknowledge the downstream payload")
            raise _RouterFinished

        router._send = send_then_stop
        report.send(
            {
                "stage": "ready",
                "domain": domain,
                "pid": os.getpid(),
                "endpoint": endpoint,
                "pickle_ipc": io_struct._USE_PICKLE_IPC,
            }
        )
        try:
            router.event_loop()
        except _RouterFinished:
            pass
        result = {
            "stage": "done",
            "domain": router.delivery_clock_domain,
            "count": registry.get_sample_value(_METRIC + "_count"),
            "sum": registry.get_sample_value(_METRIC + "_sum"),
        }
    except BaseException:
        failure = traceback.format_exc()
    finally:
        try:
            if router is not None:
                receive_context = router.recv_from_scheduler.context
                mapping_context = router.socket_mapping._zmq_context
                receive_context.destroy(linger=0)
                mapping_context.destroy(linger=0)
                if result is not None:
                    result["resources_closed"] = (
                        router.recv_from_scheduler.closed
                        and receive_context.closed
                        and mapping_context.closed
                    )
        except BaseException:
            failure = (failure or "") + traceback.format_exc()
    try:
        if failure is not None:
            report.send({"stage": "error", "traceback": failure})
            raise RuntimeError(failure)
        report.send(result)
    finally:
        report.close()


class TestClockDomainIdentity(CustomTestCase):
    def test_boot_and_time_namespace_changes_produce_distinct_identities(self):
        with (
            patch.object(delivery_clock.sys, "platform", "linux"),
            patch.object(
                delivery_clock.Path,
                "read_text",
                return_value="12345678-1234-1234-1234-123456789abc",
            ) as read_boot,
            patch.object(
                delivery_clock.os, "readlink", return_value="time:[4026531834]"
            ) as read_namespace,
        ):
            original = delivery_clock.read_clock_domain()
            self.assertIsInstance(original, bytes)
            self.assertEqual(len(original), 24)
            self.assertEqual(delivery_clock.read_clock_domain(), original)
            read_boot.return_value = "87654321-4321-4321-4321-cba987654321"
            other_boot = delivery_clock.read_clock_domain()
            self.assertNotEqual(other_boot, original)
            read_namespace.return_value = "time:[4026531835]"
            other_namespace = delivery_clock.read_clock_domain()
            self.assertNotEqual(other_namespace, other_boot)
            # The source is read in the current process; a parent-cached value
            # would miss a later child time-namespace change.
            self.assertEqual(read_boot.call_count, 4)
            self.assertEqual(read_namespace.call_count, 4)

    def test_unavailable_or_invalid_identity_is_unknown(self):
        cases = (
            ("boot-unavailable", OSError("no boot id"), "time:[1]"),
            ("boot-invalid", "invalid-uuid", "time:[1]"),
            (
                "namespace-unavailable",
                "12345678-1234-1234-1234-123456789abc",
                OSError(),
            ),
            ("namespace-invalid", "12345678-1234-1234-1234-123456789abc", "mnt:[1]"),
            ("namespace-negative", "12345678-1234-1234-1234-123456789abc", "time:[-1]"),
        )
        for name, boot, namespace in cases:
            with (
                self.subTest(name=name),
                patch.object(delivery_clock.sys, "platform", "linux"),
                patch.object(delivery_clock.Path, "read_text") as read_boot,
                patch.object(delivery_clock.os, "readlink") as read_namespace,
            ):
                if isinstance(boot, Exception):
                    read_boot.side_effect = boot
                else:
                    read_boot.return_value = boot
                if isinstance(namespace, Exception):
                    read_namespace.side_effect = namespace
                else:
                    read_namespace.return_value = namespace
                self.assertIsNone(delivery_clock.read_clock_domain())
        with (
            patch.object(delivery_clock.sys, "platform", "win32"),
            patch.object(
                delivery_clock.Path,
                "read_text",
                side_effect=AssertionError("proc read"),
            ),
        ):
            self.assertIsNone(delivery_clock.read_clock_domain())


class TestRouterClockDomain(CustomTestCase):
    def _create_channels(
        self, port_args, *, domain=_DOMAIN, enabled=True, is_leader=True
    ):
        with (
            patch.object(ipc_channels.zmq, "Context"),
            patch.object(ipc_channels, "get_zmq_socket", return_value=MagicMock()),
            patch.object(ipc_channels, "read_clock_domain", return_value=domain),
        ):
            return SchedulerIpcChannels.create(
                port_args=port_args,
                is_rank_zero=is_leader,
                skip_tokenizer_init=False,
                metrics_enabled=enabled,
                enable_scripted_runtime=False,
                detokenizer_router_metrics_enabled=enabled,
            )

    def test_native_local_non_loopback_endpoint_is_timestamped(self):
        endpoint = "tcp://192.0.2.1:23123"
        channels = self._create_channels(
            _test_port_args(endpoint, local_endpoint=endpoint)
        )
        self.assertTrue(channels.send_to_detokenizer.timestamped)
        self.assertEqual(channels.send_to_detokenizer.clock_domain, _DOMAIN)

    def test_non_loopback_endpoint_without_local_candidate_is_not_timestamped(self):
        endpoint = "tcp://192.0.2.1:23123"
        for missing in (False, True):
            with self.subTest(missing=missing):
                ports = _test_port_args(endpoint)
                if missing:
                    del ports.detokenizer_router_local_endpoint
                channels = self._create_channels(ports)
                self.assertFalse(channels.send_to_detokenizer.timestamped)

    def test_candidate_does_not_trust_a_changed_endpoint(self):
        ports = _test_port_args(
            "tcp://192.0.2.1:23124", local_endpoint="tcp://192.0.2.1:23123"
        )
        channels = self._create_channels(ports)
        self.assertFalse(channels.send_to_detokenizer.timestamped)

    def test_unknown_domain_metrics_off_and_nonleader_do_not_time(self):
        endpoint = "tcp://192.0.2.1:23123"
        cases = (
            {"domain": None},
            {"enabled": False},
            {"is_leader": False},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                channels = self._create_channels(
                    _test_port_args(endpoint, local_endpoint=endpoint), **kwargs
                )
                self.assertFalse(channels.send_to_detokenizer.timestamped)
                with patch.object(
                    io_struct.time,
                    "monotonic_ns",
                    side_effect=AssertionError("disabled sender read the clock"),
                ):
                    channels.send_to_detokenizer.send_output(AbortReq(rid="disabled"))


class TestClockDomainWire(CustomTestCase):
    def setUp(self):
        self.original_codec = io_struct._USE_PICKLE_IPC
        self.addCleanup(setattr, io_struct, "_USE_PICKLE_IPC", self.original_codec)

    def test_metadata_and_legacy_round_trip_both_codecs(self):
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                io_struct._USE_PICKLE_IPC = pickle_ipc
                message = AbortReq(rid="identity-frame")
                with patch.object(
                    io_struct.time, "monotonic_ns", return_value=_SEND_NS
                ):
                    io_struct.sock_send_with_timestamp(
                        sender, message, clock_domain=_DOMAIN
                    )
                frames = receiver.recv_multipart()
                self.assertEqual(len(frames), 2)
                self.assertEqual(len(frames[1]), 33)
                self.assertEqual(int.from_bytes(frames[1][:8], "big"), _SEND_NS)
                self.assertEqual(frames[1][8], 1)
                self.assertEqual(frames[1][9:], _DOMAIN)
                sender.send_multipart(frames)
                self.assertEqual(
                    io_struct.sock_recv_with_timestamp(
                        receiver,
                        expected_clock_domain=_DOMAIN,
                        require_clock_domain=True,
                    ),
                    (_SEND_NS, message),
                )

                with patch.object(
                    io_struct.time, "monotonic_ns", return_value=_SEND_NS
                ):
                    io_struct.sock_send_with_timestamp(sender, message)
                legacy_frames = receiver.recv_multipart()
                self.assertEqual(len(legacy_frames[1]), 8)
                sender.send_multipart(legacy_frames)
                self.assertEqual(
                    io_struct.sock_recv_with_timestamp(
                        receiver,
                        expected_clock_domain=_DOMAIN,
                        require_clock_domain=True,
                    ),
                    (None, message),
                )
                self.assertFalse(receiver.getsockopt(zmq.RCVMORE))

    def test_malformed_metadata_lengths_are_rejected_both_codecs(self):
        for pickle_ipc in (False, True):
            for size in (0, 7, 9, 32, 34):
                with (
                    self.subTest(pickle_ipc=pickle_ipc, size=size),
                    _socket_pair() as (sender, receiver),
                ):
                    io_struct._USE_PICKLE_IPC = pickle_ipc
                    message = AbortReq(rid="malformed-metadata")
                    io_struct.sock_send_with_timestamp(
                        sender, message, clock_domain=_DOMAIN
                    )
                    frames = receiver.recv_multipart()
                    frames[1] = bytes(size)
                    sender.send_multipart(frames)
                    with self.assertRaisesRegex(ValueError, "timestamp frame length"):
                        io_struct.sock_recv_with_timestamp(
                            receiver,
                            expected_clock_domain=_DOMAIN,
                            require_clock_domain=True,
                        )

    def test_future_metadata_version_routes_without_sampling(self):
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                io_struct._USE_PICKLE_IPC = pickle_ipc
                message = AbortReq(rid="future-version")
                io_struct.sock_send_with_timestamp(
                    sender, message, clock_domain=_DOMAIN
                )
                frames = receiver.recv_multipart()
                frames[1] = frames[1][:8] + bytes([2]) + frames[1][9:]
                sender.send_multipart(frames)
                self.assertEqual(
                    io_struct.sock_recv_with_timestamp(
                        receiver,
                        expected_clock_domain=_DOMAIN,
                        require_clock_domain=True,
                    ),
                    (None, message),
                )
                self.assertFalse(receiver.getsockopt(zmq.RCVMORE))


class TestRouterClockDomainIntegration(CustomTestCase):
    def setUp(self):
        self.original_codec = io_struct._USE_PICKLE_IPC
        self.addCleanup(setattr, io_struct, "_USE_PICKLE_IPC", self.original_codec)

    def test_mixed_domains_and_legacy_messages_on_one_zmq_channel(self):
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                io_struct._USE_PICKLE_IPC = pickle_ipc
                # Names describe the simulated identity source, not real remote hosts.
                messages = [
                    ("match-first", _DOMAIN),
                    ("other-boot", _OTHER_BOOT_DOMAIN),
                    ("other-time-namespace", _OTHER_NAMESPACE_DOMAIN),
                    ("unknown-identity", bytes(24)),
                    ("legacy-timestamp", None),
                    ("legacy-single-frame", None),
                    ("future-version", _DOMAIN),
                    ("match-last", _DOMAIN),
                ]
                for rid, domain in messages:
                    message = AbortReq(rid=rid)
                    if rid == "legacy-single-frame":
                        io_struct.sock_send(sender, message)
                    elif rid == "future-version":
                        payload = (
                            pickle.dumps(message)
                            if pickle_ipc
                            else io_struct.msgpack_encode(message)
                        )
                        sender.send_multipart(
                            [payload, _SEND_NS.to_bytes(8, "big") + bytes([2]) + domain]
                        )
                    else:
                        with patch.object(
                            io_struct.time, "monotonic_ns", return_value=_SEND_NS
                        ):
                            io_struct.sock_send_with_timestamp(
                                sender, message, clock_domain=domain
                            )
                with _router_for_socket(receiver, _DOMAIN) as (router, registry):
                    routed = _collect_routed_messages(router, len(messages))
                    with (
                        patch(
                            "sglang.srt.managers.multi_tokenizer_mixin.monotonic_ns",
                            return_value=_SEND_NS + 1_000_000_000,
                        ) as receive_clock,
                        self.assertRaises(_RouterFinished),
                    ):
                        router.event_loop()
                    self.assertEqual(receive_clock.call_count, 2)
                    self.assertEqual(
                        [obj.rid for obj in routed], [x[0] for x in messages]
                    )
                    self.assertEqual(registry.get_sample_value(_METRIC + "_count"), 2.0)
                    self.assertEqual(registry.get_sample_value(_METRIC + "_sum"), 2.0)

    def test_mismatched_or_unknown_router_domain_never_reads_receive_clock(self):
        for router_domain in (_DOMAIN, None):
            with (
                self.subTest(router_domain=router_domain),
                _socket_pair() as (sender, receiver),
            ):
                io_struct.sock_send_with_timestamp(
                    sender, AbortReq(rid="untrusted"), clock_domain=_OTHER_BOOT_DOMAIN
                )
                with _router_for_socket(receiver, router_domain) as (router, registry):
                    routed = _collect_routed_messages(router, 1)
                    with (
                        patch(
                            "sglang.srt.managers.multi_tokenizer_mixin.monotonic_ns",
                            side_effect=AssertionError(
                                "untrusted receive read the clock"
                            ),
                        ),
                        self.assertRaises(_RouterFinished),
                    ):
                        router.event_loop()
                    self.assertEqual(routed[0].rid, "untrusted")
                    self.assertEqual(registry.get_sample_value(_METRIC + "_count"), 0.0)

    def test_real_non_loopback_tcp_channels_to_production_router(self):
        addresses = [
            item.address
            for entries in psutil.net_if_addrs().values()
            for item in entries
            if item.family == socket.AF_INET
            and not ipaddress.ip_address(item.address).is_loopback
            and not ipaddress.ip_address(item.address).is_unspecified
        ]
        if not addresses:
            self.skipTest("No local non-loopback IPv4 interface is available")
        # This is a local transport test; it does not claim a multi-host runtime.
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.LINGER, 0)
        receiver.setsockopt(zmq.RCVTIMEO, 3000)
        port = receiver.bind_to_random_port("tcp://" + addresses[0])
        endpoint = "tcp://" + addresses[0] + ":" + str(port)
        channels = None
        try:
            with tempfile.TemporaryDirectory(prefix="sglang-clock-tcp-") as tmp:
                ports = _test_port_args(endpoint, local_endpoint=endpoint)
                for field in (
                    "scheduler_input_ipc_name",
                    "rpc_ipc_name",
                    "tokenizer_ipc_name",
                    "metrics_ipc_name",
                ):
                    setattr(ports, field, "ipc://" + tmp + "/" + field)
                with patch.object(
                    ipc_channels, "read_clock_domain", return_value=_DOMAIN
                ):
                    channels = SchedulerIpcChannels.create(
                        port_args=ports,
                        is_rank_zero=True,
                        skip_tokenizer_init=False,
                        metrics_enabled=True,
                        enable_scripted_runtime=False,
                        detokenizer_router_metrics_enabled=True,
                    )
                self.assertTrue(channels.send_to_detokenizer.timestamped)
                channels.send_to_detokenizer.socket.setsockopt(zmq.SNDTIMEO, 3000)
                channels.send_to_detokenizer.send_output(AbortReq(rid="real-tcp"))
                with _router_for_socket(receiver, _DOMAIN) as (router, registry):
                    routed = _collect_routed_messages(router, 1)
                    with self.assertRaises(_RouterFinished):
                        router.event_loop()
                    self.assertEqual(routed[0].rid, "real-tcp")
                    self.assertEqual(registry.get_sample_value(_METRIC + "_count"), 1.0)
                    self.assertGreaterEqual(
                        registry.get_sample_value(_METRIC + "_sum"), 0.0
                    )
        finally:
            if channels is not None:
                channels.send_to_detokenizer.socket.context.destroy(linger=0)
            receiver.close(0)
            context.term()


class _ReachedSchedulerLaunch(Exception):
    pass


class _UnknownEngineBackend(Engine):
    pass


class TestEngineClockDomainCandidate(CustomTestCase):
    def _candidate_at_scheduler_launch(self, **overrides):
        settings = {
            "node_rank": 0,
            "metrics": True,
            "workers": 2,
            "skip": False,
            "joiner": False,
            "rust": False,
            "engine_cls": Engine,
        }
        settings.update(overrides)
        endpoint = "tcp://192.0.2.1:23123"
        ports = _test_port_args(endpoint, local_endpoint="stale-candidate")
        seen = []

        def stop_at_scheduler_launch(_server_args, child_ports, _scheduler):
            self.assertIs(child_ports, ports)
            seen.append(child_ports.detokenizer_router_local_endpoint)
            raise _ReachedSchedulerLaunch

        # Keep the production launch body and candidate decision intact while
        # stopping before any scheduler/model/process can start.
        with ExitStack() as stack:
            for name in (
                "configure_logger",
                "_set_envs_and_config",
                "load_plugins",
                "resolve_auto_parsers",
                "snapshot_context",
                "publish",
            ):
                stack.enter_context(patch.object(engine_module, name))
            stack.enter_context(
                patch.object(
                    engine_module, "_server_args_for_logging", return_value="cpu-test"
                )
            )
            for name, value in (
                ("get_parallel", SimpleNamespace(node_rank=settings["node_rank"])),
                (
                    "get_exec",
                    SimpleNamespace(
                        moe=SimpleNamespace(is_ep_joiner=settings["joiner"])
                    ),
                ),
                (
                    "get_observability",
                    SimpleNamespace(enable_metrics=settings["metrics"]),
                ),
                (
                    "get_serving",
                    SimpleNamespace(
                        detokenizer_worker_num=settings["workers"],
                        skip_tokenizer_init=settings["skip"],
                    ),
                ),
                (
                    "get_model",
                    SimpleNamespace(
                        remote_instance_weight_loader_start_seed_via_transfer_engine=(
                            False
                        ),
                        weight_cache_mode="disabled",
                    ),
                ),
            ):
                stack.enter_context(
                    patch.object(engine_module, name, return_value=value)
                )
            stack.enter_context(
                patch.object(
                    engine_module.envs.SGLANG_RUST_SERVER,
                    "get",
                    return_value=settings["rust"],
                )
            )
            launch = stack.enter_context(
                patch.object(
                    settings["engine_cls"],
                    "_launch_scheduler_processes",
                    side_effect=stop_at_scheduler_launch,
                )
            )
            with self.assertRaises(_ReachedSchedulerLaunch):
                settings["engine_cls"]._launch_subprocesses(
                    server_args=MagicMock(),
                    init_tokenizer_manager_func=MagicMock(),
                    run_scheduler_process_func=MagicMock(),
                    run_detokenizer_process_func=MagicMock(),
                    port_args=ports,
                )
            launch.assert_called_once()
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_native_candidate_is_set_before_scheduler_launch(self):
        self.assertEqual(self._candidate_at_scheduler_launch(), "tcp://192.0.2.1:23123")

    def test_excluded_launches_clear_a_preexisting_candidate(self):
        cases = (
            ("follower", {"node_rank": 1}),
            ("metrics-off", {"metrics": False}),
            ("skip-tokenizer", {"skip": True}),
            ("single-worker", {"workers": 1}),
            ("joiner", {"joiner": True}),
            ("unknown-subclass", {"engine_cls": _UnknownEngineBackend}),
            ("rust", {"rust": True}),
        )
        for name, overrides in cases:
            with self.subTest(name=name):
                self.assertIsNone(self._candidate_at_scheduler_launch(**overrides))


class TestDpClockDomainCandidate(CustomTestCase):
    def test_rebuilt_ports_propagate_only_the_unchanged_router_target(self):
        endpoint = "tcp://192.0.2.1:23123"
        cases = (
            ("same-target", endpoint, False, endpoint, endpoint),
            (
                "different-target",
                "tcp://192.0.2.1:23124",
                False,
                "tcp://192.0.2.1:23124",
                None,
            ),
            (
                "joiner-rewrites-target",
                endpoint,
                True,
                "tcp://192.0.2.2:26002",
                None,
            ),
        )
        for (
            name,
            rebuilt_endpoint,
            joiner,
            expected_endpoint,
            expected_candidate,
        ) in cases:
            with self.subTest(name=name):
                parent_ports = _test_port_args(endpoint, local_endpoint=endpoint)
                parent_ports.nccl_port = 24000
                parent_ports.instance_id = "clock-domain-test"
                rebuilt_ports = _test_port_args(rebuilt_endpoint)
                server_args = MagicMock()
                controller = dp_module.DataParallelController.__new__(
                    dp_module.DataParallelController
                )
                controller.env_lock = threading.Lock()
                controller._scheduler_init_info_lock = threading.Lock()
                controller._scheduler_init_infos = []
                controller.local_kv_event_sources = []
                controller.attn_replica_size = 1
                pipe = MagicMock()
                pipe.recv.return_value = {
                    "max_total_num_tokens": 64,
                    "max_req_input_len": 32,
                }
                controller.prelaunched_schedulers = {(0, 0): (12345, pipe)}
                parallel = SimpleNamespace(
                    attn_dp_enabled=True,
                    pp_size=1,
                    nnodes=1,
                    node_rank=0,
                    tp_size=1,
                    attn_dp_size=1,
                    attn_cp_size=1,
                    moe_dp_size=1,
                    ep_size=1,
                    ep_join_rank_offset=0,
                    dist_init_addr="192.0.2.2:26000",
                )
                execution = SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False),
                    moe=SimpleNamespace(is_ep_scale_joiner=joiner),
                )
                with (
                    patch.object(dp_module, "get_parallel", return_value=parallel),
                    patch.object(dp_module, "get_exec", return_value=execution),
                    patch.object(
                        dp_module,
                        "get_device",
                        return_value=SimpleNamespace(base_gpu_id=0, gpu_id_step=1),
                    ),
                    patch.object(
                        dp_module.PortArgs, "init_new", return_value=rebuilt_ports
                    ) as init_ports,
                    patch.object(dp_module.TorchMemorySaverAdapter, "create"),
                    patch.object(
                        dp_module,
                        "maybe_reindex_device_id",
                        return_value=nullcontext(0),
                    ),
                    patch.object(dp_module.mp, "Process") as process,
                ):
                    controller.launch_tensor_parallel_group(
                        server_args, parent_ports, 0, None, [25006]
                    )
                    process.assert_not_called()
                    init_ports.assert_called_once_with(server_args, 0, [25006])
                pipe.send.assert_called_once()
                scheduler_args, display_ranks = pipe.send.call_args.args[0]
                child_ports = scheduler_args[1]
                self.assertIs(child_ports, rebuilt_ports)
                self.assertEqual(child_ports.detokenizer_ipc_name, expected_endpoint)
                self.assertEqual(
                    child_ports.detokenizer_router_local_endpoint, expected_candidate
                )
                self.assertEqual(child_ports.nccl_port, parent_ports.nccl_port)
                self.assertEqual(child_ports.instance_id, parent_ports.instance_id)

                # Exercise the actual prelaunch receiver rather than inspecting
                # a copied reconstruction of the scheduler argument protocol.
                child_pipe = MagicMock()
                child_pipe.recv.return_value = (scheduler_args, display_ranks)
                scheduler_entry = MagicMock()
                with patch.object(dp_module, "kill_itself_when_parent_died"):
                    dp_module.run_scheduler_process_with_init_pipe(
                        scheduler_entry, child_pipe
                    )
                scheduler_entry.assert_called_once_with(
                    *scheduler_args, child_pipe, *display_ranks
                )
                self.assertIs(scheduler_entry.call_args.args[1], child_ports)
                child_pipe.close.assert_called_once()

    def test_pure_dp_rebuild_passes_router_endpoint_and_candidate_to_thread(self):
        endpoint = "tcp://192.0.2.1:23123"
        for candidate in (endpoint, None):
            with self.subTest(candidate=candidate):
                parent_ports = _test_port_args(endpoint, local_endpoint=candidate)
                parent_ports.instance_id = "pure-dp-clock-domain-test"
                rebuilt_ports = _test_port_args(
                    "tcp://192.0.2.1:23124",
                    local_endpoint="stale-rebuilt-candidate",
                )
                rebuilt_ports.nccl_port = 24000
                server_args = MagicMock()
                controller = dp_module.DataParallelController.__new__(
                    dp_module.DataParallelController
                )
                controller.context = MagicMock()
                controller.workers = [None]
                parallel = SimpleNamespace(
                    num_dp_ranks=1, tp_size=1, pp_size=1, node_rank=0
                )
                with (
                    patch.object(dp_module, "get_parallel", return_value=parallel),
                    patch.object(
                        dp_module,
                        "get_device",
                        return_value=SimpleNamespace(gpu_id_step=1),
                    ),
                    patch.object(
                        dp_module.PortArgs, "init_new", return_value=rebuilt_ports
                    ) as init_ports,
                    patch.object(dp_module, "bind_port") as bind_port,
                    patch.object(dp_module, "get_zmq_socket") as get_socket,
                    patch.object(dp_module.threading, "Event") as event,
                    patch.object(dp_module.threading, "Thread") as thread,
                ):
                    controller.launch_dp_schedulers(server_args, parent_ports)
                    init_ports.assert_called_once_with(server_args)
                    bind_port.assert_called_once_with(rebuilt_ports.nccl_port)
                    bind_port.return_value.close.assert_called_once()
                    thread.assert_called_once()
                    self.assertEqual(
                        thread.call_args.kwargs["target"],
                        controller.launch_tensor_parallel_group_thread,
                    )
                    child_args = thread.call_args.kwargs["args"]
                    self.assertEqual(
                        child_args,
                        (server_args, rebuilt_ports, 0, 0, event.return_value),
                    )
                    child_ports = child_args[1]
                    self.assertEqual(child_ports.detokenizer_ipc_name, endpoint)
                    self.assertEqual(
                        child_ports.detokenizer_router_local_endpoint, candidate
                    )
                    self.assertEqual(child_ports.instance_id, parent_ports.instance_id)
                    get_socket.assert_called_once_with(
                        controller.context,
                        zmq.PUSH,
                        rebuilt_ports.scheduler_input_ipc_name,
                        True,
                    )
                    thread.return_value.start.assert_called_once()
                    event.return_value.wait.assert_called_once()


class TestSpawnedRouterClockDomain(CustomTestCase):
    def test_spawned_router_uses_real_domain_and_non_loopback_tcp(self):
        if sys.platform != "linux":
            self.skipTest("Real boot/time-namespace identities require Linux")
        addresses = [
            item.address
            for entries in psutil.net_if_addrs().values()
            for item in entries
            if item.family == socket.AF_INET
            and not ipaddress.ip_address(item.address).is_loopback
            and not ipaddress.ip_address(item.address).is_unspecified
        ]
        if not addresses:
            self.skipTest("No local non-loopback IPv4 interface is available")
        parent_domain = delivery_clock.read_clock_domain()
        self.assertIsInstance(parent_domain, bytes)
        self.assertEqual(len(parent_domain), 24)
        process_context = mp.get_context("spawn")
        report, child_report = process_context.Pipe(duplex=False)
        received = process_context.Event()
        channels = None
        failure = None
        child_errors = []
        terminated = False

        def read_report(process, timeout):
            if not report.poll(timeout):
                raise TimeoutError(
                    f"Router report timed out: pid={process.pid}, "
                    f"exitcode={process.exitcode}"
                )
            result = report.recv()
            if result["stage"] == "error":
                raise AssertionError(result["traceback"])
            return result

        with (
            tempfile.TemporaryDirectory(prefix="sglang-spawn-clock-") as tmp,
            zmq.Context() as worker_context,
        ):
            workers = [
                worker_context.socket(zmq.PULL),
                worker_context.socket(zmq.PULL),
            ]
            worker_endpoints = [
                "ipc://" + tmp + "/worker-a",
                "ipc://" + tmp + "/worker-b",
            ]
            for worker, endpoint in zip(workers, worker_endpoints):
                worker.setsockopt(zmq.LINGER, 0)
                worker.bind(endpoint)
            process = process_context.Process(
                target=_spawn_real_tcp_router,
                args=(addresses[0], worker_endpoints, child_report, received),
            )
            try:
                process.start()
                child_report.close()
                ready = read_report(process, 45)
                self.assertEqual(ready["stage"], "ready")
                self.assertEqual(ready["pid"], process.pid)
                self.assertNotEqual(ready["pid"], os.getpid())
                self.assertEqual(ready["domain"], parent_domain)
                self.assertEqual(ready["pickle_ipc"], io_struct._USE_PICKLE_IPC)
                endpoint = ready["endpoint"]
                self.assertTrue(endpoint.startswith("tcp://" + addresses[0] + ":"))
                ports = _test_port_args(endpoint, local_endpoint=endpoint)
                for field in (
                    "scheduler_input_ipc_name",
                    "rpc_ipc_name",
                    "tokenizer_ipc_name",
                    "metrics_ipc_name",
                ):
                    setattr(ports, field, "ipc://" + tmp + "/" + field)
                channels = SchedulerIpcChannels.create(
                    port_args=ports,
                    is_rank_zero=True,
                    skip_tokenizer_init=False,
                    metrics_enabled=True,
                    enable_scripted_runtime=False,
                    detokenizer_router_metrics_enabled=True,
                )
                self.assertTrue(channels.send_to_detokenizer.timestamped)
                self.assertEqual(
                    channels.send_to_detokenizer.clock_domain, parent_domain
                )
                channels.send_to_detokenizer.socket.setsockopt(zmq.SNDTIMEO, 5000)
                message = AbortReq(rid="spawned-real-tcp")
                channels.send_to_detokenizer.send_output(message)
                poller = zmq.Poller()
                for worker in workers:
                    poller.register(worker, zmq.POLLIN)
                readable = dict(poller.poll(15000))
                self.assertEqual(len(readable), 1, "Missing downstream payload")
                forwarded = io_struct.sock_recv(next(iter(readable)))
                self.assertEqual(forwarded, message)
                received.set()
                done = read_report(process, 15)
                self.assertEqual(done["stage"], "done")
                self.assertEqual(done["domain"], parent_domain)
                self.assertEqual(done["count"], 1.0)
                self.assertGreaterEqual(done["sum"], 0.0)
                self.assertLess(done["sum"], 30.0)
                self.assertTrue(done["resources_closed"])
            except Exception as exc:
                failure = exc
            finally:
                received.set()
                if channels is not None:
                    sender_context = channels.send_to_detokenizer.socket.context
                    sender_context.destroy(linger=0)
                for worker in workers:
                    worker.close(0)
                if process.pid is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        terminated = True
                        process.terminate()
                        process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                while report.poll(0):
                    try:
                        pending = report.recv()
                    except EOFError:
                        break
                    if pending["stage"] == "error":
                        child_errors.append(pending["traceback"])
                child_report.close()
                report.close()
                child_pid = process.pid
                child_exitcode = process.exitcode
                child_alive = process.is_alive()
                if child_pid is not None and not child_alive:
                    process.close()
            diagnostic = (
                f"router pid={child_pid}, exitcode={child_exitcode}, "
                f"terminated_after_timeout={terminated}"
            )
            if child_errors:
                diagnostic += "\n" + "\n".join(child_errors)
            if failure is not None:
                raise AssertionError(f"{failure}\n{diagnostic}") from failure
            self.assertFalse(terminated, diagnostic)
            self.assertFalse(child_alive, diagnostic)
            self.assertEqual(child_exitcode, 0, diagnostic)
            self.assertFalse(child_errors, diagnostic)
            self.assertTrue(all(worker.closed for worker in workers))
            self.assertTrue(sender_context.closed)
            self.assertTrue(report.closed)
            self.assertTrue(child_report.closed)
        self.assertTrue(worker_context.closed)


if __name__ == "__main__":
    unittest.main()
