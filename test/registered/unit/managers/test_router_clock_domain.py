import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import delivery_clock, io_struct
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_components import ipc_channels
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.test.router_delivery_test_utils import RouterFinished as _RouterFinished
from sglang.test.router_delivery_test_utils import (
    collect_routed_messages as _collect_routed_messages,
)
from sglang.test.router_delivery_test_utils import (
    create_test_channels,
)
from sglang.test.router_delivery_test_utils import (
    router_for_socket as _router_for_socket,
)
from sglang.test.router_delivery_test_utils import socket_pair as _socket_pair
from sglang.test.router_delivery_test_utils import test_port_args as _test_port_args

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DOMAIN = bytes(range(24))
_TEST_CLOCK_DOMAIN = _DOMAIN
_OTHER_BOOT_DOMAIN = bytes(range(24, 48))
_OTHER_NAMESPACE_DOMAIN = bytes(range(48, 72))
_SEND_NS = 1_000_000_000
_METRIC = "sglang:detokenizer_router_delivery_seconds"


class TestTimestampFraming(CustomTestCase):
    def test_round_trip_and_version_compatibility(self):
        message = AbortReq(rid="framing")
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch.object(
                    delivery_clock.time, "monotonic_ns", return_value=_SEND_NS
                ),
                _socket_pair() as (sender, receiver),
            ):
                io_struct.sock_send(sender, message)
                self.assertEqual(
                    delivery_clock.sock_recv_with_timestamp(receiver), (None, message)
                )

                delivery_clock.sock_send_with_timestamp(sender, message)
                legacy_frames = receiver.recv_multipart()
                self.assertEqual(len(legacy_frames), 2)
                self.assertEqual(legacy_frames[1], _SEND_NS.to_bytes(8, "big"))

                delivery_clock.sock_send_with_timestamp(
                    sender, message, clock_domain=_DOMAIN
                )
                current_frames = receiver.recv_multipart()
                self.assertEqual(len(current_frames), 2)
                self.assertEqual(
                    current_frames[1],
                    _SEND_NS.to_bytes(8, "big") + bytes([1]) + _DOMAIN,
                )
                future_frames = [
                    current_frames[0],
                    current_frames[1][:8] + bytes([2]) + _DOMAIN,
                ]

                cases = (
                    (legacy_frames, None, False, _SEND_NS),
                    (legacy_frames, _DOMAIN, True, None),
                    (current_frames, _DOMAIN, True, _SEND_NS),
                    (current_frames, _OTHER_BOOT_DOMAIN, True, None),
                    (future_frames, _DOMAIN, True, None),
                )
                for (
                    frames,
                    expected_domain,
                    require_domain,
                    expected_timestamp,
                ) in cases:
                    with self.subTest(
                        expected_domain=expected_domain,
                        require_domain=require_domain,
                        metadata=frames[1],
                    ):
                        sender.send_multipart(frames)
                        self.assertEqual(
                            delivery_clock.sock_recv_with_timestamp(
                                receiver,
                                expected_clock_domain=expected_domain,
                                require_clock_domain=require_domain,
                            ),
                            (expected_timestamp, message),
                        )
                        self.assertFalse(receiver.getsockopt(zmq.RCVMORE))

    def test_send_timestamp_precedes_serialization(self):
        message = AbortReq(rid="encode-boundary")
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                encoder_owner = io_struct.pickle if pickle_ipc else io_struct
                encoder_name = "dumps" if pickle_ipc else "msgpack_encode"
                original_encoder = getattr(encoder_owner, encoder_name)
                clock_ns = [1_000_000_000]

                def advance_during_encoding(*args, **kwargs):
                    clock_ns[0] += 2_000_000_000
                    return original_encoder(*args, **kwargs)

                with (
                    patch.object(
                        delivery_clock.time,
                        "monotonic_ns",
                        side_effect=lambda: clock_ns[0],
                    ),
                    patch.object(
                        encoder_owner, encoder_name, side_effect=advance_during_encoding
                    ),
                ):
                    delivery_clock.sock_send_with_timestamp(sender, message)
                timestamp, decoded = delivery_clock.sock_recv_with_timestamp(receiver)
                self.assertEqual(timestamp, 1_000_000_000)
                self.assertEqual(clock_ns[0], 3_000_000_000)
                self.assertEqual(decoded, message)

    def test_invalid_metadata_and_extra_frames(self):
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                message = AbortReq(rid="malformed-metadata")
                payload = (
                    pickle.dumps(message)
                    if pickle_ipc
                    else io_struct.msgpack_encode(message)
                )
                for size in (0, 7, 9, 32, 34):
                    with self.subTest(size=size):
                        sender.send_multipart([payload, bytes(size)])
                        with self.assertRaisesRegex(
                            ValueError, "timestamp frame length"
                        ):
                            delivery_clock.sock_recv_with_timestamp(
                                receiver,
                                expected_clock_domain=_DOMAIN,
                                require_clock_domain=True,
                            )
                sender.send_multipart([payload, bytes(8), b"extra"])
                with self.assertRaisesRegex(ValueError, "Expected at most two frames"):
                    delivery_clock.sock_recv_with_timestamp(receiver)
                socket_mock = MagicMock()
                with self.assertRaisesRegex(ValueError, "identity length"):
                    delivery_clock.sock_send_with_timestamp(
                        socket_mock, message, clock_domain=bytes(23)
                    )
                self.assertEqual(socket_mock.mock_calls, [])

    def test_encoding_failure_sends_no_partial_message(self):
        for pickle_ipc in (False, True):
            encoder = "pickle.dumps" if pickle_ipc else "msgpack_encode"
            socket_mock = MagicMock()
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch(
                    f"sglang.srt.managers.io_struct.{encoder}",
                    side_effect=ValueError("encoding failed"),
                ),
                self.assertRaisesRegex(ValueError, "encoding failed"),
            ):
                delivery_clock.sock_send_with_timestamp(
                    socket_mock, AbortReq(rid="encoding")
                )
            self.assertEqual(socket_mock.mock_calls, [])


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


class TestClockDomainGating(CustomTestCase):
    def test_timestamp_only_for_local_endpoints(self):
        cases = (
            ("ipc:///tmp/detokenizer", True),
            ("inproc://detokenizer", True),
            ("tcp://127.0.0.1:23123", True),
            ("tcp://127.0.0.2:23123", True),
            ("tcp://[::1]:23123", True),
            ("tcp://192.0.2.1:23123", False),
            ("tcp://remote.example:23123", False),
            ("tcp://localhost:23123", False),
            ("tcp://127.0.0.1:0;192.0.2.1:23123", False),
            ("tcp://192.0.2.1:0;127.0.0.1:23123", False),
        )
        for endpoint, expected in cases:
            with (
                self.subTest(endpoint=endpoint),
                patch(
                    "sglang.srt.managers.scheduler_components.ipc_channels.read_clock_domain",
                    return_value=_TEST_CLOCK_DOMAIN,
                ),
                patch(
                    "sglang.srt.managers.scheduler_components.ipc_channels.zmq.Context"
                ),
                patch(
                    "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket",
                    return_value=MagicMock(),
                ),
            ):
                channels = SchedulerIpcChannels.create(
                    port_args=_test_port_args(endpoint),
                    is_rank_zero=True,
                    skip_tokenizer_init=False,
                    metrics_enabled=True,
                    enable_scripted_runtime=False,
                    detokenizer_router_metrics_enabled=True,
                )
                self.assertEqual(
                    channels.send_to_detokenizer.clock_domain is not None, expected
                )

    def test_remote_sender_keeps_single_frame_without_clock_read(self):
        original_codec = io_struct._USE_PICKLE_IPC
        self.addCleanup(setattr, io_struct, "_USE_PICKLE_IPC", original_codec)
        endpoint = "tcp://192.0.2.1:23123"
        for pickle_ipc in (False, True):
            with (
                self.subTest(pickle_ipc=pickle_ipc),
                _socket_pair() as (sender, receiver),
            ):
                io_struct._USE_PICKLE_IPC = pickle_ipc

                def socket_for_endpoint(_context, _kind, name, _bind):
                    return sender if name == endpoint else MagicMock()

                with (
                    patch(
                        "sglang.srt.managers.scheduler_components.ipc_channels.zmq.Context"
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket",
                        side_effect=socket_for_endpoint,
                    ),
                ):
                    channels = SchedulerIpcChannels.create(
                        port_args=_test_port_args(endpoint),
                        is_rank_zero=True,
                        skip_tokenizer_init=False,
                        metrics_enabled=True,
                        enable_scripted_runtime=False,
                        detokenizer_router_metrics_enabled=True,
                    )
                message = AbortReq(rid="remote-legacy")
                with patch.object(
                    delivery_clock.time,
                    "monotonic_ns",
                    side_effect=AssertionError("clock read"),
                ):
                    channels.send_to_detokenizer.send_output(message)
                self.assertEqual(
                    delivery_clock.sock_recv_with_timestamp(receiver), (None, message)
                )
                self.assertFalse(receiver.getsockopt(zmq.RCVMORE))


class TestSchedulerClockDomain(CustomTestCase):
    def test_unknown_domain_metrics_off_and_nonleader_do_not_time(self):
        endpoint = "ipc:///detokenizer"
        cases = (
            {"domain": None},
            {"enabled": False},
            {"is_leader": False},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                channels = create_test_channels(_test_port_args(endpoint), **kwargs)
                self.assertIsNone(channels.send_to_detokenizer.clock_domain)
                with patch.object(
                    delivery_clock.time,
                    "monotonic_ns",
                    side_effect=AssertionError("disabled sender read the clock"),
                ):
                    channels.send_to_detokenizer.send_output(AbortReq(rid="disabled"))

    def test_scheduler_metrics_modes_use_the_expected_framing(self):
        cases = (
            ("enabled", True, 2, False, False, True),
            ("rust", True, 2, False, True, False),
            ("metrics-off", False, 2, False, False, False),
            ("single-detokenizer", True, 1, False, False, False),
            ("skip-tokenizer", True, 2, True, False, False),
        )
        for name, metrics, workers, skip, rust, timestamped in cases:
            for pickle_ipc in (False, True):
                scheduler = scheduler_module.Scheduler.__new__(
                    scheduler_module.Scheduler
                )
                scheduler.skip_tokenizer_init = skip
                ports = _test_port_args("ipc:///detokenizer")
                observability = SimpleNamespace(
                    enable_metrics=metrics,
                    enable_metrics_for_all_schedulers=False,
                    load_snapshot_publish_interval=1.0,
                )
                parallel = SimpleNamespace(attn_tp_rank=0, dp_rank=0, num_dp_ranks=1)
                serving = SimpleNamespace(detokenizer_worker_num=workers)
                socket_mock = MagicMock()
                with (
                    self.subTest(mode=name, pickle_ipc=pickle_ipc),
                    patch.object(
                        scheduler_module, "is_scheduler_io_rank", return_value=True
                    ),
                    patch.object(
                        scheduler_module,
                        "get_observability",
                        return_value=observability,
                    ),
                    patch.object(
                        scheduler_module, "get_parallel", return_value=parallel
                    ),
                    patch.object(scheduler_module, "get_serving", return_value=serving),
                    patch.object(scheduler_module, "create_load_snapshot_writer"),
                    patch.object(
                        scheduler_module.envs.SGLANG_RUST_SERVER,
                        "get",
                        return_value=rust,
                    ),
                    patch.object(ipc_channels.zmq, "Context"),
                    patch.object(
                        ipc_channels, "get_zmq_socket", return_value=socket_mock
                    ) as get_socket,
                    patch.object(
                        ipc_channels, "read_clock_domain", return_value=_DOMAIN
                    ) as read_domain,
                    patch.object(
                        delivery_clock.time, "monotonic_ns", return_value=_SEND_NS
                    ) as read_clock,
                    patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                ):
                    scheduler.init_ipc_channels(ports)
                    sender = scheduler.ipc_channels.send_to_detokenizer
                    sender.send_output(AbortReq(rid="scheduler-mode-framing"))
                    if timestamped:
                        read_domain.assert_called_once()
                        read_clock.assert_called_once()
                        socket_mock.send_multipart.assert_called_once()
                        self.assertEqual(
                            len(socket_mock.send_multipart.call_args.args[0]), 2
                        )
                    else:
                        read_domain.assert_not_called()
                        read_clock.assert_not_called()
                        socket_mock.send_multipart.assert_not_called()
                        (
                            socket_mock.send_pyobj if pickle_ipc else socket_mock.send
                        ).assert_called_once()
                    endpoints = [call.args[2] for call in get_socket.call_args_list]
                    self.assertEqual(
                        endpoints.count(ports.tokenizer_ipc_name), 2 if skip else 1
                    )
                    self.assertEqual(
                        endpoints.count(ports.detokenizer_ipc_name), 0 if skip else 1
                    )


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
                            delivery_clock.time, "monotonic_ns", return_value=_SEND_NS
                        ):
                            delivery_clock.sock_send_with_timestamp(
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
                delivery_clock.sock_send_with_timestamp(
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


if __name__ == "__main__":
    unittest.main()
