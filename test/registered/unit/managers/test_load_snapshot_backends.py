"""Unit tests for LoadSnapshot SHM and ZMQ backends."""

import os
import socket
import tempfile
import time
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.kv_events import load_pub_endpoint
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    PubLoadSnapshotWriter,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    ZmqLoadSnapshotWriter,
    ZmqShmLoadSnapshotReader,
    _zmq_addr_for,
    create_load_pub_writer,
    create_load_snapshot_reader,
    create_load_snapshot_writer,
    should_use_zmq,
    zmq_reader_owner,
)
from sglang.srt.managers.scheduler_components.load_publisher import (
    FAIL_WARN_PERIOD_S,
    SchedulerLoadPublisher,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


def _ipc_addr() -> str:
    fd, path = tempfile.mkstemp(prefix="sglang_test_zmq_", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return f"ipc://{path}"


def _warmup_zmq(writers, reader, attempts=20, interval=0.05):
    """Send warmup messages until the reader receives from all writers."""
    expected = {w.dp_rank for w in writers}
    received = set()
    for _ in range(attempts):
        for w in writers:
            w.write(LoadSnapshot(dp_rank=w.dp_rank, timestamp=-1.0, num_running_reqs=0))
        time.sleep(interval)
        for rank in expected:
            load = reader.read(rank)
            if load is not None:
                received.add(rank)
        if received >= expected:
            return
    raise RuntimeError(f"warmup failed: expected {expected}, received {received}")


class TestShmRoundTrip(CustomTestCase):
    def test_single_rank_write_read(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            writer.write(
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=5,
                    timestamp=1.0,
                    num_active_tokens=4096,
                    total_prefill_uncached_tokens=1000,
                    total_prefill_busy_us=250_000,
                    decode_moments=[2, 30, 3000, 500, 50_000, 60],
                )
            )
            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 5)
            self.assertEqual(load.timestamp, 1.0)
            self.assertEqual(load.num_active_tokens, 4096)
            # The cumulative total_* counters round-trip like any other
            # core scalar.
            self.assertEqual(load.total_prefill_uncached_tokens, 1000)
            self.assertEqual(load.total_prefill_busy_us, 250_000)
            self.assertEqual(load.decode_moments[0], 2)
            self.assertEqual(load.decode_moments[5], 60)
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_multi_rank_write_read_all(self):
        path = _temp_path()
        writers = []
        try:
            for rank in range(4):
                w = ShmLoadSnapshotWriter(path, dp_size=4, dp_rank=rank)
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        num_running_reqs=rank * 10,
                        timestamp=1.0,
                    )
                )
                writers.append(w)

            reader = ShmLoadSnapshotReader(path, dp_size=4)
            loads = reader.read_all()
            self.assertEqual(len(loads), 4)
            for i, load in enumerate(loads):
                self.assertEqual(load.dp_rank, i)
                self.assertEqual(load.num_running_reqs, i * 10)
            reader.close()
        finally:
            for w in writers:
                w.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_reader_empty_before_writer(self):
        path = _temp_path()
        reader = ShmLoadSnapshotReader(path, dp_size=2)
        self.assertEqual(reader.read_all(), [])
        self.assertIsNone(reader.read(0))
        reader.close()


class TestZmqRoundTrip(CustomTestCase):
    def test_single_rank_zmq_to_shm(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=2)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=2, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=7, timestamp=2.0))
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 7)
            self.assertEqual(load.timestamp, 2.0)
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_multi_rank_zmq(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 4
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)
        writers = []
        try:
            for rank in range(dp_size):
                w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
                writers.append(w)

            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(dp_rank=rank, num_running_reqs=rank + 1, timestamp=3.0)
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)
            for load in loads:
                self.assertEqual(load.num_running_reqs, load.dp_rank + 1)
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_read_returns_latest(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            for i in range(10):
                writer.write(
                    LoadSnapshot(dp_rank=0, num_running_reqs=i, timestamp=float(i))
                )
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 9)
            self.assertEqual(load.timestamp, 9.0)
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_zmq_writer_noblock_without_reader(self):
        addr = _ipc_addr()
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=1, timestamp=1.0))
        finally:
            writer.close()
            ipc_path = addr[len("ipc://") :]
            if os.path.exists(ipc_path):
                os.unlink(ipc_path)

    def test_reader_ipc_cleanup(self):
        addr = _ipc_addr()
        shm_path = _temp_path()
        ipc_path = addr[len("ipc://") :]
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        self.assertTrue(os.path.exists(ipc_path))
        reader.close()
        self.assertFalse(os.path.exists(ipc_path))
        if os.path.exists(shm_path):
            os.unlink(shm_path)


class TestFactoryFunctions(CustomTestCase):
    def test_shm_mode(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
            tokenizer_worker_num=1,
        )
        port_args = SimpleNamespace(instance_id="test_shm_factory")
        writer = create_load_snapshot_writer(
            server_args, port_args, dp_size=1, dp_rank=0
        )
        self.assertIsInstance(writer, ShmLoadSnapshotWriter)
        reader = create_load_snapshot_reader(
            server_args, port_args, caller="TokenizerManager"
        )
        self.assertIsInstance(reader, ShmLoadSnapshotReader)
        reader.close()
        writer.close()
        from sglang.srt.managers.load_snapshot import shm_path_for

        path = shm_path_for("test_shm_factory")
        if os.path.exists(path):
            os.unlink(path)

    def test_zmq_mode_via_env(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
            tokenizer_worker_num=1,
        )
        port_args = SimpleNamespace(instance_id="test_zmq_factory")
        os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"] = "1"
        try:
            writer = create_load_snapshot_writer(
                server_args, port_args, dp_size=1, dp_rank=0
            )
            self.assertIsInstance(writer, ZmqLoadSnapshotWriter)
            reader = create_load_snapshot_reader(
                server_args, port_args, caller="TokenizerManager"
            )
            self.assertIsInstance(reader, ZmqShmLoadSnapshotReader)
            reader.close()
            writer.close()
        finally:
            del os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"]

    def test_should_use_zmq_multinode_dp_attention(self):
        args = SimpleNamespace(enable_dp_attention=True, nnodes=2)
        self.assertTrue(should_use_zmq(args))


def _kv_args(endpoint="tcp://*:5557", dp_size=1, load_publish_endpoint=None):
    return SimpleNamespace(
        enable_dp_attention=False,
        nnodes=1,
        dp_size=dp_size,
        load_balance_method="round_robin",
        node_rank=0,
        tokenizer_worker_num=1,
        page_size=64,
        kv_events_config=(
            None
            if endpoint is None
            else f'{{"publisher":"zmq","endpoint":"{endpoint}"}}'
        ),
        load_publish_endpoint=load_publish_endpoint,
    )


class TestLoadPubEndpoint(CustomTestCase):
    """The load range is packed after the KV range; underivable => no publishing."""

    def test_packed_after_kv_range(self):
        # dp_size 2 => KV holds 5557-5558, load base is 5559.
        args = _kv_args(dp_size=2)
        self.assertEqual(load_pub_endpoint(args, 0), "tcp://*:5559")
        self.assertEqual(load_pub_endpoint(args, 1), "tcp://*:5560")

    def test_advertised_base_matches_bound_port(self):
        """`/server_info` must never advertise a range the writer won't bind.

        Both resolve through `_load_pub_range`; this pins that the descriptor
        really does route through it. The individual decline reasons are covered
        below -- one representative case here is enough to catch the descriptor
        deriving the range on its own again.
        """
        from sglang.srt.server_args import ServerArgs

        def _args(endpoint, dp_size):
            return ServerArgs(
                model_path="dummy",
                dp_size=dp_size,
                page_size=64,
                kv_events_config=(f'{{"publisher":"zmq","endpoint":"{endpoint}"}}'),
            )

        for dp_size in (1, 2, 8):
            with self.subTest(dp_size=dp_size):
                sa = _args("tcp://*:5557", dp_size)
                advertised = sa.describe_kv_events_publisher()[
                    "load_endpoint_port_base"
                ]
                for rank in range(dp_size):
                    self.assertEqual(
                        load_pub_endpoint(sa, rank),
                        f"tcp://*:{advertised + rank}",
                    )

        declined = _args("tcp://10.0.0.5:5557", 1)
        self.assertNotIn(
            "load_endpoint_port_base",
            declined.describe_kv_events_publisher() or {},
        )
        self.assertIsNone(load_pub_endpoint(declined, 0))

    def test_declines_non_tcp_endpoint(self):
        # ipc:// is valid for KV events but has no port to offset.
        self.assertIsNone(load_pub_endpoint(_kv_args(endpoint="ipc:///tmp/kv"), 0))

    def test_declines_when_range_overflows_u16(self):
        self.assertIsNone(load_pub_endpoint(_kv_args(endpoint="tcp://*:65535"), 0))

    def test_declines_without_kv_events(self):
        self.assertIsNone(load_pub_endpoint(_kv_args(endpoint=None), 0))

    def test_declines_null_publisher(self):
        args = _kv_args()
        args.kv_events_config = '{"publisher":"null"}'
        self.assertIsNone(load_pub_endpoint(args, 0))

    def test_declines_concrete_host(self):
        # Would be connected to rather than bound -- see `_load_pub_range`.
        self.assertIsNone(
            load_pub_endpoint(_kv_args(endpoint="tcp://10.0.0.5:5557"), 0)
        )

    def test_declines_concrete_ipv6_host(self):
        """`::` appears in every IPv6 address, so a substring test would call a
        concrete remote host bindable -- advertised, then EADDRNOTAVAIL."""
        self.assertIsNone(
            load_pub_endpoint(_kv_args(endpoint="tcp://[2001:db8::5]:5557"), 0)
        )

    def test_declines_explicit_range_inside_the_kv_range(self):
        """KV binds its own range later and unguarded, so an explicit endpoint
        landing inside it would kill startup blaming the KV publisher."""
        args = _kv_args(
            endpoint="tcp://*:5557", dp_size=4, load_publish_endpoint="tcp://*:5558"
        )
        self.assertIsNone(load_pub_endpoint(args, 0))

    def test_declines_replay_collision_for_a_concrete_replay_host(self):
        """The replay socket is bound for any host, so the collision check
        cannot be gated on the endpoint being wildcard-shaped."""
        args = _kv_args()
        args.kv_events_config = (
            '{"publisher":"zmq","endpoint":"tcp://*:5557",'
            '"replay_endpoint":"tcp://127.0.0.1:5558"}'
        )
        self.assertIsNone(load_pub_endpoint(args, 0))

    def test_declines_range_colliding_with_replay_endpoint(self):
        """The replay socket binds later and unguarded, so taking its port
        would kill startup blaming the KV publisher."""
        args = _kv_args()
        args.kv_events_config = (
            '{"publisher":"zmq","endpoint":"tcp://*:5557",'
            '"replay_endpoint":"tcp://*:5558"}'
        )
        self.assertIsNone(load_pub_endpoint(args, 0))

    def test_explicit_endpoint_needs_an_advertisable_kv_config(self):
        """Discovery rides on the kv-events descriptor, so binding a range that
        /server_info will not describe claims a port no router can find."""
        self.assertIsNone(
            load_pub_endpoint(
                _kv_args(endpoint=None, load_publish_endpoint="tcp://*:6000"), 0
            )
        )
        self.assertIsNone(
            load_pub_endpoint(
                _kv_args(
                    endpoint="ipc:///tmp/kv", load_publish_endpoint="tcp://*:6000"
                ),
                0,
            )
        )

    def test_binding_and_advertising_agree(self):
        """The invariant both directions: never advertise what we will not
        bind, never bind what we will not advertise."""
        from sglang.srt.server_args import ServerArgs

        cases = [
            ('{"publisher":"zmq","endpoint":"tcp://*:5557"}', None, 64),
            ('{"publisher":"zmq","endpoint":"tcp://10.0.0.5:5557"}', None, 64),
            ('{"publisher":"zmq","endpoint":"ipc:///tmp/kv"}', "tcp://*:6000", 64),
            ('{"publisher":"null"}', "tcp://*:6000", 64),
            ('{"publisher":"zmq","endpoint":"tcp://*:5557"}', "tcp://*:6000", 64),
            ('{"publisher":"zmq","endpoint":"tcp://*:5557"}', None, 0),
        ]
        for kv, explicit, page_size in cases:
            with self.subTest(kv=kv, explicit=explicit, page_size=page_size):
                sa = ServerArgs(
                    model_path="dummy",
                    dp_size=1,
                    page_size=page_size,
                    kv_events_config=kv,
                    load_publish_endpoint=explicit,
                )
                descriptor = sa.describe_kv_events_publisher() or {}
                self.assertEqual(
                    "load_endpoint_port_base" in descriptor,
                    load_pub_endpoint(sa, 0) is not None,
                    "advertising and binding must agree",
                )

    def test_declines_ambiguous_bare_ipv6(self):
        # `::1:5557` cannot be split into host and port unambiguously; it is
        # bind-style by the wildcard test, so only the address parse rejects it.
        self.assertIsNone(load_pub_endpoint(_kv_args(endpoint="tcp://::1:5557"), 0))

    def test_accepts_every_bind_style_host(self):
        for host in ("*", "0.0.0.0", "[::]"):
            with self.subTest(host=host):
                args = _kv_args(endpoint=f"tcp://{host}:5557")
                self.assertEqual(load_pub_endpoint(args, 0), f"tcp://{host}:5558")

    def test_kv_events_config_yields_a_router_facing_writer(self):
        """Without this, deleting the router writer entirely -- or any
        construction error swallowed into a log line -- leaves the suite green
        while the feature is a no-op."""
        # A free ephemeral port, so a busy 5558 cannot turn this into a flake
        # whose only symptom is create_load_pub_writer swallowing the bind.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        args = _kv_args(
            endpoint="tcp://*:5557", load_publish_endpoint=f"tcp://*:{free_port}"
        )
        writer = create_load_pub_writer(args, dp_rank=0)
        self.assertIsInstance(writer, PubLoadSnapshotWriter)
        try:
            self.assertEqual(writer.endpoint, f"tcp://*:{free_port}")
        finally:
            writer.close()

    def test_no_router_writer_without_a_range(self):
        self.assertIsNone(create_load_pub_writer(_kv_args(endpoint=None), dp_rank=0))


class _FakeWriter:
    """Minimal stand-in exposing the sink contract the publisher depends on."""

    def __init__(self, fail=False):
        self.fail = fail
        self.written = []

    def write(self, snapshot):
        if self.fail:
            raise ValueError("boom")
        self.written.append(snapshot)

    def close(self):
        pass


def _publisher(internal=None, router=None, get_loads=None, **kw):
    inquirer = SimpleNamespace(get_loads=get_loads)
    return SchedulerLoadPublisher(
        inquirer=inquirer, internal_writer=internal, router_writer=router, **kw
    )


class TestLoadPublisher(CustomTestCase):
    """Each sink runs on its own schedule; the snapshot is collected once."""

    def _counting_loads(self):
        calls = []

        def get_loads():
            calls.append(1)
            return LoadSnapshot(dp_rank=0, num_running_reqs=len(calls))

        return get_loads, calls

    def test_collects_once_for_both_sinks(self):
        """get_loads walks the waiting queue and four disagg queues, so
        collecting per sink would multiply that cost per forward pass."""
        get_loads, calls = self._counting_loads()
        a, b = _FakeWriter(), _FakeWriter()
        _publisher(internal=a, router=b, get_loads=get_loads).publish()

        self.assertEqual(len(calls), 1, "one collection per call, not per sink")
        self.assertIs(a.written[0], b.written[0], "both sinks get the same object")

    def test_no_sinks_never_collects(self):
        get_loads, calls = self._counting_loads()
        _publisher(get_loads=get_loads).publish(force=True)
        self.assertEqual(calls, [], "no sink => no reason to walk the queues")

    def test_internal_sink_is_iteration_throttled(self):
        get_loads, _ = self._counting_loads()
        w = _FakeWriter()
        pub = _publisher(internal=w, get_loads=get_loads, internal_interval=3)
        for _ in range(3):
            pub.publish()
        self.assertEqual(len(w.written), 1)

    def test_force_bypasses_the_internal_throttle_and_resets_it(self):
        get_loads, _ = self._counting_loads()
        w = _FakeWriter()
        pub = _publisher(internal=w, get_loads=get_loads, internal_interval=100)
        pub.publish(force=True)
        self.assertEqual(len(w.written), 1)
        pub.publish()
        pub.publish()
        self.assertEqual(len(w.written), 1, "force must reset, not skip, the counter")

    def test_router_sink_is_wall_clock_throttled_and_force_cannot_bypass_it(self):
        """The whole point of the separate cadence: an idle scheduler forces a
        publish on every loop spin, and that must not reach the network."""
        get_loads, _ = self._counting_loads()
        w = _FakeWriter()
        pub = _publisher(router=w, get_loads=get_loads, router_min_period_s=30.0)
        for _ in range(1000):
            pub.publish(force=True)
        self.assertEqual(len(w.written), 1, "1000 forced spins inside the floor => 1")

    def test_router_sink_publishes_again_after_the_floor_elapses(self):
        get_loads, _ = self._counting_loads()
        w = _FakeWriter()
        pub = _publisher(router=w, get_loads=get_loads, router_min_period_s=0.0)
        pub.publish()
        pub.publish()
        self.assertEqual(len(w.written), 2)

    def test_one_failing_sink_does_not_starve_the_other(self):
        """The router socket must never cost /v1/loads its snapshot."""
        get_loads, _ = self._counting_loads()
        bad, good = _FakeWriter(fail=True), _FakeWriter()
        _publisher(router=bad, internal=good, get_loads=get_loads).publish()
        self.assertEqual(len(good.written), 1)

    def test_collection_failure_publishes_nothing_and_does_not_raise(self):
        def get_loads():
            raise RuntimeError("collector down")

        w = _FakeWriter()
        _publisher(internal=w, get_loads=get_loads).publish()
        self.assertEqual(w.written, [])

    def test_repeated_failures_are_throttled_by_wall_clock(self):
        """A count-based bound would still scale with the loop rate, which is
        the flood it is supposed to stop -- so the bound is in seconds."""
        get_loads, _ = self._counting_loads()
        bad = _FakeWriter(fail=True)
        pub = _publisher(internal=bad, get_loads=get_loads)

        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.load_publisher", level="WARNING"
        ) as logs:
            for _ in range(5000):
                pub.publish(force=True)

        self.assertEqual(
            len(logs.output), 1, "5000 rapid failures inside one warn period => 1 line"
        )
        self.assertIn("5000 consecutive", logs.output[0])

    def test_failure_counter_resets_after_a_success(self):
        get_loads, _ = self._counting_loads()
        w = _FakeWriter(fail=True)
        pub = _publisher(internal=w, get_loads=get_loads)
        pub.publish(force=True)
        w.fail = False
        pub.publish(force=True)
        self.assertEqual(pub._failures, {}, "'consecutive' must mean consecutive")

    def test_warn_period_is_wall_clock(self):
        self.assertIsInstance(FAIL_WARN_PERIOD_S, float)


class TestPubLoadSnapshotWriter(CustomTestCase):
    """Wire framing the router's SUB tasks require: 3 frames, BE i64 seq."""

    def test_emits_three_frame_multipart_decodable_as_snapshot(self):
        import zmq

        from sglang.srt.managers.load_snapshot import snapshot_decoder

        # ipc:// takes the writer's bind branch and needs no free TCP port, so
        # this neither collides under parallel CI nor silently degrades into a
        # connect-connect pair where nothing listens.
        endpoint = _ipc_addr()
        writer = PubLoadSnapshotWriter(endpoint, dp_size=1, dp_rank=0)
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(endpoint)
        try:
            # PUB/SUB drops messages sent before the subscription propagates.
            deadline = time.time() + 5
            frames = None
            while time.time() < deadline and frames is None:
                writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=7))
                if sub.poll(100):
                    frames = sub.recv_multipart()
            self.assertIsNotNone(frames, "no load frame received within 5s")

            self.assertEqual(len(frames), 3, "router drops any non-3-frame message")
            self.assertEqual(frames[0], b"", "load socket publishes an empty topic")
            self.assertEqual(len(frames[1]), 8, "seq frame must be 8 bytes")
            self.assertGreaterEqual(
                int.from_bytes(frames[1], "big", signed=True),
                0,
                "seq counts up from 0, never the -1 the frame format reserves",
            )
            decoded = snapshot_decoder.decode(frames[2])
            self.assertEqual(decoded.num_running_reqs, 7)
        finally:
            sub.close()
            writer.close()

    def test_rejects_mismatched_dp_rank(self):
        writer = PubLoadSnapshotWriter(_ipc_addr(), dp_size=2, dp_rank=1)
        try:
            with self.assertRaises(ValueError):
                writer.write(LoadSnapshot(dp_rank=0))
        finally:
            writer.close()

    def test_refuses_a_non_bindable_endpoint(self):
        """Connecting would reach nobody and report nothing, so construction
        fails instead of producing a writer indistinguishable from a healthy one."""
        with self.assertRaises(ValueError):
            PubLoadSnapshotWriter("tcp://10.0.0.5:6000", dp_size=1, dp_rank=0)

    def test_binds_rather_than_connects(self):
        """A second bind on the same address fails, proving the first bound."""
        import zmq

        addr = _ipc_addr()
        first = PubLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            with self.assertRaises(
                zmq.ZMQError, msg="second bind must fail, proving the first bound"
            ):
                PubLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        finally:
            first.close()

    def test_snapshot_wire_shape_is_a_map_keyed_by_field_name(self):
        """Out-of-process subscribers decode this by field name.

        Declaring `array_like=True` on LoadSnapshot, or renaming a field, would
        keep the rest of the suite green while breaking every subscriber. Not
        hypothetical: the five nested sub-structs in this very payload
        (MemoryMetrics, QueueMetrics, ...) are all declared array_like, so it is
        an easy edit to make by analogy.

        `test_v1_loads_aggregate.py` pins the same names through the SHM path.
        """
        import msgspec.msgpack

        from sglang.srt.managers.load_snapshot import snapshot_encoder

        payload = snapshot_encoder.encode(
            LoadSnapshot(
                num_running_reqs=5,
                num_waiting_reqs=2,
                num_used_tokens=100,
                max_total_num_tokens=1000,
            )
        )
        decoded = msgspec.msgpack.decode(payload)
        self.assertIsInstance(
            decoded, dict, "router decodes a msgpack map, not an array"
        )
        self.assertEqual(decoded["num_running_reqs"], 5)
        self.assertEqual(decoded["num_waiting_reqs"], 2)
        self.assertEqual(decoded["num_used_tokens"], 100)
        self.assertEqual(decoded["max_total_num_tokens"], 1000)


class TestZmqReaderOwner(CustomTestCase):
    """At most one process binds the zmq PULL socket across all callers."""

    CALLERS = ("TokenizerManager", "MultiTokenizerRouter", "DataParallelController")

    @staticmethod
    def _args(**overrides):
        base = dict(
            enable_dp_attention=True,
            nnodes=2,
            node_rank=0,
            dp_size=1,
            load_balance_method="round_robin",
            tokenizer_worker_num=1,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _owners(self, args):
        return {c for c in self.CALLERS if zmq_reader_owner(args, c)}

    def test_zmq_disabled_no_owner(self):
        args = self._args(enable_dp_attention=False, nnodes=1)
        self.assertEqual(self._owners(args), set())

    def test_non_zero_node_rank_no_owner(self):
        args = self._args(node_rank=1, dp_size=4, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), set())

    def test_tokenizer_manager_owns_when_dp1(self):
        self.assertEqual(self._owners(self._args(dp_size=1)), {"TokenizerManager"})

    def test_multi_tokenizer_router_owns_in_multi_tokenizer_dp1(self):
        args = self._args(dp_size=1, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), {"MultiTokenizerRouter"})

    def test_multi_tokenizer_router_owns_in_multi_tokenizer_round_robin(self):
        args = self._args(dp_size=4, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), {"MultiTokenizerRouter"})

    def test_data_parallel_controller_owns_load_aware(self):
        for method in ("total_tokens", "total_requests"):
            args = self._args(
                dp_size=4, tokenizer_worker_num=8, load_balance_method=method
            )
            self.assertEqual(self._owners(args), {"DataParallelController"})

    def test_tokenizer_manager_owns_dp4_round_robin(self):
        args = self._args(dp_size=4, tokenizer_worker_num=1)
        self.assertEqual(self._owners(args), {"TokenizerManager"})

    def test_at_most_one_owner_across_configs(self):
        for dp_size in (1, 4):
            for tw in (1, 8):
                for method in ("round_robin", "total_tokens", "total_requests"):
                    for node_rank in (0, 1):
                        args = self._args(
                            dp_size=dp_size,
                            tokenizer_worker_num=tw,
                            load_balance_method=method,
                            node_rank=node_rank,
                        )
                        self.assertLessEqual(len(self._owners(args)), 1, args)


class TestZmqAddr(CustomTestCase):
    def test_ipc_for_single_node(self):
        port_args = SimpleNamespace(instance_id="myinstance")
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("ipc://"))
        self.assertIn("myinstance", addr)

    def test_tcp_from_port_args(self):
        from sglang.srt.utils.network import NetworkAddress

        port_args = SimpleNamespace(
            instance_id="myinstance",
            load_collector_ipc_name=NetworkAddress("10.0.0.1", 29506).to_tcp(),
        )
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("tcp://"))
        self.assertIn("10.0.0.1", addr)


class TestEndToEndZmqSimulation(CustomTestCase):
    """Simulate multi-node DP attention on single machine using IPC."""

    def test_full_flow_dp_size_2(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 2

        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)

        writers = []
        for rank in range(dp_size):
            w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
            writers.append(w)

        try:
            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=1.0,
                        num_running_reqs=10 + rank,
                        num_waiting_reqs=5 + rank,
                        num_total_tokens=100 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)
            self.assertEqual(loads[0].num_running_reqs, 10)
            self.assertEqual(loads[1].num_running_reqs, 11)
            self.assertEqual(loads[0].num_total_tokens, 100)
            self.assertEqual(loads[1].num_total_tokens, 150)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=2.0,
                        num_running_reqs=20 + rank,
                        num_waiting_reqs=0,
                        num_total_tokens=200 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(loads[0].num_running_reqs, 20)
            self.assertEqual(loads[1].num_running_reqs, 21)
            self.assertEqual(loads[0].num_total_tokens, 200)
            self.assertEqual(loads[1].num_total_tokens, 250)
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)


if __name__ == "__main__":
    unittest.main()
