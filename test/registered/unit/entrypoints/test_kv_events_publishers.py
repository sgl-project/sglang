import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.validation_hook import warn_multi_node_kv_events
from sglang.srt.disaggregation.kv_events import (
    EventPublisher,
    EventPublisherFactory,
    KVEventsConfig,
    build_advertised_publisher_entry,
    is_loopback_host,
    merge_kv_event_publishers,
    offset_advertisable_tcp,
    resolve_advertised_host,
    resolve_advertised_tcp_endpoint,
)
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.entrypoints.server_utils import build_server_info
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_publisher import (
    SchedulerLoadPublisher,
)
from sglang.srt.runtime_context import (
    describe_kv_events_publisher,
    get_context,
    get_exec,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.rust_server import config as rust_config
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _call_http_server_info(
    server_args: ServerArgs,
    scheduler_info: dict | None = None,
) -> dict:
    async def _fake_internal_state():
        return [{"max_req_input_len": 1024}]

    tokenizer_manager = SimpleNamespace(
        server_args=server_args,
        get_internal_state=_fake_internal_state,
        startup_time={},
    )
    stub_state = SimpleNamespace(
        tokenizer_manager=tokenizer_manager,
        scheduler_info=scheduler_info if scheduler_info is not None else {},
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(stub_state)
    try:
        chat = SimpleNamespace(supports_inline_system=False)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=chat))
        )
        return asyncio.run(http_server.server_info(request))
    finally:
        http_server._global_state = prior_state


class TestAdvertisedHostAndEndpointResolution(CustomTestCase):
    def test_wildcard_hosts_resolve_via_local_ip_auto_ipv4_and_ipv6(self):
        for wildcard in ("*", "0.0.0.0", "::", "[::]"):
            with (
                self.subTest(wildcard=wildcard, family="ipv4"),
                patch(
                    "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                    return_value="10.0.1.15",
                ),
            ):
                self.assertEqual(resolve_advertised_host(wildcard), "10.0.1.15")

        for wildcard in ("*", "::", "[::]"):
            with (
                self.subTest(wildcard=wildcard, family="ipv6"),
                patch(
                    "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                    return_value="2001:db8::42",
                ),
            ):
                self.assertEqual(resolve_advertised_host(wildcard), "2001:db8::42")

        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            return_value="::1",
        ):
            # IPv4 wildcard bind does not serve ::1; map loopback IPv6 to 127.0.0.1.
            self.assertEqual(resolve_advertised_host("0.0.0.0"), "127.0.0.1")

    def test_wildcard_hosts_prefer_explicit_advertise_host(self):
        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            side_effect=AssertionError(
                "should not probe local IP when advertise_host is set"
            ),
        ):
            self.assertEqual(
                resolve_advertised_tcp_endpoint(
                    "tcp://*:5557", advertise_host="10.20.30.40"
                ),
                "tcp://10.20.30.40:5557",
            )
            self.assertEqual(
                resolve_advertised_tcp_endpoint(
                    "tcp://0.0.0.0:5558", advertise_host="node-a.cluster.local"
                ),
                "tcp://node-a.cluster.local:5558",
            )
            self.assertEqual(
                resolve_advertised_tcp_endpoint(
                    "tcp://[::]:5559", advertise_host="2001:db8::7"
                ),
                "tcp://[2001:db8::7]:5559",
            )
            self.assertEqual(
                resolve_advertised_tcp_endpoint(
                    "tcp://[::]:5560", advertise_host="[2001:db8::8]"
                ),
                "tcp://[2001:db8::8]:5560",
            )

    def test_concrete_hosts_are_preserved_and_ipv6_bracketed(self):
        self.assertEqual(
            resolve_advertised_tcp_endpoint(
                "tcp://10.0.0.5:5557", advertise_host="10.9.9.9"
            ),
            "tcp://10.0.0.5:5557",
        )
        self.assertEqual(
            resolve_advertised_tcp_endpoint("tcp://[2001:db8::5]:5557"),
            "tcp://[2001:db8::5]:5557",
        )

    def test_non_tcp_or_overflow_endpoints_return_none(self):
        for bad in (None, "", "inproc://cache", "ipc:///tmp/kv.sock", "tcp://*:abc"):
            with self.subTest(endpoint=bad):
                self.assertIsNone(resolve_advertised_tcp_endpoint(bad))
                self.assertIsNone(offset_advertisable_tcp(bad, 1))
        self.assertIsNone(offset_advertisable_tcp("tcp://*:65535", 1))
        self.assertEqual(
            offset_advertisable_tcp("tcp://[::]:5557", 2), "tcp://[::]:5559"
        )

    def test_is_loopback_host(self):
        for host in (
            "127.0.0.1",
            "127.0.1.1",
            "::1",
            "[::1]",
            "localhost",
            "LOCALHOST",
        ):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_host(host))
        for host in (
            "*",
            "0.0.0.0",
            "::",
            "10.0.0.1",
            "2001:db8::1",
            "node-0.internal",
        ):
            with self.subTest(host=host):
                self.assertFalse(is_loopback_host(host))

    def test_build_advertised_publisher_entry_optional_fields(self):
        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            return_value="10.0.0.10",
        ):
            minimal = build_advertised_publisher_entry(
                dp_rank=0, kv_endpoint="tcp://*:5557"
            )
            self.assertEqual(
                minimal,
                {"dp_rank": 0, "endpoint": "tcp://10.0.0.10:5557"},
            )
            self.assertNotIn("replay_endpoint", minimal)
            self.assertNotIn("load_endpoint", minimal)

            with_replay = build_advertised_publisher_entry(
                dp_rank=1,
                kv_endpoint="tcp://*:5558",
                replay_endpoint="tcp://*:5658",
            )
            self.assertEqual(
                with_replay,
                {
                    "dp_rank": 1,
                    "endpoint": "tcp://10.0.0.10:5558",
                    "replay_endpoint": "tcp://10.0.0.10:5658",
                },
            )
            self.assertNotIn("load_endpoint", with_replay)

            with_load = build_advertised_publisher_entry(
                dp_rank=2,
                kv_endpoint="tcp://*:5559",
                load_endpoint="tcp://*:5563",
            )
            self.assertEqual(
                with_load,
                {
                    "dp_rank": 2,
                    "endpoint": "tcp://10.0.0.10:5559",
                    "load_endpoint": "tcp://10.0.0.10:5563",
                },
            )
            self.assertNotIn("replay_endpoint", with_load)

            full = build_advertised_publisher_entry(
                dp_rank=3,
                kv_endpoint="tcp://[::]:5560",
                replay_endpoint="tcp://[::]:5660",
                load_endpoint="tcp://[::]:5760",
                advertise_host="2001:db8::3",
            )
            self.assertEqual(
                full,
                {
                    "dp_rank": 3,
                    "endpoint": "tcp://[2001:db8::3]:5560",
                    "replay_endpoint": "tcp://[2001:db8::3]:5660",
                    "load_endpoint": "tcp://[2001:db8::3]:5760",
                },
            )

    def test_factory_strips_advertise_host_before_constructing_publisher(self):
        captured = {}

        class _DummyPublisher(EventPublisher):
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def publish(self, events) -> None:
                return

            def shutdown(self) -> None:
                return

        key = "_test_dummy_advertise_host_pub"
        EventPublisherFactory.register_publisher(key, _DummyPublisher)
        self.addCleanup(EventPublisherFactory._registry.pop, key, None)

        cfg = json.dumps(
            {
                "publisher": key,
                "endpoint": "tcp://*:5557",
                "advertise_host": "10.0.0.88",
            }
        )
        parsed = KVEventsConfig.from_cli(cfg)
        self.assertEqual(parsed.advertise_host, "10.0.0.88")

        EventPublisherFactory.create(cfg, attn_dp_rank=2)
        self.assertEqual(captured["attn_dp_rank"], 2)
        self.assertNotIn("advertise_host", captured)
        self.assertNotIn("format", captured)


class TestServerInfoKvEventsPublishers(CustomTestCase):
    def test_multinode_dp2_and_dp4_publishers_with_backward_compat_fields(self):
        args_dp2 = ServerArgs(
            model_path="dummy",
            kv_events_config=json.dumps(
                {
                    "publisher": "zmq",
                    "endpoint": "tcp://*:5557",
                    "replay_endpoint": "tcp://*:5567",
                    "topic": "kv@prefill",
                }
            ),
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=2,
        )
        dp2_publishers = [
            {
                "dp_rank": 1,
                "endpoint": "tcp://10.0.0.2:5558",
                "replay_endpoint": "tcp://10.0.0.2:5568",
                "load_endpoint": "tcp://10.0.0.2:5560",
            },
            {
                "dp_rank": 0,
                "endpoint": "tcp://10.0.0.1:5557",
                "replay_endpoint": "tcp://10.0.0.1:5567",
                "load_endpoint": "tcp://10.0.0.1:5559",
            },
        ]
        sched_info = {
            "max_req_input_len": 1024,
            "kv_event_publishers": dp2_publishers,
        }
        expected_kv = {
            "publisher": "zmq",
            "endpoint_host": "*",
            "endpoint_port_base": 5557,
            "topic": "kv@prefill",
            "block_size": 64,
            "dp_size": 2,
            "load_endpoint_port_base": 5559,
            "load_topic": "load",
            "replay_endpoint_port_base": 5567,
            "publishers": [
                {
                    "dp_rank": 0,
                    "endpoint": "tcp://10.0.0.1:5557",
                    "replay_endpoint": "tcp://10.0.0.1:5567",
                    "load_endpoint": "tcp://10.0.0.1:5559",
                },
                {
                    "dp_rank": 1,
                    "endpoint": "tcp://10.0.0.2:5558",
                    "replay_endpoint": "tcp://10.0.0.2:5568",
                    "load_endpoint": "tcp://10.0.0.2:5560",
                },
            ],
        }

        # 1. build_server_info (called twice to verify sched_info is not mutated)
        for _ in range(2):
            built = build_server_info(args_dp2, sched_info)
            self.assertEqual(built["kv_events"], expected_kv)
            self.assertNotIn("kv_event_publishers", built)

        # 2. HTTP /server_info
        http_info = _call_http_server_info(args_dp2, scheduler_info=sched_info)
        self.assertEqual(http_info["kv_events"], expected_kv)
        self.assertNotIn("kv_event_publishers", http_info)

        # 3. gRPC RuntimeHandle.get_server_info
        bridge = RuntimeHandle.__new__(RuntimeHandle)
        bridge.tokenizer_manager = SimpleNamespace(
            server_args=args_dp2, mm_processor=None
        )
        bridge.scheduler_info = sched_info
        grpc_info = json.loads(bridge.get_server_info())
        self.assertEqual(grpc_info["kv_events"], expected_kv)
        self.assertNotIn("kv_event_publishers", grpc_info)

        # 4. Embedded Rust _launch_server_info
        stub_scheduler = SimpleNamespace(
            server_args=args_dp2,
            max_total_num_tokens=4096,
            max_req_input_len=1024,
            kv_event_publishers=dp2_publishers,
        )
        rust_info = json.loads(rust_config._launch_server_info(stub_scheduler))
        self.assertEqual(rust_info["kv_events"], expected_kv)
        self.assertNotIn("kv_event_publishers", rust_info)

        # 5. Python Engine.get_server_info
        async def _internal_state():
            return [{"max_req_input_len": 1024}]

        engine = Engine.__new__(Engine)
        engine.tokenizer_manager = SimpleNamespace(
            server_args=args_dp2,
            get_internal_state=_internal_state,
            startup_time={},
        )
        engine.loop = asyncio.new_event_loop()
        self.addCleanup(engine.loop.close)
        engine._scheduler_init_result = SimpleNamespace(scheduler_infos=[sched_info])
        engine_info = engine.get_server_info()
        self.assertEqual(engine_info["kv_events"], expected_kv)
        self.assertNotIn("kv_event_publishers", engine_info)

    def test_multinode_dp4_mixed_ipv4_and_ipv6_hosts(self):
        args_dp4 = ServerArgs(
            model_path="dummy",
            kv_events_config=json.dumps(
                {"publisher": "zmq", "endpoint": "tcp://[::]:5557", "topic": ""}
            ),
            page_size=128,
            tp_size=8,
            attn_dp_size=4,
            nnodes=2,
        )
        publishers = [
            {"dp_rank": 0, "endpoint": "tcp://10.0.0.10:5557"},
            {"dp_rank": 1, "endpoint": "tcp://10.0.0.10:5558"},
            {"dp_rank": 2, "endpoint": "tcp://[2001:db8::11]:5559"},
            {"dp_rank": 3, "endpoint": "tcp://[2001:db8::11]:5560"},
        ]
        desc = describe_kv_events_publisher(args_dp4, publishers=publishers)
        self.assertEqual(desc["endpoint_host"], "[::]")
        self.assertEqual(desc["endpoint_port_base"], 5557)
        self.assertEqual(desc["dp_size"], 4)
        self.assertEqual(desc["block_size"], 128)
        self.assertNotIn("replay_endpoint_port_base", desc)
        self.assertNotIn("load_endpoint_port_base", desc)
        self.assertEqual(desc["publishers"], publishers)

    def test_single_node_fallback_and_disabled_configs(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            page_size=64,
            dp_size=1,
        )
        # Without gathered publishers, "publishers" is omitted for legacy compat.
        desc = describe_kv_events_publisher(args)
        self.assertNotIn("publishers", desc)
        built = build_server_info(args, {})
        self.assertNotIn("publishers", built["kv_events"])

        # Disabled / invalid config stays None even if publishers list is passed.
        for bad_args in (
            ServerArgs(model_path="dummy", page_size=64),
            ServerArgs(
                model_path="dummy",
                kv_events_config='{"publisher": "null"}',
                page_size=64,
            ),
            ServerArgs(
                model_path="dummy",
                kv_events_config='{"publisher": "zmq", "endpoint": "inproc://kv"}',
                page_size=64,
            ),
            ServerArgs(
                model_path="dummy",
                kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
                page_size=0,
            ),
        ):
            self.assertIsNone(
                describe_kv_events_publisher(
                    bad_args,
                    publishers=[{"dp_rank": 0, "endpoint": "tcp://10.0.0.1:5557"}],
                )
            )


class TestSchedulerAndControllerPublisherGathering(CustomTestCase):
    def tearDown(self):
        reset_context()
        super().tearDown()

    def _make_scheduler(
        self,
        server_args: ServerArgs,
        *,
        enable_kv: bool = True,
        load_endpoint: str | None = None,
    ) -> Scheduler:
        component = SchedulerKvEventsPublisher.__new__(SchedulerKvEventsPublisher)
        component.kv_events_config = server_args.kv_events_config
        component.enable_kv_cache_events = enable_kv
        component.kv_event_publisher = MagicMock() if enable_kv else None

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = server_args
        scheduler.max_total_num_tokens = 2048
        scheduler.max_req_input_len = 512
        scheduler.startup_time = {}
        scheduler.page_size = server_args.page_size or 64
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.kv_events_publisher = component
        scheduler.load_publisher = SimpleNamespace(
            enable=load_endpoint is not None,
            endpoint=load_endpoint,
        )
        return scheduler

    def test_multinode_world_group_gather_collects_only_publishing_ranks(self):
        cfg = json.dumps(
            {
                "publisher": "zmq",
                "endpoint": "tcp://*:5557",
                "replay_endpoint": "tcp://*:5657",
            }
        )
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=cfg,
            load_publish_endpoint="auto",
            page_size=64,
            tp_size=4,
            attn_dp_size=2,
            nnodes=2,
            dist_init_addr="10.0.0.10:29500",
        )
        publish(args, role="scheduler")

        # Simulate 4 ranks across 2 nodes:
        #   node 0: dp_rank 0 (attn_tp_rank 0, publishes) + dp_rank 0 (attn_tp_rank 1, non-pub)
        #   node 1: dp_rank 1 (attn_tp_rank 0, publishes) + dp_rank 1 (attn_tp_rank 1, non-pub)
        rank_specs = [
            (0, 0, True, "10.0.0.10", "tcp://*:5559"),
            (0, 1, False, "10.0.0.10", None),
            (1, 0, True, "10.0.0.11", "tcp://*:5560"),
            (1, 1, False, "10.0.0.11", None),
        ]
        local_entries = []
        for attn_dp_rank, attn_tp_rank, is_pub, node_ip, load_ep in rank_specs:
            sched = self._make_scheduler(args, enable_kv=is_pub, load_endpoint=load_ep)
            with (
                get_parallel().override(
                    pp_rank=0,
                    attn_dp_size=2,
                    attn_dp_rank=attn_dp_rank,
                    attn_tp_rank=attn_tp_rank,
                    attn_cp_rank=0,
                    dp_rank=None,
                ),
                patch(
                    "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                    return_value=node_ip,
                ),
            ):
                entry = sched.kv_events_publisher.describe_advertised_publisher(
                    load_endpoint=load_ep
                )
            local_entries.append(entry)

        # Only the two attn_tp_rank==0 ranks produced entries.
        self.assertIsNotNone(local_entries[0])
        self.assertIsNone(local_entries[1])
        self.assertIsNotNone(local_entries[2])
        self.assertIsNone(local_entries[3])

        # Now verify Scheduler.init_kv_event_publishers() calls world_group.all_gather_object
        # and populates get_init_info()["kv_event_publishers"].
        fake_world_group = MagicMock(world_size=4)
        fake_world_group.all_gather_object.return_value = local_entries

        sched0 = self._make_scheduler(
            args, enable_kv=True, load_endpoint="tcp://*:5559"
        )
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=2,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=None,
                world_group=fake_world_group,
            ),
            patch(
                "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                return_value="10.0.0.10",
            ),
        ):
            sched0.init_kv_event_publishers()
            init_info = sched0.get_init_info()

        fake_world_group.all_gather_object.assert_called_once_with(local_entries[0])
        expected = [
            {
                "dp_rank": 0,
                "endpoint": "tcp://10.0.0.10:5557",
                "replay_endpoint": "tcp://10.0.0.10:5657",
                "load_endpoint": "tcp://10.0.0.10:5559",
            },
            {
                "dp_rank": 1,
                "endpoint": "tcp://10.0.0.11:5558",
                "replay_endpoint": "tcp://10.0.0.11:5658",
                "load_endpoint": "tcp://10.0.0.11:5560",
            },
        ]
        self.assertEqual(sched0.kv_event_publishers, expected)
        self.assertEqual(init_info["kv_event_publishers"], expected)

        # Feed directly into build_server_info to verify end-to-end wire contract.
        server_info = build_server_info(args, init_info)
        self.assertEqual(server_info["kv_events"]["publishers"], expected)
        self.assertEqual(server_info["kv_events"]["endpoint_port_base"], 5557)
        self.assertEqual(server_info["kv_events"]["replay_endpoint_port_base"], 5657)
        self.assertEqual(server_info["kv_events"]["load_endpoint_port_base"], 5559)

    def test_single_node_and_disabled_and_ep_joiner_skip_collective(self):
        # 1. Single-node world_size=1 skips all_gather_object and still populates publishers
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            page_size=64,
        )
        publish(args, role="scheduler")
        fake_wg_single = MagicMock(world_size=1)
        sched = self._make_scheduler(args, enable_kv=True)
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=None,
                world_group=fake_wg_single,
            ),
            patch(
                "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                return_value="192.168.1.5",
            ),
        ):
            sched.init_kv_event_publishers()
        fake_wg_single.all_gather_object.assert_not_called()
        self.assertEqual(
            sched.kv_event_publishers,
            [{"dp_rank": 0, "endpoint": "tcp://192.168.1.5:5557"}],
        )

        # 2. EP joiner skips all_gather_object even when world_size > 1
        fake_wg_multi = MagicMock(world_size=4)
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=None,
                world_group=fake_wg_multi,
            ),
            get_exec().moe.override(is_ep_joiner=True),
            patch(
                "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                return_value="192.168.1.6",
            ),
        ):
            sched.init_kv_event_publishers()
        fake_wg_multi.all_gather_object.assert_not_called()
        self.assertEqual(
            sched.kv_event_publishers,
            [{"dp_rank": 0, "endpoint": "tcp://192.168.1.6:5557"}],
        )

        # 3. Disabled kv_events_config skips collective and leaves get_init_info clean
        reset_context()
        args_off = ServerArgs(model_path="dummy", page_size=64)
        publish(args_off, role="scheduler")
        sched_off = self._make_scheduler(args_off, enable_kv=False)
        with get_parallel().override(world_group=fake_wg_multi):
            sched_off.init_kv_event_publishers()
            init_info = sched_off.get_init_info()
        fake_wg_multi.all_gather_object.assert_not_called()
        self.assertEqual(sched_off.kv_event_publishers, [])
        self.assertNotIn("kv_event_publishers", init_info)

    def test_runtime_loopback_warning_on_multinode(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            page_size=64,
            tp_size=2,
            attn_dp_size=2,
            nnodes=2,
            dist_init_addr="10.0.0.10:29500",
        )
        publish(args, role="scheduler")
        sched = self._make_scheduler(args, enable_kv=True)
        fake_wg = MagicMock(world_size=2)
        fake_wg.all_gather_object.return_value = [
            {"dp_rank": 0, "endpoint": "tcp://127.0.0.1:5557"},
            {"dp_rank": 1, "endpoint": "tcp://10.0.0.11:5558"},
        ]
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=2,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=None,
                world_group=fake_wg,
            ),
            patch(
                "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
                return_value="127.0.0.1",
            ),
            self.assertLogs("sglang.srt.managers.scheduler", level="WARNING") as logs,
        ):
            sched.init_kv_event_publishers()
        self.assertTrue(any("loopback" in msg for msg in logs.output))

    def test_load_publisher_endpoint_property_tracks_bind_success_and_close(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            load_publish_endpoint="auto",
            page_size=64,
            dp_size=2,
        )
        publish(args, role="scheduler")
        fake_sock = MagicMock()
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=1,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_publisher._open_pub_socket",
                return_value=fake_sock,
            ),
        ):
            pub = SchedulerLoadPublisher(
                kv_events_config=args.kv_events_config,
                load_publish_endpoint="auto",
            )
        self.assertTrue(pub.enable)
        self.assertEqual(pub.endpoint, "tcp://*:5560")
        pub.close()
        self.assertFalse(pub.enable)
        self.assertIsNone(pub.endpoint)

        # Bind failure leaves endpoint as None.
        with (
            get_parallel().override(
                pp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_tp_rank=0,
                attn_cp_rank=0,
                dp_rank=0,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_publisher._open_pub_socket",
                side_effect=OSError("port busy"),
            ),
        ):
            pub_fail = SchedulerLoadPublisher(
                kv_events_config=args.kv_events_config,
                load_publish_endpoint="auto",
            )
        self.assertFalse(pub_fail.enable)
        self.assertIsNone(pub_fail.endpoint)

    def test_dp_controller_merges_and_deduplicates_publishers(self):
        from sglang.srt.managers import data_parallel_controller as module

        controller = DataParallelController.__new__(DataParallelController)
        controller.env_lock = threading.Lock()
        controller._scheduler_init_info_lock = threading.Lock()
        controller._scheduler_init_infos = []
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.kv_event_publishers = []
        controller.run_scheduler_process_func = MagicMock()
        controller.attn_replica_size = 1
        controller.prelaunched_schedulers = None

        pub0 = {"dp_rank": 0, "endpoint": "tcp://10.0.0.1:5557"}
        pub1 = {"dp_rank": 1, "endpoint": "tcp://10.0.0.2:5558"}
        # Simulate two local schedulers each carrying the full gathered list.
        readers = [
            MagicMock(
                recv=MagicMock(
                    return_value={
                        "max_total_num_tokens": 64,
                        "max_req_input_len": 32,
                        "kv_event_publishers": [pub1, pub0],
                    }
                )
            ),
            MagicMock(
                recv=MagicMock(
                    return_value={
                        "max_total_num_tokens": 64,
                        "max_req_input_len": 32,
                        "kv_event_publishers": [pub0, pub1],
                    }
                )
            ),
        ]
        with (
            get_context().override_server_args(
                tp_size=2,
                attn_dp_size=2,
                nnodes=1,
                node_rank=0,
                pp_size=1,
                attn_cp_size=1,
            ),
            patch.object(module.mp, "Process"),
            patch.object(
                module.mp,
                "Pipe",
                side_effect=[(reader, MagicMock()) for reader in readers],
            ),
            patch.object(module.PortArgs, "init_new", return_value=MagicMock()),
        ):
            controller.launch_tensor_parallel_group(
                ServerArgs(model_path="dummy"), MagicMock(), 0, None
            )

        self.assertEqual(controller.kv_event_publishers, [pub0, pub1])

    def test_merge_kv_event_publishers_dedup_sort_and_sanitize(self):
        merged = merge_kv_event_publishers(
            [
                None,
                {
                    "dp_rank": 2,
                    "endpoint": "tcp://10.0.0.2:5559",
                    "replay_endpoint": None,
                },
                {
                    "dp_rank": 0,
                    "endpoint": "tcp://10.0.0.1:5557",
                    "load_endpoint": "tcp://10.0.0.1:5561",
                },
                {"dp_rank": 2, "endpoint": "tcp://10.0.0.2:5559"},
                {"dp_rank": 1, "endpoint": ""},
            ]
        )
        self.assertEqual(
            merged,
            [
                {
                    "dp_rank": 0,
                    "endpoint": "tcp://10.0.0.1:5557",
                    "load_endpoint": "tcp://10.0.0.1:5561",
                },
                {"dp_rank": 2, "endpoint": "tcp://10.0.0.2:5559"},
            ],
        )


class TestWarnMultiNodeKvEvents(CustomTestCase):
    def _assert_no_warning(self, server_args: ServerArgs):
        with patch("sglang.srt.arg_groups.validation_hook.logger.warning") as mock_warn:
            warn_multi_node_kv_events(server_args)
            mock_warn.assert_not_called()

    def _assert_warns(self, server_args: ServerArgs) -> str:
        with patch("sglang.srt.arg_groups.validation_hook.logger.warning") as mock_warn:
            warn_multi_node_kv_events(server_args)
            mock_warn.assert_called_once()
            return mock_warn.call_args.args[0]

    def test_multinode_wildcard_bind_does_not_warn(self):
        for endpoint in ("tcp://*:5557", "tcp://0.0.0.0:5557", "tcp://[::]:5557"):
            with self.subTest(endpoint=endpoint):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=json.dumps(
                        {"publisher": "zmq", "endpoint": endpoint}
                    ),
                    load_publish_endpoint="auto",
                    page_size=64,
                    tp_size=8,
                    attn_dp_size=4,
                    nnodes=2,
                    node_rank=0,
                )
                self._assert_no_warning(args)

    def test_multinode_wildcard_with_non_loopback_advertise_host_does_not_warn(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=json.dumps(
                {
                    "publisher": "zmq",
                    "endpoint": "tcp://*:5557",
                    "advertise_host": "10.0.0.10",
                }
            ),
            load_publish_endpoint="auto",
            page_size=64,
            tp_size=8,
            attn_dp_size=4,
            nnodes=2,
            node_rank=0,
        )
        self._assert_no_warning(args)

    def test_multinode_loopback_bind_warns(self):
        for endpoint in (
            "tcp://127.0.0.1:5557",
            "tcp://[::1]:5557",
            "tcp://localhost:5557",
        ):
            with self.subTest(endpoint=endpoint):
                args = ServerArgs(
                    model_path="dummy",
                    kv_events_config=json.dumps(
                        {"publisher": "zmq", "endpoint": endpoint, "bind": True}
                    ),
                    page_size=64,
                    tp_size=8,
                    attn_dp_size=4,
                    nnodes=2,
                    node_rank=0,
                )
                msg = self._assert_warns(args)
                self.assertIn("loopback", msg)
                self.assertIn("KV events", msg)

    def test_multinode_wildcard_with_loopback_advertise_host_warns(self):
        args = ServerArgs(
            model_path="dummy",
            kv_events_config=json.dumps(
                {
                    "publisher": "zmq",
                    "endpoint": "tcp://*:5557",
                    "advertise_host": "127.0.0.1",
                }
            ),
            load_publish_endpoint="auto",
            page_size=64,
            tp_size=8,
            attn_dp_size=4,
            nnodes=2,
            node_rank=0,
        )
        msg = self._assert_warns(args)
        self.assertIn("KV events", msg)
        self.assertIn("load reports", msg)
        self.assertIn("127.0.0.1", msg)

    def test_single_node_follower_and_connect_style_do_not_warn(self):
        # Single node
        self._assert_no_warning(
            ServerArgs(
                model_path="dummy",
                kv_events_config=json.dumps(
                    {
                        "publisher": "zmq",
                        "endpoint": "tcp://127.0.0.1:5557",
                        "bind": True,
                    }
                ),
                page_size=64,
                dp_size=2,
                nnodes=1,
                node_rank=0,
            )
        )
        # Follower node
        self._assert_no_warning(
            ServerArgs(
                model_path="dummy",
                kv_events_config=json.dumps(
                    {
                        "publisher": "zmq",
                        "endpoint": "tcp://127.0.0.1:5557",
                        "bind": True,
                    }
                ),
                page_size=64,
                tp_size=8,
                attn_dp_size=4,
                nnodes=2,
                node_rank=1,
            )
        )
        # Connect-style (bind=False)
        self._assert_no_warning(
            ServerArgs(
                model_path="dummy",
                kv_events_config=json.dumps(
                    {
                        "publisher": "zmq",
                        "endpoint": "tcp://127.0.0.1:5557",
                        "bind": False,
                    }
                ),
                page_size=64,
                tp_size=8,
                attn_dp_size=4,
                nnodes=2,
                node_rank=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
