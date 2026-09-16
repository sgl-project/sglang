"""CPU coverage for node-local metadata and follower gRPC lifecycle."""

import json
import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints import engine, grpc_metadata
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def source(rank):
    return dict(
        dp_rank=rank,
        endpoint=f"tcp://127.0.0.1:{5557 + rank}",
        topic="kv",
        block_size=64,
    )


class TestGrpcMetadata(unittest.TestCase):
    def test_leader_and_follower_use_the_same_metadata_contract(self):
        for node_rank, ranks in ((0, range(4)), (1, range(4, 8)), (1, [])):
            with self.subTest(node_rank=node_rank, ranks=ranks):
                args = SimpleNamespace(
                    resolved_dict=lambda: dict(
                        node_rank=node_rank,
                        nnodes=2,
                        dp_size=8,
                        dist_init_addr="leader:5000",
                        disaggregation_mode="prefill",
                    ),
                    launch_command="sglang serve ...",
                )
                info = {"kv_event_sources": [source(rank) for rank in ranks]}
                handle = RuntimeHandle.__new__(RuntimeHandle)
                handle.tokenizer_manager = SimpleNamespace(server_args=args)
                handle.scheduler_info = info
                with patch.object(
                    grpc_metadata, "describe_kv_events_publisher", return_value=None
                ):
                    actual = grpc_metadata.get_server_info_json(args, info)
                    self.assertEqual(handle.get_server_info(), actual)
                decoded = json.loads(actual)
                self.assertEqual(decoded["kv_event_sources"], info["kv_event_sources"])
                self.assertEqual(decoded["node_rank"], node_rank)
                self.assertEqual(decoded["dist_init_addr"], "leader:5000")
                self.assertEqual(decoded["disaggregation_mode"], "prefill")
                self.assertEqual(decoded["launch_command"], args.launch_command)
                self.assertIsNone(decoded["kv_events"])

    def test_aggregates_direct_scheduler_and_controller_sources(self):
        for dp_size, infos in (
            (1, [{"kv_event_sources": []}, {"kv_event_sources": [source(0)]}]),
            (8, [{"kv_event_sources": [source(5), source(4)]}]),
            (1, [{"kv_event_sources": []}]),
            (1, [{"status": "ready"}]),
        ):
            expected = sorted(
                [item for info in infos for item in info.get("kv_event_sources", [])],
                key=lambda item: item["dp_rank"],
            )
            with (
                self.subTest(dp_size=dp_size, infos=infos),
                get_context().override_server_args(
                    dp_size=dp_size, tp_size=1, pp_size=1, nnodes=1, node_rank=0
                ),
                patch.object(engine.mp, "Process"),
                patch.object(
                    engine.mp, "Pipe", return_value=(MagicMock(), MagicMock())
                ),
                patch.object(engine.TorchMemorySaverAdapter, "create"),
                patch.object(engine.numa_utils, "configure_subprocess"),
                patch.object(
                    engine, "maybe_reindex_device_id", return_value=nullcontext(0)
                ),
                patch.object(engine, "_wait_for_scheduler_ready", return_value=infos),
            ):
                result, _ = engine.Engine._launch_scheduler_processes(
                    MagicMock(), MagicMock(), MagicMock()
                )
                result.wait_for_ready()
            if "kv_event_sources" in infos[0]:
                self.assertEqual(
                    result.scheduler_infos[0]["kv_event_sources"], expected
                )
            else:
                self.assertNotIn("kv_event_sources", result.scheduler_infos[0])


class TestFollowerGrpcLifecycle(unittest.TestCase):
    @contextmanager
    def launch(self, *, blocking=True, sources=None, **config):
        events = []
        result = engine.SchedulerInitResult(
            scheduler_infos=[{"kv_event_sources": sources or []}],
            wait_for_ready=MagicMock(side_effect=lambda: events.append("ready")),
            block_until_scheduler_exits=MagicMock(
                side_effect=lambda: events.append("blocking")
            ),
        )
        handle = MagicMock()
        handle.shutdown.side_effect = lambda: events.append("shutdown")
        native = MagicMock()
        native.start_metadata_server.side_effect = lambda **kwargs: (
            events.append("grpc") or handle
        )
        with (
            get_context().override_server_args(
                **(
                    dict(
                        node_rank=1,
                        nnodes=2,
                        dp_size=8,
                        host="127.0.0.1",
                        grpc_port=50051,
                    )
                    | config
                )
            ),
            patch.dict(
                os.environ,
                {"SGLANG_BLOCK_NONZERO_RANK_CHILDREN": "1" if blocking else "0"},
            ),
            patch.object(engine, "configure_logger"),
            patch.object(engine, "_set_envs_and_config"),
            patch.object(engine, "load_plugins"),
            patch.object(engine, "publish"),
            patch.object(
                engine,
                "resolving_view",
                return_value=SimpleNamespace(
                    reasoning_parser=None, tool_call_parser=None
                ),
            ),
            patch.object(
                engine.Engine,
                "_launch_scheduler_processes",
                return_value=(result, [MagicMock(pid=12345)]),
            ),
            patch(
                "sglang.srt.rust_extensions.load_rust_extension", return_value=native
            ),
            patch.object(
                grpc_metadata, "get_server_info_json", return_value='{"node_rank":1}'
            ) as metadata,
            patch.object(
                engine,
                "launch_dummy_health_check_server",
                side_effect=lambda *args: events.append("health"),
            ) as health,
            patch.object(engine, "kill_process_tree") as kill,
        ):
            yield SimpleNamespace(
                run=lambda: engine.Engine._launch_subprocesses(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    port_args=MagicMock(),
                ),
                result=result,
                handle=handle,
                native=native,
                metadata=metadata,
                events=events,
                health=health,
                kill=kill,
            )

    def test_starts_after_scheduler_ready_and_stops_on_exit(self):
        for sources in ([source(4)], []):
            with (
                self.subTest(sources=sources),
                self.launch(sources=sources) as follower,
            ):
                follower.run()
                follower.native.start_metadata_server.assert_called_once_with(
                    host="127.0.0.1", port=50051, server_info_json='{"node_rank":1}'
                )
                self.assertEqual(
                    follower.metadata.call_args.args[1], {"kv_event_sources": sources}
                )
                self.assertEqual(
                    follower.events, ["ready", "grpc", "health", "blocking", "shutdown"]
                )

    def test_disabled_and_legacy_grpc_do_not_start_native_server(self):
        for config in (
            {"grpc_port": None},
            {"smg_grpc_mode": True},
            {"grpc_mode": True},
        ):
            with self.subTest(config=config), self.launch(**config) as follower:
                follower.run()
                follower.native.start_metadata_server.assert_not_called()
                follower.handle.shutdown.assert_not_called()

    def test_blocking_failure_stops_server(self):
        with self.launch() as follower:
            follower.result.block_until_scheduler_exits.side_effect = RuntimeError(
                "scheduler stopped"
            )
            with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
                follower.run()
            follower.handle.shutdown.assert_called_once_with()

    def test_nonblocking_engine_owns_server_shutdown(self):
        with self.launch(blocking=False) as follower:
            returned = follower.run()
            self.assertIs(returned[3], follower.result)
            self.assertIs(follower.result.grpc_server, follower.handle)
            follower.health.assert_not_called()
            instance = engine.Engine.__new__(engine.Engine)
            instance.tokenizer_manager = None
            instance._scheduler_init_result = follower.result
            instance.shutdown()
            instance.shutdown()
            follower.handle.shutdown.assert_called_once_with()
            self.assertIsNone(follower.result.grpc_server)

    def test_bind_failure_reaps_schedulers(self):
        with self.launch() as follower:
            follower.native.start_metadata_server.side_effect = OSError(
                "address in use"
            )
            with self.assertRaisesRegex(OSError, "address in use"):
                follower.run()
            follower.health.assert_not_called()
            follower.kill.assert_called_once_with(12345, wait_timeout=60)


if __name__ == "__main__":
    unittest.main()
