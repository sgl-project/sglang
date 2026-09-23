"""CPU coverage for node-local metadata and follower gRPC lifecycle."""

import json
import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints import engine
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import common
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
    def test_aggregates_direct_scheduler_and_controller_sources(self):
        for dp_size, infos, expected in (
            (
                1,
                [{"kv_event_sources": []}, {"kv_event_sources": [source(0)]}],
                [source(0)],
            ),
            (8, [{"kv_event_sources": [source(5), source(4)]}], [source(4), source(5)]),
        ):
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
            self.assertEqual(result.scheduler_infos[0]["kv_event_sources"], expected)

    def test_disabled_and_legacy_grpc_do_not_load_native_server(self):
        for config in (
            {"grpc_port": None},
            {"grpc_port": 50051, "smg_grpc_mode": True},
            {"grpc_port": 50051, "grpc_mode": True},
        ):
            with (
                self.subTest(config=config),
                get_context().override_server_args(**config),
                patch("sglang.srt.rust_extensions.load_rust_extension") as load,
            ):
                self.assertIsNone(common.start_follower_grpc_server(None, {}))
                load.assert_not_called()


class TestFollowerGrpcLifecycle(unittest.TestCase):
    @contextmanager
    def launch(self, *, blocking=True, sources=None):
        result = engine.SchedulerInitResult(
            scheduler_infos=[], block_until_scheduler_exits=MagicMock()
        )
        # Publishing readiness supplies the metadata: starting the server too
        # early must fail rather than reading a pre-populated fixture.
        result.wait_for_ready = lambda: result.scheduler_infos.append(
            {"kv_event_sources": sources or []}
        )
        args = MagicMock(launch_command="sglang serve ...")
        args.resolved_dict.return_value = {
            "node_rank": 1,
            "dist_init_addr": "leader:5000",
        }
        handle = MagicMock()
        native = MagicMock(start_metadata_server=MagicMock(return_value=handle))
        with (
            get_context().override_server_args(
                node_rank=1, nnodes=2, host="127.0.0.1", grpc_port=50051
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
            patch.object(common, "describe_kv_events_publisher", return_value=None),
            patch.object(engine, "launch_dummy_health_check_server"),
            patch.object(engine, "kill_process_tree") as kill,
        ):
            yield SimpleNamespace(
                run=lambda: engine.Engine._launch_subprocesses(
                    args,
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    port_args=MagicMock(),
                ),
                result=result,
                handle=handle,
                native=native,
                kill=kill,
            )

    def test_starts_after_scheduler_ready_and_stops_on_exit(self):
        with self.launch(sources=[source(4)]) as follower:
            follower.result.block_until_scheduler_exits.side_effect = (
                follower.handle.shutdown.assert_not_called
            )
            follower.run()
            follower.result.block_until_scheduler_exits.assert_called_once_with()
            follower.native.start_metadata_server.assert_called_once()
            info = json.loads(
                follower.native.start_metadata_server.call_args.kwargs[
                    "server_info_json"
                ]
            )
            self.assertEqual(info["kv_event_sources"], [source(4)])
            self.assertEqual(info["node_rank"], 1)
            self.assertEqual(info["dist_init_addr"], "leader:5000")
            follower.handle.shutdown.assert_called_once_with()

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
            # Nodes without publishers still expose metadata.
            follower.native.start_metadata_server.assert_called_once()
            follower.handle.shutdown.assert_not_called()
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
            follower.kill.assert_called_once_with(12345, wait_timeout=60)


if __name__ == "__main__":
    unittest.main()
