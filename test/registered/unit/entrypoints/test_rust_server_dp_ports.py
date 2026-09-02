import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.environ import envs
from sglang.srt.runtime_context import reset_context
from sglang.srt.rust_server.server import RustServer
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRustServerDpLocalPorts(CustomTestCase):
    def _launch_rust_servers(self, nnodes, ranks):
        server_cls = MagicMock()
        server_args = SimpleNamespace(nnodes=nnodes)

        with (
            patch(
                "sglang.srt.rust_extensions.load_rust_extension",
                return_value=SimpleNamespace(Server=server_cls),
            ),
            patch(
                "sglang.srt.rust_server.server.get_serving",
                return_value=SimpleNamespace(
                    host="0.0.0.0", port=30000, preferred_sampling_params=None
                ),
            ),
            patch(
                "sglang.srt.rust_server.server._partition_cores",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.rust_server.server._build_server_args",
                return_value="{}",
            ),
        ):
            for tp_rank, dp_rank, attn_tp_size, dp_size in ranks:
                scheduler = SimpleNamespace(
                    server_args=server_args,
                    ps=SimpleNamespace(
                        tp_rank=tp_rank,
                        tp_size=4,
                        pp_size=1,
                        attn_tp_size=attn_tp_size,
                        attn_cp_size=1,
                        attn_dp_rank=dp_rank,
                        dp_size=dp_size,
                    ),
                    model_config=SimpleNamespace(is_multimodal=False),
                )
                RustServer.launch(scheduler)

        return [call.kwargs["port_offset"] for call in server_cls.call_args_list]

    def _launch_nonzero_node(self, *, nnodes, node_rank, tp_size, dp_size):
        server_args = ServerArgs(
            model_path="dummy",
            nnodes=nnodes,
            node_rank=node_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            enable_dp_attention=True,
        )
        server_args.check_server_args = MagicMock()
        self.addCleanup(reset_context)
        scheduler_init_result = SimpleNamespace(
            all_child_pids=[],
            scheduler_infos=[],
            wait_for_ready=MagicMock(),
            block_until_scheduler_exits=MagicMock(),
            engine_info_bootstrap_server=None,
        )
        with (
            envs.SGLANG_RUST_SERVER.override(True),
            patch.object(engine_module, "configure_logger"),
            patch.object(engine_module, "_set_envs_and_config"),
            patch.object(engine_module, "load_plugins"),
            patch.object(
                Engine,
                "_launch_scheduler_processes",
                return_value=(scheduler_init_result, []),
            ),
            patch.object(engine_module, "launch_dummy_health_check_server") as launch,
        ):
            Engine._launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=MagicMock(),
                run_scheduler_process_func=MagicMock(),
                run_detokenizer_process_func=MagicMock(),
                port_args=SimpleNamespace(),
            )

        scheduler_init_result.wait_for_ready.assert_called_once_with()
        scheduler_init_result.block_until_scheduler_exits.assert_called_once_with()
        return launch, server_args

    def test_two_nodes_reuse_the_same_http_ports(self):
        offsets = self._launch_rust_servers(
            nnodes=2,
            ranks=(
                (0, 0, 1, 4),
                (1, 1, 1, 4),
                (2, 2, 1, 4),
                (3, 3, 1, 4),
            ),
        )
        self.assertEqual(offsets, [0, 1, 0, 1])

    def test_more_nodes_than_dp_ranks_reuse_the_base_port(self):
        offsets = self._launch_rust_servers(
            nnodes=4,
            ranks=((0, 0, 2, 2), (2, 1, 2, 2)),
        )
        self.assertEqual(offsets, [0, 0])

    def test_nonzero_node_does_not_start_dummy_server_for_rust_dp(self):
        launch, _ = self._launch_nonzero_node(
            nnodes=2, node_rank=1, tp_size=4, dp_size=4
        )

        launch.assert_not_called()

    def test_non_server_node_starts_dummy_server_for_rust_dp(self):
        launch, server_args = self._launch_nonzero_node(
            nnodes=4, node_rank=1, tp_size=4, dp_size=2
        )

        launch.assert_called_once_with(
            server_args.host, server_args.port, server_args.enable_metrics
        )


if __name__ == "__main__":
    unittest.main()
