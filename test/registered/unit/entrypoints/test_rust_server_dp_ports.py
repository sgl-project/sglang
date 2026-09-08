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
    def _launch_rust_servers(self, nnodes, ranks, *, tp_size=4, cp_size=1, pp_size=1):
        server_cls = MagicMock()
        server_args = SimpleNamespace()

        with (
            patch(
                "sglang.srt.rust_server.server.get_parallel",
                return_value=SimpleNamespace(nnodes=nnodes),
            ),
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
                        tp_size=tp_size,
                        pp_size=pp_size,
                        attn_tp_size=attn_tp_size,
                        attn_cp_size=cp_size,
                        attn_dp_rank=dp_rank,
                        dp_size=dp_size,
                    ),
                    model_config=SimpleNamespace(is_multimodal=False),
                )
                with self.assertLogs(
                    "sglang.srt.rust_server.server", level="INFO"
                ) as logs:
                    server = RustServer.launch(scheduler)
                offset = server_cls.call_args.kwargs["port_offset"]
                self.assertEqual(server.http_port, 30000 + (offset or 0))
                self.assertIn(f"0.0.0.0:{server.http_port}", logs.output[-1])
                if dp_size > 1:
                    self.assertIn(f"DP rank {dp_rank}/{dp_size}", logs.output[-1])
                else:
                    self.assertNotIn("DP rank", logs.output[-1])
                self.assertEqual(scheduler.ps.attn_dp_rank, dp_rank)

        return [call.kwargs["port_offset"] for call in server_cls.call_args_list]

    def _launch_nonzero_node(
        self,
        *,
        nnodes,
        node_rank,
        tp_size,
        dp_size=None,
        pp_size=1,
        cp_size=1,
        rust=True,
    ):
        server_args = ServerArgs(
            model_path="dummy",
            nnodes=nnodes,
            node_rank=node_rank,
            tp_size=tp_size,
            pp_size=pp_size,
            attn_cp_size=cp_size,
            enable_dp_attention=True,
            host="0.0.0.0",
            port=30000,
            enable_metrics=True,
            **({"dp_size": dp_size} if dp_size is not None else {}),
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
            envs.SGLANG_RUST_SERVER.override(rust),
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
        return launch

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

    def test_cp_and_pp_do_not_change_node_local_ports(self):
        offsets = self._launch_rust_servers(
            nnodes=4,
            tp_size=8,
            cp_size=2,
            pp_size=2,
            ranks=((0, 0, 1, 4), (2, 1, 1, 4), (4, 2, 1, 4), (6, 3, 1, 4)),
        )
        self.assertEqual(offsets, [0, 1, 0, 1])

    def test_no_dp_keeps_port_offset_unset(self):
        offsets = self._launch_rust_servers(nnodes=2, ranks=((0, 0, 4, 1),))
        self.assertEqual(offsets, [None])

    def test_dummy_server_only_runs_without_a_rust_listener(self):
        cases = (
            (dict(nnodes=2, node_rank=1, tp_size=4, dp_size=4), False),
            (dict(nnodes=4, node_rank=1, tp_size=4, dp_size=2), True),
            (dict(nnodes=4, node_rank=2, tp_size=4, dp_size=2), False),
            (dict(nnodes=2, node_rank=1, tp_size=4), True),
            (dict(nnodes=2, node_rank=1, tp_size=4, dp_size=4, pp_size=2), True),
            (
                dict(nnodes=4, node_rank=1, tp_size=8, dp_size=4, cp_size=2, pp_size=2),
                False,
            ),
            (dict(nnodes=2, node_rank=1, tp_size=4, dp_size=4, rust=False), True),
        )
        for topology, needs_dummy in cases:
            with self.subTest(**topology):
                launch = self._launch_nonzero_node(**topology)
                if needs_dummy:
                    launch.assert_called_once_with("0.0.0.0", 30000, True)
                else:
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
