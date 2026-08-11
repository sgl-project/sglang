import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.environ import envs
from sglang.srt.managers.rust_server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRustServerDpLocalPorts(CustomTestCase):
    def test_two_nodes_reuse_the_same_http_ports(self):
        server_cls = MagicMock()
        server_args = SimpleNamespace(
            host="0.0.0.0",
            port=30000,
            nnodes=2,
            preferred_sampling_params=None,
            mm_processor_worker_num=0,
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.server": SimpleNamespace(_core=SimpleNamespace()),
                    "sglang.srt.server._core": SimpleNamespace(Server=server_cls),
                },
            ),
            patch.object(RustServer, "_partition_cores", return_value=(None, None)),
            patch.object(RustServer, "_build_server_args", return_value="{}"),
        ):
            for global_rank in range(8):
                scheduler = SimpleNamespace(
                    server_args=server_args,
                    ps=SimpleNamespace(attn_dp_rank=global_rank, dp_size=8),
                    model_config=SimpleNamespace(is_multimodal=False),
                )
                RustServer.launch(scheduler)

        ports = [call.kwargs["http_addr"] for call in server_cls.call_args_list]
        self.assertEqual(
            ports,
            [
                "0.0.0.0:30000",
                "0.0.0.0:30001",
                "0.0.0.0:30002",
                "0.0.0.0:30003",
            ]
            * 2,
        )

    def test_nonzero_node_does_not_start_dummy_server_for_rust_dp(self):
        server_args = SimpleNamespace(
            check_server_args=MagicMock(),
            remote_instance_weight_loader_start_seed_via_transfer_engine=False,
            reasoning_parser=None,
            tool_call_parser=None,
            weight_cache_mode=None,
            enable_elastic_expert_backup=False,
            elastic_ep_backend=None,
            node_rank=1,
            host="0.0.0.0",
            port=30000,
            enable_metrics=False,
            dp_size=8,
        )
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

        launch.assert_not_called()
        scheduler_init_result.wait_for_ready.assert_called_once_with()
        scheduler_init_result.block_until_scheduler_exits.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
