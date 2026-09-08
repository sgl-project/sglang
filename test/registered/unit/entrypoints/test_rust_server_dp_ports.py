import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt import rust_extensions
from sglang.srt.rust_server import server as rust_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRustServerDpLocalPorts(CustomTestCase):
    def test_dp_leaders_reuse_node_local_ports(self):
        # TP4/DP4: two listeners per node. TP4/DP2: each DP group spans two nodes.
        for nnodes, dp_size, attn_tp_size, ranks, expected in (
            (2, 4, 1, (0, 1, 2, 3), [0, 1, 0, 1]),
            (4, 2, 2, (0, 2), [0, 0]),
        ):
            with (
                self.subTest(nnodes=nnodes, dp_size=dp_size),
                patch.object(rust_extensions, "load_rust_extension") as extension,
                patch.object(
                    rust_server,
                    "get_parallel",
                    return_value=SimpleNamespace(nnodes=nnodes),
                ),
                patch.object(
                    rust_server,
                    "get_serving",
                    return_value=SimpleNamespace(host="0.0.0.0", port=30000),
                ),
                patch.object(
                    rust_server, "_partition_cores", return_value=(None, None)
                ),
                patch.object(rust_server, "_build_server_args"),
            ):
                ports = []
                for dp_rank, tp_rank in enumerate(ranks):
                    scheduler = SimpleNamespace(
                        server_args=SimpleNamespace(),
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
                    ports.append(rust_server.RustServer.launch(scheduler).http_port)

                calls = extension.return_value.Server.call_args_list
                self.assertEqual([c.kwargs["port_offset"] for c in calls], expected)
                # P/D bootstrap must register against the same ports Rust binds.
                self.assertEqual(ports, [30000 + offset for offset in expected])


if __name__ == "__main__":
    unittest.main()
