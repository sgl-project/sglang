from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt import rust_extensions
from sglang.srt.entrypoints.engine import node_hosts_rust_server
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.rust_server import server as rust_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "nnodes,tp_size,dp_size,ep_join_mode,ranks,expected",
    [
        (2, 4, 4, None, (0, 1, 2, 3), [0, 1, 0, 1]),
        (4, 4, 2, None, (0, 2), [0, 0]),
        (2, 2, 2, "scale", (0, 1), [0, 1]),
    ],
    ids=["multiple-listeners-per-node", "dp-spans-nodes", "scale-joiner"],
)
def test_dp_leaders_reuse_node_local_ports(
    nnodes, tp_size, dp_size, ep_join_mode, ranks, expected
):
    with (
        get_context().override_server_args(
            nnodes=nnodes,
            tp_size=tp_size,
            dp_size=dp_size,
            enable_dp_attention=True,
            ep_join_mode=ep_join_mode,
            host="0.0.0.0",
            port=30000,
        ),
        patch.object(rust_extensions, "load_rust_extension") as extension,
        patch.object(rust_server, "_partition_cores", return_value=(None, None)),
        patch.object(rust_server, "_build_server_args"),
    ):
        parallel = get_parallel()
        ports = []
        for dp_rank, tp_rank in enumerate(ranks):
            scheduler = SimpleNamespace(
                server_args=SimpleNamespace(),
                ps=SimpleNamespace(
                    tp_rank=tp_rank,
                    tp_size=parallel.tp_size,
                    pp_size=parallel.pp_size,
                    attn_tp_size=parallel.attn_tp_size,
                    attn_cp_size=parallel.attn_cp_size,
                    attn_dp_rank=dp_rank,
                    dp_size=dp_size,
                ),
                model_config=SimpleNamespace(is_multimodal=False),
            )
            ports.append(rust_server.RustServer.launch(scheduler).http_port)

        calls = extension.return_value.Server.call_args_list
        assert [c.kwargs["port_offset"] for c in calls] == expected
        # P/D bootstrap must register against the same ports Rust binds.
        assert ports == [30000 + offset for offset in expected]


@pytest.mark.parametrize(
    "pp_size,expected",
    [(1, [True, False, True, False]), (2, [True, True, False, False])],
)
@pytest.mark.parametrize("node_rank", range(4))
def test_node_listener_placement(pp_size, expected, node_rank):
    with get_context().override_server_args(
        nnodes=4,
        node_rank=node_rank,
        tp_size=4,
        pp_size=pp_size,
        dp_size=2,
        enable_dp_attention=True,
        attn_cp_size=2,
    ):
        assert node_hosts_rust_server() == expected[node_rank]


def test_scale_joiner_hosts_listener():
    with get_context().override_server_args(
        nnodes=2,
        node_rank=1,
        tp_size=1,
        dp_size=1,
        enable_dp_attention=True,
        ep_join_mode="scale",
    ):
        assert node_hosts_rust_server()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
