from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt import rust_extensions
from sglang.srt.entrypoints.engine import node_hosts_rust_server
from sglang.srt.runtime_context import get_context, get_parallel, get_serving
from sglang.srt.rust_server import server as rust_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "nnodes,tp_size,dp_size,ep_join_mode,ranks,bound_ports",
    [
        (2, 4, 4, None, (0, 1, 2, 3), [42107, 43329, 42107, 43329]),
        (4, 4, 2, None, (0, 2), [42107, 42107]),
        (2, 2, 2, "scale", (0, 1), [42107, 43329]),
    ],
    ids=["multiple-listeners-per-node", "dp-spans-nodes", "scale-joiner"],
)
def test_dp_leaders_publish_allocated_worker_ports(
    nnodes, tp_size, dp_size, ep_join_mode, ranks, bound_ports
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
        extension.return_value.Server.side_effect = [
            SimpleNamespace(http_port=port) for port in bound_ports
        ]
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
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                    pp_rank=0,
                    attn_tp_rank=0,
                    attn_cp_rank=0,
                ),
                model_config=SimpleNamespace(is_multimodal=False),
            )
            frontend = rust_server.RustServer.launch(scheduler)
            ports.append(frontend.http_port)
            assert frontend.topology.dp_rank == dp_rank

        calls = extension.return_value.Server.call_args_list
        assert [call.kwargs["http_port"] for call in calls] == [0] * dp_size
        assert all(call.kwargs.get("port_offset") is None for call in calls)
        assert get_serving().port == 30000
        # Discovery and P/D bootstrap must publish the ports Rust actually bound.
        assert ports == bound_ports


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
