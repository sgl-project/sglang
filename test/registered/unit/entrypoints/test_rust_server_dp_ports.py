from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt import rust_extensions
from sglang.srt.entrypoints.engine import node_hosts_rust_server
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.rust_server import config as rust_config
from sglang.srt.rust_server import server as rust_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scheduler_for_typed_config():
    return SimpleNamespace(
        server_args=SimpleNamespace(enable_return_hidden_states=False),
        model_config=SimpleNamespace(
            context_len=2048,
            vocab_size=1000,
            is_multimodal=False,
            hf_config=SimpleNamespace(model_type=None),
            get_default_sampling_params=lambda: {},
        ),
        rust_server_tokenizer_path=lambda: "tokenizer",
        max_total_num_tokens=1024,
    )


def _typed_config_extension():
    return SimpleNamespace(
        DisaggregationMode=SimpleNamespace(
            Null="null", Prefill="prefill", Decode="decode"
        ),
        ModelConfig=MagicMock(return_value="model-config"),
        DefaultSamplingParams=MagicMock(return_value="sampling-defaults"),
        ServerArgs=MagicMock(return_value="server-args"),
    )


@pytest.mark.parametrize(
    "legacy_args,expected",
    [({}, 50051), ({"smg_grpc_mode": True}, None), ({"grpc_mode": True}, None)],
    ids=["native", "legacy-smg", "deprecated-legacy-smg"],
)
def test_typed_config_only_forwards_native_grpc_port(legacy_args, expected):
    extension = _typed_config_extension()
    with (
        get_context().override_server_args(grpc_port=50051, **legacy_args),
        patch.object(rust_extensions, "load_rust_extension", return_value=extension),
        patch.object(rust_config, "compute_num_reserved_tokens", return_value=0),
    ):
        assert (
            rust_config._build_server_args(_scheduler_for_typed_config())
            == "server-args"
        )

    assert extension.ServerArgs.call_args.kwargs["grpc_port"] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_typed_config_forwards_request_decompression(enabled):
    extension = _typed_config_extension()
    with (
        get_context().override_server_args(),
        envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.override(enabled),
        patch.object(rust_extensions, "load_rust_extension", return_value=extension),
        patch.object(rust_config, "compute_num_reserved_tokens", return_value=0),
    ):
        rust_config._build_server_args(_scheduler_for_typed_config())

    kwargs = extension.ServerArgs.call_args.kwargs
    assert kwargs["enable_request_decompression"] is enabled


@pytest.mark.parametrize(
    "nnodes,tp_size,attn_dp_size,ep_join_mode,ranks,expected",
    [
        (2, 4, 4, None, (0, 1, 2, 3), [0, 1, 0, 1]),
        (4, 4, 2, None, (0, 2), [0, 0]),
        (2, 2, 2, "scale", (0, 1), [0, 1]),
    ],
    ids=["multiple-listeners-per-node", "dp-spans-nodes", "scale-joiner"],
)
def test_dp_leaders_reuse_node_local_ports(
    nnodes, tp_size, attn_dp_size, ep_join_mode, ranks, expected
):
    with (
        get_context().override_server_args(
            nnodes=nnodes,
            tp_size=tp_size,
            attn_dp_size=attn_dp_size,
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
                model_config=SimpleNamespace(is_multimodal=False),
            )
            with parallel.override(
                tp_rank=tp_rank,
                attn_dp_rank=dp_rank,
                attn_tp_rank=tp_rank % parallel.attn_tp_size,
                attn_cp_rank=0,
            ):
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
        attn_dp_size=2,
        attn_cp_size=2,
    ):
        assert node_hosts_rust_server() == expected[node_rank]


def test_scale_joiner_hosts_listener():
    with get_context().override_server_args(
        nnodes=2,
        node_rank=1,
        tp_size=1,
        ep_join_mode="scale",
    ):
        assert node_hosts_rust_server()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
