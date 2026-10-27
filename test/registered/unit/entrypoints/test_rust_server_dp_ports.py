import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt import rust_extensions
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base.conn import BaseKVBootstrapServer
from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.entrypoints.engine import node_hosts_rust_server
from sglang.srt.environ import envs
from sglang.srt.managers import disagg_service
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.rust_server import config as rust_config
from sglang.srt.rust_server import server as rust_server
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
        get_init_info=MagicMock(
            side_effect=AssertionError("typed config must not read late init info")
        ),
        kv_events_publisher=SimpleNamespace(
            local_kv_event_sources=MagicMock(
                return_value=[{"endpoint": "tcp://127.0.0.1:5557"}]
            )
        ),
        page_size=16,
        rust_server_tokenizer_path=lambda: "tokenizer",
        tokenizer=None,
        max_total_num_tokens=1024,
        load_snapshot_writer=None,
    )


@pytest.mark.parametrize(
    "overrides,kwarg,expected",
    [
        ({}, "grpc_port", 50051),
        ({"hicache_storage_backend": "file"}, "hicache_storage_backend", "file"),
        (
            {"hicache_storage_backend_extra_config": "{}"},
            "hicache_storage_backend_extra_config",
            "{}",
        ),
        (
            {"hicache_storage_prefetch_policy": "timeout"},
            "hicache_storage_prefetch_policy",
            "timeout",
        ),
        ({"hicache_write_policy": "write_back"}, "hicache_write_policy", "write_back"),
        ({"smg_grpc_mode": True}, "grpc_port", None),
        ({"grpc_mode": True}, "grpc_port", None),
        ({"default_priority_value": 5}, "default_priority_value", None),
        (
            {"enable_priority_scheduling": True, "default_priority_value": 5},
            "default_priority_value",
            5,
        ),
        ({}, "limit_mm_data_per_request", []),
        (
            {"limit_mm_data_per_request": {"video": 1, "image": 2}},
            "limit_mm_data_per_request",
            [("video", 1), ("image", 2)],
        ),
    ],
    ids=[
        "native-grpc",
        "hicache_storage_backend",
        "hicache_storage_backend_extra_config",
        "hicache_storage_prefetch_policy",
        "hicache_write_policy",
        "legacy-smg",
        "deprecated-legacy-smg",
        "priority-off",
        "priority-on",
        "mm-limits-unset",
        "mm-limits-keep-order",
    ],
)
def test_typed_config_forwards_resolved_launch_fields(overrides, kwarg, expected):
    extension = SimpleNamespace(
        DisaggregationMode=SimpleNamespace(
            Null="null", Prefill="prefill", Decode="decode"
        ),
        ModelConfig=MagicMock(return_value="model-config"),
        DefaultSamplingParams=MagicMock(return_value="sampling-defaults"),
        ServerArgs=MagicMock(return_value="server-args"),
    )
    scheduler = _scheduler_for_typed_config()
    with (
        get_context().override_server_args(grpc_port=50051, **overrides),
        get_parallel().override(attn_dp_rank=0, dp_rank=None),
        patch.object(rust_extensions, "load_rust_extension", return_value=extension),
        patch.object(rust_config, "compute_num_reserved_tokens", return_value=0),
        patch.object(rust_config, "_launch_server_info", return_value="{}"),
    ):
        assert (
            rust_config._build_server_args(scheduler, default_max_new_tokens=None)
            == "server-args"
        )

    assert extension.ServerArgs.call_args.kwargs[kwarg] == expected
    assert extension.ServerArgs.call_args.kwargs["default_max_new_tokens"] is None
    scheduler.get_init_info.assert_not_called()
    scheduler.kv_events_publisher.local_kv_event_sources.assert_called_once_with(16)
    deployment_info = json.loads(
        extension.ServerArgs.call_args.kwargs["deployment_info"]
    )
    assert deployment_info["kv_event_sources"] == [{"endpoint": "tcp://127.0.0.1:5557"}]


@pytest.mark.parametrize(
    "server_args,ranks,expected",
    [
        ({"tp_size": 4, "attn_dp_size": 4}, {"attn_dp_rank": 2, "dp_rank": None}, 2),
        ({"dp_size": 2}, {"attn_dp_rank": 0, "dp_rank": 1}, 1),
        ({}, {"attn_dp_rank": 0, "dp_rank": None}, 0),
    ],
    ids=["attention-dp", "replica-dp", "no-dp"],
)
def test_deployment_info_reports_listener_dp_rank(server_args, ranks, expected):
    """Each DP listener must advertise its own rank, matching its KV-event
    source; a per-listener orchestrator cannot otherwise tell listeners apart."""
    extension = SimpleNamespace(
        DisaggregationMode=SimpleNamespace(
            Null="null", Prefill="prefill", Decode="decode"
        ),
        ModelConfig=MagicMock(return_value="model-config"),
        DefaultSamplingParams=MagicMock(return_value="sampling-defaults"),
        ServerArgs=MagicMock(return_value="server-args"),
    )
    with (
        get_context().override_server_args(**server_args),
        get_parallel().override(**ranks),
        patch.object(rust_extensions, "load_rust_extension", return_value=extension),
        patch.object(rust_config, "compute_num_reserved_tokens", return_value=0),
        patch.object(rust_config, "_launch_server_info", return_value="{}"),
    ):
        rust_config._build_server_args(_scheduler_for_typed_config())

    deployment_info = json.loads(
        extension.ServerArgs.call_args.kwargs["deployment_info"]
    )
    assert deployment_info["dp_rank"] == expected


@pytest.mark.parametrize("enable_metrics", [False, True])
def test_typed_config_carries_tokenizer_metrics(enable_metrics):
    extension = MagicMock()
    with (
        get_context().override_server_args(
            enable_metrics=enable_metrics,
            served_model_name="m",
            enable_priority_scheduling=True,
            extra_metric_labels={"cluster": "c"},
            bucket_inter_token_latency=[0.001, 0.01],
            generation_tokens_buckets=["custom", "10", "1"],
        ),
        get_parallel().override(attn_dp_rank=0, dp_rank=None),
        patch.dict("os.environ", {"PROMETHEUS_MULTIPROC_DIR": "/tmp/prom"}),
        patch.object(rust_extensions, "load_rust_extension", return_value=extension),
        patch.object(rust_config, "compute_num_reserved_tokens", return_value=0),
        patch.object(rust_config, "_launch_server_info", return_value="{}"),
    ):
        rust_config._build_server_args(
            _scheduler_for_typed_config(), default_max_new_tokens=None
        )

    metrics = extension.ServerArgs.call_args.kwargs["metrics"]
    if not enable_metrics:
        assert metrics is None
        return
    assert metrics is extension.MetricsConfig.return_value
    extension.MetricsConfig.assert_called_once_with(
        multiproc_dir="/tmp/prom",
        labels={
            "model_name": "m",
            "engine_type": "unified",
            "priority": "",
            "cluster": "c",
        },
        priority_label=True,
        bucket_time_to_first_token=None,
        bucket_inter_token_latency=[0.001, 0.01],
        bucket_e2e_request_latency=None,
        # The default rule keeps the Rust side's defaults; others resolve here.
        bucket_prompt_tokens=None,
        bucket_generation_tokens=[1.0, 10.0],
    )


@pytest.mark.parametrize(
    "tokenizer,expected",
    [
        (None, -1),
        (SimpleNamespace(), -1),
        (SimpleNamespace(model_max_length=4096), 4096),
        # HF's unset sentinel, int(1e30), is not a 64-bit integer.
        (SimpleNamespace(model_max_length=int(1e30)), 2048),
    ],
    ids=["no-tokenizer", "no-attribute", "configured", "unset-sentinel"],
)
def test_tokenizer_model_max_length_matches_python_tokenize(tokenizer, expected):
    assert rust_config._tokenizer_model_max_length(tokenizer, 2048) == expected


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


class _HttpRegistry(CommonKVBootstrapServer):
    def __init__(self, host, port):
        raise AssertionError("the Rust server serves this registry")


class _OwnBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host, port):
        self.address = (host, port)


@pytest.mark.parametrize(
    "server_class,bootstrap_port,expected",
    [
        (_HttpRegistry, 8998, 31000),
        (_HttpRegistry, 31000, 31000),
        (_HttpRegistry, 33000, ValueError),
        (_OwnBootstrapServer, 8998, 8998),
        (_OwnBootstrapServer, 33000, 33000),
    ],
    ids=[
        "http-registry-default",
        "http-registry-shared",
        "http-registry-separate",
        "backend-bootstrap-server-default",
        "backend-bootstrap-server-separate",
    ],
)
def test_rust_prefill_bootstrap_port_alias(server_class, bootstrap_port, expected):
    args = ServerArgs(
        model_path="dummy",
        port=31000,
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=bootstrap_port,
    )
    with (
        envs.SGLANG_RUST_SERVER.override(True),
        patch.object(disagg_service, "get_kv_class", return_value=server_class),
    ):
        if expected is ValueError:
            with pytest.raises(ValueError, match="conflicts with --port"):
                handle_pd_disaggregation(args)
            return
        handle_pd_disaggregation(args)
    assert resolving_view(args).disaggregation_bootstrap_port == expected


@pytest.mark.parametrize(
    "mode,server_class,starts",
    [
        ("prefill", _HttpRegistry, False),
        ("prefill", _OwnBootstrapServer, True),
        ("decode", _OwnBootstrapServer, False),
    ],
    ids=["http-registry", "backend-bootstrap-server", "decode"],
)
def test_rust_mode_starts_only_a_bootstrap_server_rust_cannot_serve(
    mode, server_class, starts
):
    with (
        get_context().override_server_args(
            host="127.0.0.1",
            disaggregation_mode=mode,
            disaggregation_transfer_backend="nixl",
            disaggregation_bootstrap_port=33000,
        ),
        patch.object(disagg_service, "get_kv_class", return_value=server_class),
    ):
        server = disagg_service.start_disagg_service_for_rust_server()

    if starts:
        assert server.address == ("127.0.0.1", 33000)
    else:
        assert server is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
