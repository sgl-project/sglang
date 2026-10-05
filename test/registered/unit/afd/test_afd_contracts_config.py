from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
import copy
import json
import multiprocessing
import sys
import types
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as config
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import ffn_server as afd_ffn_server
from sglang.srt.afd import integration as afd_integration
from sglang.srt.afd import profiles as profiles
from sglang.srt.afd import role_graph as afd_graph
from sglang.srt.afd import transport as afd_transport
from sglang.srt.afd.model_adapters import base as adapter
from sglang.srt.afd.model_adapters import qwen3_moe as qwen3
from sglang.test.afd.config_fixtures import _lane_server_args, _server_args


def _spawn_role_readback(result_queue, value):
    result_queue.put(
        config.execution_mode_from_server_args(
            SimpleNamespace(afd_execution_mode=value)
        ).value
    )


@pytest.mark.parametrize(
    "case", ["missing", "order", "wire-rank", "world-rank", "duplicate-lane", "width"]
)
def test_topology_negative_matrix_fails_before_startup(case):
    from msgspec.structs import replace

    lanes = 2 if case == "duplicate-lane" else 1
    endpoints = list(contracts.AFDPairedTopology.paired(lanes=lanes).endpoints)
    if case == "missing":
        endpoints = endpoints[1:]
    elif case == "order":
        endpoints.reverse()
    elif case == "wire-rank":
        endpoints = [replace(e, transport_rank=1 - e.transport_rank) for e in endpoints]
    elif case == "world-rank":
        endpoints[1] = replace(endpoints[1], coordination_rank=0)
    elif case == "duplicate-lane":
        endpoints[1] = replace(endpoints[1], ordinal=0)
    else:
        endpoints = contracts.AFDPairedTopology.paired(lanes=2).endpoints
    topology = contracts.AFDPairedTopology(endpoints=tuple(endpoints), lanes=lanes)
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_PAIRED_LAYOUT_REQUIRED"):
        topology.validate()


@pytest.mark.parametrize("lanes", [0, -1, True, 1.0, "2"])
def test_topology_rejects_non_positive_lane_counts(lanes):
    """A lane count that is not a positive int must fail at construction."""

    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_LANE_COUNT_INVALID"):
        contracts.AFDPairedTopology.paired(lanes=lanes)


@pytest.mark.parametrize(
    "ffn_lanes,attention_lanes",
    [(1, 1), (2, 2), (4, 4), (8, 8), (4, 8), (4, 12), (4, 16), (2, 8), (1, 4)],
)
def test_lane_group_fanin_routes_k_attention_lanes_into_one_ffn_rank(
    ffn_lanes, attention_lanes
):
    """M = k*N must give every attention lane one peer and every FFN rank k."""

    group = attention_lanes // ffn_lanes
    topology = contracts.AFDPairedTopology.paired(
        lanes=ffn_lanes,
        attention_lanes=attention_lanes,
    )
    topology.validate()
    assert (topology.ffn_size, topology.attention_size) == (ffn_lanes, attention_lanes)
    assert topology.expects_dp_attention(role=contracts.AFDRole.FFN) is False
    assert topology.expects_dp_attention(role=contracts.AFDRole.ATTENTION) is (
        attention_lanes > 1
    )
    assert topology.lanes_per_ffn == group
    assert topology.coordination_world_size == ffn_lanes + attention_lanes
    # One wire group per FFN rank: that rank plus the k lanes it serves.
    assert topology.pair_world_size == 1 + group
    assert topology.expected_parallelism(role=contracts.AFDRole.ATTENTION) == (
        attention_lanes,
        attention_lanes,
        1,
        1,
    )
    assert topology.expected_parallelism(role=contracts.AFDRole.FFN) == (
        ffn_lanes,
        1,
        ffn_lanes,
        1,
    )
    for ordinal in range(attention_lanes):
        endpoint = topology.local(role=contracts.AFDRole.ATTENTION, ordinal=ordinal)
        assert endpoint.transport_rank == 1 + ordinal % group
        assert endpoint.coordination_rank == ffn_lanes + ordinal
        assert (
            topology.group_ordinal(role=contracts.AFDRole.ATTENTION, ordinal=ordinal)
            == ordinal // group
        )
        assert topology.peers(role=contracts.AFDRole.ATTENTION, ordinal=ordinal) == (
            topology.local(role=contracts.AFDRole.FFN, ordinal=ordinal // group),
        )
    for ordinal in range(ffn_lanes):
        endpoint = topology.local(role=contracts.AFDRole.FFN, ordinal=ordinal)
        assert (endpoint.transport_rank, endpoint.coordination_rank) == (0, ordinal)
        assert (
            topology.group_ordinal(role=contracts.AFDRole.FFN, ordinal=ordinal)
            == ordinal
        )
        assert topology.attention_lane_group(ffn_ordinal=ordinal) == tuple(
            range(ordinal * group, (ordinal + 1) * group)
        )
        peers = topology.peers(role=contracts.AFDRole.FFN, ordinal=ordinal)
        assert peers == tuple(
            topology.local(role=contracts.AFDRole.ATTENTION, ordinal=lane)
            for lane in topology.attention_lane_group(ffn_ordinal=ordinal)
        )
        # Distinct wire ranks inside the group are what let one FFN rank address
        # k lanes without k separate rendezvous.
        assert tuple(peer.transport_rank for peer in peers) == tuple(
            range(1, group + 1)
        )
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_ROLE_IDENTITY_INVALID"):
        topology.local(role=contracts.AFDRole.FFN, ordinal=ffn_lanes)
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_ROLE_IDENTITY_INVALID"):
        topology.local(role=contracts.AFDRole.ATTENTION, ordinal=attention_lanes)


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_stating_the_symmetric_lane_count_changes_nothing(lanes):
    """k == 1 must be the same topology however it was spelled.

    The mAnF work is only allowed to add a shape, not move the symmetric one, and
    the endpoint tuple is what every rank, port and buffer key is derived from.
    """

    implicit = contracts.AFDPairedTopology.paired(lanes=lanes)
    explicit = contracts.AFDPairedTopology.paired(lanes=lanes, attention_lanes=lanes)
    for topology in (implicit, explicit):
        topology.validate()
        assert topology.lanes_per_ffn == 1
        assert topology.coordination_world_size == 2 * lanes
    assert implicit.endpoints == explicit.endpoints
    for role in (contracts.AFDRole.ATTENTION, contracts.AFDRole.FFN):
        assert implicit.expected_parallelism(
            role=role
        ) == explicit.expected_parallelism(role=role)
        assert implicit.expects_dp_attention(
            role=role
        ) == explicit.expects_dp_attention(role=role)
        for ordinal in range(lanes):
            assert implicit.peers(role=role, ordinal=ordinal) == explicit.peers(
                role=role, ordinal=ordinal
            )


@pytest.mark.parametrize("attention_lanes", [0, -4, True, 4.0, "8"])
def test_topology_rejects_invalid_attention_counts(
    attention_lanes,
):
    """Balanced ingress admits nonintegral ratios, never invalid rank counts."""

    with pytest.raises(
        contracts.AFDError, match="AFD_TOPOLOGY_LANE_GROUP_RATIO_INVALID"
    ):
        contracts.AFDPairedTopology.paired(lanes=4, attention_lanes=attention_lanes)


def test_a_hand_built_partial_lane_group_still_fails_validate():
    """`paired` is not the only constructor, so validate must repeat the check."""

    topology = contracts.AFDPairedTopology(
        endpoints=contracts.AFDPairedTopology.paired(
            lanes=2, attention_lanes=4
        ).endpoints,
        lanes=2,
        attention_lanes=3,
    )
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_PAIRED_LAYOUT_REQUIRED"):
        topology.validate()


@pytest.mark.parametrize(
    "batch_size,stages,expected",
    [
        (8, 2, ((0, 4, 0, 4), (4, 8, 4, 8))),
        (5, 2, ((0, 3, 0, 4), (3, 5, 4, 10))),
    ],
)
def test_s2_request_and_token_ranges_are_contiguous(batch_size, stages, expected):
    """Changing request partition math must preserve request/token identity and order."""

    token_lengths = (1,) * batch_size if batch_size == 8 else (1, 2, 1, 2, 4)
    actual = adapter._stage_ranges(
        batch_size=batch_size,
        token_lengths=token_lengths,
        stages=stages,
    )
    assert actual == expected
    assert tuple(item[0] for item in actual) == (0,) + tuple(
        item[1] for item in actual[:-1]
    )
    assert tuple(item[2] for item in actual) == (0,) + tuple(
        item[3] for item in actual[:-1]
    )


@pytest.mark.parametrize(
    "stages,expected",
    [
        (2, ((0, 1, 0, 8), (1, 1, 8, 8))),
    ],
)
def test_single_request_stage_ranges_keep_empty_tail_stages(stages, expected):
    assert (
        adapter._stage_ranges(
            batch_size=1,
            token_lengths=(8,),
            stages=stages,
        )
        == expected
    )


def _join_stages(rows, residuals):
    stages = [
        adapter.AFDStage(
            index=index,
            request_start=0,
            request_stop=0,
            token_start=0,
            token_stop=count,
            hidden_states=torch.arange(count * 2).reshape(count, 2) + index * 100,
            residual=None
            if residual is None
            else torch.tensor(residual, dtype=torch.int64).reshape(-1, 2),
            positions=None,
            forward_batch=None,
        )
        for index, (count, residual) in enumerate(zip(rows, residuals))
    ]
    return object.__new__(qwen3.Qwen3AFDAdapter).join_step(stages=stages)


@pytest.mark.parametrize(
    "rows,residuals,expected_hidden,expected_residual",
    [
        ((2, 0), ([10, 11, 12, 13], None), [0, 1, 2, 3], [10, 11, 12, 13]),
        ((2, 0), ([10, 11, 12, 13], []), [0, 1, 2, 3], [10, 11, 12, 13]),
        ((0, 1), (None, [20, 21]), [100, 101], [20, 21]),
        ((0, 1), ([], [20, 21]), [100, 101], [20, 21]),
        (
            (2, 1),
            ([10, 11, 12, 13], [20, 21]),
            [0, 1, 2, 3, 100, 101],
            [10, 11, 12, 13, 20, 21],
        ),
        ((2, 0), (None, []), [0, 1, 2, 3], None),
        ((0, 0), (None, None), [], None),
        ((0, 0), ([], []), [], []),
    ],
)
def test_join_s2_preserves_values_and_empty_stage_residuals(
    rows, residuals, expected_hidden, expected_residual
):
    hidden, residual = _join_stages(rows, residuals)
    torch.testing.assert_close(
        hidden, torch.tensor(expected_hidden, dtype=torch.int64).reshape(-1, 2)
    )
    if expected_residual is None:
        assert residual is None
    else:
        torch.testing.assert_close(
            residual, torch.tensor(expected_residual, dtype=torch.int64).reshape(-1, 2)
        )


@pytest.mark.parametrize(
    "rows,residuals,code",
    [
        ((1, 1), ([10, 11], None), "IDENTITY_DRIFT"),
        ((0, 0), ([], None), "IDENTITY_DRIFT"),
        ((2, 0), ([10, 11], None), "SHAPE_DRIFT"),
        ((2, 0), ([10, 11, 12, 13], [20, 21]), "SHAPE_DRIFT"),
    ],
)
def test_join_s2_rejects_residual_identity_or_shape_drift(rows, residuals, code):
    with pytest.raises(contracts.AFDError, match="AFD_STAGE_RESIDUAL_" + code):
        _join_stages(rows, residuals)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"tp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"dp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"ep_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"pp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"nnodes": 2}, "AFD_TOPOLOGY_ROLE_NODE_SPLIT_UNSUPPORTED"),
        ({"enable_two_batch_overlap": True}, "AFD_TBO_UNSUPPORTED"),
        (
            {"enable_single_batch_overlap": True},
            "AFD_SINGLE_BATCH_OVERLAP_UNSUPPORTED",
        ),
        ({"moe_a2a_backend": "deepep"}, "AFD_INTERNAL_COLLECTIVE_UNSUPPORTED"),
        ({"speculative_algorithm": "EAGLE"}, "AFD_MTP_SPECULATIVE_UNSUPPORTED"),
        ({"enable_dp_attention": True}, "AFD_TOPOLOGY_DP_ATTENTION_MISMATCH"),
    ],
)
def test_startup_exclusions_fail_closed(overrides, code):
    """An excluded execution family must fail at argument validation, not mid-request."""

    with pytest.raises(contracts.AFDError, match=code):
        config.validate_afd_server_args(_server_args(**overrides))


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_cpu_overlap_schedule_is_admitted(role):
    """Overlap is a supported configuration, not an excluded family.

    Asserted positively because lifting the old refusal removed the only
    reference to this behaviour, so a regression that restored the raise would
    otherwise leave the suite green.
    """

    config.validate_afd_server_args(
        _server_args(afd_execution_mode=role, disable_overlap_schedule=False)
    )


def _role_args(role, lanes=1, attention_lanes=None, **overrides):
    width = lanes if role == "ffn" else (attention_lanes or lanes)
    values = dict(
        tp_size=width,
        dp_size=width if role == "attention" else 1,
        ep_size=lanes if role == "ffn" else 1,
        enable_dp_attention=role == "attention" and width > 1,
    )
    values.update(overrides)
    return _lane_server_args(
        role=role, lanes=lanes, attention_lanes=attention_lanes, **values
    )


@pytest.mark.parametrize(
    "lanes,attention_lanes", [(2, None), (4, None), (2, 4), (4, 8), (4, 16)]
)
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_startup_parallelism_matches_each_roles_width(role, lanes, attention_lanes):
    args = _role_args(role, lanes, attention_lanes)
    config.validate_afd_server_args(args)
    args.enable_dp_attention = not args.enable_dp_attention
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_DP_ATTENTION_MISMATCH"):
        config.validate_afd_server_args(args)
    args.enable_dp_attention = not args.enable_dp_attention
    if attention_lanes is not None:
        args.tp_size = lanes if role == "attention" else attention_lanes
        if role == "attention":
            args.dp_size = lanes
        else:
            args.ep_size = attention_lanes
    elif role == "attention":
        args.tp_size = args.dp_size = 1
    else:
        args.dp_size, args.ep_size = lanes, 1
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_PARALLELISM_MISMATCH"):
        config.validate_afd_server_args(args)


def test_lane_count_outside_admitted_range_fails_config_validation():
    """The config gate bounds lanes; the per-profile lane list stays authoritative."""

    for lanes in (0, 33):
        with pytest.raises(contracts.AFDError, match="AFD_GRAPH_LANE_COUNT_INVALID"):
            config.AFDConfig(lanes=lanes).validate()
    # Three lanes clear the cheap bound but no profile admits them.
    config.AFDConfig(lanes=3).validate()
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_LANE_COUNT_UNSUPPORTED"):
        profiles.QWEN3_PAIRED_C1.validate_shape(config=config.AFDConfig(lanes=3))


@pytest.mark.parametrize("value", [0, -1, True, 5.0, "7"])
def test_attention_lane_count_requires_a_positive_integer(value):
    with pytest.raises(
        contracts.AFDError, match="AFD_GRAPH_ATTENTION_LANE_COUNT_INVALID"
    ):
        config.AFDConfig(lanes=4, attention_lanes=value).validate()


@pytest.mark.parametrize("a,f", [(1, 1), (1, 4), (4, 1), (5, 4), (7, 4), (36, 4)])
@pytest.mark.parametrize("backend", ["fa4", "nsa"])
def test_model_profiles_admit_independent_a_counts_without_an_enum(a, f, backend):
    cfg = config.AFDConfig(lanes=f, attention_lanes=a, attention_backend=backend)
    profile = profiles.QWEN3_PAIRED_C1 if backend == "fa4" else profiles.GLM5_PAIRED_C1
    profile.validate_shape(config=cfg)
    assert profile.contract()["attention_lane_policy"] == "positive-integer"
    for role in ("attention", "ffn"):
        args = _role_args(role, lanes=f, attention_lanes=a)
        args.afd_config, args.attention_backend = cfg, backend
        args.enable_dp_lm_head = role == "attention" and a > 1
        config.validate_afd_server_args(args)
        # The single A path retains native non-DP attention semantics.
        assert args.enable_dp_attention == (role == "attention" and a > 1)


@pytest.mark.parametrize(
    "role,lanes,attention_lanes,nnodes,admitted",
    [
        ("attention", 4, 8, 2, True),
        ("attention", 4, None, 2, True),
        ("attention", 4, 16, 4, True),
        ("ffn", 4, 8, 2, True),
        ("attention", 1, None, 2, False),
        ("ffn", 4, 8, 3, False),
        ("attention", 4, 8, 5, False),
        ("attention", 4, 8, 0, False),
    ],
)
def test_a_role_may_span_hosts_only_when_its_own_rank_count_divides(
    role, lanes, attention_lanes, nnodes, admitted
):
    """The node gate is per role now: 8 attention ranks split over 2 hosts, 4 FFN do not."""

    server_args = _role_args(role, lanes, attention_lanes, nnodes=nnodes)
    if admitted:
        config.validate_afd_server_args(server_args)
        return
    with pytest.raises(
        contracts.AFDError,
        match="AFD_TOPOLOGY_ROLE_NODE_SPLIT_UNSUPPORTED",
    ):
        config.validate_afd_server_args(server_args)


def test_ffn_startup_gate_precedes_torch_distributed_and_model_load(monkeypatch):
    """An invalid child topology must fail before any FFN CUDA side effect."""

    calls = []
    monkeypatch.setattr(
        afd_ffn_server,
        "_init_distributed",
        lambda **kwargs: calls.append(("distributed", kwargs)),
    )
    monkeypatch.setattr(
        afd_ffn_server,
        "_load_model",
        lambda **kwargs: calls.append(("model", kwargs)),
    )
    server_args = _server_args(
        afd_execution_mode="ffn",
        tp_size=2,
    )
    server_args.check_server_args = lambda: config.validate_afd_server_args(server_args)
    with pytest.raises(
        contracts.AFDError,
        match="AFD_TOPOLOGY_PARALLELISM_MISMATCH",
    ):
        afd_ffn_server.launch_server(server_args)
    assert calls == []
    from sglang import launch_server

    monkeypatch.setattr(launch_server, "resolving_view", lambda args: args)
    server_args.smg_grpc_mode = server_args.grpc_mode = False
    server_args.resolve_once = lambda: None
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_PARALLELISM_MISMATCH"):
        launch_server.run_server(server_args)
    assert calls == []


def test_ffn_root_runs_general_checks_and_requires_resolved_ffn(monkeypatch):
    """Common gates and resolved role must fail before FFN CUDA/model work."""

    calls = []
    monkeypatch.setattr(
        afd_ffn_server,
        "_init_distributed",
        lambda **kwargs: calls.append(("distributed", kwargs)),
    )
    monkeypatch.setattr(
        afd_ffn_server,
        "_load_model",
        lambda **kwargs: calls.append(("model", kwargs)),
    )
    invalid_gpu = _server_args(afd_execution_mode="ffn")
    invalid_gpu.check_server_args = lambda: (_ for _ in ()).throw(
        AssertionError("base_gpu_id must be non-negative")
    )
    with pytest.raises(AssertionError, match="base_gpu_id"):
        afd_ffn_server.launch_server(invalid_gpu)
    wrong_role = _server_args(afd_execution_mode="attention")
    wrong_role.check_server_args = lambda: None
    with pytest.raises(contracts.AFDError, match="FFN_EXECUTION_MODE_REQUIRED"):
        afd_ffn_server.launch_server(wrong_role)
    assert calls == []


def _ffn_lane_harness(monkeypatch, *, lanes, tp_rank=None):
    from sglang.srt.entrypoints import engine

    monkeypatch.setattr(engine, "_set_envs_and_config", lambda args: None)
    calls = []
    monkeypatch.setattr(
        afd_ffn_server,
        "_init_distributed",
        lambda **kwargs: calls.append(("distributed", kwargs)),
    )
    monkeypatch.setattr(
        afd_ffn_server,
        "_load_model",
        lambda **kwargs: (calls.append(("model", kwargs)), ("model", "bfloat16"))[1],
    )
    monkeypatch.setattr(
        afd_ffn_server,
        "build_ffn_pipeline",
        lambda **kwargs: (
            calls.append(("pipeline", kwargs))
            or SimpleNamespace(run_once=lambda: False, close=lambda: None)
        ),
    )
    torch = types.ModuleType("torch")
    torch.device = lambda kind, index: (kind, index)
    monkeypatch.setitem(sys.modules, "torch", torch)
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_group=SimpleNamespace(rank_in_group=tp_rank)),
    )
    monkeypatch.setattr(
        runtime_context,
        "publish",
        lambda value, *, role, ranks: calls.append(("publish", role, ranks)),
    )
    overrides = types.ModuleType("sglang.srt.arg_groups.overrides")
    overrides.resolving_view = lambda value: value
    monkeypatch.setitem(sys.modules, overrides.__name__, overrides)
    utils = types.ModuleType("sglang.srt.utils")
    utils.configure_logger = lambda args, prefix="": calls.append(("logger", prefix))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)
    server_args = _server_args(
        afd_execution_mode="ffn",
        afd_config=config.AFDConfig(lanes=lanes),
        tp_size=lanes,
        ep_size=lanes,
        base_gpu_id=4,
        nccl_port=None,
        load_format="auto",
        download_dir=None,
    )
    server_args.check_server_args = lambda: None
    return server_args, calls


def test_single_lane_ffn_still_serves_in_process(monkeypatch):
    """The measured 1A1F path must not acquire a process fork."""

    server_args, calls = _ffn_lane_harness(monkeypatch, lanes=1, tp_rank=0)
    monkeypatch.setattr(
        afd_ffn_server,
        "_run_lanes",
        lambda *args, **kwargs: pytest.fail("single lane must not spawn"),
    )
    afd_ffn_server.launch_server(server_args)
    kinds = [item[0] for item in calls]
    # Logging first: the pipeline built later is what emits the graph telemetry.
    assert kinds == ["logger", "publish", "distributed", "model", "pipeline"]
    assert calls[1][1] == "scheduler"
    assert calls[1][2].world_rank == 0 and calls[1][2].gpu_id == 4
    assert calls[0][1] == " FFN0"
    assert calls[2][1]["lane"] == 0
    assert calls[2][1]["device"] == ("cuda", 4)
    assert calls[4][1]["lane"] == 0


def test_multi_lane_ffn_spawns_one_worker_per_lane(monkeypatch):
    """Each FFN lane is its own process, owning one rank of the FFN group."""

    server_args, _ = _ffn_lane_harness(monkeypatch, lanes=4)
    started = []

    class Worker:
        def __init__(self, *, target, args):
            self.target = target
            self.args = args
            self.sentinel = len(started)
            self.name = f"worker-{self.sentinel}"
            self.exitcode = 0
            self.pid = None
            started.append(self)

        def start(self):
            self.started = True
            self.pid = self.sentinel + 100

        def join(self, timeout=None):
            pass

        def terminate(self):
            pytest.fail("a clean exit must not terminate peers")

    multiprocessing = types.ModuleType("multiprocessing")
    multiprocessing.get_context = lambda kind: SimpleNamespace(
        Process=lambda target, args: Worker(target=target, args=args)
    )
    connection = types.ModuleType("multiprocessing.connection")
    connection.wait = lambda pending, timeout=None: list(pending)
    monkeypatch.setitem(sys.modules, "multiprocessing", multiprocessing)
    monkeypatch.setitem(sys.modules, "multiprocessing.connection", connection)

    afd_ffn_server.launch_server(server_args)
    assert [worker.args[1] for worker in started] == [0, 1, 2, 3]
    assert all(worker.target is afd_ffn_server.run_lane for worker in started)
    assert all(worker.started for worker in started)


@pytest.mark.parametrize("lane", [0, 3])
@pytest.mark.parametrize("nnodes,addr", [(1, None), (2, "10.0.0.1:4405"), (2, None)])
@pytest.mark.parametrize("dist_timeout", [None, 91])
def test_ffn_lane_joins_a_lane_sized_tp_and_ep_group(
    monkeypatch, lane, nnodes, addr, dist_timeout
):
    import inspect

    from sglang.srt.distributed.parallel_state import initialize_model_parallel
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    parallel_signature = inspect.signature(initialize_model_parallel)
    dp_signature = inspect.signature(initialize_dp_attention)
    recorded = {}
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(set_device=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    model_config = types.ModuleType("sglang.srt.configs.model_config")
    model_config.ModelConfig = SimpleNamespace(from_server_args=lambda args: args)
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", model_config)
    distributed = types.ModuleType("sglang.srt.distributed")

    def init_environment(**kwargs):
        recorded["environment"] = kwargs

    def init_parallel(**kwargs):
        parallel_signature.bind(**kwargs)
        recorded["parallel"] = kwargs
        recorded["flags_before_groups"] = "flags" in recorded
        assert "moe_config" in recorded

    distributed.init_distributed_environment = init_environment
    distributed.initialize_model_parallel = init_parallel
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    bootstrap = types.ModuleType("sglang.srt.distributed.bootstrap")
    bootstrap._set_all_reduce_flags = lambda **kwargs: recorded.setdefault(
        "flags", kwargs
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.bootstrap", bootstrap)
    dp_attention = types.ModuleType("sglang.srt.layers.dp_attention")

    def init_dp(**kwargs):
        dp_signature.bind(**kwargs)
        recorded["dp"] = kwargs

    dp_attention.initialize_dp_attention = init_dp
    dp_attention.init_dp_gathered_buffer = lambda model: recorded.setdefault(
        "buffer", model
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.dp_attention", dp_attention)
    moe = types.ModuleType("sglang.srt.layers.moe")
    moe.initialize_moe_config = lambda: recorded.setdefault("moe_config", True)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", moe)

    server_args = _server_args(
        afd_execution_mode="ffn",
        ep_size=4,
        afd_config=config.AFDConfig(
            lanes=4,
            rendezvous_port=4400,
        ),
        nccl_port=None,
        nnodes=nnodes,
        node_rank=0,
        dist_init_addr=addr,
        dist_timeout=dist_timeout,
    )
    if nnodes > 1 and addr is None:
        with pytest.raises(contracts.AFDError, match="AFD_FFN_DIST_INIT_ADDR_REQUIRED"):
            afd_ffn_server._init_distributed(
                server_args=server_args,
                device=SimpleNamespace(index=4 + lane),
                lane=lane,
            )
        assert "environment" not in recorded
        return
    afd_ffn_server._init_distributed(
        server_args=server_args,
        device=SimpleNamespace(index=4 + lane),
        lane=lane,
    )
    assert recorded["environment"]["timeout"] == dist_timeout
    assert recorded["environment"]["world_size"] == 4
    assert recorded["environment"]["rank"] == lane
    assert recorded["environment"]["local_rank"] == 4 + lane
    # One port above the coordination world and its four per-lane pairs.
    assert recorded["environment"]["distributed_init_method"] == (
        f"tcp://{addr}" if nnodes > 1 else "tcp://127.0.0.1:4405"
    )
    # Widths come from the published context; removed kwargs must not return.
    assert recorded["parallel"] == {}
    assert recorded["dp"] == {"server_args": server_args}
    assert recorded["buffer"] is server_args
    # This role bypasses init_torch_distributed, so nothing else would apply
    # --disable-custom-all-reduce and friends to its group.
    assert recorded["flags"] == {}
    assert recorded["flags_before_groups"] is True
    assert recorded["moe_config"] is True


def test_ffn_lane_rank_disagreement_fails_closed(monkeypatch):
    """A lane merging rows at another lane's offset must not start."""

    server_args, _ = _ffn_lane_harness(monkeypatch, lanes=4, tp_rank=1)
    with pytest.raises(contracts.AFDError, match="AFD_FFN_LANE_RANK_MISMATCH"):
        afd_ffn_server.run_lane(server_args, 2)


def test_non_off_raw_envelope_is_checked_in_early_config_handler():
    """Reject incomplete AFD config before touching even a nonexistent model."""
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="/nonexistent/afd-test-model", afd_execution_mode="attention"
    )
    with pytest.raises(contracts.AFDError, match="AFD_GRAPH_CONFIG_REQUIRED"):
        args.resolve_once()
    native = ServerArgs(model_path="dummy")
    native.resolve_once()
    assert native.afd_execution_mode == "off"
    assert native.afd_config is None


def test_default_graph_contract_is_bounded():
    """Validate supported defaults and resource-control bounds."""

    value = config.AFDConfig()
    assert value.stages == 2
    assert value.attention_backend == "fa3"
    assert value.close_timeout_seconds == 30
    assert value.nccl_num_channels == 8
    with pytest.raises(contracts.AFDError, match="CLOSE_TIMEOUT_INVALID"):
        config.AFDConfig(close_timeout_seconds=0).validate()
    with pytest.raises(contracts.AFDError, match="FA_BACKEND_INVALID"):
        config.AFDConfig(attention_backend="flashinfer").validate()
    with pytest.raises(contracts.AFDError, match="NCCL_CHANNELS_INVALID"):
        config.AFDConfig(nccl_num_channels=0).validate()
    with pytest.raises(contracts.AFDError, match="NCCL_CHANNELS_INVALID"):
        config.AFDConfig(nccl_num_channels=33).validate()


@pytest.mark.parametrize("stages", [0, 1, 3, 4])
def test_out_of_range_stage_graphs_fail_closed(stages):
    with pytest.raises(contracts.AFDError, match="STAGE_COUNT_UNSUPPORTED"):
        config.AFDConfig(stages=stages).validate()


def test_profiles_use_native_two_microbatch_execution():
    for profile in (profiles.QWEN3_PAIRED_C1, profiles.GLM5_PAIRED_C1):
        assert profile.stages == (2,)
    assert config.AFDConfig().stages == 2


def _qwen_model(*, hidden_size=2048, num_hidden_layers=48):
    model_type = type("Qwen3MoeForCausalLM", (), {})
    model = model_type()
    model.model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_experts=128,
            num_experts_per_tok=8,
        ),
        layers=[object()] * num_hidden_layers,
        start_layer=0,
        end_layer=num_hidden_layers,
    )
    return model


def test_model_adapter_accepts_only_admitted_qwen3_moe_identities(monkeypatch):
    """Validated Qwen3-MoE shapes pass; a nearby size fails before any allocation."""

    base = sys.modules[qwen3.Qwen3AFDAdapter.__mro__[1].__module__]
    cfg = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, enable_prefill_cp=False)
    monkeypatch.setattr(base, "get_parallel", lambda: cfg)
    for layers, hidden in ((48, 2048), (94, 4096)):
        qwen3.Qwen3AFDAdapter(
            role=contracts.AFDRole.FFN,
            model=_qwen_model(num_hidden_layers=layers, hidden_size=hidden),
            attention_backend=None,
        ).validate_model()
    with pytest.raises(contracts.AFDError, match="QWEN3_MOE_REQUIRED"):
        qwen3.Qwen3AFDAdapter(
            role=contracts.AFDRole.FFN,
            model=_qwen_model(hidden_size=4096),
            attention_backend=None,
        ).validate_model()


@pytest.mark.parametrize(
    "field,value",
    [("attn_cp_size", 2), ("attn_dcp_size", 2), ("enable_prefill_cp", True)],
)
def test_qwen_rejects_actual_runtime_context_parallel(monkeypatch, field, value):
    base = sys.modules[qwen3.Qwen3AFDAdapter.__mro__[1].__module__]
    cfg = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, enable_prefill_cp=False)
    setattr(cfg, field, value)
    monkeypatch.setattr(base, "get_parallel", lambda: cfg)
    with pytest.raises(contracts.AFDError, match="AFD_CONTEXT_PARALLEL_UNSUPPORTED"):
        qwen3.Qwen3AFDAdapter(
            role=contracts.AFDRole.FFN, model=_qwen_model(), attention_backend=None
        ).validate_model()


def _qwen_adapter(role, backend_name="fa3"):
    backend = type(
        "FlashAttentionBackend",
        (),
        {
            "__module__": "sglang.srt.layers.attention.flashattention_backend",
            "prefill_attention_backend_str": backend_name,
            "decode_attention_backend_str": backend_name,
        },
    )()
    return qwen3.Qwen3AFDAdapter(
        role=role,
        model=_qwen_model(),
        attention_backend=backend if role == contracts.AFDRole.ATTENTION else None,
    )


def _model_descriptor(*, adapter, config):
    return afd_integration._model_descriptor(
        capture_sizes=(8, 16, 32),
        role=adapter.role,
        adapter=adapter,
        profile=profiles.QWEN3_PAIRED_C1,
        config=config,
        dtype="torch.bfloat16",
    )


def test_attention_adapter_rejects_non_fa3_fa4_before_transport():
    """Unsupported attention metadata must fail before connector construction."""

    with pytest.raises(
        contracts.AFDError,
        match="AFD_FA_METADATA_BACKEND_UNSUPPORTED",
    ):
        _qwen_adapter(
            contracts.AFDRole.ATTENTION, backend_name="flashinfer"
        ).validate_model()


def test_peer_runtime_capability_drift_fails_before_nccl_construction():
    """A validly re-digested peer with a wider cache contract is still rejected."""

    attention_adapter = _qwen_adapter(contracts.AFDRole.ATTENTION)
    ffn_adapter = _qwen_adapter(contracts.AFDRole.FFN)
    local_config = config.AFDConfig()
    attention_descriptor = _model_descriptor(
        adapter=attention_adapter,
        config=local_config,
    )
    ffn_descriptor = _model_descriptor(
        adapter=ffn_adapter,
        config=local_config,
    )
    transport = object.__new__(afd_transport.AFDPairedP2PTransport)
    transport._topology = contracts.AFDPairedTopology.paired()
    transport._validate_descriptors(
        descriptors=[attention_descriptor, ffn_descriptor],
        local_descriptor=attention_descriptor,
    )

    drifted = copy.deepcopy(ffn_descriptor)
    drifted["runtime_contract"]["capture_stage_sizes"] = (8, 32)
    drifted["runtime_contract_digest"] = contracts.contract_digest(
        drifted["runtime_contract"]
    )
    drifted_body = {
        key: value for key, value in drifted.items() if key != "capability_digest"
    }
    drifted["capability_digest"] = contracts.contract_digest(drifted_body)
    with pytest.raises(
        contracts.AFDError,
        match="STARTUP_IDENTITY_MISMATCH",
    ):
        transport._validate_descriptors(
            descriptors=[attention_descriptor, drifted],
            local_descriptor=attention_descriptor,
        )


def test_runtime_fa_backend_must_match_configured_peer_capability():
    """The JSON backend is the only truth accepted by the runtime digest."""

    attention_adapter = _qwen_adapter(contracts.AFDRole.ATTENTION)
    with pytest.raises(
        contracts.AFDError,
        match="ATTENTION_BACKEND_CONFIG_RUNTIME_DRIFT",
    ):
        _model_descriptor(
            adapter=attention_adapter,
            config=config.AFDConfig(attention_backend="fa4"),
        )
    descriptor = _model_descriptor(
        adapter=attention_adapter,
        config=config.AFDConfig(),
    )
    runtime = descriptor["runtime_contract"]
    assert "base_identity" not in runtime
    assert runtime["capability_profile"] == profiles.QWEN3_PAIRED_C1.contract()
    assert runtime["capability_profile_digest"] == profiles.QWEN3_PAIRED_C1.digest
    assert runtime["afd_abi"]["revision"] == "afd-c1-abi-native-s2-startup-capture-r7"
    assert len(runtime["afd_abi"]["contract_digest"]) == 64


def test_spawn_child_resolves_role_from_process_local_server_args():
    """A spawned model worker must not inherit or require the parent's mode global."""

    assert (
        config.execution_mode_from_server_args(
            SimpleNamespace(afd_execution_mode="attention")
        )
        == config.AFDExecutionMode.ATTENTION
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_role_readback,
        args=(result_queue, "ffn"),
    )
    try:
        process.start()
        # Spawn imports the real runtime again; cold Linux CI imports can exceed
        # five seconds even when role resolution itself completes immediately.
        process.join(timeout=30)
        assert process.exitcode == 0
        assert result_queue.get(timeout=1) == "ffn"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def test_capability_registry_holds_one_profile_per_family_and_rejects_unknown():
    assert tuple(profiles.AFD_PROFILE_REGISTRY) == (
        profiles.QWEN3_PAIRED_C1_ID,
        profiles.GLM5_PAIRED_C1_ID,
    )
    assert profiles.registered_capability_profiles() == (
        profiles.QWEN3_PAIRED_C1,
        profiles.GLM5_PAIRED_C1,
    )
    assert profiles.QWEN3_PAIRED_C1.digest != profiles.GLM5_PAIRED_C1.digest
    assert profiles.QWEN3_PAIRED_C1.model_family == "qwen3_moe"
    assert profiles.GLM5_PAIRED_C1.model_family == "glm_moe_dsa"
    assert (
        profiles.GLM5_PAIRED_C1.metadata_contract
        is contracts.MetadataContract.PRIVATE_DSA
    )
    assert profiles.QWEN3_PAIRED_C1.fa_backends == ("fa3", "fa4")
    assert profiles.GLM5_PAIRED_C1.fa_backends == ("nsa",)
    server_args = _server_args()
    assert profiles.validate_startup_capabilities(
        server_args=server_args,
        config=config.AFDConfig(),
    ) == (profiles.QWEN3_PAIRED_C1,)
    assert profiles.validate_startup_capabilities(
        server_args=server_args,
        config=config.AFDConfig(attention_backend="nsa"),
    ) == (profiles.GLM5_PAIRED_C1,)
    with pytest.raises(
        contracts.AFDError,
        match="AFD_CAPABILITY_PROFILE_STARTUP_UNADMITTED",
    ) as unadmitted:
        profiles.validate_startup_capabilities(
            server_args=_server_args(tp_size=2),
            config=config.AFDConfig(),
        )
    assert profiles.QWEN3_PAIRED_C1_ID in unadmitted.value.detail
    assert profiles.GLM5_PAIRED_C1_ID in unadmitted.value.detail
    with pytest.raises(FrozenInstanceError):
        profiles.QWEN3_PAIRED_C1.identity = "mutated"
    with pytest.raises(TypeError):
        profiles.AFD_PROFILE_REGISTRY["unknown"] = object()
    with pytest.raises(
        contracts.AFDError,
        match="AFD_CAPABILITY_PROFILE_UNRESOLVED",
    ):
        profiles.resolve_capability_factories(
            model=SimpleNamespace(),
            config=config.AFDConfig(),
        )


def test_registered_factories_compose_exact_c1_types_and_identity(monkeypatch):
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_observability",
        lambda: SimpleNamespace(decode_log_interval=17),
    )
    base = sys.modules[qwen3.Qwen3AFDAdapter.__mro__[1].__module__]
    cfg = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, enable_prefill_cp=False)
    monkeypatch.setattr(base, "get_parallel", lambda: cfg)
    factories = profiles.resolve_capability_factories(
        model=_qwen_model(),
        config=config.AFDConfig(),
    )
    topology = factories.make_topology(lanes=1)
    qwen_adapter = factories.make_adapter(
        role=contracts.AFDRole.FFN,
        model=_qwen_model(),
        attention_backend=None,
    )
    assert type(topology) is contracts.AFDPairedTopology
    assert (topology.coordination_world_size, topology.pair_world_size) == (2, 2)
    assert tuple(endpoint.transport_rank for endpoint in topology.endpoints) == (
        0,
        1,
    )
    assert type(qwen_adapter) is qwen3.Qwen3AFDAdapter
    assert qwen_adapter.metadata_contract is contracts.MetadataContract.STANDARD_FA
    assert factories.transport_factory is afd_transport.AFDPairedP2PTransport
    graph = factories.make_graph_strategy(
        capture_sizes=(8, 16, 32),
        role=contracts.AFDRole.FFN,
        config=config.AFDConfig(),
        num_layers=48,
        base_hbm_bytes=0,
        device="cuda:0",
    )
    assert type(graph) is afd_graph.AFDRoleGraphService
    assert graph._log_interval == 17
    assert factories.profile.digest == profiles.QWEN3_PAIRED_C1.digest


def test_shape_merge_plan_uses_padded_rows_so_a_bucket_survives_real_drift():
    first, drifted, widened = [
        planned_shape(
            lane=1,
            lane_rows=rows,
            hidden_size=2048,
            dtype="bfloat16",
            config=config.AFDConfig(lanes=2),
        )
        for rows in [((17, 9), (40, 40)), ((21, 3), (33, 64)), ((65, 9), (40, 40))]
    ]
    assert first.stage_rows == (40, 40)
    assert first.lane_bucket_rows == ((64, 64), (64, 64))
    assert first.merge_plan(stage=0) == first.merge_plan(stage=1) == (64, 64)
    assert drifted.merge_plan(stage=0) == first.merge_plan(stage=0)
    assert drifted.digest == first.digest
    assert widened.merge_plan(stage=0) == (128, 128)
    assert widened.digest != first.digest


def _lane_plan(*, gathered, local_rows, lanes, lane=0, extend=False):
    return qwen3.Qwen3AFDAdapter(
        role=contracts.AFDRole.ATTENTION,
        model=_qwen_model(),
        attention_backend=None,
    ).lane_stage_rows(
        forward_batch=SimpleNamespace(
            is_extend_in_batch=extend,
            global_num_tokens_cpu=gathered,
        ),
        local_rows=local_rows,
        lanes=lanes,
        lane=lane,
    )


def test_lane_row_plan_reuses_the_dp_gathered_token_counts():
    """Multi-lane rows must cost no collective beyond what DP attention did."""

    assert _lane_plan(
        gathered=[9, 8, 4, 0],
        local_rows=(5, 4),
        lanes=4,
        lane=0,
    ) == ((5, 4), (4, 4), (2, 2), (0, 0))
    assert _lane_plan(
        gathered=[9, 8, 4, 0],
        local_rows=(2, 2),
        lanes=4,
        lane=2,
    ) == ((5, 4), (4, 4), (2, 2), (0, 0))
    # The single-lane path is the local split verbatim and never reads the batch.
    assert qwen3.Qwen3AFDAdapter(
        role=contracts.AFDRole.ATTENTION,
        model=_qwen_model(),
        attention_backend=None,
    ).lane_stage_rows(
        forward_batch=None,
        local_rows=(5, 4),
        lanes=1,
        lane=0,
    ) == ((5, 4),)


@pytest.mark.parametrize(
    "kwargs,code",
    [
        (
            {"gathered": None, "local_rows": (5, 4), "lanes": 2},
            "AFD_LANE_ROW_PLAN_UNAVAILABLE",
        ),
        (
            {"gathered": [9, 8, 4], "local_rows": (5, 4), "lanes": 2},
            "AFD_LANE_ROW_PLAN_UNAVAILABLE",
        ),
        (
            {"gathered": [9, 8], "local_rows": (4, 4), "lanes": 2},
            "AFD_LANE_ROW_PLAN_LOCAL_DISAGREES",
        ),
    ],
)
def test_lane_row_plan_fails_closed_rather_than_guessing(kwargs, code):
    with pytest.raises(contracts.AFDError, match=code):
        _lane_plan(**kwargs)


def _patch_lane_group(monkeypatch, gathered_rows, seen):
    class _LaneGroup:
        @staticmethod
        def all_gather_object(obj):
            seen.append(obj)
            return gathered_rows

    monkeypatch.setattr(
        adapter, "get_parallel", lambda: SimpleNamespace(tp_group=_LaneGroup())
    )


def test_lane_row_plan_all_gathers_extend_rows_over_the_lane_group(monkeypatch):
    """Extend rows follow each lane's own sequence lengths, so they must be sent."""

    seen: list = []
    _patch_lane_group(monkeypatch, [(5, 4), (1, 0), (7, 6), (0, 0)], seen)
    assert _lane_plan(
        gathered=None,
        local_rows=(5, 4),
        lanes=4,
        extend=True,
    ) == ((5, 4), (1, 0), (7, 6), (0, 0))
    assert seen == [(5, 4)]


def test_lane_row_plan_rejects_an_extend_gather_that_misses_a_lane(monkeypatch):
    """A short gather would silently drop a lane's rows from the merge width."""

    _patch_lane_group(monkeypatch, [(5, 4), (1, 0)], [])
    with pytest.raises(contracts.AFDError, match="AFD_LANE_ROW_PLAN_UNAVAILABLE"):
        _lane_plan(gathered=None, local_rows=(5, 4), lanes=4, extend=True)


def _idle(batch_size):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        batch_size=batch_size,
        extend_seq_lens_cpu=None,
    )


@pytest.mark.parametrize("rows", [0, 32], ids=["empty", "dp-padded"])
def test_idle_rank_uses_dp_padded_rows_without_collective(rows):
    assert adapter._token_lengths(_idle(rows)) == (1,) * rows
    half = rows // 2
    assert adapter._stage_ranges(
        batch_size=rows, token_lengths=(1,) * rows, stages=2
    ) == ((0, half, 0, half), (half, rows, half, rows))
    assert (
        _lane_plan(gathered=[rows] * 4, local_rows=(half, half), lanes=4, lane=3)
        == ((half, half),) * 4
    )


def _ffn_args(*, lanes, nnodes, node_rank, **overrides):
    args = _server_args(
        afd_execution_mode="ffn",
        afd_config=config.AFDConfig(lanes=lanes),
        nnodes=nnodes,
        node_rank=node_rank,
        base_gpu_id=0,
        nccl_port=None,
        dist_init_addr=None,
        **overrides,
    )
    return args


@pytest.mark.parametrize(
    "lanes,nnodes,node_rank,owned",
    [
        (4, 1, 0, (0, 1, 2, 3)),
        (8, 2, 0, (0, 1, 2, 3)),
        (8, 2, 1, (4, 5, 6, 7)),
        (8, 4, 2, (4, 5)),
        (1, 1, 0, (0,)),
    ],
)
def test_the_ffn_role_owns_a_contiguous_lane_block_per_host(
    lanes, nnodes, node_rank, owned
):
    """N is a MoE property, not a machine one, so it may exceed a host's GPUs.

    GB300 has 4 per host, so N=8 cannot start at all without this split -- lane 4
    used to ask for cuda:4. Blocks must be contiguous so the global lane ordinal
    stays the global TP rank, which is what the wire groups key on.
    """

    args = _ffn_args(lanes=lanes, nnodes=nnodes, node_rank=node_rank)
    got_nnodes, per_node, got_owned = afd_ffn_server._node_split(args)
    assert (got_nnodes, per_node, got_owned) == (nnodes, lanes // nnodes, owned)
    # Local slots must be 0-based on every host, or node 1 indexes past its GPUs.
    assert [afd_ffn_server._local_slot(args, lane) for lane in owned] == list(
        range(len(owned))
    )
    for foreign in set(range(lanes)) - set(owned):
        with pytest.raises(contracts.AFDError, match="AFD_FFN_LANE_NOT_LOCAL"):
            afd_ffn_server._local_slot(args, foreign)


@pytest.mark.parametrize(
    "lanes,nnodes,node_rank,code",
    [
        (4, 3, 0, "AFD_FFN_NODE_SPLIT_UNSUPPORTED"),
        (8, 3, 0, "AFD_FFN_NODE_SPLIT_UNSUPPORTED"),
        (4, 2, 2, "AFD_FFN_NODE_RANK_INVALID"),
        (4, 2, -1, "AFD_FFN_NODE_RANK_INVALID"),
    ],
)
def test_an_ffn_split_that_would_strand_a_lane_fails_closed(
    lanes, nnodes, node_rank, code
):
    """A partial split leaves lanes nobody serves, which hangs the wire group."""

    with pytest.raises(contracts.AFDError, match=code):
        afd_ffn_server._node_split(
            _ffn_args(lanes=lanes, nnodes=nnodes, node_rank=node_rank)
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
def test_afd_config_preserves_community_moe_runner_option(role, runner):
    cfg = config.AFDConfig.from_json(
        json.dumps(
            {
                "lanes": 4,
                "attention_backend": "nsa",
            }
        )
    )
    args = _role_args(role, lanes=4, moe_runner_backend=runner)
    args.afd_config = cfg
    config.validate_afd_server_args(args)
    assert args.moe_runner_backend == runner
    assert not any(
        "runner" in field or "gemm" in field
        for field in config.AFDConfig.__struct_fields__
    )


@pytest.mark.parametrize(
    "value,eligible,code",
    [
        *[
            (value, False, "MODE_INVALID")
            for value in (None, 0, 1, "false", "true", [], {})
        ],
        (True, True, "MODE_MISMATCH"),
    ],
)
def test_step_descriptor_rejects_invalid_batch_mode(value, eligible, code):
    with pytest.raises(contracts.AFDError, match="AFD_STEP_DESCRIPTOR_" + code):
        contracts.AFDStepDescriptor(
            kind="STEP",
            step_id=0,
            lane_stage_rows=((8, 8),),
            hidden_size=8,
            dtype="bfloat16",
            num_layers=1,
            graph_eligible=eligible,
            is_extend_in_batch=value,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_ffn_reports_run_outcome_before_collective_teardown(monkeypatch, fail):
    args, _ = _ffn_lane_harness(monkeypatch, lanes=4, tp_rank=2)
    events = []

    def run_once():
        if fail:
            raise ValueError("original worker failure")
        return False

    monkeypatch.setattr(
        afd_ffn_server,
        "build_ffn_pipeline",
        lambda **kwargs: SimpleNamespace(run_once=run_once),
    )
    monkeypatch.setattr(
        afd_ffn_server.logger,
        "exception",
        lambda *args: events.append("original_failure"),
    )
    monkeypatch.setattr(
        afd_ffn_server.logger,
        "info",
        lambda fmt, *args: (
            events.append("peer_close") if "AFD_FFN_PEER_CLOSE" in fmt else None
        ),
    )

    def close(*args, **kwargs):
        events.append("teardown")
        raise RuntimeError("secondary teardown failure")

    monkeypatch.setattr(afd_ffn_server, "_close_within_deadline", close)
    with pytest.raises(RuntimeError, match="secondary teardown"):
        afd_ffn_server.run_lane(args, 2)
    assert events == ["original_failure" if fail else "peer_close", "teardown"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))


@pytest.mark.parametrize(
    "extra",
    [
        {"load_format": "remote_instance"},
        {"remote_instance_weight_loader_start_seed_via_transfer_engine": True},
        {"weight_cache_mode": "daemon"},
        {"weight_cache_mode": "client"},
    ],
)
def test_ffn_rejects_loaders_requiring_unowned_lifecycle(extra):
    with pytest.raises(contracts.AFDError, match="FFN_LOADER_LIFECYCLE_UNSUPPORTED"):
        config.validate_afd_server_args(_server_args(afd_execution_mode="ffn", **extra))


def test_ffn_load_uses_native_builder_and_lane_rank(monkeypatch):
    from sglang.srt import model_loader
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.eplb import expert_location
    from sglang.srt.model_executor.model_runner_components import load_model_utils

    cfg = SimpleNamespace(dtype="bfloat16")
    args = _server_args(afd_execution_mode="ffn")
    built = object()
    calls = {}
    monkeypatch.setattr(ModelConfig, "from_server_args", lambda value: cfg)
    monkeypatch.setattr(
        expert_location, "compute_initial_expert_location_metadata", lambda **kw: None
    )
    monkeypatch.setattr(
        expert_location, "set_global_expert_location_metadata", lambda value: None
    )

    def build(**kw):
        calls.update(kw)
        return built

    monkeypatch.setattr(load_model_utils, "build_load_config", build)

    def load(**kw):
        assert kw["load_config"] is built
        assert kw["model_config"] is cfg
        return "loaded"

    monkeypatch.setattr(model_loader, "get_model", load)
    assert afd_ffn_server._load_model(
        server_args=args, device=SimpleNamespace(index=0), lane=3
    ) == ("loaded", "bfloat16")
    assert calls["server_args"] is args
    assert calls["tp_rank"] == 3
    assert calls["weight_cache_mode"] == "off"
    assert calls["remote_instance_weight_transporter_engine"] is None


def test_native_load_builder_preserves_ffn_loader_options(monkeypatch):
    from sglang.srt.arg_groups.fields.model import Model
    from sglang.srt.model_executor.model_runner_components import load_model_utils

    model = Model(
        model_path="unused",
        load_format="safetensors",
        download_dir="/tmp/afd-load-test",
        model_loader_extra_config='{"num_threads": 3}',
        modelopt_checkpoint_restore_path="/tmp/afd-modelopt",
    )
    monkeypatch.setattr(load_model_utils, "get_model", lambda: model)
    actual = load_model_utils.build_load_config(
        server_args=SimpleNamespace(modelexpress_config=None),
        tp_rank=3,
        remote_instance_weight_transporter_engine=None,
        remote_instance_weight_transporter_session_id="",
        draft_model_idx=None,
        weight_cache_mode="off",
        weight_cache_socket=None,
    )
    assert actual.tp_rank == 3
    assert actual.download_dir == model.download_dir
    assert actual.model_loader_extra_config == {"num_threads": 3}
    assert (
        actual.modelopt_config.checkpoint_restore_path
        == model.modelopt_checkpoint_restore_path
    )


@pytest.mark.parametrize("failure", ["start", "lane", "drain"])
def test_ffn_parent_reaps_owned_children_on_failure(monkeypatch, failure):
    import multiprocessing.connection

    workers = []

    class Worker:
        def __init__(self, **kwargs):
            self.name = str(len(workers))
            self.sentinel = len(workers)
            self.pid = None
            self.exitcode = None
            self.signals = []
            self.joins = []
            workers.append(self)

        def start(self):
            if failure == "start" and self.sentinel == 1:
                raise RuntimeError("spawn failed")
            self.pid = 100 + self.sentinel

        def terminate(self):
            self.signals.append("TERM")

        def kill(self):
            self.signals.append("KILL")
            self.exitcode = -9

        def join(self, timeout):
            self.joins.append(timeout)

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda kind: SimpleNamespace(Process=Worker)
    )
    calls = []

    def ready(pending, timeout=None):
        calls.append(timeout)
        if len(calls) > 1:
            return []
        workers[0].exitcode = 2 if failure == "lane" else 0
        return [0]

    monkeypatch.setattr(multiprocessing.connection, "wait", ready)
    error = (
        "spawn failed"
        if failure == "start"
        else ("LANE_EXITED" if failure == "lane" else "DRAIN_TIMEOUT")
    )
    with pytest.raises(Exception, match=error):
        afd_ffn_server._run_lanes(_server_args(), owned=(0, 1))
    for worker in workers:
        if worker.pid is None:
            assert worker.signals == worker.joins == []
        elif worker.exitcode == -9:
            assert worker.signals == ["TERM", "KILL"]
            assert len(worker.joins) == 2
            assert all(timeout >= 0 for timeout in worker.joins)
    if failure == "drain":
        assert calls[0] is None
        assert 0 <= calls[1] <= 3 * config.AFDConfig().close_timeout_seconds


def _ignore_term_until_killed(ready):
    import signal
    import time

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(1)


def test_ffn_cleanup_kills_term_resistant_child_but_preserves_unowned_process():
    import time

    context = multiprocessing.get_context("spawn")
    ready = [context.Event(), context.Event()]
    children = [
        context.Process(target=_ignore_term_until_killed, args=(event,))
        for event in ready
    ]
    try:
        for child in children:
            child.start()
        assert all(event.wait(30) for event in ready)
        start = time.monotonic()
        afd_ffn_server._stop_workers(children[:1], seconds=0.5)
        assert time.monotonic() - start < 10
        assert children[0].exitcode == -9
        assert children[1].is_alive()
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.kill()
                child.join(timeout=5)
