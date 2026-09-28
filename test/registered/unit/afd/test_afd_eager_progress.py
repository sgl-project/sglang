"""CPU regressions for first-shape S2 eager transport progress."""

from types import SimpleNamespace

import pytest

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import pipeline as pipeline_mod
from sglang.srt.afd import role_graph as role_graph
from sglang.srt.afd.model_adapters import base as qwen
from sglang.test.afd.graph_fixtures import CAPTURE_SIZES, RecordingDriver
from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.afd.pipeline_fixtures import (
    AttentionAdapter,
    DecodeBatch,
    DecodeMode,
    EagerGraph,
    FakeTensor,
    FFNAdapter,
    RetainedGraph,
    RetainedTransport,
    hidden_attention,
    hidden_ffn,
)
from sglang.test.ci.ci_register import register_cpu_ci


def _descriptor(rows, *, layers=1, eligible=False, step_id=10, lanes=1, kind="STEP"):
    return contracts.AFDStepDescriptor(
        kind=kind,
        step_id=step_id,
        lane_stage_rows=(tuple(rows),) * lanes,
        hidden_size=8,
        dtype="bfloat16",
        num_layers=layers,
        graph_eligible=eligible,
    )


def _run_ffn(
    rows,
    *,
    layers=1,
    eligible=False,
    graph=None,
    capturing=False,
    ffn_lanes=1,
    attention_lanes=None,
    lane=0,
):
    matrix = tuple(
        tuple(value + index for value in rows)
        for index in range(attention_lanes or ffn_lanes)
    )
    pipeline, _, transport = hidden_ffn(
        matrix,
        layers,
        graph=graph,
        capturing=capturing,
        ffn_lanes=ffn_lanes,
        eligible=eligible,
        lane=lane,
    )
    assert pipeline.run_once()
    return transport


def test_s2_extend_keeps_one_receive_at_a_time_when_eager():
    token_lengths = tuple((index * 37) % 511 + 1 for index in range(29))
    ranges = qwen._stage_ranges(
        batch_size=29,
        token_lengths=token_lengths,
        stages=2,
    )
    assert [(item[0], item[1]) for item in ranges] == [(0, 15), (15, 29)]
    rows = tuple(item[3] - item[2] for item in ranges)
    assert all(rows)
    # No prefetch on this path. A receive posted a boundary early would write the
    # buffer the previous boundary is still reading, and the edge that keeps those
    # apart is the fork the transport only draws while capturing.
    assert _run_ffn(rows, layers=2).timeline[:-1] == [
        ("recv", 0),
        ("wait", 0),
        ("local", 0, 0),
        ("return", 0),
        ("recv", 1),
        ("wait", 1),
        ("local", 0, 1),
        ("return", 1),
        ("recv", 0),
        ("wait", 0),
        ("local", 1, 0),
        ("return", 0),
        ("recv", 1),
        ("wait", 1),
        ("local", 1, 1),
        ("return", 1),
    ]


@pytest.mark.parametrize(
    "ffn_lanes,attention_lanes,lane,widths",
    [
        (1, 1, 0, [15, 14]),
        (2, 4, 1, [17, 18, 16, 17]),
    ],
)
def test_capture_prefetch_and_lane_ownership(ffn_lanes, attention_lanes, lane, widths):
    """Ordinal 1 of two FFN ranks over four lanes holds lanes 2 and 3, in order."""

    graph = EagerGraph()
    transport = _run_ffn(
        (15, 14),
        layers=2,
        graph=graph,
        capturing=True,
        ffn_lanes=ffn_lanes,
        attention_lanes=attention_lanes,
        lane=lane,
    )
    group = attention_lanes // ffn_lanes
    # Literal two-layer oracle. Fan-in repeats only wire edges, never compute/wait.
    order = [
        ("recv", 0),
        ("recv", 1),
        ("wait", 0),
        ("local", 0, 0),
        ("return", 0),
        ("recv", 0),
        ("wait", 1),
        ("local", 0, 1),
        ("return", 1),
        ("recv", 1),
        ("wait", 0),
        ("local", 1, 0),
        ("return", 0),
        ("wait", 1),
        ("local", 1, 1),
        ("return", 1),
        ("rejoin",),
    ]
    assert transport.timeline == [
        event
        for event in order
        for _ in range(group if event[0] in ("recv", "return") else 1)
    ]
    assert transport.recv_widths == widths * 2
    # One flat tuple per stage, one tensor per lane. The whole-role graph walks
    # exactly one level in to find its replay sentinel, so a nested tuple would
    # hand it a tuple where it needs a tensor and only fail on a device.
    assert len(graph.step_outputs) == 2
    for values in graph.step_outputs:
        assert len(values) == group
        assert all(hasattr(value, "shape") for value in values)


def test_captured_receive_lookahead_is_exactly_one_boundary():
    """Pin the depth, not the op shape.

    NCCL's requirement is a consistent order per communicator, and both keep it:
    a2e carries receives 0,1,2,... against the peer's sends 0,1,2,..., and e2a
    carries the returns in the same step. What a deeper lookahead would break is
    the write-after-read edge on the per-stage buffers, which hold exactly one
    boundary of slack, and the bound on host run-ahead the cold-start deadlock fix
    rests on -- the attention role issues `stages` dispatches before its first
    wait.
    """

    for stages, layers in ((2, 3), (2, 2), (2, 48)):
        rows = tuple(4 + index for index in range(stages))
        timeline = _run_ffn(
            rows, layers=layers, graph=EagerGraph(), capturing=True
        ).timeline
        posted = waited = depth = 0
        for entry in timeline:
            if entry[0] == "recv":
                posted += 1
                depth = max(depth, posted - waited)
            elif entry[0] == "wait":
                waited += 1
        boundaries = stages * layers
        assert posted == waited == boundaries
        assert depth == 2, (stages, layers, depth)
        order = [index % stages for index in range(boundaries)]
        assert [item[1] for item in timeline if item[0] == "recv"] == order
        assert [item[1] for item in timeline if item[0] == "return"] == order
        assert [item[1:] for item in timeline if item[0] == "local"] == [
            (layer, stage) for layer in range(layers) for stage in range(stages)
        ]


def test_zero_tail_keeps_stage_identity_and_order():
    assert _run_ffn((8, 0)).timeline[:-1] == [
        ("recv", 0),
        ("wait", 0),
        ("local", 0, 0),
        ("return", 0),
    ]
    assert [item for item in _run_ffn((3, 2)).timeline if item[0] == "return"] == [
        ("return", 0),
        ("return", 1),
    ]


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class MeasuredDriver(RecordingDriver):
    def capture(self, *, spec):
        program = super().capture(spec=spec)
        program.retained_hbm_bytes = 20_000
        return program

    def memory_usage(self, *, device):
        # Larger than the backing estimate: capture growth must remain charged.
        allocated = sum(program.retained_hbm_bytes for program in self.programs)
        return allocated, allocated


def _retained_usage(graph):
    usage = graph.usage()
    bucket = next(iter(usage["buckets"].values()))
    return usage["retained_hbm_bytes"], bucket["usage"]


def _make_retained_pipeline(*, role, stages):
    cfg = config_mod.AFDConfig(
        stages=stages,
    )
    graph = role_graph.AFDRoleGraphService(
        capture_sizes=CAPTURE_SIZES,
        role=role,
        config=cfg,
        num_layers=2,
        driver=MeasuredDriver(),
        device="cuda:0",
    )
    transport = RetainedTransport(role=role)
    connector = SimpleNamespace(
        transport=transport,
        graph_strategy=graph,
        shutdown_requested=False,
    )
    if role == contracts.AFDRole.ATTENTION:
        pipeline = pipeline_mod.AFDAttentionPipeline(
            adapter=AttentionAdapter(),
            connector=connector,
            config=cfg,
            shape_factory=planned_shape,
        )

        def run(step):
            pipeline._startup_capture = step == 0
            return pipeline.execute(
                hidden_states=FakeTensor(2 * stages),
                residual=None,
                positions=FakeTensor(2 * stages, 1),
                forward_batch=DecodeBatch(),
            )

    else:
        adapter = FFNAdapter(layers=2)
        adapter.timeline = transport.timeline
        pipeline = pipeline_mod.AFDFFNPipeline(
            adapter=adapter,
            connector=connector,
            config=cfg,
            device="cpu",
            dtype="bfloat16",
            shape_factory=planned_shape,
        )

        def run(step):
            transport.descriptor = _descriptor(
                (2,) * stages,
                layers=2,
                eligible=True,
                step_id=step,
                kind="CAPTURE" if step == 0 else "STEP",
            )
            return pipeline.run_once()

    return graph, transport, run


@pytest.mark.parametrize(
    "role",
    (contracts.AFDRole.ATTENTION, contracts.AFDRole.FFN),
)
@pytest.mark.parametrize("stages", (2,))
def test_installed_pipeline_reuses_hbm_reservation_without_growth(role, stages):
    graph, transport, run = _make_retained_pipeline(
        role=role,
        stages=stages,
    )
    run(0)
    run(1)
    installed_hbm, installed_usage = _retained_usage(graph)
    bucket = graph._cache.buckets[0]
    installed_backing_hbm = bucket.retained_backing_hbm_bytes
    assert installed_usage["installs"] == 1
    run(2)
    transport.reported_hbm_delta = -1
    run(3)
    replay_hbm, replay_usage = _retained_usage(graph)
    assert replay_hbm == installed_hbm
    assert replay_usage["replays"] == 3
    assert bucket.retained_backing_hbm_bytes == installed_backing_hbm
    assert transport.retained_allocations == 1


@pytest.mark.parametrize(
    "role",
    (contracts.AFDRole.ATTENTION, contracts.AFDRole.FFN),
)
@pytest.mark.parametrize("stages", (2,))
def test_installed_hbm_growth_aborts_and_ends_step(role, stages):
    graph, transport, run = _make_retained_pipeline(
        role=role,
        stages=stages,
    )
    run(0)
    run(1)
    bucket = graph._cache.buckets[0]
    programs = (bucket.program,)
    transport.reported_hbm_delta = 1

    with pytest.raises(contracts.AFDError, match="STEP_NOT_GRAPHABLE"):
        run(2)
    usage = graph.usage()
    bucket_usage = next(iter(usage["buckets"].values()))
    assert bucket_usage["phase"] == "terminal_eager"
    assert bucket_usage["terminal_reason"] == contracts.AFDReason.HBM_LIMIT.value
    assert usage["retained_hbm_bytes"] == usage["capture_hbm_bytes"] == 20_000
    assert bucket.retained_backing_hbm_bytes == 0
    assert bucket.program is None
    assert all(program.closed for program in programs)
    assert transport.releases == 1
    assert graph._selection is None

    with pytest.raises(contracts.AFDError, match="STEP_NOT_GRAPHABLE"):
        run(3)
    assert graph._selection is None


def test_eager_and_captured_role_steps_preserve_stream_waits():
    """Both execute the retained role path without per-boundary host blocking."""

    eager = _run_ffn((2, 2), layers=3).timeline

    assert {item[0] for item in eager if item[0] in ("wait", "sync")} == {"wait"}

    # An unjoined side stream leaves its work outside the graph's ordering, so
    # the captured region must rejoin exactly once, at the very end.
    assert eager[-1] == ("rejoin",)
    assert [item[0] for item in eager].count("rejoin") == 1

    # Capture changes receive lookahead, preserving compute and return order.
    captured = _run_ffn((2, 2), layers=3, capturing=True).timeline
    assert {item[0] for item in captured if item[0] in ("wait", "sync")} == {"wait"}
    assert captured[-1] == ("rejoin",)
    assert [item[0] for item in captured].count("rejoin") == 1
    for kinds in (("local", "return"), ("recv",)):
        assert [item for item in eager if item[0] in kinds] == [
            item for item in captured if item[0] in kinds
        ]
    assert [item[0] for item in eager][:2] == ["recv", "wait"]
    assert [item[0] for item in captured][:2] == ["recv", "recv"]


def test_a_captured_step_receives_at_the_bucket_width_not_real_rows():
    """Inside a fixed-shape capture every tensor is the bucket's padded width.

    This is how the whole-role graph failed on GB300. Handing the loop row-sliced
    receive buffers makes the first layer's output narrow to real rows while the
    residual it is added to stays padded, and the fused rmsnorm rejects the
    mismatch. Eager execution keeps real-row slices; captured execution uses the
    retained bucket width.
    """

    width = CAPTURE_SIZES[0]
    eager = _run_ffn((2, 2), layers=2)
    armed = _run_ffn((2, 2), layers=2, graph=RetainedGraph())

    assert set(eager.recv_widths) == {2}
    assert set(armed.recv_widths) == {width}


def _run_attention(rows, *, layers=2, graph=None):
    cfg = config_mod.AFDConfig(
        stages=len(rows),
    )
    transport = RetainedTransport(role=contracts.AFDRole.ATTENTION)
    pipeline = pipeline_mod.AFDAttentionPipeline(
        adapter=AttentionAdapter(),
        connector=SimpleNamespace(
            transport=transport,
            graph_strategy=graph or EagerGraph(),
            shutdown_requested=False,
        ),
        config=cfg,
        shape_factory=planned_shape,
    )
    total = sum(rows)
    pipeline.execute(
        hidden_states=FakeTensor(total),
        residual=None,
        positions=FakeTensor(total, 1),
        forward_batch=DecodeBatch(),
    )
    return transport


def test_the_attention_role_applies_the_same_captured_step_width_rule():
    """The crash was on the attention ranks, so pin the rule on both roles.

    Same rule as the FFN side: a graphed step runs at the bucket width throughout,
    a step both roles decline runs eager at real rows.
    """

    width = CAPTURE_SIZES[0]
    eager = _run_attention((2, 2))
    armed = _run_attention((2, 2), graph=RetainedGraph())

    assert set(eager.recv_widths) == {2}
    assert set(armed.recv_widths) == {width}


@pytest.mark.parametrize("participates", [(True, False), (False, True)])
@pytest.mark.parametrize("capturing", [False, True])
def test_single_active_stage_keeps_hidden_until_compute(participates, capturing):
    pipeline = object.__new__(pipeline_mod.AFDFFNPipeline)
    timeline = []
    active = participates.index(True)
    hidden = [0]
    payload = (hidden,)
    count = [0]

    def receive(buffers):
        assert buffers is payload
        count[0] += 1
        hidden[0] = count[0]
        timeline.append("recv")
        return object()

    def compute(*, layer, stage, hidden_states, residual):
        assert stage.index == active
        assert hidden_states[0][0] == layer + 1
        timeline.append("compute")
        return payload, None

    pipeline._adapter = SimpleNamespace(num_layers=3, local_compute=compute)
    pipeline._connector = SimpleNamespace(
        transport=SimpleNamespace(
            capturing=capturing,
            wait=lambda event: timeline.append("wait"),
            receive_dispatch=receive,
            return_result=lambda output: timeline.append("send"),
        )
    )
    output = pipeline._run_layers(
        stages=[SimpleNamespace(index=i) for i in range(len(participates))],
        recv_buffers=[payload] * len(participates),
        participates=participates,
    )
    assert timeline == ["recv", "wait", "compute", "send"] * 3
    assert output[active] is payload
    assert all(not value for i, value in enumerate(output) if i != active)


def _runtime_enum(relative_path, name):
    """Import the production enum through its normal package."""
    import importlib

    module = "sglang.srt." + relative_path.removesuffix(".py").replace("/", ".")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "mode_name,global_extend,rows,expected_extend,eligible",
    [
        ("EXTEND", False, ((3, 5),), True, False),
        ("MIXED", True, ((3, 5), (2, 4)), True, False),
        ("DECODE", True, ((3, 5), (2, 4)), True, False),
        ("IDLE", True, ((3, 5), (2, 4)), True, False),
        ("DECODE", False, ((3, 5), (2, 4)), False, True),
        ("IDLE", False, ((3, 5), (2, 4)), False, True),
        ("DECODE", False, ((3, 0), (2, 4)), False, False),
        ("IDLE", False, ((0, 0),), False, False),
    ],
)
def test_attention_step_mode_comes_from_typed_mode_not_graph_eligibility(
    mode_name, global_extend, rows, expected_extend, eligible
):
    mode = _runtime_enum("model_executor/forward_batch_info.py", "ForwardMode")
    _, _, transport, run = hidden_attention(rows, 2)
    run(SimpleNamespace(forward_mode=mode[mode_name], is_extend_in_batch=global_extend))
    assert transport.descriptor.is_extend_in_batch is expected_extend
    assert transport.descriptor.graph_eligible is eligible


@pytest.mark.parametrize("global_extend", [None, 0, 1, "false"])
def test_attention_refuses_malformed_global_mode_before_dispatch(global_extend):
    _, _, transport, run = hidden_attention(((2, 2),), 2)
    with pytest.raises(contracts.AFDError, match="AFD_ATTENTION_FORWARD_MODE_INVALID"):
        run(
            SimpleNamespace(forward_mode=DecodeMode(), is_extend_in_batch=global_extend)
        )
    assert transport.descriptor is None and not transport.sent


def test_attention_refuses_unrecognized_typed_mode_even_with_extend_flag():
    mode = _runtime_enum("model_executor/forward_batch_info.py", "ForwardMode")
    _, _, transport, run = hidden_attention(((2, 2),), 2)
    with pytest.raises(
        contracts.AFDError, match="AFD_ATTENTION_FORWARD_MODE_UNSUPPORTED"
    ):
        run(SimpleNamespace(forward_mode=mode.PREBUILT, is_extend_in_batch=True))
    assert transport.descriptor is None and not transport.sent


@pytest.mark.parametrize("value", [None, 0, 1, "false", True])
def test_ffn_revalidates_unchecked_step_mode_before_data_receive(value):
    import msgspec

    pipeline, _, transport = hidden_ffn(((3, 5),), 1)
    msgspec.structs.force_setattr(transport.descriptor, "is_extend_in_batch", value)
    code = "MODE_MISMATCH" if value is True else "MODE_INVALID"
    with pytest.raises(contracts.AFDError, match=code):
        pipeline.run_once()
    assert not transport.packets


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))


def test_metadata_guards_are_built_only_for_capture(monkeypatch):
    calls = []
    monkeypatch.setattr(
        AttentionAdapter,
        "metadata_guard",
        staticmethod(lambda **kwargs: calls.append(kwargs["stage"].index)),
    )
    graph, _, run = _make_retained_pipeline(role=contracts.AFDRole.ATTENTION, stages=2)
    run(0)
    assert calls == [0, 1]
    run(1)
    assert calls == [0, 1]
    run(2)
    run(3)
    assert calls == [0, 1]
    assert sum(b.usage.replays for b in graph._cache.buckets) == 3
