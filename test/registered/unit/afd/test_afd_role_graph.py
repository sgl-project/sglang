"""Whole-role graph strategy: the seam, the sentinel, and the fail-closed mode."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import contracts
from sglang.srt.afd import integration as afd_integration
from sglang.srt.afd import profiles, role_graph
from sglang.srt.afd.contracts import AFDError
from sglang.srt.afd.role_graph import AFDRoleGraphProgramSpec, TorchRoleGraphDriver
from sglang.srt.model_executor.runner_utils import pool as pool_utils
from sglang.test.afd.graph_fixtures import (
    CAPTURE_SIZES,
    RecordingDriver,
    RecordingProgram,
    make_shape,
    role_service,
    role_step,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tensor_rows(rows):
    return torch.zeros(rows, dtype=torch.bfloat16)


def _shape(*, config, stage_rows=(1, 1)):
    return make_shape(
        lane=0,
        lane_rows=(stage_rows,),
        hidden_size=1,
        dtype="bfloat16",
        config=config,
    )


def test_fixed_graph_identity_is_part_of_actual_peer_contract_and_digest():
    adapter = SimpleNamespace(
        num_layers=2,
        hidden_size=16,
        attention_capability=lambda **kwargs: {"backend": kwargs["configured_backend"]},
    )
    descriptor = afd_integration._model_descriptor(
        capture_sizes=(8, 16, 32),
        role=contracts.AFDRole.ATTENTION,
        adapter=adapter,
        profile=profiles.QWEN3_PAIRED_C1,
        config=config_mod.AFDConfig(),
        dtype="bfloat16",
    )
    runtime = descriptor["runtime_contract"]
    assert runtime["graph_strategy"] == profiles.ROLE_GRAPH_STRATEGY_ID
    assert (
        runtime["capability_profile"]["graph_strategy"]
        == profiles.ROLE_GRAPH_STRATEGY_ID
    )
    assert descriptor["runtime_contract_digest"] == contracts.contract_digest(runtime)
    changed = dict(runtime, graph_strategy="local-compute-child-graph-v1")
    assert contracts.contract_digest(changed) != descriptor["runtime_contract_digest"]


def test_a_capture_that_writes_nothing_takes_the_role_down(monkeypatch):
    def sentinel_static(self):
        raise contracts.AFDError("AFD_ROLE_GRAPH_SELF_TEST_SENTINEL_STATIC")

    monkeypatch.setattr(RecordingProgram, "require_sentinel_written", sentinel_static)
    driver = RecordingDriver()
    cfg, service = role_service(driver=driver)
    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_SELF_TEST_SENTINEL_STATIC",
    ):
        role_step(service, cfg, step_id=1)
    assert driver.programs[0].armed == 1
    # It did replay -- that is the round the sentinel proves ran -- and the check
    # sits outside the fallback so no eager path can swallow it.
    assert driver.programs[0].replays == 1


def test_step_rows_must_match_the_shape_the_bucket_was_selected_for():
    cfg, service = role_service()
    service.begin_step(
        capture=True,
        step_id=1,
        shape=_shape(config=cfg, stage_rows=(1, 1)),
        eligible=True,
    )
    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_STEP_ROWS_INVALID",
    ):
        service.execute_step(
            stage_args=((tensor_rows(2),), (tensor_rows(1),)),
            stage_rows=(2, 1),
            forward_batches=(None, None),
            compute=lambda staged: staged,
            metadata_guards=(None, None),
        )


def test_the_usage_snapshot_names_which_strategy_produced_it(caplog):
    cfg, service = role_service()
    with caplog.at_level("INFO"):
        role_step(service, cfg, step_id=1)
    snapshots = [
        record.getMessage()
        for record in caplog.records
        if "AFD_GRAPH_USAGE_SNAPSHOT" in record.getMessage()
    ]
    assert snapshots
    assert '"strategy": "role"' in snapshots[-1]
    assert '"expected_operations": 1' in snapshots[-1]


def test_the_ffn_role_captures_although_it_stages_no_tensors():
    driver = RecordingDriver()
    cfg, service = role_service(driver=driver, role=contracts.AFDRole.FFN)
    service.begin_step(
        capture=True,
        step_id=1,
        shape=_shape(config=cfg, stage_rows=(1, 1)),
        eligible=True,
    )
    ran = []
    result = service.execute_step(
        # Empty per stage: nothing is staged and nothing is extracted.
        stage_args=((), ()),
        stage_rows=(1, 1),
        forward_batches=(None, None),
        compute=lambda staged: (ran.append(staged), ((), ()))[1],
        metadata_guards=(None, None),
    )
    assert service.end_step() is None
    assert result.reason is contracts.AFDReason.ARMING
    assert len(driver.programs) == 1
    assert driver.programs[0].spec.device == "cuda:0"
    assert driver.programs[0].armed == 1


def test_a_role_graph_without_a_device_refuses_to_be_built():
    with pytest.raises(
        contracts.AFDError,
        match="AFD_ROLE_GRAPH_DEVICE_REQUIRED",
    ):
        role_graph.AFDRoleGraphService(
            capture_sizes=CAPTURE_SIZES,
            role=contracts.AFDRole.FFN,
            config=config_mod.AFDConfig(),
            num_layers=2,
            driver=RecordingDriver(),
        )


class _GraphBoundary:
    def __init__(self, events):
        self.events = events
        self.replays = 0
        self.on_replay = None
        self.failure = None

    def reset(self):
        self.events.append("reset")

    def replay(self):
        self.replays += 1
        self.events.append("replay")
        if self.failure is not None:
            raise self.failure
        if self.on_replay is not None:
            self.on_replay()


@pytest.fixture
def cuda_boundary(monkeypatch):
    events, graphs, pools, captures = ([], [], [], [])
    stream = object()
    resources = SimpleNamespace(graph_memory_pool=None, graph_pool_borrow=None)
    monkeypatch.setattr(pool_utils, "get_resources", lambda: resources)

    def make_graph():
        graph = _GraphBoundary(events)
        graphs.append(graph)
        return graph

    def make_pool():
        pool = object()
        pools.append(pool)
        return pool

    @contextmanager
    def device(device):
        events.append(("device", str(device)))
        yield

    @contextmanager
    def capture(cuda_graph, *, pool, stream=None):
        captures.append((cuda_graph, pool, stream))
        events.append("capture_enter")
        try:
            yield
        finally:
            events.append("capture_exit")

    @contextmanager
    def stream_scope(value):
        assert value is stream
        events.append("stream_enter")
        try:
            yield
        finally:
            events.append("stream_exit")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", make_graph)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", make_pool)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    monkeypatch.setattr(
        pool_utils, "get_or_create_global_graph_capture_stream", lambda: stream
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: events.append("sync"))
    monkeypatch.setenv("SGLANG_ENABLE_GRAPH_POOL_PRECARVE", "0")
    monkeypatch.setenv("SGLANG_ENABLE_GRAPH_POOL_BORROW", "0")
    return SimpleNamespace(
        events=events, graphs=graphs, pools=pools, captures=captures, stream=stream
    )


def _guard(events, label):

    def record(method):
        return lambda *args: events.append((label, method))

    guard = SimpleNamespace(
        capture=record("capture"),
        activate_in_graph=record("activate"),
        prepare_replay=record("prepare_replay"),
        assert_stable=record("assert_stable"),
        restore=record("restore"),
    )
    guard.close = lambda: guard.restore()
    return guard


def _capture(
    boundary,
    *,
    failure=None,
    driver=None,
    digest="test-shape",
    restore_failure=None,
):
    """The compute closure records weak references, never extra tensor owners."""
    calls, input_refs = ([], [])
    guard = _guard(boundary.events, "metadata")
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def compute_role(stages):
        calls.append("forward")
        boundary.events.append("forward")
        input_refs.extend((weakref.ref(values[0]) for values in stages))
        if failure is not None:
            raise failure
        return tuple(((values[0], None) for values in stages))

    driver = driver or TorchRoleGraphDriver()
    if restore_failure is not None:

        def fail_restore():
            raise restore_failure

        guard.restore = fail_restore
    spec = AFDRoleGraphProgramSpec(
        shape_digest=digest,
        device="cpu",
        bucket_rows=(4, 4),
        stage_rows=(2, 1),
        stage_args=((source,), (torch.ones(1, 3),)),
        compute=compute_role,
        forward_batches=(None, None),
        metadata_guards=(guard, None),
    )
    program = driver.capture(spec=spec)
    pool = boundary.captures[-1][1]
    return (program, calls, input_refs, pool)


def _replay(program, value=7.0):
    source = torch.full((1, 3), value)
    return program.replay(
        stage_args=((source,), (torch.full((2, 3), value + 1),)),
        stage_rows=(1, 2),
        forward_batches=(None, None),
    )


def test_native_warmup_precedes_single_capture_and_replay_has_no_python_forward(
    cuda_boundary,
):
    program, calls, inputs, pool = _capture(cuda_boundary)
    assert calls == ["forward"] * 3
    assert len(cuda_boundary.captures) == 1
    before, during = (
        cuda_boundary.events.index("capture_enter"),
        cuda_boundary.events.index("capture_exit"),
    )
    assert cuda_boundary.events[:before].count("forward") == 2
    assert cuda_boundary.events[before:during].count("forward") == 1
    assert cuda_boundary.captures[0][1] is pool
    assert cuda_boundary.captures[0][2] is cuda_boundary.stream
    assert cuda_boundary.events.index("stream_enter") < cuda_boundary.events.index(
        "forward"
    )
    assert cuda_boundary.events.index("stream_exit") > during
    assert cuda_boundary.graphs[0].replays == 0
    assert cuda_boundary.events.count("sync") == 3
    assert cuda_boundary.events.index(
        ("metadata", "capture")
    ) < cuda_boundary.events.index("capture_enter")
    assert cuda_boundary.events.index(
        ("metadata", "activate")
    ) < cuda_boundary.events.index("capture_exit")
    assert cuda_boundary.events[-1] == ("metadata", "restore")
    cuda_boundary.events.clear()
    outputs = _replay(program)
    first = outputs[0][0]
    assert torch.equal(first, torch.full((1, 3), 7.0))
    assert (
        first.untyped_storage().data_ptr() == inputs[0]().untyped_storage().data_ptr()
    )
    assert torch.count_nonzero(inputs[0]()[1:]) == 0
    assert calls == ["forward"] * 3
    assert cuda_boundary.graphs[0].replays == 1
    assert cuda_boundary.events == [
        ("metadata", "prepare_replay"),
        ("metadata", "assert_stable"),
        "replay",
        ("metadata", "restore"),
    ]
    assert torch.equal(outputs[1][0], torch.full((2, 3), 8.0))
    assert inputs[0]().data_ptr() != inputs[1]().data_ptr()
    program.close()


def test_program_owns_static_inputs_until_close_and_close_is_idempotent(cuda_boundary):
    program, _, inputs, _ = _capture(cuda_boundary)
    gc.collect()
    assert all((ref() is not None for ref in inputs))
    program.close()
    after_close = list(cuda_boundary.events)
    assert after_close.count("reset") == 1
    program.close()
    assert cuda_boundary.events == after_close
    gc.collect()
    assert all((ref() is None for ref in inputs))
    with pytest.raises(AFDError, match="GRAPH_PROGRAM_CLOSED"):
        _replay(program)
    assert cuda_boundary.graphs[0].replays == 0


def test_capture_failure_restores_metadata_without_replay_or_extra_compute(
    cuda_boundary,
):
    failure = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed"):
        _capture(cuda_boundary, failure=failure)
    assert len(cuda_boundary.captures) == 0
    assert cuda_boundary.graphs == []
    assert cuda_boundary.events[-1] == ("metadata", "restore")


def test_replay_failure_restores_metadata_without_private_retry(cuda_boundary):
    program, calls, _, _ = _capture(cuda_boundary)
    cuda_boundary.graphs[0].failure = RuntimeError("replay failed")
    with pytest.raises(RuntimeError, match="replay failed"):
        _replay(program)
    assert calls == ["forward"] * 3
    assert cuda_boundary.graphs[0].replays == 1
    assert cuda_boundary.events[-1] == ("metadata", "restore")
    program.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
