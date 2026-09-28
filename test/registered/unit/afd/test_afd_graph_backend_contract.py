"""CPU contracts required before composing AFD with a shared graph backend.

These exercise real drivers/programs and real tensor storage. Only CUDA and TP
boundaries are replaced. A no-op graph replay is sufficient for alias/ownership
checks; the sentinel test simulates just its device copy. None of these tests
claim to validate CUDA capture, NCCL rounds, or inference performance.
"""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.contracts import AFDError
from sglang.srt.afd.role_graph import AFDRoleGraphProgramSpec, TorchRoleGraphDriver
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as full
from sglang.srt.model_executor.runner_utils import pool as pool_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


def test_output_views_alias_program_storage_without_cross_program_reuse(cuda_boundary):
    first, _, first_inputs, _ = _capture(cuda_boundary)
    second, _, second_inputs, _ = _capture(cuda_boundary)
    assert cuda_boundary.captures[0][1] is cuda_boundary.captures[1][1]
    outputs = _replay(first, 4.0)
    held_output = outputs[0][0]
    _replay(second, 20.0)
    assert torch.equal(held_output, torch.full((1, 3), 4.0))
    assert first_inputs[0]().data_ptr() != second_inputs[0]().data_ptr()
    _replay(first, 9.0)
    assert torch.equal(held_output, torch.full((1, 3), 9.0))
    first.close()
    assert second_inputs[0]() is not None
    assert torch.equal(held_output, torch.full((1, 3), 9.0))
    second.close()


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


def test_role_sentinel_checks_the_existing_replay_without_posting_an_extra_round(
    cuda_boundary,
):
    program, calls, _, _ = _capture(cuda_boundary)
    graph = cuda_boundary.graphs[0]
    program.arm_sentinel()
    with pytest.raises(AFDError, match="SELF_TEST_SENTINEL_STATIC"):
        program.require_sentinel_written()
    assert graph.replays == 0
    sentinel_ref, source_ref = (
        weakref.ref(program._sentinel),
        weakref.ref(program._sentinel_source),
    )
    graph.on_replay = lambda: sentinel_ref().copy_(source_ref().reshape(-1)[:1])
    _replay(program)
    program.require_sentinel_written()
    assert graph.replays == 1
    assert calls == ["forward"] * 3
    program.close()


def _native_backend(boundary, monkeypatch):
    shared_pool = object()
    monkeypatch.setattr(
        full, "get_or_create_global_graph_memory_pool", lambda device: shared_pool
    )
    monkeypatch.setattr(
        full,
        "set_graph_pool_id",
        lambda pool: boundary.events.append(("global_pool", pool)),
    )
    runner = SimpleNamespace(
        device_module=torch.cuda,
        model_runner=SimpleNamespace(
            tp_group=SimpleNamespace(barrier=lambda: boundary.events.append("barrier"))
        ),
        enable_profile_cuda_graph=False,
    )
    return (full.FullCudaGraphBackend(runner, reuse_output_buffer=False), shared_pool)


def test_native_backend_still_owns_two_warmups_barriers_and_global_pool(
    cuda_boundary, monkeypatch
):
    backend, pool = _native_backend(cuda_boundary, monkeypatch)
    calls = []
    stream = object()

    def forward():
        calls.append("forward")
        return torch.ones(4, 3)

    key = ShapeKey(size=4)
    with backend.capture_session(stream):
        backend.capture_one(key, forward)
    assert calls == ["forward"] * 3
    assert cuda_boundary.events.count("barrier") == 2
    assert cuda_boundary.graphs[0].replays == 0
    assert cuda_boundary.captures == [(cuda_boundary.graphs[0], pool, stream)]
    assert backend._capture_stream is None
    output = backend.replay(key, static_forward_batch=None)
    assert torch.equal(output, torch.ones(4, 3))
    assert cuda_boundary.graphs[0].replays == 1
    assert calls == ["forward"] * 3
    backend.cleanup()
    backend.cleanup()
    assert not backend.can_run(None, key)


def test_native_backend_keeps_explicit_operation_keys_and_owned_inputs(
    cuda_boundary, monkeypatch
):
    backend, _ = _native_backend(cuda_boundary, monkeypatch)
    owned_inputs = (torch.ones(4, 3),)
    source_ref = weakref.ref(owned_inputs[0])
    first = ShapeKey(size=4, variant_label="layer0-stage0")
    second = ShapeKey(size=4, variant_label="layer0-stage1")
    with backend.capture_session(object()):
        backend.capture_one(
            first,
            lambda values=owned_inputs: values[0] + 1,
            capture_inputs=owned_inputs,
        )
        backend.capture_one(second, lambda: torch.full((4, 3), 9.0))
    del owned_inputs
    gc.collect()
    assert source_ref() is not None
    assert len(backend._graphs) == 2
    assert torch.equal(backend.replay(first, None), torch.full((4, 3), 2.0))
    assert torch.equal(backend.replay(second, None), torch.full((4, 3), 9.0))
    backend.cleanup()
    gc.collect()
    assert source_ref() is None


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_close_releases_storage_when_restore_fails_and_preserves_first_error(
    failure_type, cuda_boundary
):
    program, _, inputs, _ = _capture(cuda_boundary)
    failure = failure_type("first restore failed")
    first_guard = program._guards[0]

    def first_restore():
        cuda_boundary.events.append("first_restore_failed")
        raise failure

    def last_restore():
        cuda_boundary.events.append("last_restore_failed")
        raise RuntimeError("later restore failed")

    first_guard.restore = first_restore
    second_guard = _guard(cuda_boundary.events, "second_metadata")
    third_guard = _guard(cuda_boundary.events, "third_metadata")
    third_guard.restore = last_restore
    program._guards = (third_guard, second_guard, first_guard)
    with pytest.raises(failure_type) as caught:
        program.close()
    assert caught.value is failure
    assert program._backend is None
    assert program._stage_inputs == ()
    assert program._sentinel is None
    assert program._sentinel_source is None
    assert cuda_boundary.events[-3:] == [
        "first_restore_failed",
        ("second_metadata", "restore"),
        "last_restore_failed",
    ]
    gc.collect()
    assert all((ref() is None for ref in inputs))
    after_close = list(cuda_boundary.events)
    program.close()
    assert cuda_boundary.events == after_close
    assert cuda_boundary.graphs[0].replays == 0


@pytest.mark.parametrize("failure_site", ("capture", "restore"))
def test_shared_backend_failure_and_release_leave_other_bucket_replayable(
    cuda_boundary, failure_site
):
    driver = TorchRoleGraphDriver()
    first, _, inputs, pool1 = _capture(cuda_boundary, driver=driver, digest="shape1")
    failure = RuntimeError("new bucket failed")
    kwargs = {"failure" if failure_site == "capture" else "restore_failure": failure}
    with pytest.raises(RuntimeError, match="new bucket failed"):
        _capture(cuda_boundary, driver=driver, digest="bad", **kwargs)
    assert len(driver._backend._graphs) == 1
    second, _, inputs2, pool2 = _capture(cuda_boundary, driver=driver, digest="shape2")
    assert first._backend is second._backend is driver._backend
    assert pool1 is pool2
    assert driver._backend._pool is pool1
    assert torch.equal(_replay(first, 4.0)[0][0], torch.full((1, 3), 4.0))
    first.close()
    gc.collect()
    assert all(ref() is None for ref in inputs)
    assert len(driver._backend._graphs) == 1
    assert torch.equal(_replay(second, 9.0)[0][0], torch.full((1, 3), 9.0))
    second.close()
    gc.collect()
    assert all(ref() is None for ref in inputs2)
    assert not driver._backend._graphs
    assert not driver._backend._outputs
    assert not driver._backend._capture_inputs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))


@pytest.mark.parametrize("shape", [(), (4,), (5, 7)])
def test_sentinel_converts_only_first_value_including_strided_outputs(shape):
    from torch.utils._python_dispatch import TorchDispatchMode

    from sglang.srt.afd.role_graph import TorchRoleGraphProgram

    source = torch.full(shape, 3.5, dtype=torch.bfloat16)
    if source.ndim == 2:
        source = source.T
    converted_sizes = []

    class CastRecorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._to_copy.default:
                converted_sizes.append(args[0].numel())
            return func(*args, **(kwargs or {}))

    program = object.__new__(TorchRoleGraphProgram)
    program._torch = torch
    with CastRecorder():
        program._capture_sentinel(outputs=((source,),), device="cpu")
    assert program._sentinel.item() == 3.5
    assert converted_sizes == [1]
    assert program._sentinel_source is source
