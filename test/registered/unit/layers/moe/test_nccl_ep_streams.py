"""Real CUDA stream/Graph dependencies with a one-rank external EP double."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.fake_ep import FakeHandle, dispatcher_environment
from nccl_ep_test.sglang_graph import backend_for

from sglang.srt.layers.moe.token_dispatcher.nccl_ep_stream import (
    destroy_nccl_ep_streams,
    get_nccl_ep_stream,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import (
    get_flags,
    get_resources,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def ep_case():
    with get_flags().moe.override(sbo_enabled=True), dispatcher_environment(
        capacity=8
    ) as environment:
        x = torch.ones(4, 2048, device="cuda", dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.full((4, 2), 0.5, device="cuda"),
            torch.tensor([[0, 1]] * 4, device="cuda"),
            None,
        )
        yield environment, x, topk


def test_communication_stream_requires_warmup_and_reuses_current_device(monkeypatch):
    try:
        with monkeypatch.context() as capture:
            capture.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            with pytest.raises(RuntimeError, match="during warmup"):
                get_nccl_ep_stream("cuda")
        stream = get_nccl_ep_stream("cuda")
        assert stream is get_nccl_ep_stream(
            torch.device("cuda", torch.cuda.current_device())
        )
    finally:
        destroy_nccl_ep_streams()


def test_failed_eager_handle_creation_releases_graph_transaction(monkeypatch, ep_case):
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        NcclEpGraphResources,
    )

    environment, x, topk = ep_case
    owner = NcclEpGraphResources(capacity=8, shutdown=lambda: None)
    monkeypatch.setitem(get_resources().buffers, "nccl_ep_graph_resources", owner)
    dispatcher = environment.dispatcher(layer_id=0)

    def fail_creation(**kwargs):
        raise RuntimeError("Injected handle allocation failure")

    with monkeypatch.context() as failure:
        failure.setattr(dispatcher.buffer.group, "create_handle", fail_creation)
        with pytest.raises(RuntimeError, match="Injected handle allocation"):
            dispatcher.dispatch_a(x, topk)
    assert dispatcher._eager_session is None
    assert dispatcher.buffer.borrower is None
    # A failed allocation must not strand the session lock or poison a retry.
    with owner.submission_session("eager"):
        dispatcher.dispatch(x, topk)
        actual = dispatcher.combine(dispatcher._active_buffer.recv_tokens)
    torch.testing.assert_close(actual, x, atol=0, rtol=0)


def test_native_submission_uses_side_stream_only_for_graph(monkeypatch, ep_case):
    streams = []
    original = FakeHandle.dispatch

    def observe(self, *args, **kwargs):
        streams.append((kwargs["stream"], torch.cuda.current_stream().cuda_stream))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FakeHandle, "dispatch", observe)
    environment, x, topk = ep_case
    dispatcher = environment.dispatcher(layer_id=0)
    main = torch.cuda.current_stream().cuda_stream

    def forward():
        dispatcher.dispatch(x, topk)
        return dispatcher.combine(dispatcher._active_buffer.recv_tokens)

    torch.testing.assert_close(forward(), x)
    assert streams and all(native == current == main for native, current in streams)
    streams.clear()
    backend = backend_for(environment.coordinator, 8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream), backend.capture_session(stream):
            backend.capture_one(ShapeKey(4), forward)
        torch.cuda.current_stream().wait_stream(stream)
        assert len(streams) == 3  # Two Graph warmups and the actual capture.
        assert all(
            native == current == dispatcher._comm.stream.cuda_stream
            for native, current in streams
        )
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
            nccl_ep_eager_session,
        )

        for _ in range(2):
            with backend.replay_session():
                torch.testing.assert_close(backend.replay(ShapeKey(4), None), x)
            streams.clear()
            with nccl_ep_eager_session():
                torch.testing.assert_close(forward(), x)
            assert streams and all(
                native == current == main for native, current in streams
            )
    finally:
        backend.cleanup()


@pytest.mark.parametrize(
    "case",
    [
        "graph_output_survives_next_layer_shared_scratch",
        "stream_switch_fences_input_writes_and_output_consumers",
        "recapture_closes_executables_before_persistent_resources",
        "failed_capture_can_start_a_fresh_generation",
        "eager_graph_alternation_keeps_groups_isolated",
        "concurrent_runner_submissions_fail_before_input_writes",
        "failed_combine_does_not_allow_another_transaction",
    ],
)
def test_overlap_stream_lifecycle(case):
    from registered.unit.layers.moe import test_nccl_ep_graph_lifecycle as checks

    with get_flags().moe.override(sbo_enabled=True):
        getattr(checks, f"test_{case}")()


def test_graph_completion_does_not_wait_for_independent_work(monkeypatch, ep_case):
    """Delayed fake EP checks captured dependencies, not native EP speed."""
    environment, x, topk = ep_case
    ready = torch.cuda.Event(enable_timing=True, external=True)
    blocked = torch.cuda.Event(enable_timing=True, external=True)
    original = FakeHandle.complete
    dispatcher = environment.dispatcher(layer_id=0)

    def delayed(handle, *args, **kwargs):
        if handle.pending == "dispatch":
            torch.cuda._sleep(2_000_000)
            ready.record()
        original(handle, *args, **kwargs)

    monkeypatch.setattr(FakeHandle, "complete", delayed)

    def forward():
        dispatcher.dispatch_a(x, topk)
        torch.cuda._sleep(40_000_000)
        blocked.record()
        dispatcher.dispatch_b()
        return dispatcher.combine(dispatcher._active_buffer.recv_tokens)

    backend = backend_for(environment.coordinator, 8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream), backend.capture_session(stream):
            backend.capture_one(ShapeKey(4), forward)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            with backend.replay_session():
                actual = backend.replay(ShapeKey(4), None)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, x, rtol=0, atol=0)
            assert ready.elapsed_time(blocked) > 0.5
    finally:
        backend.cleanup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
