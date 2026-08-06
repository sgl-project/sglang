"""Unit test for FullCudaGraphBackend.capture_one ordering — the device
synchronize + TP-group barrier must run between the last warmup forward and
CUDAGraph() construction (PR #33795 / #33356 regression guard).

Bug (black-box): during DSpark compact ragged-verify CUDA-graph capture,
async TVM/DeepGEMM JIT compilation triggered by a new non-uniform
verify_lens shape can still be issuing CUDA driver calls (cuModuleLoadData)
when `torch.cuda.CUDAGraph()` enters the stream-capture region, yielding
CUDA_ERROR_ILLEGAL_ADDRESS / cudaErrorStreamCaptureUnsupported at startup.
The fix inserts a synchronize() + barrier() between the last warmup forward
and graph construction so no JIT work is in flight inside the capture
context. This case pins that ordering invariant: removing or re-ordering
the post-warmup sync reopens the race and turns this test red.

Deterministic: pure call-order check via mocks — no GPU, no real JIT race.
"""

from unittest.mock import MagicMock

import pytest

from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _backend(call_log):
    """FullCudaGraphBackend with fully mocked collaborators.

    Built via __new__ + attribute stubs (same pattern as the runner unit
    tests) so no GPU initialization runs on CPU CI.
    """
    device_module = MagicMock()
    device_module.synchronize.side_effect = lambda: call_log.append("synchronize")

    # graph() returns a context manager for the capture body.
    capture_cm = MagicMock()
    capture_cm.__enter__ = MagicMock(return_value=None)
    capture_cm.__exit__ = MagicMock(return_value=False)
    device_module.graph.return_value = capture_cm

    tp_group = MagicMock()
    tp_group.barrier.side_effect = lambda: call_log.append("barrier")

    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._pool = None
    backend._capture_stream = MagicMock()
    backend._device_module = device_module
    backend._tp_group = tp_group
    backend._memory_saver_adapter = None
    return backend


def test_capture_one_syncs_and_barriers_before_cudagraph(monkeypatch):
    call_log = []

    def forward_fn():
        call_log.append("forward")

    class FakeCUDAGraph:
        def __init__(self):
            call_log.append("cudagraph_create")

    monkeypatch.setattr("torch.cuda.CUDAGraph", FakeCUDAGraph)

    _backend(call_log).capture_one(
        shape_key="test_shape",
        forward_fn=forward_fn,
        capture_inputs=None,
        post_warmup_hook=None,
    )

    # 2 warmup forwards + 1 in-capture forward.
    assert call_log.count("forward") == 3, f"Unexpected forwards: {call_log}"
    assert call_log.count("cudagraph_create") == 1, f"Unexpected graphs: {call_log}"

    cudagraph_idx = call_log.index("cudagraph_create")

    # Last warmup forward = last forward strictly before CUDAGraph().
    last_warmup_forward_idx = max(
        i for i, v in enumerate(call_log) if v == "forward" and i < cudagraph_idx
    )

    # KEY: a synchronize() must appear between the last warmup forward and
    # CUDAGraph() construction. Removing it reopens the JIT-in-capture race.
    sync_between = [
        i
        for i, v in enumerate(call_log)
        if v == "synchronize" and last_warmup_forward_idx < i < cudagraph_idx
    ]
    assert len(sync_between) >= 1, (
        "No synchronize() between last warmup forward and CUDAGraph() "
        f"construction: {call_log}"
    )

    # KEY: the TP-group barrier() must sit in the same window.
    barrier_between = [
        i
        for i, v in enumerate(call_log)
        if v == "barrier" and last_warmup_forward_idx < i < cudagraph_idx
    ]
    assert len(barrier_between) >= 1, (
        "No barrier() between last warmup forward and CUDAGraph() "
        f"construction: {call_log}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
