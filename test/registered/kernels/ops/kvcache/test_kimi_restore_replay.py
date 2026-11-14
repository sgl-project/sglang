# SPDX-License-Identifier: Apache-2.0
"""Real CUDA restore events must precede a specialized captured state reader.

This compact fixture uses K3 local cache/state shapes, the native restore counter,
state router and binding replay hook. Pinned-host copies stand in for the host
transfer kernels; no attention backend, model, CuTe kernel or TP group is built.
"""

import ctypes
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA streams and host callbacks",
)


class _CopyGate:
    """Hold one copy stream without GPU sleeps or a timing-dependent race."""

    def __init__(self, stream):
        from cuda.bindings import runtime

        self.release = threading.Event()
        self.entered = threading.Event()
        self.timed_out = threading.Event()

        def wait(_):
            # Host functions must not call CUDA APIs. Event.wait releases the
            # GIL, so the main thread can enqueue replay and release the gate.
            self.entered.set()
            if not self.release.wait(timeout=10):
                self.timed_out.set()

        # Both wrappers must outlive the stream work, not only the launch call.
        self.callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(wait)
        self.callback_handle = runtime.cudaHostFn_t(
            ctypes.cast(self.callback, ctypes.c_void_p).value
        )
        (error,) = runtime.cudaLaunchHostFunc(
            stream.cuda_stream, self.callback_handle, 0
        )
        assert error == runtime.cudaError_t.cudaSuccess, error


def test_replay_joins_the_current_restore_slot_before_reading_state(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia.binding import MegakernelBinding
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter

    # One MLA and one KDA model layer, with the real K3 per-rank state geometry.
    mla = torch.zeros((128, 1, 576), dtype=torch.uint8, device="cuda")
    conv = torch.zeros((4, 3, 4608), dtype=torch.bfloat16, device="cuda")
    temporal = torch.zeros((4, 12, 128, 128), dtype=torch.float32, device="cuda")
    counter = LayerDoneCounter(2)
    pool = SimpleNamespace(start_layer=0, layer_transfer_counter=counter)
    state = SimpleNamespace(conv=[conv], temporal=temporal)
    router = KVLayerRouter(pool, sub_pools=(), views={}, layers={})
    router.bind_part(sub_pool="state", part=STATE_PART, view=state, indices={1: 0})
    router.load_parts_layer_wise()

    def current_mla():
        counter.wait_until(0)
        return mla

    def current_state():
        return router.read(1, STATE_PART).view

    binding = MegakernelBinding(model=torch.nn.Module())
    # Pool/backend construction and sealing are covered by the binding contract
    # tests. Seed those sealed getters here to isolate the actual CUDA join:
    # prepare_replay -> assert_live -> native counter/router -> current stream.
    binding._hybrid_sources = object()
    binding._seal("mla_cache[0]", mla, current_mla)
    binding._seal("kda_conv[1]", conv, lambda: current_state().conv[0])
    binding._seal("kda_temporal[1]", temporal, lambda: current_state().temporal)

    sources = (mla[64], conv[2], temporal[2])
    outputs = tuple(torch.empty_like(source) for source in sources)
    observed = tuple(
        torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
        for source in sources
    )
    host = tuple(torch.empty_like(value, pin_memory=True) for value in observed)
    reader, copier = torch.cuda.Stream(), torch.cuda.Stream()
    reader.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=reader):
        for output, source in zip(outputs, sources, strict=True):
            output.copy_(source)
    # All allocations and capture are complete before a host callback can block.
    torch.cuda.synchronize()

    waits = []
    real_wait_event = torch.cuda.Stream.wait_event

    def observed_wait_event(stream, event):
        waits.append((stream.cuda_stream, event))
        return real_wait_event(stream, event)

    # Observe the dependency submissions without replacing their CUDA behavior.
    # Data alone could pass if an accidentally missing hook races a fast copy.
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", observed_wait_event)

    def replay_after_restore(*, join, value):
        for source in sources:
            source.zero_()
        for index, value_host in enumerate(host):
            value_host.fill_(value + index)
        torch.cuda.synchronize()
        slot = counter.update_producer()
        gate = _CopyGate(copier)
        try:
            with torch.cuda.stream(copier):
                sources[0].copy_(host[0], non_blocking=True)
                counter.events[slot].complete(0)
                for destination, value_host in zip(sources[1:], host[1:], strict=True):
                    destination.copy_(value_host, non_blocking=True)
                counter.events[slot].complete(1)
            assert gate.entered.wait(timeout=5), "copy callback did not start"
            counter.set_consumer(slot)
            done = torch.cuda.Event()
            waits.clear()
            with torch.cuda.stream(reader):
                if join:
                    binding.prepare_replay()
                graph.replay()
                for destination, output in zip(observed, outputs, strict=True):
                    destination.copy_(output, non_blocking=True)
                done.record()
            load_events = counter.events[slot].load_events
            expected_waits = (
                [(reader.cuda_stream, load_events[layer]) for layer in (0, 1, 1)]
                if join
                else []
            )
            assert waits == expected_waits, (
                "replay did not join the current restore slot"
            )
            if join:
                gate.release.set()
            # In the negative control the producer is still host-gated: this
            # graph necessarily reads the old zeros, independent of GPU speed.
            done.synchronize()
            for index, actual in enumerate(observed):
                expected = value + index if join else 0
                assert torch.equal(actual, torch.full_like(actual, expected))
            return slot
        finally:
            gate.release.set()
            copier.synchronize()
            assert not gate.timed_out.is_set(), "copy gate exceeded its timeout"

    replay_after_restore(join=False, value=11)
    counter.reset()
    # Exercise every actual event set and wrap around to a reused consumer slot.
    slots = [replay_after_restore(join=True, value=value) for value in (21, 31, 41, 51)]
    assert slots == [0, 1, 2, 0]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
