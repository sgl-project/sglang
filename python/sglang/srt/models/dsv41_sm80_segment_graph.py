"""Opt-in, lazy CUDA graphs for a *pure*, single-token V4.1 FFN segment.

Integration contract (the caller, not this helper, owns model/forward state):
* Create one helper per layer after loading weights, or lazily on first use.
* All participating ranks must use identical eligibility, layer/capture order,
  tensor signatures and exactly three warmups. No rank-local fallback is safe.
* ``fn(h, ids, ids_global)`` returns one Tensor and has no model/cache mutations
  (custom-all-reduce scratch is allowed). All changing tensor inputs must be
  explicit arguments; captured Python metadata/weights must remain invariant.
* No enclosing CUDA OR distributed graph_capture context; these communicator
  contexts are not nestable. This is not a whole-model graph runner.
* Consume the returned stable buffer before the next call to this layer. Do not
  call concurrently. Optional shared pools require globally ordered, sequential
  capture/replay, with outputs consumed before later graphs reuse pool memory.

The distributed context surrounds both warmup and capture, and exits before
replay to register custom-AR buffers. Warmup outputs may be zeros and are never
returned. No CUDA allocation, stream, or pool is created by the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs

TensorFn = Callable[
    [torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], torch.Tensor
]


@dataclass
class _CapturedSegment:
    graph: object
    inputs: tuple
    output: torch.Tensor
    stream: object
    # Retain the first closure and its captured immutable forward metadata /
    # tensor owners for the complete graph lifetime. Dynamic tensors must still
    # be explicit args; keeping a closure does not refresh changing metadata.
    fn: TensorFn


def _signature(tensor):
    if tensor is None:
        return None
    return (tuple(tensor.shape), tensor.dtype, tensor.device)


class Dsv41Sm80FfnSegmentGraph:
    """Per-layer lazy graph cache; ordinary calls remain eager by default.

    ``is_decode`` must be derived from caller ForwardMode, not tensor shape.
    The gate additionally requires h.shape == (1, hidden), both exact env flags,
    model_type == 'deepseek_v41', and the input's *actual* CUDA device SM80.
    Capture/replay failures permanently poison this helper and propagate; the
    serving process should abort rather than continue unmatched collectives.
    """

    WARMUP_COUNT = 3

    def __init__(self, model_type: str, *, pool=None):
        self.model_type = model_type
        self.pool = pool
        self._cache = {}
        self._failure = None

    def _eligible(self, h, input_ids, input_ids_global, is_decode):
        if not (
            envs.SGLANG_DSV41_SM80_FFN_GRAPH.get() == "1"
            and envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() == "1"
            and self.model_type == "deepseek_v41"
            and is_decode
            and h.ndim == 2
            and h.shape[0] == 1
            and h.device.type == "cuda"
            and torch.version.hip is None
        ):
            return False
        if any(
            t is not None and t.device != h.device
            for t in (input_ids, input_ids_global)
        ):
            return False
        return torch.cuda.get_device_capability(h.device) == (8, 0)

    def __call__(
        self,
        fn: TensorFn,
        h: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        if self._failure is not None:
            raise RuntimeError(
                "FFN segment graph previously failed; abort serving"
            ) from self._failure
        with torch.no_grad():
            if not self._eligible(h, input_ids, input_ids_global, is_decode):
                return fn(h, input_ids, input_ids_global)
            # Never nest inside a whole-model CUDA graph. Distributed-only outer
            # contexts cannot be detected by CUDA; excluding those is a caller
            # obligation (see module contract).
            with torch.cuda.device(h.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("FFN segment graph cannot nest CUDA captures")
                args = (h, input_ids, input_ids_global)
                key = tuple(_signature(t) for t in args)
                try:
                    segment = self._cache.get(key)
                    if segment is None:
                        segment = self._capture(fn, args)
                        self._cache[key] = segment
                    # Refresh *every* tensor, including hash-routing token IDs,
                    # on the caller stream before every replay (also the first).
                    for static, dynamic in zip(segment.inputs, args):
                        if static is not None:
                            static.copy_(dynamic)
                    segment.graph.replay()
                    return segment.output
                except Exception as exc:
                    self._failure = exc
                    raise RuntimeError(
                        "FFN segment CUDA graph failed; no rank-local eager fallback is safe"
                    ) from exc

    def _capture(self, fn, args):
        # Import lazily: disabled use does not initialize distributed machinery.
        from sglang.srt.distributed.parallel_state import graph_capture

        device = args[0].device
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        static = tuple(t.clone() if t is not None else None for t in args)
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        with graph_capture(stream=stream) as context:
            # This API accepts stream=, NOT a positional device. It selects the
            # same stream for each communicator and registers IPC on context exit.
            with torch.cuda.stream(context.stream):
                for _ in range(self.WARMUP_COUNT):
                    fn(*static)
                context.stream.synchronize()
                with torch.cuda.graph(graph, pool=self.pool, stream=context.stream):
                    output = fn(*static)
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("FFN segment closure must return a Tensor")
        # Includes all capture-stream work; replay is launched on caller stream.
        current.wait_stream(stream)
        return _CapturedSegment(graph, static, output, stream, fn)
