"""CUDA-graph-compatible GPU-to-host layer progress signaling.

A layer publishes completion by enqueueing a four-byte device-to-host copy on
the current compute stream. CUDA graph capture records that copy as a graph
node, so it is replayed for every forward instead of only running during
capture. The host-side destination is pinned and can be polled without querying
the compute stream.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

import torch


class LayerProgress:
    """Monotonic per-forward layer completion counter.

    ``start_forward`` publishes a new generation before a forward is launched.
    During the forward, ``record`` publishes ``layer_id + 1`` after selected
    layers. Observing a value from an older generation returns zero.
    """

    def __init__(
        self,
        num_layers: int,
        device: torch.device,
        ring_layers: Iterable[int] | None = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if device.type != "cuda":
            raise ValueError(f"layer progress requires a CUDA device, got {device}")

        self.num_layers = int(num_layers)
        self.device = device
        # A power-of-two stride makes generation decoding cheap while supporting
        # models with more than the formerly hard-coded 1023 layers.
        self.stride = 1 << self.num_layers.bit_length()
        self._max_generation = (
            torch.iinfo(torch.int32).max - self.num_layers
        ) // self.stride
        if self._max_generation < 1:
            raise ValueError(
                f"too many layers for int32 progress encoding: {num_layers}"
            )
        self._base = torch.arange(self.num_layers + 1, dtype=torch.int32, device=device)
        self._progress = self._base.clone()
        self._host_value = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._generation = 0
        self._generation_lock = threading.Lock()

        selected = (
            set(range(self.num_layers))
            if ring_layers is None
            else {int(layer_id) for layer_id in ring_layers}
        )
        invalid = sorted(
            layer_id
            for layer_id in selected
            if layer_id < 0 or layer_id >= self.num_layers
        )
        if invalid:
            raise ValueError(f"ring layer ids are out of range: {invalid}")

        self._views: tuple[torch.Tensor | None, ...] = tuple(
            self._progress[layer_id + 1 : layer_id + 2]
            if layer_id in selected
            else None
            for layer_id in range(self.num_layers)
        )
        self.num_ring_points = len(selected)

    def start_forward(self, stream: torch.cuda.Stream | None = None) -> int:
        """Start a generation on the stream that will launch the forward."""
        with self._generation_lock:
            self._generation = self._generation % self._max_generation + 1
            generation = self._generation

        base = generation * self.stride
        if stream is None:
            self._progress.copy_(self._base + base, non_blocking=True)
        else:
            with torch.cuda.stream(stream):
                self._progress.copy_(self._base + base, non_blocking=True)
        self._host_value.fill_(base)
        return generation

    @property
    def generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def record(self, layer_id: int) -> None:
        """Enqueue a capturable completion update on the current CUDA stream."""
        if layer_id < 0 or layer_id >= self.num_layers:
            return
        view = self._views[layer_id]
        if view is not None:
            self._host_value.copy_(view, non_blocking=True)

    def completed_layers(self, generation: int) -> int:
        """Return completed global layers for ``generation`` or zero if stale."""
        value = int(self._host_value[0].item())
        base = generation * self.stride
        if value < base or value >= base + self.stride:
            return 0
        return min(value - base, self.num_layers)

    @property
    def host_value(self) -> torch.Tensor:
        """Pinned host counter, exposed for diagnostics and focused tests."""
        return self._host_value
