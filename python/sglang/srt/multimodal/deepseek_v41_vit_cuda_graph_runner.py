"""Per-grid CUDA graph runner for the DeepSeek-V4.1 image encoder."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable, Tuple

import torch

logger = logging.getLogger(__name__)

# (patches, n_vit_h, n_vit_w) -> LLM span of the image.
EncodeFn = Callable[[torch.Tensor, int, int], torch.Tensor]
# (n_vit_h, n_vit_w, device) -> device tensors the encoder reads for that grid.
InputsFn = Callable[[int, int, torch.device], Any]


class DeepseekV41ViTCudaGraphRunner:
    """Replays vision tower + aligner + span assembly as one graph per grid.

    The encoder is replicated on every TP rank and host-bound (hundreds of
    small launches per image), so host jitter on one rank becomes collective
    wait on the others. A grid is captured on its second sighting, so one-off
    sizes stay eager; a failed capture stays eager for that grid.
    """

    def __init__(
        self,
        encode: EncodeFn,
        inputs: InputsFn,
        *,
        capacity: int = 16,
        min_hits: int = 2,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if min_hits <= 0:
            raise ValueError(f"min_hits must be positive, got {min_hits}")
        self.encode = encode
        self.inputs = inputs
        self.capacity = capacity
        self.min_hits = min_hits
        # (n_h, n_w) -> (graph, static_patches, static_span, inputs), LRU first.
        self.graphs: OrderedDict[Tuple[int, int], tuple] = OrderedDict()
        self.seen: OrderedDict[Tuple[int, int], int] = OrderedDict()
        self.failed_keys: set[Tuple[int, int]] = set()
        self._pool: Any = None

    def _record_hit(self, key: Tuple[int, int]) -> int:
        count = self.seen.pop(key, 0) + 1
        self.seen[key] = count
        if len(self.seen) > self.capacity * 8:
            self.seen.popitem(last=False)
        return count

    def run(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        key = (n_h, n_w)
        if torch.cuda.is_current_stream_capturing():
            return self.encode(patches, n_h, n_w)
        entry = self.graphs.get(key)
        if entry is None:
            if key in self.failed_keys or self._record_hit(key) < self.min_hits:
                return self.encode(patches, n_h, n_w)
            # No warm-up run: the earlier eager sightings warmed this grid.
            entry = self._capture(key, patches)
            if entry is None:
                return self.encode(patches, n_h, n_w)
        else:
            self.graphs.move_to_end(key)
        graph, static_patches, static_span, _ = entry
        static_patches.copy_(patches)
        graph.replay()
        # Spans live on in the multimodal embedding cache; the next replay
        # overwrites static_span.
        return static_span.clone()

    def _capture(self, key: Tuple[int, int], patches: torch.Tensor):
        n_h, n_w = key
        device = patches.device
        static_patches = patches.clone()
        # Built before capture so no host-to-device copy is recorded; the entry
        # keeps them alive for as long as the graph reads them.
        inputs = self.inputs(n_h, n_w, device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        reserved_before = torch.cuda.memory_reserved(device)
        graph = torch.cuda.CUDAGraph()
        try:
            # thread_local: other threads (e.g. the TP watchdog) keep running
            # CUDA calls during capture.
            with torch.cuda.graph(
                graph, pool=self._pool, capture_error_mode="thread_local"
            ):
                static_span = self.encode(static_patches, n_h, n_w)
        except Exception:
            self.failed_keys.add(key)
            logger.exception(
                "DeepSeek-V4.1 ViT CUDA graph capture failed for grid %s; "
                "using eager fallback",
                key,
            )
            return None
        entry = (graph, static_patches, static_span, inputs)
        self.graphs[key] = entry
        self.seen.pop(key, None)
        if len(self.graphs) > self.capacity:
            self.graphs.popitem(last=False)
        logger.info(
            "Captured DeepSeek-V4.1 ViT CUDA graph for grid %s (%d/%d), "
            "reserved_delta_mib=%.1f",
            key,
            len(self.graphs),
            self.capacity,
            (torch.cuda.memory_reserved(device) - reserved_before) / 2**20,
        )
        return entry
