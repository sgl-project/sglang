"""Glue-graph capture of the per-replay attention-metadata prep.

``decode_cuda_graph_runner.load_batch`` runs
``attn_backend.init_forward_metadata_out_graph(fb_view)`` eagerly on every
replay. At bs=1 spec decode this is an "op soup": dozens of tiny tensor ops
whose HOST dispatch cost dominates the inter-phase seam, while every device
input/output lives at a stable address — the replay fb view hands backends the
runner's static buffers, and pool tensors are persistent. Capturing the op
sequence once per replay key collapses the per-step host cost to a single
graph launch.

Correctness contract:

- The caller only routes here when the replay is padding-free
  (raw_bs == padded bs) and TBO / pdmux / LoRA are off, so every
  Python-visible branch inside the backends is constant per key.
- Python side effects (each backend's ``forward_metadata`` object) are
  snapshotted at capture time and re-installed on every replay; the graph
  replays only the device ops that refresh the tensors those objects point to.
- ``NUM_WARMUP`` eager runs precede capture so triton JIT compile / autotune
  happen outside capture.
- Any capture failure (e.g. a backend syncing or reading host values inside
  its prep) permanently disables the glue graph and falls back to eager.
- Backends whose prep computes values on the HOST each replay (e.g. the
  DFlash-family host-fed fast verify plans) must never be glued: capture
  records only device ops, so the host-written plan inputs would replay
  frozen at their capture-time values. Note the failure is SILENT — capture
  succeeds, outputs stay correct, only accept length collapses. Callers must
  gate such configurations off before routing here
  (``decode_cuda_graph_runner`` force-disables the glue for DFlash-family
  spec).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class MetadataGlueGraph:
    NUM_WARMUP = 2

    def __init__(self, device):
        self.device = device
        self.disabled = False
        # Since the first warm-up: prep steps served by a graph replay, and steps
        # that ran eagerly after it (capture failure, disabled, or never routed here).
        self.replays = 0
        self.eager = 0
        self._states: Dict[Any, dict] = {}
        self._capture_stream = None

    def reset(self):
        """Drop captured graphs (call when the runner recaptures its graphs —
        static buffers and backend state may have been rebuilt)."""
        self._states.clear()

    def note_not_routed(self) -> None:
        """A replay the caller prepared eagerly without calling ``run`` (padding,
        TBO, pdmux, LoRA, disabled)."""
        if self._states or self.disabled:
            self.eager += 1

    @staticmethod
    def _leaves(attn_backend) -> List[Any]:
        backends = [attn_backend]
        if attn_backend.attn_backend_list is not None:
            backends.extend(attn_backend.attn_backend_list)
        return backends

    def is_captured(self, key) -> bool:
        st = self._states.get(key)
        return not self.disabled and st is not None and st["graph"] is not None

    def reinstall(self, key) -> None:
        """Point each backend at the metadata ``key``'s graph writes."""
        for backend, metadata in self._states[key]["meta"]:
            backend.forward_metadata = metadata

    def replay(self, key) -> None:
        """Replay ``key``'s captured prep on the current stream (a staging side stream
        included); the caller reinstalls its metadata before the forward."""
        self.reinstall(key)
        self._states[key]["graph"].replay()
        self.replays += 1

    def run(self, attn_backend, fb_view, key) -> None:
        """Run ``init_forward_metadata_out_graph`` for this replay, through the
        captured glue graph once it is ready."""
        st = self._states.get(key)
        if st is None:
            st = {"warmups": 0, "graph": None, "meta": None}
            self._states[key] = st

        if st["graph"] is not None:
            for backend, metadata in st["meta"]:
                backend.forward_metadata = metadata
            st["graph"].replay()
            self.replays += 1
            return

        if st["warmups"] < self.NUM_WARMUP:
            st["warmups"] += 1
            attn_backend.init_forward_metadata_out_graph(fb_view)
            return

        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        try:
            # captured while serving: other threads may use CUDA meanwhile
            with torch.cuda.graph(
                graph, stream=self._capture_stream, capture_error_mode="thread_local"
            ):
                attn_backend.init_forward_metadata_out_graph(fb_view)
        except Exception:
            logger.warning(
                "Metadata glue-graph capture failed for key %s; falling back "
                "to eager metadata prep permanently.",
                key,
                exc_info=True,
            )
            self.disabled = True
            self.eager += 1
            # Ops under a failed capture were recorded, not executed — run
            # this step's prep for real.
            attn_backend.init_forward_metadata_out_graph(fb_view)
            return

        st["meta"] = [(b, b.forward_metadata) for b in self._leaves(attn_backend)]
        st["graph"] = graph
        # Capture records without executing; replay once to do this step's prep.
        graph.replay()
