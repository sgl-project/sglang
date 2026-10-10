"""Glue-graph capture of the per-replay attention-metadata prep.

``decode_cuda_graph_runner.load_batch`` runs
``attn_backend.init_forward_metadata_out_graph(fb_view)`` eagerly on every
replay. At bs=1 spec decode this is an "op soup": dozens of tiny tensor ops
whose HOST dispatch cost dominates the inter-phase seam, while every device
input/output lives at a stable address — the replay fb view hands backends the
runner's static buffers, and pool tensors are persistent. Capturing the op
sequence once per replay key collapses the per-step host cost to a single
graph launch.

Two-phase split (DFlash-family host-fed plans):
Some backends (DFlash TARGET_VERIFY) recompute plan inputs on the host every
replay. Capturing those host writes would freeze them at capture-time values
(silent accept-length collapse). Those backends implement
``prepare_host_metadata`` / ``apply_device_metadata``:

- ``prepare_host_metadata`` runs **eagerly every replay** (not captured) and
  writes current-step plan arrays into pointer-stable static/pinned buffers.
- ``apply_device_metadata`` issues only device ops that read those buffers
  and is what the glue graph captures.

Backends that do not implement the split keep the single-phase
``init_forward_metadata_out_graph`` path.

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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MetadataGlueGraph:
    NUM_WARMUP = 2

    def __init__(self, device):
        self.device = device
        self.disabled = False
        self._states: Dict[Any, dict] = {}
        self._capture_stream = None

    def reset(self):
        """Drop captured graphs (call when the runner recaptures its graphs —
        static buffers and backend state may have been rebuilt)."""
        self._states.clear()

    @staticmethod
    def _leaves(attn_backend) -> List[Any]:
        backends = [attn_backend]
        if attn_backend.attn_backend_list is not None:
            backends.extend(attn_backend.attn_backend_list)
        return backends

    def _run_prep(self, attn_backend, fb_view, host_inputs: Optional[dict]) -> None:
        if host_inputs is not None:
            attn_backend.apply_device_metadata(fb_view, host_inputs)
        else:
            attn_backend.init_forward_metadata_out_graph(fb_view)

    def run(self, attn_backend, fb_view, key) -> None:
        """Run metadata prep for this replay, through the captured glue graph
        once it is ready.

        Host-eager phase (``prepare_host_metadata``) always runs outside the
        graph so DFlash-style plan inputs stay fresh. The device phase is
        captured per key.
        """
        host_inputs = attn_backend.prepare_host_metadata(fb_view)

        st = self._states.get(key)
        if st is None:
            st = {"warmups": 0, "graph": None, "meta": None}
            self._states[key] = st

        if st["graph"] is not None:
            for backend, metadata in st["meta"]:
                backend.forward_metadata = metadata
            st["graph"].replay()
            return

        if st["warmups"] < self.NUM_WARMUP:
            st["warmups"] += 1
            self._run_prep(attn_backend, fb_view, host_inputs)
            return

        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=self._capture_stream):
                self._run_prep(attn_backend, fb_view, host_inputs)
        except Exception:
            logger.warning(
                "Metadata glue-graph capture failed for key %s; falling back "
                "to eager metadata prep permanently.",
                key,
                exc_info=True,
            )
            self.disabled = True
            # Ops under a failed capture were recorded, not executed — run
            # this step's prep for real.
            self._run_prep(attn_backend, fb_view, host_inputs)
            return

        st["meta"] = [(b, b.forward_metadata) for b in self._leaves(attn_backend)]
        st["graph"] = graph
        # Capture records without executing; replay once to do this step's prep.
        graph.replay()
