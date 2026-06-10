# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Breakable CUDA graph (BCG) runner for diffusion DiT transformers.

Captures a DiT ``transformer.forward`` as a sequence of
``torch.cuda.CUDAGraph`` segments split at the attention modules (see
``layers/attention/layer.py``), so the linear/norm/FFN math of each block runs
from a static CUDA graph while sequence-parallel all-to-all, varlen packing,
and dynamic/sparse attention kernels run eagerly between segments.

Why this is simpler than the LLM BCG runner (``sglang.srt``): within a single
generate request the DiT input shapes are fixed across all denoising steps, so
we capture lazily on first use (keyed by the tensor-input signature) and replay
for every subsequent step. Every tensor input — including tensors nested inside
list/tuple/dict kwargs such as Wan's ``encoder_hidden_states`` prompt-embed list
— is copied into a persistent static buffer before each replay, so per-step
latents/timestep AND per-CFG-branch conditioning are refreshed correctly. The
attention break points re-run eagerly and re-read the live forward context, so
per-timestep attention metadata (e.g. sparse-video-attention masks) is also
picked up correctly on replay.

This runner shares the model-agnostic BCG primitives in
:mod:`sglang.srt.breakable_cuda_graph` with the LLM runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang.srt.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)

logger = logging.getLogger(__name__)


def _map_tensors(obj, fn):
    """Rebuild ``obj`` applying ``fn`` to every tensor leaf, recursing into
    list/tuple/dict containers; everything else passes through unchanged."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [_map_tensors(o, fn) for o in obj]
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    return obj


def _flatten_tensors(obj, out: list):
    """Depth-first collect every tensor leaf into ``out`` (deterministic order:
    dicts traversed in sorted-key order to match across calls)."""
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _flatten_tensors(o, out)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _flatten_tensors(obj[k], out)


def _flatten_kwargs(kwargs: dict[str, Any]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for name in sorted(kwargs):
        _flatten_tensors(kwargs[name], out)
    return out


@dataclass
class _CaptureEntry:
    graph: BreakableCUDAGraph
    # full captured kwargs with persistent static buffers at every tensor leaf
    static_kwargs: dict[str, Any]
    # the same static buffers, flattened in _flatten_kwargs order (replay copies
    # live tensors into these positionally)
    static_leaves: list[torch.Tensor]
    output: Any
    num_segments: int


class DiffusionBreakableCudaGraphRunner:
    """Lazily capture and replay a diffusion DiT ``transformer`` with BCG.

    Usage::

        runner = DiffusionBreakableCudaGraphRunner(transformer, device)
        noise_pred = runner(hidden_states=..., timestep=..., ...)

    Falls back to a plain eager call (and disables itself for the offending
    signature) if capture fails, so a model/shape the runner cannot handle
    never breaks generation — it just runs eagerly.
    """

    def __init__(
        self,
        transformer: nn.Module,
        device: torch.device,
        pool=None,
    ) -> None:
        self.transformer = transformer
        self.device = device
        self.device_module = torch.get_device_module(device)
        # One shared mempool across all captured graphs/segments so per-block
        # intermediates can be reclaimed and weak-ref'd safely.
        self._pool = pool if pool is not None else self.device_module.graph_pool_handle()
        self._capture_stream = self.device_module.Stream(device=device)
        self.entries: dict[tuple, _CaptureEntry] = {}
        # Signatures we have given up capturing (capture raised); run eager.
        self._blocked: set[tuple] = set()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, **kwargs) -> Any:
        key = self._signature(kwargs)
        if key in self._blocked:
            return self.transformer(**kwargs)

        entry = self.entries.get(key)
        if entry is None:
            try:
                entry = self._capture(kwargs, key)
            except Exception as e:  # noqa: BLE001 — never break generation on capture
                logger.warning(
                    "[Diffusion BCG] capture failed for signature %s (%s); "
                    "falling back to eager for this signature.",
                    key,
                    e,
                )
                self._blocked.add(key)
                return self.transformer(**kwargs)
            self.entries[key] = entry
        return self._replay(entry, kwargs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _signature(self, kwargs: dict[str, Any]) -> tuple:
        """Capture key: shape+dtype of every tensor leaf (including tensors
        nested in list/tuple/dict kwargs), in deterministic order. Non-tensor
        leaves are assumed structurally constant within a request and are baked
        into the captured graph."""
        return tuple(
            (tuple(t.shape), str(t.dtype)) for t in _flatten_kwargs(kwargs)
        )

    def _capture(self, kwargs: dict[str, Any], key: tuple) -> _CaptureEntry:
        # Persistent static buffers at every tensor leaf; bake non-tensors.
        def _to_static(t: torch.Tensor) -> torch.Tensor:
            # Static buffers live on the capture device. A CPU input (e.g. a
            # scalar timestep/sigma or an index tensor built on the host)
            # would otherwise force a CPU->CUDA copy inside the captured
            # region, which is illegal; place its buffer on the device so the
            # only host->device copy happens here, before capture, and replay
            # is device-to-device.
            if t.device.type == "cpu":
                buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            else:
                buf = torch.empty_like(t)
            buf.copy_(t)
            return buf

        static_kwargs = {
            name: _map_tensors(v, _to_static) for name, v in kwargs.items()
        }
        static_leaves = _flatten_kwargs(static_kwargs)

        # Warm up on the capture stream so cuBLAS/cuDNN/Triton workspaces and
        # any lazy JIT are materialized before capture (mirrors the LLM runner
        # and torch.cuda.make_graphed_callables).
        self.device_module.synchronize()
        with self.device_module.stream(self._capture_stream):
            for _ in range(2):
                self.transformer(**static_kwargs)
        self._capture_stream.synchronize()
        self.device_module.synchronize()

        graph = BreakableCUDAGraph()
        with enable_breakable_cuda_graph():
            with BreakableCUDAGraphCapture(
                cuda_graph=graph, pool=self._pool, stream=self._capture_stream
            ):
                output = self.transformer(**static_kwargs)
        self.device_module.synchronize()

        logger.info(
            "[Diffusion BCG] captured %d segment(s), %d tensor input(s) for "
            "signature %s",
            len(graph._segments),
            len(static_leaves),
            key,
        )
        return _CaptureEntry(
            graph=graph,
            static_kwargs=static_kwargs,
            static_leaves=static_leaves,
            output=output,
            num_segments=len(graph._segments),
        )

    def _replay(self, entry: _CaptureEntry, kwargs: dict[str, Any]) -> Any:
        live_leaves = _flatten_kwargs(kwargs)
        if len(live_leaves) != len(entry.static_leaves):
            # Structure changed under a matching shape key — should not happen;
            # fall back to eager rather than copy mismatched buffers.
            return self.transformer(**kwargs)
        for buf, live in zip(entry.static_leaves, live_leaves):
            buf.copy_(live, non_blocking=True)
        entry.graph.replay()
        # Clone so the caller can hold the result across the next replay / the
        # other CFG branch (which shares this static output buffer when shapes
        # match). The clone is one cheap DtoD copy relative to the full DiT.
        return _clone_output(entry.output)


def _clone_output(out: Any) -> Any:
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, tuple):
        return tuple(_clone_output(o) for o in out)
    if isinstance(out, list):
        return [_clone_output(o) for o in out]
    return out
