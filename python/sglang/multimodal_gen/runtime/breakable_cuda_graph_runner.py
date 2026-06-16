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
import os
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[Diffusion BCG] ignoring invalid integer %s=%r", name, raw)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[Diffusion BCG] ignoring invalid float %s=%r", name, raw)
        return default


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


def _signature_leaf(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return ("tensor", tuple(obj.shape), str(obj.dtype))
    if isinstance(obj, tuple):
        return ("tuple", tuple(_signature_leaf(o) for o in obj))
    if isinstance(obj, list):
        return ("list", tuple(_signature_leaf(o) for o in obj))
    if isinstance(obj, dict):
        return (
            "dict",
            tuple((k, _signature_leaf(obj[k])) for k in sorted(obj)),
        )
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return ("const", obj)
    return ("object", type(obj).__module__, type(obj).__qualname__, id(obj))


def _signature_kwargs(kwargs: dict[str, Any]) -> tuple:
    return tuple((name, _signature_leaf(kwargs[name])) for name in sorted(kwargs))


def _signature_summary_leaf(sig: Any, *, depth: int = 0) -> Any:
    if not isinstance(sig, tuple) or not sig:
        return sig

    tag = sig[0]
    if tag == "tensor":
        return sig
    if tag == "const":
        value = sig[1]
        if isinstance(value, str) and len(value) > 64:
            value = value[:61] + "..."
        return (tag, value)
    if tag == "object":
        return sig[:3]
    if depth >= 2:
        return (tag, "...")
    if tag in ("tuple", "list"):
        items = sig[1]
        preview = tuple(
            _signature_summary_leaf(item, depth=depth + 1) for item in items[:4]
        )
        if len(items) > 4:
            preview += (("...", len(items) - 4),)
        return (tag, len(items), preview)
    if tag == "dict":
        items = sig[1]
        preview = tuple(
            (key, _signature_summary_leaf(value, depth=depth + 1))
            for key, value in items[:4]
        )
        if len(items) > 4:
            preview += (("...", len(items) - 4),)
        return (tag, len(items), preview)
    return sig


def _signature_summary(key: tuple) -> tuple:
    return tuple(
        (name, _signature_summary_leaf(value)) for name, value in key[:16]
    ) + ((("...", len(key) - 16),) if len(key) > 16 else ())


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


class _CaptureRejected(RuntimeError):
    pass


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
        self._disabled_reason: str | None = None
        self.max_entries = max(0, _env_int("SGLANG_DIFFUSION_BCG_MAX_ENTRIES", 8))
        self.max_segments = max(0, _env_int("SGLANG_DIFFUSION_BCG_MAX_SEGMENTS", 128))
        max_reserved_gb = max(
            0.0, _env_float("SGLANG_DIFFUSION_BCG_MAX_RESERVED_GB", 12.0)
        )
        self.max_reserved_bytes = int(max_reserved_gb * (1024**3))
        self._reserved_baseline_bytes = self._memory_reserved()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, **kwargs) -> Any:
        if self._disabled_reason is not None:
            return self.transformer(**kwargs)

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
                    _signature_summary(key),
                    e,
                )
                self._blocked.add(key)
                return self.transformer(**kwargs)
            self.entries[key] = entry
            self._evict_entries_if_needed()
        return self._replay(entry, kwargs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _signature(self, kwargs: dict[str, Any]) -> tuple:
        """Capture key for tensor leaves and non-tensor control values.

        Tensor leaves are keyed by shape+dtype so their values can change per
        replay. Non-tensor leaves are baked into the captured Python control
        flow, so simple constants must be part of the key as well. Mutable
        objects are keyed by identity to avoid replaying a graph whose eager
        break points still reference a previous request's state object.
        """
        return _signature_kwargs(kwargs)

    def _memory_reserved(self) -> int:
        memory_reserved = getattr(self.device_module, "memory_reserved", None)
        if not callable(memory_reserved):
            return 0
        try:
            return int(memory_reserved(self.device))
        except TypeError:
            return int(memory_reserved())

    def _empty_cache(self) -> None:
        empty_cache = getattr(self.device_module, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    @staticmethod
    def _drop_entry(entry: _CaptureEntry) -> None:
        entry.graph._break_fns.clear()
        entry.graph._segments.clear()
        entry.static_kwargs.clear()
        entry.static_leaves.clear()
        entry.output = None

    def reset(self, *, disabled_reason: str | None = None) -> None:
        for entry in self.entries.values():
            self._drop_entry(entry)
        self.entries.clear()
        self._blocked.clear()
        self._pool = None
        self._empty_cache()
        if disabled_reason is not None:
            self._disabled_reason = disabled_reason

    def _capture_limit_reason(self, entry: _CaptureEntry) -> str | None:
        if self.max_segments and entry.num_segments > self.max_segments:
            return (
                f"captured {entry.num_segments} segments, above "
                f"SGLANG_DIFFUSION_BCG_MAX_SEGMENTS={self.max_segments}"
            )
        if self.max_reserved_bytes:
            reserved_delta = self._memory_reserved() - self._reserved_baseline_bytes
            if reserved_delta > self.max_reserved_bytes:
                return (
                    f"reserved graph memory grew by {reserved_delta / (1024**3):.2f}GiB, "
                    "above SGLANG_DIFFUSION_BCG_MAX_RESERVED_GB="
                    f"{self.max_reserved_bytes / (1024**3):.2f}"
                )
        return None

    def _evict_entries_if_needed(self) -> None:
        if not self.max_entries:
            return
        while len(self.entries) > self.max_entries:
            evicted_key = next(iter(self.entries))
            entry = self.entries.pop(evicted_key)
            self._drop_entry(entry)
            logger.info(
                "[Diffusion BCG] evicted oldest capture for signature %s "
                "(SGLANG_DIFFUSION_BCG_MAX_ENTRIES=%d)",
                _signature_summary(evicted_key),
                self.max_entries,
            )
        self._empty_cache()

    def _capture(self, kwargs: dict[str, Any], key: tuple) -> _CaptureEntry:
        if self._pool is None:
            self._pool = self.device_module.graph_pool_handle()

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
            _signature_summary(key),
        )
        entry = _CaptureEntry(
            graph=graph,
            static_kwargs=static_kwargs,
            static_leaves=static_leaves,
            output=output,
            num_segments=len(graph._segments),
        )
        limit_reason = self._capture_limit_reason(entry)
        if limit_reason is not None:
            self._drop_entry(entry)
            self.reset(disabled_reason=limit_reason)
            raise _CaptureRejected(
                f"{limit_reason}; disabling this BCG runner and using eager"
            )
        return entry

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
