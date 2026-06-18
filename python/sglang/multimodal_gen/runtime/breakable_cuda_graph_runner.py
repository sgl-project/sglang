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

A runner wraps a callable ``nn.Module`` and turns it into an *eager runner* that
transparently proxies every attribute to the wrapped module and, when called,
replays a previously captured graph for the input signature — or runs the
module eagerly when no graph was captured for that signature. Capture is an
explicit, idempotent ``capture()`` call (driven at warmup) so that serving never
triggers a fresh capture.

This file is intentionally local to ``multimodal_gen``: diffusion reuses the
low-level SRT BCG primitives, but the capture/replay runner owns diffusion DiT
signature handling, static tensor buffers, prompt-bucket warmup, and fallback
behavior.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)

# Log under the multimodal_gen namespace so the diffusion server's logging
# config (which configures sglang.multimodal_gen.* at INFO and writes them to
# the server log) surfaces the "[Diffusion BCG] captured ..." lines. A plain
# __name__ logger lives under sglang.srt.* and is not written to the diffusion
# server log, which would hide BCG capture/eviction diagnostics.
logger = logging.getLogger("sglang.multimodal_gen.runtime.breakable_cuda_graph_runner")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[BCG] ignoring invalid integer %s=%r", name, raw)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[BCG] ignoring invalid float %s=%r", name, raw)
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
    return tuple((name, _signature_summary_leaf(value)) for name, value in key[:16]) + (
        (("...", len(key) - 16),) if len(key) > 16 else ()
    )


def _clone_output(out: Any) -> Any:
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, tuple):
        return tuple(_clone_output(o) for o in out)
    if isinstance(out, list):
        return [_clone_output(o) for o in out]
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


class _CaptureRejected(RuntimeError):
    pass


class BaseBreakableCudaGraphRunner:
    """Eager runner around ``transformer`` with an explicit capture/replay API.

    The capture/replay contract:

    * :meth:`capture` captures a BCG graph for the given input signature, once
      (idempotent). It is intended to be driven at warmup so that every
      signature served later is already captured.
    * :meth:`replay` copies live inputs into the captured static buffers and
      replays the graph, returning a clone of the captured output.
    * :meth:`__call__` is the *eager runner*: it replays when a graph exists for
      the signature and otherwise runs ``transformer`` eagerly. It never
      captures, so serving never pays a capture cost.

    Any attribute not defined on the runner is proxied to ``transformer`` so the
    runner can stand in for the wrapped module ("other functions directly
    pass").
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
        self._pool = (
            pool if pool is not None else self.device_module.graph_pool_handle()
        )
        self._capture_stream = self.device_module.Stream(device=device)
        self.entries: dict[tuple, _CaptureEntry] = {}
        # Signatures we have given up capturing (capture raised); run eager.
        self._blocked: set[tuple] = set()
        self._disabled_reason: str | None = None
        self.max_entries = max(0, _env_int("SGLANG_DIFFUSION_BCG_MAX_ENTRIES", 32))
        self.max_segments = max(0, _env_int("SGLANG_DIFFUSION_BCG_MAX_SEGMENTS", 128))

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the runner itself does not define; proxy
        # them to the wrapped transformer so callers can treat the runner as a
        # transparent stand-in. Use __dict__ to avoid recursing through
        # __getattr__ before ``transformer`` is assigned in __init__.
        try:
            transformer = self.__dict__["transformer"]
        except KeyError as e:  # pragma: no cover - during/ before __init__
            raise AttributeError(name) from e
        return getattr(transformer, name)

    # ------------------------------------------------------------------ #
    # Public capture / replay API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def capture(self, **kwargs) -> bool:
        """Capture a graph for ``kwargs``'s signature if not already captured.

        Idempotent: returns ``True`` when a graph is available for the
        signature afterwards (already captured or newly captured), ``False``
        when capture is disabled/blocked or failed (the caller then runs eager).
        """
        if self._disabled_reason is not None:
            return False
        key = self._signature(kwargs)
        if key in self._blocked:
            return False
        if key in self.entries:
            return True
        try:
            entry = self._capture(kwargs, key)
        except Exception as e:  # noqa: BLE001 — never break generation on capture
            logger.warning(
                "[Diffusion BCG] capture failed for signature %s (%s); "
                "this signature will run eager.",
                _signature_summary(key),
                e,
            )
            self._blocked.add(key)
            return False
        self.entries[key] = entry
        self._evict_entries_if_needed()
        return True

    def _should_capture_on_call(self, key: tuple) -> bool:
        """Whether ``__call__`` may lazily capture an unseen signature.

        Base runners only ever capture through the explicit :meth:`capture`
        API, so this returns ``False``: serving never records a fresh graph.
        Subclasses gate lazy capture on a warmup window (see the diffusion
        runner) so warmup can capture by simply driving the forward as usual.
        """
        return False

    @torch.no_grad()
    def __call__(self, **kwargs) -> Any:
        """Eager runner: replay a captured graph, else run ``transformer``.

        While serving this never captures, so no new graph is recorded once
        warmup is over. During the warmup window subclasses opt into lazy
        capture via :meth:`_should_capture_on_call`.
        """
        if self._disabled_reason is not None:
            return self.transformer(**kwargs)
        key = self._signature(kwargs)
        entry = self.entries.get(key)
        if entry is None:
            if not self._should_capture_on_call(key):
                return self.transformer(**kwargs)
            if not self.capture(**kwargs):
                return self.transformer(**kwargs)
            entry = self.entries[key]
        return self.replay(entry, kwargs)

    def replay(self, entry: _CaptureEntry, kwargs: dict[str, Any]) -> Any:
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


class DiffusionBreakableCudaGraphRunner(BaseBreakableCudaGraphRunner):
    """Capture/replay a diffusion DiT ``transformer`` with BCG.

    Unknown attributes proxy to the wrapped transformer, so the runner can
    stand in for the module while only intercepting ``forward`` calls.
    """

    def _should_capture_on_call(self, key) -> bool:
        try:
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )

            forward_batch = get_forward_context().forward_batch
        except Exception:
            return False
        return bool(getattr(forward_batch, "is_warmup", False))
