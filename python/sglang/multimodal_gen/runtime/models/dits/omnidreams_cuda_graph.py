# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture wrapper for the OmniDreams autoregressive DiT.

Ported from FlashDreams ``flashdreams/infra/cuda_graph.py`` (``CUDAGraphWrapper``
+ ``set_or_copy``), trimmed to the single-stream OmniDreams use-case.

The OmniDreams AR rollout calls ``self.transformer(...)`` three times per chunk
(two flow-match denoise steps + one context re-forward) with *identical tensor
shapes* once the KV-cache window is steady-state. Per-launch CPU overhead across
those repeated identical-shape calls is what this wrapper eliminates: it captures
one forward into a ``torch.cuda.CUDAGraph`` against static input buffers and
replays it, copying the per-call-varying tensors (noisy latent, timestep,
cond-mask, RoPE freqs, HD-map tokens) into the static buffers each call.

Capture invariants the AR-loop wiring MUST uphold (see
``OmniDreamsDenoisingStage``):

* Capture ONLY at KV steady-state. ``BlockKVCache.cached_k()`` returns a
  variable-length prefix slice while the window is filling and the full
  fixed-shape buffer only once steady (``is_steady_state()``). A graph captured
  mid-fill would bake the wrong slice shape. Run fill-phase chunks eager.
* The first-frame ``pin`` injection and the ``before_update``/``after_update``
  window roll stay EAGER, outside the wrapped call. Only the raw
  ``self.transformer(...)`` is captured. The in-place ``BlockKVCache.update``
  write happens inside the forward and is captured against the cache's
  pointer-stable ``_k``/``_v`` buffers (correct because the steady-state write
  position is constant).
* The eager window roll and the graph replay run on the same (default) stream,
  so the roll's writes to ``_k``/``_v`` are ordered before the replay's reads.
  Do not move either onto a side stream without adding an explicit event sync.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def set_or_copy(state: dict, key: Any, new_value: torch.Tensor) -> None:
    """Write ``new_value`` into ``state[key]`` preserving the storage pointer.

    First write clones into a fresh buffer; subsequent same-shape writes
    ``copy_`` in place. Pointer stability is required for CUDA-graph capture,
    since captured kernels reference the slot's storage address.
    """
    cur = state.get(key)
    if cur is not None and cur.shape == new_value.shape:
        cur.copy_(new_value)
    else:
        state[key] = new_value.clone()


class CUDAGraphWrapper:
    """Capture a stateful CUDA callable into a replayable graph.

    The callable runs eagerly for ``warmup_iters`` calls so kernels JIT-load and
    the allocator stabilises (and, when ``fn`` is ``torch.compile``-d, so
    Inductor's lazy triton autotunes -- illegal during capture -- run on the
    eager path first). The next call captures the whole forward into a
    ``torch.cuda.CUDAGraph`` against static input buffers; every same-shape call
    after that copies inputs into those buffers and replays the graph, returning
    clones of the captured outputs.

    Only top-level tensor positional args and kwargs are copied into static
    buffers. Everything else (ints, ``None``, lists, dicts, custom objects)
    passes through verbatim -- this is intentional so mutable state passed
    through containers (e.g. the ``list[BlockKVCache]`` whose ``_k``/``_v`` are
    already pointer-stable) keeps its in-place semantics. Pass any
    per-call-varying tensor as its own top-level arg.

    A change in the staged-tensor signature drops the graph and restarts warmup.
    ``reset`` does the same explicitly -- call it per request (fresh KV caches).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        warmup_iters: int = 2,
        *,
        capture_error_mode: str = "thread_local",
    ):
        self.fn = fn
        self.warmup_iters = warmup_iters
        self.capture_error_mode = capture_error_mode
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_args: list[Any] = []
        self._static_kwargs: dict[str, Any] = {}
        self._static_out_leaves: Optional[list[Any]] = None
        self._out_spec: Any = None
        self._warmup_remaining = warmup_iters

    def reset(self) -> None:
        self._graph = None
        self._static_args = []
        self._static_kwargs = {}
        self._static_out_leaves = None
        self._out_spec = None
        self._warmup_remaining = self.warmup_iters

    @property
    def is_capturing_or_captured(self) -> bool:
        """True once warmup has been consumed (graph captured, or next call captures)."""
        return self._graph is not None or self._warmup_remaining == 0

    # --- input staging ---

    @staticmethod
    def _slot_compatible(slot: Any, fresh: Any) -> bool:
        if isinstance(slot, torch.Tensor):
            return (
                isinstance(fresh, torch.Tensor)
                and slot.shape == fresh.shape
                and slot.dtype == fresh.dtype
            )
        return not isinstance(fresh, torch.Tensor)

    def _slots_compatible_with(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> bool:
        if len(self._static_args) != len(args):
            return False
        if set(self._static_kwargs) != set(kwargs):
            return False
        for slot, fresh in zip(self._static_args, args):
            if not self._slot_compatible(slot, fresh):
                return False
        for name, slot in self._static_kwargs.items():
            if not self._slot_compatible(slot, kwargs[name]):
                return False
        return True

    @staticmethod
    def _make_slot(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value).contiguous()
        return value

    def _stage(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self._slots_compatible_with(args, kwargs):
            self.reset()
            self._static_args = [self._make_slot(a) for a in args]
            self._static_kwargs = {k: self._make_slot(v) for k, v in kwargs.items()}

        staged_args: list[Any] = []
        for slot, fresh in zip(self._static_args, args):
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_args.append(slot)
            else:
                staged_args.append(fresh)

        staged_kwargs: dict[str, Any] = {}
        for name, fresh in kwargs.items():
            slot = self._static_kwargs[name]
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_kwargs[name] = slot
            else:
                staged_kwargs[name] = fresh

        return tuple(staged_args), staged_kwargs

    # --- output handling ---

    def _clone_output(self) -> Any:
        assert self._static_out_leaves is not None and self._out_spec is not None
        cloned = [
            leaf.clone() if isinstance(leaf, torch.Tensor) else leaf
            for leaf in self._static_out_leaves
        ]
        return tree_unflatten(cloned, self._out_spec)

    # --- public entry points ---

    def drain(self, *args: Any, **kwargs: Any) -> Any:
        """Eager autotune drain through the shared static buffers.

        Runs ``fn`` eagerly against the same static buffers + strides that
        ``__call__`` will later capture against, so a ``torch.compile``-d ``fn``
        finishes its lazy Inductor specialisation before capture. Does not
        consume ``warmup_iters`` and does not capture.
        """
        args, kwargs = self._stage(args, kwargs)
        return self.fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = self._stage(args, kwargs)

        if self._graph is not None:
            self._graph.replay()
            return self._clone_output()

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return self.fn(*args, **kwargs)

        # Capture: trace one full forward against the static buffers.
        # cudaStreamBeginCapture only records kernels -- it does not execute
        # them -- so the static outputs and in-place cache updates are no-ops
        # during capture. Replay once immediately to actually compute the
        # output and advance the cache. Keep the graph local until capture and
        # the first replay both succeed, so a failed capture does not leave a
        # stored-but-invalid graph that hides the real error on the next call.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=self.capture_error_mode):
            out = self.fn(*args, **kwargs)
        out_leaves, out_spec = tree_flatten(out)
        graph.replay()
        self._graph = graph
        self._out_spec = out_spec
        self._static_out_leaves = out_leaves
        return self._clone_output()
