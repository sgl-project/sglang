# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class VLAPrefixGraphSignature:
    batch_size: int
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    static_layout: tuple[Any, ...]
    parallel_layout: str


@dataclass
class _CapturedPrefixGraph:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_output: PrefixContext


@dataclass(frozen=True)
class VLADenoiseGraphSignature:
    batch_size: int
    prefix_len: int
    prefix_full_attention: bool
    action_horizon: int
    action_dim: int
    dtype: str
    parallel_layout: str


@dataclass
class _CapturedDenoiseGraph:
    graph: torch.cuda.CUDAGraph
    static_prefix_context: PrefixContext
    static_x_t: torch.Tensor
    static_timestep: torch.Tensor
    static_output: torch.Tensor
    current_context_id: int | None = None
    current_context_digest: str | None = None


@dataclass(frozen=True)
class VLAGraphCacheInfo:
    size: int
    max_entries: int
    hits: int
    misses: int
    captures: int
    evictions: int
    failures: int
    evict_on_miss: bool


class _BoundedCaptureCache:
    """Bounded LRU owning CUDA graphs and their static buffers."""

    def __init__(self, name: str, max_entries: int, *, evict_on_miss: bool):
        self.name = name
        self.max_entries = max(0, int(max_entries))
        self.evict_on_miss = evict_on_miss
        self.entries: OrderedDict[Any, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.captures = 0
        self.evictions = 0
        self.failures = 0

    @staticmethod
    def _release(entry: Any) -> None:
        reset = getattr(entry.graph, "reset", None)
        if callable(reset):
            reset()

    def get(self, signature: Any) -> Any | None:
        entry = self.entries.get(signature)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(signature)
        return entry

    def can_admit(self, signature: Any) -> bool:
        return self.max_entries > 0 and (
            signature in self.entries
            or len(self.entries) < self.max_entries
            or self.evict_on_miss
        )

    def _evict_lru(self) -> None:
        evicted_signature, evicted = self.entries.popitem(last=False)
        self._release(evicted)
        self.evictions += 1
        logger.debug(
            "Evicted VLA %s CUDA graph for signature %s (entries=%d/%d)",
            self.name,
            evicted_signature,
            len(self.entries),
            self.max_entries,
        )

    def prepare_admission(self, signature: Any) -> None:
        if (
            signature in self.entries
            or len(self.entries) < self.max_entries
            or not self.evict_on_miss
        ):
            return
        self._evict_lru()

    def put(self, signature: Any, entry: Any) -> bool:
        if self.max_entries == 0 or not self.can_admit(signature):
            self._release(entry)
            return False
        previous = self.entries.pop(signature, None)
        if previous is not None:
            self._release(previous)
        self.entries[signature] = entry
        self.captures += 1

        if len(self.entries) > self.max_entries:
            self._evict_lru()
        return True

    def discard(self, signature: Any) -> None:
        entry = self.entries.pop(signature, None)
        if entry is not None:
            self._release(entry)

    def clear(self) -> None:
        for entry in self.entries.values():
            self._release(entry)
        self.entries.clear()

    def mark_failure(self) -> None:
        self.failures += 1

    def info(self) -> VLAGraphCacheInfo:
        return VLAGraphCacheInfo(
            size=len(self.entries),
            max_entries=self.max_entries,
            hits=self.hits,
            misses=self.misses,
            captures=self.captures,
            evictions=self.evictions,
            failures=self.failures,
            evict_on_miss=self.evict_on_miss,
        )


def _clone_past_key_values(past_key_values: Any) -> Any:
    return VLADensePrefixCache(
        tuple(
            (keys.detach().clone(), values.detach().clone(), sliding_window)
            for keys, values, sliding_window in past_key_values
        )
    )


def _copy_past_key_values_(dst: Any, src: Any) -> None:
    for (dst_keys, dst_values, _), (src_keys, src_values, _) in zip(
        dst, src, strict=True
    ):
        dst_keys.copy_(src_keys)
        dst_values.copy_(src_values)


def _clone_prefix_context(prefix_context: PrefixContext) -> PrefixContext:
    return PrefixContext(
        past_key_values=_clone_past_key_values(prefix_context.past_key_values),
        prefix_pad_masks=prefix_context.prefix_pad_masks.detach().clone(),
        prefix_len=prefix_context.prefix_len,
        layout=dict(prefix_context.layout),
        cache_key_digest=prefix_context.cache_key_digest,
    )


def _copy_prefix_context_(dst: PrefixContext, src: PrefixContext) -> None:
    dst.prefix_pad_masks.copy_(src.prefix_pad_masks)
    _copy_past_key_values_(dst.past_key_values, src.past_key_values)
    dst.cache_key_digest = src.cache_key_digest


class VLAPrefixGraphRunner:
    """Full CUDA graph runner for VLA prefix encoding shape buckets."""

    def __init__(
        self,
        enabled: bool = True,
        max_entries: int = 1,
        *,
        evict_on_miss: bool = False,
    ):
        self._cache = _BoundedCaptureCache(
            "prefix",
            max_entries,
            evict_on_miss=evict_on_miss,
        )
        self.max_entries = self._cache.max_entries
        self.enabled = enabled and self.max_entries > 0
        self._disabled_signatures: set[VLAPrefixGraphSignature] = set()
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph_pool: Any = None

    def cache_info(self) -> VLAGraphCacheInfo:
        return self._cache.info()

    def clear(self) -> None:
        self._cache.clear()
        self._disabled_signatures.clear()

    def _capture(
        self,
        signature: VLAPrefixGraphSignature,
        step_fn: Callable[[tuple[torch.Tensor, ...]], PrefixContext],
        inputs: tuple[torch.Tensor, ...],
    ) -> _CapturedPrefixGraph:
        static_inputs = tuple(tensor.detach().clone() for tensor in inputs)
        device_module = torch.get_device_module(inputs[0].device)
        if self._capture_stream is None:
            self._capture_stream = device_module.Stream(device=inputs[0].device)
        if self._graph_pool is None:
            self._graph_pool = get_or_create_global_graph_memory_pool(device_module)
            set_graph_pool_id(self._graph_pool)

        device_module.synchronize()
        with device_module.stream(self._capture_stream), torch.inference_mode():
            step_fn(static_inputs)
        self._capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with (
            device_module.graph(
                cuda_graph=graph,
                pool=self._graph_pool,
                stream=self._capture_stream,
            ),
            torch.inference_mode(),
        ):
            static_output = step_fn(static_inputs)
        self._capture_stream.synchronize()
        static_output.layout["mutable_graph_output"] = True

        captured = _CapturedPrefixGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_output=static_output,
        )
        logger.info(
            "Captured VLA prefix CUDA graph: batch=%d inputs=%s (capacity=%d)",
            signature.batch_size,
            signature.input_shapes,
            self.max_entries,
        )
        return captured

    def capture_or_run(
        self,
        signature: VLAPrefixGraphSignature,
        step_fn: Callable[[tuple[torch.Tensor, ...]], PrefixContext],
        inputs: tuple[torch.Tensor, ...],
    ) -> PrefixContext:
        if (
            not self.enabled
            or signature in self._disabled_signatures
            or not inputs
            or inputs[0].device.type != "cuda"
        ):
            return step_fn(inputs)

        captured = self._cache.get(signature)
        if captured is None and not self._cache.can_admit(signature):
            return step_fn(inputs)
        try:
            if captured is None:
                self._cache.prepare_admission(signature)
                captured = self._capture(signature, step_fn, inputs)
                self._cache.put(signature, captured)
            else:
                for static_input, current_input in zip(
                    captured.static_inputs, inputs, strict=True
                ):
                    static_input.copy_(current_input)
            captured.graph.replay()
            torch.get_device_module(inputs[0].device).current_stream(
                device=inputs[0].device
            ).synchronize()
            return captured.static_output
        except Exception:
            self._disabled_signatures.add(signature)
            self._cache.discard(signature)
            self._cache.mark_failure()
            logger.warning(
                "VLA prefix CUDA graph disabled for signature %s",
                signature,
                exc_info=True,
            )
            return step_fn(inputs)


class VLADenoiseGraphRunner:
    """Full CUDA graph runner for one VLA action-denoise step.

    Each signature owns fixed input and output buffers. This does not use
    diffusion BCG and does not capture prefix encoding or token decode.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_entries: int = 1,
        *,
        evict_on_miss: bool = False,
    ):
        self._cache = _BoundedCaptureCache(
            "action-denoise",
            max_entries,
            evict_on_miss=evict_on_miss,
        )
        self.max_entries = self._cache.max_entries
        self.enabled = enabled and self.max_entries > 0
        self._disabled_signatures: set[VLADenoiseGraphSignature] = set()
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph_pool: Any = None
        self._capacity_warning_emitted = False

    def cache_info(self) -> VLAGraphCacheInfo:
        return self._cache.info()

    def clear(self) -> None:
        self._cache.clear()
        self._disabled_signatures.clear()
        self._capacity_warning_emitted = False

    def _sync_context_if_needed(
        self,
        captured: _CapturedDenoiseGraph,
        prefix_context: PrefixContext,
    ) -> None:
        context_id = id(prefix_context.past_key_values)
        context_digest = prefix_context.cache_key_digest
        mutable_graph_output = bool(
            prefix_context.layout.get("mutable_graph_output", False)
        )
        if not mutable_graph_output and (
            context_digest is not None
            and captured.current_context_digest == context_digest
        ):
            captured.current_context_id = context_id
            return
        if not mutable_graph_output and captured.current_context_id == context_id:
            return
        _copy_prefix_context_(captured.static_prefix_context, prefix_context)
        captured.current_context_id = context_id
        captured.current_context_digest = context_digest

    def _capture(
        self,
        signature: VLADenoiseGraphSignature,
        step_fn: Callable[..., torch.Tensor],
        prefix_context: PrefixContext,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _CapturedDenoiseGraph:
        static_prefix_context = _clone_prefix_context(prefix_context)
        static_x_t = x_t.detach().clone()
        static_timestep = timestep.detach().clone()

        device_module = torch.get_device_module(x_t.device)
        if self._capture_stream is None:
            self._capture_stream = device_module.Stream(device=x_t.device)
        if self._graph_pool is None:
            self._graph_pool = get_or_create_global_graph_memory_pool(device_module)
            set_graph_pool_id(self._graph_pool)

        # warm up lazy kernels and workspaces before capture
        device_module.synchronize()
        with device_module.stream(self._capture_stream), torch.inference_mode():
            step_fn(
                static_prefix_context,
                static_x_t,
                static_timestep,
            )
        self._capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with (
            device_module.graph(
                cuda_graph=graph,
                pool=self._graph_pool,
                stream=self._capture_stream,
            ),
            torch.inference_mode(),
        ):
            static_output = step_fn(
                static_prefix_context,
                static_x_t,
                static_timestep,
            )
        self._capture_stream.synchronize()

        captured = _CapturedDenoiseGraph(
            graph=graph,
            static_prefix_context=static_prefix_context,
            static_x_t=static_x_t,
            static_timestep=static_timestep,
            static_output=static_output,
            current_context_id=id(prefix_context.past_key_values),
            current_context_digest=prefix_context.cache_key_digest,
        )
        logger.info(
            "Captured VLA denoise CUDA graph: batch=%d prefix=%d action=%dx%d "
            "dtype=%s (capacity=%d)",
            signature.batch_size,
            signature.prefix_len,
            signature.action_horizon,
            signature.action_dim,
            signature.dtype,
            self.max_entries,
        )
        return captured

    def capture_or_run(
        self,
        signature: VLADenoiseGraphSignature,
        step_fn: Callable[..., torch.Tensor],
        prefix_context: PrefixContext,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled or signature in self._disabled_signatures:
            return step_fn(prefix_context, x_t, timestep)

        if x_t.device.type != "cuda":
            return step_fn(prefix_context, x_t, timestep)

        captured = self._cache.get(signature)
        if captured is None and not self._cache.can_admit(signature):
            if not self._capacity_warning_emitted:
                logger.info(
                    "VLA denoise CUDA graph capacity reached (%d); "
                    "new signatures run eagerly",
                    self.max_entries,
                )
                self._capacity_warning_emitted = True
            return step_fn(prefix_context, x_t, timestep)
        try:
            if captured is None:
                self._cache.prepare_admission(signature)
                captured = self._capture(
                    signature, step_fn, prefix_context, x_t, timestep
                )
                self._cache.put(signature, captured)
                captured.graph.replay()
            else:
                self._sync_context_if_needed(captured, prefix_context)
                captured.static_x_t.copy_(x_t)
                captured.static_timestep.copy_(timestep)
                captured.graph.replay()
            return captured.static_output
        except Exception:
            self._disabled_signatures.add(signature)
            self._cache.discard(signature)
            self._cache.mark_failure()
            logger.warning(
                "VLA denoise CUDA graph disabled for signature %s",
                signature,
                exc_info=True,
            )
            return step_fn(prefix_context, x_t, timestep)
