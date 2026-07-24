# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
class VLADenoiseGraphSignature:
    batch_size: int
    prefix_len: int
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


class VLADenoiseGraphRunner:
    """Full CUDA graph runner for one VLA action-denoise step.

    Each signature owns fixed input and output buffers. This does not use
    diffusion BCG and does not capture prefix encoding or token decode.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._captured: dict[VLADenoiseGraphSignature, _CapturedDenoiseGraph] = {}
        self._disabled_signatures: set[VLADenoiseGraphSignature] = set()
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph_pool: Any = None

    def _sync_context_if_needed(
        self,
        captured: _CapturedDenoiseGraph,
        prefix_context: PrefixContext,
    ) -> None:
        context_id = id(prefix_context.past_key_values)
        context_digest = prefix_context.cache_key_digest
        if (
            context_digest is not None
            and captured.current_context_digest == context_digest
        ):
            captured.current_context_id = context_id
            return
        if captured.current_context_id == context_id:
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
        self._captured[signature] = captured
        logger.info(
            "Captured VLA denoise CUDA graph: batch=%d prefix=%d action=%dx%d "
            "dtype=%s",
            signature.batch_size,
            signature.prefix_len,
            signature.action_horizon,
            signature.action_dim,
            signature.dtype,
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

        captured = self._captured.get(signature)
        try:
            if captured is None:
                captured = self._capture(
                    signature, step_fn, prefix_context, x_t, timestep
                )
                captured.graph.replay()
            else:
                self._sync_context_if_needed(captured, prefix_context)
                captured.static_x_t.copy_(x_t)
                captured.static_timestep.copy_(timestep)
                captured.graph.replay()
            return captured.static_output
        except Exception:
            self._disabled_signatures.add(signature)
            self._captured.pop(signature, None)
            logger.warning(
                "VLA denoise CUDA graph disabled for signature %s",
                signature,
                exc_info=True,
            )
            return step_fn(prefix_context, x_t, timestep)
