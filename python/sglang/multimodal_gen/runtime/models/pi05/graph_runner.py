# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.models.pi05.prefix_cache import PrefixContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pi05DenoiseShapeBucket:
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


def _clone_past_key_values(past_key_values: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(
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
        prefix_position_ids=prefix_context.prefix_position_ids.detach().clone(),
        prefix_len=prefix_context.prefix_len,
        dtype=prefix_context.dtype,
        device=prefix_context.device,
        layout=dict(prefix_context.layout),
        cache_key_digest=prefix_context.cache_key_digest,
    )


def _copy_prefix_context_(dst: PrefixContext, src: PrefixContext) -> None:
    dst.prefix_pad_masks.copy_(src.prefix_pad_masks)
    dst.prefix_position_ids.copy_(src.prefix_position_ids)
    _copy_past_key_values_(dst.past_key_values, src.past_key_values)
    dst.cache_key_digest = src.cache_key_digest


class Pi05DenoiseGraphRunner:
    """Shape-bucketed action denoise runner.

    This class is the Pi0.5-facing seam for reusing SRT's piecewise CUDA graph
    machinery. It deliberately targets one action-expert denoise step, not VLM
    prefix prefill and not token decode.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._captured: dict[Pi05DenoiseShapeBucket, Any] = {}
        self._disabled_buckets: set[Pi05DenoiseShapeBucket] = set()

    def can_replay(self, bucket: Pi05DenoiseShapeBucket) -> bool:
        return self.enabled and bucket in self._captured

    def _sync_context_if_needed(
        self,
        captured: _CapturedDenoiseGraph,
        prefix_context: PrefixContext,
    ) -> None:
        context_id = id(prefix_context.past_key_values)
        if captured.current_context_id == context_id:
            return
        _copy_prefix_context_(captured.static_prefix_context, prefix_context)
        captured.current_context_id = context_id

    def _capture(
        self,
        bucket: Pi05DenoiseShapeBucket,
        step_fn: Callable[..., Any],
        prefix_context: PrefixContext,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _CapturedDenoiseGraph:
        static_prefix_context = _clone_prefix_context(prefix_context)
        static_x_t = x_t.detach().clone()
        static_timestep = timestep.detach().clone()

        stream = torch.cuda.Stream(device=x_t.device)
        stream.wait_stream(torch.cuda.current_stream(device=x_t.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            static_output = step_fn(
                static_prefix_context,
                static_x_t,
                static_timestep,
            )
        torch.cuda.current_stream(device=x_t.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_output = step_fn(
                static_prefix_context,
                static_x_t,
                static_timestep,
            )

        captured = _CapturedDenoiseGraph(
            graph=graph,
            static_prefix_context=static_prefix_context,
            static_x_t=static_x_t,
            static_timestep=static_timestep,
            static_output=static_output,
            current_context_id=id(prefix_context.past_key_values),
        )
        self._captured[bucket] = captured
        return captured

    def capture_or_run(
        self,
        bucket: Pi05DenoiseShapeBucket,
        step_fn: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        if (
            not self.enabled
            or kwargs
            or bucket in self._disabled_buckets
            or len(args) != 3
        ):
            return step_fn(*args, **kwargs)

        prefix_context, x_t, timestep = args
        if not isinstance(prefix_context, PrefixContext) or x_t.device.type != "cuda":
            return step_fn(*args, **kwargs)

        captured = self._captured.get(bucket)
        try:
            if captured is None:
                captured = self._capture(bucket, step_fn, prefix_context, x_t, timestep)
                captured.graph.replay()
            else:
                self._sync_context_if_needed(captured, prefix_context)
                captured.static_x_t.copy_(x_t)
                captured.static_timestep.copy_(timestep)
                captured.graph.replay()
            return captured.static_output
        except Exception:
            self._disabled_buckets.add(bucket)
            self._captured.pop(bucket, None)
            logger.warning(
                "Pi05 denoise CUDA graph disabled for bucket %s",
                bucket,
                exc_info=True,
            )
            return step_fn(*args, **kwargs)
