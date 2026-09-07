"""Bounded full-tower CUDA graph runner for Kimi K3 vision."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, List, Tuple

import torch

from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)

if TYPE_CHECKING:
    from sglang.srt.models.kimi_k3_vl import (
        GridTHW,
        KimiK3VisionForwardMetadata,
        KimiK3VisionTower,
    )

logger = logging.getLogger(__name__)


@dataclass
class _CapturedVisionGraph:
    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor
    outputs: Tuple[torch.Tensor, ...]
    metadata: KimiK3VisionForwardMetadata


class KimiK3ViTCudaGraphRunner:
    def __init__(
        self,
        tower: KimiK3VisionTower,
        *,
        capacity: int,
        min_hits: int,
        max_seqlen: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if min_hits <= 0:
            raise ValueError(f"min_hits must be positive, got {min_hits}")
        if max_seqlen is not None and max_seqlen <= 0:
            raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
        self.tower = tower
        self.capacity = capacity
        self.min_hits = min_hits
        self.max_seqlen = max_seqlen
        self.graphs: dict[Hashable, _CapturedVisionGraph] = {}
        self.seen: OrderedDict[Hashable, int] = OrderedDict()
        self.failed_keys: set[Hashable] = set()
        self._graph_memory_pool: Any = None
        self._capacity_logged = False
        self._max_seqlen_logged = False
        logger.info(
            "Kimi-K3 ViT CUDA graph: capacity=%d min_hits=%d max_seqlen=%s",
            capacity,
            min_hits,
            max_seqlen,
        )

    @staticmethod
    def graph_key(grid_thw_list: Tuple[GridTHW, ...]) -> Hashable:
        return grid_thw_list

    def _record_hit(self, key: Hashable) -> int:
        count = self.seen.pop(key, 0) + 1
        self.seen[key] = count
        seen_capacity = max(self.capacity * 8, self.capacity)
        if len(self.seen) > seen_capacity:
            self.seen.popitem(last=False)
        return count

    def _run_eager(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        grid_thw_list: Tuple[GridTHW, ...],
    ) -> tuple[List[torch.Tensor], KimiK3VisionForwardMetadata]:
        metadata = self.tower.prepare_forward_metadata(
            grid_thws,
            grid_thw_list=grid_thw_list,
            total_tokens=pixel_values.shape[0],
            dtype=pixel_values.dtype,
        )
        outputs = self.tower._forward_eager(
            pixel_values,
            grid_thws,
            grid_thw_list=grid_thw_list,
            forward_metadata=metadata,
        )
        return outputs, metadata

    def _capture(
        self,
        key: Hashable,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        grid_thw_list: Tuple[GridTHW, ...],
        metadata: KimiK3VisionForwardMetadata,
    ) -> _CapturedVisionGraph:
        allocated_before = torch.cuda.memory_allocated(pixel_values.device)
        reserved_before = torch.cuda.memory_reserved(pixel_values.device)
        input_buffer = torch.empty_like(pixel_values)
        input_buffer.copy_(pixel_values)
        graph = torch.cuda.CUDAGraph()
        if self._graph_memory_pool is None:
            self._graph_memory_pool = get_or_create_global_graph_memory_pool(torch.cuda)
        with torch.cuda.graph(graph, pool=self._graph_memory_pool):
            outputs = self.tower._forward_eager(
                input_buffer,
                grid_thws,
                grid_thw_list=grid_thw_list,
                forward_metadata=metadata,
            )
        torch.cuda.synchronize(pixel_values.device)
        entry = _CapturedVisionGraph(
            graph=graph,
            input_buffer=input_buffer,
            outputs=tuple(outputs),
            metadata=metadata,
        )
        position_embeddings = getattr(metadata, "position_embeddings", None)
        position_cache_bytes = (
            position_embeddings.numel() * position_embeddings.element_size()
            if position_embeddings is not None
            else 0
        )
        logger.info(
            "Captured Kimi-K3 ViT CUDA graph: key=%s cache=%d/%d "
            "allocated_delta_mib=%.1f reserved_delta_mib=%.1f "
            "position_cache_mib=%.1f allocated_total_mib=%.1f "
            "reserved_total_mib=%.1f",
            key,
            len(self.graphs) + 1,
            self.capacity,
            (torch.cuda.memory_allocated(pixel_values.device) - allocated_before)
            / 2**20,
            (torch.cuda.memory_reserved(pixel_values.device) - reserved_before) / 2**20,
            position_cache_bytes / 2**20,
            torch.cuda.memory_allocated(pixel_values.device) / 2**20,
            torch.cuda.memory_reserved(pixel_values.device) / 2**20,
        )
        return entry

    def run(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        grid_thw_list: Tuple[GridTHW, ...],
    ) -> List[torch.Tensor]:
        key = self.graph_key(grid_thw_list)
        entry = self.graphs.get(key)
        if entry is not None:
            entry.input_buffer.copy_(pixel_values)
            with graph_pool_replay_scope():
                entry.graph.replay()
            return list(entry.outputs)

        max_seqlen = max(t * h * w for t, h, w in grid_thw_list)
        if self.max_seqlen is not None and max_seqlen > self.max_seqlen:
            if not self._max_seqlen_logged:
                logger.info(
                    "Kimi-K3 ViT CUDA graph uses eager fallback above "
                    "max_seqlen=%d to avoid low-value captures and HBM use",
                    self.max_seqlen,
                )
                self._max_seqlen_logged = True
            outputs, _ = self._run_eager(pixel_values, grid_thws, grid_thw_list)
            return outputs

        if key in self.failed_keys or self._record_hit(key) < self.min_hits:
            outputs, _ = self._run_eager(pixel_values, grid_thws, grid_thw_list)
            return outputs

        if len(self.graphs) >= self.capacity:
            if not self._capacity_logged:
                logger.warning(
                    "Kimi-K3 ViT CUDA graph cache reached capacity=%d; "
                    "new shapes use eager fallback to bound HBM",
                    self.capacity,
                )
                self._capacity_logged = True
            outputs, _ = self._run_eager(pixel_values, grid_thws, grid_thw_list)
            return outputs

        metadata = self.tower.prepare_forward_metadata(
            grid_thws,
            grid_thw_list=grid_thw_list,
            total_tokens=pixel_values.shape[0],
            dtype=pixel_values.dtype,
        )
        with graph_pool_capture_scope():
            try:
                entry = self._capture(
                    key, pixel_values, grid_thws, grid_thw_list, metadata
                )
            except Exception:
                self.failed_keys.add(key)
                logger.exception(
                    "Kimi-K3 ViT CUDA graph capture failed for key=%s; "
                    "using eager fallback",
                    key,
                )
                outputs, _ = self._run_eager(pixel_values, grid_thws, grid_thw_list)
                return outputs

            self.graphs[key] = entry
            entry.input_buffer.copy_(pixel_values)
            entry.graph.replay()
        return list(entry.outputs)
