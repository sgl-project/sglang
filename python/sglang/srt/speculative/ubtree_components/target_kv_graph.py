# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""CUDA Graph replay for UBTree accepted-path target KV movement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from sglang.kernels.ops.speculative.ubtree import prepare_target_kv_move_metadata


@dataclass
class _CapturedTargetKVMoveBucket:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    accept_index: torch.Tensor
    accept_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    target_cache_loc: torch.Tensor
    source_cache_loc: torch.Tensor
    graph: torch.cuda.CUDAGraph


class UBTreeTargetKVMoveGraphRunner:
    """Capture accepted-path metadata preparation and all-layer target KV copy.

    Each exact batch-size bucket owns fixed-address input buffers. Live request
    metadata is copied into those buffers before replay; the graph then compacts
    the accepted path and moves a fixed ``batch_size * accept_width`` set of KV
    rows. Unused rows map slot 0 to slot 0, matching the eager padded mover.
    """

    def __init__(
        self,
        *,
        kvcache,
        req_to_token: torch.Tensor,
        batch_sizes: Iterable[int],
        accept_width: int,
        verify_node_count: int | None = None,
        verify_node_counts: Iterable[int] | None = None,
        device: torch.device | str,
    ) -> None:
        self.kvcache = kvcache
        self.req_to_token = req_to_token
        self.accept_width = int(accept_width)
        if verify_node_counts is None:
            if verify_node_count is None:
                raise ValueError("verify_node_count or verify_node_counts is required")
            verify_node_counts = (verify_node_count,)
        self.verify_node_counts = tuple(
            sorted({int(value) for value in verify_node_counts if int(value) > 0})
        )
        if not self.verify_node_counts:
            raise ValueError("verify_node_counts must contain a positive value")
        self.verify_node_count = self.verify_node_counts[-1]
        self.device = torch.device(device)
        self._captured: dict[tuple[int, int], _CapturedTargetKVMoveBucket] = {}
        requested = sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        if self.device.type == "cuda":
            for batch_size in requested:
                for node_count in self.verify_node_counts:
                    self._captured[(batch_size, node_count)] = self._capture(
                        batch_size, node_count
                    )

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(sorted({key[0] for key in self._captured}))

    def _capture(
        self, batch_size: int, verify_node_count: int
    ) -> _CapturedTargetKVMoveBucket:
        req_pool_indices = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        accept_index = torch.zeros(
            (batch_size, self.accept_width),
            dtype=torch.int32,
            device=self.device,
        )
        accept_lens = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        out_cache_loc = torch.zeros(
            batch_size * verify_node_count,
            dtype=torch.int64,
            device=self.device,
        )
        target_cache_loc = torch.zeros(
            batch_size * self.accept_width,
            dtype=torch.int64,
            device=self.device,
        )
        source_cache_loc = torch.zeros_like(target_cache_loc)

        def run() -> None:
            prepare_target_kv_move_metadata(
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                seq_lens=seq_lens,
                accept_index=accept_index,
                accept_lens=accept_lens,
                out_cache_loc=out_cache_loc,
                target_cache_loc_out=target_cache_loc,
                source_cache_loc_out=source_cache_loc,
            )
            self.kvcache.move_kv_cache(target_cache_loc, source_cache_loc)

        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        current_stream.wait_stream(capture_stream)
        return _CapturedTargetKVMoveBucket(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            accept_index=accept_index,
            accept_lens=accept_lens,
            out_cache_loc=out_cache_loc,
            target_cache_loc=target_cache_loc,
            source_cache_loc=source_cache_loc,
            graph=graph,
        )

    def __call__(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> bool:
        if accept_index.ndim != 2:
            raise ValueError("accept_index must be two-dimensional")
        batch_size, accept_width = map(int, accept_index.shape)
        if int(out_cache_loc.numel()) % batch_size != 0:
            return False
        verify_node_count = int(out_cache_loc.numel()) // batch_size
        captured = self._captured.get((batch_size, verify_node_count))
        if captured is None or accept_width != self.accept_width:
            return False
        if any(
            int(tensor.numel()) != batch_size
            for tensor in (req_pool_indices, seq_lens, accept_lens)
        ):
            raise ValueError("request metadata must contain one value per request")

        captured.req_pool_indices.copy_(req_pool_indices.reshape(-1))
        captured.seq_lens.copy_(seq_lens.reshape(-1))
        captured.accept_index.copy_(accept_index)
        captured.accept_lens.copy_(accept_lens.reshape(-1))
        captured.out_cache_loc.copy_(out_cache_loc.reshape(-1))
        captured.graph.replay()
        return True


__all__ = ["UBTreeTargetKVMoveGraphRunner"]
