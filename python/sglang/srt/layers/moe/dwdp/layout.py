# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Expert ownership and page-aligned memory layout computation for DWDP."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.vmm import align_down, align_up

# one (start, end_capped) expert range per peer DWDP rank
PeerRanges = List[Tuple[int, int]]

LayerWeightSpecs = Dict[int, Dict[str, "WeightSpec"]]


class DwdpExpertLayout:
    def __init__(
        self,
        num_routed_experts: int,
        dwdp_size: int,
        dwdp_rank: int,
    ):
        self.num_routed_experts = num_routed_experts
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank

        num_experts_per_worker = num_routed_experts // dwdp_size
        self.num_experts_per_worker = num_experts_per_worker

        self.num_prefetch_experts = math.ceil(
            (num_routed_experts - num_experts_per_worker) / (dwdp_size - 1)
        )
        self.local_expert_start = min(
            self.num_prefetch_experts * dwdp_rank,
            num_routed_experts - num_experts_per_worker,
        )
        self.local_expert_end = self.local_expert_start + num_experts_per_worker

        self.peer_ranges = compute_peer_ranges(
            dwdp_size=dwdp_size,
            num_experts_per_worker=num_experts_per_worker,
            num_prefetch_experts=self.num_prefetch_experts,
            num_experts_total=num_routed_experts,
        )


class WeightSpec:
    def __init__(
        self,
        num_experts: int,
        chunk_shape: Tuple[int, ...],
        full_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.num_experts = num_experts
        self.chunk_shape = chunk_shape
        self.full_shape = full_shape
        self.dtype = dtype

    @property
    def expert_bytes(self) -> int:
        n = 1
        for d in self.full_shape[1:]:
            n *= d
        return n * torch.tensor([], dtype=self.dtype).element_size()

    @property
    def chunk_bytes(self) -> int:
        n = 1
        for d in self.chunk_shape:
            n *= d
        return n * torch.tensor([], dtype=self.dtype).element_size()

    @property
    def local_experts(self) -> int:
        return self.chunk_shape[0]


class EdgeInfo:
    def __init__(
        self,
        data_offset: int,
        leading_edge: int,
        trailing_edge: int,
        page_start: int,
        page_end: int,
        expert_bytes: int,
    ):
        self.data_offset = data_offset
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.page_start = page_start
        self.page_end = page_end
        self.expert_bytes = expert_bytes


class PageAlignedLayout:
    # composite VA: [pre_region (pool pages) | mnnvl_region (fabric handle) | post_region (pool pages)]
    def __init__(
        self,
        expert_bytes: int,
        num_experts: int,
        local_start: int,
        local_end: int,
        granularity: int,
        pool_granularity: int,
        page_start: int,
        page_end: int,
        pre_size: int,
        mnnvl_size: int,
        post_size: int,
        pre_padding: int,
        post_padding: int,
        data_offset: int,
        leading_edge: int,
        trailing_edge: int,
        total_size: int,
        handle_phys_size: int,
    ):
        self.expert_bytes = expert_bytes
        self.num_experts = num_experts
        self.local_start = local_start
        self.local_end = local_end
        self.granularity = granularity
        self.pool_granularity = pool_granularity
        self.page_start = page_start
        self.page_end = page_end
        self.pre_size = pre_size
        self.mnnvl_size = mnnvl_size
        self.post_size = post_size
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.data_offset = data_offset
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.total_size = total_size
        self.handle_phys_size = handle_phys_size

    @classmethod
    def compute(
        cls,
        expert_bytes: int,
        num_experts: int,
        local_start: int,
        local_end: int,
        granularity: int,
        handle_phys_size: int,
        pool_granularity: Optional[int] = None,
    ) -> PageAlignedLayout:
        if pool_granularity is None:
            pool_granularity = granularity

        local_start_bytes = local_start * expert_bytes
        local_end_bytes = local_end * expert_bytes
        total_expert_bytes = num_experts * expert_bytes

        page_start = align_down(local_start_bytes, granularity)
        page_end = align_up(local_end_bytes, granularity)

        data_offset = local_start_bytes - page_start
        leading_edge = data_offset
        trailing_edge = page_end - local_end_bytes

        mnnvl_size = page_end - page_start

        if mnnvl_size > handle_phys_size:
            raise ValueError(
                f"mnnvl_size ({mnnvl_size}) exceeds handle_phys_size ({handle_phys_size})"
            )

        pre_size = align_up(page_start, pool_granularity)
        pre_padding = pre_size - page_start

        post_size_raw = align_up(total_expert_bytes, granularity) - page_end
        post_size = align_up(post_size_raw, pool_granularity)
        post_padding = post_size - post_size_raw

        total_size = pre_size + mnnvl_size + post_size

        return cls(
            expert_bytes=expert_bytes,
            num_experts=num_experts,
            local_start=local_start,
            local_end=local_end,
            granularity=granularity,
            pool_granularity=pool_granularity,
            page_start=page_start,
            page_end=page_end,
            pre_size=pre_size,
            mnnvl_size=mnnvl_size,
            post_size=post_size,
            pre_padding=pre_padding,
            post_padding=post_padding,
            data_offset=data_offset,
            leading_edge=leading_edge,
            trailing_edge=trailing_edge,
            total_size=total_size,
            handle_phys_size=handle_phys_size,
        )

    def get_edge_info(self) -> EdgeInfo:
        return EdgeInfo(
            data_offset=self.data_offset,
            leading_edge=self.leading_edge,
            trailing_edge=self.trailing_edge,
            page_start=self.page_start,
            page_end=self.page_end,
            expert_bytes=self.expert_bytes,
        )

    @property
    def pre_pages(self) -> int:
        return self.pre_size // self.pool_granularity if self.pool_granularity else 0

    @property
    def post_pages(self) -> int:
        return self.post_size // self.pool_granularity if self.pool_granularity else 0

    @property
    def remote_pages(self) -> int:
        return self.pre_pages + self.post_pages


class MnnvlHandleSet:
    def __init__(
        self,
        handles: Dict[Tuple[int, str], int],
        sizes: Dict[Tuple[int, str], int],
    ):
        self.handles = handles
        self.sizes = sizes

    def get_handle(self, layer_idx: int, name: str) -> int:
        return self.handles[(layer_idx, name)]

    def get_size(self, layer_idx: int, name: str) -> int:
        return self.sizes[(layer_idx, name)]

    @property
    def layer_indices(self) -> List[int]:
        return sorted(set(li for li, _ in self.handles.keys()))

    def weight_names(self, layer_idx: int) -> List[str]:
        return [n for (li, n) in self.handles.keys() if li == layer_idx]


def compute_peer_ranges(
    *,
    dwdp_size: int,
    num_experts_per_worker: int,
    num_prefetch_experts: int,
    num_experts_total: int,
) -> PeerRanges:
    ranges: PeerRanges = []
    for peer_rank in range(dwdp_size):
        start = peer_rank * num_prefetch_experts
        end_capped = min(start + num_experts_per_worker, num_experts_total)
        ranges.append((start, end_capped))
    return ranges


def lookup_owner(expert_id: int, peer_ranges: PeerRanges) -> int:
    for peer_rank, (start, end) in enumerate(peer_ranges):
        if start <= expert_id < end:
            return peer_rank
    raise ValueError(
        f"expert_id={expert_id} not owned by any peer in peer_ranges={peer_ranges}"
    )


def build_layer_weight_specs(
    local_params: Dict[Tuple[int, str], torch.Tensor],
    num_routed_experts: int,
) -> LayerWeightSpecs:
    specs: LayerWeightSpecs = {}
    for (layer_idx, name), param in local_params.items():
        chunk_shape = tuple(param.shape)
        specs.setdefault(layer_idx, {})[name] = WeightSpec(
            num_experts=num_routed_experts,
            chunk_shape=chunk_shape,
            full_shape=(num_routed_experts,) + chunk_shape[1:],
            dtype=param.dtype,
        )
    return specs
