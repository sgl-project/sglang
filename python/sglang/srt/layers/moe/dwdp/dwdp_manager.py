"""Global singleton orchestrating the DWDP lifecycle from setup(model) to cleanup()."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    build_layer_weight_specs,
    lookup_owner,
)
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_EXPERT_WEIGHT_NAMES = (
    "w13_weight",
    "w2_weight",
)


class DwdpManager:
    def __init__(self, server_args: ServerArgs):
        self.dwdp_size = server_args.dwdp_size
        self.dwdp_rank = get_parallel().tp_rank
        self.device_id = torch.cuda.current_device()
        self.layout: Optional[DwdpExpertLayout] = None

        self._weight_manager: Optional[DWDPWeightManager] = None
        self._moe_layer_indices: List[int] = []

    def setup(self, model: nn.Module) -> None:
        if self._weight_manager is not None:
            return

        moe_layers = self._collect_moe_layers(model)
        if not moe_layers:
            raise RuntimeError(
                f"DWDP is enabled but no FusedMoE layers were found in "
                f"{type(model).__name__}"
            )
        self._moe_layer_indices = [li for li, _ in moe_layers]

        expert_counts = {e.num_global_routed_experts for _, e in moe_layers}
        if len(expert_counts) != 1:
            raise RuntimeError(
                f"DWDP requires a uniform routed expert count across MoE layers, "
                f"got {sorted(expert_counts)}"
            )
        num_routed = expert_counts.pop()
        if num_routed % self.dwdp_size != 0:
            raise ValueError(
                f"DWDP requires num_routed_experts ({num_routed}) to be divisible "
                f"by dwdp_size ({self.dwdp_size})"
            )
        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
        )
        logger.info(
            f"DWDP layout: {self.layout.num_routed_experts} experts, "
            f"local [{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"prefetch_per_peer={self.layout.num_prefetch_experts}"
        )

        local_params = {}
        for li, experts in moe_layers:
            local_params[(li, "w13_weight")] = experts.w13_weight.data
            local_params[(li, "w2_weight")] = experts.w2_weight.data
        layer_weight_specs = build_layer_weight_specs(
            local_params, self.layout.num_routed_experts
        )

        group = get_parallel().tp_group
        transport = DWDPTransport.create(
            layer_weight_specs=layer_weight_specs,
            local_params=local_params,
            group=group,
            layout=self.layout,
            device_id=self.device_id,
        )

        weight_buffer = WeightBuffer.create(
            layer_weight_specs=layer_weight_specs,
            handles=transport.handle_set,
            local_start=self.layout.local_expert_start,
            local_end=self.layout.local_expert_end,
            dwdp_size=self.dwdp_size,
            device_id=self.device_id,
        )

        self._fill_edge_bytes(weight_buffer, transport.peer_views)

        self._weight_manager = DWDPWeightManager(
            weight_buffer=weight_buffer,
            peer_views=transport.peer_views,
            peer_ranges=self.layout.peer_ranges,
            moe_layer_indices=self._moe_layer_indices,
            weight_names=list(_EXPERT_WEIGHT_NAMES),
            dwdp_rank=self.dwdp_rank,
            dwdp_size=self.dwdp_size,
            transport=transport,
        )

        for li, experts in moe_layers:
            experts.bind_full_expert_weights(
                {
                    name: weight_buffer.get_full_tensor(li, name)
                    for name in weight_buffer.weight_names(li)
                }
            )
        self._allgather_small_params(moe_layers, group)

        logger.info("DWDP setup complete.")

    def prefetch_first_layers(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.prefetch_first_layers()

    def wait_prefetch(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.wait_prefetch(layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.record_compute_and_prefetch_next(layer_idx)

    def cleanup(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.release()
            self._weight_manager = None

    @staticmethod
    def _collect_moe_layers(model: nn.Module) -> List[Tuple[int, FusedMoE]]:
        decoder = model.model if hasattr(model, "model") else model
        moe_layers = []
        for layer_idx, layer in enumerate(decoder.layers):
            experts = next(
                (m for m in layer.modules() if isinstance(m, FusedMoE)), None
            )
            if experts is not None:
                moe_layers.append((layer_idx, experts))
        return moe_layers

    def _fill_edge_bytes(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
    ) -> None:
        local_start = self.layout.local_expert_start
        local_end = self.layout.local_expert_end
        peer_ranges = self.layout.peer_ranges

        for li in weight_buffer.layer_indices:
            for name in weight_buffer.weight_names(li):
                edge = weight_buffer.get_edge_info(li, name)
                if edge.leading_edge == 0 and edge.trailing_edge == 0:
                    continue

                full_tensor = weight_buffer.get_full_tensor(li, name)

                if edge.leading_edge > 0 and local_start > 0:
                    prev = local_start - 1
                    peer = lookup_owner(prev, peer_ranges)
                    ps, _ = peer_ranges[peer]
                    key = (peer, li, name)
                    if key in peer_views:
                        full_tensor[prev].copy_(peer_views[key][prev - ps])

                if edge.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                    nxt = local_end
                    peer = lookup_owner(nxt, peer_ranges)
                    ps, _ = peer_ranges[peer]
                    key = (peer, li, name)
                    if key in peer_views:
                        full_tensor[nxt].copy_(peer_views[key][nxt - ps])

        torch.cuda.synchronize(weight_buffer.device_id)

    def _allgather_small_params(
        self, moe_layers: List[Tuple[int, FusedMoE]], group
    ) -> None:
        local_experts = self.layout.num_experts_per_worker
        num_total = self.layout.num_routed_experts

        for li, experts in moe_layers:
            for pname, data in experts.named_per_expert_tensors(local_experts):
                shards = [torch.empty_like(data) for _ in range(self.dwdp_size)]
                dist.all_gather(shards, data, group=group.device_group)
                full = torch.cat(shards, dim=0)[:num_total].contiguous()
                experts.replace_expert_tensor(pname, full)

                logger.debug(
                    f"Layer {li}: allgathered {pname} "
                    f"({local_experts} -> {full.shape[0]}) "
                    f"shape={tuple(full.shape)} dtype={full.dtype} "
                    f"size={full.numel() * full.element_size() / 1e6:.1f}MB"
                )
