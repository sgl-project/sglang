"""Global singleton orchestrating the DWDP lifecycle from setup(model) to cleanup()."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.moe.dwdp.common import restore_storage_rank
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    build_layer_weight_specs,
    lookup_owner,
)
from sglang.srt.layers.moe.dwdp.tensor_schema import DwdpTensorSchema
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DwdpManager:
    def __init__(self, server_args: ServerArgs):
        self.dwdp_size = server_args.dwdp_size
        self.dwdp_rank = get_parallel().tp_rank
        self.device_id = torch.cuda.current_device()
        self.layout: Optional[DwdpExpertLayout] = None

        self._weight_manager: Optional[DWDPWeightManager] = None
        self._moe_layer_indices: List[int] = []
        self._moe_layers: List[Tuple[int, FusedMoE]] = []
        self._schemas: Dict[int, DwdpTensorSchema] = {}
        self._original_main_tensors: Dict[Tuple[int, str], torch.Tensor] = {}
        self._original_side_tensors: Dict[Tuple[int, str], torch.Tensor] = {}

    def setup(self, model: nn.Module) -> None:
        if self._weight_manager is not None:
            return

        self._moe_layers = self._collect_moe_layers(model)
        if not self._moe_layers:
            raise RuntimeError(
                f"DWDP is enabled but no FusedMoE layers were found in "
                f"{type(model).__name__}"
            )
        self._moe_layer_indices = [li for li, _ in self._moe_layers]

        expert_counts = {e.num_global_routed_experts for _, e in self._moe_layers}
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
        shared_counts = {
            int(getattr(experts, "num_fused_shared_experts", 0))
            for _, experts in self._moe_layers
        }
        if len(shared_counts) != 1:
            raise RuntimeError(
                "DWDP requires a uniform fused shared expert count, got "
                f"{sorted(shared_counts)}"
            )
        num_shared = shared_counts.pop()
        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_fused_shared_experts=num_shared,
        )
        logger.info(
            f"DWDP layout: {self.layout.num_routed_experts} experts, "
            f"local [{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"prefetch_per_peer={self.layout.num_prefetch_experts}"
        )

        local_params = {}
        local_routed = self.layout.num_experts_per_worker
        local_count = self.layout.local_expert_end - self.layout.local_expert_start
        main_weight_names = None
        for li, experts in self._moe_layers:
            if experts.quant_method is None:
                raise RuntimeError(f"Layer {li} has no MoE quantization method")
            schema = experts.quant_method.get_dwdp_tensor_schema(experts)
            schema.validate(experts)
            self._schemas[li] = schema
            if main_weight_names is None:
                main_weight_names = schema.main_weights
            elif schema.main_weights != main_weight_names:
                raise RuntimeError(
                    "DWDP requires identical main weight names across layers"
                )
            for name in schema.main_weights:
                storage_tensor = getattr(experts, name)
                if isinstance(storage_tensor, torch.nn.Parameter):
                    storage_tensor = storage_tensor.data
                self._original_main_tensors[(li, name)] = storage_tensor
                original = experts.quant_method.get_dwdp_tensor(experts, name)
                required = local_routed + num_shared
                if original.ndim == 0 or original.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {li} tensor {name} has shape "
                        f"{tuple(original.shape)}, expected at least {required} experts"
                    )
                local = original.narrow(0, 0, local_count)
                if not local.is_contiguous():
                    local = local.contiguous()
                local_params[(li, name)] = local
        layer_weight_specs = build_layer_weight_specs(
            local_params, self.layout.num_experts
        )

        group = get_parallel().tp_group
        transport = None
        weight_buffer = None
        try:
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
            self._allgather_small_params(self._moe_layers, self._schemas, group)

            manager = DWDPWeightManager(
                weight_buffer=weight_buffer,
                peer_views=transport.peer_views,
                peer_ranges=self.layout.peer_ranges,
                moe_layer_indices=self._moe_layer_indices,
                weight_names=list(main_weight_names),
                dwdp_rank=self.dwdp_rank,
                dwdp_size=self.dwdp_size,
                transport=transport,
            )
            for li, experts in self._moe_layers:
                experts.bind_full_expert_weights(
                    {
                        name: weight_buffer.get_full_tensor(li, name)
                        for name in main_weight_names
                    }
                )
            self._weight_manager = manager
            for tensor in self._original_main_tensors.values():
                tensor.untyped_storage().resize_(0)
            torch.cuda.empty_cache()
        except Exception:
            layers = dict(self._moe_layers)
            for (layer_idx, name), original in self._original_main_tensors.items():
                layers[layer_idx].replace_expert_tensor(name, original)
            for (layer_idx, name), original in self._original_side_tensors.items():
                layers[layer_idx].replace_expert_tensor(name, original)
            for _, experts in self._moe_layers:
                experts.unbind_dwdp_weights()
            if weight_buffer is not None:
                weight_buffer.release()
            if transport is not None:
                transport.release()
            self._original_main_tensors.clear()
            self._original_side_tensors.clear()
            raise

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
        if self._weight_manager is None:
            return
        if dist.is_initialized():
            get_parallel().tp_group.barrier()
        self._weight_manager.release()
        self._weight_manager = None
        self._original_main_tensors.clear()
        self._original_side_tensors.clear()
        self._schemas.clear()

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
        self,
        moe_layers: List[Tuple[int, FusedMoE]],
        schemas: Dict[int, DwdpTensorSchema],
        group,
    ) -> None:
        local_experts = self.layout.num_experts_per_worker
        num_shared = self.layout.num_fused_shared_experts

        for li, experts in moe_layers:
            schema = schemas[li]
            names = tuple(
                dict.fromkeys(
                    tuple(
                        name
                        for name in schema.partitioned
                        if name not in schema.main_weights
                    )
                    + schema.replicated
                )
            )
            for pname in names:
                storage_tensor = getattr(experts, pname)
                if isinstance(storage_tensor, torch.nn.Parameter):
                    storage_tensor = storage_tensor.data
                self._original_side_tensors.setdefault(
                    (li, pname),
                    storage_tensor,
                )
                data = experts.quant_method.get_dwdp_tensor(experts, pname)
                required = local_experts + num_shared
                if data.ndim == 0 or data.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {li} side tensor {pname} has shape "
                        f"{tuple(data.shape)}, expected at least {required} experts"
                    )
                routed = data.narrow(0, 0, local_experts)
                shards = [torch.empty_like(routed) for _ in range(self.dwdp_size)]
                dist.all_gather(shards, routed, group=group.device_group)
                full = torch.cat(shards, dim=0)
                if num_shared:
                    full = torch.cat(
                        [
                            full,
                            data.narrow(0, local_experts, num_shared),
                        ],
                        dim=0,
                    )
                full = restore_storage_rank(storage_tensor, full)
                experts.replace_expert_tensor(pname, full)

                logger.debug(
                    f"Layer {li}: allgathered {pname} "
                    f"({data.shape[0]} -> {full.shape[0]}) "
                    f"shape={tuple(full.shape)} dtype={full.dtype} "
                    f"size={full.numel() * full.element_size() / 1e6:.1f}MB"
                )
