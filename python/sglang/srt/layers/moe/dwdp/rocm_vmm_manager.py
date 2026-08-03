"""ROCm DWDP manager backed by HIP VMM composite virtual addresses."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.moe.dwdp import hip_vmm
from sglang.srt.layers.moe.dwdp.common import restore_storage_rank
from sglang.srt.layers.moe.dwdp.hsa_copy import HsaSdmaCopyEngine
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    build_layer_weight_specs,
    lookup_owner,
)
from sglang.srt.layers.moe.dwdp.rocm_vmm_transport import RocmVmmTransport
from sglang.srt.layers.moe.dwdp.tensor_schema import DwdpTensorSchema
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class RocmVmmDwdpManager:
    weight_backend = "vmm"

    def __init__(self, server_args):
        self.dwdp_size = int(server_args.dwdp_size)
        self.dwdp_rank = get_parallel().tp_rank
        self.device_id = torch.cuda.current_device()
        self.layout: Optional[DwdpExpertLayout] = None
        self._weight_manager: Optional[DWDPWeightManager] = None
        self._schemas: Dict[int, DwdpTensorSchema] = {}
        self._moe_layers: List[Tuple[int, FusedMoE]] = []
        self._old_main_tensors: List[torch.Tensor] = []

    @classmethod
    def is_available(cls) -> bool:
        available, _ = hip_vmm.extension_availability()
        if not available or not torch.cuda.is_available():
            return False
        try:
            return hip_vmm.is_supported(torch.cuda.current_device())
        except Exception:
            return False

    @staticmethod
    def _collect_moe_layers(model: nn.Module) -> List[Tuple[int, FusedMoE]]:
        decoder = model.model if hasattr(model, "model") else model
        result = []
        for layer_idx, layer in enumerate(decoder.layers):
            experts = next(
                (module for module in layer.modules() if isinstance(module, FusedMoE)),
                None,
            )
            if experts is not None:
                result.append((layer_idx, experts))
        return result

    def setup(self, model: nn.Module) -> None:
        if self._weight_manager is not None:
            return
        arch = torch.cuda.get_device_properties(self.device_id).gcnArchName
        if "gfx950" not in arch:
            raise NotImplementedError(
                f"ROCm VMM DWDP requires gfx950, got {arch!r}"
            )
        if torch.cuda.device_count() < self.dwdp_size:
            raise RuntimeError(
                f"ROCm VMM DWDP needs {self.dwdp_size} visible GPUs on one node, "
                f"got {torch.cuda.device_count()}"
            )
        inaccessible = [
            peer
            for peer in range(torch.cuda.device_count())
            if peer != self.device_id
            and not torch.cuda.can_device_access_peer(self.device_id, peer)
        ]
        if inaccessible:
            raise RuntimeError(
                f"ROCm VMM DWDP requires full XGMI P2P; device {self.device_id} "
                f"cannot access peers {inaccessible}"
            )
        self._moe_layers = self._collect_moe_layers(model)
        if not self._moe_layers:
            raise RuntimeError(
                f"DWDP is enabled but no FusedMoE layers were found in "
                f"{type(model).__name__}"
            )

        expert_counts = {
            experts.num_global_routed_experts for _, experts in self._moe_layers
        }
        shared_counts = {
            int(getattr(experts, "num_fused_shared_experts", 0))
            for _, experts in self._moe_layers
        }
        if len(expert_counts) != 1 or len(shared_counts) != 1:
            raise RuntimeError(
                "ROCm VMM DWDP requires uniform routed/shared expert counts"
            )
        num_routed = expert_counts.pop()
        num_shared = shared_counts.pop()
        if num_routed % self.dwdp_size != 0:
            raise ValueError(
                f"num_routed_experts ({num_routed}) must be divisible by "
                f"dwdp_size ({self.dwdp_size})"
            )

        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_fused_shared_experts=num_shared,
        )
        local_routed = self.layout.num_experts_per_worker
        local_count = self.layout.local_expert_end - self.layout.local_expert_start

        local_params: Dict[Tuple[int, str], torch.Tensor] = {}
        reference_main_names = None
        for layer_idx, experts in self._moe_layers:
            if experts.quant_method is None:
                raise RuntimeError(f"Layer {layer_idx} has no MoE quantization method")
            schema = experts.quant_method.get_dwdp_tensor_schema(experts)
            schema.validate(experts)
            self._schemas[layer_idx] = schema
            if reference_main_names is None:
                reference_main_names = schema.main_weights
            elif schema.main_weights != reference_main_names:
                raise RuntimeError(
                    "ROCm VMM DWDP requires identical main weight names across layers"
                )

            for name in schema.main_weights:
                original = experts.quant_method.get_dwdp_tensor(experts, name)
                required = local_routed + num_shared
                if original.ndim == 0 or original.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {layer_idx} tensor {name} has shape "
                        f"{tuple(original.shape)}, expected at least {required} experts"
                    )
                local = original.narrow(0, 0, local_count)
                if not local.is_contiguous():
                    local = local.contiguous()
                local_params[(layer_idx, name)] = local
                self._old_main_tensors.append(original)

        specs = build_layer_weight_specs(local_params, self.layout.num_experts)
        transport = None
        weight_buffer = None
        try:
            group = get_parallel().tp_group
            transport = RocmVmmTransport.create(
                layer_weight_specs=specs,
                local_params=local_params,
                group=group,
                layout=self.layout,
                device_id=self.device_id,
            )
            weight_buffer = WeightBuffer.create(
                layer_weight_specs=specs,
                handles=transport.handle_set,
                local_start=self.layout.local_expert_start,
                local_end=self.layout.local_expert_end,
                dwdp_size=self.dwdp_size,
                device_id=self.device_id,
                vmm_ops=hip_vmm,
            )
            self._fill_edge_bytes(weight_buffer, transport.peer_views)
            self._replicate_side_tensors(group)

            manager = DWDPWeightManager(
                weight_buffer=weight_buffer,
                peer_views=transport.peer_views,
                peer_ranges=self.layout.peer_ranges,
                moe_layer_indices=[idx for idx, _ in self._moe_layers],
                weight_names=list(reference_main_names),
                dwdp_rank=self.dwdp_rank,
                dwdp_size=self.dwdp_size,
                transport=transport,
                copy_engine=HsaSdmaCopyEngine.create_or_none(),
                peer_device_ids=transport.peer_device_ids,
            )
            for layer_idx, experts in self._moe_layers:
                experts.bind_full_expert_weights(
                    {
                        name: weight_buffer.get_full_tensor(layer_idx, name)
                        for name in reference_main_names
                    }
                )
            self._weight_manager = manager

            for tensor in self._old_main_tensors:
                tensor.untyped_storage().resize_(0)
            self._old_main_tensors.clear()
            torch.cuda.empty_cache()
        except Exception:
            if weight_buffer is not None:
                weight_buffer.release()
            if transport is not None:
                transport.release()
            raise

        logger.info(
            "ROCm VMM DWDP setup complete: rank=%d/%d, routed=%d, shared=%d",
            self.dwdp_rank,
            self.dwdp_size,
            num_routed,
            num_shared,
        )

    def _fill_edge_bytes(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
    ) -> None:
        assert self.layout is not None
        local_start = self.layout.local_expert_start
        local_end = self.layout.local_expert_end
        peer_ranges = self.layout.peer_ranges
        for layer_idx in weight_buffer.layer_indices:
            for name in weight_buffer.weight_names(layer_idx):
                edge = weight_buffer.get_edge_info(layer_idx, name)
                if edge.leading_edge == 0 and edge.trailing_edge == 0:
                    continue
                full_tensor = weight_buffer.get_full_tensor(layer_idx, name)
                if edge.leading_edge > 0 and local_start > 0:
                    previous = local_start - 1
                    peer = lookup_owner(previous, peer_ranges)
                    peer_start, _ = peer_ranges[peer]
                    full_tensor[previous].copy_(
                        peer_views[(peer, layer_idx, name)][previous - peer_start]
                    )
                if edge.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                    following = local_end
                    peer = lookup_owner(following, peer_ranges)
                    peer_start, _ = peer_ranges[peer]
                    full_tensor[following].copy_(
                        peer_views[(peer, layer_idx, name)][following - peer_start]
                    )
        torch.cuda.synchronize(self.device_id)

    def _replicate_side_tensors(self, group) -> None:
        assert self.layout is not None
        local_routed = self.layout.num_experts_per_worker
        num_shared = self.layout.num_fused_shared_experts
        for layer_idx, experts in self._moe_layers:
            schema = self._schemas[layer_idx]
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
            for name in names:
                storage_tensor = getattr(experts, name)
                if isinstance(storage_tensor, torch.nn.Parameter):
                    storage_tensor = storage_tensor.data
                tensor = experts.quant_method.get_dwdp_tensor(experts, name)
                required = local_routed + num_shared
                if tensor.ndim == 0 or tensor.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {layer_idx} side tensor {name} has shape "
                        f"{tuple(tensor.shape)}, expected at least {required} experts"
                    )
                routed = tensor.narrow(0, 0, local_routed)
                shards = [torch.empty_like(routed) for _ in range(self.dwdp_size)]
                dist.all_gather(shards, routed, group=group.device_group)
                full = torch.cat(shards, dim=0)
                if num_shared:
                    full = torch.cat(
                        [full, tensor.narrow(0, local_routed, num_shared)],
                        dim=0,
                    )
                experts.replace_expert_tensor(
                    name,
                    restore_storage_rank(storage_tensor, full),
                )

    def prefetch_first_layers(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.prefetch_first_layers()

    def wait_prefetch(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.wait_prefetch(layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.record_compute_and_prefetch_next(layer_idx)

    def cleanup(self, synchronize_ranks: bool = True) -> None:
        if self._weight_manager is None:
            return
        self._weight_manager.release()
        self._weight_manager = None
        self._schemas.clear()
