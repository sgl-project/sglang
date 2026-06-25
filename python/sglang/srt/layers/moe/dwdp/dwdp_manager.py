"""DwdpManager: top-level lifecycle for DWDP weight prefetch.

Global singleton that orchestrates:
  1. setup(model): after weight loading — transport, VA, prefetch init
  2. prefetch_first_layers(): at forward_extend entry
  3. wait_prefetch(layer_id): before each MoE layer forward
  4. record_compute_and_prefetch_next(layer_id): after each MoE layer forward
  5. cleanup(): on shutdown
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.moe.dwdp.specs import (
    DwdpExpertLayout,
    LayerWeightSpecs,
    WeightSpec,
    lookup_owner,
)
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager

logger = logging.getLogger(__name__)

# Global singleton
_GLOBAL_DWDP_MANAGER: Optional[DwdpManager] = None


def get_global_dwdp_manager() -> Optional[DwdpManager]:
    return _GLOBAL_DWDP_MANAGER


def set_global_dwdp_manager(manager: Optional[DwdpManager]) -> None:
    global _GLOBAL_DWDP_MANAGER
    _GLOBAL_DWDP_MANAGER = manager


# Weight parameter names to handle via the full DWDP pipeline.
# These are the expert weight tensors that go through
# Transport → WeightBuffer → WeightManager.
_EXPERT_WEIGHT_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_sf",
    "w2_weight_sf",
    "w1_alpha",
    "w2_alpha",
)


class DwdpManager:
    """Lifecycle manager for DWDP weight prefetch."""

    def __init__(self, server_args):
        from sglang.srt.runtime_context import get_parallel

        self.dwdp_size = server_args.dwdp_size
        self.dwdp_rank = get_parallel().dwdp_rank
        self.device_id = torch.cuda.current_device()

        self.layout = DwdpExpertLayout(
            num_routed_experts=0,  # set during setup()
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_experts_per_worker=server_args.dwdp_num_experts_per_worker,
        )

        self._weight_manager: Optional[DWDPWeightManager] = None
        self._moe_layer_indices: List[int] = []
        self._setup_done = False

    def setup(self, model: nn.Module) -> None:
        """Run after weight loading. Sets up transport, VA, and prefetch."""
        if self._setup_done:
            return

        from sglang.srt.runtime_context import get_parallel

        logger.info(
            f"[DwdpManager] Starting setup: rank={self.dwdp_rank}/{self.dwdp_size}"
        )

        # 1. Collect MoE expert weight params from model
        local_params, weight_names, moe_layer_indices, num_routed_experts = (
            self._collect_moe_params(model)
        )
        self._moe_layer_indices = moe_layer_indices

        # Update layout with actual expert count
        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed_experts,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_experts_per_worker=(
                self.layout.num_experts_per_worker
                if self.layout.num_experts_per_worker
                else None
            ),
        )

        logger.info(
            f"[DwdpManager] Layout: {num_routed_experts} experts, "
            f"local [{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"prefetch_per_peer={self.layout.num_prefetch_experts}"
        )

        # 2. Build weight specs
        layer_weight_specs = self._build_weight_specs(
            local_params, weight_names, num_routed_experts
        )

        # 3. Transport: alloc handles, copy weights, exchange
        group = get_parallel().dwdp_group
        transport = DWDPTransport.create(
            layer_weight_specs=layer_weight_specs,
            local_params=local_params,
            group=group,
            layout=self.layout,
            device_id=self.device_id,
        )

        # 4. WeightBuffer: composite VA layout
        weight_buffer = WeightBuffer.create(
            layer_weight_specs=layer_weight_specs,
            handles=transport.handle_set,
            local_start=self.layout.local_expert_start,
            local_end=self.layout.local_expert_end,
            dwdp_size=self.dwdp_size,
            device_id=self.device_id,
        )

        # 5. Fill edge bytes
        self._fill_edge_bytes(
            weight_buffer=weight_buffer,
            peer_views=transport.peer_views,
            peer_ranges=self.layout.peer_ranges,
        )

        # 6. WeightManager
        self._weight_manager = DWDPWeightManager(
            weight_buffer=weight_buffer,
            peer_views=transport.peer_views,
            peer_ranges=self.layout.peer_ranges,
            moe_layer_indices=moe_layer_indices,
            weight_names=weight_names,
            dwdp_rank=self.dwdp_rank,
            dwdp_size=self.dwdp_size,
        )
        # Keep transport alive (handles underpin VA mappings)
        self._weight_manager._transport = transport

        # 7. Fixup MoE backends: patch ep_size=1, bind composite VA tensors
        self._fixup_moe_backends(model, weight_buffer, num_routed_experts)

        # 8. Allgather small scale params
        self._allgather_small_params(model, group)

        self._setup_done = True
        logger.info("[DwdpManager] Setup complete.")

    # ------------------------------------------------------------------
    # Forward hooks (called by DeepseekV2MoE.forward_dwdp)
    # ------------------------------------------------------------------

    def prefetch_first_layers(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.prefetch_first_layers()

    def wait_prefetch(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.wait_prefetch(layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        if self._weight_manager is not None:
            self._weight_manager.record_compute_and_prefetch_next(layer_idx)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._weight_manager is not None:
            self._weight_manager.release()
            self._weight_manager = None
        self._setup_done = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_moe_params(self, model: nn.Module) -> Tuple[
        Dict[Tuple[int, str], torch.Tensor],
        List[str],
        List[int],
        int,
    ]:
        """Collect expert weight tensors from all MoE layers.

        Returns (local_params, weight_names, moe_layer_indices, num_routed_experts).
        """
        local_params: Dict[Tuple[int, str], torch.Tensor] = {}
        weight_names_set: List[str] = []
        moe_layer_indices: List[int] = []
        num_routed_experts = 0

        # Navigate to the model with .layers
        decoder = self._get_decoder_model(model)
        layers = decoder.layers

        for layer_idx, layer in enumerate(layers):
            experts = self._get_experts_module(layer)
            if experts is None:
                continue

            moe_layer_indices.append(layer_idx)

            # Detect num_routed_experts from first MoE layer
            if num_routed_experts == 0:
                nre = getattr(experts, "_num_global_routed", None)
                if nre is None:
                    nre = getattr(experts, "num_experts", None)
                if nre is not None:
                    num_routed_experts = nre

            found = []
            for wname in _EXPERT_WEIGHT_NAMES:
                param = getattr(experts, wname, None)
                if param is not None and isinstance(
                    param, (torch.Tensor, nn.Parameter)
                ):
                    data = param.data if isinstance(param, nn.Parameter) else param
                    if data.numel() > 0:
                        local_params[(layer_idx, wname)] = data
                        found.append(wname)

            if not weight_names_set and found:
                weight_names_set = found

        logger.info(
            f"[DwdpManager] Collected params from {len(moe_layer_indices)} MoE layers, "
            f"weights={weight_names_set}, num_routed_experts={num_routed_experts}"
        )
        return local_params, weight_names_set, moe_layer_indices, num_routed_experts

    def _build_weight_specs(
        self,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        weight_names: List[str],
        num_experts_total: int,
    ) -> LayerWeightSpecs:
        specs: LayerWeightSpecs = {}
        layers_seen: Dict[int, List[str]] = {}
        for li, wn in local_params:
            layers_seen.setdefault(li, []).append(wn)

        for li in sorted(layers_seen):
            specs[li] = {}
            for wn in weight_names:
                key = (li, wn)
                if key not in local_params:
                    continue
                p = local_params[key]
                chunk_shape = tuple(p.shape)
                full_shape = (num_experts_total,) + chunk_shape[1:]
                specs[li][wn] = WeightSpec(
                    num_experts=num_experts_total,
                    chunk_shape=chunk_shape,
                    full_shape=full_shape,
                    dtype=p.dtype,
                )
        return specs

    def _fill_edge_bytes(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges,
    ) -> None:
        """Fill page-alignment edge bytes from peers (one-time setup)."""
        local_start = self.layout.local_expert_start
        local_end = self.layout.local_expert_end

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

    def _fixup_moe_backends(
        self,
        model: nn.Module,
        weight_buffer: WeightBuffer,
        num_experts_total: int,
    ) -> None:
        """Patch each FusedMoE to see all experts (ep_size=1) and bind VA tensors."""
        decoder = self._get_decoder_model(model)

        for li in self._moe_layer_indices:
            layer = decoder.layers[li]
            experts = self._get_experts_module(layer)
            if experts is None:
                continue

            # Patch EP attributes
            experts.moe_ep_size = 1
            experts.moe_ep_rank = 0
            experts._num_local_routed = experts._num_global_routed
            experts.num_local_experts = experts._num_global_routed + getattr(
                experts, "_has_fused_shared", 0
            )

            # Bind composite VA tensors to param.data
            for name in weight_buffer.weight_names(li):
                param = getattr(experts, name, None)
                if param is not None:
                    full_tensor = weight_buffer.get_full_tensor(li, name)
                    if isinstance(param, nn.Parameter):
                        param.data = full_tensor
                    else:
                        setattr(experts, name, full_tensor)

            logger.debug(
                f"[DwdpManager] Patched layer {li}: ep_size=1, all experts bound"
            )

    def _allgather_small_params(self, model: nn.Module, group) -> None:
        """Allgather small EP-sharded params (alpha, bias) via torch.distributed."""
        decoder = self._get_decoder_model(model)
        local_experts = self.layout.num_experts_per_worker
        num_total = self.layout.num_routed_experts

        for li in self._moe_layer_indices:
            layer = decoder.layers[li]
            experts = self._get_experts_module(layer)
            if experts is None:
                continue

            # Look for small scale/alpha params that are EP-sharded
            for pname in list(vars(experts).keys()):
                param = getattr(experts, pname, None)
                if not isinstance(param, (torch.Tensor, nn.Parameter)):
                    continue
                data = param.data if isinstance(param, nn.Parameter) else param
                if data.ndim == 0 or data.shape[0] != local_experts:
                    continue
                # Skip large weight params (handled by transport)
                if pname in _EXPERT_WEIGHT_NAMES:
                    continue
                if not any(kw in pname for kw in ("alpha", "scale", "bias")):
                    continue

                # Allgather
                shards = [torch.empty_like(data) for _ in range(self.dwdp_size)]
                dist.all_gather(shards, data, group=group.device_group)
                full = torch.cat(shards, dim=0)[:num_total].contiguous()

                if isinstance(param, nn.Parameter):
                    param.data = full
                else:
                    setattr(experts, pname, full)

                logger.debug(
                    f"[DwdpManager] Layer {li}: allgathered {pname} "
                    f"({local_experts} -> {full.shape[0]})"
                )

    @staticmethod
    def _get_decoder_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "layers"):
            return model
        for attr in ("transformer", "decoder", "backbone"):
            child = getattr(model, attr, None)
            if child is not None and hasattr(child, "layers"):
                return child
        raise RuntimeError(
            f"Cannot find decoder model with .layers in {type(model).__name__}"
        )

    @staticmethod
    def _get_experts_module(layer: nn.Module) -> Optional[nn.Module]:
        """Find the FusedMoE experts module from a decoder layer."""
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None
        # DeepseekV2MoE.experts is the FusedMoE
        experts = getattr(mlp, "experts", None)
        if experts is not None and hasattr(experts, "w13_weight"):
            return experts
        # Maybe the mlp itself is FusedMoE
        if hasattr(mlp, "w13_weight"):
            return mlp
        return None
