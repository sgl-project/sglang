# SPDX-License-Identifier: Apache-2.0
"""
LoRA Manager for multi-LoRA batching in SGLang Diffusion.

This module provides support for using multiple LoRA adapters simultaneously
in a single batch, similar to SRT's multi-LoRA batching for LLMs.

Multi-GPU / Tensor Parallelism Support:
- LoRA weights are loaded on rank 0 and broadcast to other ranks to avoid
  redundant file I/O and ensure consistency.
- Batch LoRA assignments are synchronized across all TP ranks.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, TypedDict

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core import Req


class LoRAAdapterConfig(TypedDict):
    alpha: float
    rank: int


@dataclass
class LoRAAdapter:
    """Represents a loaded LoRA adapter."""

    nickname: str
    path: str
    weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    rank: int = 16
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=list)
    loaded: bool = True


class DiffusionLoRAManager:
    """
    Manages multiple LoRA adapters for diffusion models.

    Similar to SRT's LoRAManager but adapted for diffusion pipelines.
    Supports loading and managing multiple LoRA adapters that can be used
    simultaneously in a single batch.

    Multi-GPU Support:
    - In tensor parallel mode, rank 0 loads the LoRA weights and broadcasts
      them to other ranks to ensure consistency and reduce redundant I/O.
    - Batch LoRA assignments are synchronized via broadcast_object.
    """

    def __init__(
        self,
        max_loras_per_batch: int = 8,
        lora_memory_pool_size: int = 16,
        device: Optional[torch.device] = None,
        server_args=None,
        modules: Optional[Dict[str, torch.nn.Module]] = None,
    ):
        self.max_loras_per_batch = max_loras_per_batch
        self.lora_memory_pool_size = lora_memory_pool_size
        self.device = device or get_local_torch_device()
        self.server_args = server_args
        self.modules = modules or {}

        # Tensor parallelism awareness
        self.tp_rank = get_tp_rank() if dist.is_initialized() else 0
        self.tp_size = get_tp_world_size() if dist.is_initialized() else 1
        self._tp_group = None  # Lazy init to avoid issues before dist init

        # Storage for all loaded LoRA adapters
        self.lora_adapters: Dict[str, LoRAAdapter] = {}

        # Mapping: request_id -> lora_nickname
        self.request_lora_map: Dict[str, str] = {}

    @property
    def tp_group(self):
        """Lazy getter for TP group coordinator."""
        if self._tp_group is None and dist.is_initialized() and self.tp_size > 1:
            try:
                self._tp_group = get_tp_group()
            except Exception:
                self._tp_group = None
        return self._tp_group

    def _load_lora_weights_local(
        self, lora_path: str, lora_nickname: str
    ) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from disk/HF hub (local loading, no broadcast)."""
        lora_local_path = maybe_download_lora(lora_path)
        lora_state_dict = load_file(lora_local_path)

        weights: Dict[str, torch.Tensor] = {}

        if self.server_args is not None and self.modules:
            # Prefer mapping dicts from pipeline config; fall back to transformer attributes if present.
            # Be defensive: pipeline configs/modules evolve across upstream versions.
            arch_config = getattr(
                getattr(
                    getattr(self.server_args, "pipeline_config", None),
                    "dit_config",
                    None,
                ),
                "arch_config",
                None,
            )
            transformer = self.modules.get("transformer")

            param_names_mapping_dict = None
            lora_param_names_mapping_dict = None
            if arch_config is not None:
                param_names_mapping_dict = getattr(
                    arch_config, "param_names_mapping", None
                )
                lora_param_names_mapping_dict = getattr(
                    arch_config, "lora_param_names_mapping", None
                )
            if not param_names_mapping_dict and transformer is not None:
                param_names_mapping_dict = getattr(
                    transformer, "param_names_mapping", None
                )
            if not lora_param_names_mapping_dict and transformer is not None:
                lora_param_names_mapping_dict = getattr(
                    transformer, "lora_param_names_mapping", None
                )

            # If we don't have both mappings, fall back to the simple name mapping below.
            if not param_names_mapping_dict or not lora_param_names_mapping_dict:
                logger.warning(
                    "LoRA name mapping not found in pipeline config/modules; falling back to simple mapping."
                )
                for name, weight in lora_state_dict.items():
                    name = name.replace("diffusion_model.", "")
                    name = name.replace(".weight", "")
                    weights[name] = weight.to(self.device)
                return weights

            param_names_mapping_fn = get_param_names_mapping(param_names_mapping_dict)
            lora_param_names_mapping_fn = get_param_names_mapping(
                lora_param_names_mapping_dict
            )

            to_merge_params: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)

            for name, weight in lora_state_dict.items():
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                # misc-format -> HF-format
                name, _, _ = lora_param_names_mapping_fn(name)
                # HF-format (LoRA) -> SGLang-dit-format
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                    name
                )
                # for (in_dim, r) @ (r, out_dim * n), we only merge (r, out_dim * n)
                if merge_index is not None and "lora_B" in name:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        sorted_tensors = [
                            to_merge_params[target_name][i]
                            for i in range(num_params_to_merge)
                        ]
                        weight = torch.cat(sorted_tensors, dim=1)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in weights:
                    raise ValueError(
                        f"Dit target weight name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                weights[target_name] = weight.to(self.device)
        else:
            # Fallback: simple name mapping
            for name, weight in lora_state_dict.items():
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                weights[name] = weight.to(self.device)

        return weights

    def _load_lora_weights(
        self, lora_path: str, lora_nickname: str
    ) -> Dict[str, torch.Tensor]:
        """
        Load LoRA weights with multi-GPU support.

        In TP mode (tp_size > 1):
        - Rank 0 loads from disk and broadcasts to other ranks.
        - This avoids redundant file I/O and ensures all ranks have identical weights.

        In single-GPU mode (tp_size == 1):
        - Loads directly without broadcast overhead.
        """
        if self.tp_size == 1 or self.tp_group is None:
            # Single GPU: load directly
            return self._load_lora_weights_local(lora_path, lora_nickname)

        # Multi-GPU with TP: rank 0 loads, then broadcasts
        if self.tp_rank == 0:
            weights = self._load_lora_weights_local(lora_path, lora_nickname)
            # Broadcast weight dict from rank 0 to all other ranks
            self.tp_group.broadcast_tensor_dict(weights, src=0)
        else:
            # Receive weights from rank 0
            weights = self.tp_group.broadcast_tensor_dict(None, src=0)
            # Move received weights to local device
            if weights:
                for name in weights:
                    weights[name] = weights[name].to(self.device)

        return weights or {}

    def load_lora_adapter(
        self,
        lora_path: str,
        lora_nickname: str,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> LoRAAdapter:
        """
        Load a LoRA adapter from disk/HF hub.

        Args:
            lora_path: Path to LoRA adapter (local file or HF hub ID)
            lora_nickname: Nickname for the adapter
            rank: LoRA rank (optional, will infer from weights if not provided)
            alpha: LoRA alpha (optional, defaults to rank if not provided)
        """
        if lora_nickname in self.lora_adapters:
            logger.info("LoRA adapter %s already loaded", lora_nickname)
            return self.lora_adapters[lora_nickname]

        # Fix Bug 2: Use separate variable for distributed rank
        distributed_rank = dist.get_rank() if dist.is_initialized() else 0
        lora_rank = rank  # keep original parameter meaning (LoRA rank)

        logger.info(
            "Rank %d: Loading LoRA adapter: %s from %s",
            distributed_rank,
            lora_nickname,
            lora_path,
        )

        weights = self._load_lora_weights(lora_path, lora_nickname)

        if lora_rank is None and weights:
            for name, weight in weights.items():
                if "lora_A" in name:
                    lora_rank = weight.shape[0]
                    break

        adapter = LoRAAdapter(
            nickname=lora_nickname,
            path=lora_path,
            weights=weights,
            rank=lora_rank or 16,
            alpha=alpha or (lora_rank or 16.0),
            target_modules=[],
        )

        self.lora_adapters[lora_nickname] = adapter

        current_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(
            "Rank %d: Loaded LoRA adapter %s with %d weight tensors",
            current_rank,
            lora_nickname,
            len(weights),
        )
        return adapter

    def _sync_batch_info_across_ranks(
        self,
        active_loras: Set[str],
        request_lora_map: Dict[str, str],
        lora_paths: Dict[str, str],
    ) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
        """
        Synchronize batch LoRA information across all TP ranks.

        In multi-GPU mode, rank 0 broadcasts the batch info to ensure all ranks
        process the same set of LoRAs in the same order.

        Returns:
            Synchronized (active_loras, request_lora_map, lora_paths)
        """
        if self.tp_size == 1 or self.tp_group is None:
            return active_loras, request_lora_map, lora_paths

        # Pack batch info for broadcast
        batch_info = {
            "active_loras": list(active_loras),
            "request_lora_map": request_lora_map,
            "lora_paths": lora_paths,
        }

        if self.tp_rank == 0:
            synced_info = self.tp_group.broadcast_object(batch_info, src=0)
        else:
            synced_info = self.tp_group.broadcast_object(None, src=0)

        return (
            set(synced_info["active_loras"]),
            synced_info["request_lora_map"],
            synced_info["lora_paths"],
        )

    def prepare_lora_batch(
        self,
        requests: List["Req"],
        lora_layers: Dict[str, BaseLayerWithLoRA],
    ) -> Tuple[
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Dict[str, int],
        Dict[str, LoRAAdapterConfig],
    ]:
        """
        Prepare LoRA weights for a batch of requests.

        Multi-GPU Support:
        - Batch LoRA assignments are synchronized across all TP ranks.
        - Missing adapters are loaded with broadcast from rank 0.

        Returns:
            - layer_name -> {nickname -> (A, B)}
            - nickname -> index
            - nickname -> {alpha, rank}
        """
        active_loras: Set[str] = set()
        request_lora_map: Dict[str, str] = {}
        lora_paths: Dict[str, str] = {}  # nickname -> path for loading

        for req in requests:
            lora_nickname = getattr(req, "lora_nickname", None)
            if lora_nickname:
                active_loras.add(lora_nickname)
                if hasattr(req, "request_id") and req.request_id:
                    request_lora_map[req.request_id] = lora_nickname
                # Collect paths for potential loading
                lora_path = getattr(req, "lora_path", None)
                if lora_path and lora_nickname not in lora_paths:
                    lora_paths[lora_nickname] = lora_path

        # Sync batch info across all TP ranks
        active_loras, request_lora_map, lora_paths = self._sync_batch_info_across_ranks(
            active_loras, request_lora_map, lora_paths
        )

        if len(active_loras) > self.max_loras_per_batch:
            raise ValueError(
                f"Too many LoRAs in batch: {len(active_loras)} > {self.max_loras_per_batch}"
            )

        # Load missing adapters (with broadcast in TP mode)
        for nickname in active_loras:
            if nickname not in self.lora_adapters:
                lora_path = lora_paths.get(nickname)
                if not lora_path:
                    raise ValueError(
                        f"LoRA adapter '{nickname}' not loaded and no path provided"
                    )
                self.load_lora_adapter(lora_path, nickname)

        batch_lora_weights: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

        for layer_name, _layer in lora_layers.items():
            batch_lora_weights[layer_name] = {}
            for nickname in active_loras:
                adapter = self.lora_adapters[nickname]
                lora_A_name = f"{layer_name}.lora_A"
                lora_B_name = f"{layer_name}.lora_B"

                if lora_A_name in adapter.weights and lora_B_name in adapter.weights:
                    batch_lora_weights[layer_name][nickname] = (
                        adapter.weights[lora_A_name],
                        adapter.weights[lora_B_name],
                    )

        lora_nickname_to_index: Dict[str, int] = {
            nickname: idx for idx, nickname in enumerate(sorted(active_loras))
        }

        lora_adapter_configs: Dict[str, LoRAAdapterConfig] = {}
        for nickname in active_loras:
            adapter = self.lora_adapters[nickname]
            lora_adapter_configs[nickname] = {
                "alpha": adapter.alpha,
                "rank": adapter.rank,
            }

        self.request_lora_map = request_lora_map
        return batch_lora_weights, lora_nickname_to_index, lora_adapter_configs

    def get_lora_adapter(self, lora_nickname: str) -> Optional[LoRAAdapter]:
        return self.lora_adapters.get(lora_nickname)

    def unload_lora_adapter(self, lora_nickname: str) -> None:
        if lora_nickname in self.lora_adapters:
            del self.lora_adapters[lora_nickname]
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info("Rank %d: Unloaded LoRA adapter: %s", rank, lora_nickname)
