import logging
import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import (
    EMBEDDING_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_lm_head_lora_b_shard_size,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
)
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class EmptySlot:
    """
    Singleton class to represent an empty slot in the memory pool.
    This is used to improve readability by not using special str as a placeholder.
    """

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
        lora_added_tokens_size: int,
        experts_shared_outer_loras: bool = False,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: int = lora_added_tokens_size
        self.max_lora_rank: int = max_lora_rank
        self.target_modules: Set[str] = target_modules
        self.experts_shared_outer_loras: bool = experts_shared_outer_loras

        # Initialize eviction policy
        self.eviction_policy = get_eviction_policy(eviction_policy)

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # Standard LoRA (3D): [num_loras, rank, hidden_dim]
        # MoE LoRA (4D): [num_loras, num_experts, rank, hidden_dim]
        # The dimensionality is determined by the module type (MoE vs standard)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.lm_head_A_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_B_buffer: Dict[str, torch.Tensor] = {}
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        self.embedding_dim: int = self.base_hf_config.hidden_size

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as `EmptySlot` for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_loras_per_batch

        # Cache lm_head shard_indices from the base model so that buffer
        # allocation uses the same sharding as the base ParallelLMHead layer.
        self.lm_head_shard_indices = None
        if "lm_head" in target_modules and tp_size > 1:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            for _, module in base_model.named_modules():
                if isinstance(module, ParallelLMHead):
                    self.lm_head_shard_indices = module.shard_indices
                    break

        self.init_buffers(base_model)

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """
        Check if the memory pool can support the given LoRA adapters.
        """

        def _can_support(config: LoRAConfig) -> bool:
            """
            Check if the memory pool can support a single LoRA adapter.
            """
            if config.r > self.max_lora_rank:
                return False
            if config.lora_added_tokens_size > self.lora_added_tokens_size:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            if "all" in target_module_names:
                return True
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def is_moe_module(self, module_name: str) -> bool:
        """Check if module is part of MoE experts."""
        return "moe" in module_name

    @staticmethod
    def _get_num_experts(base_model: torch.nn.Module) -> int:
        cfg = base_model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        return (
            getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None)
            or getattr(cfg, "n_routed_experts", None)
            or 1
        )

    def _get_standard_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """Get 3D shape for standard (non-MoE) modules."""
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (self.max_loras_per_batch, max_lora_dim * c, input_dim)

    def get_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Get shape for LoRA A weights. Automatically returns 3D or 4D based on module type.

        Returns:
            - Standard: [num_loras, rank, hidden_dim]
            - MoE: [num_loras, num_experts, rank, hidden_dim]
        """
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)

        if self.is_moe_module(module_name):
            num_experts = self._get_num_experts(base_model)
            expert_dim = num_experts
            if self.experts_shared_outer_loras and module_name == "gate_up_proj_moe":
                expert_dim = 1
            return (
                self.max_loras_per_batch,
                expert_dim,
                max_lora_dim * c,
                input_dim,
            )
        else:
            return (self.max_loras_per_batch, max_lora_dim * c, input_dim)

    def get_embedding_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )
        # Embedding LoRA A is kept unsharded (full vocab) across TP ranks.
        # Each rank does a full lookup; no vocab-dimension splitting needed.
        return (
            self.max_loras_per_batch,
            max_lora_dim,
            input_dim,
        )

    def get_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Get shape for LoRA B weights. Automatically returns 3D or 4D based on module type.

        Returns:
            - Standard: [num_loras, output_dim, rank]
            - MoE: [num_loras, num_experts, output_dim, rank]
        """
        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, layer_idx
        )
        if self.tp_size > 1 and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            output_dim = divide(output_dim, self.tp_size)

        # Check if MoE module and return appropriate shape
        if self.is_moe_module(module_name):
            num_experts = self._get_num_experts(base_model)
            expert_dim = num_experts
            if self.experts_shared_outer_loras and module_name == "down_proj_moe":
                expert_dim = 1
            return (self.max_loras_per_batch, expert_dim, output_dim, max_lora_dim)
        else:
            return (self.max_loras_per_batch, output_dim, max_lora_dim)

    def get_embedding_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )
        # lm_head is column-parallel so B is sharded; embed_tokens B stays
        # unsharded (base output is all-reduced to full embed_dim).
        if module_name == "lm_head":
            output_dim = get_lm_head_lora_b_shard_size(
                output_dim,
                shard_indices=self.lm_head_shard_indices,
            )
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            cfg = base_model.config
            if hasattr(cfg, "get_text_config"):
                cfg = cfg.get_text_config()
            has_shared_experts = (
                hasattr(cfg, "shared_expert_intermediate_size")
                and cfg.shared_expert_intermediate_size > 0
            ) or (getattr(cfg, "n_shared_experts", 0) or 0) > 0
            has_moe = self._get_num_experts(base_model) > 1

            # Shape functions automatically handle both 3D (standard) and 4D (MoE)
            target_modules = target_modules - set(EMBEDDING_NAMES)
            for module_name in target_modules:
                # Special handling for ambiguous target modules that can be in different contexts
                ambiguous_modules = {"gate_up_proj", "down_proj"}
                if module_name in ambiguous_modules and has_moe:
                    # Allocate shared expert version (3D) only when model has shared experts
                    if has_shared_experts:
                        buffer[module_name] = [
                            torch.zeros(
                                get_lora_shape_fn(
                                    module_name, base_model, self.max_lora_rank, idx
                                ),
                                dtype=self.dtype,
                                device=device,
                            )
                            for idx in range(self.num_layer)
                        ]

                    # MoE expert version (4D)
                    moe_key = f"{module_name}_moe"
                    buffer[moe_key] = [
                        torch.empty(
                            get_lora_shape_fn(
                                moe_key, base_model, self.max_lora_rank, idx
                            ),
                            dtype=self.dtype,
                            device=device,
                        )
                        for idx in range(self.num_layer)
                    ]
                else:
                    # Standard allocation for unambiguous modules
                    buffer[module_name] = [
                        torch.empty(
                            get_lora_shape_fn(
                                module_name,
                                base_model,
                                self.max_lora_rank,
                                idx,
                            ),
                            dtype=self.dtype,
                            device=device,
                        )
                        for idx in range(self.num_layer)
                    ]

        def init_embedding_buffer(
            buffer: Dict[str, torch.Tensor],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[int], Tuple[int]],
        ):
            target_modules = target_modules & set(EMBEDDING_NAMES)
            for module_name in target_modules:
                buffer[module_name] = torch.empty(
                    get_lora_shape_fn(
                        module_name,
                        base_model,
                        self.max_lora_rank,
                        0,
                    ),
                    dtype=self.dtype,
                    device=device,
                )

        if self.lora_added_tokens_size > 0:
            self.new_embeddings_buffer["input_embeddings"] = torch.empty(
                (
                    self.max_loras_per_batch,
                    self.lora_added_tokens_size,
                    self.embedding_dim,
                ),
                dtype=self.dtype,
                device=device,
            )

        if "embed_tokens" in self.target_modules:
            init_embedding_buffer(
                self.embedding_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.embedding_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        if "lm_head" in self.target_modules:
            init_embedding_buffer(
                self.lm_head_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.lm_head_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        init_buffer(
            self.A_buffer,
            self.target_modules,
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.target_modules,
            self.get_lora_B_shape,
        )

    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_refs: Dict[str, LoRARef],
        lora_embed_tokens_module: Optional[BaseLayerWithLoRA],
        lora_lm_head_module: Optional[BaseLayerWithLoRA],
    ):
        def get_available_buffer_slot():
            # 1. Prioritize empty slots
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            # 2. Memory pool is full, need to evict using policy
            candidates = set()

            for buffer_id in range(self.max_loras_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]

                # Skip if this adapter is needed by current batch
                if uid in cur_uids:
                    continue

                # Skip if this adapter is pinned
                if uid is not None:
                    lora_ref = lora_refs.get(uid)
                    if lora_ref and lora_ref.pinned:
                        continue

                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of active (pinned) loras is less than max_loras_per_batch."
                )

            # Prefer evicting LoRA adapters over the base model (None).
            # Only evict None when the batch consists entirely of LoRA requests
            # and no other adapters can be evicted.
            non_none_candidates = candidates - {None}
            if non_none_candidates:
                # Prioritize evicting actual LoRA adapters
                candidates_to_use = non_none_candidates
            else:
                # Only None is available for eviction (batch is all LoRA requests)
                candidates_to_use = candidates

            # Select victim using eviction policy
            victim_uid = self.eviction_policy.select_victim(candidates_to_use)

            # Evict the selected victim
            victim_buffer_id = self.uid_to_buffer_id[victim_uid]
            self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting LoRA {victim_uid} from buffer slot {victim_buffer_id}."
            )
            return victim_buffer_id

        # Mark all adapters in current batch as used (for LRU tracking)
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid,
                    buffer_id,
                    lora_adapter,
                    lora_modules,
                    lora_embed_tokens_module,
                    lora_lm_head_module,
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_embed_tokens_module: Optional[BaseLayerWithLoRA],
        lora_lm_head_module: Optional[BaseLayerWithLoRA],
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor, weight: Optional[torch.Tensor]
        ):
            if weight is None:
                # If the particular weight is not present in the adapter, we initialize the buffer to zero
                # to avoid contamination from the residual weight of the evicted adapters.
                buffer_view.zero_()
            else:
                assert (
                    buffer_view.shape == weight.shape
                ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."
                buffer_view.copy_(weight, non_blocking=True)

        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0

            for k in self.embedding_A_buffer.keys():
                self.embedding_A_buffer[k][buffer_id] = 0

            for k in self.lm_head_A_buffer.keys():
                self.lm_head_A_buffer[k][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.r
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            # - Standard: module_name -> torch.Tensor
            # - MoE: module_name -> Dict[expert_id -> torch.Tensor]
            temp_A_buffer: Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_buffer: Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]] = {
                target_module: None for target_module in self.B_buffer
            }

            for name, weights in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)

                # Check if this is an MoE weight (has expert index in name)
                expert_match = re.search(r"experts\.(\d+)\.", name)

                if expert_match:
                    # Per-expert MoE weight — 2D tensors, one per expert
                    target_module = target_module + "_moe"
                    if temp_A_buffer[target_module] is None:
                        temp_A_buffer[target_module] = {}
                        temp_B_buffer[target_module] = {}

                    expert_id = int(expert_match.group(1))
                    if "lora_A" in name:
                        temp_A_buffer[target_module][expert_id] = weights
                    else:
                        temp_B_buffer[target_module][expert_id] = weights
                elif "experts" in name and weights.dim() == 3:
                    # Shared outer MoE weight — 3D tensor [expert_dim, rank, hidden]
                    target_module = target_module + "_moe"
                    if "lora_A" in name:
                        temp_A_buffer[target_module] = weights
                    else:
                        temp_B_buffer[target_module] = weights
                else:
                    # Standard weight — single tensor per module
                    if "lora_A" in name:
                        temp_A_buffer[target_module] = weights
                    else:
                        temp_B_buffer[target_module] = weights

            if self.tp_size > 1:
                cur_layer_modules = lora_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    # TODO (Jonahcb): check if the code can be refactored to avoid the special handling for FusedMoEWithLoRA
                    # Handle FusedMoEWithLoRA specially - it contains multiple target modules
                    from sglang.srt.lora.layers import FusedMoEWithLoRA

                    if isinstance(module, FusedMoEWithLoRA):
                        moe_target_modules = ["gate_up_proj_moe", "down_proj_moe"]
                        for target_module in moe_target_modules:
                            if temp_A_buffer.get(target_module) is not None:
                                temp_A_buffer[target_module] = (
                                    module.slice_moe_lora_a_weights(
                                        temp_A_buffer[target_module],
                                        self.tp_rank,
                                        target_module,
                                    )
                                )
                            if temp_B_buffer.get(target_module) is not None:
                                temp_B_buffer[target_module] = (
                                    module.slice_moe_lora_b_weights(
                                        temp_B_buffer[target_module],
                                        self.tp_rank,
                                        target_module,
                                    )
                                )

                        continue

                    # Handle regular modules
                    target_module = get_target_module_name(
                        module_name, self.target_modules
                    )

                    if temp_A_buffer[target_module] is None:
                        # Skip weight slicing if the weight is not present in the adapter
                        continue

                    # Handle standard modules
                    temp_A_buffer[target_module] = module.slice_lora_a_weights(
                        temp_A_buffer[target_module], self.tp_rank
                    )
                    temp_B_buffer[target_module] = module.slice_lora_b_weights(
                        temp_B_buffer[target_module], self.tp_rank
                    )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                target_buffer = self.A_buffer[name][layer_id]

                if name in ["gate_up_proj_moe", "down_proj_moe"]:
                    if self.experts_shared_outer_loras and name == "gate_up_proj_moe":
                        if isinstance(weights, torch.Tensor) and weights.dim() == 3:
                            buffer_view = target_buffer[
                                buffer_id, 0, : lora_rank * c, :
                            ]
                            load_lora_weight_tensor(buffer_view, weights[0])
                        elif isinstance(weights, dict) and len(weights) > 0:
                            rep = next(iter(weights.values()))
                            buffer_view = target_buffer[
                                buffer_id, 0, : lora_rank * c, :
                            ]
                            load_lora_weight_tensor(buffer_view, rep)
                        else:
                            raise ValueError(
                                f"Unexpected weight format for shared outer gate_up_proj_moe lora_A: "
                                f"type={type(weights)}, "
                                f"shape={weights.shape if isinstance(weights, torch.Tensor) else 'N/A'}"
                            )
                    elif isinstance(weights, torch.Tensor) and weights.dim() == 3:
                        for eid in range(weights.shape[0]):
                            buffer_view = target_buffer[
                                buffer_id, eid, : lora_rank * c, :
                            ]
                            load_lora_weight_tensor(buffer_view, weights[eid])
                    elif isinstance(weights, dict):
                        for expert_id, expert_weight in weights.items():
                            buffer_view = target_buffer[
                                buffer_id, expert_id, : lora_rank * c, :
                            ]
                            load_lora_weight_tensor(buffer_view, expert_weight)
                else:
                    buffer_view = target_buffer[buffer_id, : lora_rank * c, :]
                    load_lora_weight_tensor(buffer_view, weights)

            for name, weights in temp_B_buffer.items():
                target_buffer = self.B_buffer[name][layer_id]

                if name in ["gate_up_proj_moe", "down_proj_moe"]:
                    if self.experts_shared_outer_loras and name == "down_proj_moe":
                        if isinstance(weights, torch.Tensor) and weights.dim() == 3:
                            buffer_view = target_buffer[buffer_id, 0, :, :lora_rank]
                            w = weights[0]
                            if w is not None:
                                w = w * lora_adapter.scaling
                            load_lora_weight_tensor(buffer_view, w)
                        elif isinstance(weights, dict) and len(weights) > 0:
                            rep = next(iter(weights.values()))
                            buffer_view = target_buffer[buffer_id, 0, :, :lora_rank]
                            if rep is not None:
                                rep = rep * lora_adapter.scaling
                            load_lora_weight_tensor(buffer_view, rep)
                        else:
                            raise ValueError(
                                f"Unexpected weight format for shared outer down_proj_moe lora_B: "
                                f"type={type(weights)}, "
                                f"shape={weights.shape if isinstance(weights, torch.Tensor) else 'N/A'}"
                            )
                    elif isinstance(weights, torch.Tensor) and weights.dim() == 3:
                        for eid in range(weights.shape[0]):
                            buffer_view = target_buffer[buffer_id, eid, :, :lora_rank]
                            w = weights[eid]
                            if w is not None:
                                w = w * lora_adapter.scaling
                            load_lora_weight_tensor(buffer_view, w)
                    elif isinstance(weights, dict):
                        for expert_id, expert_weight in weights.items():
                            buffer_view = target_buffer[
                                buffer_id, expert_id, :, :lora_rank
                            ]
                            w = expert_weight
                            if w is not None:
                                w = w * lora_adapter.scaling
                            load_lora_weight_tensor(buffer_view, w)
                else:
                    buffer_view = target_buffer[buffer_id, :, :lora_rank]
                    load_lora_weight_tensor(buffer_view, weights)

        if lora_adapter.embedding_layers:
            org_vocab_size = self.base_hf_config.vocab_size
            lora_added_tokens_size = lora_adapter.config.lora_added_tokens_size
            # Only when LoRA is applied to the embedding layer will it have the extra-token issue that needs to be resolved.
            # Load embeddings weights for extra tokens to buffer
            if lora_adapter.added_tokens_embeddings:
                for name, weights in lora_adapter.added_tokens_embeddings.items():
                    if "input_embeddings" in name:
                        buffer_view = self.new_embeddings_buffer["input_embeddings"][
                            buffer_id, :lora_added_tokens_size
                        ]
                        load_lora_weight_tensor(buffer_view, weights)

            # load vocab_emb and lm_head
            for name, weights in lora_adapter.embedding_layers.items():
                target_module = get_target_module_name(name, self.target_modules)
                if (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.embedding_A_buffer[target_module][
                        buffer_id,
                        :lora_rank,
                        : (org_vocab_size + lora_added_tokens_size),
                    ]
                    load_lora_weight_tensor(buffer_view, weights)
                elif (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    lora_b_weights = weights
                    # TP is supported by keeping embedding LoRA B unsharded;
                    # no slicing needed.

                    buffer_view = self.embedding_B_buffer[target_module][
                        buffer_id, :, :lora_rank
                    ]
                    load_lora_weight_tensor(buffer_view, lora_b_weights)

                elif (
                    target_module == "lm_head"
                    and "lm_head" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.lm_head_A_buffer[target_module][
                        # buffer_id, :, :lora_rank
                        buffer_id,
                        :lora_rank,
                        :,
                    ]
                    load_lora_weight_tensor(buffer_view, weights)
                elif (
                    target_module == "lm_head"
                    and "lm_head" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    lora_b_weights = weights
                    # Slice B along vocab dimension for this TP rank
                    if self.tp_size > 1 and lora_lm_head_module is not None:
                        lora_b_weights = lora_lm_head_module.slice_lora_b_weights(
                            lora_b_weights, self.tp_rank
                        )

                    buffer_view = self.lm_head_B_buffer[target_module][
                        buffer_id,
                        : lora_b_weights.shape[0],
                        :lora_rank,
                    ]
                    load_lora_weight_tensor(buffer_view, lora_b_weights)
        else:
            # Zero out embedding/lm_head buffers for adapters without embedding LoRA
            # to avoid using garbage values from uninitialized memory
            for k in self.embedding_A_buffer.keys():
                self.embedding_A_buffer[k][buffer_id].zero_()
            for k in self.embedding_B_buffer.keys():
                self.embedding_B_buffer[k][buffer_id].zero_()
            for k in self.lm_head_A_buffer.keys():
                self.lm_head_A_buffer[k][buffer_id].zero_()
            for k in self.lm_head_B_buffer.keys():
                self.lm_head_B_buffer[k][buffer_id].zero_()
            if (
                self.lora_added_tokens_size > 0
                and "input_embeddings" in self.new_embeddings_buffer
            ):
                self.new_embeddings_buffer["input_embeddings"][buffer_id].zero_()

    def get_embedding_tensor(
        self, target_module: str, lora_type: LoRAType
    ) -> Optional[torch.Tensor]:
        """
        Get LoRA tensor for non-layer modules (embed_tokens, lm_head).

        Args:
            target_module: Module name, either "embed_tokens" or "lm_head"
            lora_type: Either LoRAType.LORA_A or LoRAType.LORA_B

        Returns:
            The corresponding buffer tensor, or None if not available
        """

        if target_module == "added_tokens":
            if (
                self.lora_added_tokens_size is not None
                and self.lora_added_tokens_size > 0
            ):
                return self.new_embeddings_buffer["input_embeddings"]
            return None
        elif target_module == "embed_tokens":
            if lora_type == LoRAType.LORA_A:
                return self.embedding_A_buffer[target_module]
            return self.embedding_B_buffer[target_module]
        elif target_module == "lm_head":
            if lora_type == LoRAType.LORA_A:
                return self.lm_head_A_buffer[target_module]
            return self.lm_head_B_buffer[target_module]

        raise ValueError(
            f"Invalid target_module '{target_module}'. "
            f"Expected 'embed_tokens' or 'lm_head'."
        )

    def get_tensor(
        self, target_module: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        """
        Get LoRA tensor buffer (automatically handles both 3D and 4D tensors).

        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[target_module][layer_id]

        Args:
            target_module: Target module name (e.g., 'gate_up_proj' or 'gate_up_proj_moe' for MoE)
            layer_id: Layer index
            lora_type: LoRAType.LORA_A or LoRAType.LORA_B

        Returns:
            - 3D tensor [num_loras, rank, hidden] for standard modules
            - 4D tensor [num_loras, num_experts, rank, hidden] for MoE modules
        """
        buffer_dict = self.A_buffer if lora_type == LoRAType.LORA_A else self.B_buffer

        return buffer_dict[target_module][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
