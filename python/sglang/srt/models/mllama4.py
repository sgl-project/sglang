import json as json_lib
import logging
import os
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import Llama4Config
from transformers.models.llama4.modeling_llama4 import (
    Llama4MultiModalProjector,
    Llama4VisionModel,
)

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, is_cpu

_is_cpu = is_cpu()
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Check if this is a text-only model (modelopt fp8 llama4 has no vision components)
        self.has_vision = self._has_vision_weights(config)
        if not self.has_vision:
            logger.warning(
                "No vision weights found in checkpoint. Model will run in text-only mode. "
                "Multimodal capabilities (image processing) will be unavailable."
            )

        if self.has_vision:
            self.vision_model = Llama4VisionModel(config.vision_config)
            self.multi_modal_projector = Llama4MultiModalProjector(config)
        else:
            self.vision_model = None
            self.multi_modal_projector = None

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config if hasattr(config, "text_config") else config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(
            config.text_config if hasattr(config, "text_config") else config
        )

    def _has_vision_weights(self, config) -> bool:
        """Check if the model has vision components by examining the checkpoint."""
        model_path = getattr(config, "_name_or_path", None)
        if not model_path:
            return False

        # Check if this is a local path first
        if os.path.isdir(model_path):
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                return self._check_vision_weights_in_index(index_file)

        # For HuggingFace models, we need to check the actual checkpoint
        # The config might say it's multimodal, but the checkpoint might be text-only
        try:
            # Try to access the HuggingFace cache directory
            from huggingface_hub import try_to_load_from_cache

            # Check if index file exists in cache
            index_file_path = try_to_load_from_cache(
                repo_id=model_path,
                filename="model.safetensors.index.json",
                cache_dir=None,
            )

            if index_file_path and os.path.exists(index_file_path):
                return self._check_vision_weights_in_index(index_file_path)

        except Exception:
            # If we can't access the cache, fall back to config-based detection
            pass

        # Fallback， assume text-only
        return False

    def _check_vision_weights_in_index(self, index_file: str) -> bool:
        """Check if the model.safetensors.index.json contains vision weights."""
        try:
            with open(index_file, "r") as f:
                index_data = json_lib.load(f)

            vision_patterns = ["vision_model", "vision_tower", "multi_modal_projector"]
            weight_names = index_data.get("weight_map", {}).keys()

            return any(
                pattern in weight_name
                for weight_name in weight_names
                for pattern in vision_patterns
            )
        except (OSError, json_lib.JSONDecodeError, KeyError):
            return False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(
        self,
        items: List[MultimodalDataItem],
    ) -> torch.Tensor:
        # For text-only models, return None or raise an error
        if not self.has_vision or self.vision_model is None:
            raise ValueError("Vision model not available for text-only checkpoint")

        pixel_values = (
            torch.concat([item.pixel_values for item in items])
            .to(next(self.vision_model.parameters()).device)
            .type(next(self.vision_model.parameters()).dtype)
        )

        image_outputs = self.vision_model(pixel_values, output_hidden_states=False)
        image_features = image_outputs.last_hidden_state
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat)
        return projected_vision_flat

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:

        # For text-only models, pass None for image_data_embedding_func
        image_embedding_func = self.get_image_feature if self.has_vision else None

        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hs

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.language_model.config.head_dim * n_heads
            attn_out = self.language_model.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            if _is_cpu:
                dim = self.language_model.config.original_total_num_kv_heads
            else:
                dim = self.language_model.config.num_key_value_heads
            loaded_weight = permute(loaded_weight, dim)
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            if _is_cpu:
                dim = self.language_model.config.original_num_attention_heads
            else:
                dim = self.language_model.config.num_attention_heads
            loaded_weight = permute(loaded_weight, dim)

        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        num_experts = (
            self.config.text_config.num_local_experts
            if hasattr(self.config, "text_config")
            else self.config.num_local_experts
        )

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        for name, loaded_weight in weights:
            if self._should_skip_weight(name):
                continue

            name = self._transform_weight_name(name)

            if "vision" not in name:
                name, loaded_weight = self.permute_qk_weight_for_rotary(
                    name, loaded_weight
                )

            if self._handle_scale_remapping(name, params_dict):
                continue

            if self._handle_stacked_params(
                name, loaded_weight, stacked_params_mapping, params_dict
            ):
                continue

            if self._handle_expert_weights(
                name, loaded_weight, expert_params_mapping, params_dict, num_experts
            ):
                continue

            self._handle_default_weight(name, loaded_weight, params_dict)

    def _should_skip_weight(self, name: str) -> bool:
        """Check if we should skip loading this weight."""
        return "vision" in name and not self.has_vision

    def _transform_weight_name(self, name: str) -> str:
        """Transform weight name by adding language_model prefix if needed."""
        if (
            not name.startswith("language_model.")
            and "vision" not in name
            and "multi_modal_projector" not in name
        ):
            return f"language_model.{name}"
        return name

    def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            return remapped_name is None
        return False

    def _handle_stacked_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        stacked_params_mapping: list,
        params_dict: dict,
    ) -> bool:
        """Handle stacked parameter loading. Returns True if handled."""
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name and "vision" not in name:
                transformed_name = name.replace(weight_name, param_name)
                param = params_dict[transformed_name]
                param.weight_loader(param, loaded_weight, shard_id)
                return True
        return False

    def _handle_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        expert_params_mapping: list,
        params_dict: dict,
        num_experts: int,
    ) -> bool:
        """Handle expert weight loading for MoE (Mixture of Experts) layers.

        Args:
            name: Parameter name from the checkpoint
            loaded_weight: The weight tensor to be loaded
            expert_params_mapping: Mapping of parameter names to expert configurations
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts in the MoE layer

        Returns:
            bool: True if the parameter was handled (is an expert parameter), False otherwise
        """
        if ".experts" not in name:
            return False

        if "experts.gate_up_proj" not in name and "experts.down_proj" not in name:
            return self._handle_other_expert_params(
                name, loaded_weight, expert_params_mapping, params_dict
            )

        if "scale" in name:
            return self._handle_expert_scale_params(
                name, loaded_weight, params_dict, num_experts
            )
        else:
            return self._handle_expert_weight_params(
                name, loaded_weight, params_dict, num_experts
            )

    def _handle_other_expert_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        expert_params_mapping: list,
        params_dict: dict,
    ) -> bool:
        """Handle expert parameters that are not gate_up_proj or down_proj weights.

        Args:
            name: Parameter name from the checkpoint
            loaded_weight: The weight tensor to be loaded
            expert_params_mapping: List of tuples mapping checkpoint names to model parameters
            params_dict: Dictionary of model parameters

        Returns:
            bool: True if parameter was found and handled, False otherwise
        """
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            if weight_name in name:
                transformed_name = name.replace(weight_name, param_name)
                param = params_dict[transformed_name]
                param.weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                return True
        return False

    def _transform_expert_name(
        self, name: str, is_weight: bool = False
    ) -> Tuple[str, str, List[str]]:
        """Transform expert parameter name and get shard information.

        Args:
            name: The original parameter name
            is_weight: Whether this is a weight parameter (adds _weight suffix)

        Returns:
            Tuple of (transformed_name, shard_id, shard_id_list)
        """
        suffix = "_weight" if is_weight else ""

        if ".gate_up_proj" in name:
            transformed_name = name.replace(
                ".experts.gate_up_proj", f".experts.w13{suffix}"
            )
            shard_id = "w13"
            shard_id_list = ["w1", "w3"]
        else:  # down_proj
            transformed_name = name.replace(
                ".experts.down_proj", f".experts.w2{suffix}"
            )
            shard_id = "w2"
            shard_id_list = ["w2"]

        return transformed_name, shard_id, shard_id_list

    def _handle_expert_scale_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        num_experts: int,
    ) -> bool:
        """Handle quantization scale parameters for expert weights.

        Args:
            name: Parameter name containing scale information
            loaded_weight: Scale tensor to be loaded
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts for broadcast operations

        Returns:
            bool: True (always handles scale parameters)
        """
        import re

        # Check if this matches the expert parameter pattern: experts.{expert_id}.{param_name}
        expert_match = re.search(r"experts\.(\d+)\.", name)

        # Transform name
        transformed_name, _, _ = self._transform_expert_name(name)

        if transformed_name not in params_dict:
            return True

        param = params_dict[transformed_name]

        # Handle scale parameters
        if expert_match:
            # If we have a specific expert ID, only load for that expert
            expert_id = int(expert_match.group(1))
            # For scale parameters, we can directly set the value
            param.data[expert_id] = loaded_weight
        else:
            # No expert ID found - this is a single scale for all experts
            # Load the same scale for all experts
            for expert_id in range(num_experts):
                param.data[expert_id] = loaded_weight

        return True

    def _handle_expert_weight_params(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        num_experts: int,
    ) -> bool:
        """Handle actual weight tensors for expert layers (gate_up_proj and down_proj).

        Args:
            name: Parameter name (should contain gate_up_proj or down_proj)
            loaded_weight: Weight tensor(s) to be loaded
            params_dict: Dictionary of model parameters
            num_experts: Total number of experts for tensor distribution

        Returns:
            bool: True (always handles weight parameters)
        """
        # Transform name and get shard info
        transformed_name, _, shard_id_list = self._transform_expert_name(
            name, is_weight=True
        )

        if ".gate_up_proj" in name:
            loaded_weight_list = loaded_weight.chunk(2, dim=-1)
        else:  # down_proj
            loaded_weight_list = [loaded_weight]

        for param_name, weight_chunk, shard_id in zip(
            [transformed_name] * len(shard_id_list), loaded_weight_list, shard_id_list
        ):
            if param_name not in params_dict:
                continue

            param = params_dict[param_name]
            weight_loader = param.weight_loader

            # Handle the case where loaded_weight might be a single tensor for all experts
            if weight_chunk.dim() == 2:
                # Single tensor case - load for all experts
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        weight_chunk.T,
                        param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
            else:
                # Multiple experts case - load each expert's weights
                for expert_id in range(num_experts):
                    weight_loader(
                        param,
                        weight_chunk[expert_id].T,
                        param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )

        return True

    def _handle_default_weight(
        self, name: str, loaded_weight: torch.Tensor, params_dict: dict
    ):
        """Handle default weight loading."""
        # Skip loading extra bias for GPTQ models
        if name.endswith(".bias") and name not in params_dict:
            return

        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if hasattr(self.language_model, "set_eagle3_layers_to_capture"):
            self.language_model.set_eagle3_layers_to_capture(layer_ids)

    def get_embed_and_head(self):
        # For EAGLE3, we delegate to the language model which should have this method
        # If the language model doesn't have lm_head (like EAGLE3), we return None for head
        embed = self.language_model.get_embed()
        if hasattr(self.language_model, "get_embed_and_head"):
            return self.language_model.get_embed_and_head()
        elif hasattr(self.language_model, "lm_head"):
            return embed, self.language_model.lm_head.weight
        else:
            # For EAGLE3, head might not be needed
            return embed, None

    def set_embed_and_head(self, embed, head):
        if hasattr(self.language_model, "set_embed_and_head"):
            return self.language_model.set_embed_and_head(embed, head)
        else:
            # For EAGLE3, only set embed
            return self.language_model.set_embed(embed)

    def get_embed(self):
        return self.language_model.get_embed()

    def set_embed(self, embed):
        return self.language_model.set_embed(embed)


EntryClass = Llama4ForConditionalGeneration
