import json
from collections.abc import Iterable
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import Llama4Config, Llama4VisionModel
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, is_cpu

_is_cpu = is_cpu()


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

        import json as json_lib
        import os

        # Get the model path from the config if available
        model_path = getattr(config, "_name_or_path", None)
        has_vision_weights = False

        if model_path and os.path.exists(
            os.path.join(model_path, "model.safetensors.index.json")
        ):
            try:
                with open(
                    os.path.join(model_path, "model.safetensors.index.json"), "r"
                ) as f:
                    index_data = json_lib.load(f)
                # Check if any vision model weights exist in the checkpoint
                vision_weight_patterns = [
                    "vision_model",
                    "vision_tower",
                    "multi_modal_projector",
                ]
                has_vision_weights = any(
                    any(pattern in weight_name for pattern in vision_weight_patterns)
                    for weight_name in index_data.get("weight_map", {}).keys()
                )
            except:
                # Fallback: assume text-only since most checkpoints are text-only
                has_vision_weights = False
        else:
            # No checkpoint index found, assume text-only
            has_vision_weights = False

        self.has_vision = has_vision_weights

        if self.has_vision:
            self.vision_model = Llama4VisionModel(config.vision_config)
            self.multi_modal_projector = Llama4MultiModalProjector(config)
        else:
            # For text-only models, set these to None
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
            image_data_embedding_func=image_embedding_func,
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
        # Only print debug info from rank 0 to avoid spam with multiple GPUs
        is_main_rank = get_tensor_model_parallel_rank() == 0

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

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        for name, loaded_weight in weights:
            # Skip vision weights if this is a text-only model
            if "vision" in name and not self.has_vision:
                continue

            # General prefix transformation: add language_model prefix if not already present
            # All model parameters have this prefix, but modelopt fp8 llama4 checkpoint weights don't
            if not name.startswith("language_model.") and "vision" not in name:
                name = f"language_model.{name}"

            if not "vision" in name:
                name, loaded_weight = self.permute_qk_weight_for_rotary(
                    name, loaded_weight
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "vision" in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    # NOTE: llama4 fp8 has different weight format for experts
                    # exclude shared_expert.xxx
                    # Scale parameters are now handled by the MoE layer's weight_loader directly
                    if (
                        "experts.gate_up_proj" not in name
                        and "experts.down_proj" not in name
                    ):
                        # Handle other expert parameters through mapping, experts.8.down_proj.weight, this is for RH llama4 fp8 checkpoint
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            param = params_dict[name]
                            weight_loader = param.weight_loader

                            weight_loader(
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                            break
                    else:
                        # Handle scale parameters differently from weight parameters
                        if "scale" in name:
                            # Scale parameters might have expert IDs in their names
                            # Check if this matches the expert parameter pattern: experts.{expert_id}.{param_name}
                            import re

                            expert_match = re.search(r"experts\.(\d+)\.", name)
                            if expert_match:
                                expert_id = int(expert_match.group(1))
                                if is_main_rank:
                                    print(
                                        f"Debug: Found expert_id {expert_id} in scale parameter: {name}"
                                    )

                                # Transform the name
                                if ".gate_up_proj" in name:
                                    transformed_name = name.replace(
                                        ".experts.gate_up_proj", ".experts.w13"
                                    )
                                    shard_id = "w13"
                                else:  # down_proj
                                    transformed_name = name.replace(
                                        ".experts.down_proj", ".experts.w2"
                                    )
                                    shard_id = "w2"

                                if is_main_rank:
                                    print(
                                        f"Debug: Loading scale parameter: {name} -> {transformed_name}"
                                    )

                                if transformed_name in params_dict:
                                    param = params_dict[transformed_name]
                                    weight_loader = param.weight_loader

                                    # Load the same scale for all experts
                                    for expert_id in range(num_experts):
                                        # For gate_up_proj scales, load for both w1 and w3 shards
                                        if ".gate_up_proj" in name:
                                            # Load the same scale for both w1 and w3
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w1",
                                                expert_id,
                                            )
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w3",
                                                expert_id,
                                            )
                                        else:
                                            # For down_proj, just load w2
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w2",
                                                expert_id,
                                            )
                            else:
                                # No expert ID found - this is a single scale for all experts
                                # Replicate the same scale across all experts
                                # Transform the name
                                if ".gate_up_proj" in name:
                                    transformed_name = name.replace(
                                        ".experts.gate_up_proj", ".experts.w13"
                                    )
                                    shard_id = "w13"
                                else:  # down_proj
                                    transformed_name = name.replace(
                                        ".experts.down_proj", ".experts.w2"
                                    )
                                    shard_id = "w2"

                                if transformed_name in params_dict:
                                    param = params_dict[transformed_name]
                                    weight_loader = param.weight_loader

                                    # Load the same scale for all experts
                                    for expert_id in range(num_experts):
                                        # For gate_up_proj scales, load for both w1 and w3 shards
                                        if ".gate_up_proj" in name:
                                            # Load the same scale for both w1 and w3
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w1",
                                                expert_id,
                                            )
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w3",
                                                expert_id,
                                            )
                                        else:
                                            # For down_proj, just load w2
                                            weight_loader(
                                                param,
                                                loaded_weight,
                                                transformed_name,
                                                "w2",
                                                expert_id,
                                            )
                                else:
                                    pass  # Parameter not found, skip
                        else:
                            # Weight parameters: Handle with expert iteration
                            if ".gate_up_proj" in name:
                                name_list = [
                                    name.replace(
                                        ".experts.gate_up_proj", ".experts.w13_weight"
                                    )
                                ] * 2
                                loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                                shard_id_list = ["w1", "w3"]
                            else:
                                name_list = [
                                    name.replace(
                                        ".experts.down_proj", ".experts.w2_weight"
                                    )
                                ]
                                shard_id_list = ["w2"]
                                loaded_weight_list = [loaded_weight]

                            for name, loaded_weight, shard_id in zip(
                                name_list, loaded_weight_list, shard_id_list
                            ):
                                param = params_dict[name]
                                weight_loader = param.weight_loader
                                for expert_id in range(num_experts):
                                    weight_loader(
                                        param,
                                        loaded_weight[expert_id].T,
                                        name,
                                        shard_id=shard_id,
                                        expert_id=expert_id,
                                    )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
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
