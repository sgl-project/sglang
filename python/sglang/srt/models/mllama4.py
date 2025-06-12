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

        # Check if this is a text-only model (no vision components)
        # We need a more robust check since transformers auto-generates vision_config
        # for Llama4ForConditionalGeneration even for text-only models

        # First check the original config.json to see if vision_config was explicitly defined
        import json as json_lib
        import os

        # Get the model path from the config if available
        model_path = getattr(config, "_name_or_path", None)
        original_has_vision = False

        if model_path and os.path.exists(os.path.join(model_path, "config.json")):
            try:
                with open(os.path.join(model_path, "config.json"), "r") as f:
                    original_config = json_lib.load(f)
                # Check if vision_config exists in the original file
                original_has_vision = "vision_config" in original_config
                if get_tensor_model_parallel_rank() == 0:
                    print(
                        f"Debug: Original config has explicit vision_config: {original_has_vision}"
                    )
            except:
                # Fallback to checking the processed config
                original_has_vision = (
                    hasattr(config, "vision_config")
                    and config.vision_config is not None
                )
                if get_tensor_model_parallel_rank() == 0:
                    print(
                        f"Debug: Could not read original config, using processed config"
                    )
        else:
            # Fallback to checking the processed config
            original_has_vision = (
                hasattr(config, "vision_config") and config.vision_config is not None
            )
            if get_tensor_model_parallel_rank() == 0:
                print(
                    f"Debug: No model path or config.json found, using processed config"
                )

        self.has_vision = original_has_vision
        if get_tensor_model_parallel_rank() == 0:
            print(f"Debug: has_vision: {self.has_vision}")
            print(f"Debug: model_path: {model_path}")

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

        if is_main_rank:
            print(
                f"Debug: Loading weights for {'multimodal' if self.has_vision else 'text-only'} model"
            )

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
        if is_main_rank:
            print(f"Debug: Model has {len(params_dict)} parameters")

            # Debug: Show scale parameter names in the model
            scale_params = [
                name
                for name in params_dict.keys()
                if "scale" in name and "expert" in name
            ]
            print(
                f"Debug: Scale parameters in model: {scale_params[:30]}"
            )  # Show first 10

        num_experts = (
            self.config.text_config.num_local_experts
            if hasattr(self.config, "text_config")
            else self.config.num_local_experts
        )
        if is_main_rank:
            print(f"Debug: Number of experts: {num_experts}")

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        # Note: Scale parameters will be handled separately since they have
        # a different name pattern (language_model.model.layers.X... vs model.layers.X...)

        for name, loaded_weight in weights:
            # Debug: Show scale parameter names
            if is_main_rank and "scale" in name and "expert" in name:
                print(f"DEBUG: Processing checkpoint scale parameter: {name}")

            # Skip vision weights if this is a text-only model
            if "vision" in name and not self.has_vision:
                # if is_main_rank:
                #     print(f"Skipping vision weight: {name} (text-only model)")
                continue

                # Special handling for scale parameters that have language_model. prefix
            if (
                "scale" in name
                and "language_model.model.layers" in name
                and "experts" in name
            ):
                # Transform language_model.model.layers.X.feed_forward.experts.*
                # to model.layers.X.feed_forward.experts.*
                transformed_name = name.replace("language_model.", "")

                # Try multiple possible target names since the model may have duplicate scale parameters
                target_names = []

                if "gate_up_proj" in transformed_name:
                    # Try both gate_up_proj (direct) and w13 (FusedMoE) naming
                    target_names.append(transformed_name)  # Direct: gate_up_proj
                    target_names.append(
                        transformed_name.replace("gate_up_proj", "w13")
                    )  # FusedMoE: w13
                elif "down_proj" in transformed_name:
                    # Try both down_proj (direct) and w2 (FusedMoE) naming
                    target_names.append(transformed_name)  # Direct: down_proj
                    target_names.append(
                        transformed_name.replace("down_proj", "w2")
                    )  # FusedMoE: w2
                else:
                    target_names.append(transformed_name)

                # Try each possible target name
                loaded = False
                for target_name in target_names:
                    if target_name in params_dict:
                        if is_main_rank:
                            print(
                                f"DEBUG: Loading scale parameter: {name} -> {target_name}"
                            )
                        param = params_dict[target_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded = True
                        break

                if loaded:
                    continue
                else:
                    if is_main_rank:
                        print(
                            f"DEBUG: Scale parameter {name} -> {target_names} not found in model"
                        )
                    continue

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
                if name not in params_dict:
                    # if is_main_rank:
                    #     print(f"Warning: Parameter {name} not found in model, skipping")
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    # NOTE: llama4 fp8 has different weight format for experts
                    # Scale parameters are handled separately above
                    if "scale" in name:
                        # Scale parameters should already be handled above
                        if is_main_rank:
                            print(
                                f"DEBUG: Skipping scale parameter in expert section: {name}"
                            )
                        continue
                    elif (
                        "experts.gate_up_proj" not in name
                        and "experts.down_proj" not in name
                    ):
                        # Handle other expert parameters through mapping
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                # if is_main_rank:
                                #     print(f"Warning: Expert parameter {name} not found in model, skipping")
                                break
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
                        if ".gate_up_proj" in name:
                            # Add safety checks for tensor dimensions
                            # if is_main_rank:
                            #     print(f"Debug: Processing gate_up_proj weight {name}, shape: {loaded_weight.shape}, dtype: {loaded_weight.dtype}")

                            if loaded_weight.dim() == 0:
                                # if is_main_rank:
                                #     print(f"Warning: Skipping scalar weight {name}")
                                continue
                            elif loaded_weight.size(-1) < 2:
                                # if is_main_rank:
                                #     print(f"Warning: Cannot chunk weight {name} with last dimension {loaded_weight.size(-1)}")
                                continue

                            name_list = [
                                name.replace(
                                    ".experts.gate_up_proj", ".experts.w13_weight"
                                )
                            ] * 2
                            loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                            shard_id_list = ["w1", "w3"]
                        elif ".down_proj" in name:
                            # Add debugging for down_proj as well
                            # if is_main_rank:
                            #     print(f"Debug: Processing down_proj weight {name}, shape: {loaded_weight.shape}, dtype: {loaded_weight.dtype}")

                            if loaded_weight.dim() == 0:
                                # if is_main_rank:
                                #     print(f"Warning: Skipping scalar weight {name}")
                                continue

                            name_list = [
                                name.replace(".experts.down_proj", ".experts.w2_weight")
                            ]
                            shard_id_list = ["w2"]
                            loaded_weight_list = [loaded_weight]
                        else:
                            # Unrecognized expert parameter
                            # if is_main_rank:
                            #     print(f"Warning: Unrecognized expert parameter {name}, skipping")
                            continue

                        for name, loaded_weight, shard_id in zip(
                            name_list, loaded_weight_list, shard_id_list
                        ):
                            if name not in params_dict:
                                # if is_main_rank:
                                #     print(f"Warning: Expert parameter {name} not found in model, skipping")
                                continue
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
                    # Skip parameters that don't exist in the text-only model
                    if name not in params_dict:
                        # if is_main_rank:
                        #     print(f"Warning: Parameter {name} not found in model, skipping")
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
