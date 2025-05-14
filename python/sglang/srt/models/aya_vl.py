# Copyright 2023-2024 SGLang Team
# Adapted from vLLM's Aya Vision implementation and SGLang's Llava implementation.
# Copyright 2024 Cohere and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-only AyaVision model compatible with HuggingFace weights."""

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AyaVisionConfig, SiglipVisionModel
from transformers.activations import ACT2FN

# Assume Cohere2ForCausalLM has been ported and exists here;
# TODO: support Cohere2ForCausalLM properly

try:
    from sglang.srt.models.cohere2 import Cohere2ForCausalLM
except ImportError:
    logging.warning(
        "sglang.srt.models.cohere2.Cohere2ForCausalLM not found. "
        "AyaVision model requires this dependency."
    )

    # define a dummy class to avoid crashing during initial load attempts.
    class Cohere2ForCausalLM(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError(
                "Cohere2ForCausalLM model definition is missing in SGLang."
            )

        def load_weights(self, *args, **kwargs):
            raise NotImplementedError()

        def forward(self, *args, **kwargs):
            raise NotImplementedError()


from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, flatten_nested_list

logger = logging.getLogger(__name__)


class AyaVisionMultiModalProjector(nn.Module):
    """Projector layer specific to AyaVision."""

    def __init__(self, config: AyaVisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        # Default to text hidden size if alignment_intermediate_size is not specified
        self.alignment_intermediate_size = getattr(
            config, "alignment_intermediate_size", config.text_config.hidden_size
        )
        self.layernorm = nn.LayerNorm(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            eps=config.adapter_layer_norm_eps,
        )

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            bias=True,
        )

        # SwiGLU uses SiLU activation
        self.act = ACT2FN.get("silu", ACT2FN["silu"])
        # For SwiGLU, project down to half size since we split intermediate dim
        self.linear_2 = nn.Linear(
            self.alignment_intermediate_size // 2,
            config.text_config.hidden_size,
            bias=True,
        )

        # Assign weight loaders if necessary for custom quantization/loading;;;
        # set_weight_attrs(self.layernorm.weight, {"weight_loader": ...})
        # set_weight_attrs(self.linear_1.weight, {"weight_loader": ...})
        # set_weight_attrs(self.linear_2.weight, {"weight_loader": ...})

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU activation
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features: torch.Tensor) -> torch.Tensor:  # B, S, D
        """Helper function for pixel shuffling specific to AyaVision projector."""
        batch_size, seq_length, _ = image_features.shape
        # Assuming square input patches for seq_length calculation
        height = width = int(seq_length**0.5)
        # Reshape to spatial dimensions
        image_features = image_features.view(image_features.shape[0], width, height, -1)
        channels = image_features.shape[-1]

        # Apply downsampling based on the factor
        # Reshape for height downsampling
        image_features = image_features.view(
            batch_size,
            width,
            int(height / self.downsample_factor),
            int(channels * self.downsample_factor),
        )
        # Permute for width downsampling
        image_features = image_features.permute(0, 2, 1, 3)
        # Reshape for width downsampling
        image_features = image_features.view(
            batch_size,
            int(height / self.downsample_factor),
            int(width / self.downsample_factor),
            -1,  # Channels dimension automatically calculated
        )
        # Permute back to original dimension order (if necessary, seems layout might be different now)
        # The vLLM code permutes again, check if needed based on expected output layout
        image_features = image_features.permute(0, 2, 1, 3)

        # Flatten back to (batch_size, new_seq_length, new_hidden_dim)
        # The shape after shuffling seems different in vLLM's code, verify the final flatten step
        # The vLLM code returns shape [B, W/factor, H/factor, C*factor^2], let's flatten that
        image_features = image_features.flatten(start_dim=1, end_dim=2)

        return image_features


class AyaVisionBaseForCausalLM(nn.Module):
    def __init__(self, config: AyaVisionConfig):
        super().__init__()
        self.config = config
        self.vision_tower: Optional[SiglipVisionModel] = None
        self.multi_modal_projector: Optional[AyaVisionMultiModalProjector] = None

    def _select_image_features(
        self, image_features: torch.Tensor, *, strategy: str
    ) -> torch.Tensor:
        """Selects features from the vision tower output based on strategy."""

        if strategy == "default":
            # Default for CLIP-like models is to exclude CLS token.
            return image_features[:, 1:]
        elif strategy == "full":
            # SigLIP uses "full", keeping all tokens including the CLS-like one
            return image_features
        else:
            raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encodes images using the vision tower and multimodal projector."""
        if self.vision_tower is None or self.multi_modal_projector is None:
            raise ValueError(
                "Vision tower and projector must be initialized before encoding images."
            )

        vision_dtype = next(self.vision_tower.parameters()).dtype
        vision_device = next(self.vision_tower.parameters()).device
        pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)

        # 1. Get features from vision tower
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # 2. Select the correct layer
        # Layer indexing needs to handle negative indices like in HF/vLLM
        layer_idx = self.config.vision_feature_layer
        if layer_idx < 0:
            layer_idx = self.config.vision_config.num_hidden_layers + layer_idx
        selected_image_feature = image_outputs.hidden_states[layer_idx]

        selected_image_feature = self._select_image_features(
            selected_image_feature,
            strategy=self.config.vision_feature_select_strategy,
        )

        # 4. Project features
        image_embeds = self.multi_modal_projector(selected_image_feature)
        return image_embeds


class AyaVisionCohereForCausalLM(AyaVisionBaseForCausalLM):
    def __init__(
        self,
        config: AyaVisionConfig,  # Expecting AyaVisionConfig here.
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.quant_config = quant_config  # Store quant_config if needed by LM

        self.multi_modal_projector = AyaVisionMultiModalProjector(config)

        language_model_prefix = add_prefix("language_model", prefix)
        self.language_model = Cohere2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=language_model_prefix,
        )

        # Vision tower is initialized in load_weights
        self.vision_tower = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass logic, handling both text and image inputs.
        """
        image_inputs: Optional[MultimodalInputs] = forward_batch.mm_inputs

        if forward_batch.forward_mode.is_extend():
            # Logic similar to SGLang Llava for embedding generation/merging

            # Determine which sequences in the batch actually need vision processing now
            # This requires image_offsets and image_pad_len from the processor
            if image_inputs:
                max_image_offset_list = []
                for i, im_input in enumerate(image_inputs):
                    if im_input and im_input.image_offsets:
                        max_image_offset_list.append(
                            np.max(
                                np.array(im_input.image_offsets)
                                + np.array(im_input.image_pad_len)
                            )
                        )
                    else:
                        max_image_offset_list.append(-1)

                start_positions = (
                    positions[forward_batch.extend_start_loc].cpu().numpy()
                )
                need_vision = start_positions <= np.array(max_image_offset_list)

            else:  # No image inputs in the batch
                need_vision = np.zeros(forward_batch.batch_size, dtype=bool)

            if input_embeds is None and need_vision.any():
                # Calculate text embeddings first
                input_embeds = self.language_model.model.embed_tokens(input_ids)

                bs = forward_batch.batch_size
                # Extract pixel_values only for sequences that need vision processing
                pixel_values_list = []
                num_tiles_per_needed_seq = []
                if image_inputs:
                    for i in range(bs):
                        if need_vision[i] and image_inputs[i]:
                            seq_pixels = flatten_nested_list(
                                [item.pixel_values for item in image_inputs[i].mm_items]
                            )
                            pixel_values_list.extend(seq_pixels)
                            num_tiles_per_needed_seq.append(
                                len(seq_pixels)
                            )  # Num tiles for this sequence

                if pixel_values_list:
                    concatenated_pixel_values = torch.cat(pixel_values_list, dim=0)

                    # Encode all needed image tiles at once
                    image_features_encoded = self.encode_images(
                        concatenated_pixel_values
                    )
                    # image_features_encoded shape: (total_tiles_needed, num_tokens_per_tile, D)

                    # Split the results back per sequence that needed vision
                    split_image_features = torch.split(
                        image_features_encoded, num_tiles_per_needed_seq, dim=0
                    )

                    # Combine features for multi-tile images within each sequence
                    final_image_embeds_list = []
                    pt_split = 0
                    if image_inputs:
                        for i in range(bs):
                            if need_vision[i] and image_inputs[i]:
                                seq_features = split_image_features[pt_split]

                                combined_features = seq_features.view(
                                    -1, seq_features.shape[-1]
                                )

                                final_image_embeds_list.append(combined_features)
                                pt_split += 1

                    # Merge image embeddings into text embeddings
                    extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                    extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
                    pt_embed = 0
                    for i in range(bs):
                        if need_vision[i] and image_inputs and image_inputs[i]:
                            start_idx = extend_start_loc_cpu[i]
                            seq_len = extend_seq_lens[i]
                            prefix_len = prefix_lens_cpu[i]

                            current_image_embeds = final_image_embeds_list[pt_embed]
                            pt_embed += 1

                            for img_idx, image_offset in enumerate(
                                image_inputs[i].image_offsets
                            ):
                                pad_len = image_inputs[i].image_pad_len[img_idx]
                                img_embed_chunk = current_image_embeds  # TODO: Handle multiple images correctly

                                # Calculate the slice in input_embeds to overwrite
                                input_offset = image_offset - prefix_len
                                left_idx = start_idx + input_offset
                                right_idx = left_idx + pad_len

                                # Adjust slice based on the current extend window
                                slice_left_embed = 0
                                slice_right_embed = pad_len

                                if left_idx < start_idx:  # Starts before current window
                                    slice_left_embed = start_idx - left_idx
                                    left_idx = start_idx
                                if (
                                    right_idx > start_idx + seq_len
                                ):  # Ends after current window
                                    slice_right_embed = pad_len - (
                                        right_idx - (start_idx + seq_len)
                                    )
                                    right_idx = start_idx + seq_len

                                if left_idx < right_idx:
                                    try:
                                        input_embeds[left_idx:right_idx] = (
                                            img_embed_chunk[
                                                slice_left_embed:slice_right_embed
                                            ]
                                        )
                                    except RuntimeError as e:
                                        logger.error(
                                            f"RuntimeError during embedding merge: {e}"
                                        )
                                        logger.error(
                                            f"Shapes: input_embeds[{left_idx}:{right_idx}]={input_embeds[left_idx:right_idx].shape}, img_chunk[{slice_left_embed}:{slice_right_embed}]={img_embed_chunk[slice_left_embed:slice_right_embed].shape}"
                                        )
                                        logger.error(
                                            f"Indices: {i=}, {img_idx=}, {image_offset=}, {pad_len=}, {prefix_len=}, {seq_len=}, {start_idx=}"
                                        )
                                        # Potentially raise or log more context

                # After merging (or if no vision needed), call the language model
                # If images were merged, input_ids should be None
                lm_input_ids = None if need_vision.any() else input_ids
                return self.language_model(
                    lm_input_ids, positions, forward_batch, input_embeds=input_embeds
                )

            else:  # No vision needed, or input_embeds already provided
                return self.language_model(
                    input_ids, positions, forward_batch, input_embeds=input_embeds
                )

        elif forward_batch.forward_mode.is_decode():
            # Decode phase only needs text processing
            return self.language_model(input_ids, positions, forward_batch)

        else:
            raise ValueError(f"Unknown forward mode: {forward_batch.forward_mode}")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Loads weights for AyaVision, handling vision tower, projector, and LM."""
        # 1. Initialize Vision Tower (SigLIP) from HF
        # Assuming config has mm_vision_tower path
        vision_tower_path = getattr(self.config, "mm_vision_tower", None) or getattr(
            self.config, "vision_model", None
        )  # Check common names
        if not vision_tower_path:
            raise ValueError("Vision tower path not found in config.")

        try:
            logger.info(f"Loading SiglipVisionModel from: {vision_tower_path}")
            # Determine dtype from language model if available, else default to float16
            lm_dtype = (
                next(self.language_model.parameters()).dtype
                if self.language_model
                else torch.float16
            )
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_tower_path, torch_dtype=lm_dtype
            )
            self.vision_tower.cuda().eval()
            logger.info("Vision tower loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SiglipVisionModel: {e}", exc_info=True)
            raise

        # 2. Prepare for weight loading
        params_dict = dict(self.named_parameters())
        projector_prefix = "multi_modal_projector."
        # Determine the vision tower prefix used in the checkpoint file. Common ones are "vision_tower." or "vision_model."
        # Check a few sample weight names if unsure. Assume "vision_tower." based on vLLM.
        vision_tower_prefix_in_ckpt = "vision_tower."
        language_model_prefix_in_ckpt = "language_model."  # Or potentially "model."

        remaining_weights = []

        # 3. Iterate through checkpoint weights and assign
        vision_tower_weights_loaded = set()
        projector_weights_loaded = set()

        for name, loaded_weight in weights:
            if name.startswith(vision_tower_prefix_in_ckpt):
                # Map checkpoint name to HF SiglipVisionModel parameter name
                hf_vision_param_name = name[len(vision_tower_prefix_in_ckpt) :]
                try:
                    param = self.vision_tower.get_parameter(hf_vision_param_name)
                    param.data.copy_(loaded_weight)
                    vision_tower_weights_loaded.add(name)
                except AttributeError:
                    logger.warning(
                        f"Parameter {hf_vision_param_name} not found in loaded SiglipVisionModel. Skipping weight: {name}"
                    )
            elif name.startswith(projector_prefix):
                if name in params_dict:
                    param = params_dict[name]

                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    projector_weights_loaded.add(name)
                else:
                    logger.warning(
                        f"Parameter {name} found in checkpoint but not in SGLang projector model."
                    )
            elif name.startswith(language_model_prefix_in_ckpt):
                # Pass weights intended for the language model, adjusting prefix if needed
                lm_name = name[len(language_model_prefix_in_ckpt) :]
                remaining_weights.append((lm_name, loaded_weight))
            else:
                # Handle weights that might belong directly to the language model if prefix differs
                # Or warn about unexpected weights
                logger.warning(
                    f"Unexpected weight prefix in checkpoint: {name}. Assuming it belongs to LM or skipping."
                )
                # Heuristic: if it's in the LM params_dict (without prefix), assume it's LM
                if name in self.language_model.state_dict():
                    remaining_weights.append((name, loaded_weight))

        logger.info(
            f"Loaded {len(vision_tower_weights_loaded)} weights for vision tower."
        )
        logger.info(f"Loaded {len(projector_weights_loaded)} weights for projector.")
        logger.info(f"Passing {len(remaining_weights)} weights to language model.")

        # 4. Load remaining weights into the language model
        self.language_model.load_weights(remaining_weights)
        logger.info("Weight loading complete.")


EntryClass = AyaVisionCohereForCausalLM
