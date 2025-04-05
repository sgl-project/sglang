# Copyright 2024 SGLang Team
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
# ==============================================================================
"""
Inference-only Pixtral HF Vision model compatible with HuggingFace Transformers (Llava Architecture).
Using mistral-community/pixtral-12b as reference.
TODO: add support for mistral format (mistralai/Pixtral-12B as a reference)
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PixtralVisionConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    position_ids_in_meshgrid,
)

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader


class PixtralHFEncoderInfo:
    """Information about the Pixtral Vision Encoder configuration."""

    def __init__(self, vision_config: PixtralVisionConfig):
        self.vision_config = vision_config

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        ncols, nrows = self.get_patch_grid_size(
            image_width=image_width,
            image_height=image_height,
        )

        tokens = (ncols + 1) * nrows
        # Consider the image_break_token
        return tokens

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size()
        tokens = self.get_num_image_tokens(
            image_width=image_size,
            image_height=image_size,
        )
        return tokens

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        patch_size = (
            self.vision_config.patch_size * self.vision_config.spatial_merge_size
        )
        return patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        grid_length = image_size // patch_size
        # Since interpolation is applied, the image size need not be divisible
        return grid_length

    def get_patch_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_width = max_height = self.get_image_size()
        patch_width = patch_height = self.get_patch_size()

        ratio = max(image_width / max_width, image_height / max_height)

        orig_width, orig_height = image_width, image_height
        if ratio > 1:
            image_width = int(math.floor(image_width / ratio))
            image_height = int(math.floor(image_height / ratio))

        nrows, ncols = _get_pixtral_hf_num_image_tokens(
            (image_height, image_width),
            (patch_height, patch_width),
        )

        return ncols, nrows


class PixtralHFMLP(nn.Module):
    """MLP for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert config.intermediate_size is not None

        # Use MergedColumnParallelLinear for gate_up_proj to handle combined weights
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_output, _ = self.gate_up_proj(x)

        # Apply SiLU activation and multiply
        gate_up = self.act_fn(gate_up_output)

        # Project back to hidden size
        out, _ = self.down_proj(gate_up)
        return out


class PixtralHFTransformerBlock(nn.Module):
    """Transformer block for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)

        # Use SGLang's VisionAttention instead of vLLM's PixtralHFAttention
        self.attention = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            quant_config=quant_config,
            dropout=0.0,
            bias=False,
            use_context_forward=False,
            softmax_in_single_precision=False,
            flatten_batch=False,
            prefix=f"{prefix}.attention",
        )

        self.feed_forward = PixtralHFMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )

        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # Ensure hidden_states has the batch dimension [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Apply attention norm - normalize along the last dimension
        attn_normalized = self.attention_norm(hidden_states.view(-1, hidden_dim)).view(
            batch_size, seq_len, hidden_dim
        )

        # Pass through attention layer
        attention_output = self.attention(
            attn_normalized,
            attention_mask=attention_mask,
            cu_seqlens=None,
            position_embeddings=position_embeddings,
        )

        # Apply first residual connection
        hidden_states = hidden_states + attention_output

        # Apply feed-forward norm - normalize along the last dimension
        ffn_normalized = self.ffn_norm(hidden_states.view(-1, hidden_dim)).view(
            batch_size, seq_len, hidden_dim
        )

        # Pass through feed-forward layer
        # First reshape to 2D for the feed-forward network, then reshape back
        ffn_output = self.feed_forward(ffn_normalized)

        # Apply second residual connection
        output = hidden_states + ffn_output

        return output


class PixtralHFTransformer(nn.Module):
    """Transformer for PixtralHFVisionModel using SGLang components."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        num_hidden_layers = config.num_hidden_layers
        if num_hidden_layers_override is not None:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                PixtralHFTransformerBlock(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        return_all_hidden_states: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through transformer layers.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            position_embeddings: Optional position embeddings for rotary attention
            return_all_hidden_states: Whether to return all hidden states

        Returns:
            Either the final hidden state, or a list of all hidden states if
            return_all_hidden_states is True
        """
        # For HF model compatibility, always start with the input
        hidden_states = x
        all_hidden_states = [hidden_states] if return_all_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return all_hidden_states
        return hidden_states


def resolve_visual_encoder_outputs(
    outputs: Union[torch.Tensor, List[torch.Tensor]],
    feature_sample_layers: Optional[List[int]],
    post_norm: Optional[nn.Module],
    num_hidden_layers: int,
) -> torch.Tensor:
    """Resolve outputs from visual encoder based on feature_sample_layers."""
    if feature_sample_layers is None:
        # Just use the last layer's output
        if isinstance(outputs, list):
            outputs = outputs[-1]
        if post_norm is not None:
            outputs = post_norm(outputs)
        return outputs

    # Handle the case where we want to use specific layers
    if not isinstance(outputs, list):
        raise ValueError(
            "Expected outputs to be a list when feature_sample_layers is provided"
        )

    # Validate layer indices
    for layer_idx in feature_sample_layers:
        if layer_idx < 0 or layer_idx > num_hidden_layers:
            raise ValueError(
                f"Feature sample layer index {layer_idx} is out of range "
                f"[0, {num_hidden_layers}]"
            )

    # Collect outputs from specified layers
    selected_outputs = [outputs[layer_idx] for layer_idx in feature_sample_layers]

    # Combine the outputs
    combined_outputs = torch.cat(selected_outputs, dim=-1)

    if post_norm is not None:
        combined_outputs = post_norm(combined_outputs)

    return combined_outputs


class PixtralHFVisionModel(nn.Module):
    """Hugging Face Pixtral Vision Model implemented using SGLang components."""

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)

        self.transformer = PixtralHFTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.transformer",
        )

        # Check that num_hidden_layers is valid
        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} "
                "layers."
            )

        if require_post_norm is True:
            raise ValueError("PixtralHFVisionModel does not have post-layernorm")

        # Initialize patch position embedding
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: list[torch.Tensor],
        output_hidden_states: bool = False,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Args:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This is a list of tensors because multiple
                requests batched can have multiple images, each with their
                own shape potentially.
            output_hidden_states: Whether to return all hidden states.
            feature_sample_layers: Layer indices whose features should be
                concatenated and used as the visual encoder output. If none
                are provided, the last layer is used.

        Returns:
            A tuple containing:
              - hidden_states: Final model outputs (or selected layers if feature_sample_layers given)
              - hidden_states tuple (optional): All hidden states if output_hidden_states=True
        """
        # Process images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in pixel_values
        ]

        # Reshape patches to token sequences
        patch_embeds = [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        # Combine all patches into a single sequence
        patch_embeds = torch.cat(patch_embeds, dim=1)

        # Apply pre-layer norm
        # RMSNorm expects 2D input, so reshape, apply norm, and reshape back
        batch_size, seq_len, hidden_dim = patch_embeds.shape
        patch_embeds = self.ln_pre(patch_embeds.view(-1, hidden_dim)).view(
            batch_size, seq_len, hidden_dim
        )

        # Compute positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        ).to(self.device)

        # The original PixtralRotaryEmbedding expects 2D input but returns a tuple of tensors (cos, sin)
        # These tensors are used by apply_rotary_pos_emb in the transformer blocks
        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)

        # Create attention mask for each image (block diagonal)
        # We'll use a simple mask approach instead of xformers
        batch_size = len(patch_embeds_list)
        seq_lengths = [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]
        max_seq_len = max(seq_lengths)

        # Create a block diagonal mask
        attention_mask = torch.zeros(
            (batch_size, 1, max_seq_len, max_seq_len),
            device=self.device,
            dtype=torch.bool,
        )

        # Fill in the blocks
        start_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len
            attention_mask[i, 0, start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        # Process through transformer
        return_all_hidden_states = (
            output_hidden_states or feature_sample_layers is not None
        )
        transformer_outputs = self.transformer(
            patch_embeds,  # Already has shape [batch_size, seq_len, hidden_dim]
            attention_mask,
            position_embedding,
            return_all_hidden_states=return_all_hidden_states,
        )

        # Store all hidden states if requested
        all_hidden_states = None
        if isinstance(transformer_outputs, list):
            all_hidden_states = transformer_outputs
            # Use the last layer by default if feature_sample_layers is not specified
            if feature_sample_layers is None:
                out = transformer_outputs[-1]
            else:
                # Resolve outputs based on feature sample layers
                out = resolve_visual_encoder_outputs(
                    transformer_outputs,
                    feature_sample_layers,
                    None,
                    self.config.num_hidden_layers,
                )
        else:
            out = transformer_outputs

        # Split back into separate tensors for each image
        final_outputs = torch.split(
            out.squeeze(0) if out.size(0) == 1 else out, embed_sizes
        )

        # Format return to be compatible with HuggingFace vision models
        if output_hidden_states:
            return type(
                "VisualOutput",
                (),
                {
                    "last_hidden_state": final_outputs,
                    "hidden_states": all_hidden_states,
                },
            )
        else:
            return final_outputs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights from a HuggingFace checkpoint with proper parameter mapping."""
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        # Define mappings for stacked parameters
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            (".attention.qkv_proj", ".attention.q_proj", "q"),
            (".attention.qkv_proj", ".attention.k_proj", "k"),
            (".attention.qkv_proj", ".attention.v_proj", "v"),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        # Process each weight
        processed_count = 0
        stacked_count = 0
        for name, loaded_weight in weights:
            # Check if this is a stacked parameter (q_proj/k_proj/v_proj or gate_proj/up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    # Replace the weight name part with the combined parameter name
                    transformed_name = name.replace(weight_name, param_name)
                    if transformed_name in params_dict:
                        param = params_dict[transformed_name]
                        # Use the weight_loader method with shard_id to load into the correct portion
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(transformed_name)
                        stacked_count += 1
                        break
            else:
                # Handle regular parameters (not stacked ones)
                # For attention.proj => attention.o_proj mapping
                if ".attention.o_proj" in name:
                    name = name.replace(".attention.o_proj", ".attention.proj")

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


# Register the model classes for external access
EntityClass = [PixtralHFVisionModel]
