# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 Transformer model implementation.

NOTE: This initial implementation uses diffusers' JointTransformerBlock directly.
A native SGLang attention implementation is needed for FlashAttention, TP/SP,
quantization, and LoRA support.
"""

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.normalization import AdaLayerNormContinuous

from sglang.multimodal_gen.configs.models.dits.stablediffusion3 import (
    StableDiffusion3TransformerConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SD3Transformer2DModel(CachableDiT):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["JointTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    def __init__(
        self,
        config: StableDiffusion3TransformerConfig,
        hf_config: dict[str, Any] | None = None,
        quant_config=None,
    ):
        super().__init__(config=config, hf_config=hf_config)
        self.config = config
        arch_config = config.arch_config
        sample_size = arch_config.sample_size
        patch_size = arch_config.patch_size
        in_channels = arch_config.in_channels
        num_layers = arch_config.num_layers
        attention_head_dim = arch_config.attention_head_dim
        num_attention_heads = arch_config.num_attention_heads
        joint_attention_dim = arch_config.joint_attention_dim
        caption_projection_dim = arch_config.caption_projection_dim
        pooled_projection_dim = arch_config.pooled_projection_dim
        out_channels = arch_config.out_channels
        pos_embed_max_size = arch_config.pos_embed_max_size
        dual_attention_layers = arch_config.dual_attention_layers
        qk_norm = arch_config.qk_norm

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=i in dual_attention_layers,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        block_controlnet_hidden_states: list | None = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        skip_layers: list[int] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided.")
        if pooled_projections is None:
            raise ValueError("pooled_projections must be provided.")

        encoder_embeddings = encoder_hidden_states

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_embeddings = self.context_embedder(encoder_embeddings)

        skip_layer_set = set(skip_layers) if skip_layers else set()

        if block_controlnet_hidden_states is not None:
            interval_control = len(self.transformer_blocks) / len(
                block_controlnet_hidden_states
            )
        else:
            interval_control = 0

        for index_block, block in enumerate(self.transformer_blocks):
            if index_block not in skip_layer_set:
                encoder_embeddings, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_embeddings,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[
                        int(index_block / interval_control)
                    ]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        return output


# Entry class for registry
EntryClass = SD3Transformer2DModel
