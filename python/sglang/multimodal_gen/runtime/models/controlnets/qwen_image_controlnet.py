# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
# Adapted for SGLang from InstantX/Qwen-Image-ControlNet-Union
"""
QwenImage ControlNet Model

This module implements the ControlNet adapter for QwenImage, enabling
precise spatial control over image generation.

The architecture mirrors InstantX/Qwen-Image-ControlNet-Union to ensure
weight compatibility when loading from HuggingFace.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    Used for zero-initialization of ControlNet output layers.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class QwenImageControlNetOutput:
    """Output class for QwenImageControlNetModel"""

    controlnet_block_samples: Tuple[torch.Tensor, ...]


class QwenImageControlNetModel(nn.Module):
    """
    ControlNet for QwenImage - mirrors InstantX/Qwen-Image-ControlNet-Union architecture.

    This implementation reuses SGLang's optimized QwenImageTransformerBlock and related
    components to ensure compatibility with the loading system while maintaining
    performance optimizations.

    Architecture:
    - pos_embed: QwenEmbedRope for rotary position embeddings
    - time_text_embed: QwenTimestepProjEmbeddings for timestep conditioning
    - txt_norm, txt_in: Text processing layers
    - img_in: Image input projection
    - transformer_blocks: N transformer blocks (typically 5 for ControlNet-Union)
    - controlnet_blocks: Zero-initialized output projections
    - controlnet_x_embedder: Zero-initialized condition embedding
    """

    # Required class attributes for SGLang's FSDP loading system
    _fsdp_shard_conditions = [
        lambda n, m: isinstance(m, QwenImageTransformerBlock),
    ]

    # Parameter name mapping - identity mapping since we match InstantX architecture
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: QwenImageDitConfig,
        hf_config: Dict[str, Any],
    ):
        super().__init__()

        # Extract config parameters
        # Use hf_config values if available (from InstantX config.json), else fall back to dit_config
        patch_size = hf_config.get("patch_size", config.arch_config.patch_size)
        in_channels = hf_config.get("in_channels", config.arch_config.in_channels)
        out_channels = hf_config.get("out_channels", config.arch_config.out_channels)
        num_layers = hf_config.get("num_layers", 5)  # ControlNet-Union uses 5 blocks
        attention_head_dim = hf_config.get(
            "attention_head_dim", config.arch_config.attention_head_dim
        )
        num_attention_heads = hf_config.get(
            "num_attention_heads", config.arch_config.num_attention_heads
        )
        joint_attention_dim = hf_config.get(
            "joint_attention_dim", config.arch_config.joint_attention_dim
        )
        axes_dims_rope = hf_config.get(
            "axes_dims_rope", config.arch_config.axes_dims_rope
        )
        extra_condition_channels = hf_config.get("extra_condition_channels", 0)

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.extra_condition_channels = extra_condition_channels

        logger.info(
            f"Initializing QwenImageControlNet with {num_layers} blocks, "
            f"inner_dim={self.inner_dim}, in_channels={in_channels}"
        )

        # Position embeddings - matches transformer architecture
        self.pos_embed = QwenEmbedRope(
            theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
        )

        # Timestep embeddings
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        # Text processing
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        # Image input projection
        self.img_in = nn.Linear(in_channels, self.inner_dim)

        # Transformer blocks - reuse SGLang's optimized implementation
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # ControlNet output blocks - zero-initialized for stable training
        self.controlnet_blocks = nn.ModuleList(
            [
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
                for _ in range(num_layers)
            ]
        )

        # ControlNet condition embedder - zero-initialized
        self.controlnet_x_embedder = zero_module(
            nn.Linear(in_channels + extra_condition_channels, self.inner_dim)
        )

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[QwenImageControlNetOutput, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through ControlNet.

        Args:
            hidden_states: Input hidden states [batch, seq_len, in_channels]
            controlnet_cond: Control condition tensor [batch, seq_len, in_channels]
            conditioning_scale: Scale factor for ControlNet outputs
            encoder_hidden_states: Text embeddings [batch, text_seq_len, joint_attention_dim]
            encoder_hidden_states_mask: Attention mask for text
            timestep: Current denoising timestep
            img_shapes: List of (frame, height, width) tuples for RoPE
            txt_seq_lens: List of text sequence lengths
            freqs_cis: Pre-computed rotary position embeddings (if available)
            joint_attention_kwargs: Additional kwargs for attention
            return_dict: Whether to return QwenImageControlNetOutput

        Returns:
            ControlNet block samples to be added to transformer hidden states
        """
        # Handle list input for encoder_hidden_states (same as transformer)
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        # Project image input
        hidden_states = self.img_in(hidden_states)

        # Add control condition embedding
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        # Normalize timestep (same as transformer: timestep / 1000)
        timestep = (timestep / 1000).to(hidden_states.dtype)

        # Compute timestep embedding
        temb = self.time_text_embed(timestep, hidden_states)

        # Compute rotary position embeddings
        # Use pre-computed freqs_cis if available, otherwise compute from pos_embed
        if freqs_cis is not None:
            image_rotary_emb = freqs_cis
        else:
            image_rotary_emb = self.pos_embed(
                img_shapes, txt_seq_lens, device=hidden_states.device
            )

        # Process text embeddings
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Process through transformer blocks
        block_samples = []
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            block_samples.append(hidden_states)

        # Apply controlnet output projections with zero-init
        controlnet_block_samples = []
        for block_sample, controlnet_block in zip(
            block_samples, self.controlnet_blocks
        ):
            controlnet_block_samples.append(
                controlnet_block(block_sample) * conditioning_scale
            )

        controlnet_block_samples = tuple(controlnet_block_samples)

        if not return_dict:
            return controlnet_block_samples

        return QwenImageControlNetOutput(
            controlnet_block_samples=controlnet_block_samples
        )


class QwenImageMultiControlNetModel(nn.Module):
    """
    Wrapper class for multiple QwenImageControlNetModel instances.

    Supports two modes:
    1. ControlNet-Union: Single network with multiple conditions (e.g., canny + depth)
    2. Multi-ControlNet: Multiple separate networks, each with its own condition
    """

    def __init__(self, controlnets: List[QwenImageControlNetModel]):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: List[torch.Tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[QwenImageControlNetOutput, Tuple]:
        """
        Forward pass through multiple ControlNets.

        Args:
            hidden_states: Input hidden states [batch, seq_len, in_channels]
            controlnet_cond: List of control condition tensors (one per condition/network)
            conditioning_scale: List of scales (one per condition/network)
            ...

        Modes:
            - Single net (Union): Iterate conditions through the same network
            - Multiple nets: Each network processes its corresponding condition
        """
        control_block_samples = None

        if len(self.nets) == 1:
            # ControlNet-Union mode: single network, multiple conditions
            controlnet = self.nets[0]

            for i, (cond, scale) in enumerate(zip(controlnet_cond, conditioning_scale)):
                block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=cond,
                    conditioning_scale=scale,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )

                if control_block_samples is None:
                    control_block_samples = list(block_samples)
                else:
                    control_block_samples = [
                        prev + curr
                        for prev, curr in zip(control_block_samples, block_samples)
                    ]
        else:
            # Multi-ControlNet mode: multiple networks, each with its own condition
            # Number of conditions should match number of networks
            if len(controlnet_cond) != len(self.nets):
                raise ValueError(
                    f"Number of conditions ({len(controlnet_cond)}) must match "
                    f"number of ControlNets ({len(self.nets)})"
                )

            for i, (controlnet, cond, scale) in enumerate(
                zip(self.nets, controlnet_cond, conditioning_scale)
            ):
                block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=cond,
                    conditioning_scale=scale,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )

                if control_block_samples is None:
                    control_block_samples = list(block_samples)
                else:
                    # Sum residuals from all ControlNets
                    control_block_samples = [
                        prev + curr
                        for prev, curr in zip(control_block_samples, block_samples)
                    ]

        if return_dict:
            return QwenImageControlNetOutput(
                controlnet_block_samples=(
                    tuple(control_block_samples) if control_block_samples else ()
                )
            )
        return tuple(control_block_samples) if control_block_samples else ()


# Register as EntryClass for automatic discovery by ModelRegistry
EntryClass = QwenImageControlNetModel
