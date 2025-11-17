# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 Transformer model implementation."""

from typing import Any, Dict, List, Optional

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
        hf_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, hf_config=hf_config)
        self.config = config
        arch_config = getattr(config, "arch_config", None)
        sample_size = getattr(arch_config, "sample_size", 128)
        patch_size = getattr(arch_config, "patch_size", 2)
        in_channels = getattr(arch_config, "in_channels", 16)
        num_layers = getattr(arch_config, "num_layers", 18)
        attention_head_dim = getattr(arch_config, "attention_head_dim", 64)
        num_attention_heads = getattr(arch_config, "num_attention_heads", 18)
        joint_attention_dim = getattr(arch_config, "joint_attention_dim", 4096)
        caption_projection_dim = getattr(arch_config, "caption_projection_dim", 1152)
        pooled_projection_dim = getattr(arch_config, "pooled_projection_dim", 2048)
        out_channels = getattr(arch_config, "out_channels", 16)
        pos_embed_max_size = getattr(arch_config, "pos_embed_max_size", 96)
        dual_attention_layers = getattr(arch_config, "dual_attention_layers", ())
        qk_norm = getattr(arch_config, "qk_norm", None)

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
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
                    use_dual_attention=True if i in dual_attention_layers else False,
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
        encoder_hidden_states: List[torch.Tensor] = [],
        pooled_projections: torch.Tensor = None,  # TODO: this should probably be removed
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # should be convert [prompt_embeds,pooled_embeds]
        assert len(encoder_hidden_states) == 2
        pooled_projections = encoder_hidden_states[1]
        encoder_hidden_states = encoder_hidden_states[0]

        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(
            hidden_states
        )  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        # Copied from diffusers/models/transformers/transformer_sd3.py#SD3Transformer2DModel, but without the self.image_proj method; commented out temporarily for robustness.
        # if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        #     ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        #     ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
        #
        #     joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = (
                True
                if skip_layers is not None and index_block in skip_layers
                else False
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        joint_attention_kwargs,
                    )
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                interval_control = len(self.transformer_blocks) / len(
                    block_controlnet_hidden_states
                )
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[
                        int(index_block / interval_control)
                    ]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
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
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
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
