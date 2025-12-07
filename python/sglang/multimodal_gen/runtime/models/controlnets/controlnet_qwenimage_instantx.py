# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
# Use SGLang's QwenImage implementation
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
    QwenEmbedRope,
)
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class QwenImageControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]


class QwenImageControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config=None,  # SGLang config object
        hf_config=None,  # HuggingFace config dict
        # Diffusers-style individual parameters (for compatibility)
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        extra_condition_channels: int = 0, # for controlnet-inpainting
    ):
        # If SGLang config is provided, extract parameters from it
        if config is not None:
            from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
            if isinstance(config, QwenImageDitConfig):
                patch_size = config.arch_config.patch_size
                in_channels = config.arch_config.in_channels
                out_channels = config.arch_config.out_channels
                attention_head_dim = config.arch_config.attention_head_dim
                num_attention_heads = config.arch_config.num_attention_heads
                joint_attention_dim = config.arch_config.joint_attention_dim
                axes_dims_rope = config.arch_config.axes_dims_rope

        # If hf_config is provided, extract num_layers from it
        if hf_config is not None:
            num_layers = hf_config.get("num_layers", num_layers)
            extra_condition_channels = hf_config.get("extra_condition_channels", extra_condition_channels)

        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Store config for param_names_mapping compatibility
        if config is not None:
            self.param_names_mapping = config.arch_config.param_names_mapping

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

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

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
        self.controlnet_x_embedder = zero_module(torch.nn.Linear(in_channels + extra_condition_channels, self.inner_dim))

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 5,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
        extra_condition_channels: int = 0,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        config["extra_condition_channels"] = extra_condition_channels

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.img_in.load_state_dict(transformer.img_in.state_dict())
            controlnet.txt_in.load_state_dict(transformer.txt_in.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

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
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.img_in(hidden_states)

        # add
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            block_samples = block_samples + (hidden_states,)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples)

        return QwenImageControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
        )


class QwenImageMultiControlNetModel(ModelMixin):
    r"""
    `QwenImageMultiControlNetModel` wrapper class for Multi-QwenImageControlNetModel

    This module is a wrapper for multiple instances of the `QwenImageControlNetModel`. The `forward()` API is designed to be
    compatible with `QwenImageControlNetModel`.

    Args:
        controlnets (`List[QwenImageControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `QwenImageControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[QwenImageControlNetOutput, Tuple]:
        # ControlNet-Union with multiple conditions
        # only load one ControlNet for saving memories
        if len(self.nets) == 1:
            controlnet = self.nets[0]

            for i, (image, scale) in enumerate(zip(controlnet_cond, conditioning_scale)):
                block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    conditioning_scale=scale,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                        ]
        else:
            raise ValueError("QwenImageMultiControlNetModel only supports controlnet-union now.")

        return control_block_samples

# Register as EntryClass for automatic discovery
EntryClass = QwenImageControlNetModel
