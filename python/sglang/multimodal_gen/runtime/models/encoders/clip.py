# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/clip.py
# Adapted from transformers: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py
"""Minimal implementation of CLIPVisionModel intended to be only used
within a vision language model."""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import ImageEncoder, TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.vision import (
    resolve_visual_encoder_outputs,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.clip import (
    CLIPEncoder,
    CLIPTextEmbeddings,
    CLIPVisionEmbeddings,
    prepare_clip_attention_mask,
)


def _srt_clip_param_name(name: str) -> str:
    return name.replace(".out_proj.", ".proj.")


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPTextEmbeddings(config)

        self.encoder = CLIPEncoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=prefix,
            causal=True,
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        attention_mask = prepare_clip_attention_mask(
            input_shape,
            hidden_states.dtype,
            hidden_states.device,
            attention_mask,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
        )

        if output_hidden_states:
            all_hidden_states = encoder_outputs
            last_hidden_state = encoder_outputs[-1]
        else:
            last_hidden_state = encoder_outputs
            all_hidden_states = [encoder_outputs]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                    dim=-1
                ),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                    == self.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]

        return BaseEncoderOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )


class CLIPTextModel(TextEncoder):
    def __init__(
        self,
        config: CLIPTextConfig,
    ) -> None:
        super().__init__(config)
        self.text_model = CLIPTextTransformer(
            config=config, quant_config=config.quant_config, prefix=config.prefix
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:

        outputs: BaseEncoderOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        # Define mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = _srt_clip_param_name(name)
            # Handle q_proj, k_proj, v_proj -> qkv_proj mapping
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    # Replace the weight name with the parameter name
                    model_param_name = name.replace(weight_name, param_name)

                    if model_param_name in params_dict:
                        param = params_dict[model_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(model_param_name)
                    break
            else:
                # Use default weight loader for all other parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class CLIPTextModelWithProjection(CLIPTextModel):
    """
    CLIP text encoder with projection head for pooled_output.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
    ) -> None:
        super().__init__(config)
        self.text_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        outputs: BaseEncoderOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs.pooler_output
        if pooled_output is not None:
            pooled_output = self.text_projection(pooled_output)

        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        feature_sample_layers: list[int] | None = None,
    ) -> BaseEncoderOutput:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        return_all_hidden_states = output_hidden_states or (
            feature_sample_layers is not None
        )

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        if not return_all_hidden_states:
            # Handle post-norm (if applicable) and stacks feature layers if needed
            encoder_outputs = resolve_visual_encoder_outputs(
                encoder_outputs,
                feature_sample_layers,
                self.post_layernorm,
                self.config.num_hidden_layers,
            )

        if return_all_hidden_states:
            return BaseEncoderOutput(hidden_states=encoder_outputs)

        return BaseEncoderOutput(last_hidden_state=encoder_outputs)


class CLIPVisionModel(ImageEncoder):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    # CLIPEncoder and its linears come from SRT, so their serialized checkpoint
    # format must use the matching SRT quantization implementation.
    checkpoint_quantization_backend = "srt"

    def __init__(self, config: CLIPVisionConfig) -> None:
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=config.quant_config,
            num_hidden_layers_override=config.num_hidden_layers_override,
            require_post_norm=config.require_post_norm,
            prefix=f"{config.prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: list[int] | None = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        base_encoder_output = self.vision_model(
            pixel_values,
            output_hidden_states=output_hidden_states,
            feature_sample_layers=feature_sample_layers,
        )

        return base_encoder_output

    @property
    def device(self):
        return next(self.parameters()).device

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            if name.startswith("visual_projection"):
                continue
            name = _srt_clip_param_name(name)
            # post_layernorm is not needed in CLIPVisionModel
            if (
                name.startswith("vision_model.post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (
                param_name,
                weight_name,
                shard_id,
            ) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BertModel(CLIPTextModel):
    pass


EntryClass = [CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModel]
