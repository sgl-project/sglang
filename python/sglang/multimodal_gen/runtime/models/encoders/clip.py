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
from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig

# TODO: support quantization
# from vllm.model_executor.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import ImageEncoder, TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.vision import (
    resolve_visual_encoder_outputs,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164 # noqa
class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPTextEmbeddings(nn.Module):

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: CLIPVisionConfig | CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tp_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = LocalAttention(
            self.num_heads_per_partition,
            self.head_dim,
            self.num_heads_per_partition,
            softmax_scale=self.scale,
            causal=False,
            supported_attention_backends=config._supported_attention_backends,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        # use flash_attn_func
        query_states = query_states.reshape(
            query_states.shape[0],
            query_states.shape[1],
            self.num_heads_per_partition,
            self.head_dim,
        )
        key_states = key_states.reshape(
            key_states.shape[0],
            key_states.shape[1],
            self.num_heads_per_partition,
            self.head_dim,
        )
        value_states = value_states.reshape(
            value_states.shape[0],
            value_states.shape[1],
            self.num_heads_per_partition,
            self.head_dim,
        )
        attn_output = self.attn(query_states, key_states, value_states)

        attn_output = attn_output.reshape(
            attn_output.shape[0],
            attn_output.shape[1],
            self.num_heads_per_partition * self.head_dim,
        )
        attn_output, _ = self.out_proj(attn_output)

        return attn_output, None


class CLIPMLP(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig | CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        config: CLIPVisionConfig | CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self, inputs_embeds: torch.Tensor, return_all_hidden_states: bool
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return [hidden_states]


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
        r"""
        Returns:

        """
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

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        # causal_attention_mask = _create_4d_causal_attention_mask(
        #     input_shape, hidden_states.dtype, device=hidden_states.device
        # )

        # # expand attention_mask
        # if attention_mask is not None and not self._use_flash_attention_2:
        #     raise NotImplementedError("attention_mask is not supported for CLIPTextTransformer")
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # attention_mask=attention_mask,
            # causal_attention_mask=causal_attention_mask,
            # output_attentions=output_attentions,
            return_all_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[-1]
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
            hidden_states=encoder_outputs,
            # attentions=encoder_outputs.attentions,
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
            encoder_outputs = encoder_outputs[0]

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


EntryClass = [CLIPTextModel, CLIPVisionModel]
