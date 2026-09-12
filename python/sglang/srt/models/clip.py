# Adapted from
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/clip/modeling_clip.py

from functools import partial
from typing import Iterable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from sglang.srt.layers.activation import QuickGELU, get_act_fn
from sglang.srt.layers.conv import Conv2dLayer
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.model_runner import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, flatten_nested_list


def prepare_clip_attention_mask(
    input_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    batch_size, sequence_length = input_shape
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None].expand(batch_size, 1, -1, -1)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    return causal_mask + attention_mask


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = Conv2dLayer(
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
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        max_positions = self.position_embedding.weight.shape[0]
        if seq_length > max_positions:
            raise ValueError(
                f"Sequence length {seq_length} exceeds the maximum {max_positions}."
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPMLP(nn.Module):
    def __init__(
        self,
        config,
        act_layer: Optional[Type[nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        if act_layer is not None:
            self.act = act_layer()
        elif config.hidden_act == "quick_gelu":
            self.act = QuickGELU()
        else:
            self.act = get_act_fn(config.hidden_act)
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


class CLIPAttention(nn.Module):
    def __init__(
        self,
        config: Union[CLIPTextConfig, CLIPVisionConfig],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        causal: bool = False,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        self.num_heads = config.num_attention_heads // parallel.attn_tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.causal = causal
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
        )
        self.proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        qkv_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query = query.view(qkv_shape).transpose(1, 2)
        key = key.view(qkv_shape).transpose(1, 2)
        value = value.view(qkv_shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attention_mask is None,
            scale=self.scale,
        )
        output = output.transpose(1, 2).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        output, _ = self.proj(output)
        return output


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        act_layer: Optional[Type[nn.Module]] = None,
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        causal: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layer_norm1 = norm_layer(config.hidden_size)
        self.layer_norm2 = norm_layer(config.hidden_size)
        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            causal=causal,
        )
        self.mlp = CLIPMLP(
            config,
            act_layer=act_layer,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attn_mask,
            # causal_attention_mask=causal_attention_mask,
        )

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
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_hidden_layers_override: Optional[int] = None,
        act_layer: Optional[Type[nn.Module]] = None,
        causal: bool = False,
    ) -> None:
        super().__init__()

        self.config = config

        num_hidden_layers = (
            config.num_hidden_layers
            if num_hidden_layers_override is None
            else num_hidden_layers_override
        )
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    config=config,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                    causal=causal,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        return_all_hidden_states: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, attention_mask, causal_attention_mask
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
            causal=True,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @property
    def device(self) -> torch.device:
        return self.encoder.layers[0].layer_norm1.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids, position_ids)
        attention_mask = prepare_clip_attention_mask(
            input_ids.shape,
            hidden_states.dtype,
            hidden_states.device,
            attention_mask,
        )
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        return last_hidden_state


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_model = CLIPTextTransformer(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("text_model", prefix),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        return self.text_model(input_ids, position_ids=position_ids)


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
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
            prefix=add_prefix("encoder", prefix),
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @property
    def device(self) -> torch.device:
        return self.encoder.layers[0].layer_norm1.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device))
        hidden_states = self.pre_layrnorm(hidden_states)

        return_all_hidden_states = False

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class CLIPVisionModel(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config, quant_config, prefix=add_prefix("vision_model", prefix)
        )

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(self, pixel_values: torch.Tensor):
        return self.vision_model(pixel_values)


class CLIPModel(nn.Module):
    def __init__(
        self,
        config: CLIPConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )

        text_model = CLIPTextModel(
            text_config, quant_config, prefix=add_prefix("text_model", prefix)
        )
        vision_model = CLIPVisionModel(
            vision_config, quant_config, prefix=add_prefix("vision_model", prefix)
        )
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        monkey_patch_weight_loader()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = True,
    ):
        assert get_embedding, "CLIPEmbeddingModel is only used for embedding"
        mm_inputs = []
        if forward_batch.mm_inputs is not None:
            mm_inputs = forward_batch.mm_inputs
        pixel_values_list = [
            item.feature
            for item in flatten_nested_list(
                [mm_input.mm_items for mm_input in mm_inputs if mm_input is not None]
            )
        ]
        if len(pixel_values_list) != 0:
            pixel_values = torch.concat(pixel_values_list)
            vision_outputs = self.vision_model(pixel_values)
            pooled_output = vision_outputs[:, 0, :]
            image_embeds = self.visual_projection(pooled_output)
            image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1)
            return EmbeddingPoolerOutput(embeddings=image_embeds)

        else:
            text_outputs = self.text_model(input_ids, position_ids=positions)
            pooled_output = self.pooler(text_outputs[0], forward_batch)
            return EmbeddingPoolerOutput(
                embeddings=self.text_projection(pooled_output.embeddings)
            )

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Clip embeddings models handle text/image separately, so we don't need to pad input ids
        return input_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "position_ids" in name:
                continue
            if "out_proj" in name:
                name = name.replace("out_proj", "proj")
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# monkey patch weight loader to remove open_clip file
def monkey_patch_weight_loader():
    import glob
    import os

    from sglang.srt.model_loader.loader import DefaultModelLoader
    from sglang.srt.model_loader.weight_utils import (
        download_weights_from_hf,
        filter_files_not_needed_for_inference,
    )

    def prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        model_name_or_path = (
            self._maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        use_safetensors = False
        allow_patterns = ["*.bin"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))

        hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        # remove open_clip file
        hf_weights_files = [
            file for file in hf_weights_files if "open_clip" not in file
        ]

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    setattr(DefaultModelLoader, "_prepare_weights", prepare_weights)


EntryClass = CLIPModel
