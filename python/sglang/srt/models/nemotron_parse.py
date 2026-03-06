# Copyright 2025 SGLang Team
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
# Adapted from vllm commit ee21291 nemotron_parse.py

import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from sglang.srt.configs.radio import RadioConfig
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.radio import RadioModel
from sglang.srt.models.whisper import WhisperAttention


class RadioWithNeck(nn.Module):

    def __init__(self, encoder_config, quant_config=None):
        super().__init__()
        self.encoder_config = encoder_config

        args = getattr(encoder_config, "args", {})
        if isinstance(args, dict):
            model_name = args.get("model", "vit_huge_patch16_224")
            reg_tokens = args.get("register_multiple", None)
            cpe_max_size = args.get("cpe_max_size", 2048)
            teachers = args.get("teachers", None)
            cls_token_per_teacher = args.get("cls_token_per_teacher", False)
        else:
            model_name = getattr(args, "model", "vit_huge_patch16_224")
            reg_tokens = getattr(args, "register_multiple", None)
            cpe_max_size = getattr(args, "cpe_max_size", 2048)
            teachers = getattr(args, "teachers", None)
            cls_token_per_teacher = getattr(args, "cls_token_per_teacher", False)

        preferred_resolution = getattr(encoder_config, "preferred_resolution", [768])
        if isinstance(preferred_resolution, list):
            image_size = preferred_resolution[0]
        else:
            image_size = preferred_resolution

        patch_size = getattr(encoder_config, "patch_size", 16)

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            max_img_size=cpe_max_size,
            reg_tokens=reg_tokens,
            teachers=teachers,
            cls_token_per_teacher=cls_token_per_teacher,
        )

        self.radio_model = RadioModel(config=radio_config, quant_config=quant_config)
        self.patch_size = patch_size

        hidden_size = 1024
        encoder_hidden_size = 1280

        self.conv1 = nn.Conv1d(encoder_hidden_size, hidden_size, 1)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.conv2 = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=(1, 4), stride=(1, 4), bias=False
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        summary_dim = 3840
        self.sum_proj = ColumnParallelLinear(summary_dim, hidden_size, bias=True)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=1e-6)

    # The neck bridges RADIO's ViT-H encoder (1280-dim patch features) to the
    # mBART decoder (1024-dim).  The pipeline is:
    #   1. conv1 (1x1 1D): project 1280 -> 1024 per patch token
    #   2. conv2 (1x4 2D): reshape to spatial grid, then compress width by 4x
    #      to cut the sequence length down to something the decoder can attend
    #      over without blowing up cross-attention memory
    #   3. summary: RADIO produces per-teacher CLS tokens; we concatenate the
    #      ones marked use_summary (3 teachers x 1280 = 3840-dim), project to
    #      1024, and append as an extra token so the decoder has a global
    #      image-level representation alongside the spatial patch features
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        summary, features = self.radio_model(pixel_values, return_summary=True)

        output = self.conv1(features.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer_norm1(output)

        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size
        output = rearrange(output, "b (h w) d -> b d h w", h=h, w=w)

        output = self.conv2(output)
        output = rearrange(output, "b d h w -> b (h w) d")
        output = self.layer_norm2(output)

        summary, _ = self.sum_proj(summary)
        summary = self.layer_norm3(summary)
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)

        return output

    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]):
        radio_weights = {}
        neck_params = dict(self.named_parameters())

        for name, weight in weights:
            if name.startswith("encoder.model_encoder."):
                radio_name = name[len("encoder.model_encoder.") :]
                radio_weights[radio_name] = weight
            elif name.startswith("encoder."):
                neck_name = name[len("encoder.") :]
                if neck_name in neck_params:
                    param = neck_params[neck_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, weight)

        if radio_weights:
            self.radio_model.load_weights(radio_weights)


class BartScaledWordEmbedding(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, pad_token_id: int = 1):
        super().__init__()
        self.embed = VocabParallelEmbedding(vocab_size, embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.padding_idx = pad_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids) * self.embed_scale


class MBartDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        activation_function: str = "gelu",
        self_attn_layer_id: int = 0,
        cross_attn_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = d_model

        self.self_attn = WhisperAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            layer_id=self_attn_layer_id,
            quant_config=quant_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = WhisperAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            layer_id=cross_attn_layer_id,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.activation_fn = get_act_fn(activation_function, quant_config=quant_config)
        self.fc1 = ColumnParallelLinear(d_model, ffn_dim)
        self.fc2 = RowParallelLinear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states, forward_batch, encoder_hidden_states
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MBartDecoderNoPos(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        decoder_layers: int,
        decoder_attention_heads: int,
        decoder_ffn_dim: int,
        pad_token_id: int = 1,
        activation_function: str = "gelu",
        scale_embedding: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.embed_tokens = BartScaledWordEmbedding(vocab_size, d_model, pad_token_id)
        self.layernorm_embedding = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList(
            [
                MBartDecoderLayer(
                    d_model=d_model,
                    num_heads=decoder_attention_heads,
                    ffn_dim=decoder_ffn_dim,
                    activation_function=activation_function,
                    self_attn_layer_id=layer_idx,
                    cross_attn_layer_id=decoder_layers + layer_idx,
                    quant_config=quant_config,
                )
                for layer_idx in range(decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.layernorm_embedding(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, forward_batch)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class NemotronParseForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        decoder_config = config.decoder

        self.encoder = RadioWithNeck(config.encoder, quant_config)

        self.decoder = MBartDecoderNoPos(
            vocab_size=decoder_config.vocab_size,
            d_model=decoder_config.d_model,
            decoder_layers=decoder_config.decoder_layers,
            decoder_attention_heads=decoder_config.decoder_attention_heads,
            decoder_ffn_dim=decoder_config.decoder_ffn_dim,
            pad_token_id=getattr(decoder_config, "pad_token_id", 1),
            activation_function=getattr(decoder_config, "activation_function", "gelu"),
            scale_embedding=getattr(decoder_config, "scale_embedding", True),
            quant_config=quant_config,
        )

        self.lm_head = ParallelLMHead(
            decoder_config.vocab_size, decoder_config.d_model, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(decoder_config)

        self._encoder_cache = {}

    def pad_input_ids(self, input_ids: List[int], _mm_inputs: MultimodalInputs):
        return input_ids

    # This is an encoder-decoder model, so the forward pass has two distinct
    # phases that map onto sglang's prefill/decode split.  During prefill we run
    # the vision encoder once per request and stash the result in _encoder_cache
    # keyed by req_pool_index.  During decode we just look up the cached encoder
    # output -- the image doesn't change between tokens, so re-encoding would be
    # pure waste.  The encoder output is fed as cross-attention context to every
    # decoder layer on every step, which is why it needs to stick around for the
    # entire lifetime of the request.
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        is_decode = forward_batch.forward_mode.is_decode()

        if is_decode:
            encoder_outputs = None
            if forward_batch.req_pool_indices is not None:
                req_indices = forward_batch.req_pool_indices.tolist()
                encoder_list = []
                for req_idx in req_indices:
                    if req_idx in self._encoder_cache:
                        encoder_list.append(self._encoder_cache[req_idx])
                if encoder_list:
                    encoder_outputs = torch.cat(encoder_list, dim=0)
        else:
            encoder_list = []
            mm_inputs_list = forward_batch.mm_inputs if forward_batch.mm_inputs else []
            req_indices = (
                forward_batch.req_pool_indices.tolist()
                if forward_batch.req_pool_indices is not None
                else []
            )

            for req_idx, mm_input in zip(req_indices, mm_inputs_list):
                if mm_input is None or not mm_input.mm_items:
                    continue

                pixel_values = mm_input.mm_items[0].feature
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)

                dtype = self.encoder.conv1.weight.dtype
                req_encoder_outputs = self.encoder(pixel_values.to(dtype))
                req_encoder_outputs = req_encoder_outputs.squeeze(0)

                self._encoder_cache[req_idx] = req_encoder_outputs
                encoder_list.append(req_encoder_outputs)

            if encoder_list:
                encoder_outputs = torch.cat(encoder_list, dim=0)
            else:
                encoder_outputs = None

        decoder_outputs = self.decoder(input_ids, encoder_outputs, forward_batch)

        logits = self.logits_processor(
            input_ids=input_ids,
            lm_head=self.lm_head,
            hidden_states=decoder_outputs,
            logits_metadata=forward_batch,
        )

        return logits

    # Weight loading has three complications worth knowing about:
    #
    # 1. The checkpoint splits into encoder weights (prefixed "encoder.") which
    #    get routed to RadioWithNeck, and decoder weights which stay here.  The
    #    encoder further splits: "encoder.model_encoder.*" goes to RadioModel,
    #    everything else is the neck (conv1/conv2/sum_proj/layer_norms).
    #
    # 2. BartScaledWordEmbedding wraps VocabParallelEmbedding as .embed, so
    #    "decoder.embed_tokens.weight" needs remapping to
    #    "decoder.embed_tokens.embed.weight".
    #
    # 3. The original mBART checkpoint has separate q/k/v projection weights for
    #    self-attention and separate k/v for cross-attention.  We fuse them into
    #    qkv_proj and kv_proj respectively via stacked_params_mapping.  The
    #    checkpoint also ships no bias for cross-attention k_proj (only q_proj
    #    has one), but WhisperAttention's fused kv_proj expects a bias for both
    #    k and v.  We synthesize a zero bias for k_proj to keep the math right.
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        weights_list = list(weights)

        encoder_weights = []

        weights_dict = dict(weights_list)
        decoder_layers = self.config.decoder.decoder_layers
        for layer_idx in range(decoder_layers):
            prefix = f"decoder.layers.{layer_idx}.encoder_attn."
            k_proj_key = prefix + "k_proj.weight"
            if k_proj_key in weights_dict:
                bias_key = prefix + "k_proj.bias"
                if bias_key not in weights_dict:
                    weights_dict[bias_key] = torch.zeros(
                        weights_dict[k_proj_key].size(0)
                    )

        for name, loaded_weight in weights_dict.items():
            if name.startswith("encoder."):
                encoder_weights.append((name, loaded_weight))
                continue

            if name == "lm_head.weight":
                if "lm_head.weight" in params_dict:
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                continue

            mapped_name = name
            if mapped_name == "decoder.embed_tokens.weight":
                mapped_name = "decoder.embed_tokens.embed.weight"

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                mapped_name = mapped_name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        if encoder_weights:
            self.encoder.load_weights(encoder_weights)


EntryClass = [NemotronParseForConditionalGeneration]
