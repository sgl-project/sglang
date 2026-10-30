# Copyright 2026 SGLang Team
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
"""Nemotron 3.5 Parse: RADIO vision with a causal visual-prefix mBART decoder."""

import math
from array import array

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.radio import RadioConfig
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import general_mm_embed_routine, pad_mm_tokens
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.radio import RadioModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

NECK_COMPRESSION = 4


def content_normalized_positions(
    pos_embed: torch.Tensor,
    image_size: tuple[int, int],
    content_size: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    height, width = content_size
    rows, cols = (size // patch_size for size in image_size)
    source_side = math.isqrt(pos_embed.shape[1])
    field = (
        pos_embed.reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2).float()
    )
    content_grid = (math.ceil(height / patch_size), math.ceil(width / patch_size))
    if min(content_grid) == 1:
        field = F.adaptive_avg_pool2d(field, content_grid)
    else:
        field = F.interpolate(
            field,
            size=content_grid,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def centers(count, extent):
        starts = (
            torch.arange(count, device=pos_embed.device, dtype=torch.float32)
            * patch_size
        )
        # Partial edge patches use the center of their unpadded content.
        spans = (extent - starts).clamp(0, patch_size)
        return (starts + spans / 2).clamp(max=extent) / extent * 2 - 1

    y, x = torch.meshgrid(centers(rows, height), centers(cols, width), indexing="ij")
    grid = torch.stack((x, y), dim=-1).unsqueeze(0)
    positions = F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return positions.flatten(2).transpose(1, 2).to(pos_embed.dtype)


class NemotronParseVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder = config.encoder
        radio_config = RadioConfig(
            model_name=encoder.args["model"],
            patch_size=encoder.patch_size,
            max_img_size=encoder.max_resolution,
            reg_tokens=encoder.args["register_multiple"],
        )
        self.model_encoder = RadioModel(radio_config)
        self.pack_pages = (
            self.model_encoder.model.encoder.layers[0].attn.attn.qkv_backend_name
            != "sdpa"
        )
        self.patch_size = radio_config.patch_size
        self.num_prefix_tokens = self.model_encoder.model.patch_generator.num_skip
        hidden = config.decoder.d_model
        self.conv1 = nn.Conv1d(radio_config.hidden_size, hidden, 1)
        self.layer_norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.conv2 = nn.Conv2d(
            hidden,
            hidden,
            (1, NECK_COMPRESSION),
            stride=(1, NECK_COMPRESSION),
            bias=False,
        )
        self.layer_norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.register_buffer(
            "summary_idxs",
            torch.arange(config.vision_summary_dim // radio_config.hidden_size),
        )
        self.sum_proj = nn.Linear(config.vision_summary_dim, hidden)
        self.layer_norm3 = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        backbone = self.model_encoder.model
        patches = backbone.patch_generator
        pages, sizes, boundaries = [], [], [0]
        for item in items:
            size = tuple(item.model_specific_data["image_size"])
            content = tuple(item.model_specific_data["content_size"])
            pixels = item.feature.to(
                device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
            )
            page = patches.embedder(pixels).unsqueeze(0)
            page = page + content_normalized_positions(
                patches.pos_embed, size, content, self.patch_size
            )
            page = patches.cls_token(page)
            pages.append(page)
            sizes.append(size)
            boundaries.append(boundaries[-1] + page.shape[1])
        if self.pack_pages:
            hidden = backbone.encoder(
                inputs_embeds=torch.cat(pages, dim=1),
                cu_seqlens=torch.tensor(
                    boundaries, dtype=torch.int32, device=pages[0].device
                ),
                return_dict=False,
            )[0]
        else:
            # SDPA expects padded batches, so encode variable-size pages separately.
            hidden = torch.cat(
                [
                    backbone.encoder(inputs_embeds=page, return_dict=False)[0]
                    for page in pages
                ],
                dim=1,
            )
        features = []
        for i, (height, width) in enumerate(sizes):
            page = hidden[:, boundaries[i] : boundaries[i + 1]]
            summary = self.layer_norm3(
                self.sum_proj(page[:, self.summary_idxs].flatten(1))
            ).unsqueeze(1)
            spatial = page[:, self.num_prefix_tokens :]
            spatial = self.layer_norm1(
                self.conv1(spatial.transpose(1, 2)).transpose(1, 2)
            )
            spatial = spatial.transpose(1, 2).reshape(
                1, -1, height // self.patch_size, width // self.patch_size
            )
            spatial = self.layer_norm2(self.conv2(spatial).flatten(2).transpose(1, 2))
            features.append(torch.cat((spatial, summary), dim=1).squeeze(0))
        return torch.cat(features)


class NemotronParseAttention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        tp_size = get_parallel().tp_size
        total_heads = config.decoder_attention_heads
        total_kv_heads = config.self_attention_kv_heads
        if total_heads % tp_size or (
            total_kv_heads >= tp_size and total_kv_heads % tp_size
        ):
            raise ValueError(
                "Parse attention heads must be divisible by the tensor parallel size"
            )
        if total_kv_heads < tp_size and tp_size % total_kv_heads:
            raise ValueError(
                "Tensor parallel size must divide or be divisible by Parse KV heads"
            )
        self.head_dim = config.d_model // total_heads
        self.q_size = total_heads // tp_size * self.head_dim
        self.kv_size = max(1, total_kv_heads // tp_size) * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.d_model,
            self.head_dim,
            total_heads,
            total_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            config.d_model,
            config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        # The trained visual prefix is causal and uses no positional rotation.
        self.attn = RadixAttention(
            total_heads // tp_size,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=max(1, total_kv_heads // tp_size),
            layer_id=layer_id,
            quant_config=quant_config,
        )

    def forward(self, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        output = self.attn(q, k, v, forward_batch)
        return self.out_proj(output)[0]


class NemotronParseDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.self_attn = NemotronParseAttention(
            config, layer_id, quant_config, add_prefix("self_attn", prefix)
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation = get_act_fn(config.activation_function)
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.decoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states, forward_batch):
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states), forward_batch
        )
        output, _ = self.fc1(self.final_layer_norm(hidden_states))
        output, _ = self.fc2(self.activation(output))
        return hidden_states + output


class NemotronParseEmbedding(VocabParallelEmbedding):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.d_model)
        self.scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

    def forward(self, input_ids):
        return super().forward(input_ids) * self.scale


class NemotronParseDecoder(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.embed_tokens = NemotronParseEmbedding(config)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList(
            [
                NemotronParseDecoderLayer(
                    config, i, quant_config, add_prefix(f"layers.{i}", prefix)
                )
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_embeds, forward_batch):
        hidden_states = self.layernorm_embedding(input_embeds)
        for layer in self.layers:
            hidden_states = layer(hidden_states, forward_batch)
        return self.layer_norm(hidden_states)


class RadioMBartPrefixForCausalLM(nn.Module):
    merge_by_field_config = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Nemotron Parse supports document images only")

    def __init__(self, *, config, quant_config=None, prefix="", **kwargs):
        super().__init__()
        if not config.legacy_causal_vision_prefix or config.decoder.add_cross_attention:
            raise ValueError("Nemotron Parse requires the causal visual-prefix decoder")
        self.config = config
        self.encoder = NemotronParseVision(config)
        self.decoder = NemotronParseDecoder(
            config.decoder, quant_config, add_prefix("decoder", prefix)
        )
        self.lm_head = self.decoder.embed_tokens
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def get_embed_and_head(self):
        return self.decoder.embed_tokens.weight, self.lm_head.weight

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self.encoder(items)

    def forward(
        self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None
    ):
        if input_embeds is None:
            return general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self,
                multimodal_model=self,
                positions=positions,
            )
        hidden_states = self.decoder(input_embeds, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        radio_weights = []
        loaded = set()
        for name, weight in weights:
            # Auxiliary projections belong to MTP, not normal decoding.
            if name.startswith(("decoder.extra_heads.", "decoder.extra_proj.")):
                continue
            if name.startswith("encoder.model_encoder."):
                radio_name = name.removeprefix("encoder.model_encoder.")
                if radio_name == "radio_model.summary_idxs":
                    default_weight_loader(self.encoder.summary_idxs, weight)
                    continue
                if radio_name in {
                    "radio_model.input_conditioner.norm_mean",
                    "radio_model.input_conditioner.norm_std",
                }:
                    # The published image processor already normalizes the patches.
                    continue
                radio_weights.append((radio_name, weight))
                continue
            if name == "lm_head.weight":
                name = "decoder.embed_tokens.weight"
            for source, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                if f".self_attn.{source}." in name:
                    name = name.replace(f".self_attn.{source}.", ".self_attn.qkv_proj.")
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    loaded.add(name)
                    break
            else:
                param = params[name]
                if name.startswith("decoder.embed_tokens.") or any(
                    part in name for part in (".self_attn.out_proj.", ".fc1.", ".fc2.")
                ):
                    param.weight_loader(param, weight)
                else:
                    default_weight_loader(param, weight)
                loaded.add(name)
        radio_loaded = self.encoder.model_encoder.load_weights(radio_weights)
        loaded.update(f"encoder.model_encoder.{name}" for name in radio_loaded)
        # The shared RADIO layers initialize LayerScale to one; SO400M has none.
        synthetic = {
            name
            for name in params
            if name.startswith("encoder.model_encoder.")
            and name.endswith((".ls1", ".ls2"))
        }
        missing = params.keys() - loaded - synthetic
        if missing:
            raise ValueError(
                f"Missing Nemotron Parse checkpoint parameters: {sorted(missing)}"
            )
        return loaded


EntryClass = RadioMBartPrefixForCausalLM
