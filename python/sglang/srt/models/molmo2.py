import copy
import re
from functools import partial
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionConfig

from sglang.srt.configs.molmo2 import Molmo2Config
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.managers.mm_utils import general_mm_embed_routine, pad_mm_tokens
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.siglip import SiglipEncoder

_VISION_PREFIX = "model.vision_backbone."
_VIT_BLOCK = re.compile(r"image_vit\.transformer\.resblocks\.(\d+)\.(.+)")
_VIT_BLOCK_RENAMES = {
    "attention_norm.": "layer_norm1.",
    "ffn_norm.": "layer_norm2.",
    "attention.wo.": "self_attn.proj.",
    "feed_forward.w1.": "mlp.fc1.",
    "feed_forward.w2.": "mlp.fc2.",
}
_VIT_QKV_SHARDS = {"attention.wq.": "q", "attention.wk.": "k", "attention.wv.": "v"}
_ADAPTER_RENAMES = {
    "image_pooling_2d.wq.": "image_pooling.q_proj.",
    "image_pooling_2d.wk.": "image_pooling.k_proj.",
    "image_pooling_2d.wv.": "image_pooling.v_proj.",
    "image_pooling_2d.wo.": "image_pooling.o_proj.",
    "image_projector.w2.": "image_projector.down_proj.",
    "image_vit.patch_embedding.": "vision_tower.patch_embedding.",
}
_PROJECTOR_SHARDS = {"image_projector.w1.": 0, "image_projector.w3.": 1}
_TEXT_BLOCK = re.compile(r"model\.transformer\.blocks\.(\d+)\.(.+)")
_TEXT_BLOCK_RENAMES = {
    "self_attn.attn_out.": "self_attn.o_proj.",
    "self_attn.q_norm.": "self_attn.q_norm.",
    "self_attn.k_norm.": "self_attn.k_norm.",
    "attn_norm.": "input_layernorm.",
    "ff_norm.": "post_attention_layernorm.",
    "mlp.ff_out.": "mlp.down_proj.",
}


class Molmo2VisionTower(nn.Module):
    def __init__(
        self,
        vit_config: dict,
        feature_layers: Tuple[int, ...],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        patch_size = vit_config["image_patch_size"]
        hidden_size = vit_config["hidden_size"]
        self.feature_layers = feature_layers
        encoder_config = SiglipVisionConfig(
            hidden_size=hidden_size,
            intermediate_size=vit_config["intermediate_size"],
            num_hidden_layers=max(feature_layers) + 1,
            num_attention_heads=vit_config["num_attention_heads"],
            layer_norm_eps=vit_config["layer_norm_eps"],
            hidden_act=vit_config["hidden_act"],
            patch_size=patch_size,
            image_size=vit_config["image_default_input_size"][0],
        )
        self.patch_embedding = ReplicatedLinear(
            patch_size * patch_size * 3,
            hidden_size,
            bias=True,
            prefix=add_prefix("patch_embedding", prefix),
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(vit_config["image_num_pos"], hidden_size)
        )
        self.encoder = SiglipEncoder(
            encoder_config,
            quant_config=quant_config,
            act_layer=partial(nn.GELU, approximate="tanh"),
            flatten_batch=False,
            prefix=add_prefix("encoder", prefix),
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.patch_embedding(crops)
        hidden_states = hidden_states + self.positional_embedding.to(
            hidden_states.dtype
        )
        features = {}
        for index, layer in enumerate(self.encoder.layers):
            hidden_states = layer(hidden_states, None, None)
            if index in self.feature_layers:
                features[index] = hidden_states
        return torch.cat([features[layer] for layer in self.feature_layers], dim=-1)


class Molmo2ImagePooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_dim = num_heads * head_dim
        self.q_proj = ReplicatedLinear(
            input_dim, attn_dim, bias=True, prefix=add_prefix("q_proj", prefix)
        )
        self.k_proj = ReplicatedLinear(
            input_dim, attn_dim, bias=True, prefix=add_prefix("k_proj", prefix)
        )
        self.v_proj = ReplicatedLinear(
            input_dim, attn_dim, bias=True, prefix=add_prefix("v_proj", prefix)
        )
        self.o_proj = ReplicatedLinear(
            attn_dim, hidden_size, bias=True, prefix=add_prefix("o_proj", prefix)
        )

    def forward(self, patches: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        num_tokens, pool_size, _ = patches.shape
        num_valid = valid.sum(-1, keepdim=True).clamp(min=1).to(patches.dtype)
        query = patches.sum(1, keepdim=True) / num_valid[..., None]
        q, _ = self.q_proj(query)
        k, _ = self.k_proj(patches)
        v, _ = self.v_proj(patches)
        q = q.view(num_tokens, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(num_tokens, pool_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_tokens, pool_size, self.num_heads, self.head_dim).transpose(1, 2)
        pooled = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=valid[:, None, None, :]
        )
        pooled = pooled.transpose(1, 2).reshape(num_tokens, -1).to(patches.dtype)
        pooled, _ = self.o_proj(pooled)
        return pooled


class Molmo2ImageProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        output_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_dim,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            output_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Molmo2ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Molmo2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        vit_config = config.vit_config
        adapter_config = config.adapter_config
        text_config = config.text_config
        feature_layers = tuple(
            layer if layer >= 0 else layer + vit_config["num_hidden_layers"]
            for layer in adapter_config["vit_layers"]
        )

        self.vision_tower = Molmo2VisionTower(
            vit_config,
            feature_layers=feature_layers,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )
        self.image_pooling = Molmo2ImagePooling(
            input_dim=vit_config["hidden_size"] * len(feature_layers),
            hidden_size=adapter_config["hidden_size"],
            num_heads=adapter_config["num_attention_heads"],
            head_dim=adapter_config["head_dim"],
            prefix=add_prefix("image_pooling", prefix),
        )
        self.image_projector = Molmo2ImageProjector(
            input_dim=adapter_config["hidden_size"],
            intermediate_size=adapter_config["intermediate_size"],
            output_dim=adapter_config["text_hidden_size"],
            quant_config=quant_config,
            prefix=add_prefix("image_projector", prefix),
        )

        self.additional_vocab_size = config.additional_vocab_size
        model_text_config = copy.deepcopy(text_config)
        model_text_config.vocab_size += self.additional_vocab_size
        self.language_model = Qwen3ForCausalLM(
            config=model_text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.language_model.logits_processor = LogitsProcessor(text_config)
        self.qkv_sizes = (
            text_config.num_attention_heads * text_config.head_dim,
            text_config.num_key_value_heads * text_config.head_dim,
            text_config.num_key_value_heads * text_config.head_dim,
        )
        self.register_buffer(
            "image_patch_embedding",
            torch.zeros(text_config.hidden_size),
            persistent=False,
        )

    def pad_input_ids(self, input_ids, mm_inputs: MultimodalInputs):
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        weight = self.vision_tower.patch_embedding.weight
        crops = torch.cat([item.feature for item in items]).to(
            device=weight.device, dtype=weight.dtype
        )
        pooling = _batch_pooling_indices(items, patches_per_crop=crops.shape[1]).to(
            weight.device
        )
        features = self.vision_tower(crops).flatten(0, 1)
        valid = pooling >= 0
        patches = features[pooling.clamp(min=0)] * valid[..., None].to(features.dtype)
        image_embeds = self.image_projector(self.image_pooling(patches, valid))
        return image_embeds + self.image_patch_embedding.to(image_embeds.dtype)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        self.language_model.load_weights(self._text_weights(weights, params))
        if self.language_model.pp_group.is_first_rank:
            self._cache_image_patch_embedding()

    def _text_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], params: dict
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        embedding_parts = {}
        for name, weight in weights:
            if name.startswith(_VISION_PREFIX):
                self._load_vision_weight(name[len(_VISION_PREFIX) :], weight, params)
            elif name.startswith("model.transformer.wte."):
                embedding_parts[name.rsplit(".", 1)[1]] = weight
                if len(embedding_parts) == 2:
                    yield (
                        "model.embed_tokens.weight",
                        torch.cat(
                            [
                                embedding_parts["embedding"],
                                embedding_parts["new_embedding"],
                            ]
                        ),
                    )
            elif name == "lm_head.weight":
                yield (
                    name,
                    torch.cat(
                        [
                            weight,
                            weight.new_zeros(
                                self.additional_vocab_size, weight.shape[1]
                            ),
                        ]
                    ),
                )
            elif name == "model.transformer.ln_f.weight":
                yield "model.norm.weight", weight
            else:
                yield from _text_block_weights(name, weight, qkv_sizes=self.qkv_sizes)

    def _load_vision_weight(self, name: str, weight: torch.Tensor, params: dict):
        if name == "image_vit.positional_embedding":
            default_weight_loader(params["vision_tower.positional_embedding"], weight)
            return
        match = _VIT_BLOCK.fullmatch(name)
        if match is not None:
            layer, rest = match.groups()
            if int(layer) >= len(self.vision_tower.encoder.layers):
                return
            prefix = f"vision_tower.encoder.layers.{layer}."
            for old, shard in _VIT_QKV_SHARDS.items():
                if rest.startswith(old):
                    param = params[prefix + "self_attn.qkv_proj." + rest[len(old) :]]
                    param.weight_loader(param, weight, shard)
                    return
            for old, new in _VIT_BLOCK_RENAMES.items():
                if rest.startswith(old):
                    param = params[prefix + new + rest[len(old) :]]
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )
                    return
            raise ValueError(f"Unexpected Molmo2 vision weight: {name}")
        for old, shard in _PROJECTOR_SHARDS.items():
            if name.startswith(old):
                param = params["image_projector.gate_up_proj." + name[len(old) :]]
                param.weight_loader(param, weight, shard)
                return
        for old, new in _ADAPTER_RENAMES.items():
            if name.startswith(old):
                param = params[new + name[len(old) :]]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                return
        raise ValueError(f"Unexpected Molmo2 vision weight: {name}")

    @torch.no_grad()
    def _cache_image_patch_embedding(self):
        embed_tokens = self.language_model.model.embed_tokens
        token = torch.tensor(
            [self.config.image_patch_id], device=embed_tokens.weight.device
        )
        self.image_patch_embedding.copy_(embed_tokens(token)[0])


def _text_block_weights(
    name: str, weight: torch.Tensor, *, qkv_sizes: Tuple[int, int, int]
) -> Iterator[Tuple[str, torch.Tensor]]:
    match = _TEXT_BLOCK.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected Molmo2 weight: {name}")
    layer, rest = match.groups()
    prefix = f"model.layers.{layer}."
    if rest.startswith("self_attn.att_proj."):
        suffix = rest[len("self_attn.att_proj.") :]
        for proj, part in zip(("q_proj", "k_proj", "v_proj"), weight.split(qkv_sizes)):
            yield f"{prefix}self_attn.{proj}.{suffix}", part
    elif rest.startswith("mlp.ff_proj."):
        suffix = rest[len("mlp.ff_proj.") :]
        up, gate = weight.chunk(2)
        yield f"{prefix}mlp.gate_proj.{suffix}", gate
        yield f"{prefix}mlp.up_proj.{suffix}", up
    else:
        for old, new in _TEXT_BLOCK_RENAMES.items():
            if rest.startswith(old):
                yield prefix + new + rest[len(old) :], weight
                return
        raise ValueError(f"Unexpected Molmo2 weight: {name}")


def _batch_pooling_indices(
    items: List[MultimodalDataItem], patches_per_crop: int
) -> torch.Tensor:
    poolings = []
    patch_offset = 0
    for item in items:
        pooling = item.model_specific_data["image_token_pooling"]
        poolings.append(torch.where(pooling >= 0, pooling + patch_offset, pooling))
        patch_offset += item.feature.shape[0] * patches_per_crop
    return torch.cat(poolings)


EntryClass = Molmo2ForConditionalGeneration
