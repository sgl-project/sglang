import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import einops
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, PretrainedConfig
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# TODO: This should be a config in huggingface...
@dataclass
class MolmoVisionBackboneConfig:
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 23
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "quick_gelu"
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class PlaceholderConfig:
    hidden_size: int = 3584
    intermediate_size: int = 37888


class MolmoViTMLP(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.image_emb_dim, config.image_mlp_dim, bias=True)
        self.w2 = nn.Linear(config.image_mlp_dim, config.image_emb_dim, bias=True)
        self.act = QuickGELU()

    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class MolmoMultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig, num_vit_layers: int = 1):
        super().__init__()
        self.config = config
        self.hidden_size = config.image_emb_dim
        self.total_num_heads = config.image_num_heads
        self.total_num_kv_heads = config.image_num_key_value_heads
        self.head_dim = config.image_emb_dim // config.image_num_heads
        self.num_vit_layers = num_vit_layers

        self.wq = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.total_num_heads * self.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            self.total_num_heads * self.head_dim, self.hidden_size, bias=True
        )

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.total_num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.total_num_heads * self.head_dim,)
        )

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor = None):
        if inputs_kv is None:
            inputs_k = inputs_q
            inputs_v = inputs_q
        else:
            inputs_k = inputs_kv
            inputs_v = inputs_kv

        # (bsz, seq_len, embed_dim)
        q = self.wq(inputs_q)
        k = self.wk(inputs_k)
        v = self.wv(inputs_v)

        # TODO: Implement flash attention or xformers
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # (bsz, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # (bsz, num_heads, seq_len, head_dim) -> transpose to (bsz, seq_len, num_heads, head_dim)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)

        # (bsz, seq_len, num_heads, head_dim) -> (bsz, seq_len, embed_dim)
        attn_output = self._merge_heads(attn_output)
        output = self.wo(attn_output)

        return output


class MolmoResidualAttentionBlock(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.attention = MolmoMultiHeadDotProductAttention(config)
        self.feed_forward = MolmoViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim, eps=config.image_norm_eps
        )
        self.ffn_norm = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class BlockCollection(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.resblocks = nn.ModuleList(
            [
                MolmoResidualAttentionBlock(config)
                for _ in range(config.image_num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        hidden_states = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)

        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.scale = config.image_emb_dim**-0.5
        self.patch_num = config.image_num_patch
        self.num_prefix_tokens = 1
        self.class_embedding = nn.Parameter(
            torch.randn(config.image_emb_dim) * self.scale
        )
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.image_emb_dim) * self.scale
        )
        self.patch_embedding = nn.Linear(
            config.image_patch_size * config.image_patch_size * 3,
            config.image_emb_dim,
            bias=False,
        )
        self.transformer = BlockCollection(config)
        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def _expand_token(self, token: torch.Tensor, batch_size: int) -> torch.Tensor:
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def _add_pos_embed(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_embed = self.positional_embedding[0:1]
        pos_embed = self.positional_embedding[1:]

        pos_embed = pos_embed.reshape(
            (
                int(math.sqrt(pos_embed.shape[0])),
                int(math.sqrt(pos_embed.shape[0])),
                pos_embed.shape[1],
            )
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_embed.shape[0] != patch_num_0 or pos_embed.shape[1] != patch_num_1:
            # (1, patch_h, patch_w, emb_dim) -> (1, emb_dim, patch_h, patch_w)
            pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # (1, emb_dim, patch_h, patch_w) -> (patch_h, patch_w, emb_dim)
            pos_embed = pos_embed.permute(0, 2, 3, 1).squeeze(0)

        pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])

        # cat: (1, embed_dim) + (patch_h * patch_w, embed_dim) -> (1, patch_h * patch_w + 1, embed_dim)
        x = x + torch.cat([cls_embed[None, :, :], pos_embed[None, :, :]], dim=1).to(
            x.dtype
        )
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        if patch_num is None:
            patch_num = self.patch_num

        B, N, D = x.shape
        x = self.patch_embedding(x)

        # Expand the class embedding to batch size then concate along the patch dimension
        x = torch.cat(
            [
                self._expand_token(self.class_embedding, x.shape[0]).to(
                    x.device, x.dtype
                ),
                x,
            ],
            dim=1,
        )
        x = self._add_pos_embed(x, patch_num)
        x = self.pre_ln(x)
        hidden_states = self.transformer(x)
        return hidden_states


class MolmoAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads or self.total_num_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.k_norm: Optional[nn.Module] = None
        self.q_norm: Optional[nn.Module] = None
        if config.attention_layer_norm:
            self.k_norm = RMSNorm(self.kv_size, eps=config.layer_norm_eps)
            self.q_norm = RMSNorm(self.q_size, eps=config.layer_norm_eps)

        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )
        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm.forward_native(q)
            k = self.k_norm.forward_native(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MolmoMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size // 2
        self.gate_up_proj = MergedColumnParallelLinear(
            input_dim or config.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act(gate_up)
        x, _ = self.down_proj(x)
        return x


class MolmoDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.self_attn = MolmoAttention(config, layer_id, quant_config)
        self.mlp = MolmoMLP(config, quant_config=quant_config)

        # LayerNorm
        assert config.layer_norm_type == "rms"
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.norm_after = config.norm_after

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ):
        if self.norm_after:
            residual = hidden_states
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            input_metadata=input_metadata,
        )

        if self.norm_after:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = hidden_states + residual
            residual = hidden_states

        if not self.norm_after:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        hidden_states = self.mlp(hidden_states)
        if self.norm_after:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + residual
            residual = None

        return hidden_states, residual


class MolmoModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_size = config.embedding_size or config.vocab_size
        # TODO: extra embedding_size hard-coded for now. Consider making it configurable.
        self.embedding_size += 128
        self.embed_tokens = VocabParallelEmbedding(
            self.embedding_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        self.layers = nn.ModuleList(
            [
                MolmoDecoderLayer(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )

        assert config.layer_norm_type == "rms"
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, positions, input_metadata, residual
            )

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class MolmoVisionBackbone(nn.Module):
    def __init__(
        self,
        config,
        vision_config: MolmoVisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.config = vision_config
        self.vit_layers = [-2, -9]
        self.image_vit = VisionTransformer(vision_config)
        self.image_projector = MolmoMLP(
            config,
            input_dim=vision_config.image_emb_dim,
            quant_config=quant_config,
        )
        self.pad_embed = nn.Parameter(
            torch.zeros((2, vision_config.image_emb_dim * len(self.vit_layers)))
        )
        self.image_pooling_2d = MolmoMultiHeadDotProductAttention(
            vision_config, num_vit_layers=len(self.vit_layers)
        )

        self.image_num_patch = vision_config.image_num_patch
        self.llm_patches_per_crop = (
            (self.image_num_patch[0] + 1) // 2,
            (self.image_num_patch[1] + 1) // 2,
        )

    def encode_image(self, images: torch.Tensor):
        B, T, N, D = images.shape

        # Mask checks if all tokens are equal to -1
        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)
        images = images.view(B * T, N, D)

        hidden_states = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(hidden_states[layer])
            image_features = torch.cat(features, dim=-1)

        # Since num_prefix_tokens is always 1, we can just take the first token
        cls_embed = image_features[:, 0]
        image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1)

        return image_features, cls_embed

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor):
        batch_size, num_images = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        og_dtype = image_features.dtype
        assert image_masks is not None
        pad_embed = self.pad_embed[:, None, None, None, :]
        all_pad = image_masks == 0
        partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(
            device=image_features.device, dtype=torch.float32
        )
        all_pad = all_pad.to(device=image_features.device, dtype=torch.float32)
        image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
        image_features = image_features + pad_embed[1] * torch.unsqueeze(
            partial_pad, -1
        )

        image_features = image_features.to(og_dtype)

        # Dropout was none
        image_features = image_features.reshape(
            (batch_size, num_images) + self.config.image_num_patch + (-1,)
        )

        if self.image_num_patch[0] % 2 == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )

        # Divide into 2x2 patches
        image_features = einops.rearrange(
            image_features,
            "b n (h dh) (w dw) c -> (b n h w) (dh dw) c",
            dh=2,
            dw=2,
        )

        # NOTE(chris): to float16?
        query = image_features.mean(-2, keepdim=True)
        image_features = self.image_pooling_2d(query, image_features)

        h, w = self.llm_patches_per_crop
        image_features = image_features.reshape(batch_size, num_images, h * w, -1)

        # MLP layer to map the feature.
        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features, cls_embed


class MolmoForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.model = MolmoModel(config, quant_config)
        self.vision_backbone = MolmoVisionBackbone(
            config, MolmoVisionBackboneConfig(), quant_config
        )

        if self.config.weight_tying:
            self.lm_head = self.model.transformer.wte
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        self.logits_processor = LogitsProcessor(config)

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        if image_inputs is not None and image_inputs.input_ids is not None:
            return image_inputs.input_ids.tolist()
        else:
            return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = input_metadata.image_inputs

        if input_metadata.forward_mode.is_extend():
            images = []
            image_masks = []
            image_input_idx = []

            for im in image_inputs:
                if im is not None and im.pixel_values is not None:
                    images.append(im.pixel_values)
                    image_masks.append(im.image_masks)
                    image_input_idx.append(im.image_input_idx)

            if images is not None and len(images) > 0:
                images = torch.stack(images, dim=0)
                image_masks = torch.stack(image_masks, dim=0)
                image_input_idx = torch.stack(image_input_idx, dim=0)

                input_embeds = self.model.embed_tokens(input_ids)

                # Is unsqueeze necessary?
                seq_len = torch.tensor(
                    len(input_ids), device=input_embeds.device, dtype=torch.long
                ).unsqueeze(0)

                images = images.to(device=input_embeds.device, dtype=input_embeds.dtype)
                image_features, cls_embed = self.vision_backbone(
                    images=images, image_masks=image_masks
                )
                batch_size = images.shape[0]
                num_image, num_patch = image_features.shape[1:3]

                assert image_input_idx.shape == (batch_size, num_image, num_patch)

                image_features = image_features.to(input_embeds.device)
                image_features = image_features.view(
                    batch_size, num_image * num_patch, -1
                )
                image_input_idx = image_input_idx.view(
                    batch_size, num_image * num_patch
                ).to(device=image_features.device)

                valid = (image_input_idx >= 0).to(
                    device=image_features.device, dtype=image_features.dtype
                )
                image_features = image_features * valid[:, :, None]
                image_features = image_features.view(
                    batch_size * num_image * num_patch, -1
                ).contiguous()

                image_input_idx = image_input_idx * valid
                offset = torch.cat(
                    [seq_len.new_zeros((1)), seq_len.cumsum(dim=0)[:-1]], dim=0
                )[:, None]
                image_input_idx = image_input_idx + offset.to(image_input_idx.dtype)
                image_input_idx = image_input_idx.flatten()[:, None]
                mat = (
                    image_input_idx
                    == torch.arange(seq_len.sum().item(), device=input_embeds.device)[
                        None, :
                    ]
                )
                mat = mat.to(image_features.dtype)
                input_embeds = input_embeds + torch.einsum(
                    "nd,nm->md", image_features, mat
                )
            else:
                input_embeds = self.model.embed_tokens(input_ids)

            input_ids = None

            hidden_states = self.model(
                input_ids, positions, input_metadata, input_embeds=input_embeds
            )
        elif input_metadata.forward_mode.is_decode():
            hidden_states = self.model(input_ids, positions, input_metadata)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, input_metadata
        )

    def try_load_weights(
        self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Module]
    ):
        try:
            param = params_dict[name]
        except KeyError:
            print(f"no {name}")
            raise
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            weight_loader(param, loaded_weight)
        except:
            raise

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        embedding_weight = {}
        projector_weight = {}
        language_model_weights_mapping = {
            "model.transformer": "model",
            "blocks": "layers",
            "ln_f": "norm",
            "attn_out": "self_attn.o_proj",
            "att_proj": "self_attn.qkv_proj",
            "q_norm": "self_attn.q_norm",
            "k_norm": "self_attn.k_norm",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        }
        projector_weights_mapping = {
            "w2": "down_proj",
        }
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                log.info(f"Skipping {name}")
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if "wte.embedding" in name:
                embedding_weight["embedding"] = loaded_weight
                continue

            if "wte.new_embedding" in name:
                embedding_weight["new_embedding"] = loaded_weight
                continue

            if "vision_backbone" in name:
                if name.startswith("model"):
                    name = name[len("model.") :]

                if "image_projector" in name:
                    if "w1" in name:
                        projector_weight["gate_proj"] = loaded_weight
                    elif "w2" in name:
                        projector_weight["down_proj"] = loaded_weight
                    elif "w3" in name:
                        projector_weight["up_proj"] = loaded_weight
                    continue
            else:
                if "ff_proj" in name:
                    name = name.replace("ff_proj", "mlp.gate_up_proj")
                    assert "weight" in name
                    up_weight, gate_weight = loaded_weight.chunk(2, dim=0)
                    loaded_weight = torch.cat([gate_weight, up_weight], dim=0)

                for k, v in language_model_weights_mapping.items():
                    if k in name:
                        name = name.replace(k, v)

                if "ff_out" in name:
                    if "layers" in name:
                        name = name.replace("ff_out", "mlp.down_proj")
                    else:
                        # lm head
                        name = name.replace("model.ff_out", "lm_head")

            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        gate_up_proj_weight = torch.cat(
            [projector_weight["gate_proj"], projector_weight["up_proj"]], dim=0
        )
        name = "vision_backbone.image_projector.gate_up_proj.weight"
        self.try_load_weights(name, gate_up_proj_weight, params_dict)
        down_proj_weight = projector_weight["down_proj"]
        name = "vision_backbone.image_projector.down_proj.weight"
        self.try_load_weights(name, down_proj_weight, params_dict)
        embedding_weight = torch.cat(
            [embedding_weight["embedding"], embedding_weight["new_embedding"]], dim=0
        )
        name = "model.embed_tokens.weight"
        self.try_load_weights(name, embedding_weight, params_dict)


EntryClass = [MolmoForCausalLM]
