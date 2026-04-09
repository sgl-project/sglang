# Copyright 2025 The LG AI Research Team
# Copyright 2023-2025 SGLang Team
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
"""Inference-only EXAONE-4.5 vision-language model."""

import logging
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.exaone4 import (
    Exaone4ForCausalLM,
    get_attention_sliding_window_size,
)
from sglang.srt.models.utils import WeightsMapper, compute_cu_seqlens_from_grid_numpy
from sglang.srt.utils import add_prefix, is_npu
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


# === Vision Encoder === #


class Exaone45VisionMLP(nn.Module):
    """Gated SiLU MLP for EXAONE-4.5 vision encoder."""

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            in_features,
            [intermediate_size] * 2,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            in_features,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _gqa_rope_applier(q, k, position_embeddings, x_shape):
    """Apply rotary position embeddings for GQA where Q and K have different head counts."""
    cos, sin = position_embeddings
    if cos.size(-1) * 2 == q.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    q = apply_rotary_pos_emb(q, q, cos, sin)[0]
    k = apply_rotary_pos_emb(k, k, cos, sin)[0]
    return q, k


class Exaone45VisionBlock(nn.Module):
    """Transformer block for EXAONE-4.5 vision encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        # Use customized_position_embedding_applier for GQA compatibility
        # VisionAttention's default RoPE path assumes Q and K have same head count
        use_gqa = num_kv_heads != num_heads
        rope_applier = _gqa_rope_applier if use_gqa else None

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            customized_position_embedding_applier=rope_applier,
        )
        if use_gqa:
            self._setup_gqa(dim, num_heads, num_kv_heads, quant_config, prefix)

        self.mlp = Exaone45VisionMLP(
            dim,
            intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def _setup_gqa(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        """Replace the QKV projection with one that supports GQA."""
        head_size = dim // num_heads
        tp_size = self.attn.tp_size
        tp_rank = self.attn.tp_rank

        self.attn.num_attention_kv_heads_per_partition = max(1, num_kv_heads // tp_size)
        self.attn.kv_size = self.attn.num_attention_kv_heads_per_partition * head_size

        self.attn.qkv_proj = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_size,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=True,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            prefix=add_prefix("attn.qkv_proj", prefix),
        )

        # Update the backend's kv head count
        self.attn.qkv_backend.num_kv_heads = (
            self.attn.num_attention_kv_heads_per_partition
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class Exaone45VisionPatchEmbed(nn.Module):
    """3D patch embedding for EXAONE-4.5 vision encoder."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = Conv3dLayer(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.embed_dim)
        return x


class Exaone45VisionPatchMerger(nn.Module):
    """Spatial patch merger for EXAONE-4.5 vision encoder."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Exaone45VisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding for EXAONE-4.5 vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            self._freqs_cached = torch.outer(seq, self.inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Exaone45VisionTransformer(nn.Module):
    """EXAONE-4.5 vision encoder transformer."""

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.spatial_merge_size = vision_config.spatial_merge_size

        self.patch_embed = Exaone45VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_chans=vision_config.in_channels,
            embed_dim=vision_config.hidden_size,
        )

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Exaone45VisionRotaryEmbedding(head_dim // 2)

        num_kv_heads = getattr(
            vision_config, "num_key_value_heads", vision_config.num_heads
        )
        self.blocks = nn.ModuleList(
            [
                Exaone45VisionBlock(
                    dim=vision_config.hidden_size,
                    num_heads=vision_config.num_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=vision_config.intermediate_size,
                    norm_eps=norm_eps,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(vision_config.depth)
            ]
        )
        self.merger = Exaone45VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            spatial_merge_size=vision_config.spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.down_proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")

        # transformer blocks
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        # merge spatial patches
        x = self.merger(x)
        return x


# === VLM Wrapper === #


class Exaone4_5_ForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # To ensure correct weight loading and mapping.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
        },
    )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        vision_config = config.vision_config
        text_config = config.text_config

        self.visual = Exaone45VisionTransformer(
            vision_config,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.language_model = Exaone4ForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def _init_language_model_wrapper(self):
        """Create a wrapper around Exaone4Model that is compatible with
        general_mm_embed_routine (needs get_input_embeddings() -> nn.Embedding
        and forward() -> hidden_states)."""
        model = self.language_model.model

        # Exaone4Model.get_input_embeddings(input_ids) takes an arg,
        # but general_mm_embed_routine expects get_input_embeddings() -> nn.Embedding.
        # Support both signatures to avoid breaking internal calls.
        if not hasattr(self, "_lm_wrapper"):
            original_fn = model.get_input_embeddings

            def flexible_get_input_embeddings(*args, **kwargs):
                if args or kwargs:
                    return original_fn(*args, **kwargs)
                return model.embed_tokens

            model.get_input_embeddings = flexible_get_input_embeddings
            self._lm_wrapper = model
        return self._lm_wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        lm_wrapper = self._init_language_model_wrapper()

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=lm_wrapper,
            multimodal_model=self,
            positions=positions,
        )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        else:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.language_model.lm_head,
                forward_batch,
            )

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config.text_config)

    def _remap_weight_name(self, name: str) -> str:
        """Remap HF weight names to SGLang parameter names."""
        # Apply prefix remapping (longest prefix first)
        prefix_map = [
            ("model.language_model.", "language_model.model."),
            ("model.visual.", "visual."),
            ("lm_head.", "language_model.lm_head."),
        ]
        for old_prefix, new_prefix in prefix_map:
            if name.startswith(old_prefix):
                name = new_prefix + name[len(old_prefix) :]
                break

        # Vision attention QKV naming
        if "visual" in name:
            name = name.replace("attn.qkv.", "attn.qkv_proj.")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Skip MTP weights — loaded by exaone4_5_mtp.py
            if name.startswith("mtp."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Remap HF weight names to SGLang parameter names
            name = self._remap_weight_name(name)

            # Vision QKV weights are already fused — skip stacked mapping
            # to avoid false matches (e.g., "v_proj" matching inside "qkv_proj")
            is_vision_qkv = "visual" in name and "qkv_proj" in name

            if not is_vision_qkv:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name.endswith(".bias") and name not in params_dict:
                        break
                    if name not in params_dict:
                        break
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    # No stacked match — load directly
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            else:
                # Load vision QKV directly (already fused)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Exaone4_5_ForConditionalGeneration
