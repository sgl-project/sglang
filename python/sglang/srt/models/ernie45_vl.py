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
"""Inference-only Ernie45-VL model compatible with HuggingFace weights."""
import logging
from functools import lru_cache, partial
from typing import Iterable, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig

from sglang.srt.layers.activation import QuickGELU
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.ernie45_moe_vl import Ernie4_5_VLMoeModel
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)


# === Vision Encoder === #


class Ernie4_5_VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        act_layer: Type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.act = act_layer()
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


class Ernie4_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Ernie4_5_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_layer=act_layer,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
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


class Ernie4_5_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(target_dtype)
        hidden_states = self.proj(hidden_states)

        return hidden_states


class VariableResolutionResamplerModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        spatial_conv_size,
        temporal_conv_size,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        self.spatial_linear1 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.spatial_linear1",
        )

        self.spatial_gelu = nn.GELU()

        self.spatial_linear2 = ColumnParallelLinear(
            self.spatial_dim,
            self.spatial_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.spatial_linear2",
        )

        self.spatial_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        if self.use_temporal_conv:
            self.temporal_linear1 = ColumnParallelLinear(
                self.temporal_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, "quant_config", None),
                prefix=f"{prefix}.temporal_linear1",
            )

            self.temporal_gelu = nn.GELU()

            self.temporal_linear2 = ColumnParallelLinear(
                self.spatial_dim,
                self.spatial_dim,
                bias=True,
                gather_output=True,
                quant_config=getattr(config, "quant_config", None),
                prefix=f"{prefix}.temporal_linear2",
            )

            self.temporal_norm = nn.LayerNorm(self.spatial_dim, eps=1e-6)

        self.mlp = ColumnParallelLinear(
            self.spatial_dim,
            self.out_dim,
            bias=True,
            gather_output=True,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.mlp",
        )

        self.after_norm = RMSNorm(
            hidden_size=out_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def spatial_conv_reshape(self, x, spatial_conv_size):
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, grid_thw):
        def fwd_spatial(x):
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x, _ = self.spatial_linear1(x)
            x = self.spatial_gelu(x)
            x, _ = self.spatial_linear2(x)
            x = self.spatial_norm(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to(
                x.device
            )

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to(
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x, _ = self.temporal_linear1(x)
            x = self.temporal_gelu(x)
            x, _ = self.temporal_linear2(x)
            x = self.temporal_norm(x)
            return x

        def fwd_mlp(x):
            x, _ = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Ernie4_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


class Ernie4_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: PretrainedConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        in_chans: int = vision_config.in_chans
        hidden_size: int = vision_config.hidden_size
        embed_dim: int = vision_config.embed_dim
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        mlp_ratio: float = vision_config.mlp_ratio

        self.spatial_merge_size = spatial_merge_size

        self.patch_embed = Ernie4_5_VisionPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Ernie4_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Ernie4_5_VisionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(depth)
            ]
        )

        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

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
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        final_output = self.ln(x)

        if final_output.ndim == 3:
            final_output = final_output.squeeze(dim=1)

        return final_output


cached_get_processor = lru_cache(get_processor)


class Ernie4_5_VLMoeForConditionalGeneration(nn.Module):
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
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_model = Ernie4_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )

        self.model = Ernie4_5_VLMoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        self.resampler_model = VariableResolutionResamplerModel(
            self.config.pixel_hidden_size,
            self.config.hidden_size,
            self.config.spatial_conv_size,
            self.config.temporal_conv_size,
            config=self.config,
            prefix=add_prefix("resampler_model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling
        self.logits_processor = LogitsProcessor(config)

        if getattr(self.config, "im_patch_id", None):
            visual_token_ids = [
                token_id
                for token_id in [
                    self.config.im_patch_id,
                    getattr(self.config, "image_start_token_id", None),
                    getattr(self.config, "image_end_token_id", None),
                    getattr(self.config, "video_start_token_id", None),
                    getattr(self.config, "video_end_token_id", None),
                ]
                if token_id is not None
            ]
            self._visual_token_ids_tensor_cache = torch.tensor(
                visual_token_ids, dtype=torch.long
            )
        else:
            self._visual_token_ids_tensor_cache = None

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0]
            if grid_thw.numel() % 3 != 0:
                raise ValueError(
                    f"grid_thw has {grid_thw.numel()} elements after filtering,"
                    "which is not divisible by 3."
                )
            grid_thw = grid_thw.reshape(-1, 3)
            # example: [[1,64,64],[2,80,80]] -> [[1,64,64],[1,80,80],[1,80,80]]
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_features = self.vision_model(pixel_values, grid_thw)
        return image_features

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_model.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_feature = self._vision_forward(pixel_values, grid_thw=image_grid_thw)
        image_embeds = self.resampler_model(image_feature, image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_model.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        video_feature = self._vision_forward(pixel_values, grid_thw=video_grid_thw)
        video_embeds = self.resampler_model(video_feature, video_grid_thw)
        return video_embeds

    def _set_visual_token_mask(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        """Set mask for visual tokens (image/video patches and delimiters)."""
        if self._visual_token_ids_tensor_cache is None:
            self.visual_token_mask = None
            return
        # Create tensor on the correct device
        visual_token_ids_tensor = self._visual_token_ids_tensor_cache.to(
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        pad_values = []
        if hasattr(forward_batch, "mm_inputs") and forward_batch.mm_inputs is not None:
            for mm_input in forward_batch.mm_inputs:
                if mm_input is None:
                    continue
                for item in mm_input.mm_items:
                    pad_values.append(item.pad_value)
        placeholder_tensor = torch.as_tensor(
            pad_values,
            device=input_ids.device,
        )
        pad_visual_token_ids_tensor = torch.cat(
            [visual_token_ids_tensor, placeholder_tensor], dim=0
        )
        self.visual_token_mask = torch.isin(
            input_ids, pad_visual_token_ids_tensor
        ).reshape(-1, 1)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def should_apply_lora(self, module_name: str) -> bool:
        # skip vision_model
        return not module_name.startswith("vision_model")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        """Run forward pass for Ernie45-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        self._set_visual_token_mask(input_ids, forward_batch)

        assert (
            input_ids.numel() == positions.shape[-1]
        ), f"input_ids {input_ids.shape} and position_ids {positions.shape} should have the same length"

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            visual_token_mask=self.visual_token_mask,
        )

        self.visual_token_mask = None

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        # resampler_weight_mappings
        resampler_weight_mapping = {
            "spatial_linear.0.": "spatial_linear1.",
            "spatial_linear.2.": "spatial_linear2.",
            "spatial_linear.3.": "spatial_norm.",
            "temporal_linear.0.": "temporal_linear1.",
            "temporal_linear.2.": "temporal_linear2.",
            "temporal_linear.3.": "temporal_norm.",
        }

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max(self.config.moe_num_experts),
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_model" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                if name.startswith("model.resampler_model"):
                    name = name.replace("model.resampler_model", "resampler_model")

                for (
                    old_weight_name,
                    new_weight_name,
                ) in resampler_weight_mapping.items():
                    if old_weight_name in name:
                        name = name.replace(old_weight_name, new_weight_name, 1)
                        break

                # Distinguish between vision experts and text experts
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    vision_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(
                            f".experts.{moe_offset}",
                            f".vision_experts.{moe_offset - vision_expert_start_idx}",
                        )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Distinguish between vision experts and text experts
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1

                    name = name.replace(weight_name, param_name)
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".vision_experts.")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
                    break
                else:
                    # Distinguish between vision expert gate
                    # and text expert gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight"
                        )
                        loaded_weight = loaded_weight.T

                    if "e_score_correction_bias" in name:
                        name = name.replace(".moe_statics.", ".")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = [Ernie4_5_VLMoeForConditionalGeneration]
