# Copyright 2023-2024 SGLang Team
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

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4v/modular_glm4v.py
"""Inference-only GLM-4.1V model compatible with HuggingFace weights."""

import logging
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.glm4v.configuration_glm4v import Glm4vConfig, Glm4vVisionConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Glm4vRMSNorm(RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.contiguous().reshape(-1, original_shape[-1])
        x_2d = super().forward(x_2d)
        x = x_2d.reshape(original_shape)
        return x


class Glm4vVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_qkv_bias: bool = True,
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-5,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=False,
            qkv_bias=attn_qkv_bias,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Glm4vVisionMLP(
            dim,
            intermediate_dim,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.proj(x).view(-1, self.hidden_size)
        return x


class Glm4vPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = d_model
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )
        self.post_projection_norm = LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(
                0, hidden_size, device=device, dtype=pos_embed_weight.dtype
            )
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(
                    image_shapes, device=device, dtype=torch.long
                )

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat(
                [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)
            target_w = torch.cat(
                [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
            ).to(device=device, dtype=torch.float32)

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d,
                grid,
                mode="bicubic",
                align_corners=False,
                padding_mode="border",
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = (
                interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            )
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
                embeddings.device
            )

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionRotaryEmbedding(nn.Module):
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
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0,
                        self.dim,
                        2,
                        dtype=torch.float,
                        device=self.inv_freq.device,
                    )
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Glm4vVisionModel(nn.Module):
    def __init__(
        self,
        vision_config: Glm4vVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.use_data_parallel = use_data_parallel

        self.patch_embed = Glm4vVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Glm4vVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=self.out_hidden_size,
                    num_heads=self.num_heads,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    rms_norm_eps=vision_config.rms_norm_eps,
                    attn_qkv_bias=vision_config.attention_bias,
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(depth)
            ]
        )

        self.merger = Glm4vPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        self.embeddings = Glm4vVisionEmbeddings(vision_config)

        self.post_conv_layernorm = Glm4vRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
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
        return rotary_pos_emb, pos_ids

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        x = self.post_conv_layernorm(x)

        # compute position embedding
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        x = self.embeddings(
            x, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
        )

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb_tuple = (emb.cos(), emb.sin())

        # x.shape: (s, b, d) where b=1 for vision processing
        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=rotary_pos_emb_tuple)

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x


class Glm4vForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Glm4vConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.visual = Glm4vVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        vision_utils.update_vit_attn_dummy_heads_config(self.config)

        self.model = Glm4Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in GLM-V, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in GLM-V, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)

        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = (
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            )
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                flattened_video_grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            video_embeds = self.visual(pixel_values, grid_thw=flattened_video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for GLM-4.1V.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for GLM-4.1V
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

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):
        """pad attn qkv weights for dummy heads"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads
        if num_dummy_heads == 0:
            return loaded_weight
        head_dim = self.config.vision_config.head_dim

        if "attn.qkv_proj" in name:
            wq, wk, wv = loaded_weight.chunk(3, dim=0)
            if name.endswith(".weight"):
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]
            elif name.endswith(".bias"):
                dummy_shape = [num_dummy_heads, head_dim]
            else:
                raise RuntimeError(f"Unsupported weight with name={name}")
            pad_func = lambda x: torch.cat(
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
            ).flatten(0, 1)
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
            loaded_weight = torch.cat([wq, wk, wv], dim=0)
        elif "attn.proj.weight" in name:
            padded_weight = loaded_weight.new_zeros(
                loaded_weight.shape[0], head_dim * num_dummy_heads
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
        elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
        return loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # For the PP case, we add special handling for lm_head.weight,
        # - On non–last ranks: we continue, because this stage is supposed to
        #   be just an empty PPMissingLayer shell.
        # - On the last rank: params_dict is expected to contain lm_head.weight,
        #   so it will never hit the branch "if name not in params_dict".
        #
        # For all other parameters, such like
        # "model.visual.blocks.20.mlp.gate_proj.weight", the unified rule is:
        # If this name does not exist in the current rank’s params_dict,
        # it does not belong to this pipeline stage, thus we simply continue.
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")
            if name.startswith("lm_head.") and not self.pp_group.is_last_rank:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if "visual" in name:
                    loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                        self.config, name, loaded_weight
                    )
                weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [Glm4vForConditionalGeneration]
