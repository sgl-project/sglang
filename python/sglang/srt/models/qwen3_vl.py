# Copyright 2025 Qwen Team
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
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""
import logging
import math
import re
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.utils import (
    RotaryPosMixin,
    WeightsMapper,
    compute_cu_seqlens_from_grid_numpy,
)
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_int_env_var, is_npu
from sglang.srt.utils.hf_transformers_utils import get_processor

import nvtx  # wili
import os  # wili
import numpy as np  # wili
from vfly.utils.parallel import dit_sp_split, dit_sp_gather  # wili

logger = logging.getLogger(__name__)


# === Vision Encoder === #


class Qwen3_VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        if bool(int(os.environ.get('ENABLE_VFLY', '0'))):  # wili, for vfly
            self.tp_size = 1  # wili, reuse TP group but keep TP size as 1
            self.tp_rank = 0  # wili
        else:  # wili, original code
            self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
            self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output


# wili, original code of class Qwen3VLVisionPatchEmbed
original_Qwen3VLVisionPatchEmbed = """
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states
"""


# wili, improved version of class Qwen3VLVisionPatchEmbed
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )
        k = self.in_channels * self.temporal_patch_size * self.patch_size ** 2
        self.linear = nn.Linear(in_features=k, out_features=self.embed_dim, bias=True, dtype=self.proj.weight.dtype)
        
    def copy_conv3d_weight_to_linear(self):
        # Call this after model loading in `sglang/srt/model_loader/loader.py: load_weights_and_postprocess()`
        print("Copy weights from Conv3d to Linear in PatchEmbed")
        with torch.no_grad():
            self.linear.weight.copy_(self.proj.weight.view(self.embed_dim, -1))
            self.linear.bias.copy_(self.proj.bias)
        del self.proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act="silu",
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_ws: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_ws=output_ws,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x += attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x += mlp
        return x


class Qwen3VLMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        if bool(int(os.environ.get('ENABLE_VFLY', '0'))):  # wili, for vfly
            self.tp_size = 1  # wili, reuse TP group but keep TP size as 1
            self.tp_rank = 0  # wili
        else:  # wili, original code
            self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
            self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3VLMoeVisionModel(nn.Module, RotaryPosMixin):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.num_grid = self.num_grid_per_side * self.num_grid_per_side
        self.align_corners = (
            get_global_server_args().enable_precise_embedding_interpolation
        )
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.use_data_parallel = use_data_parallel
        # layer indexes of which layer's output should be deep-stacked
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        if self.pp_group.is_first_rank:
            self.pos_embed = VocabParallelEmbedding(
                self.num_position_embeddings,
                self.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("pos_embed", prefix),
            )
        else:
            self.pos_embed = PPMissingLayer()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )
        self.enable_vfly = bool(int(os.environ.get('ENABLE_VFLY', '0')))  # wili

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = Qwen3VLMoeVisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    dim=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"deepstack_merger_list.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        if bool(int(os.environ.get('ENABLE_VFLY', '0'))):  # wili, for vfly
            self.tp_size = 1
        else:  # wili, original code, but seems useless?
            self.tp_size = (
                1 if use_data_parallel else get_tensor_model_parallel_world_size()
            )
        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = ViTCudaGraphRunner(self)

    @property
    def dtype(self) -> torch.dtype:
        # return self.patch_embed.proj.weight.dtype  # wili, Conv3d -> Linear
        return self.patch_embed.linear.weight.dtype  # wili, Conv3d -> Linear

    @property
    def device(self) -> torch.device:
        # return self.patch_embed.proj.weight.device  # wili, Conv3d -> Linear
        return self.patch_embed.linear.weight.device  # wili, Conv3d -> Linear

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def rot_pos_emb_v2(self, grid_thw):  # wili, TODO: align logic with original code
        """
        grid_thw: LongTensor on CPU / GPU, shape [N, 3], value (t,h,w) per row
        return  : bfloat16 tensor on GPU, shape [Σ(t*h*w), 2 * 18]
        """
        device = grid_thw.device
        m = self.spatial_merge_size
        
        pos_ids_list = []
        
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            
            hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            
            wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            
            sample_pos = torch.stack([hpos_ids, wpos_ids], dim=-1)  # [h*w, 2]
            pos_ids_list.append(sample_pos.repeat(t, 1))  # [t*h*w, 2]
        
        pos_ids = torch.cat(pos_ids_list, dim=0)
        
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def _get_interpolation_indices(self, dim_size: int) -> torch.Tensor:
        """
        Compute continuous interpolation indices for a single dimension.

        Returns continuous indices.
        """
        if self.align_corners:
            indices = np.linspace(
                0, self.num_grid_per_side - 1, dim_size, dtype=np.float32
            )
        else:
            indices = (np.arange(dim_size, dtype=np.float32) + 0.5) * (
                self.num_grid_per_side / dim_size
            ) - 0.5
            indices = np.clip(indices, 0, self.num_grid_per_side - 1)
        return indices

    def _calculate_indices_and_weights(self, h_idxs, w_idxs):
        """
        Compute bilinear interpolation indices and weights.

        Returns tuple of (indices, weights), each as 4 numpy arrays for the 4 corner points.
        """
        h_f = np.floor(h_idxs).astype(np.int64)
        h_c = np.clip(h_f + 1, 0, self.num_grid_per_side - 1)
        dh = h_idxs - h_f

        w_f = np.floor(w_idxs).astype(np.int64)
        w_c = np.clip(w_f + 1, 0, self.num_grid_per_side - 1)
        dw = w_idxs - w_f

        side = self.num_grid_per_side

        indices = [
            (h_f[:, None] * side + w_f).flatten(),
            (h_f[:, None] * side + w_c).flatten(),
            (h_c[:, None] * side + w_f).flatten(),
            (h_c[:, None] * side + w_c).flatten(),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)).flatten(),
            ((1 - dh)[:, None] * dw).flatten(),
            (dh[:, None] * (1 - dw)).flatten(),
            (dh[:, None] * dw).flatten(),
        ]
        return indices, weights

    def _get_position_embedding(self, patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        """
        Tile and reorganize position embeddings to align with the token sequence.
        """
        result_parts = []
        merge_size = self.spatial_merge_size

        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)

            h_merge = h // merge_size
            w_merge = w // merge_size

            pos_embed = (
                pos_embed.view(t, h_merge, merge_size, w_merge, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )

            result_parts.append(pos_embed)

        return torch.cat(result_parts, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw):
        """Interpolate position embeddings for (batch, 3) size input dimensions.

        Performs bilinear interpolation on spatial dimensions (height, width) and replicates
        along temporal dimension. The result is reorganized according to spatial_merge_size.

        Args:
            grid_thw: Tensor of shape [batch_size, 3] with (temporal, height, width) dimensions
                     in patches for each sample.

        Returns:
            Interpolated position embeddings tensor.
        """
        grid_thw_cpu = grid_thw.cpu().numpy()

        # transfer data to CPU before loop
        temporal_dims = grid_thw_cpu[:, 0].tolist()
        height_dims = grid_thw_cpu[:, 1].tolist()
        width_dims = grid_thw_cpu[:, 2].tolist()

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        patches_size = [h * w for h, w in zip(height_dims, width_dims)]
        total_patches = sum(patches_size)
        all_indices_np = np.zeros((4, total_patches), dtype=np.int64)
        all_weights_np = np.zeros((4, total_patches), dtype=np.float32)

        current_idx = 0

        # calculate indices and weights on CPU
        for t, h, w in zip(temporal_dims, height_dims, width_dims):
            h_idxs = self._get_interpolation_indices(h)
            w_idxs = self._get_interpolation_indices(w)

            indices, weights = self._calculate_indices_and_weights(h_idxs, w_idxs)

            end_idx = current_idx + h * w
            for i in range(4):
                all_indices_np[i, current_idx:end_idx] = indices[i]
                all_weights_np[i, current_idx:end_idx] = weights[i]
            current_idx = end_idx

        idx_tensor = torch.from_numpy(all_indices_np).to(device)
        weight_tensor = torch.from_numpy(all_weights_np).to(dtype=dtype, device=device)

        # calculate interpolation
        pos_embeds = self.pos_embed(idx_tensor.view(-1))
        pos_embeds = pos_embeds.view(4, total_patches, -1)
        patch_pos_embeds = (pos_embeds * weight_tensor.unsqueeze(-1)).sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split(patches_size)
        return self._get_position_embedding(
            patch_pos_embeds, temporal_dims, height_dims, width_dims
        )


    def fast_pos_embed_interpolate_v2(self, grid_thw: torch.Tensor):  # wili, TODO: align logic with original code
        """
        grid_thw: LongTensor on CPU / GPU, shape [N, 3], value (t,h,w) per row
        return  : bfloat16 tensor on GPU, shape [Σ(t*h*w), self.pos_embed.embedding_dim]
        """
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        grid_thw = grid_thw.to(device, non_blocking=True)
        num_grid = int(self.num_position_embeddings ** 0.5)
        m_size = self.spatial_merge_size
        embedding_dim = self.pos_embed.embedding_dim

        num_patch_per_clip = grid_thw.prod(dim=1)  # [t_i * h_i * w_i for i in range len(grid_thw)]
        num_patch_quad = num_patch_per_clip * 4  # 4 indice / weights per patch
        num_elements = int(num_patch_per_clip.sum())  # number of total patches, on CPU

        offset = torch.cat([torch.tensor([0], dtype=torch.long, device=device), num_patch_per_clip.cumsum(0)])
        offset_quad = offset * 4

        idx_all = torch.empty(num_patch_quad.sum(), dtype=torch.long, device=device)
        wgt_all = torch.empty(num_patch_quad.sum(), dtype=dtype, device=device)

        for st, ed, (t, h, w) in zip(offset_quad[:-1], offset_quad[1:], grid_thw):
            h_idx = torch.linspace(0, num_grid - 1, h, device=device)
            w_idx = torch.linspace(0, num_grid - 1, w, device=device)

            h_floor = h_idx.floor().long()
            w_floor = w_idx.floor().long()
            h_ceil = (h_floor + 1).clamp_max(num_grid - 1)
            w_ceil = (w_floor + 1).clamp_max(num_grid - 1)

            hf, wf = torch.meshgrid(h_floor, w_floor, indexing='ij')
            hc, wf = torch.meshgrid(h_ceil, w_floor, indexing='ij')
            hf, wc = torch.meshgrid(h_floor, w_ceil, indexing='ij')
            hc, wc = torch.meshgrid(h_ceil, w_ceil, indexing='ij')
            idx4 = torch.stack([hf * num_grid + wf, hf * num_grid + wc, hc * num_grid + wf, hc * num_grid + wc], dim=-1)

            dh = (h_idx - h_floor.float()).view(-1, 1)
            dw = (w_idx - w_floor.float()).view(1, -1)
            w4 = torch.stack([(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw], dim=-1)

            idx_all[st:ed] = idx4.flatten().repeat_interleave(t)
            wgt_all[st:ed] = w4.flatten().repeat_interleave(t)

        patch_pos_embed = self.pos_embed(idx_all) * wgt_all.unsqueeze(1)
        patch_pos_embed = patch_pos_embed.view(-1, 4, embedding_dim).sum(dim=1)

        out = torch.empty([num_elements, embedding_dim], dtype=dtype, device=device)
        for st, ed, (t, h, w) in zip(offset[:-1], offset[1:], grid_thw):
            emb = patch_pos_embed[st:ed]
            emb = emb.view(t, h // m_size, m_size, w // m_size, m_size, embedding_dim)
            emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0, 4)
            out[st:ed] = emb

        return out

    def fast_pos_embed_interpolate_v3(self, grid_thw: torch.Tensor):  # wili, TODO: align logic with 0.5.7
        """
        grid_thw: LongTensor on CPU / GPU, shape [N, 3], value (t,h,w) per row
        return  : bfloat16 tensor on GPU, shape [Σ(t*h*w), self.pos_embed.embedding_dim]
        """
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        grid_thw_cpu = grid_thw.detach().cpu().numpy()
        num_grid = int(self.num_position_embeddings ** 0.5)
        m_size = self.spatial_merge_size
        embedding_dim = self.pos_embed.embedding_dim

        num_patch_per_clip = [int(t * h * w) for t, h, w in grid_thw_cpu]
        total_patches = sum(num_patch_per_clip)
        
        idx_all = np.empty((total_patches, 4), dtype=np.int64)
        wgt_all = np.empty((total_patches, 4), dtype=np.float32)
        
        offset = 0
        for t, h, w in grid_thw_cpu:
            h_idx = np.linspace(0, num_grid - 1, h)
            w_idx = np.linspace(0, num_grid - 1, w)
            h_floor = np.floor(h_idx).astype(int)
            w_floor = np.floor(w_idx).astype(int)
            h_ceil = np.clip(h_floor + 1, 0, num_grid - 1)
            w_ceil = np.clip(w_floor + 1, 0, num_grid - 1)

            hf, wf = np.meshgrid(h_floor, w_floor, indexing='ij')
            hc, wf2 = np.meshgrid(h_ceil, w_floor, indexing='ij')
            hf2, wc = np.meshgrid(h_floor, w_ceil, indexing='ij')
            hc2, wc2 = np.meshgrid(h_ceil, w_ceil, indexing='ij')
            idx4 = np.stack([
                hf * num_grid + wf,
                hf2 * num_grid + wc,
                hc * num_grid + wf2,
                hc2 * num_grid + wc2
            ], axis=-1)  # [h, w, 4]

            dh = (h_idx - h_floor).reshape(-1, 1)
            dw = (w_idx - w_floor).reshape(1, -1)
            w4 = np.stack([
                (1 - dh) * (1 - dw),
                (1 - dh) * dw,
                dh * (1 - dw),
                dh * dw
            ], axis=-1)  # [h, w, 4]

            idx4 = np.tile(idx4, (t, 1, 1, 1))  # [t, h, w, 4]
            w4 = np.tile(w4, (t, 1, 1, 1))      # [t, h, w, 4]
            
            patch_count = t * h * w
            idx_all[offset:offset+patch_count] = idx4.reshape(-1, 4)
            wgt_all[offset:offset+patch_count] = w4.reshape(-1, 4)
            offset += patch_count

        idx_all = torch.from_numpy(idx_all.reshape(-1)).to(device)
        wgt_all = torch.from_numpy(wgt_all.reshape(-1)).to(device, dtype=dtype)

        patch_pos_embed = self.pos_embed(idx_all) * wgt_all.unsqueeze(1)
        patch_pos_embed = patch_pos_embed.view(-1, 4, embedding_dim).sum(dim=1)

        offset_cumsum = np.cumsum([0] + num_patch_per_clip)
        out = torch.empty([total_patches, embedding_dim], dtype=dtype, device=device)

        # PErmute indices rather than values
        all_indices = np.empty(total_patches, dtype=np.int32)
        for i, (st, ed, (t, h, w)) in enumerate(zip(offset_cumsum[:-1], offset_cumsum[1:], grid_thw_cpu)):
            base_idx = np.arange(st, ed).reshape(t, h, w)
            base_idx = base_idx.reshape(t, h // m_size, m_size, w // m_size, m_size)
            base_idx = base_idx.transpose(0, 1, 3, 2, 4)
            base_idx = base_idx.reshape(-1)
            all_indices[st:ed] = base_idx
        
        all_indices = torch.from_numpy(all_indices)
        out[:] = patch_pos_embed[all_indices]
        
        return out

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            return self.forward_with_cuda_graph(x, grid_thw)

        x = x.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)  # wili, TODO: align logic with original code
        with nvtx.annotate("self.patch_embed(x)", color="yellow"):  # wili
            x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        with nvtx.annotate("self.fast_pos_embed_interpolate(x)", color="yellow"):  # wili
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)  # wili
        # with nvtx.annotate("self.fast_pos_embed_interpolate_v2(x)", color="yellow"):  # wili
        #     pos_embeds = self.fast_pos_embed_interpolate_v2(grid_thw)  # wili
        # with nvtx.annotate("self.fast_pos_embed_interpolate_v3(x)", color="yellow"):  # wili
        #     pos_embeds = self.fast_pos_embed_interpolate_v3(grid_thw)  # wili, TODO: align logic with original code

        x += pos_embeds

        with nvtx.annotate("self.rot_pos_emb(x)", color="yellow"):  # wili
            rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)  # wili, original code
        # with nvtx.annotate("self.rot_pos_emb_v2(x)", color="yellow"):  # wili
        #     rotary_pos_emb = self.rot_pos_emb_v2(grid_thw)  # wili, TODO: align logic with original code

        # compute cu_seqlens
        with nvtx.annotate("compute cu_seqlens(x)", color="yellow"):  # wili
            seq_lens_list = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])  # wili
            cu_seqlens = seq_lens_list.cumsum(dim=0)
            cu_seqlens = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
                cu_seqlens.to(torch.int32),
            ])
        # wili, original code
        # cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        # # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        # if not is_npu():
        #     cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        # else:
        #     cu_seqlens = cu_seqlens.to("cpu")

        x = x.unsqueeze(1)  # wili, [seq_len, batch_size=1, hidden_size], transpose to [batch_size, seq_len, hidden_size] in Qwen3_VisionBlock just before attention
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        # print("[wili] ==== shape before blocks ====================================")
        # print(f"{x.shape = }")
        # print(f"{rotary_pos_emb_cos.shape = }")
        # print("[wili] ==== shape before blocks ====================================")

        if self.enable_vfly:  # wili
            # wili, pad sequence length to be multiple of 32
            # For example we use a picture of 1608x828 as input,
            # It is resize (by `smart_resize()` in library transformers) to 1600x832 with aligned 32
            # Then it is merged pixels (by `_preprocess()` in library transformers) to sequence length of 1600/16*832/16=5200 with 16x16 per a block
            # So the the sequence length here (x.shape[1]) is 5200
            # Using cp=8, the sequence length per worker will be 5200 / 8 = 650,
            # Noticing this line (Qwen3VLMoeVisionPatchMerger.forward, around Line 278 in this file):
            #     x = self.norm(x.view(-1, self.hidden_size))  # here x.shape == [650, 1152], self.hidden_size == 4608
            # So 650 / 4 = 162.5, leads to a error
            # As a workaround, we pad the sequence 5200 to 5216 here with aligned 32,
            # So 5216 / 8 / 4 = 163, OK for the PatchMerger.
            seq_len = x.shape[0]
            pad_length = int((seq_len + 31) / 32) * 32 - seq_len
            pad_size = [x.size(i) for i in range(x.ndim)]
            pad_size[0] = pad_length
            x = torch.cat([x, x.new_zeros(*pad_size)], dim=0).contiguous()
            pad_size = [rotary_pos_emb_cos.size(i) for i in range(rotary_pos_emb_cos.ndim)]
            pad_size[0] = pad_length
            rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos.new_zeros(*pad_size)], dim=0).contiguous()
            rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin.new_zeros(*pad_size)], dim=0).contiguous()

            x = dit_sp_split(x, dim=0)  # wili, split sequence parts for distributed processing
            rotary_pos_emb_cos = dit_sp_split(rotary_pos_emb_cos, dim=0)  # wili
            rotary_pos_emb_sin = dit_sp_split(rotary_pos_emb_sin, dim=0)  # wili

            # print("[wili] ==== shape after dit_sp_split ====================================")
            # print(f"{x.shape = }")
            # print(f"{rotary_pos_emb_cos.shape = }")
            # print("[wili] ==== shape after dit_sp_split ====================================")

        deepstack_feature_lists = []
        num_deepstack_captured = 0

        for layer_num, blk in enumerate(self.blocks):
            with nvtx.annotate("blk", color="yellow"):  # wili
                x = blk(
                    x,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb_cos=rotary_pos_emb_cos,
                    rotary_pos_emb_sin=rotary_pos_emb_sin,
                )

                if layer_num in self.deepstack_visual_indexes:
                    deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](
                        x
                    )
                    deepstack_feature_lists.append(deepstack_feature)
                    num_deepstack_captured += 1

        with nvtx.annotate("x = self.merger(x)", color="yellow"):  # wili
            x = self.merger(x)
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        if self.enable_vfly:  # wili
            hidden_states = hidden_states.unsqueeze(0)  # wili, [batch_size=1, seq_len, hidden_size * (1 + depth_of_deepstack)]
            # print("[wili] ==== shape before dit_sp_gather ====================================")
            # print(f"{hidden_states.shape = }")
            # print("[wili] ==== shape before dit_sp_gather ====================================")

            hidden_states = dit_sp_gather(hidden_states, dim=1)  # wili, gather sequence parts back
            
            hidden_states = hidden_states.squeeze(0)

        # print("[wili] ==== shape after blocks ====================================")
        # print(f"{hidden_states.shape = }")
        # print("[wili] ==== shape after blocks ====================================")

        return hidden_states

    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds

        # rotary embedding -> (cos, sin)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # compute cu_seqlens
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        if not isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = torch.tensor(cu_seqlens, device=x.device, dtype=torch.int32)
        else:
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        cu_seqlens = cu_seqlens.contiguous()

        # blocks + merger + deepstack(optional) via CUDA Graph Runner
        return self.cuda_graph_runner.run(
            x=x,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=None,
            output_indices=None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
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


cached_get_processor = lru_cache(get_processor)


class Qwen3LLMModel(Qwen3Model):

    def __init__(
        self,
        *,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        if not self.pp_group.is_first_rank:
            assert self.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), "start_layer should be greater than or equal to len(deepstack_visual_indexes)"

        self.hidden_size = config.hidden_size
        self.deepstack_embed_to_decoder_layer = range(
            len(config.vision_config.deepstack_visual_indexes)
        )

    def get_deepstack_embeds(
        self, layer_idx: int, input_deepstack_embeds: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Get deepstack embeddings for a given layer index, or None if not applicable."""
        if (
            input_deepstack_embeds is None
            or layer_idx not in self.deepstack_embed_to_decoder_layer
        ):
            return None
        sep = self.hidden_size * layer_idx
        return input_deepstack_embeds[:, sep : sep + self.hidden_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer
            if layer_idx in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )

            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.
            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549
            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack
            # The order matters because addition with different tensors is not associative in practice.
            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.
            deepstack_embeds = self.get_deepstack_embeds(
                layer_idx - 1, input_deepstack_embeds
            )
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                post_residual_addition=deepstack_embeds,
            )

        # Handle deepstack for the last processed layer if it exists.
        last_deepstack = self.get_deepstack_embeds(
            self.end_layer - 1, input_deepstack_embeds
        )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(
                        hidden_states, residual, post_residual_addition=last_deepstack
                    )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen3VLForConditionalGeneration(nn.Module):
    # To ensure correct weight loading and mapping.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3LLMModel,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()

        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.visual = Qwen3VLMoeVisionModel(
            config.vision_config,
            # NOTE: Qwen3-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        # TODO: make it more elegant
        if language_model_cls is Qwen3LLMModel:
            self.config: Qwen3VLConfig = config  # for qwen3-vl
        else:
            self.config = config.text_config  # for qwen3-omni
            self.config.encoder_only = getattr(config, "encoder_only", False)
            self.config.language_only = getattr(config, "language_only", False)

        if not hasattr(config, "encoder_only") or not config.encoder_only:
            self.model = language_model_cls(
                config=self.config,
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
                        use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                self.lm_head = PPMissingLayer()
        else:
            # encoder_only mode: no language model, so no lm_head needed
            self.lm_head = None

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(self.config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # like {8:0, 16:1, 24:2}, which stands for the captured deepstack features on
        # 8, 16, 24 layer will be merged to 0, 1, 2 layer of decoder output hidden_states

        # deepstack
        self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        self.use_deepstack = {Modality.IMAGE: True, Modality.VIDEO: True}

    def separate_deepstack_embeds(self, embedding):
        assert (
            embedding.shape[-1] % (1 + self.num_deepstack_embeddings) == 0
        ), f"hidden_state of {embedding.shape} should be divisible by ({1 + self.num_deepstack_embeddings})"

        separate_index = self.config.hidden_size
        input_embeds = embedding[:, :separate_index]
        input_deepstack_embeds = embedding[:, separate_index:]
        return input_embeds, input_deepstack_embeds

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        max_patches_per_call = get_int_env_var("SGLANG_VLM_MAX_PATCHES_PER_VIT", 0)
        max_images_per_call = get_int_env_var("SGLANG_VLM_MAX_IMAGES_PER_VIT", 0)

        if max_patches_per_call == 0 and max_images_per_call == 0:
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values,
                    image_grid_thw.tolist(),
                    rope_type="rope_3d",
                )
            else:
                return self.visual(pixel_values, grid_thw=image_grid_thw)

        # compute the number of patches per image and the slice positions in pixel_values
        grid_thw_list = (
            image_grid_thw.tolist()
        )  # List[List[int]], each is [T, H, W] or similar
        patches_per_image = [int(math.prod(g)) for g in grid_thw_list]
        num_images = len(patches_per_image)

        # cumulative sum used to slice pixel_values along the image dimension
        cum_patches = [0]
        for p in patches_per_image:
            cum_patches.append(cum_patches[-1] + p)
        total_patches = cum_patches[-1]

        assert pixel_values.size(0) == total_patches, (
            f"pixel_values rows ({pixel_values.size(0)}) "
            f"!= total patches ({total_patches})"
        )

        # split into chunks in image order, each chunk obeys the patch/image limits
        all_chunk_embeds: List[torch.Tensor] = []
        img_start = 0

        # start = torch.cuda.Event(enable_timing=True)  # wili
        # end   = torch.cuda.Event(enable_timing=True)  # wili
        # torch.cuda.synchronize()  # wili
        # start.record()  # wili
        while img_start < num_images:
            img_end = img_start
            patches_in_chunk = 0
            images_in_chunk = 0

            # try to pack more images into the current chunk until some limit would be exceeded
            while img_end < num_images:
                next_patches = patches_per_image[img_end]

                # if adding this image would exceed the patch limit, stop
                if (
                    max_patches_per_call > 0
                    and patches_in_chunk + next_patches > max_patches_per_call
                ):
                    break

                # if adding this image would exceed the image-count limit, also stop
                if (
                    max_images_per_call > 0
                    and images_in_chunk + 1 > max_images_per_call
                ):
                    break

                patches_in_chunk += next_patches
                images_in_chunk += 1
                img_end += 1

            # extreme case: the first image alone exceeds the patch limit -> at least ensure img_end > img_start
            if img_end == img_start:
                img_end = img_start + 1
                patches_in_chunk = patches_per_image[img_start]
                images_in_chunk = 1

            # slice pixel_values and grid_thw according to [img_start:img_end]
            patch_start = cum_patches[img_start]
            patch_end = cum_patches[img_end]
            pixel_chunk = pixel_values[patch_start:patch_end]
            grid_chunk = image_grid_thw[img_start:img_end]

            # run ViT once on this chunk without extra padding
            with nvtx.annotate("VisionModel", color="green"):  # wili
                if self.use_data_parallel:
                    chunk_embeds = run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_chunk,
                        grid_chunk.tolist(),
                        rope_type="rope_3d",
                    )
                else:
                    chunk_embeds = self.visual(pixel_chunk, grid_thw=grid_chunk)

            # chunk_embeds: (sum_patches_after_merge_this_chunk, hidden)
            all_chunk_embeds.append(chunk_embeds)

            # next batch
            img_start = img_end

        # end.record()  # wili
        # torch.cuda.synchronize()  # wili
        # elapsed_ms = start.elapsed_time(end)  # wili
        # print(f"VisionModel,{elapsed_ms=:.3f}ms")  # wili

        # concatenate back the full image embedding sequence
        with nvtx.annotate("Concat after Vision Model", color="green"):  # wili
            res = torch.cat(all_chunk_embeds, dim=0)
        return res

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        for item in items:
            item.feature = item.feature.to(self.visual.device)
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        # Memory optimization for item.feature:
        # 1. item.feature is released when request finished
        # 2. High concurrency may cause device OOM due to delayed release
        # 3. Fix: Offload item.feature to CPU, move to device only when needed
        for item in items:
            item.feature = item.feature.to("cpu")
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Qwen3-VL.

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

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            layer_id = get_layer_id(name)

            # Only copy embed_tokens to lm_head when tie_word_embeddings=True
            # For models with tie_word_embeddings=False (e.g. 8B), lm_head has independent weights
            if (
                self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and self.config.tie_word_embeddings
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            is_visual = "visual" in name
            if (
                not is_visual
                and layer_id is not None
                and hasattr(self, "model")
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading visual/language model weights
                if (
                    self.config.encoder_only or self.config.language_only
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                    else:
                        continue

                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def enable_vision_fly(self):  # wili
        # Copy from /work/qwen3-vl/third_party/VisionFly/examples/qwenimage.py
        from .common import BaseArgumentParser, create_vfly_config, validate_parallel_config
        from vfly import setup_configs
        from vfly.layers import VflyAttnProcessor, apply_vfly_linear, apply_vfly_norm
        from vfly.utils import get_logger
        from vfly.configs.parallel import get_dit_parallel_config, DiTParallelConfig
        from vfly.configs.pipeline import PipelineConfig

        logger = get_logger(__name__)

        class VflyQwenDoubleStreamAttnProcessor2_0(VflyAttnProcessor):

            def __init__(self):
                super().__init__()
                logger.debug("VflyQwenDoubleStreamAttnProcessor2_0 initialized")

            def __call__(
                self,
                q: torch.FloatTensor,
                k: torch.FloatTensor,
                v: torch.FloatTensor,
                cu_seqlens: torch.IntTensor,
            ) -> torch.FloatTensor:
                pfg = get_dit_parallel_config()
                world_size = pfg.ulysses_size()  # Only ulysses is supported

                seq_lens_list = cu_seqlens.diff()
                max_seqlen = torch.max(seq_lens_list)
                total_seq_len = cu_seqlens[-1].item()
                seq_len_padded = (total_seq_len + world_size - 1) // world_size * world_size
                uneven_number = seq_len_padded - total_seq_len
                seq_len_cur_rank = q.shape[1]
                if torch.distributed.get_rank() == world_size - 1:
                    seq_len_cur_rank = seq_len_cur_rank - uneven_number

                parallel_config = DiTParallelConfig()
                parallel_config.set_config(
                    cfg_size=pfg.cp_size(),
                    ulysses_size=pfg.ulysses_size(),
                    ring_size=pfg.ring_size(),
                )
                PipelineConfig.set_uneven_cp_config(total_seq_len, seq_len_padded, seq_len_cur_rank, parallel_config)

                return self.vfly_attn(
                    q,
                    k,
                    v,
                    tensor_layout="NHD",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens.clone(),
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                )

        # Setup argument parser
        parser = BaseArgumentParser("")
        parser.set_defaults(ulysses=torch.distributed.get_world_size(), attn_type="flash-attn3")
        args = parser.parse_args([])
        """
        enable_autotuner = False
        if args.linear_type == "auto" or args.attn_type == "auto":
            enable_autotuner = True
            if not args.disable_torch_compile:
                logger.warning("Disable torch compile when using autotuner")
                args.disable_torch_compile = True
            if args.enable_vfly_cpu_offload:
                logger.warning("Disable vfly cpu offload when using autotuner")
                args.enable_vfly_cpu_offload = False
        """
        # Validate configuration
        validate_parallel_config(args)

        # Load pipeline
        vfly_configs = create_vfly_config(args)
        setup_configs(**vfly_configs)

        pipe = self.visual
        for name, module in pipe.blocks.named_modules():
            if isinstance(module, VisionAttention):
                attn_processor = VflyQwenDoubleStreamAttnProcessor2_0()
                attn_processor.name = name
                module.processor = attn_processor
        apply_vfly_linear(pipe, load_parameters=True)
        apply_vfly_norm(
            pipe,
            rmsnorm=["norm_q", "norm_k", "norm_added_q", "norm_added_k"],
            load_parameters=True,
        )
        """
        if not args.disable_torch_compile:
            self.visual = torch.compile(self.visual, mode=args.torch_compile_mode)
        """
        return

EntryClass = Qwen3VLForConditionalGeneration
