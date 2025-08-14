# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
import logging
from functools import lru_cache, partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import utils as dist_utils
from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.utils import add_prefix, is_npu

_is_npu = is_npu()

if _is_npu:
    import torch_npu

MIN_PAD_SIZE = 64
MAX_PAD_SIZE = 128

logger = logging.getLogger(__name__)


class Qwen2_5_VLMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel_gate, _ = self.gate_proj(x)
        x_parallel_gate = self.act(x_parallel_gate)
        x_parallel_up, _ = self.up_proj(x)
        x_parallel = x_parallel_gate * x_parallel_up
        x, _ = self.down_proj(x_parallel)
        return x


class AscendQwen2_5_VisionAttention(VisionAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            use_qkv_parallel,
            qkv_backend,
            quant_config,
            dropout,
            softmax_in_single_precision,
            flatten_batch,
            prefix,
            proj_bias,
            num_dummy_heads,
            qkv_bias,
            qk_normalization,
            layer_norm_eps,
            customized_position_embedding_applier,
            **kwargs,
        )
        self.origin_hidden_size_per_attention_head = self.hidden_size_per_attention_head
        self.pad_hidden_size_per_attention_head = 0
        if (
            self.hidden_size_per_attention_head > MIN_PAD_SIZE
            and self.hidden_size_per_attention_head < MAX_PAD_SIZE
        ):
            self.pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head
            )
            self.hidden_size_per_attention_head = MAX_PAD_SIZE
        self.q_size = self.num_attention_heads_per_partition * (
            self.head_size + self.pad_hidden_size_per_attention_head
        )
        self.kv_size = self.num_attention_kv_heads_per_partition * (
            self.head_size + self.pad_hidden_size_per_attention_head
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        bsz, s, _ = x.shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        # [b, s, embed_dim] --> [b, s, embed_dim]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [b, s, embed_dim] --> [b, s, head, head_size]
        q = q.reshape(bsz, s, head, -1).contiguous()
        k = k.reshape(bsz, s, kv_head, -1).contiguous()
        v = v.reshape(bsz, s, kv_head, -1).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = torch_npu.npu_rotary_mul(q, cos, sin)
            k = torch_npu.npu_rotary_mul(k, cos, sin)

        # [b, s, head, head_size] --> [b * s, head, head_size]
        q, k, v = [rearrange(x, "b s ... -> (b s) ...").contiguous() for x in (q, k, v)]

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        output = torch.empty_like(q)
        # operator requires pta version >= 2.5.1
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.origin_hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            out=output,
        )

        assert output.dim() == 3, output.shape

        # [b * s, h, head_size] --> [b, s, h * head_size]
        output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

        # [b, s, h * head_size] --> [b, s, h * head_size]
        output, _ = self.proj(output)

        return output


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.norm2 = RMSNorm(dim, eps=1e-6)

        if attn_implementation is None:
            softmax_in_single_precision = False
            qkv_backend = None
            flatten_batch = True
        elif attn_implementation == "sdpa":
            softmax_in_single_precision = False
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_2":
            softmax_in_single_precision = False
            qkv_backend = "triton_attn"
            flatten_batch = True
        elif attn_implementation == "eager":
            softmax_in_single_precision = True
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_3":
            softmax_in_single_precision = False
            qkv_backend = "fa3"
            flatten_batch = True

        if _is_npu:
            vision_attention_class = AscendQwen2_5_VisionAttention
        else:
            vision_attention_class = VisionAttention
        self.attn = vision_attention_class(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=flatten_batch,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
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
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x = x + mlp
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
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
                    dim,
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


class AscendQwen2_5_VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.matmul(
            self.proj.weight.data.view(self.embed_dim, -1).transpose(0, 1)
        )
        return hidden_states


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = vision_config.intermediate_size
        if _is_npu:
            self.patch_embed = AscendQwen2_5_VisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                embed_dim=hidden_size,
            )
        else:
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                embed_dim=hidden_size,
            )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.gate_proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

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

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=grid_thw.device),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).cumsum(dim=0),
            ]
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)

        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x


class AscendQwen2_5_VisionTransformer(Qwen2_5_VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)
        self.interleaved = interleaved
        self.enable_pad = False
        self.hidden_size: int = vision_config.hidden_size
        num_heads: int = vision_config.num_heads
        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, num_heads
        )

        if (
            self.hidden_size_per_attention_head > MIN_PAD_SIZE
            and self.hidden_size_per_attention_head < MAX_PAD_SIZE
        ):
            self.enable_pad = True
            self.origin_hidden_size_per_attention_head = (
                self.hidden_size_per_attention_head
            )
            self.half_origin_hidden_size_per_attention_head = (
                self.hidden_size_per_attention_head // 2
            )
            self.half_pad_hidden_size_per_attention_head = (
                MAX_PAD_SIZE - self.hidden_size_per_attention_head
            ) // 2
            self.hidden_size_per_attention_head = MAX_PAD_SIZE

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()  # [seqlen, rotary_dim / 2]
        sin = rotary_pos_emb.sin()
        if self.enable_pad:
            cos = torch.nn.functional.pad(
                cos, (0, self.half_pad_hidden_size_per_attention_head)
            )
            sin = torch.nn.functional.pad(
                sin, (0, self.half_pad_hidden_size_per_attention_head)
            )

        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(
                torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2
            )
            sin_new = rearrange(
                torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2
            )
        cos_new = cos_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def pad_qkv_bias(self, bias: torch.Tensor):
        first_half = bias.reshape(-1, 3, self.origin_hidden_size_per_attention_head)[
            :, :, : self.half_origin_hidden_size_per_attention_head
        ]
        second_half = bias.reshape(-1, 3, self.origin_hidden_size_per_attention_head)[
            :, :, self.half_origin_hidden_size_per_attention_head :
        ]
        first_half_padded = torch.nn.functional.pad(
            first_half, (0, self.half_pad_hidden_size_per_attention_head)
        )
        second_half_padded = torch.nn.functional.pad(
            second_half, (0, self.half_pad_hidden_size_per_attention_head)
        )
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(-1)
        return bias_final

    def pad_qkv_weight(self, data: torch.Tensor):
        last_dim = data.shape[-1]
        qkv_weight_first_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, last_dim
        )[:, :, : self.half_origin_hidden_size_per_attention_head, :]
        qkv_weight_second_half = data.reshape(
            -1, 3, self.origin_hidden_size_per_attention_head, last_dim
        )[:, :, self.half_origin_hidden_size_per_attention_head :, :]

        qkv_weight_first_half_padded = torch.nn.functional.pad(
            qkv_weight_first_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head),
        )
        qkv_weight_second_half_padded = torch.nn.functional.pad(
            qkv_weight_second_half,
            (0, 0, 0, self.half_pad_hidden_size_per_attention_head),
        )
        qkv_weight_padded = torch.cat(
            [qkv_weight_first_half_padded, qkv_weight_second_half_padded], dim=2
        )
        qkv_weight_final = qkv_weight_padded.reshape(-1, last_dim)
        return qkv_weight_final

    def pad_proj_weight(self, data: torch.Tensor):
        if data.shape[-1] == 1:
            # weight_scale or weight_offset
            return data
        out_weight = torch.nn.functional.pad(
            data.reshape(
                self.hidden_size, -1, self.half_origin_hidden_size_per_attention_head
            ),
            (0, self.half_pad_hidden_size_per_attention_head, 0, 0),
        ).reshape(self.hidden_size, -1)
        return out_weight

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

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        position_embeddings = self.cal_cos_sin(rotary_pos_emb)

        # compute cu_seqlens
        cu_seqlens = (
            torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            .to(torch.int32)
        )

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)

        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x


cached_get_processor = lru_cache(get_processor)


class Qwen2_5_VLForConditionalGeneration(nn.Module):
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
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        if _is_npu:
            vision_transformer_class = AscendQwen2_5_VisionTransformer
        else:
            vision_transformer_class = Qwen2_5_VisionTransformer
        self.visual = vision_transformer_class(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            # NOTE: Qwen2_5-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.model = Qwen2Model(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
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
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

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
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        """Run forward pass for Qwen2_5-VL.

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
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

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

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
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
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                if (
                    _is_npu
                    and isinstance(self.visual, AscendQwen2_5_VisionTransformer)
                    and self.visual.enable_pad
                ):
                    if "attn.proj.weight" in name:
                        param.data = self.visual.pad_proj_weight(param.data)
                    if "attn.qkv_proj.weight" in name:
                        param.data = self.visual.pad_qkv_weight(param.data)
                    if (
                        "attn.qkv_proj.bias" in name
                        or "attn.qkv_proj.deq_scale" in name
                        or "attn.qkv_proj.quant_bias" in name
                    ):
                        param.data = self.visual.pad_qkv_bias(param.data)


EntryClass = [Qwen2_5_VLForConditionalGeneration]
