# Copyright 2025 The HunYuan team.
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
"""HYV4 vision tower (ViT + spatial-merge projector)."""

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from sglang.srt.configs.hy_v4 import HYV4VLVisionConfig
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel


def _vision_reduces_over_attn_tp() -> bool:
    # The tower shards over attention TP; attention DP and prefill CP narrow that
    # group below TP, and reducing over full TP would mix other ranks' inputs.
    parallel = get_parallel()
    return parallel.attn_tp_size < parallel.tp_size


class HYV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: HYV4VLVisionConfig):
        super().__init__()
        self.config = config
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size, dtype=torch.float32
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        config = self.config
        patches = pixel_values.reshape(
            -1, config.num_channels, config.patch_size, config.patch_size
        )
        patches = self.patch_embedding(patches).flatten(2).squeeze(-1).unsqueeze(0)
        positions = (
            self.position_embedding.weight[1:]
            .float()
            .reshape(1, self.position_edge, self.position_edge, config.hidden_size)
            .permute(0, 3, 1, 2)
        )
        positional = []
        for t, h, w in grid_thw:
            resized = F.interpolate(
                positions,
                scale_factor=(
                    (h + 0.1) / self.position_edge,
                    (w + 0.1) / self.position_edge,
                ),
                mode=config.interpolate_mode,
                align_corners=False,
            )
            positional.append(
                resized.flatten(2).transpose(1, 2).to(patches.dtype).repeat(1, t, 1)
            )
        return patches + torch.cat(positional, dim=1)


class HYV4VisionMLP(nn.Module):
    def __init__(
        self,
        config: HYV4VLVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("dense_h_to_4h", prefix),
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("dense_4h_to_h", prefix),
            use_dp_attention_reduce=_vision_reduces_over_attn_tp(),
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.dense_h_to_4h(x)[0])
        return self.dense_4h_to_h(x)[0]


class HYV4VisionBlock(nn.Module):
    def __init__(
        self,
        config: HYV4VLVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            use_dp_attention_reduce=_vision_reduces_over_attn_tp(),
        )
        self.mlp = HYV4VisionMLP(config, quant_config, add_prefix("mlp", prefix))

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            forward_metadata=forward_metadata,
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class HYV4VisionPatchMerger(nn.Module):
    def __init__(self, config: HYV4VLVisionConfig):
        super().__init__()
        self.config = config
        dim, output, merge = (
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
        )
        hidden = config.perceive_intermediate_size or dim * merge**2
        self.extra_mlp = config.projector_type is None
        if config.projector_type == "rmsnorm_mlp_adapter":
            self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
            self.proj = nn.Sequential(
                nn.Conv2d(dim, hidden, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(hidden, output, 1),
            )
            newline_size = output
        elif config.projector_type is None:
            self.proj = nn.Sequential(
                nn.Conv2d(dim, dim * merge, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(dim * merge, hidden, 1),
            )
            self.mlp = nn.Linear(hidden, output)
            # Checkpoint-owned parameter; the perceive forward never reads it.
            self.image_sep = nn.Parameter(torch.empty(output))
            newline_size = hidden
        else:
            raise ValueError(
                f"Unsupported HYV4 projector_type: {config.projector_type}"
            )
        if config.use_image_newline_token:
            self.image_newline = nn.Parameter(torch.empty(newline_size))
        if config.cat_extra_token:
            self.image_begin = nn.Parameter(torch.empty(output))
            self.image_end = nn.Parameter(torch.empty(output))
        self.before_rms = (
            RMSNorm(dim, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_pre_norm
            else None
        )
        self.after_rms = (
            RMSNorm(output, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_post_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self.config
        if not self.extra_mlp:
            x = self.norm(x)
        if self.before_rms is not None:
            x = self.before_rms(x)
        dtype = x.dtype
        x = self.proj(x.permute(0, 3, 1, 2).contiguous())
        if x.shape[0] > 1:
            x = F.avg_pool3d(
                x.permute(1, 0, 2, 3),
                (config.temporal_patch_size, 1, 1),
                stride=(config.temporal_patch_size, 1, 1),
            ).permute(1, 0, 2, 3)
            x = F.avg_pool2d(
                x, config.spatial_patch_size, stride=config.spatial_patch_size
            )
        b, c, h, _ = x.shape
        if config.use_image_newline_token:
            x = torch.cat(
                (
                    x,
                    self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype),
                ),
                dim=-1,
            )
        x = x.flatten(2).transpose(1, 2)
        if self.extra_mlp:
            x = self.mlp(x)
        if config.cat_extra_token:
            x = torch.cat(
                (
                    self.image_begin.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                    x,
                    self.image_end.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                ),
                dim=1,
            )
        x = x.reshape(-1, x.shape[-1])
        return self.after_rms(x) if self.after_rms is not None else x


class HYV4VisionTransformer(nn.Module):
    def __init__(
        self,
        config: HYV4VLVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = HYV4VisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HYV4VisionBlock(config, quant_config, add_prefix(f"layers.{i}", prefix))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.perceive = HYV4VisionPatchMerger(config)
        self.rotary_pos_emb = None
        if config.use_2d_rope:
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_pos_emb = get_rope(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=self.embeddings.position_edge + 1,
                base=config.rope_2d_theta,
                is_neox_style=True,
                partial_rotary_factor=0.5,
                dtype=torch.float32,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        cos_table, sin_table = self.rotary_pos_emb.get_cos_sin(max_grid)
        indices = [
            torch.stack(
                (torch.arange(h).repeat_interleave(w), torch.arange(w).repeat(h)),
                dim=-1,
            ).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        indices = torch.cat(indices).to(cos_table.device)
        return cos_table[indices].flatten(1), sin_table[indices].flatten(1)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> list[torch.Tensor]:
        config = self.config
        grid_thw = [[int(v) for v in grid] for grid in grid_thw]
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        if sum(lengths) != pixel_values.shape[0] or not lengths or min(lengths) <= 0:
            raise ValueError(
                "HYV4 pixel_values length does not match positive grid_thw"
            )
        hidden = self.embeddings(
            pixel_values.to(device=self.device, dtype=self.dtype), grid_thw
        )
        forward_metadata = prepare_vision_attention_metadata(
            torch.from_numpy(np.cumsum([0, *lengths], dtype=np.int32)),
            self.device,
            max_seqlen=max(lengths),
        )
        cos = sin = None
        if self.rotary_pos_emb is not None:
            cos, sin = self.rot_pos_emb(grid_thw)
        for layer in self.layers:
            hidden = layer(
                hidden,
                cu_seqlens=forward_metadata.cu_seqlens,
                forward_metadata=forward_metadata,
                rotary_pos_emb_cos=cos,
                rotary_pos_emb_sin=sin,
            )
        outputs, offset = [], 0
        for t, h, w in grid_thw:
            states = hidden[:, offset : offset + t * h * w].reshape(t, h, w, -1)
            offset += t * h * w
            if t > 1 and t % config.temporal_patch_size:
                pad = config.temporal_patch_size - t % config.temporal_patch_size
                states = torch.cat((states, states[-1:].expand(pad, -1, -1, -1)), dim=0)
            outputs.append(self.perceive(states))
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("self_attn.qkv_proj.", "self_attn.q_proj.", "q"),
            ("self_attn.qkv_proj.", "self_attn.k_proj.", "k"),
            ("self_attn.qkv_proj.", "self_attn.v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, loaded_weight in weights:
            if "self_attn.o_proj." in name:
                name = name.replace("self_attn.o_proj.", "self_attn.proj.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded.add(name)
        return loaded
