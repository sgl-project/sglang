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
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
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
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

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
    ):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output


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


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act="silu",
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        attn_implementation: Optional[str] = "sdpa",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if attn_implementation == "sdpa":
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

        self.attn = VisionAttention(
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
        self.mlp = Qwen3_VisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
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
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim
        )
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
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


class Qwen3VLMoeVisionModel(nn.Module):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        # layer indexes of which layer's output should be deep-stacked
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    attn_implementation="flash_attention_3",
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
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
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
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

    def fast_pos_embed_interpolate(self, grid_thw):
        num_grid_per_side = int(self.num_position_embeddings**0.5)

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        # TODO: use torch instand of np
        for t, h, w in grid_thw:
            h_idxs = np.linspace(0, num_grid_per_side - 1, h)
            w_idxs = np.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.astype(int)
            w_idxs_floor = w_idxs.astype(int)
            h_idxs_ceil = (h_idxs.astype(int) + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.astype(int) + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            idx_list[0].extend(
                ((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_floor[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[1].extend(
                ((h_idxs_floor * num_grid_per_side)[None].T + w_idxs_ceil[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[2].extend(
                ((h_idxs_ceil * num_grid_per_side)[None].T + w_idxs_floor[None])
                .flatten()
                .tolist()
                * t
            )
            idx_list[3].extend(
                ((h_idxs_ceil * num_grid_per_side)[None].T + w_idxs_ceil[None])
                .flatten()
                .tolist()
                * t
            )

            weight_list[0].extend(
                ((1 - dh)[None].T * (1 - dw)[None]).flatten().tolist() * t
            )
            weight_list[1].extend(((1 - dh)[None].T * dw[None]).flatten().tolist() * t)
            weight_list[2].extend((dh[None].T * (1 - dw)[None]).flatten().tolist() * t)
            weight_list[3].extend((dh[None].T * dw[None]).flatten().tolist() * t)

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        p0 = (
            self.pos_embed(torch.tensor(idx_list[0], dtype=torch.long, device=device))
            * torch.tensor(weight_list[0], dtype=dtype, device=device)[:, None]
        )
        p1 = (
            self.pos_embed(torch.tensor(idx_list[1], dtype=torch.long, device=device))
            * torch.tensor(weight_list[1], dtype=dtype, device=device)[:, None]
        )
        p2 = (
            self.pos_embed(torch.tensor(idx_list[2], dtype=torch.long, device=device))
            * torch.tensor(weight_list[2], dtype=dtype, device=device)[:, None]
        )
        p3 = (
            self.pos_embed(torch.tensor(idx_list[3], dtype=torch.long, device=device))
            * torch.tensor(weight_list[3], dtype=dtype, device=device)[:, None]
        )

        patch_pos_embeds = p0 + p1 + p2 + p3
        patch_pos_embeds = patch_pos_embeds.split([t * h * w for t, h, w in grid_thw])
        patch_pos_embeds_permute = []
        m_size = self.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_thw):
            pos_embed = (
                pos_embed.view(t, h // m_size, m_size, w // m_size, m_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = x.size()
        rotary_pos_emb = rotary_pos_emb.to(x.device)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
                cu_seqlens.to(torch.int32),
            ]
        )

        x = x.unsqueeze(1)

        deepstack_feature_lists = []
        num_deepstack_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](
                    x
                )
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1
        x = self.merger(x)
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

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

            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

            # process deepstack
            if (
                input_deepstack_embeds is not None
                and layer_idx in self.deepstack_embed_to_decoder_layer
            ):
                sep = self.hidden_size * layer_idx
                hidden_states += input_deepstack_embeds[:, sep : sep + self.hidden_size]

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
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3LLMModel,
    ) -> None:
        super().__init__()

        self.visual = Qwen3VLMoeVisionModel(
            config.vision_config,
            # NOTE: Qwen3-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
            # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
        )

        # TODO: make it more elegant
        if language_model_cls is Qwen3LLMModel:
            self.config: Qwen3VLConfig = config  # for qwen3-vl
        else:
            self.config = config.text_config  # for qwen3-omni

        self.model = language_model_cls(
            config=self.config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(self.config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        # like {8:0, 16:1, 24:2}, which stands for the captured deepstack features on
        # 8, 16, 24 layer will be merged to 0, 1, 2 layer of decoder output hidden_states

        # deepstack
        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
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
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")

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
                    name = name.replace(r"model.visual.", r"visual.")

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


EntryClass = Qwen3VLForConditionalGeneration
