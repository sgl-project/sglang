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

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
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
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
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
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
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
                enable_tp=not is_dp_attention_enabled(),
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

        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = ViTCudaGraphRunner(self)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

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

    def fast_pos_embed_interpolate(self, grid_thw):
        patch_pos_embeds_permute = []
        m_size = self.spatial_merge_size

        embeds = torch.arange(self.num_grid, device=self.pos_embed.weight.device)
        embeds = (
            self.pos_embed(embeds)
            .permute(1, 0)
            .reshape(1, -1, self.num_grid_per_side, self.num_grid_per_side)
        )
        for t, h, w in grid_thw:
            pos_embed = torch.nn.functional.interpolate(
                embeds, size=(h, w), mode="bilinear", align_corners=self.align_corners
            )
            pos_embed = pos_embed.reshape(
                -1,
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            pos_embed = pos_embed.permute(1, 3, 2, 4, 0)
            pos_embed = pos_embed.flatten(0, 3).repeat(t, 1)
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            return self.forward_with_cuda_graph(x, grid_thw)

        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds

        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # compute cu_seqlens
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if not is_npu():
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        else:
            cu_seqlens = cu_seqlens.to("cpu")
        x = x.unsqueeze(1)

        deepstack_feature_lists = []
        num_deepstack_captured = 0

        for layer_num, blk in enumerate(self.blocks):
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
        x = self.merger(x)
        hidden_states = torch.cat(
            [x] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
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

        # concatenate back the full image embedding sequence
        return torch.cat(all_chunk_embeds, dim=0)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
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

            if self.pp_group.is_last_rank and "model.embed_tokens.weight" in name:
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


EntryClass = Qwen3VLForConditionalGeneration
