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
from inspect import signature
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import parallel_state
from sglang.srt.hf_transformers_utils import get_processor
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
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2_vl import Qwen2VLVideoInputs
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# Add back the PPMissingLayer for non-last PP ranks
class PPMissingLayer(nn.Module):
    """A placeholder layer for pipeline parallelism used in non-last ranks."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        return None
    
    @property
    def weight(self):
        return None


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


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = "sdpa",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = Qwen2RMSNorm(dim, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(dim, eps=1e-6)
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
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
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
                    attn_implementation="sdpa",
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


cached_get_processor = lru_cache(get_processor)


# ... (imports from the beginning of the file, including `from inspect import signature`)

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
        self.pp_group = parallel_state.get_pp_group()
        self.pp_rank = self.pp_group.rank_in_group
        self.pp_size = self.pp_group.world_size

        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_rank == self.pp_size - 1:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.image_token_idx = config.image_token_id
        if self.pp_rank == 0:
            self.visual = Qwen2_5_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
            )
        else:
            # The other PP ranks use a placeholder layer to avoid occupying GPU memory / back-propagating params.
            self.visual = PPMissingLayer()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_token_id: int = mm_inputs.im_token_id
        pattern = MultiModalityDataPaddingPatternMultimodalTokens([im_token_id])
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Return the (total_seq, hidden) Tensor concatenated in the order of items.
        - PP0 is responsible for the real calculation
        - The other PP ranks get the same result through NCCL broadcast
        """
        device = next(self.parameters()).device
        dtype = next(self.model.parameters()).dtype

        if self.pp_rank == 0:
            pixel_values = torch.cat([it.pixel_values for it in items], dim=0).to(
                self.visual.dtype
            )
            grid_thw = torch.cat([it.image_grid_thw for it in items], dim=0)
            embeds = self.visual(pixel_values, grid_thw)
            embeds_flat = embeds.contiguous()

            shape_tensor = torch.tensor(
                embeds_flat.shape, device=device, dtype=torch.long
            )
            dist.broadcast(shape_tensor, src=0, group=self.pp_group)
            dist.broadcast(embeds_flat, src=0, group=self.pp_group)
        else:
            shape_tensor = torch.empty(2, dtype=torch.long, device=device)
            dist.broadcast(shape_tensor, src=0, group=self.pp_group)
            embeds_flat = torch.empty(
                tuple(shape_tensor.tolist()), dtype=dtype, device=device
            )
            dist.broadcast(embeds_flat, src=0, group=self.pp_group)

        return embeds_flat
    
    def _process_video_input(self, video_input: Qwen2VLVideoInputs) -> torch.Tensor:
        # Not implemented yet
        raise NotImplementedError("Video input is not supported yet.")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if get_embedding:
            return self.model.embed_tokens(input_ids)

        # Pipeline parallelism logic
        if pp_proxy_tensors is not None and "hidden_states" in pp_proxy_tensors.tensors:
            # This is a rank > 0. The input is the output from the previous stage.
            # We treat it as `inputs_embeds` for this stage's model pass.
            hidden_states = self.model(
                input_ids=input_ids,
                input_embeds=pp_proxy_tensors.tensors["hidden_states"],
                positions=positions,
                forward_batch=forward_batch,
            )
        else:
            # This is rank 0.
            if forward_batch.mm_data is not None:
                # Prepare multimodal embeddings
                input_embeds = general_mm_embed_routine(
                    language_model=self,
                    input_ids=input_ids,
                    positions=positions,
                    forward_batch=forward_batch,
                    image_data_embedding_func=self.get_image_feature,
                )
                hidden_states = self.model(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    positions=positions,
                    forward_batch=forward_batch,
                )
            else:
                # Text-only case
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    forward_batch=forward_batch,
                )

        if self.pp_rank == self.pp_size - 1:
            logits = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
            return logits
        else:
            return PPProxyTensors({"hidden_states": hidden_states})
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Define the prefix map: {checkpoint_prefix: sglang_model_prefix}
        prefix_map = {
            "model.": "model.",
            "lm_head.": "lm_head.",
            "visual.": "visual.",
        }
        params_dict = dict(self.named_parameters())
        for name, tensor in weights:
            if self.pp_rank != 0 and name.startswith("visual."):
                continue

            sglang_name = name
            for old_prefix, new_prefix in prefix_map.items():
                if sglang_name.startswith(old_prefix):
                    sglang_name = new_prefix + sglang_name[len(old_prefix) :]
                    break
            
            # Handle the fused QKV weights for the visual model from the checkpoint.
            if "visual.blocks" in sglang_name and ".attn.qkv." in sglang_name:
                # The checkpoint has a fused QKV weight, but the sglang model has separate
                # q_proj, k_proj, and v_proj parameters. We need to split the loaded tensor.
                try:
                    q_tensor, k_tensor, v_tensor = torch.chunk(tensor, 3, dim=0)
                    
                    # Construct the names for the separate parameters in the sglang model.
                    q_name = sglang_name.replace(".qkv.", ".q_proj.")
                    k_name = sglang_name.replace(".qkv.", ".k_proj.")
                    v_name = sglang_name.replace(".qkv.", ".v_proj.")

                    # Load each chunk into its corresponding parameter.
                    for p_name, p_tensor in [(q_name, q_tensor), (k_name, k_tensor), (v_name, v_tensor)]:
                        if p_name in params_dict:
                            param = params_dict[p_name]
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, p_tensor)
                        else:
                            logger.warning(
                                f"Weight loading failed. Skipped split visual weight: '{p_name}' was not found on this rank (PP rank {self.pp_rank})."
                            )
                    continue
                except Exception as e:
                    logger.error(f"Error splitting visual QKV weight {sglang_name}: {e}")
                    continue

            is_stacked = False
            for src_proj_name in self.bitsandbytes_stacked_params_mapping:
                pattern_base = f".{src_proj_name}"
                # Correctly check for weights ending with patterns like ".q_proj.weight" or ".q_proj.bias"
                if sglang_name.endswith(f"{pattern_base}.weight") or sglang_name.endswith(
                    f"{pattern_base}.bias"
                ):
                    (
                        stacked_param_name,
                        shard_idx,
                    ) = self.bitsandbytes_stacked_params_mapping[src_proj_name]
                    # Correctly replace the project name part with the stacked name
                    final_name = sglang_name.replace(
                        pattern_base, f".{stacked_param_name}"
                    )

                    if final_name in params_dict:
                        param = params_dict[final_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        sig = signature(weight_loader)
                        if "loaded_shard_id" in sig.parameters:
                            if src_proj_name in ["q_proj", "k_proj", "v_proj"]:
                                loaded_shard_id = src_proj_name.split("_")[0]
                            else:
                                loaded_shard_id = shard_idx
                            weight_loader(
                                param, tensor, loaded_shard_id=loaded_shard_id
                            )
                        else:
                            weight_loader(param, tensor)
                        is_stacked = True
                        break

            if is_stacked:
                continue

            if sglang_name in params_dict:
                param = params_dict[sglang_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
            else:
                logger.warning(
                    f"Weight loading failed. Skipped weight: '{name}'. "
                    f"The corresponding parameter '{sglang_name}' was not found on this rank (PP rank {self.pp_rank})."
                )



EntryClass = [Qwen2_5_VLForConditionalGeneration]
