# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.distributed import get_sp_group
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import dict_to_3d_list

try:
    from st_attn import sliding_tile_attention

    st_attn_backend_available = True
except Exception:
    st_attn_backend_available = False

logger = init_logger(__name__)


class RangeDict(dict):

    def __getitem__(self, item: int) -> str:
        for key in self.keys():
            if isinstance(key, tuple):
                low, high = key
                if low <= item <= high:
                    return str(super().__getitem__(key))
            elif key == item:
                return str(super().__getitem__(key))
        raise KeyError(f"seq_len {item} not supported for STA")


class SlidingTileAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # TODO(will-refactor): check this
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SLIDING_TILE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SlidingTileAttentionImpl"]:
        return SlidingTileAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SlidingTileAttentionMetadata"]:
        return SlidingTileAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SlidingTileAttentionMetadataBuilder"]:
        return SlidingTileAttentionMetadataBuilder


@dataclass
class SlidingTileAttentionMetadata(AttentionMetadata):
    current_timestep: int
    STA_param: list[
        list[Any]
    ]  # each timestep with one metadata, shape [num_layers, num_heads]


class SlidingTileAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        STA_param: list[list[Any]],
        current_timestep: int,
        **kwargs: dict[str, Any],
    ) -> SlidingTileAttentionMetadata:
        param = STA_param
        if param is None:
            return SlidingTileAttentionMetadata(
                current_timestep=current_timestep, STA_param=[]
            )
        return SlidingTileAttentionMetadata(
            current_timestep=current_timestep, STA_param=param[current_timestep]
        )


class SlidingTileAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if not st_attn_backend_available:
            raise ValueError("st attn not supported")
        # TODO(will-refactor): for now this is the mask strategy, but maybe we should
        # have a more general config for STA?
        config_file = envs.SGL_DIFFUSION_ATTENTION_CONFIG
        if config_file is None:
            raise ValueError("SGL_DIFFUSION_ATTENTION_CONFIG is not set")

        # TODO(kevin): get mask strategy for different STA modes
        with open(config_file) as f:
            mask_strategy = json.load(f)
        self.mask_strategy = dict_to_3d_list(mask_strategy)

        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        # STA config
        self.STA_base_tile_size = [6, 8, 8]
        self.dit_seq_shape_mapping = RangeDict(
            {
                (115200, 115456): "30x48x80",
                82944: "36x48x48",
                69120: "18x48x80",
            }
        )
        self.full_window_mapping = {
            "30x48x80": [5, 6, 10],
            "36x48x48": [6, 6, 6],
            "18x48x80": [3, 6, 10],
        }

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2],
        )

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.STA_base_tile_size[0],
            ts_h=self.STA_base_tile_size[1],
            ts_w=self.STA_base_tile_size[2],
        )
        return x

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        img_sequence_length = qkv.shape[1]
        self.dit_seq_shape_str = self.dit_seq_shape_mapping[img_sequence_length]
        self.full_window_size = self.full_window_mapping[self.dit_seq_shape_str]
        self.dit_seq_shape_int = list(map(int, self.dit_seq_shape_str.split("x")))
        self.img_seq_length = (
            self.dit_seq_shape_int[0]
            * self.dit_seq_shape_int[1]
            * self.dit_seq_shape_int[2]
        )
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: SlidingTileAttentionMetadata,
    ) -> torch.Tensor:
        if self.mask_strategy is None:
            raise ValueError("mask_strategy cannot be None for SlidingTileAttention")
        if self.mask_strategy[0] is None:
            raise ValueError("mask_strategy[0] cannot be None for SlidingTileAttention")

        timestep = attn_metadata.current_timestep
        forward_context: ForwardContext = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            raise ValueError("forward_batch cannot be None")
        # pattern:'.double_blocks.0.attn.impl' or '.single_blocks.0.attn.impl'
        layer_idx = int(self.prefix.split(".")[-3])
        if attn_metadata.STA_param is None or len(attn_metadata.STA_param) <= layer_idx:
            raise ValueError("Invalid STA_param")
        STA_param = attn_metadata.STA_param[layer_idx]

        text_length = q.shape[1] - self.img_seq_length
        has_text = text_length > 0

        query = q.transpose(1, 2).contiguous()
        key = k.transpose(1, 2).contiguous()
        value = v.transpose(1, 2).contiguous()

        head_num = query.size(1)
        sp_group = get_sp_group()
        current_rank = sp_group.rank_in_group
        start_head = current_rank * head_num

        # searching or tuning mode
        if len(STA_param) < head_num * sp_group.world_size:
            sparse_attn_hidden_states_all = []
            full_mask_window = STA_param[-1]
            for window_size in STA_param[:-1]:
                sparse_hidden_states = sliding_tile_attention(
                    query,
                    key,
                    value,
                    [window_size] * head_num,
                    text_length,
                    has_text,
                    self.dit_seq_shape_str,
                ).transpose(1, 2)
                sparse_attn_hidden_states_all.append(sparse_hidden_states)

            hidden_states = sliding_tile_attention(
                query,
                key,
                value,
                [full_mask_window] * head_num,
                text_length,
                has_text,
                self.dit_seq_shape_str,
            ).transpose(1, 2)

            attn_L2_loss = []
            attn_L1_loss = []
            # average loss across all heads
            for sparse_attn_hidden_states in sparse_attn_hidden_states_all:
                # L2 loss
                attn_L2_loss_ = (
                    torch.mean(
                        (sparse_attn_hidden_states.float() - hidden_states.float())
                        ** 2,
                        dim=[0, 1, 3],
                    )
                    .cpu()
                    .numpy()
                )
                attn_L2_loss_ = [round(float(x), 6) for x in attn_L2_loss_]
                attn_L2_loss.append(attn_L2_loss_)
                # L1 loss
                attn_L1_loss_ = (
                    torch.mean(
                        torch.abs(
                            sparse_attn_hidden_states.float() - hidden_states.float()
                        ),
                        dim=[0, 1, 3],
                    )
                    .cpu()
                    .numpy()
                )
                attn_L1_loss_ = [round(float(x), 6) for x in attn_L1_loss_]
                attn_L1_loss.append(attn_L1_loss_)

            layer_loss_save = {"L2_loss": attn_L2_loss, "L1_loss": attn_L1_loss}

            if forward_batch.is_cfg_negative:
                if forward_batch.mask_search_final_result_neg is not None:
                    forward_batch.mask_search_final_result_neg[timestep].append(
                        layer_loss_save
                    )
            else:
                if forward_batch.mask_search_final_result_pos is not None:
                    forward_batch.mask_search_final_result_pos[timestep].append(
                        layer_loss_save
                    )
        else:
            windows = [STA_param[head_idx + start_head] for head_idx in range(head_num)]

            hidden_states = sliding_tile_attention(
                query,
                key,
                value,
                windows,
                text_length,
                has_text,
                self.dit_seq_shape_str,
            ).transpose(1, 2)

        return hidden_states
