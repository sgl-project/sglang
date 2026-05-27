"""Inference-only MiMo vision model: attention + ViT."""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionPatchMerger, Qwen2_5_VLMLP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


class MiMoVLVisionConfig(PretrainedConfig):
    model_type = "mimovl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=28,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=4608,
        num_heads=32,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        window_size=128,
        out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        kv_channels=64,
        qk_channels=64,
        num_query_groups=4,
        num_key_value_heads=8,
        vit_window_attn_types=None,
        visual_token_window_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.kv_channels = kv_channels
        self.qk_channels = qk_channels
        self.num_query_groups = num_query_groups
        self.vit_window_attn_types = vit_window_attn_types or [-1] * depth
        self.visual_token_window_size = visual_token_window_size


class MiMoVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.proj_weight_linear_format = None

    @torch.no_grad()
    def sync_proj_weight_linear_format(self):
        self.proj_weight_linear_format = self.proj.weight.view(self.embed_dim, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = F.linear(
            hidden_states.to(dtype=target_dtype), self.proj_weight_linear_format
        )
        return hidden_states


class MiMoVisionBlock(nn.Module):
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
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_sink: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.use_data_parallel = use_data_parallel

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

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            qkv_bias=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=flatten_batch,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_sink=use_sink,
            window_size=window_size,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: torch.Tensor,
        full_attn: bool = True,
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
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            full_attn=full_attn,
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


class MiMoVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: MiMoVLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.server_args = get_global_server_args()
        self.vit_window_attn_types = vision_config.vit_window_attn_types
        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        num_kv_heads = getattr(vision_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_channels = getattr(vision_config, "qk_channels", None)
        self.kv_channels = getattr(vision_config, "kv_channels", None)
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.use_data_parallel = self.server_args.mm_enable_dp_encoder
        mlp_hidden_size: int = vision_config.intermediate_size
        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.use_sink = getattr(vision_config, "use_sink", False)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = (
            self.qk_channels
            if self.qk_channels is not None
            else hidden_size // num_heads
        )
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.visual_token_window_size = getattr(
            vision_config, "visual_token_window_size", -1
        )
        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    attn_implementation="flash_attention_3",
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_sink=(
                        self.use_sink if i not in self.fullatt_block_indexes else False
                    ),
                    window_size=(
                        self.visual_token_window_size,
                        self.visual_token_window_size,
                    ),
                    num_kv_heads=num_kv_heads,
                    head_dim=self.qk_channels,
                    use_data_parallel=self.use_data_parallel,
                )
                for i in range(depth)
            ]
        )

        self.vision_config = vision_config
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=self.use_data_parallel,
        )
        self._post_init()

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor):
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        tensor = tensor.flatten(0, 1)
        return tensor

    def _post_init(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()

    def get_window_index_1d(self, grid_thw, col=True):
        window_index: list = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            if col:
                index_new = index.transpose(1, 2).reshape(-1)
            else:
                index_new = index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(
            window_index,
            dim=0,
        )
        return window_index

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.gate_up_proj.weight.device

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

    def _prepare_forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(
            device=x.device
        )
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col).to(
            device=x.device
        )

        rotary_pos_emb = rotary_pos_emb.to(device=x.device)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        def get_position_embeddings(emb, x):
            position_embeddings = (emb.cos(), emb.sin())
            position_embeddings = (
                position_embeddings[0].to(x.device),
                position_embeddings[1].to(x.device),
            )
            return position_embeddings

        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                seqlens.cumsum(dim=0).to(device=x.device, dtype=torch.int32),
            ]
        )
        max_seqlen = seqlens.max().item()

        row_based_embeddings = get_position_embeddings(emb, x)
        col_based_embeddings = get_position_embeddings(
            self.apply_index(emb, window_index_1d_col), x
        )

        # transformers
        x = x.unsqueeze(1)  # [S, 1, H]

        return (
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        )

    def run_blocks(
        self,
        x: torch.Tensor,
        row_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        col_based_embeddings: Tuple[torch.Tensor, torch.Tensor],
        window_index_1d_col: torch.Tensor,
        reverse_window_index_1d_col: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer_num, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[layer_num]

            # window_attn_type = 1: col-based SWA
            if window_attn_type == 1 and (
                layer_num == 0 or self.vit_window_attn_types[layer_num - 1] != 1
            ):
                x = self.apply_index(x, window_index_1d_col)

            if (
                layer_num > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[layer_num - 1] == 1
            ):
                x = self.apply_index(x, reverse_window_index_1d_col)

            position_embeddings = (
                col_based_embeddings if window_attn_type == 1 else row_based_embeddings
            )
            full_attn = layer_num in self.fullatt_block_indexes

            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                full_attn=full_attn,
            )
        x = self.merger(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        (
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        ) = self._prepare_forward(x, grid_thw)

        return self.run_blocks(
            x,
            row_based_embeddings,
            col_based_embeddings,
            window_index_1d_col,
            reverse_window_index_1d_col,
            cu_seqlens,
            max_seqlen,
        )
