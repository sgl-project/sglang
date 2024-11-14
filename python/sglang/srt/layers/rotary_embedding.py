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
"""MRotaryEmbedding"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from vllm.model_executor.custom_op import CustomOp

from sglang.srt.layers.custom_op_util import register_custom_op


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class MRotaryEmbedding:
    """Rotary Embedding with Multimodal Sections."""

    @staticmethod
    def get_input_positions(
        input_tokens: torch.Tensor,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        vision_start_token_id: int,
        spatial_merge_size: int,
        context_len: int = 0,
    ) -> Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()

        vision_start_indices = torch.argwhere(
            input_tokens == vision_start_token_id
        ).squeeze(1)
        image_indices = vision_start_indices + 1
        image_nums = image_indices.shape[0]
        llm_pos_ids_list: list = []

        st = 0
        input_tokens_len = input_tokens.shape[0]
        for image_index in range(image_nums):
            ed = image_indices[image_index].item()
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < input_tokens_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = input_tokens_len - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:]
        mrope_position_delta = (llm_positions.max() + 1 - input_tokens_len).item()
        return llm_positions.tolist(), mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> List[List[int]]:
        return [
            list(
                range(
                    context_len + mrope_position_delta, seq_len + mrope_position_delta
                )
            )
            for _ in range(3)
        ]


@register_custom_op("sglang_rope")
class RotaryEmbedding(CustomOp):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.rope import apply_rope_pos_ids_inplace

        if offsets is not None:
            positions = positions + offsets
        seq_len, num_q_heads, num_k_heads = (
            positions.shape[0],
            query.shape[1] // self.head_size,
            key.shape[1] // self.head_size,
        )

        #  (seq_len, num_heads * head_dim) -> flashinfer input shape (nnz=seq_len, num_heads, head_dim)
        flashinfer_query, flashinfer_key = rearrange(
            query.type(torch.float16),
            "s (n_h h_d) -> s n_h h_d",
            n_h=num_q_heads,
            h_d=self.head_size,
        ), rearrange(
            key.type(torch.float16),
            "s (n_h h_d) -> s n_h h_d",
            n_h=num_k_heads,
            h_d=self.head_size,
        )
        apply_rope_pos_ids_inplace(
            flashinfer_query,
            flashinfer_key,
            pos_ids=positions,
            rotary_dim=self.rotary_dim,
            rope_theta=self.base,
            interleave=(not self.is_neox_style),
        )

        # flashinfer output shape (nnz=seq_len, num_heads, head_dim) -> (seq_len, num_heads * head_dim)
        return rearrange(
            flashinfer_query.type(self.dtype), "s n_h h_d -> s (n_h h_d)"
        ), rearrange(flashinfer_key.type(self.dtype), "s n_h h_d -> s (n_h h_d)")

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        scaling_type = rope_scaling["rope_type"]

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                )
        #     else:
        #         rotary_emb = RotaryEmbedding(
        #             head_size,
        #             rotary_dim,
        #             max_position,
        #             base,
        #             is_neox_style,
        #             dtype,
        #         )
        # elif scaling_type == "linear":
        #     scaling_factor = rope_scaling["factor"]
        #     rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
        #                                               max_position, base,
        #                                               is_neox_style,
        #                                               scaling_factor, dtype)
        # elif scaling_type == "dynamic":
        #     scaling_factor = rope_scaling["factor"]
        #     rotary_emb = DynamicNTKScalingRotaryEmbedding(
        #         head_size, rotary_dim, max_position, base, is_neox_style,
        #         scaling_factor, dtype)
        # elif scaling_type == "yarn":
        #     scaling_factor = rope_scaling["factor"]
        #     original_max_position = rope_scaling[
        #         "original_max_position_embeddings"]
        #     extra_kwargs = {
        #         k: v
        #         for k, v in rope_scaling.items()
        #         if k in ("extrapolation_factor", "attn_factor", "beta_fast",
        #                  "beta_slow")
        #     }
        #     rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
        #                                             original_max_position,
        #                                             base, is_neox_style,
        #                                             scaling_factor, dtype,
        #                                             **extra_kwargs)
        # elif scaling_type == "deepseek_yarn":
        #     scaling_factor = rope_scaling["factor"]
        #     original_max_position = rope_scaling[
        #         "original_max_position_embeddings"]
        #     # assert max_position == original_max_position * scaling_factor
        #     extra_kwargs = {
        #         k: v
        #         for k, v in rope_scaling.items()
        #         if k in ("extrapolation_factor", "attn_factor", "beta_fast",
        #                  "beta_slow", "mscale", "mscale_all_dim")
        #     }
        #     rotary_emb = DeepseekScalingRotaryEmbedding(
        #         head_size, rotary_dim, original_max_position, base,
        #         is_neox_style, scaling_factor, dtype, **extra_kwargs)
        # elif scaling_type == "longrope":
        #     short_factor = rope_scaling["short_factor"]
        #     long_factor = rope_scaling["long_factor"]
        #     original_max_position = rope_scaling[
        #         "original_max_position_embeddings"]
        #     extra_kwargs = {
        #         k: v
        #         for k, v in rope_scaling.items()
        #         if k in ("short_mscale", "long_mscale")
        #     }
        #     rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
        #         head_size, rotary_dim, max_position, original_max_position,
        #         base, is_neox_style, dtype, short_factor, long_factor,
        #         **extra_kwargs)
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
