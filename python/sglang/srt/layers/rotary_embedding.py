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
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding,
    _rotate_gptj,
    _rotate_neox,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    get_rope,
    yarn_get_mscale,
)


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


# TODO: in the DeepseekScalingRotaryEmbedding class defined in vllm,
# the device has been hard-coded to "cuda" in these two places:
# https://github.com/vllm-project/vllm/blob/8a1f938e6f02052df0f4953c149410605a2d56d8/vllm/model_executor/layers/rotary_embedding.py#L646
# https://github.com/vllm-project/vllm/blob/8a1f938e6f02052df0f4953c149410605a2d56d8/vllm/model_executor/layers/rotary_embedding.py#L665
# We port the related code to this file to make it compatible with the CPU version.
# We will add an optimized rotary embedding kernel for CPU and will remove the ported code then.
class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        device: Optional[str] = None,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.device = device
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        print("Cache shape", cache.shape)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope_cpu(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
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

    assert rope_scaling is not None
    scaling_type = rope_scaling["rope_type"]
    assert (
        scaling_type == "deepseek_yarn"
    ), "Only deepseek_yarn is supported for CPU for now"

    scaling_factor = rope_scaling["factor"]
    original_max_position = rope_scaling["original_max_position_embeddings"]
    # assert max_position == original_max_position * scaling_factor
    extra_kwargs = {
        k: v
        for k, v in rope_scaling.items()
        if k
        in (
            "extrapolation_factor",
            "attn_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
        )
    }
    extra_kwargs["device"] = device
    rotary_emb = DeepseekScalingRotaryEmbedding(
        head_size,
        rotary_dim,
        original_max_position,
        base,
        is_neox_style,
        scaling_factor,
        dtype,
        **extra_kwargs,
    )

    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
):
    if device != "cpu":
        return get_rope(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )

    return get_rope_cpu(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
        partial_rotary_factor,
        device,
    )
