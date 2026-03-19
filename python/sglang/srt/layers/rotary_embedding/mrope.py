"""MRotaryEmbedding, YaRNScalingMRotaryEmbedding, Ernie4_5_VLRotaryEmbedding,
apply_interleaved_rope for multimodal RoPE."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.triton_kernels import (
    triton_ernie45_rope_fused_inplace,
    triton_mrope_fused,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.rotary_embedding.yarn import (
    yarn_find_correction_range,
    yarn_get_mscale_simple,
    yarn_linear_ramp_mask,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()

if _is_cuda:
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

if _is_npu:
    import torch_npu


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor))
                        for section in self.mrope_section
                    ]
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    self.mrope_section = [
                        expected_sum // len(self.mrope_section)
                    ] * len(self.mrope_section)
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1
                print(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )

        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native

    def get_cos_sin_with_position(self, positions):
        if positions.ndim == 1:
            return super().get_cos_sin_with_position(positions)
        assert positions.ndim == 2
        assert self.mrope_section
        cos_sin = self.cos_sin_cache[positions]
        last_dim = cos_sin.size()[-1]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.mrope_interleaved:
            cos = apply_interleaved_rope(cos, self.mrope_section)
            sin = apply_interleaved_rope(sin, self.mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
        self.position_cos = cos.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()
        self.position_sin = sin.repeat(1, 2).view(-1, 1, 1, last_dim).contiguous()

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
        if (
            self.cos_sin_cache.device != query.device
            or self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "save kv cache is not supported for MRotaryEmbedding."
        assert positions.ndim == 1 or positions.ndim == 2

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            if self.mrope_interleaved:
                cos = apply_interleaved_rope(cos, self.mrope_section)
                sin = apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )

        seq_len_q = query.shape[0]
        query_shape = query.shape
        query = query.view(seq_len_q, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        seq_len_k = key.shape[0]
        key_shape = key.shape
        key = key.view(seq_len_k, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2
        if positions.ndim == 2 and self.mrope_section:
            return self.forward_triton(positions, query, key)
        return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)

    def forward_triton(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.mrope_section
        self._match_cos_sin_cache_dtype(query)
        triton_mrope_fused(
            query,
            key,
            self.cos_sin_cache,
            positions,
            self.mrope_section,
            self.head_size,
            self.rotary_dim,
            self.mrope_interleaved,
            self.is_neox_style,
        )
        return query, key

    def forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"
        if query.shape[1] > 4096:
            return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)
        rotary_mode = "half" if self.is_neox_style else "interleave"
        mrope_section = [0, 0, 0]
        query_out, key_out = torch_npu.npu_mrope(
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode=rotary_mode,
        )
        return query_out, key_out

    @staticmethod
    def get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        model_type,
        tokens_per_second=None,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index

        return get_rope_index(
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            model_type,
            tokens_per_second,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_qwen3_omni(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second=None,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
            get_rope_index_qwen3_omni,
        )

        return get_rope_index_qwen3_omni(
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_glm4v(
        input_ids, hf_config, image_grid_thw, video_grid_thw, attention_mask, **kwargs
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
            get_rope_index_glm4v,
        )

        return get_rope_index_glm4v(
            input_ids,
            hf_config,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            **kwargs,
        )

    @staticmethod
    def get_rope_index_ernie45(
        input_ids, hf_config, image_grid_thw, video_grid_thw, **kwargs
    ):
        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
            get_rope_index_ernie45,
        )

        return get_rope_index_ernie45(
            input_ids, hf_config, image_grid_thw, video_grid_thw, **kwargs
        )


class YaRNScalingMRotaryEmbedding(MRotaryEmbedding):
    """MRoPE-enabled rotary embedding with YaRN context scaling."""

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
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        truncate: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        self.mscale = float(yarn_get_mscale_simple(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Ernie4_5_VLRotaryEmbedding(MRotaryEmbedding):
    """3D rotary positional embedding. [h w h w h w h w... t t t...]"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )
        self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(apply_rotary_emb)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor = None,
    ):
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            section_h = self.mrope_section[0]
            section_w = self.mrope_section[1]
            section_t = self.mrope_section[2]
            assert section_h == section_w
            section_cos_t = cos[..., -section_t:]
            section_cos_h = cos[..., : section_h + section_w : 2]
            section_cos_w = cos[..., 1 : section_h + section_w : 2]
            cos_t, cos_h, cos_w = section_cos_t[0], section_cos_h[1], section_cos_w[2]
            cos_hw = torch.stack([cos_h, cos_w], dim=-1).reshape(
                cos_h.shape[:-1] + (cos_h.shape[-1] * 2,)
            )
            cos = torch.cat([cos_hw, cos_t], dim=-1)
            section_sin_t = sin[..., -section_t:]
            section_sin_h = sin[..., : section_h + section_w : 2]
            section_sin_w = sin[..., 1 : section_h + section_w : 2]
            sin_t, sin_h, sin_w = section_sin_t[0], section_sin_h[1], section_sin_w[2]
            sin_hw = torch.stack([sin_h, sin_w], dim=-1).reshape(
                sin_h.shape[:-1] + (sin_h.shape[-1] * 2,)
            )
            sin = torch.cat([sin_hw, sin_t], dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb_wrapped(
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor = None,
    ):
        assert key is not None
        assert positions.ndim in (1, 2)
        self._match_cos_sin_cache_dtype(query)

        if positions.ndim == 2:
            assert self.mrope_section is not None
            triton_ernie45_rope_fused_inplace(
                q=query,
                k=key,
                cos_sin_cache=self.cos_sin_cache,
                positions=positions,
                mrope_section=self.mrope_section,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
            )
            return query, key

        if _is_cuda and (apply_rope_with_cos_sin_cache_inplace is not None):
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
            return query, key

        return self.forward_native(positions, query, key)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2
        return self.forward_cuda(positions, query, key)
