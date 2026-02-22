"""MRotaryEmbedding, YaRNScalingMRotaryEmbedding, Ernie4_5_VLRotaryEmbedding,
apply_interleaved_rope, and Triton kernels for multimodal RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.rotary_embedding._base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding._utils import _apply_rotary_emb
from sglang.srt.layers.rotary_embedding._yarn import (
    _yarn_find_correction_range,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_npu

if TYPE_CHECKING:
    pass

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


@triton.jit
def _triton_mrope_forward_fused(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    positions_stride,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_stride
    k_ptr = k_ptr + pid * k_stride
    half_rd = rd // 2
    t = tl.load(positions_ptr + 0 * positions_stride + pid)
    h = tl.load(positions_ptr + 1 * positions_stride + pid)
    w = tl.load(positions_ptr + 2 * positions_stride + pid)
    t_cos = cos_sin_cache_ptr + t * rd
    h_cos = cos_sin_cache_ptr + h * rd
    w_cos = cos_sin_cache_ptr + w * rd
    t_sin = t_cos + half_rd
    h_sin = h_cos + half_rd
    w_sin = w_cos + half_rd
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row
    if is_neox_style:
        fhq = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fhk = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fqm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        fkm = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        q1 = tl.load(q_ptr + fhq, mask=fqm, other=0).to(sin_row.dtype)
        k1 = tl.load(k_ptr + fhk, mask=fkm, other=0).to(sin_row.dtype)
        shq = fhq + (rd // 2)
        shk = fhk + (rd // 2)
        q2 = tl.load(q_ptr + shq, mask=fqm, other=0).to(sin_row.dtype)
        k2 = tl.load(k_ptr + shk, mask=fkm, other=0).to(sin_row.dtype)
        tl.store(q_ptr + fhq, q1 * cos_row - q2 * sin_row, mask=fqm)
        tl.store(q_ptr + shq, q2 * cos_row + q1 * sin_row, mask=fqm)
        tl.store(k_ptr + fhk, k1 * cos_row - k2 * sin_row, mask=fkm)
        tl.store(k_ptr + shk, k2 * cos_row + k1 * sin_row, mask=fkm)
    else:
        bq = tl.arange(0, pad_n_qh)[:, None] * hd
        bk = tl.arange(0, pad_n_kh)[:, None] * hd
        ei = 2 * tl.arange(0, pad_hd // 2)[None, :]
        oi = ei + 1
        im = tl.arange(0, pad_hd // 2)[None, :] < (rd // 2)
        qm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & im
        km = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & im
        qe = tl.load(q_ptr + bq + ei, mask=qm, other=0).to(sin_row.dtype)
        qo = tl.load(q_ptr + bq + oi, mask=qm, other=0).to(sin_row.dtype)
        ke = tl.load(k_ptr + bk + ei, mask=km, other=0).to(sin_row.dtype)
        ko = tl.load(k_ptr + bk + oi, mask=km, other=0).to(sin_row.dtype)
        tl.store(q_ptr + bq + ei, qe * cos_row - qo * sin_row, mask=qm)
        tl.store(q_ptr + bq + oi, qo * cos_row + qe * sin_row, mask=qm)
        tl.store(k_ptr + bk + ei, ke * cos_row - ko * sin_row, mask=km)
        tl.store(k_ptr + bk + oi, ko * cos_row + ke * sin_row, mask=km)


def triton_mrope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    is_neox_style: bool,
) -> None:
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd = triton.next_power_of_2(head_size)
    _triton_mrope_forward_fused[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        is_neox_style,
    )


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
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        seq_len_k = key.shape[0]
        key_shape = key.shape
        key = key.view(seq_len_k, -1, self.head_size)
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
        from sglang.srt.layers.rotary_embedding._mrope_rope_index import get_rope_index

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
        from sglang.srt.layers.rotary_embedding._mrope_rope_index import (
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
        from sglang.srt.layers.rotary_embedding._mrope_rope_index import (
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
        from sglang.srt.layers.rotary_embedding._mrope_rope_index import (
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
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
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
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
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
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache


@triton.jit
def _triton_ernie45_rope_qk_fused(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0: tl.constexpr,
    k_stride0: tl.constexpr,
    pos_stride0: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    section_hw: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_stride0
    k_ptr = k_ptr + pid * k_stride0
    half_rd = rd // 2
    tpos = tl.load(positions_ptr + 0 * pos_stride0 + pid).to(tl.int32)
    hpos = tl.load(positions_ptr + 1 * pos_stride0 + pid).to(tl.int32)
    wpos = tl.load(positions_ptr + 2 * pos_stride0 + pid).to(tl.int32)
    ridx = tl.arange(0, pad_hd // 2)
    rmask = ridx < half_rd
    use_hw = ridx < section_hw
    use_h = (ridx & 1) == 0
    pos = tl.where(use_hw, tl.where(use_h, hpos, wpos), tpos)
    cos = tl.load(cos_sin_cache_ptr + pos * rd + ridx, mask=rmask, other=0.0)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rd + (ridx + half_rd), mask=rmask, other=0.0
    )
    if is_neox_style:
        qh = tl.arange(0, pad_n_qh)[:, None]
        kh = tl.arange(0, pad_n_kh)[:, None]
        d = tl.arange(0, pad_hd // 2)[None, :]
        qm = (qh < n_qh) & (d < half_rd)
        km = (kh < n_kh) & (d < half_rd)
        qo0 = qh * hd + d
        ko0 = kh * hd + d
        qo1 = qo0 + half_rd
        ko1 = ko0 + half_rd
        q0 = tl.load(q_ptr + qo0, mask=qm, other=0.0).to(cos.dtype)
        q1 = tl.load(q_ptr + qo1, mask=qm, other=0.0).to(cos.dtype)
        k0 = tl.load(k_ptr + ko0, mask=km, other=0.0).to(cos.dtype)
        k1 = tl.load(k_ptr + ko1, mask=km, other=0.0).to(cos.dtype)
        cb = cos[None, :]
        sb = sin[None, :]
        tl.store(q_ptr + qo0, q0 * cb - q1 * sb, mask=qm)
        tl.store(q_ptr + qo1, q1 * cb + q0 * sb, mask=qm)
        tl.store(k_ptr + ko0, k0 * cb - k1 * sb, mask=km)
        tl.store(k_ptr + ko1, k1 * cb + k0 * sb, mask=km)
    else:
        qh = tl.arange(0, pad_n_qh)[:, None]
        kh = tl.arange(0, pad_n_kh)[:, None]
        p = tl.arange(0, pad_hd // 2)[None, :]
        qm = (qh < n_qh) & (p < half_rd)
        km = (kh < n_kh) & (p < half_rd)
        even = 2 * p
        odd = even + 1
        qe = tl.load(q_ptr + qh * hd + even, mask=qm, other=0.0).to(cos.dtype)
        qo = tl.load(q_ptr + qh * hd + odd, mask=qm, other=0.0).to(cos.dtype)
        ke = tl.load(k_ptr + kh * hd + even, mask=km, other=0.0).to(cos.dtype)
        ko = tl.load(k_ptr + kh * hd + odd, mask=km, other=0.0).to(cos.dtype)
        cb = cos[None, :]
        sb = sin[None, :]
        tl.store(q_ptr + qh * hd + even, qe * cb - qo * sb, mask=qm)
        tl.store(q_ptr + qh * hd + odd, qo * cb + qe * sb, mask=qm)
        tl.store(k_ptr + kh * hd + even, ke * cb - ko * sb, mask=km)
        tl.store(k_ptr + kh * hd + odd, ko * cb + ke * sb, mask=km)


def triton_ernie45_rope_fused_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: list,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    rd = rotary_dim
    section_h, section_w, section_t = mrope_section
    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"
    assert section_h + section_w + section_t == rd // 2
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd = triton.next_power_of_2(head_size)
    num_warps = 4 if (pad_n_qh * pad_hd) <= 8192 else 8
    _triton_ernie45_rope_qk_fused[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh=n_qh,
        n_kh=n_kh,
        hd=head_size,
        rd=rd,
        pad_n_qh=pad_n_qh,
        pad_n_kh=pad_n_kh,
        pad_hd=pad_hd,
        section_hw=section_h + section_w,
        is_neox_style=is_neox_style,
        num_warps=num_warps,
    )


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
        self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(_apply_rotary_emb)

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
