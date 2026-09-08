"""Attention metadata and 2D RoPE helpers for HunyuanImage-3.

Ported from the official HunyuanImage-3 model repository
(`modeling_hunyuan_image_3.py`).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from einops import repeat

import torch
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@dataclass
class HunYuanImageAttentionMeta:
    query_lens: List[int]
    seq_lens: List[int]
    num_image_tokens: int
    first_step: bool


def create_hunyuan_image_attention_meta(
    attention_mask: torch.Tensor, num_image_tokens: int, first_step: bool
) -> HunYuanImageAttentionMeta:
    b, _, q_len1, seq_len = attention_mask.shape
    return HunYuanImageAttentionMeta(
        query_lens=[q_len1] * b,
        seq_lens=[seq_len] * b,
        num_image_tokens=num_image_tokens,
        first_step=first_step,
    )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    if cos.dim() == 3:
        cos = cos[0]
        sin = sin[0]
    ro_dim = cos.shape[-1] * 2
    cos = repeat(cos, "... d -> ... 1 (2 d)")
    sin = repeat(sin, "... d -> ... 1 (2 d)")
    return torch.cat(
        [
            q[..., :ro_dim] * cos + rotate_half(q[..., :ro_dim]) * sin,
            q[..., ro_dim:],
        ],
        dim=-1,
    ), torch.cat(
        [
            k[..., :ro_dim] * cos + rotate_half(k[..., :ro_dim]) * sin,
            k[..., ro_dim:],
        ],
        dim=-1,
    )


class HunYuanRotary2DEmbedder:
    """2D RoPE wrapper for HunyuanImage-3 attention."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states_shape[-1])

        cos, sin = custom_pos_emb
        bs, q_len = cos.shape[0], cos.shape[1]

        assert hidden_states.shape[0] == bs * q_len, f"{hidden_states.shape[0]} != {bs * q_len}"

        # [B*L, H*D] -> [B, L, H, D] for apply_rotary_pos_emb
        q = q.reshape(bs, q_len, self.num_heads, self.head_dim)
        k = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q.to(torch.float32), k.to(torch.float32), cos, sin)

        # Restore packed shape in bfloat16
        q = q.reshape(hidden_states.shape[0], self.num_heads * self.head_dim).to(torch.bfloat16)
        k = k.reshape(hidden_states.shape[0], self.num_kv_heads * self.head_dim).to(torch.bfloat16)
        hidden_states = hidden_states.reshape(hidden_states_shape)
        return q, k


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
):
    """Build 2D RoPE cos/sin tables ([seq_len, n_elem]); text uses 1D,
    image positions use 2D grid indices.
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d, 2]

    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L, device=device))
                x_sections.append(torch.arange(last_pos, L, device=device))
            elif h is None:
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                continue
            else:
                pass
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            # linspace(a, b, n+1)[:n] == arange(a, a+n); kept as-is for
            # bit-parity with the official implementation
            y_axis = torch.linspace(
                beta_y, beta_y + h, h + 1, dtype=torch.float32, device=device
            )[:h]
            x_axis = torch.linspace(
                beta_x, beta_x + w, w + 1, dtype=torch.float32, device=device
            )[:w]
            grid = torch.stack(
                torch.meshgrid(y_axis, x_axis, indexing="ij"), dim=0
            ).reshape(2, -1)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            last_pos = L + w * h
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)  # [seq_len, 1, 2]

    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2)
    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)
    return cos, sin


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
):
    """Build batched 2D RoPE cos/sin tables."""
    cos_list, sin_list = [], []
    if image_infos is None:
        image_infos = [None]
    for image_info in image_infos:
        cos, sin = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device, base=base,
        )
        cos_list.append(cos)
        sin_list.append(sin)

    return torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0)


class CachedRoPE:
    """Caches 2D RoPE cos/sin tables across diffusion steps."""

    def __init__(self, rope_theta: float, head_dim: int, rope_type: str = "2d"):
        # "2d" and "default" are the official spellings of the same scheme;
        # anything else in a checkpoint config would silently get wrong RoPE.
        if rope_type not in ("2d", "default"):
            raise NotImplementedError(f"rope_type `{rope_type}` not supported")
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_image_info = None

    def __call__(self, seq_len, device, rope_image_info=None):
        if (self.seq_len != seq_len) or (rope_image_info is not None and self.rope_image_info != rope_image_info):
            self.cos_cache, self.sin_cache = build_batch_2d_rope(
                image_infos=rope_image_info,
                seq_len=seq_len,
                n_elem=self.head_dim,
                device=device,
                base=self.rope_theta,
            )
            self.seq_len = seq_len
            self.rope_image_info = rope_image_info

        return self.cos_cache, self.sin_cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ImageKVCacheManager:
    """Full attention over the text+image sequence (no KV caching); packed
    ``[tokens, heads, dim]`` inputs, 4-D bool mask.
    """

    def __call__(self, query, key, value, attn_metadata, attention_mask=None):
        total_tokens = query.shape[0]
        bs = len(attn_metadata.query_lens)
        q_len = total_tokens // bs

        head_num_per_rank = query.shape[1]
        kv_head_num_per_rank = key.shape[1]
        repeat_num = head_num_per_rank // kv_head_num_per_rank
        head_dim = query.shape[2]

        query = query.reshape(bs, q_len, head_num_per_rank, head_dim)
        key = key.reshape(bs, q_len, kv_head_num_per_rank, head_dim)
        value = value.reshape(bs, q_len, kv_head_num_per_rank, head_dim)

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        key = repeat_kv(key, repeat_num)
        value = repeat_kv(value, repeat_num)

        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask.contiguous(),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / (query.shape[-1] ** 0.5),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(total_tokens, head_num_per_rank, head_dim)
        return attn_output
