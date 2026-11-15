"""Attention metadata and 2D RoPE helpers for HunyuanImage-3.

Ported from the official repository (``modeling_hunyuan_image_3.py``).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def _timestep_freqs(half, max_period, device):
    # host-built as in the reference, uploaded once per device
    return torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=device)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = _timestep_freqs(half, max_period, t.device)
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
    # Length of the step-invariant prefix whose K/V is cached; None disables it.
    prefix_len: Optional[int] = None
    # Separates cached prefixes of CFG halves that share one attention module.
    cache_slot: int = 0
    # Per-row (start, end, causal) segments from build_attention_segments;
    # None keeps the masked SDPA path.
    attn_segments: Optional[tuple] = None


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
    """Apply 2D RoPE with the official tiled angle layout.

    ``cos``/``sin`` from ``build_2d_rope`` are already expanded to full
    head_dim ([seq, head_dim] or [batch, seq, head_dim]); batched tables
    broadcast per row so requests with different layouts keep their own
    positions. ``q``/``k`` are [batch, seq, heads, head_dim].
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(2)  # broadcast over heads: [b, s, 1, head_dim]
    sin = sin.unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class HunYuanRotary2DEmbedder:
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

        assert hidden_states.shape[0] == bs * q_len, (
            f"{hidden_states.shape[0]} != {bs * q_len}"
        )

        # [B*L, H*D] -> [B, L, H, D] for apply_rotary_pos_emb
        q = q.reshape(bs, q_len, self.num_heads, self.head_dim)
        k = k.reshape(bs, q_len, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q.to(torch.float32), k.to(torch.float32), cos, sin)

        q = q.reshape(hidden_states.shape[0], self.num_heads * self.head_dim).to(
            torch.bfloat16
        )
        k = k.reshape(hidden_states.shape[0], self.num_kv_heads * self.head_dim).to(
            torch.bfloat16
        )
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
                y_sections.append(
                    torch.arange(sec_slice.start, sec_slice.stop, device=device)
                )
                x_sections.append(
                    torch.arange(sec_slice.start, sec_slice.stop, device=device)
                )
                continue
            else:
                pass
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            # linspace(a, b, n+1)[:n] == arange(a, a+n); kept for bit-parity
            # with the official implementation
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
        y_sections.append(
            torch.arange(last_pos, sample_seq_lens[sample_id], device=device)
        )
        x_sections.append(
            torch.arange(last_pos, sample_seq_lens[sample_id], device=device)
        )

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = (
        torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)
    )  # [seq_len, 1, 2]

    # Tile the half-dim angles up to head_dim (official layout): rotate_half
    # pairs dim i with dim i + n_elem/2, both carrying the SAME angle;
    # interleaved expansion pairs mismatched frequencies and garbles positions.
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)
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
    cos_list, sin_list = [], []
    if image_infos is None:
        image_infos = [None]
    for image_info in image_infos:
        cos, sin = build_2d_rope(
            seq_len,
            n_elem,
            image_infos=image_info,
            device=device,
            base=base,
        )
        cos_list.append(cos)
        sin_list.append(sin)

    return torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0)


class CachedRoPE:
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
        if (self.seq_len != seq_len) or (
            rope_image_info is not None and self.rope_image_info != rope_image_info
        ):
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


def build_attention_segments(seq_len: int, image_slices: List[slice]):
    """Split ``[0, seq_len)`` into ``(start, end, causal)`` segments that
    reproduce the causal mask with bidirectional image blocks.

    A row in an image block ``[s, e)`` sees keys ``[0, e)``; any other row ``i``
    sees keys ``[0, i]``. Returns None when blocks overlap, which keeps the
    masked path.
    """
    segments = []
    pos = 0
    for block in sorted(image_slices, key=lambda s: s.start):
        if block.start < pos:
            return None
        if block.start > pos:
            segments.append((pos, block.start, True))
        segments.append((block.start, block.stop, False))
        pos = block.stop
    if pos < seq_len:
        segments.append((pos, seq_len, True))
    return tuple(segments)


def segmented_attention(query, key, value, segments, attend):
    """Mask-free attention over ``build_attention_segments`` segments.

    ``query`` is ``[bs, q_len, heads, dim]`` for the last ``q_len`` of the
    ``k_len`` positions in ``key``/``value`` (``[bs, k_len, kv_heads, dim]``).
    ``attend(q, k, v, causal)`` takes ``[n, heads, dim]`` with the causal mask
    aligned to the bottom right, so queries ``[a, e)`` against keys ``[0, e)``
    see exactly ``[0, i]``.
    """
    bs, q_len = query.shape[:2]
    k_len = key.shape[1]
    q_offset = k_len - q_len
    out = torch.empty_like(query)
    for b in range(bs):
        if segments[b][-1][1] != k_len:
            raise ValueError(
                f"segments end at {segments[b][-1][1]}, but there are {k_len} keys"
            )
        for start, end, causal in segments[b]:
            if end <= q_offset:
                continue
            start = max(start, q_offset)
            out[b, start - q_offset : end - q_offset] = attend(
                query[b, start - q_offset : end - q_offset],
                key[b, :end],
                value[b, :end],
                causal,
            )
    return out


@lru_cache(maxsize=1)
def _hopper_flash_attn_varlen():
    # FA3 runs on sm90 only; elsewhere the masked SDPA path is kept.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return None
    try:
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_varlen_func,
        )
    except ImportError:
        return None
    return flash_attn_varlen_func


@lru_cache(maxsize=256)
def _cu_seqlens(n: int, device: torch.device) -> torch.Tensor:
    # one upload per length and device; callers must not mutate the result
    return torch.tensor([0, n], dtype=torch.int32, device=device)


def _flash_attend(q, k, v, causal):
    nq, nk = q.shape[0], k.shape[0]
    return _hopper_flash_attn_varlen()(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=_cu_seqlens(nq, q.device),
        cu_seqlens_k=_cu_seqlens(nk, q.device),
        max_seqlen_q=nq,
        max_seqlen_k=nk,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=causal,
    )


def _masked_sdpa(query, key, value, attention_mask):
    # [bs, seq, heads, dim] in and out; GQA K/V are expanded for SDPA.
    repeat_num = query.shape[2] // key.shape[2]
    query = query.transpose(1, 2).contiguous()
    key = repeat_kv(key.transpose(1, 2).contiguous(), repeat_num)
    value = repeat_kv(value.transpose(1, 2).contiguous(), repeat_num)
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask.contiguous(),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / (query.shape[-1] ** 0.5),
    )
    return attn_output.transpose(1, 2).contiguous()


class ImageKVCacheManager:
    """Attention over the text+image sequence; packed ``[tokens, heads, dim]``
    inputs, 4-D bool mask.

    With ``prefix_len`` set, the step-invariant prefix K/V is cached per
    ``cache_slot`` on the first step. Later steps pass only the suffix tokens as
    queries and attend to ``[cached prefix K/V, suffix K/V]``; the mask rows
    must then cover just the suffix, with all prefix columns. With
    ``attn_segments`` set on Hopper, attention runs as FA3 calls per segment
    instead of SDPA with the dense mask.
    """

    def __init__(self):
        self._prefix_kv: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def clear(self):
        """Release cached prefix storage after a generation finishes."""
        self._prefix_kv.clear()

    def __call__(self, query, key, value, attn_metadata, attention_mask=None):
        total_tokens = query.shape[0]
        bs = len(attn_metadata.query_lens)
        q_len = total_tokens // bs

        head_num_per_rank = query.shape[1]
        kv_head_num_per_rank = key.shape[1]
        head_dim = query.shape[2]

        query = query.reshape(bs, q_len, head_num_per_rank, head_dim)
        key = key.reshape(bs, q_len, kv_head_num_per_rank, head_dim)
        value = value.reshape(bs, q_len, kv_head_num_per_rank, head_dim)

        if attn_metadata.prefix_len is not None:
            key, value = self._prefix_kv_step(key, value, attn_metadata)

        if (
            attn_metadata.attn_segments is not None
            and query.is_cuda
            and _hopper_flash_attn_varlen() is not None
        ):
            attn_output = segmented_attention(
                query, key, value, attn_metadata.attn_segments, _flash_attend
            )
        else:
            attn_output = _masked_sdpa(query, key, value, attention_mask)
        return attn_output.reshape(total_tokens, head_num_per_rank, head_dim)

    def _prefix_kv_step(self, key, value, attn_metadata):
        """Store the prefix K/V on the first step; afterwards prepend it to the
        suffix K/V. Inputs and outputs are ``[bs, seq, heads, dim]``.
        """
        slot = attn_metadata.cache_slot
        prefix_len = attn_metadata.prefix_len
        if attn_metadata.first_step:
            # Slicing a contiguous prefix can still retain the full-sequence
            # allocation (notably when bs == 1). Own compact storage so the
            # generation cache does not pin the step's full K/V tensors.
            self._prefix_kv[slot] = (
                key[:, :prefix_len].clone(),
                value[:, :prefix_len].clone(),
            )
            return key, value

        if slot not in self._prefix_kv:
            raise RuntimeError(f"no cached prefix K/V for cache_slot={slot}")
        prefix_k, prefix_v = self._prefix_kv[slot]
        return (
            torch.cat([prefix_k, key], dim=1),
            torch.cat([prefix_v, value], dim=1),
        )


def split_suffix_inputs(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    custom_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    prefix_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Keep only the rows after ``prefix_len``. ``hidden_states`` is packed
    ``[bs * seq, hidden]``, the mask is ``[bs, 1, seq, seq]``, and the RoPE
    tables are ``[bs, seq, head_dim]``.
    """
    bs, _, seq_len, _ = attention_mask.shape
    hidden = hidden_states.reshape(bs, seq_len, -1)[:, prefix_len:]
    cos, sin = custom_pos_emb
    return (
        hidden.reshape(-1, hidden.shape[-1]),
        attention_mask[:, :, prefix_len:, :],
        (cos[:, prefix_len:], sin[:, prefix_len:]),
    )


def pad_prefix_rows(suffix_out: torch.Tensor, bs: int, prefix_len: int) -> torch.Tensor:
    """Re-expand suffix outputs to full ``[bs * seq, hidden]`` with zero prefix
    rows; callers only read image positions, which all lie in the suffix.
    """
    out = suffix_out.reshape(bs, -1, suffix_out.shape[-1])
    pad = out.new_zeros(bs, prefix_len, out.shape[-1])
    return torch.cat([pad, out], dim=1).reshape(-1, out.shape[-1])
