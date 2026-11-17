import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def _as_float32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


_SUPPORTED_HEAD_DIMS = (4, 8, 16, 32, 64, 128)

_INV_FREQS = tl.constexpr(
    tuple(
        (
            _as_float32(10000.0 ** (-dim / supported_head_dim))
            for supported_head_dim in _SUPPORTED_HEAD_DIMS
            for dim in range(0, supported_head_dim, 4)
        )
    )
)


@triton.jit
def _rope_cos_sin(positions_ptr, token_index, pair, HEAD_DIM: tl.constexpr):
    if HEAD_DIM == 4:
        inv_freq_offset: tl.constexpr = 0
    elif HEAD_DIM == 8:
        inv_freq_offset: tl.constexpr = 1
    elif HEAD_DIM == 16:
        inv_freq_offset: tl.constexpr = 3
    elif HEAD_DIM == 32:
        inv_freq_offset: tl.constexpr = 7
    elif HEAD_DIM == 64:
        inv_freq_offset: tl.constexpr = 15
    elif HEAD_DIM == 128:
        inv_freq_offset: tl.constexpr = 31
    else:
        tl.static_assert(False, "unsupported head dimension")
    freq_index = pair % (HEAD_DIM // 4)
    inv_freq = tl.zeros(freq_index.shape, tl.float32)
    for index in tl.static_range(0, HEAD_DIM // 4):
        inv_freq = tl.where(
            freq_index == index, _INV_FREQS[inv_freq_offset + index], inv_freq
        )
    axis_offset = tl.where(pair < HEAD_DIM // 4, 1, 0)
    position = tl.load(
        positions_ptr + 2 * token_index + axis_offset.to(tl.int64),
        mask=pair < HEAD_DIM // 2,
        other=0,
    ).to(tl.float32)
    angle = position * inv_freq
    cos = libdevice.cos(angle)
    sin = libdevice.sin(angle)
    return (cos, sin)


@triton.jit
def _rotate_pair(even, odd, cos, sin):
    return (tl.fma(even, cos, -odd * sin), tl.fma(odd, cos, even * sin))


@triton.jit(
    do_not_specialize=["num_tokens"], do_not_specialize_on_alignment=["positions_ptr"]
)
def apply_rotary_qk_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    out_q_ptr,
    out_k_ptr,
    num_tokens: int,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= num_tokens:
        return
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)[:, None]
    pair = tl.arange(0, BLOCK_PAIRS)[None, :]
    mask = (head < NUM_HEADS) & (pair < HEAD_DIM // 2)
    token_index = token.to(tl.int64)
    head_index = head.to(tl.int64)
    pair_index = pair.to(tl.int64)
    pair_offset = (token_index * NUM_HEADS + head_index) * HEAD_DIM + 2 * pair_index
    cos, sin = _rope_cos_sin(positions_ptr, token_index, pair, HEAD_DIM)
    q_even = tl.load(q_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + pair_offset + 1, mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_ptr + pair_offset + 1, mask=mask, other=0.0).to(tl.float32)
    out_q_even, out_q_odd = _rotate_pair(q_even, q_odd, cos, sin)
    out_k_even, out_k_odd = _rotate_pair(k_even, k_odd, cos, sin)
    tl.store(out_q_ptr + pair_offset, out_q_even, mask=mask)
    tl.store(out_q_ptr + pair_offset + 1, out_q_odd, mask=mask)
    tl.store(out_k_ptr + pair_offset, out_k_even, mask=mask)
    tl.store(out_k_ptr + pair_offset + 1, out_k_odd, mask=mask)


def apply_rotary_qk(q, k, positions):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    num_tokens, num_heads, head_dim = q.shape
    block_heads = min(4, triton.next_power_of_2(num_heads))
    apply_rotary_qk_kernel[(num_tokens, triton.cdiv(num_heads, block_heads))](
        q,
        k,
        positions,
        out_q,
        out_k,
        num_tokens,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=triton.next_power_of_2(head_dim // 2),
        num_warps=4,
    )
    return out_q, out_k
