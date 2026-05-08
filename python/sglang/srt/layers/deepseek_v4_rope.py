import math
from functools import lru_cache
from typing import Optional

import tilelang
import torch
import triton
import triton.language as tl

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@triton.jit
def apply_rotary_emb_triton_kernel(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    rope_dim,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)

    if USE_POS:
        position = tl.load(positions_ptr + pid_batch)
    else:
        position = pid_batch

    if IS_3D:
        base_offset = pid_batch * stride_x_batch + pid_head * stride_x_head
    else:
        base_offset = pid_batch * stride_x_batch

    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    offs_x_real = base_offset + offs_pair * 2 * stride_x_dim
    offs_x_imag = base_offset + (offs_pair * 2 + 1) * stride_x_dim

    x_real = tl.load(x_ptr + offs_x_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_x_imag, mask=mask, other=0.0).to(tl.float32)

    offs_freq_real = position * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    offs_freq_imag = position * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim

    freq_real = tl.load(freqs_ptr + offs_freq_real, mask=mask, other=0.0)
    freq_imag = tl.load(freqs_ptr + offs_freq_imag, mask=mask, other=0.0)

    if IS_INVERSE:
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + offs_x_real, out_real, mask=mask)
    tl.store(x_ptr + offs_x_imag, out_imag, mask=mask)


def apply_rotary_emb_triton(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor:
    is_3d = x.ndim == 3

    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    BLOCK_SIZE = 128

    num_blocks_dim = triton.cdiv(rope_dim // 2, BLOCK_SIZE)
    grid = (batch_size, n_heads if is_3d else 1, num_blocks_dim)

    if positions is not None:
        assert positions.shape == (
            batch_size,
        ), f"positions shape {positions.shape} != ({batch_size},)"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            positions,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=True,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        assert (
            freqs_real.shape[0] == batch_size
        ), f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            None,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=False,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return x
