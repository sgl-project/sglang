# Copyright 2026 SGLang Team
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
"""Triton store and gather-dequant kernels for the UltraQuant 4-bit KV cache.

One program handles one (token, head) pair: it rotates the key in registers,
quantizes key and value to FP4 E2M1 with UE8M0 group scales, and writes packed
codes and scale bytes into the four structure-of-arrays pool buffers.

``ultraquant_gather_dequant`` materializes cached slots back into contiguous
fp16/bf16 for dense prefill. The device helpers are shared with the UltraQuant
attention kernels. See ``sglang.srt.layers.quantization.ultraquant_tensor`` for
the format and the PyTorch reference.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.ultraquant_tensor import (
    CONSTANT_C,
    E2M1_MIDPOINTS,
    E2M1_VALUES,
    E2M1_ZERO_LEVEL_INDEX,
    GROUP_SIZE,
    UE8M0_BIAS,
    UE8M0_MAX_EXP,
    UE8M0_MIN_EXP,
    code_bytes,
    n_groups,
)

# Compile-time immediates: the store kernel is latency bound.
_MIDPOINTS = tl.constexpr(tuple(float(m) for m in E2M1_MIDPOINTS))
_NUM_MIDPOINTS = tl.constexpr(len(E2M1_MIDPOINTS))
_ZERO_LEVEL = tl.constexpr(E2M1_ZERO_LEVEL_INDEX)
_MAX_CODE = tl.constexpr(len(E2M1_VALUES) - 1)
_UE8M0_BIAS = tl.constexpr(UE8M0_BIAS)
_UE8M0_MIN_EXP = tl.constexpr(UE8M0_MIN_EXP)
_UE8M0_MAX_EXP = tl.constexpr(UE8M0_MAX_EXP)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_GATHER_OUT_DTYPES = (*_SUPPORTED_DTYPES, torch.float8_e4m3fn)


def check_pool_buffers(
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    head_num: int,
    head_dim: int,
) -> None:
    """Check the pool buffers against the layout the kernels address.

    K and V are addressed with one set of strides per buffer kind, as the pool
    allocates them.
    """
    num_slots = k_code_buffer.shape[0]
    for name, buf, row in (
        ("k_code_buffer", k_code_buffer, code_bytes(head_dim)),
        ("v_code_buffer", v_code_buffer, code_bytes(head_dim)),
        ("k_scale_buffer", k_scale_buffer, n_groups(head_dim)),
        ("v_scale_buffer", v_scale_buffer, n_groups(head_dim)),
    ):
        shape = (num_slots, head_num, row)
        if buf.dtype != torch.uint8 or buf.shape != shape or buf.stride(-1) != 1:
            raise ValueError(
                f"{name} must be a uint8 {list(shape)} tensor with unit last "
                f"stride, got {buf.dtype} {list(buf.shape)} strides {buf.stride()}"
            )
    if (
        k_code_buffer.stride() != v_code_buffer.stride()
        or k_scale_buffer.stride() != v_scale_buffer.stride()
    ):
        raise ValueError("UltraQuant K and V buffers must share strides")


def _log2_head_dim(head_dim: int) -> int:
    """Butterfly stage count of the Hadamard rotation, which needs a power of two."""
    if head_dim & (head_dim - 1):
        raise ValueError(
            f"Hadamard rotation requires a power-of-two head_dim, got {head_dim}"
        )
    return head_dim.bit_length() - 1


@triton.jit
def hadamard_rotate(x, offsets, LOG2_D: tl.constexpr):
    """Orthonormal Walsh-Hadamard transform of ``x`` held in registers.

    Equivalent to ``x @ H.T`` for the normalized Sylvester matrix built by
    ``ultraquant_tensor.hadamard_matrix``, in ``log2(D)`` butterfly stages
    instead of a D-by-D matmul. ``offsets`` is ``tl.arange(0, D)``.
    """
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    for stage in tl.static_range(LOG2_D):
        bit = 1 << stage
        partner = tl.gather(x, (offsets ^ bit).to(tl.int32), 0)
        # Lanes whose stage bit is clear compute the sum, the others the
        # difference; together that is one butterfly stage.
        sign = tl.where((offsets & bit) == 0, 1.0, -1.0).to(tl.float32)
        x = (sign * x + partner) * INV_SQRT2
    return x


@triton.jit
def quantize_fp4_ue8m0(
    x,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    CONSTANT_C_V: tl.constexpr,
):
    """Quantize ``[HEAD_DIM]`` fp32 to packed FP4 codes and UE8M0 scale bytes.

    Returns ``([HEAD_DIM // 2] uint8, [N_GROUPS_C] uint8)``. Mirrors
    ``UltraQuantKVQuantizeUtil.batched_quantize`` bit for bit, including the
    exponent clamp and the zero-group sentinel for zero or non-finite scales.
    """
    grouped = tl.reshape(x, [N_GROUPS_C, GROUP_SIZE_C])
    absmax = tl.max(tl.abs(grouped), axis=1)

    # UE8M0 snap: exponent = round(log2(c * absmax)), rounding half up.
    raw = absmax * CONSTANT_C_V
    is_zero = ~(raw > 0.0) | (raw == float("inf"))
    exponent = tl.cast(tl.floor(tl.log2(tl.where(is_zero, 1.0, raw)) + 0.5), tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, _UE8M0_MIN_EXP), _UE8M0_MAX_EXP)
    divisor = tl.where(is_zero, 1.0, tl.exp2(tl.cast(exponent, tl.float32)))

    # Bucketize against the midpoints between consecutive E2M1 levels. The
    # strict ``>`` reproduces torch.bucketize(right=False) used by the
    # reference, so ties round the same way.
    normalized = grouped / divisor[:, None]
    level = tl.zeros([N_GROUPS_C, GROUP_SIZE_C], dtype=tl.int32)
    for i in tl.static_range(_NUM_MIDPOINTS):
        level += tl.where(normalized > _MIDPOINTS.value[i], 1, 0)
    level = tl.where(is_zero[:, None], _ZERO_LEVEL, level)

    # Ascending level index to E2M1 bit pattern; see
    # ultraquant_tensor.sorted_index_to_e2m1_bits.
    bits = tl.where(level < _ZERO_LEVEL, _MAX_CODE - level, level - _ZERO_LEVEL)

    # Pack low nibble first, matching ultraquant_tensor.pack_nibbles.
    pairs = tl.reshape(tl.reshape(bits, [HEAD_DIM]), [HEAD_DIM // 2, 2])
    shifts = tl.arange(0, 2) * 4
    codes = tl.sum(pairs << shifts[None, :], axis=1).to(tl.uint8)

    # The clamp above keeps the biased exponent inside [1, 254], so byte 0 is
    # unambiguously the zero sentinel.
    scale_bytes = tl.where(is_zero, 0, exponent + _UE8M0_BIAS).to(tl.uint8)
    return codes, scale_bytes


@triton.jit
def e2m1_code_to_value(codes):
    """Decode E2M1 nibbles (int32, sign in bit 3) to fp32. Shape-generic.

    Builds the fp32 bit pattern directly: magnitude codes 2..7 are
    ``2**(e - 1) * (1 + m / 2)`` and code 1 is the subnormal 0.5.
    """
    mag = codes & 7
    exponent = mag >> 1
    mantissa = tl.where(exponent != 0, mag & 1, 0)
    bits = ((126 + exponent) << 23) | (mantissa << 22)
    values = tl.where(mag == 0, 0.0, bits.to(tl.float32, bitcast=True))
    return tl.where((codes & 8) != 0, -values, values)


@triton.jit
def ue8m0_byte_to_scale(scale_bytes):
    """Decode UE8M0 bytes to fp32; byte 0 is the exact-zero sentinel."""
    return tl.where(
        scale_bytes == 0,
        0.0,
        tl.exp2(tl.cast(scale_bytes.to(tl.int32) - _UE8M0_BIAS, tl.float32)),
    )


@triton.jit
def dequant_rows(
    codes,
    scale_bytes,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
):
    """Dequantize ``[BLOCK_N, HEAD_DIM // 2]`` packed codes and their
    ``[BLOCK_N, HEAD_DIM // GROUP_SIZE_C]`` scale bytes to fp32 rows."""
    packed = codes.to(tl.int32)
    values = tl.join(e2m1_code_to_value(packed & 0xF), e2m1_code_to_value(packed >> 4))
    values = tl.reshape(values, [BLOCK_N, HEAD_DIM // GROUP_SIZE_C, GROUP_SIZE_C])
    scales = ue8m0_byte_to_scale(scale_bytes)
    return tl.reshape(values * scales[:, :, None], [BLOCK_N, HEAD_DIM])


@triton.jit
def _ultraquant_store_kernel(
    key_ptr,
    value_ptr,
    k_code_ptr,
    k_scale_ptr,
    v_code_ptr,
    v_scale_ptr,
    loc_ptr,
    stride_key_t: tl.int64,
    stride_key_h: tl.int64,
    stride_value_t: tl.int64,
    stride_value_h: tl.int64,
    stride_code_s: tl.int64,
    stride_code_h: tl.int64,
    stride_scale_s: tl.int64,
    stride_scale_h: tl.int64,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    CONSTANT_C_V: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // HEAD_NUM
    head = pid % HEAD_NUM

    slot = tl.load(loc_ptr + token).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_code = tl.arange(0, HEAD_DIM // 2)
    offs_group = tl.arange(0, N_GROUPS_C)

    code_base = slot * stride_code_s + head * stride_code_h
    scale_base = slot * stride_scale_s + head * stride_scale_h

    # Keys are rotated before quantization, values are stored in the raw basis.
    key = tl.load(key_ptr + token * stride_key_t + head * stride_key_h + offs_d)
    key = hadamard_rotate(key.to(tl.float32), offs_d, LOG2_D)
    k_codes, k_scales = quantize_fp4_ue8m0(
        key, HEAD_DIM, GROUP_SIZE_C, N_GROUPS_C, CONSTANT_C_V
    )
    tl.store(k_code_ptr + code_base + offs_code, k_codes)
    tl.store(k_scale_ptr + scale_base + offs_group, k_scales)

    value = tl.load(value_ptr + token * stride_value_t + head * stride_value_h + offs_d)
    v_codes, v_scales = quantize_fp4_ue8m0(
        value.to(tl.float32), HEAD_DIM, GROUP_SIZE_C, N_GROUPS_C, CONSTANT_C_V
    )
    tl.store(v_code_ptr + code_base + offs_code, v_codes)
    tl.store(v_scale_ptr + scale_base + offs_group, v_scales)


@triton.jit
def _ultraquant_rotate_kernel(
    x_ptr,
    out_ptr,
    stride_xt: tl.int64,
    stride_xh: tl.int64,
    stride_ot: tl.int64,
    stride_oh: tl.int64,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // HEAD_NUM
    head = pid % HEAD_NUM

    offs_d = tl.arange(0, HEAD_DIM)
    x = tl.load(x_ptr + token * stride_xt + head * stride_xh + offs_d).to(tl.float32)
    x = hadamard_rotate(x, offs_d, LOG2_D)
    tl.store(
        out_ptr + token * stride_ot + head * stride_oh + offs_d,
        x.to(out_ptr.dtype.element_ty),
    )


def ultraquant_rotate(x: torch.Tensor, out: Optional[torch.Tensor] = None):
    """Hadamard-rotate the last dimension of ``[num_tokens, head_num, head_dim]``.

    Runs the same butterfly the store kernel applies to keys, so queries land
    in exactly the same basis. The butterfly runs in fp32 and rounds once into
    ``out``'s dtype; a float8_e4m3fn ``out`` saturates at +-448.
    """
    if x.ndim != 3:
        raise ValueError(f"ultraquant_rotate expects a 3-D tensor, got {x.shape}")
    num_tokens, head_num, head_dim = x.shape
    log2_d = _log2_head_dim(head_dim)

    if out is None:
        out = torch.empty_like(x)
    if num_tokens == 0:
        return out

    _ultraquant_rotate_kernel[(num_tokens * head_num,)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        LOG2_D=log2_d,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _ultraquant_gather_dequant_kernel(
    K_Code,
    K_Scale,
    V_Code,
    V_Scale,
    Kv_indices,
    K_out,
    V_out,
    stride_code_s: tl.int64,
    stride_code_h: tl.int64,
    stride_scale_s: tl.int64,
    stride_scale_h: tl.int64,
    stride_ko_t: tl.int64,
    stride_ko_h: tl.int64,
    stride_vo_t: tl.int64,
    stride_vo_h: tl.int64,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = pid // HEAD_NUM
    cur_head = pid % HEAD_NUM

    slot = tl.load(Kv_indices + tok).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    byte_off = offs_d // 2
    shift = (offs_d % 2) * 4
    group_off = offs_d // GROUP_SIZE_C

    code_base = slot * stride_code_s + cur_head * stride_code_h
    scale_base = slot * stride_scale_s + cur_head * stride_scale_h

    k_codes = (tl.load(K_Code + code_base + byte_off).to(tl.int32) >> shift) & 0xF
    k_scale_byte = tl.load(K_Scale + scale_base + group_off)
    k = e2m1_code_to_value(k_codes) * ue8m0_byte_to_scale(k_scale_byte)

    v_codes = (tl.load(V_Code + code_base + byte_off).to(tl.int32) >> shift) & 0xF
    v_scale_byte = tl.load(V_Scale + scale_base + group_off)
    v = e2m1_code_to_value(v_codes) * ue8m0_byte_to_scale(v_scale_byte)

    if K_out.dtype.element_ty == tl.float8e4nv:
        # E2M1 times a power of two is exact in E4M3 up to its +-448 limit.
        k = tl.clamp(k, -448.0, 448.0)
        v = tl.clamp(v, -448.0, 448.0)

    tl.store(
        K_out + tok * stride_ko_t + cur_head * stride_ko_h + offs_d,
        k.to(K_out.dtype.element_ty),
    )
    tl.store(
        V_out + tok * stride_vo_t + cur_head * stride_vo_h + offs_d,
        v.to(V_out.dtype.element_ty),
    )


def ultraquant_gather_dequant(
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    """Gather the slots in ``kv_indices`` and dequantize them into ``k_out``/``v_out``.

    The outputs are ``[num_tokens, head_num, head_dim]``, so a ragged run of
    slots becomes the contiguous layout a varlen attention kernel expects and
    ``kv_indptr`` carries over unchanged.

    Keys come back Hadamard-rotated, as stored.
    """
    num_tokens, head_num, head_dim = k_out.shape

    if v_out.shape != k_out.shape:
        raise ValueError(
            f"ultraquant_gather_dequant k/v output shape mismatch: "
            f"{k_out.shape} vs {v_out.shape}"
        )
    if v_out.dtype != k_out.dtype:
        raise ValueError(
            f"ultraquant_gather_dequant k/v output dtype mismatch: "
            f"{k_out.dtype} vs {v_out.dtype}"
        )
    if k_out.dtype not in _GATHER_OUT_DTYPES:
        raise ValueError(
            f"ultraquant_gather_dequant expects one of {_GATHER_OUT_DTYPES}, "
            f"got {k_out.dtype}"
        )
    if kv_indices.shape[0] < num_tokens:
        raise ValueError(
            f"ultraquant_gather_dequant got {kv_indices.shape[0]} indices for "
            f"{num_tokens} tokens"
        )
    check_pool_buffers(
        k_code_buffer, k_scale_buffer, v_code_buffer, v_scale_buffer, head_num, head_dim
    )

    if num_tokens == 0:
        return

    _ultraquant_gather_dequant_kernel[(num_tokens * head_num,)](
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        kv_indices,
        k_out,
        v_out,
        k_code_buffer.stride(0),
        k_code_buffer.stride(1),
        k_scale_buffer.stride(0),
        k_scale_buffer.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        GROUP_SIZE_C=GROUP_SIZE,
        num_warps=4,
        num_stages=1,
    )


def ultraquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    """Quantize ``key``/``value`` into the UltraQuant pool buffers at ``loc``.

    ``key`` and ``value`` are ``[num_tokens, head_num, head_dim]``. The four
    pool buffers are ``[num_slots, head_num, head_dim // 2]`` for codes and
    ``[num_slots, head_num, head_dim // GROUP_SIZE]`` for scales, all uint8.
    ``loc`` holds one destination slot per token.
    """
    if key.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"ultraquant_store expects one of {_SUPPORTED_DTYPES}, got {key.dtype}"
        )
    if value.dtype != key.dtype:
        raise ValueError(
            f"ultraquant_store key/value dtype mismatch: {key.dtype} vs {value.dtype}"
        )
    if value.shape != key.shape:
        raise ValueError(
            f"ultraquant_store key/value shape mismatch: {key.shape} vs {value.shape}"
        )

    num_tokens, head_num, head_dim = key.shape
    log2_d = _log2_head_dim(head_dim)
    if loc.shape[0] != num_tokens:
        raise ValueError(
            f"ultraquant_store got {loc.shape[0]} locations for {num_tokens} tokens"
        )
    check_pool_buffers(
        k_code_buffer, k_scale_buffer, v_code_buffer, v_scale_buffer, head_num, head_dim
    )

    if num_tokens == 0:
        return

    _ultraquant_store_kernel[(num_tokens * head_num,)](
        key,
        value,
        k_code_buffer,
        k_scale_buffer,
        v_code_buffer,
        v_scale_buffer,
        loc,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_code_buffer.stride(0),
        k_code_buffer.stride(1),
        k_scale_buffer.stride(0),
        k_scale_buffer.stride(1),
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        LOG2_D=log2_d,
        GROUP_SIZE_C=GROUP_SIZE,
        N_GROUPS_C=n_groups(head_dim),
        CONSTANT_C_V=CONSTANT_C,
        num_warps=4,
        num_stages=1,
    )
