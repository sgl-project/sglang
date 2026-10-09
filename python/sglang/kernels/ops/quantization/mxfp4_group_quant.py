# MXF4 (e2m1) per-token-group-32 activation quantization for the grouped-GEMM
# DeepGEMM MoE path (W4A4 experiment, SGLANG_USE_DEEPGEMM_W4A4=1).
#
# Produces the tensor format accepted by deep_gemm's
# m_grouped_fp8_fp4_gemm_nt_contiguous with recipe_a=(1, 32):
#   q : (M, K // 2)   int8   — packed e2m1, low nibble = even element
#   sf: (M, K // 128) int32  — packed ue8m0 exponents, 4 per word, mn-major
#        (storage (K // 128, ceil(M / 4) * 4) contiguous, transposed view)
# and, via `silu_mul_quant_mxfp4_masked`, the same recipe for the
# m_grouped_fp8_fp4_gemm_nt_masked (per-expert padded) layout.
#
# The e2m1 / ue8m0 coding primitives are shared with the DSA indexer fp4
# K-cache quantizer (kernels/ops/attention/dsv4/fp4_indexer.py).

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code,
)


def _check_mxfp4_input(x: torch.Tensor, name: str) -> Tuple[int, int]:
    if x.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape={tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype=torch.bfloat16, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return x.shape


def _check_sm100(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        raise RuntimeError(
            "MXFP4 hardware conversion requires SM100 or newer, "
            f"got compute capability {major}.{minor}"
        )


def _check_mxfp4_masked_m(masked_m: torch.Tensor, num_experts: int) -> None:
    if masked_m.dim() != 1 or masked_m.shape[0] != num_experts:
        raise ValueError(
            f"masked_m must be a 1D tensor of {num_experts} expert row counts, "
            f"got shape={tuple(masked_m.shape)}"
        )
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must have dtype=torch.int32, got {masked_m.dtype}")
    if not masked_m.is_cuda:
        raise ValueError("masked_m must be a CUDA tensor")


def _next_power_of_two_multiple_of_128(value: int, max_value: int) -> int:
    # Triton tl.arange requires a power-of-two extent. Keep every block
    # group-aligned so the (1, 32) quantization layout remains valid.
    block = 1 << (value - 1).bit_length()
    return min(max_value, max(128, block))


@triton.jit
def _quant_mxfp4_group32_kernel(
    x_ptr,  # (M, K) bf16 row-major
    q_ptr,  # (M, K // 2) int8 row-major
    sf_ptr,  # (K // 128, M_al) int32, column per row
    sf_row_stride,
    K,
    BLOCK_K: tl.constexpr,  # elements per program, multiple of 128
):
    m = tl.program_id(0)
    kc = tl.program_id(1)

    offs = kc * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    # int64 row bases: M * K can exceed int32 and wrap into an illegal access.
    x_base = m.to(tl.int64) * K
    q_base = m.to(tl.int64) * (K // 2)
    x = tl.load(x_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 32
    # Per-group (1, 32) amax -> ue8m0 exponent -> scale.
    absx = tl.reshape(tl.abs(x), (NG, 32))
    amax = tl.max(absx, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)  # (NG,) int32, 1..254
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_K,))

    code = _fp4_e2m1_code(x / scale_b)  # (BLOCK_K,) uint8 nibble codes

    # Pack two e2m1 codes per byte: low nibble = even element.
    c2 = tl.reshape(code, (BLOCK_K // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = kc * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    # Mask the tail block: K need only be a multiple of 128, so the last
    # program can carry fewer than BLOCK_K valid elements.
    tl.store(q_ptr + q_base + q_offs, packed.to(tl.int8), mask=q_offs < (K // 2))

    # Pack four ue8m0 exponents per int32 scale word (byte b = group 4i+b).
    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = kc * (NG // 4) + tl.arange(0, NG // 4)
    tl.store(sf_ptr + w_offs * sf_row_stride + m, word, mask=w_offs < (K // 128))


def quant_mxfp4_group32(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize (M, K) bf16 activations to packed e2m1 + packed ue8m0 (1, 32)
    scales in the layout deep_gemm's contiguous fp4 GEMM expects."""
    M, K = _check_mxfp4_input(x, "x")
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")
    if K % 128 != 0:
        raise ValueError(f"K={K} must be a multiple of 128 for (1,32) fp4 packing")

    q = torch.empty((M, K // 2), device=x.device, dtype=torch.int8)
    kb4 = K // 128
    m_al = triton.cdiv(M, 4) * 4
    # zeros: the kernel only writes m < M, so the padding columns
    # [M, m_al) must not hold garbage — deep_gemm / ep_scatter read the
    # full m_al columns and garbage bytes decode to NaN ue8m0 scales.
    sf_storage = torch.zeros((kb4, m_al), device=x.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:M, :]  # (M, kb4), stride(-2) == 1

    if M == 0:
        return q, sf
    BLOCK_K = _next_power_of_two_multiple_of_128(K, 1024)
    grid = (M, triton.cdiv(K, BLOCK_K))
    _quant_mxfp4_group32_kernel[grid](
        x, q, sf_storage, sf_storage.stride(0), K, BLOCK_K=BLOCK_K
    )
    return q, sf


# ---------------------------------------------------------------------------
# v2: hardware-cvt based quantization + fused silu-mul-quant.
# The comparison-chain codegen is replaced by the SM100 `cvt.rn.satfinite
# .e2m1x2.f32` instruction. It rounds ties to even where the chain sends them
# toward zero (0.75 -> 0.5 vs 1.0, 1.75 -> 1.5 vs 2.0, 3.5 -> 3.0 vs 4.0), so
# the two encoders agree in magnitude to within one code index rather than bit
# for bit, and they may disagree on the sign of a zero (which decodes as 0
# either way). The ue8m0 scales are identical. Pinned by
# test_mxfp4_group_quant.py::TestQuantMxfp4Group32V2::test_agrees_with_v1.
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_code_hw(x):
    # One fp32 -> one e2m1 nibble via the hardware converter (both cvt inputs
    # are the same value, so the low nibble holds the signed code).
    r = tl.inline_asm_elementwise(
        asm="{\n.reg .b8 byte0;\ncvt.rn.satfinite.e2m1x2.f32 byte0, $1, $1;\ncvt.u32.u8 $0, byte0;\n}",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return (r & 0x0F).to(tl.uint8)


@triton.jit
def _quant_mxfp4_group32_kernel_v2(
    x_ptr,
    q_ptr,
    sf_ptr,
    sf_row_stride,
    K,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    kc = tl.program_id(1)
    offs = kc * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    # int64 row bases: M * K can exceed int32 and wrap into an illegal access.
    x_base = m.to(tl.int64) * K
    q_base = m.to(tl.int64) * (K // 2)
    x = tl.load(x_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 32
    absx = tl.reshape(tl.abs(x), (NG, 32))
    amax = tl.max(absx, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_K,))

    code = _fp4_code_hw(x / scale_b)

    c2 = tl.reshape(code, (BLOCK_K // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = kc * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + q_base + q_offs, packed.to(tl.int8), mask=q_offs < (K // 2))

    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = kc * (NG // 4) + tl.arange(0, NG // 4)
    # Mask the tail block: the last program may hold fewer than BLOCK_K
    # valid elements, i.e. fewer than NG // 4 valid scale words.
    tl.store(sf_ptr + w_offs * sf_row_stride + m, word, mask=w_offs < (K // 128))


def quant_mxfp4_group32_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = _check_mxfp4_input(x, "x")
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")
    if K % 128 != 0:
        raise ValueError(f"K={K} must be a multiple of 128")
    _check_sm100(x.device)
    q = torch.empty((M, K // 2), device=x.device, dtype=torch.int8)
    kb4 = K // 128
    m_al = triton.cdiv(M, 4) * 4
    # zeros, not empty: see the padding-columns note in quant_mxfp4_group32.
    sf_storage = torch.zeros((kb4, m_al), device=x.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:M, :]
    if M == 0:
        return q, sf
    BLOCK_K = _next_power_of_two_multiple_of_128(K, 2048)
    grid = (M, triton.cdiv(K, BLOCK_K))
    _quant_mxfp4_group32_kernel_v2[grid](
        x, q, sf_storage, sf_storage.stride(0), K, BLOCK_K=BLOCK_K
    )
    return q, sf


@triton.jit
def _silu_mul_quant_mxfp4_kernel(
    gateup_ptr,  # (T, 2H) bf16: [0, H) = gate, [H, 2H) = up
    q_ptr,  # (T, H // 2) int8
    sf_ptr,  # (H // 128, T_al) int32 storage
    sf_row_stride,
    H,
    SWIGLU_LIMIT: tl.constexpr,  # 0 disables the clamp
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    hc = tl.program_id(1)
    offs = hc * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    base = t * (2 * H).to(tl.int64)
    gate = tl.load(gateup_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gateup_ptr + base + H + offs, mask=mask, other=0.0).to(tl.float32)
    if SWIGLU_LIMIT > 0:
        gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    act = gate * tl.sigmoid(gate) * up

    NG: tl.constexpr = BLOCK_H // 32
    absa = tl.reshape(tl.abs(act), (NG, 32))
    amax = tl.max(absa, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_H,))

    code = _fp4_code_hw(act / scale_b)

    c2 = tl.reshape(code, (BLOCK_H // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = hc * (BLOCK_H // 2) + tl.arange(0, BLOCK_H // 2)
    # int64 row base, same as gateup's: T * (H // 2) can exceed int32.
    tl.store(
        q_ptr + t.to(tl.int64) * (H // 2) + q_offs,
        packed.to(tl.int8),
        mask=q_offs < (H // 2),
    )

    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = hc * (NG // 4) + tl.arange(0, NG // 4)
    # Mask the tail block: the last hc program may hold fewer than BLOCK_H
    # valid elements, i.e. fewer than NG // 4 valid scale words.
    tl.store(sf_ptr + w_offs * sf_row_stride + t, word, mask=w_offs < (H // 128))


def silu_mul_quant_mxfp4(
    gateup: torch.Tensor, swiglu_limit: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU-mul + (1, 32) e2m1/ue8m0 quantization of the MoE gateup
    output, matching the two-step (_legacy_silu_and_mul + quant_mxfp4_group32)
    semantics. swiglu_limit=10 applies the DSV4 gate/up clamps."""
    T, N = _check_mxfp4_input(gateup, "gateup")
    if N <= 0:
        raise ValueError(f"gateup hidden dimension must be positive, got N={N}")
    if N % 2 != 0:
        raise ValueError(f"gateup hidden dimension must be even, got N={N}")
    H = N // 2
    if H % 128 != 0:
        raise ValueError(f"H={H} must be a multiple of 128")
    _check_sm100(gateup.device)
    q = torch.empty((T, H // 2), device=gateup.device, dtype=torch.int8)
    kb4 = H // 128
    t_al = triton.cdiv(T, 4) * 4
    # zeros, not empty: see the padding-columns note in quant_mxfp4_group32.
    sf_storage = torch.zeros((kb4, t_al), device=gateup.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:T, :]
    if T == 0:
        return q, sf
    # Keep the float precision: int() would silently truncate e.g. 10.5 -> 10
    # and diverge from the non-fused activation semantics.
    limit = float(swiglu_limit) if swiglu_limit is not None else 0
    BLOCK_H = _next_power_of_two_multiple_of_128(H, 2048)
    grid = (T, triton.cdiv(H, BLOCK_H))
    _silu_mul_quant_mxfp4_kernel[grid](
        gateup,
        q,
        sf_storage,
        sf_storage.stride(0),
        H,
        SWIGLU_LIMIT=limit,
        BLOCK_H=BLOCK_H,
    )
    return q, sf


@triton.jit
def _silu_mul_quant_mxfp4_masked_kernel(
    gateup_ptr,  # (E, m_max, 2 * size_n) bf16: [0, size_n) = gate, [size_n, 2 * size_n) = up
    stride_gateup_e,  # only the two outer strides: the innermost is contiguous
    stride_gateup_t,
    q_ptr,  # (E, m_max, size_n // 2) int8
    stride_q_e,
    stride_q_t,
    sf_ptr,  # (E, size_n // 128, m_max) int32 storage, written MN-major
    stride_sf_e,
    stride_sf_w,
    stride_sf_m,
    masked_m_ptr,  # (E,) int32
    num_experts,
    size_n,
    SWIGLU_LIMIT: tl.constexpr,  # 0 disables the clamp
    BLOCK_H: tl.constexpr,
    E_PADDED: tl.constexpr,
):
    # Flat work dim rides axis 0 (x): num_real_tokens * topk can exceed the
    # 65535 grid.y/z limit. Row `work_id` of the padded (E, m_max) mask is
    # (expert, token) pair number `work_id`, in expert-major order.
    work_id = tl.program_id(0)
    hidden_dim_block_index = tl.program_id(1)

    e_off = tl.arange(0, E_PADDED)
    mm = tl.load(masked_m_ptr + e_off, mask=e_off < num_experts, other=0)
    incl = tl.cumsum(mm)
    total = tl.sum(mm)
    # The grid is sized for the worst case (every routed slot valid), so the
    # padding programs have to bail out rather than read a wrong expert.
    if work_id >= total:
        return
    excl = incl - mm  # first global slot of each expert
    owner = (excl <= work_id) & (work_id < incl)
    expert_id = tl.sum(tl.where(owner, e_off, 0))
    token_index = work_id - tl.sum(tl.where(owner, excl, 0))

    N_GROUPS: tl.constexpr = BLOCK_H // 32
    BLOCK_W: tl.constexpr = N_GROUPS // 4

    offs = hidden_dim_block_index * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs < size_n
    # int64 row base: E * m_max * 2 * size_n exceeds int32 on the served shapes.
    in_base = gateup_ptr + (
        expert_id.to(tl.int64) * stride_gateup_e
        + token_index.to(tl.int64) * stride_gateup_t
    )
    gate = tl.load(in_base + offs, mask=mask_h, other=0.0).to(tl.float32)
    up = tl.load(in_base + size_n + offs, mask=mask_h, other=0.0).to(tl.float32)
    if SWIGLU_LIMIT > 0:
        gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    act = gate * tl.sigmoid(gate) * up

    absa = tl.reshape(tl.abs(act), (N_GROUPS, 32))
    amax = tl.max(absa, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (N_GROUPS, 32)), (BLOCK_H,))

    code = _fp4_code_hw(act / scale_b)

    c2 = tl.reshape(code, (BLOCK_H // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = hidden_dim_block_index * (BLOCK_H // 2) + tl.arange(0, BLOCK_H // 2)
    q_base = expert_id.to(tl.int64) * stride_q_e + token_index.to(tl.int64) * stride_q_t
    # Mask the tail block: size_n need only be a multiple of 128, so the last
    # program can carry fewer than BLOCK_H valid elements.
    tl.store(q_ptr + q_base + q_offs, packed.to(tl.int8), mask=q_offs < (size_n // 2))

    e2d = tl.reshape(exp, (BLOCK_W, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = hidden_dim_block_index * BLOCK_W + tl.arange(0, BLOCK_W)
    sf_base = (
        expert_id.to(tl.int64) * stride_sf_e + token_index.to(tl.int64) * stride_sf_m
    )
    tl.store(
        sf_ptr + sf_base + w_offs * stride_sf_w,
        word,
        mask=w_offs < (size_n // 128),
    )


def silu_mul_quant_mxfp4_masked(
    gateup: torch.Tensor,
    masked_m: torch.Tensor,
    topk: int,
    num_real_tokens: int,
    swiglu_limit: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked (EP-MoE) variant of `silu_mul_quant_mxfp4`.

    ``gateup`` is the per-expert padded grouped-GEMM output
    ``(E, m_max, 2 * H)`` whose first ``masked_m[e]`` rows hold expert ``e``'s
    tokens. Returns ``(q, sf)`` with ``q`` int8 ``(E, m_max, H // 2)`` packed
    e2m1 and ``sf`` int32 ``(E, m_max, H // 128)`` packed ue8m0, i.e. the same
    ``(1, 32)`` recipe the contiguous path feeds the down GEMM, in the layout
    ``m_grouped_fp8_fp4_gemm_nt_masked`` consumes.

    ``num_real_tokens * topk`` only sizes the grid (an upper bound on
    ``masked_m.sum()``); the padding programs exit without reading anything.
    """
    if gateup.dim() != 3:
        raise ValueError(
            f"gateup must be a 3D (expert, padded-token, 2H) tensor, "
            f"got shape={tuple(gateup.shape)}"
        )
    if not gateup.is_cuda:
        raise ValueError("gateup must be a CUDA tensor")
    if gateup.dtype != torch.bfloat16:
        raise ValueError(f"gateup must have dtype=torch.bfloat16, got {gateup.dtype}")
    if not gateup.is_contiguous():
        raise ValueError("gateup must be contiguous")
    num_experts, m_max, size_n_2 = gateup.shape
    if size_n_2 <= 0 or size_n_2 % 2 != 0:
        raise ValueError(
            f"gateup hidden dimension must be even and positive, got {size_n_2}"
        )
    size_n = size_n_2 // 2
    if size_n % 128 != 0:
        raise ValueError(f"down activation width {size_n} must be a multiple of 128")
    _check_mxfp4_masked_m(masked_m, num_experts)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got topk={topk}")
    _check_sm100(gateup.device)

    q = torch.empty(
        (num_experts, m_max, size_n // 2), device=gateup.device, dtype=torch.int8
    )
    # empty, not zeros: unlike the contiguous path, MN-major padded rows are
    # never read (the masked GEMM only touches rows < masked_m of each expert).
    sf_storage = torch.empty(
        (num_experts, size_n // 128, m_max), device=gateup.device, dtype=torch.int32
    )
    if m_max == 0 or num_real_tokens == 0:
        return q, sf_storage.transpose(1, 2)

    # Keep the float precision: int() would silently truncate e.g. 10.5 -> 10
    # and diverge from the non-fused activation semantics.
    limit = float(swiglu_limit) if swiglu_limit is not None else 0
    BLOCK_H = _next_power_of_two_multiple_of_128(size_n, 2048)
    grid = (num_real_tokens * topk, triton.cdiv(size_n, BLOCK_H))
    _silu_mul_quant_mxfp4_masked_kernel[grid](
        gateup,
        gateup.stride(0),
        gateup.stride(1),
        q,
        q.stride(0),
        q.stride(1),
        sf_storage,
        *sf_storage.stride(),
        masked_m,
        num_experts,
        size_n,
        SWIGLU_LIMIT=limit,
        BLOCK_H=BLOCK_H,
        E_PADDED=triton.next_power_of_2(num_experts),
    )
    return q, sf_storage.transpose(1, 2)
