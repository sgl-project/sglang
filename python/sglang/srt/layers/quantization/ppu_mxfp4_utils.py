import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_device_sm, is_ppu

MXFP_BLOCK_SIZE: tl.constexpr = tl.constexpr(32)


@triton.jit
def _downcast_to_mxfp4(
    # Output: packed e2m1 tensor (uint8, two fp4 per byte)
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    # Output: preprocessed E8M0 scales (uint16, packed + transposed layout [S_pairs, N])
    mx_scale_ptr,
    stride_mx_scale_pair,
    stride_mx_scale_outer,
    # Input: source tensor (bf16/fp16/fp32)
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    orig_quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    """
    Fused MXFP4 downcast kernel: E8M0 scale computation + e2m1 quantization in a single kernel.
    Supports PTX hardware path (SM 8.9+).
    """
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )
    tl.static_assert(
        (BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE) % 2 == 0,
        "Number of scale groups per block must be even for uint16 packing",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_ptr.dtype.element_ty == tl.uint8,
        "Output must be uint8 (packed e2m1).",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint16,
        "Scale must be uint16 (preprocessed E8M0).",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16)
        or (src_dtype == tl.float16)
        or (src_dtype == tl.float32),
        f"{src_dtype=} must be bfloat16, float16, or float32",
    )

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_SCALE_PAIRS: tl.constexpr = BLOCK_SIZE_QUANT_MX_SCALE // 2
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = (
        BLOCK_SIZE_QUANT_DIM // 2
    )  # 2 e2m1 per uint8

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_pair = quant_block * BLOCK_SIZE_QUANT_MX_SCALE_PAIRS
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_pair * stride_mx_scale_pair + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_pairs = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS)[None, :].to(
        tl.int64
    )
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < orig_quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < orig_quant_dim // 2
    full_mask_mxt = mask_mxt_quant & mask_n

    S_GROUPS_PAIRS = quant_dim // MXFP_BLOCK_SIZE // 2
    scale_mask_k = start_mx_scale_pair + offs_scale_pairs < S_GROUPS_PAIRS
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_pairs * stride_mx_scale_pair + offs_outer * stride_mx_scale_outer
    )
    mx_tensor_offsets = (
        offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    )
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    # ======== E8M0 scale computation ========
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(full_mask_src, abs_tensor, -1.0)
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    # Clamp absmax to avoid division by zero in scale computation
    max_val = tl.maximum(max_val, 1e-10)

    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # ROUND_UP: 2 ** ceil(log2(max / 6.0))
        dequant_scale = max_val / 6.0
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        # ROUND_DOWN: 2 ** floor(log2(max / 4.0))
        dequant_scale = max_val / 4.0
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000

    # dequant_scale_rounded is guaranteed non-zero (absmax >= 1e-10), skip tl.where branch
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = 1.0 / dequant_scale_rounded

    # Apply per-group scale
    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE]
    )
    quant_tensor = f32_tensor * quant_scale
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Note: padding elements become 0 * quant_scale = 0 after load with mask (other=0.0),
    # and e2m1(0.0) = 0x0, so no explicit zeroing needed for correctness.

    # Pack scale exponents into uint16 pairs directly from uint32
    # dequant_scale_exponent is [OUT, SCALE, 1] uint32 with exponent in bits [30:23]
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )
    # Zero out exponents for padding groups (beyond original quant_dim)
    ORIG_S_GROUPS = orig_quant_dim // MXFP_BLOCK_SIZE
    scale_group_idx = (
        quant_block * BLOCK_SIZE_QUANT_MX_SCALE
        + tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :]
    )
    dequant_scale_exponent = tl.where(
        scale_group_idx < ORIG_S_GROUPS, dequant_scale_exponent, 0
    )
    # Reshape to pair groups and extract exponent bytes, pack into uint16 in one go
    de_pairs = tl.reshape(
        dequant_scale_exponent, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS, 2]
    )
    lo_u32, hi_u32 = tl.split(de_pairs)
    scale_tensor = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)

    # ======== FP32 -> e2m1 conversion ========
    # SM 8.9+: PTX hardware instruction for e2m1 conversion
    pairs = tl.reshape(quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
    lo_f, hi_f = tl.split(pairs)
    lo_f32 = lo_f.to(tl.float32)
    hi_f32 = hi_f.to(tl.float32)

    # cvt.rn.satfinite.e2m1x2.f32: two f32 -> one packed e2m1x2 (uint8)
    out_tensor = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 r;
            cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
            mov.b32 $0, {r, r, r, r};
        }
        """,
        constraints="=r,f,f",
        args=[hi_f32, lo_f32],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )

    # Write scale (uint16 packed, transposed [S_pairs, N] layout) and tensor
    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


def downcast_to_mxfp4(
    src_tensor: torch.Tensor,
    axis: int,
    round_up: bool = True,
):
    """Convert src tensor to MXFP4 (packed e2m1 uint8) with preprocessed E8M0 scales.

    Scale output is directly in preprocessed layout (uint16 packed, transposed-contiguous),
    equivalent to applying preprocess_mxfp4_scales() on the raw uint8 scale.

    Args:
        src_tensor: Input tensor (bf16/fp16/fp32).
        axis:       Quantization axis.
        round_up:   True for ROUND_UP scale mode, False for ROUND_DOWN.

    Returns:
        (out_quant_tensor, out_scale):
          - out_quant_tensor: uint8, two e2m1 packed per byte.
            Shape is same as src_tensor except axis dim is halved.
          - out_scale: uint16, preprocessed E8M0 scales.
            Logical shape: [..., S_groups // 2] (quant axis replaced by S_groups//2).
            Physical: transposed-contiguous [S_groups//2, N] layout.
    """
    assert is_ppu() and get_device_sm() >= 89, (
        f"PPU MXFP4 downcast_to_mxfp4 impl requires e2m1 PTX instruction (PPU for SM 8.9+)"
    )
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    src_tensor = src_tensor.transpose(axis, ndim - 1)
    L = src_tensor.shape[-1]
    assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    # Pad quant_dim to multiple of 64 for even scale groups (uint16 packing)
    # No actual data padding (no F.pad) — kernel uses mask for OOB elements
    padded_L = ((L + 63) // 64) * 64
    out_shape = src_tensor.shape[:-1] + (L // 2,)

    S_groups = padded_L // MXFP_BLOCK_SIZE.value
    S_groups_pairs = S_groups // 2

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        N = kernel_src.shape[0]  # flattened outer dim

        # Scale output: [S_groups_pairs, N] uint16 contiguous (transposed layout)
        kernel_scale = torch.empty(
            (S_groups_pairs, N), dtype=torch.uint16, device=src_tensor.device
        )

        BLOCK_OUT = 32
        BLOCK_QUANT = 128
        NUM_WARPS = 4

        grid = (
            triton.cdiv(N, BLOCK_OUT),
            triton.cdiv(padded_L, BLOCK_QUANT),
        )

        _downcast_to_mxfp4[grid](
            kernel_quant,
            *kernel_quant.stride(),
            kernel_scale,
            *kernel_scale.stride(),
            kernel_src,
            *kernel_src.stride(),
            N,
            padded_L,
            L,
            BLOCK_SIZE_OUT_DIM=BLOCK_OUT,
            BLOCK_SIZE_QUANT_DIM=BLOCK_QUANT,
            DEQUANT_SCALE_ROUNDING_MODE=0 if round_up else 1,
            num_warps=NUM_WARPS,
        )

        # Reshape scale from [S_pairs, N] to logical [..., S_pairs]
        batch_shape = src_tensor.shape[:-1]
        out_scale = kernel_scale.t().reshape(*batch_shape, S_groups_pairs)
    else:
        batch_shape = src_tensor.shape[:-1]
        out_scale = torch.empty(
            *batch_shape, S_groups_pairs, dtype=torch.uint16, device=src_tensor.device
        )

    out_quant_tensor = out_quant_tensor.transpose(axis, ndim - 1)
    out_scale = out_scale.transpose(axis, ndim - 1)
    return out_quant_tensor, out_scale
