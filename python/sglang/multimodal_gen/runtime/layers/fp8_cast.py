import torch

FP8_CAST_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_CAST_BLOCK_SIZE = 1024

_fused_add_round_kernel = None
_triton = None


def is_fp8_cast_dtype(dtype: torch.dtype) -> bool:
    return dtype in FP8_CAST_DTYPES


def fused_add_round_fp8_cast_(
    target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int = 0
) -> None:
    if not str(original_weight.device).startswith("cuda"):
        target_weight.add_(original_weight.to(dtype=target_weight.dtype))
        return

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bias, mantissa_bits = 7, 3
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bias, mantissa_bits = 15, 2
    else:
        raise ValueError(f"Unsupported fp8-cast dtype: {original_weight.dtype}")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")
    if not original_weight.is_contiguous() or not target_weight.is_contiguous():
        raise ValueError("fp8-cast fused add expects contiguous tensors")

    triton, kernel = _get_fused_add_round_kernel()
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, FP8_CAST_BLOCK_SIZE),)
    kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        FP8_CAST_BLOCK_SIZE,
    )


def _get_fused_add_round_kernel():
    global _fused_add_round_kernel, _triton
    if _fused_add_round_kernel is not None:
        return _triton, _fused_add_round_kernel

    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        x_ptr,
        output_ptr,
        seed,
        n_elements,
        EXPONENT_BIAS,
        MANTISSA_BITS,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        rand_vals = tl.rand(seed, offsets) - 0.5

        x = tl.cast(x, tl.float16)
        delta = tl.load(output_ptr + offsets, mask=mask)
        delta = tl.cast(delta, tl.float16)
        x = x + delta

        x_bits = tl.cast(x, tl.int16, bitcast=True)
        fp16_exponent_bits = (x_bits & 0x7C00) >> 10
        fp16_normals = fp16_exponent_bits > 0
        fp16_exponent = tl.where(fp16_normals, fp16_exponent_bits - 15, -14)

        exponent = fp16_exponent + EXPONENT_BIAS
        max_exponent = 2 * EXPONENT_BIAS + 1
        exponent = tl.where(exponent > max_exponent, max_exponent, exponent)
        exponent = tl.where(exponent < 0, 0, exponent)

        eps_exp = tl.maximum(
            0, tl.minimum(31, exponent - EXPONENT_BIAS - MANTISSA_BITS + 15)
        )
        eps_normal = tl.cast(tl.cast(eps_exp << 10, tl.int16), tl.float16, bitcast=True)
        eps_subnormal = tl.cast(
            tl.cast((16 - EXPONENT_BIAS - MANTISSA_BITS) << 10, tl.int16),
            tl.float16,
            bitcast=True,
        )
        eps = tl.where(exponent > 0, eps_normal, eps_subnormal)
        eps = tl.where(x == 0, 0.0, eps)

        output = tl.cast(x + rand_vals * eps, tl.bfloat16)
        tl.store(output_ptr + offsets, output, mask=mask)

    _triton = triton
    _fused_add_round_kernel = _kernel
    return _triton, _fused_add_round_kernel
