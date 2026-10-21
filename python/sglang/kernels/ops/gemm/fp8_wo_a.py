"""DeepGEMM grouped output projection with an MXFP8 epilogue."""

import torch


def fp8_wo_a_mxfp8(
    activations: torch.Tensor,
    activation_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    recipe: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract [M,G,K] x [G,R,K] and quantize the BF16-rounded result.

    Returns contiguous E4M3 [M,G*R] and packed UE8M0 [M,G*R/128]
    with MN-major, four-row-aligned strides. These are the native input
    layout of SGLang's DeepGEMM MXFP8 linear, not FlashInfer's swizzled
    layout. The upstream epilogue rounds the accumulator to BF16 before
    finding each 32-element absmax, matching a separate BF16-output GEMM
    followed by group-32 quantization.
    """
    import deep_gemm

    rows = activations.shape[0]
    groups, rank = weight.shape[:2]
    output = torch.empty(
        (rows, groups, rank), device=activations.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty_strided(
        (rows, groups * rank // 128),
        (1, (rows + 3) // 4 * 4),
        device=activations.device,
        dtype=torch.int32,
    )
    if rows:
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (activations, activation_scales),
            (weight, weight_scales),
            output,
            recipe=recipe,
            epilogue=deep_gemm.epilogue.FP8CastWithClamp(scales, amax_eps=1e-10),
        )
    return output.flatten(1), scales
