import torch


def es_fp8_blockwise_scaled_grouped_mm(
    output,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_d,
    problem_sizes,
    expert_offsets,
    workspace,
):
    torch.ops.sgl_kernel.es_fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_d,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def es_sm100_mxfp8_blockscaled_grouped_mm(
    output, a, b, sfa, sfb, problem_sizes, expert_offsets, blockscale_offsets
):
    torch.ops.sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_mm.default(
        a, b, sfa, sfb, output, problem_sizes, expert_offsets, blockscale_offsets
    )


def es_sm100_mxfp8_blockscaled_grouped_quant(
    input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor
):
    torch.ops.sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_quant.default(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )
