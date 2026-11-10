"""SAIL activations for the DeepGEMM MoE runners.

Every platform runs the plain SiLU between GEMM1 and GEMM2. The models SAIL
serves also need the oai-swiglu gating, so that variant lives here and the
shared runner keeps its upstream activation path.
"""

import torch
import triton
import triton.language as tl


def apply_oai_swiglu(
    gateup_output: torch.Tensor,
    gemm1_alpha: float,
    gemm1_clamp_limit: float,
    gate_up_interleaved: bool,
) -> torch.Tensor:
    """Run FP32 oai-swiglu according to the weight layout; the BF16 upcast keeps the
    clamp from mutating the GEMM1 output."""
    assert gateup_output.dtype == torch.bfloat16, (
        f"expected bf16 gateup_output, got {gateup_output.dtype}"
    )
    if gate_up_interleaved:
        gate = gateup_output[..., ::2].float()
        up = gateup_output[..., 1::2].float()
    else:
        n = gateup_output.shape[-1]
        gate = gateup_output[..., : n // 2].float()
        up = gateup_output[..., n // 2 :].float()
    gate = gate.clamp_(max=gemm1_clamp_limit)
    up = up.clamp_(min=-gemm1_clamp_limit, max=gemm1_clamp_limit)
    return (gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)).to(torch.bfloat16)


@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    masked_m_ptr,
    size_n,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    GEMM1_ALPHA: tl.constexpr,
    GEMM1_CLAMP_LIMIT: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        if GEMM1_ALPHA > 0:
            # Keep the same oai-swiglu formula and FP32 intermediates as the
            # quantized activation.
            gate = tl.minimum(gate, GEMM1_CLAMP_LIMIT)
            up = tl.clamp(up, -GEMM1_CLAMP_LIMIT, GEMM1_CLAMP_LIMIT)
            gate_up = gate * tl.sigmoid(gate * GEMM1_ALPHA) * (up + 1)
        else:
            gate = gate / (1 + tl.exp(-gate))
            gate_up = up * gate
        # The activation is computed in FP32 and cast back to the input dtype.
        gate_up = gate_up.to(input_ptr.dtype.element_ty)
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            gate_up,
            mask=offs_in_d < size_n,
        )


def silu_and_mul_masked_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    gemm1_alpha: float = 0.0,
    gemm1_clamp_limit: float = 0.0,
):
    """SAIL's fused masked SiLU-mul, including the oai-swiglu variant.

    The shared ``ep_moe_kernels.silu_and_mul_masked_fwd`` stays upstream-faithful;
    the oai-swiglu branch only exists for the models SAIL serves, so it is
    forked here instead of growing the shared Triton kernel.

    input shape [expert_num, token_num_padded, hidden_dim], dtype bf16
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype bf16
    masked_m shape [expert_num]

    When gemm1_alpha > 0 this runs oai-swiglu: gate is only bounded from above,
    up is clamped on both sides and then offset by 1.
    """

    assert input.is_contiguous()
    assert output.dtype == torch.bfloat16
    assert input.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = 128
    num_warps = 4
    NUM_STAGES = 4

    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    _silu_and_mul_masked_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        masked_m,
        size_n,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        GEMM1_ALPHA=gemm1_alpha if gemm1_alpha is not None else 0.0,
        GEMM1_CLAMP_LIMIT=gemm1_clamp_limit if gemm1_clamp_limit is not None else 0.0,
        num_warps=num_warps,
    )
    return output
