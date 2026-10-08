from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_HC_COUNT = 4
_QWEN4_HIDDEN_SIZE = 2560


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round fp32 to BF16 precision while retaining an fp32 register."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _qwen4_gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_group = tl.program_id(0)
    token = token_group // HC_COUNT
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = (token < num_tokens) & (hidden < HIDDEN_SIZE)

    # `gate` arrives already rounded to bf16 by the eager multiply/reduce/divide;
    # every remaining eager bf16 rounding boundary is reproduced below.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    transformed = _round_bf16_to_fp32(root * sign)
    activated = _round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(value_ptr + token * HIDDEN_SIZE + hidden, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = token_group * HIDDEN_SIZE + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def can_fuse_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether inputs match Qwen4's fixed BF16 gate/value contract."""

    return (
        gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply Qwen4's post-reduction gate and value broadcast in one kernel."""

    if not can_fuse_qwen4_gate_value(gate, value):
        raise ValueError("unsupported input for fused Qwen4 PLE gate/value")
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_value_kernel[(gate.shape[0] * _QWEN4_HC_COUNT,)](
            gate,
            value,
            output,
            gate.shape[0],
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
    return output


@triton.jit
def _qwen4_gate_reduce_kernel(
    key_ptr,
    query_ptr,
    value_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = hidden < HIDDEN_SIZE
    key = tl.load(key_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(tl.float32)
    query = tl.load(query_ptr + group * HIDDEN_SIZE + hidden, mask, other=0).to(
        tl.float32
    )
    product = _round_bf16_to_fp32(key * query)
    gate = _round_bf16_to_fp32(tl.sum(product, 0))
    gate = _round_bf16_to_fp32(gate * (HIDDEN_SIZE**-0.5))
    magnitude = _round_bf16_to_fp32(tl.maximum(tl.abs(gate), 1.0e-6))
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0, 1.0, tl.where(gate < 0, -1.0, 0.0))
    activated = _round_bf16_to_fp32(tl.sigmoid(_round_bf16_to_fp32(root * sign)))
    value = tl.load(
        value_ptr + (group // HC_COUNT) * HIDDEN_SIZE + hidden, mask, other=0
    ).to(tl.float32)
    tl.store(output_ptr + group * HIDDEN_SIZE + hidden, activated * value, mask)


def can_fuse_qwen4_gate_reduce(key, query, value):
    return (
        key.is_cuda
        and key.dtype == torch.bfloat16
        and key.ndim == 3
        and key.shape[1:] == (_QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE)
        and key.is_contiguous()
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.device == key.device
        and query.is_contiguous()
        and value.shape == (key.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.dtype == key.dtype
        and value.device == key.device
        and value.is_contiguous()
    )


def fused_qwen4_gate_reduce(key, query, value):
    if not can_fuse_qwen4_gate_reduce(key, query, value):
        raise ValueError("unsupported input for Qwen4 PLE gate reduction")
    output = torch.empty_like(key)
    if key.shape[0]:
        _qwen4_gate_reduce_kernel[(key.shape[0] * _QWEN4_HC_COUNT,)](
            key,
            query,
            value,
            output,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            HC_COUNT=_QWEN4_HC_COUNT,
            BLOCK_SIZE=triton.next_power_of_2(_QWEN4_HIDDEN_SIZE),
            enable_fp_fusion=False,
        )
    return output
