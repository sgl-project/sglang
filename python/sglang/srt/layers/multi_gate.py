from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import Modality

MULTI_GATE_BLOCK_M = 64
TEXT_MODALITY = 0
VISION_MODALITY = 1
AUDIO_MODALITY = 2


@triton.jit
def multi_gate_kernel(
    hidden_states_ptr,
    router_logits_ptr,
    expert_bias_ptr,
    text_gate_ptr,
    image_gate_ptr,
    audio_gate_ptr,
    text_bias_ptr,
    image_bias_ptr,
    audio_bias_ptr,
    token_indices_ptr,
    modality_ids_ptr,
    num_valid_tokens: tl.constexpr,
    compute_type: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_indices = tl.load(token_indices_ptr + offs)
    token_mask = token_indices < num_valid_tokens
    modality_id = tl.load(modality_ids_ptr + pid_m).to(tl.int64)

    if modality_id == VISION_MODALITY:
        gate_ptr = image_gate_ptr
        bias_ptr = image_bias_ptr
    elif modality_id == AUDIO_MODALITY:
        gate_ptr = audio_gate_ptr
        bias_ptr = audio_bias_ptr
    else:
        gate_ptr = text_gate_ptr
        bias_ptr = text_bias_ptr

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    hidden_ptrs = hidden_states_ptr + (
        token_indices[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    gate_ptrs = gate_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        hidden = tl.load(
            hidden_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
            other=0.0,
        ).to(compute_type)
        gate = tl.load(gate_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(hidden, gate)
        hidden_ptrs += BLOCK_SIZE_K * stride_ak
        gate_ptrs += BLOCK_SIZE_K * stride_bk

    output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offsets = stride_cm * token_indices[:, None] + stride_cn * output_n[None, :]
    output_mask = token_mask[:, None] & (output_n[None, :] < N)
    tl.store(router_logits_ptr + output_offsets, accumulator, mask=output_mask)

    bias = tl.load(bias_ptr + output_n[None, :], mask=output_n[None, :] < N, other=0.0)
    tl.store(expert_bias_ptr + output_offsets, bias, mask=output_mask)


def _kernel_config(num_tokens: int) -> dict:
    configs = {
        1024: (64, 128, 64, 64, 4, 3),
        2048: (64, 32, 128, 1, 8, 3),
        4096: (64, 64, 128, 32, 4, 3),
        8192: (64, 32, 128, 64, 8, 3),
    }
    key = min(configs, key=lambda candidate: abs(candidate - num_tokens))
    block_m, block_n, block_k, group_m, num_warps, num_stages = configs[key]
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def create_multi_gate_mm_indices(
    token_modalities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group token indices into padded blocks tagged by modality."""
    if token_modalities.ndim != 1:
        raise ValueError(
            f"token_modalities must be one-dimensional, got {token_modalities.shape=}"
        )
    modality_indices = [
        (token_modalities == 0).nonzero(as_tuple=False).squeeze(-1),
        (
            (token_modalities == Modality.IMAGE.value)
            | (token_modalities == Modality.VIDEO.value)
        )
        .nonzero(as_tuple=False)
        .squeeze(-1),
        (token_modalities == Modality.AUDIO.value).nonzero(as_tuple=False).squeeze(-1),
    ]
    block_counts = [
        (indices.shape[0] + MULTI_GATE_BLOCK_M - 1) // MULTI_GATE_BLOCK_M
        for indices in modality_indices
    ]
    total_tokens = token_modalities.shape[0]
    total_blocks = sum(block_counts)
    token_indices = torch.full(
        (total_blocks * MULTI_GATE_BLOCK_M,),
        total_tokens,
        dtype=torch.int32,
        device=token_modalities.device,
    )
    modality_ids = torch.empty(
        total_blocks, dtype=torch.int32, device=token_modalities.device
    )
    block_offset = 0
    for modality, (indices, block_count) in enumerate(
        zip(modality_indices, block_counts)
    ):
        token_offset = block_offset * MULTI_GATE_BLOCK_M
        token_indices[token_offset : token_offset + indices.shape[0]] = indices
        modality_ids[block_offset : block_offset + block_count] = modality
        block_offset += block_count
    return token_indices, modality_ids


@torch.compiler.disable
def multi_gate_triton_kernel(
    hidden_states: torch.Tensor,
    multi_gate_indices: Tuple[torch.Tensor, torch.Tensor],
    text_weight: torch.Tensor,
    image_weight: torch.Tensor,
    audio_weight: torch.Tensor,
    text_bias: torch.Tensor,
    image_bias: torch.Tensor,
    audio_bias: torch.Tensor,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_indices, modality_ids = multi_gate_indices
    num_tokens = hidden_states.shape[0]
    weights = (text_weight, image_weight, audio_weight)
    biases = (text_bias, image_bias, audio_bias)
    num_experts = text_weight.shape[0]
    if any(weight.shape != text_weight.shape for weight in weights[1:]):
        raise ValueError("All modality gate weights must have the same shape")
    if any(bias is None or bias.shape != (num_experts,) for bias in biases):
        raise ValueError(
            "Multi-gate routing requires one expert-bias vector per modality"
        )

    transposed_weights = tuple(weight.transpose(0, 1) for weight in weights)
    router_logits = torch.empty(
        (num_tokens, num_experts),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dynamic_expert_bias = torch.empty(
        (num_tokens, num_experts), dtype=torch.float32, device=hidden_states.device
    )
    if text_weight.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif text_weight.dtype == torch.float16:
        compute_type = tl.float16
    elif text_weight.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported multi-gate dtype: {text_weight.dtype}")

    num_grouped_tokens = token_indices.shape[0]
    config = config or _kernel_config(num_grouped_tokens)
    if config["BLOCK_SIZE_M"] != MULTI_GATE_BLOCK_M:
        raise ValueError(
            f"Multi-gate BLOCK_SIZE_M must be {MULTI_GATE_BLOCK_M}, got {config['BLOCK_SIZE_M']}"
        )
    grid = lambda meta: (
        triton.cdiv(num_grouped_tokens, meta["BLOCK_SIZE_M"])
        * triton.cdiv(num_experts, meta["BLOCK_SIZE_N"]),
    )
    multi_gate_kernel[grid](
        hidden_states,
        router_logits,
        dynamic_expert_bias,
        *transposed_weights,
        *biases,
        token_indices,
        modality_ids,
        num_valid_tokens=num_tokens,
        compute_type=compute_type,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_bk=transposed_weights[0].stride(0),
        stride_bn=transposed_weights[0].stride(1),
        stride_cm=router_logits.stride(0),
        stride_cn=router_logits.stride(1),
        M=num_grouped_tokens,
        N=num_experts,
        K=hidden_states.shape[-1],
        **config,
    )
    return router_logits, dynamic_expert_bias
