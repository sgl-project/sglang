"""SAIL gather-read grouped GEMM; routing metadata is built once and reused by GEMM1."""

from typing import List, Optional, Tuple

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils.custom_op import register_custom_op


def _fused_metadata_fake(A, B, topk_ids):
    numel = A.shape[0] * topk_ids.shape[1]
    return (
        torch.empty(B.shape[0], dtype=torch.int32, device=topk_ids.device),
        torch.empty(numel, dtype=torch.int32, device=topk_ids.device),
        torch.empty(numel, dtype=torch.int32, device=topk_ids.device),
    )


def grouped_gemm_nt_f8f8bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    is_perchannel: bool,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_metadata_fake(A, B, topk_ids)


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f8f8bf16_fused_fake)
def grouped_gemm_nt_f8f8bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    is_perchannel: bool,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, rows, experts, tokens, blocks, inv_perm, expert_ids = (
        deep_gemm_wrapper.moe_align_block_size(
            A, B, topk_ids, perchannel_quant=is_perchannel, configs=configs
        )
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_fused(
        (A, As),
        (B, Bs),
        C,
        rows,
        experts,
        tokens,
        blocks,
        config,
        perchannel_quant=is_perchannel,
    )
    return rows, inv_perm, expert_ids


def grouped_gemm_nt_i8i8bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_metadata_fake(A, B, topk_ids)


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_i8i8bf16_fused_fake)
def grouped_gemm_nt_i8i8bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, rows, experts, tokens, blocks, inv_perm, expert_ids = (
        deep_gemm_wrapper.moe_align_block_size(
            A, B, topk_ids, perchannel_quant=True, configs=configs
        )
    )
    deep_gemm_wrapper.grouped_gemm_nt_i8i8bf16_fused(
        (A, As), (B, Bs), C, rows, experts, tokens, blocks, config
    )
    return rows, inv_perm, expert_ids


def grouped_gemm_nt_f4f4bf16_fused_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_metadata_fake(A, B, topk_ids)


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_f4f4bf16_fused_fake)
def grouped_gemm_nt_f4f4bf16_fused(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, rows, experts, tokens, blocks, inv_perm, expert_ids = (
        deep_gemm_wrapper.moe_align_block_size(
            A, B, topk_ids, perchannel_quant=False, configs=configs
        )
    )
    deep_gemm_wrapper.grouped_gemm_nt_f4f4bf16_fused(
        (A, As), (B, Bs), C, rows, experts, tokens, blocks, config
    )
    return rows, inv_perm, expert_ids


def grouped_gemm_nt_bf16i4bf16_fused_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_metadata_fake(A, B, topk_ids)


@register_custom_op(mutates_args=["C"], fake_impl=grouped_gemm_nt_bf16i4bf16_fused_fake)
def grouped_gemm_nt_bf16i4bf16_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, rows, experts, tokens, blocks, inv_perm, expert_ids = (
        deep_gemm_wrapper.moe_align_block_size(
            A, B, topk_ids, perchannel_quant=False, configs=configs
        )
    )
    deep_gemm_wrapper.grouped_gemm_nt_bf16i4bf16_fused(
        A, (B, Bs), C, rows, experts, tokens, blocks, config
    )
    return rows, inv_perm, expert_ids


def grouped_gemm_nt_bf16bf16bf16_fused_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_metadata_fake(A, B, topk_ids)


@register_custom_op(
    mutates_args=["C"], fake_impl=grouped_gemm_nt_bf16bf16bf16_fused_fake
)
def grouped_gemm_nt_bf16bf16bf16_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_ids: torch.Tensor,
    configs: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    config, rows, experts, tokens, blocks, inv_perm, expert_ids = (
        deep_gemm_wrapper.moe_align_block_size(A, B, topk_ids, configs=configs)
    )
    deep_gemm_wrapper.grouped_gemm_nt_bf16bf16bf16_fused(
        A, B, C, rows, experts, tokens, blocks, config
    )
    return rows, inv_perm, expert_ids
