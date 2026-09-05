"""Marlin W4A16 GEMMs with BF16 activations in raw pair order."""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo


class MarlinNvFp4RowState(msgspec.Struct, kw_only=True):
    hidden_states: torch.Tensor  # [num_tokens, hidden] bf16, borrowed
    topk_ids: torch.Tensor  # [num_tokens, top_k] borrowed
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    # Per-forward locks prevent aliasing across CUDA graph pools.
    lock_workspace: torch.Tensor
    pair_to_row: torch.Tensor  # [num_pairs] int32 identity
    num_pairs: int
    num_tokens: int
    top_k: int
    block_size_m: int


class MarlinNvFp4ContiguousProvider(ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="marlin_nvfp4_contiguous",
        quant_info_cls=MoeLoraNvFp4MarlinQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraNvFp4MarlinQuantInfo):
        # Packed weights use explicit logical sizes instead of shape admission.
        self.quant_info = quant_info
        self._m_alignment = 1
        self._gate_up_slices = 2
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )

        self._scalar_type_w13 = get_scalar_type(
            4, False, quant_info.w13_scales, quant_info.w13_global_scale
        )
        self._scalar_type_w2 = get_scalar_type(
            4, False, quant_info.w2_scales, quant_info.w2_global_scale
        )
        # Finalize owns the router weight; the kernel only checks the stride.
        self._unused_topk_weights = torch.zeros(
            (1, 1), dtype=torch.float32, device=quant_info.w13_qweight.device
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> MarlinNvFp4RowState:
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.srt.lora.moe.kernels.align_rows import align_rows

        num_tokens = hidden_states.shape[0]
        num_experts = self.quant_info.num_local_experts
        # fused_marlin_moe's M block size rule.
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * top_k / num_experts / block_size_m < 0.9:
                break
        sorted_token_ids, expert_ids, num_tokens_post_padded = align_rows(
            topk_ids, block_size_m, num_experts
        )
        num_pairs = topk_ids.numel()
        device = hidden_states.device
        if workspace is not None:
            # Reuse the identity map to avoid an arange launch per layer.
            pair_to_row = workspace.iota(num_pairs, device)
        else:
            pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)
        return MarlinNvFp4RowState(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            lock_workspace=marlin_make_workspace(device, max_blocks_per_sm=4),
            pair_to_row=pair_to_row,
            num_pairs=num_pairs,
            num_tokens=num_tokens,
            top_k=top_k,
            block_size_m=block_size_m,
        )

    def release_prepared_inputs(self, row_state: MarlinNvFp4RowState) -> None:
        pass

    def _invoke(
        self,
        a: torch.Tensor,
        out: torch.Tensor,
        row_state: MarlinNvFp4RowState,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
        scalar_type,
        top_k: int,
        size_m: int,
        size_n: int,
        size_k: int,
    ) -> None:
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

        moe_wna16_marlin_gemm(
            a,
            out,
            qweight,
            None,
            scales,
            global_scale,
            None,
            None,
            None,
            row_state.lock_workspace,
            row_state.sorted_token_ids,
            row_state.expert_ids,
            row_state.num_tokens_post_padded,
            self._unused_topk_weights,
            moe_block_size=row_state.block_size_m,
            top_k=top_k,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=scalar_type,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            is_k_full=True,  # NVFP4 has no GPTQ act-order; K is never permuted
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def gateup(self, row_state: MarlinNvFp4RowState, out: torch.Tensor) -> None:
        qi = self.quant_info
        self._invoke(
            row_state.hidden_states,
            out,
            row_state,
            qweight=qi.w13_qweight,
            scales=qi.w13_scales,
            global_scale=qi.w13_global_scale,
            scalar_type=self._scalar_type_w13,
            top_k=row_state.top_k,
            size_m=row_state.num_tokens,
            size_n=self._gate_up_slices * qi.intermediate_size,
            size_k=qi.hidden_size,
        )

    def down(
        self,
        row_state: MarlinNvFp4RowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        qi = self.quant_info
        self._invoke(
            act_out,
            out,
            row_state,
            qweight=qi.w2_qweight,
            scales=qi.w2_scales,
            global_scale=qi.w2_global_scale,
            scalar_type=self._scalar_type_w2,
            top_k=1,
            size_m=row_state.num_pairs,
            size_n=qi.hidden_size,
            size_k=qi.intermediate_size,
        )

    def gateup_out_shape(self, row_state: MarlinNvFp4RowState) -> tuple[int, ...]:
        return (
            row_state.num_pairs,
            self._gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: MarlinNvFp4RowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.quant_info.intermediate_size)

    def down_out_shape(self, row_state: MarlinNvFp4RowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.quant_info.hidden_size)
