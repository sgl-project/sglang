"""Marlin weight-only GEMMs (W4A16, W8A16) with BF16 activations in raw pair order."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.pair_row_domain import (
    PairRowDomainProvider,
    PairRowState,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraFp8MarlinQuantInfo,
    MoeLoraMarlinQuantInfo,
    MoeLoraMxfp4MarlinQuantInfo,
    MoeLoraNvFp4MarlinQuantInfo,
)


class MarlinRowState(PairRowState, kw_only=True):
    # Per-forward locks prevent aliasing across CUDA graph pools.
    lock_workspace: torch.Tensor
    block_size_m: int
    # Marlin tiles pad each gate/up slice; these hold the padded columns.
    padded_gateup: torch.Tensor | None = None
    padded_act: torch.Tensor | None = None


class MarlinPairProvider(PairRowDomainProvider):
    _use_atomic_add = True
    _is_fp8 = False
    derived_base_weights = "the Marlin-repacked expert weights"

    def __init__(self, quant_info: MoeLoraMarlinQuantInfo):
        super().__init__(
            quant_info,
            gate_up_slices=quant_info.gate_up_slices,
            device=quant_info.w13_qweight.device,
        )
        self._padded_intermediate_size = quant_info.w2_qweight.shape[1] * 16
        from sgl_kernel.scalar_type import scalar_types

        self._scalar_type = (
            scalar_types.float8_e4m3fn if self._is_fp8 else scalar_types.float4_e2m1f
        )

    def _block_size_m(self, num_tokens: int, top_k: int, num_experts: int) -> int:
        # fused_marlin_moe's M block size rule.
        pairs_per_expert = num_tokens * top_k / num_experts
        for block_size_m in (8, 16, 32, 48, 64):
            if pairs_per_expert / block_size_m < 0.9:
                break
        return block_size_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> MarlinRowState:
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        block_size_m = self._block_size_m(
            hidden_states.shape[0], top_k, self.quant_info.num_local_experts
        )
        rows = self._sorted_rows(
            hidden_states, topk_ids, top_k, block_size_m, workspace
        )
        num_pairs = rows["num_pairs"]
        device = hidden_states.device
        padded = self._padded_intermediate_size != self.intermediate_size
        return MarlinRowState(
            **rows,
            lock_workspace=marlin_make_workspace(device, max_blocks_per_sm=4),
            block_size_m=block_size_m,
            padded_gateup=(
                prepare_buffer(
                    workspace,
                    "marlin:padded_gateup",
                    (num_pairs, self._gate_up_slices * self._padded_intermediate_size),
                    dtype=hidden_states.dtype,
                    device=device,
                )
                if padded
                else None
            ),
            padded_act=(
                prepare_buffer(
                    workspace,
                    "marlin:padded_act",
                    (num_pairs, self._padded_intermediate_size),
                    dtype=hidden_states.dtype,
                    device=device,
                )
                if padded
                else None
            ),
        )

    def _invoke(
        self,
        a: torch.Tensor,
        out: torch.Tensor,
        row_state: MarlinRowState,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor | None,
        top_k: int,
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
            b_q_type=self._scalar_type,
            size_m=a.shape[0],
            size_n=out.shape[1],
            size_k=a.shape[1],
            is_k_full=True,  # no GPTQ act-order in FP4 or FP8; K is never permuted
            use_atomic_add=self._use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def gateup(self, row_state: MarlinRowState, out: torch.Tensor) -> None:
        qi = self.quant_info
        padded = row_state.padded_gateup
        self._invoke(
            row_state.hidden_states,
            out if padded is None else padded,
            row_state,
            qweight=qi.w13_qweight,
            scales=qi.w13_scales,
            global_scale=qi.w13_global_scale,
            top_k=row_state.top_k,
        )
        if padded is not None:
            slices = self._gate_up_slices
            out.view(row_state.num_pairs, slices, qi.intermediate_size).copy_(
                padded.view(row_state.num_pairs, slices, -1)[
                    :, :, : qi.intermediate_size
                ]
            )

    def down(
        self,
        row_state: MarlinRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        qi = self.quant_info
        padded = row_state.padded_act
        if padded is not None:
            padded[:, : qi.intermediate_size].copy_(act_out)
            # Reusable storage: clear the padded tail on every replay.
            padded[:, qi.intermediate_size :].zero_()
            act_out = padded
        self._invoke(
            act_out,
            out,
            row_state,
            qweight=qi.w2_qweight,
            scales=qi.w2_scales,
            global_scale=qi.w2_global_scale,
            top_k=1,
        )


class MarlinFp8PairProvider(MarlinPairProvider):
    contract = MoeBaseProviderContract(
        key="marlin_fp8_contiguous", quant_info_cls=MoeLoraFp8MarlinQuantInfo
    )
    _is_fp8 = True


class MarlinMxfp4PairProvider(MarlinPairProvider):
    contract = MoeBaseProviderContract(
        key="marlin_mxfp4_contiguous", quant_info_cls=MoeLoraMxfp4MarlinQuantInfo
    )
    # The E8M0-scale kernels have no atomic-add path: split-K reduces in FP32.
    _use_atomic_add = False


class MarlinNvFp4PairProvider(MarlinPairProvider):
    contract = MoeBaseProviderContract(
        key="marlin_nvfp4_contiguous", quant_info_cls=MoeLoraNvFp4MarlinQuantInfo
    )
