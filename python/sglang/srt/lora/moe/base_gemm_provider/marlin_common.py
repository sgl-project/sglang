"""Marlin weight-only GEMMs (W4A16, W8A16) with BF16 activations in raw pair order."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import prepare_buffer
from sglang.srt.lora.moe.base_gemm_provider.pair_row_domain import (
    PairRowDomainProvider,
    PairRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraMarlinQuantInfo


class MarlinRowState(PairRowState, kw_only=True):
    # Per-forward locks prevent aliasing across CUDA graph pools.
    lock_workspace: torch.Tensor
    block_size_m: int
    # Marlin tiles pad each gate/up slice; these hold the padded columns.
    padded_gateup: torch.Tensor | None = None
    padded_act: torch.Tensor | None = None


class MarlinContiguousProvider(PairRowDomainProvider):
    """Stage plumbing shared by the Marlin vendors; subclasses bind the scales."""

    # The E8M0-scale kernels have no atomic-add path: split-K reduces in FP32.
    _use_atomic_add = False
    # The packed weight format: 4-bit FP4 pairs unless a vendor sets FP8 bytes.
    _weight_bits = 4
    _is_fp8 = False
    # fused_marlin_moe's M block size rule: the first block that the average
    # pairs per expert fill to less than its fraction, 64 beyond the last.
    _block_fill = ((8, 0.9), (16, 0.9), (32, 0.9), (48, 0.9))
    # A reload replaces the repacked parameters these bind.
    derived_base_weights = "the Marlin-repacked expert weights"

    def __init__(
        self,
        quant_info: MoeLoraMarlinQuantInfo,
        *,
        global_scales: tuple[torch.Tensor | None, torch.Tensor | None],
    ):
        # Packed weights use explicit logical sizes instead of shape admission.
        super().__init__(
            quant_info,
            gate_up_slices=quant_info.gate_up_slices,
            device=quant_info.w13_qweight.device,
        )
        self._padded_intermediate_size = quant_info.w2_qweight.shape[1] * 16
        self._w13_global_scale, self._w2_global_scale = global_scales
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            get_scalar_type,
        )

        self._scalar_type_w13 = get_scalar_type(
            self._weight_bits,
            False,
            quant_info.w13_scales,
            self._w13_global_scale,
            self._is_fp8,
        )
        self._scalar_type_w2 = get_scalar_type(
            self._weight_bits,
            False,
            quant_info.w2_scales,
            self._w2_global_scale,
            self._is_fp8,
        )

        # Chosen once per layer: prepare runs in every forward.
        self._layer_block_fill = self._block_fill_for(quant_info)

    @classmethod
    def _block_fill_for(cls, quant_info) -> tuple[tuple[int, float], ...]:
        return cls._block_fill

    @staticmethod
    def _block_from_fill(
        block_fill: tuple[tuple[int, float], ...],
        num_tokens: int,
        top_k: int,
        num_experts: int,
    ) -> int:
        pairs_per_expert = num_tokens * top_k / num_experts
        for block, fill in block_fill:
            if pairs_per_expert / block < fill:
                return block
        return 64

    def _block_size_m(self, num_tokens: int, top_k: int, num_experts: int) -> int:
        return self._block_from_fill(
            self._layer_block_fill, num_tokens, top_k, num_experts
        )

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
            global_scale=self._w13_global_scale,
            scalar_type=self._scalar_type_w13,
            top_k=row_state.top_k,
            size_m=row_state.hidden_states.shape[0],
            size_n=self._gate_up_slices * self._padded_intermediate_size,
            size_k=qi.hidden_size,
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
            global_scale=self._w2_global_scale,
            scalar_type=self._scalar_type_w2,
            top_k=1,
            size_m=row_state.num_pairs,
            size_n=qi.hidden_size,
            size_k=self._padded_intermediate_size,
        )
