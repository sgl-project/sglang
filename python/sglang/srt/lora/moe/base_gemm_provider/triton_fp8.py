"""Triton FP8 GEMMs with 128-block scales and BF16 LoRA inputs/outputs."""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.triton_bf16 import (
    TritonBf16ContiguousProvider,
    TritonRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo


class TritonFp8ContiguousProvider(TritonBf16ContiguousProvider):
    contract = MoeBaseProviderContract(
        key="triton_fp8_contiguous",
        quant_info_cls=MoeLoraFp8QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        act_quant_group=128,
    )

    _config_dtype_tag = "fp8_w8a8"

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        self._config_block_shape = list(quant_info.block_shape)

    def _invoke_fp8(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        out: torch.Tensor,
        row_state: TritonRowState,
        *,
        top_k: int,
        config: dict,
        a_scale: torch.Tensor | None = None,
    ) -> None:
        import triton.language as tl

        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel,
        )

        invoke_fused_moe_kernel(
            a,
            b,
            None,
            out,
            # Supplying scales skips the launcher's activation quantization.
            a_scale,
            b_scale,
            None,
            self._unused_topk_weights,
            row_state.topk_ids,
            row_state.sorted_token_ids,
            row_state.expert_ids,
            row_state.num_tokens_post_padded,
            False,
            top_k,
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=list(self.quant_info.block_shape),
        )

    def gateup(self, row_state: TritonRowState, out: torch.Tensor) -> None:
        self._invoke_fp8(
            row_state.hidden_states,
            self.quant_info.w13_weight,
            self.quant_info.w13_scale,
            out,
            row_state,
            top_k=row_state.top_k,
            config=row_state.config,
        )

    def down(
        self,
        row_state: TritonRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        a, a_scale = act_out, None
        if row_state.act_quant is not None:
            a, a_scale = row_state.act_quant
        self._invoke_fp8(
            a,
            self.quant_info.w2_weight,
            self.quant_info.w2_scale,
            out,
            row_state,
            top_k=1,
            config=row_state.down_config or row_state.config,
            a_scale=a_scale,
        )
