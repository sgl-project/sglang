"""Triton FP8 GEMMs with BF16 LoRA inputs/outputs.

Block-scaled weights take group-128 activations; per-channel weights take
per-token activations and per-tensor weights per-tensor activations, both
quantized by the kernel launcher.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.triton_bf16 import (
    TritonBf16ContiguousProvider,
    TritonRowState,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraFp8ChannelQuantInfo,
    MoeLoraFp8QuantInfo,
    MoeLoraFp8TensorQuantInfo,
)

# Activation group width of block-scaled FP8 weights.
_QUANT_GROUP = 128


class _TritonFp8Provider(TritonBf16ContiguousProvider):
    _config_dtype_tag = "fp8_w8a8"

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
            # Pre-quantized activations carry their matching scales.
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
            per_channel_quant=self._config_per_channel_quant,
            block_shape=self._config_block_shape,
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
        if row_state.act_quant_ready:
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


class TritonFp8ContiguousProvider(_TritonFp8Provider):
    contract = MoeBaseProviderContract(
        key="triton_fp8_contiguous",
        quant_info_cls=MoeLoraFp8QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        self._config_block_shape = list(quant_info.block_shape)

    def _act_quant_target(self, row_state: TritonRowState, fused, config):
        # Quantize during the unfused activation to avoid a separate down-GEMM
        # input pass; after a fused one the launcher quantizes inside down().
        if fused:
            return None
        rows, width = self.act_out_shape(row_state)
        device = row_state.pair_to_row.device
        row_state.act_quant = (
            prepare_buffer(
                row_state.workspace,
                "act:quant_rows",
                (rows, width),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            prepare_buffer(
                row_state.workspace,
                "act:quant_scales",
                (rows, width // _QUANT_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        return (*row_state.act_quant, _QUANT_GROUP)


class TritonFp8ChannelContiguousProvider(_TritonFp8Provider):
    contract = MoeBaseProviderContract(
        key="triton_fp8_channel_contiguous",
        quant_info_cls=MoeLoraFp8ChannelQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    _config_per_channel_quant = True


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _routed_rows_scale(rows: torch.Tensor, pair_to_row: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor FP8 scale over the rows of routed pairs.

    Rows of pairs without a local expert (pair_to_row -1) are never written,
    so the launcher's scale over every row would fold stale data into them.
    """
    row_amax = torch.linalg.vector_norm(rows, float("inf"), dim=1, dtype=torch.float32)
    amax = torch.where(pair_to_row >= 0, row_amax, 0.0).amax()
    return (amax.clamp(min=1e-12) / _FP8_MAX).reshape(1)


class TritonFp8TensorContiguousProvider(_TritonFp8Provider):
    """Per-expert weight scales; the launcher quantizes activations per tensor
    with the checkpoint's static scale, or a dynamic one over the routed rows."""

    contract = MoeBaseProviderContract(
        key="triton_fp8_tensor_contiguous",
        quant_info_cls=MoeLoraFp8TensorQuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
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
            a_scale=self.quant_info.w13_input_scale,
        )

    def down(
        self,
        row_state: TritonRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        a_scale = self.quant_info.w2_input_scale
        if a_scale is None:
            a_scale = _routed_rows_scale(act_out, row_state.pair_to_row)
        self._invoke_fp8(
            act_out,
            self.quant_info.w2_weight,
            self.quant_info.w2_scale,
            out,
            row_state,
            top_k=1,
            config=row_state.down_config or row_state.config,
            a_scale=a_scale,
        )
