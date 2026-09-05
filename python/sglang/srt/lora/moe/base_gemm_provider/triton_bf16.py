"""Triton GEMMs gather sorted token IDs and scatter into raw pair order."""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo


class TritonRowState(msgspec.Struct, kw_only=True):
    hidden_states: torch.Tensor  # [num_tokens, hidden] bf16, borrowed
    topk_ids: torch.Tensor  # [num_tokens, top_k] borrowed
    sorted_token_ids: torch.Tensor  # [em_ceiling] int32 route ids, pad = numel
    expert_ids: torch.Tensor  # [em_ceiling / block_m] int32 per-block expert
    num_tokens_post_padded: torch.Tensor  # [1] int32 device scalar
    pair_to_row: torch.Tensor  # [num_pairs] int32 identity
    num_pairs: int
    top_k: int
    config: dict
    down_config: dict | None
    # Quantized activation and scales for the down GEMM.
    act_quant: tuple[torch.Tensor, torch.Tensor] | None = None


class TritonBf16ContiguousProvider(ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="triton_bf16_contiguous",
        quant_info_cls=MoeLoraBf16QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        # prepare() replaces parent compaction with the tuned BLOCK_SIZE_M.
        super().__init__(quant_info, m_alignment=1)
        # Finalize applies router weights; the launcher only checks this stride.
        self._unused_topk_weights = torch.zeros(
            (1, 1), dtype=torch.float32, device=quant_info.w2_weight.device
        )
        self._configs: dict[int, tuple[dict, dict | None]] = {}

    _config_dtype_tag: str | None = None
    _config_block_shape: list[int] | None = None

    def _config_for(self, num_tokens: int, top_k: int) -> tuple[dict, dict | None]:
        cached = self._configs.get(num_tokens)
        if cached is not None:
            return cached
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            try_get_optimal_moe_config,
        )

        config, (down_config, _max_block_m) = try_get_optimal_moe_config(
            tuple(self.quant_info.w13_weight.shape),
            tuple(self.quant_info.w2_weight.shape),
            top_k,
            self._config_dtype_tag,
            num_tokens,
            return_down_config=True,
            block_shape=self._config_block_shape,
        )
        # Both GEMMs share a sort; upstream makes their BLOCK_SIZE_M equal.
        resolved = (dict(config), dict(down_config) if down_config else None)
        # Classic Triton kernels do not accept the TMA launcher's flag.
        resolved[0].pop("USE_TMA", None)
        if resolved[1]:
            resolved[1].pop("USE_TMA", None)
        self._configs[num_tokens] = resolved
        return resolved

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> TritonRowState:
        from sglang.srt.lora.moe.kernels.align_rows import align_rows

        config, down_config = self._config_for(hidden_states.shape[0], top_k)
        sorted_token_ids, expert_ids, num_tokens_post_padded = align_rows(
            topk_ids, int(config["BLOCK_SIZE_M"]), self.quant_info.num_local_experts
        )
        num_pairs = topk_ids.numel()
        device = hidden_states.device
        if workspace is not None:
            # Reuse the identity map to avoid an arange launch per layer.
            pair_to_row = workspace.iota(num_pairs, device)
        else:
            pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)
        return TritonRowState(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            pair_to_row=pair_to_row,
            num_pairs=num_pairs,
            top_k=top_k,
            config=config,
            down_config=down_config,
        )

    def release_prepared_inputs(self, row_state: TritonRowState) -> None:
        pass

    def _invoke(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        row_state: TritonRowState,
        *,
        top_k: int,
        config: dict,
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
            None,
            None,
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
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )

    def gateup(self, row_state: TritonRowState, out: torch.Tensor) -> None:
        # GEMM1 gathers token row pair_id // top_k.
        self._invoke(
            row_state.hidden_states,
            self.quant_info.w13_weight,
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
        self._invoke(
            act_out,
            self.quant_info.w2_weight,
            out,
            row_state,
            top_k=1,
            config=row_state.down_config or row_state.config,
        )

    def gateup_out_shape(self, row_state: TritonRowState) -> tuple[int, ...]:
        return (
            row_state.num_pairs,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: TritonRowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.quant_info.intermediate_size)

    def down_out_shape(self, row_state: TritonRowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.quant_info.hidden_size)
