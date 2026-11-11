"""Triton GEMMs gather sorted token IDs and scatter into raw pair order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    admit_weight_layout,
)
from sglang.srt.lora.moe.base_gemm_provider.pair_row_domain import (
    PairRowDomainProvider,
    PairRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace


class TritonRowState(PairRowState, kw_only=True):
    topk_ids: torch.Tensor  # [num_tokens, top_k] borrowed
    config: dict
    down_config: dict | None
    workspace: LoraWorkspace | None  # borrowed
    # FP8 activation and group scales for the down GEMM, and whether this
    # forward's activation wrote them.
    act_quant: tuple[torch.Tensor, torch.Tensor] | None = None
    act_quant_ready: bool = False


class TritonBf16ContiguousProvider(PairRowDomainProvider):
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
        super().__init__(
            quant_info,
            gate_up_slices=admit_weight_layout(quant_info),
            device=quant_info.w2_weight.device,
        )
        self._configs: dict[int, tuple[dict, dict | None]] = {}

    _config_dtype_tag: str | None = None
    _config_block_shape: list[int] | None = None
    _config_per_channel_quant: bool = False

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
            per_channel_quant=self._config_per_channel_quant,
        )
        # Both GEMMs share a sort; upstream makes their BLOCK_SIZE_M equal.
        resolved = (dict(config), dict(down_config) if down_config else None)
        for stage_config in resolved:
            if stage_config is None:
                continue
            # Classic Triton kernels do not accept the TMA launcher's flag.
            stage_config.pop("USE_TMA", None)
            if self._config_block_shape is not None:
                # One dot tile must not span independently scaled K groups.
                stage_config["BLOCK_SIZE_K"] = min(
                    stage_config["BLOCK_SIZE_K"], self._config_block_shape[1]
                )
        self._configs[num_tokens] = resolved
        return resolved

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> TritonRowState:
        config, down_config = self._config_for(hidden_states.shape[0], top_k)
        return TritonRowState(
            **self._sorted_rows(
                hidden_states, topk_ids, top_k, int(config["BLOCK_SIZE_M"]), workspace
            ),
            topk_ids=topk_ids,
            config=config,
            down_config=down_config,
            workspace=workspace,
        )

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
