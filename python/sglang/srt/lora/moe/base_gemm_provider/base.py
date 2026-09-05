"""Base MoE stages with insertion points for LoRA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MoeBaseProviderContract(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    quant_info_cls: type
    # Base output order; the LoRA delta is always [gate | up].
    gate_first: bool
    interleaved: bool
    gate_up_output_dtype: torch.dtype
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype
    # Fuse FP8 quantization into activation for the down GEMM.
    act_quant_group: int | None = None


def expected_rows_per_expert(num_pairs: int, num_experts: int) -> int:
    """Rounded-up mean rows per expert; use one for an empty batch."""
    return (num_pairs - 1) // num_experts + 1 if num_pairs else 1


def prepare_buffer(workspace, name: str, shape, *, dtype, device) -> torch.Tensor:
    if workspace is not None:
        return workspace.tensor(name, shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def admit_weight_layout(quant_info) -> int:
    """Validate [E, S*I, H] / [E, H, I] weights and return S."""
    if quant_info.intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    expected_w2 = (
        quant_info.num_local_experts,
        quant_info.hidden_size,
        quant_info.intermediate_size,
    )
    if quant_info.w2_weight.shape != expected_w2:
        raise ValueError(
            f"w2_weight must be {expected_w2}, got "
            f"{tuple(quant_info.w2_weight.shape)}"
        )
    if (
        quant_info.w13_weight.ndim != 3
        or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
        or quant_info.w13_weight.shape[2] != quant_info.hidden_size
    ):
        raise ValueError(
            "w13_weight must be [num_local_experts, slices*intermediate, hidden]"
        )
    gateup_width = quant_info.w13_weight.shape[1]
    if gateup_width % quant_info.intermediate_size:
        raise ValueError(
            "w13 output width must be an integer multiple of intermediate_size"
        )
    gate_up_slices = gateup_width // quant_info.intermediate_size
    if gate_up_slices not in (1, 2):
        raise ValueError(
            "the base providers support one non-gated slice or two gated "
            f"gate/up slices, got {gate_up_slices}"
        )
    return gate_up_slices


class MoeBaseProvider:
    """One provider per layer; logical sizes come from quant_info."""

    contract: MoeBaseProviderContract

    @property
    def num_local_experts(self) -> int:
        return self.quant_info.num_local_experts

    @property
    def intermediate_size(self) -> int:
        """Logical intermediate width of one TP shard."""
        return self.quant_info.intermediate_size

    @property
    def hidden_size(self) -> int:
        return self.quant_info.hidden_size

    @property
    def gate_up_slices(self) -> int:
        return self._gate_up_slices

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ):
        """Prepare rows with graph-stable buffers when a workspace is supplied."""
        raise NotImplementedError

    def gateup(self, row_state, out: torch.Tensor) -> None:
        raise NotImplementedError

    def release_prepared_inputs(self, row_state) -> None:
        raise NotImplementedError

    def act_with_delta(
        self,
        row_state,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
    ) -> None:
        raise NotImplementedError

    def fused_act(
        self,
        row_state,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_rows: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
    ) -> None:
        raise NotImplementedError(
            f"{self.contract.key} has no fused-act implementation"
        )

    def down(
        self,
        row_state,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        row_state,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        *,
        lora_delta: torch.Tensor | None = None,
    ) -> None:
        """Weight base + unweighted LoRA delta [T, K, H], then scale once."""
        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

        num_tokens, hidden = output.shape
        post_reorder_deepgemm(
            down_out.view(-1, hidden),
            output,
            row_state.pair_to_row,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            lora_delta=lora_delta,
        )

    def shared_token_delta_finalize(
        self,
        row_state,
        *,
        down_rows: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        token_delta: torch.Tensor,
        token_route: RouteView,
        delta_config: Mapping[str, int],
        config: Mapping[str, Mapping[str, int]],
    ) -> None:
        """Reduce in rank space before applying the shared down-B per token."""
        from sglang.srt.lora.moe.kernels.finalize import (
            invoke_shared_token_delta_reduce,
            invoke_shared_token_delta_tail,
        )
        from sglang.srt.lora.moe.kernels.lora_b import grouped_lora_b

        invoke_shared_token_delta_reduce(
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            token_rank=token_rank,
            config=config["reduce"],
        )
        grouped_lora_b(
            token_rank,
            b_down,
            token_delta,
            token_route,
            destination_offsets=(0,),
            config=delta_config,
            intermediate_top_k=1,
        )
        invoke_shared_token_delta_tail(
            down_rows=down_rows,
            pair_to_row=row_state.pair_to_row,
            token_delta=token_delta,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            config=config["tail"],
        )

    def shared_one_pass_finalize(
        self,
        row_state,
        *,
        down_rows: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        from sglang.srt.lora.moe.kernels.finalize import invoke_shared_one_pass

        invoke_shared_one_pass(
            down_rows=down_rows,
            pair_to_row=row_state.pair_to_row,
            bridge=bridge,
            b_down=b_down,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            config=config,
        )

    def mapped_down_lora_a_input(
        self,
        row_state,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sentinel mappings are uninitialized; their route blocks are skipped."""
        return activation.view(-1, activation.shape[-1]), row_state.pair_to_row

    def gateup_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError
