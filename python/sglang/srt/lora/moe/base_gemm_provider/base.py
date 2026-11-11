"""Base MoE stages with insertion points for LoRA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.kernels.ops.lora.common.route_view import RouteView
    from sglang.srt.lora.moe.plan import FinalizeFamily, MoeLoraLaunchConfig
    from sglang.srt.lora.workspace import LoraWorkspace


# Token-count cutoff for the vectorized top-k gather.
_SMALL_FINALIZE_MAX_TOKENS = 64

# What a row domain's _input_buffers() hook returns beside the GEMM input rows:
# the encoding's other row-state fields by name. None is a field the mode
# leaves unset (the per-token row scales of a static build).
RowBuffers = dict[str, Optional[torch.Tensor]]


class MoeBaseProviderContract(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    quant_info_cls: type
    # Base output order; the LoRA delta is always [gate | up].
    gate_first: bool
    interleaved: bool
    gate_up_output_dtype: torch.dtype
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype


def expected_rows_per_expert(num_pairs: int, num_experts: int) -> int:
    """Rounded-up mean rows per expert; use one for an empty batch."""
    return (num_pairs - 1) // num_experts + 1 if num_pairs else 1


def prepare_buffer(
    workspace, name: str, shape, *, dtype, device, zero_on_first_allocation=False
) -> torch.Tensor:
    if workspace is not None:
        return workspace.tensor(
            name,
            shape,
            dtype=dtype,
            device=device,
            zero_on_first_allocation=zero_on_first_allocation,
        )
    if zero_on_first_allocation:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def admit_weight_layout(quant_info) -> int:
    """Validate [E, S*I, H] / [E, H, I] weights and return S."""
    if quant_info.intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    # Sub-byte formats pack K elements per byte.
    pack = getattr(quant_info, "elements_per_byte", 1)
    expected_w2 = (
        quant_info.num_local_experts,
        quant_info.hidden_size,
        quant_info.intermediate_size // pack,
    )
    if quant_info.w2_weight.shape != expected_w2:
        raise ValueError(
            f"w2_weight must be {expected_w2}, got {tuple(quant_info.w2_weight.shape)}"
        )
    if (
        quant_info.w13_weight.ndim != 3
        or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
        or quant_info.w13_weight.shape[2] != quant_info.hidden_size // pack
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
    # What the provider derives from the base weights when bound, if anything;
    # online base-weight updates cannot refresh it, so the LoRA layer declines them.
    derived_base_weights: str | None = None

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
        workspace: LoraWorkspace | None = None,
    ):
        """Prepare rows with graph-stable buffers when a workspace is supplied."""
        raise NotImplementedError

    def gateup(self, row_state, out: torch.Tensor) -> None:
        raise NotImplementedError

    def release_prepared_inputs(self, row_state) -> None:
        raise NotImplementedError

    def _act_quant_target(
        self, row_state, fused: bool, config: Mapping[str, int] | None
    ):
        """Where this activation may write the down GEMM's input: FP8 (rows,
        scales, group width), block-scaled rows, or None to leave it to down().
        ``config`` is the fused kernel's tile configuration."""
        return None

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
        from sglang.kernels.ops.lora.moe.activation_delta import act_delta

        act_quant = self._act_quant_target(row_state, fused=False, config=None)
        act_delta(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            act_quant=act_quant,
        )
        self._set_act_quant_ready(row_state, act_quant is not None)

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
        from sglang.kernels.ops.lora.moe.fused_act import fused_b_act

        act_quant = self._act_quant_target(row_state, fused=True, config=config)
        fused_b_act(
            activation=activation,
            base_gateup=base_gateup,
            act_rows=act_rows,
            act_pairs=act_pairs,
            pair_to_row=row_state.pair_to_row,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            act_quant=act_quant,
        )
        self._set_act_quant_ready(row_state, act_quant is not None)

    @staticmethod
    def _set_act_quant_ready(row_state, written: bool) -> None:
        """Row states that carry ``act_quant_ready`` say whether the activation
        that just ran wrote the down GEMM's input; down() runs its quantizer
        exactly when it did not. Cleared as well as set: a row state that is
        activated again must not keep the earlier activation's answer."""
        if hasattr(row_state, "act_quant_ready"):
            row_state.act_quant_ready = written

    def down(
        self,
        row_state,
        act_out: torch.Tensor,
        out: torch.Tensor,
        lora: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """``lora`` (down-B weights, bridge planes, slots) is only passed to
        providers that set ``supports_epilogue_down_lora``."""
        raise NotImplementedError

    def down_output(self, row_state, workspace: LoraWorkspace) -> torch.Tensor:
        return workspace.tensor(
            "base:down",
            self.down_out_shape(row_state),
            dtype=torch.bfloat16,
            device=row_state.pair_to_row.device,
        )

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
        from sglang.kernels.ops.lora.moe.finalize import invoke_small_finalize
        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm

        num_tokens, hidden = output.shape
        scaling = routed_scaling_factor if routed_scaling_factor is not None else 1.0
        if num_tokens <= _SMALL_FINALIZE_MAX_TOKENS and topk_weights.is_contiguous():
            # Gather top-k rows together instead of issuing dependent row loads.
            invoke_small_finalize(
                down_out.view(-1, hidden),
                output,
                row_state.pair_to_row,
                topk_weights,
                scaling,
                lora_delta=lora_delta,
            )
            return
        post_reorder_deepgemm(
            down_out.view(-1, hidden),
            output,
            row_state.pair_to_row,
            topk_ids,
            topk_weights.contiguous(),
            topk_ids.shape[1],
            num_tokens,
            hidden,
            scaling,
            lora_delta=lora_delta,
        )

    def shared_outer_finalize(
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
        family: FinalizeFamily,
        launch_config: MoeLoraLaunchConfig,
        workspace: LoraWorkspace,
        token_route: RouteView | None,
    ) -> None:
        """Apply shared down-B with the plan's fused or staged finalize."""
        from sglang.srt.lora.moe.plan import FinalizeFamily, Site

        if topk_weights.dtype != torch.float32:
            raise TypeError(
                "topk_weights must stay FP32 until the finalize applies them"
            )
        if family is FinalizeFamily.SHARED_ONE_PASS:
            from sglang.kernels.ops.lora.moe.finalize import invoke_shared_one_pass

            invoke_shared_one_pass(
                num_local_experts=self.num_local_experts,
                down_rows=down_rows,
                pair_to_row=row_state.pair_to_row,
                bridge=bridge,
                b_down=b_down,
                routing=routing,
                topk_weights=topk_weights,
                routed_scaling_factor=routed_scaling_factor,
                output=output,
                config=launch_config.shared_one_pass,
            )
            return
        if family is not FinalizeFamily.SHARED_TOKEN_DELTA:
            raise ValueError(f"not a shared-outer finalize family: {family}")
        if token_route is None:
            raise ValueError("shared token route was not constructed")

        num_tokens = output.shape[0]
        token_rank = workspace.tensor(
            "finalize:shared_token_rank",
            (num_tokens, bridge.shape[1]),
            dtype=bridge.dtype,
            device=bridge.device,
        )
        token_delta = workspace.tensor(
            "finalize:shared_token_delta",
            (num_tokens, self.hidden_size),
            dtype=self.contract.lora_delta_dtype,
            device=bridge.device,
        )
        from sglang.kernels.ops.lora.moe.finalize import (
            invoke_shared_token_delta_reduce,
            invoke_shared_token_delta_tail,
        )
        from sglang.kernels.ops.lora.moe.lora_b import grouped_lora_b

        invoke_shared_token_delta_reduce(
            num_local_experts=self.num_local_experts,
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            token_rank=token_rank,
            config=launch_config.shared_token_delta["reduce"],
        )
        grouped_lora_b(
            token_rank,
            b_down,
            token_delta,
            token_route,
            destination_offsets=(0,),
            config=launch_config.for_b(Site.DOWN),
            pair_bridge=True,
        )
        invoke_shared_token_delta_tail(
            num_local_experts=self.num_local_experts,
            down_rows=down_rows,
            pair_to_row=row_state.pair_to_row,
            token_delta=token_delta,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            config=launch_config.shared_token_delta["tail"],
        )

    def gateup_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError
