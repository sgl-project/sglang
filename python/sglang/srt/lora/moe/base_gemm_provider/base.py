"""The base half of the MoE forward pass, in stages.

A provider runs the base model. It defines the row layout, the weight format,
the activation, the finalize, and the workspace shapes. The runner adds the
LoRA half.

The stages are prepare, gateup, act_with_delta, down, and finalize. They are
separate for one reason: the runner inserts LoRA work between them. Python
launches each stage, so a caller can also place a stream join between two
stages. No stock ``MoeRunner`` takes part.
"""

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
    # These two flags give the column order of the base gate/up GEMM output.
    # The LoRA delta is always [gate | up].
    gate_first: bool
    interleaved: bool
    gate_up_output_dtype: torch.dtype
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype


class MappedLoraAInput(msgspec.Struct, frozen=True, kw_only=True):
    """``pair_to_row`` gives one row of ``rows`` for each routed pair.

    It is a contiguous int32 tensor. A value of ``-1`` marks an invalid pair.
    """

    rows: torch.Tensor
    pair_to_row: torch.Tensor


class MoeBaseProvider:
    """One instance for each layer and quantization type.

    Each instance binds to one ``quant_info``. This interface declares the four
    size properties below. A packed FP8 or NVFP4 provider cannot report them
    from its resident tensors. Those tensors have packed shapes. A packed shape
    does not give these sizes.
    """

    contract: MoeBaseProviderContract

    @property
    def num_local_experts(self) -> int:
        raise NotImplementedError

    @property
    def intermediate_size(self) -> int:
        """The intermediate width of one tensor-parallel shard, in logical elements."""
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    def gate_up_slices(self) -> int:
        raise NotImplementedError

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ):
        """Permute the inputs into the provider's own row layout.

        The return value is an opaque state object for the later stages. If the
        runner passes a workspace, take every per-forward tensor from it. Its
        addresses are stable, and CUDA-graph capture needs that.
        """
        raise NotImplementedError

    def gateup(self, row_state, out: torch.Tensor) -> None:
        raise NotImplementedError

    def release_prepared_inputs(self, row_state) -> None:
        """Free what ``prepare`` allocated. The gate/up GEMM is the last reader."""
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
        consume_base_pdl: bool = False,
    ) -> None:
        raise NotImplementedError

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
        """Reduce the top-k pairs into ``output``, in a fixed order.

        Apply the router weight and the routed scaling once, to the sum of the
        base pair and ``lora_delta``. ``lora_delta`` holds the unweighted
        down-LoRA result in ``[T, K, H]`` pair order.
        """
        raise NotImplementedError

    # Each implementation has a name. A config or a benchmark then selects
    # Triton or an injected CuTe kernel.
    def fused_act_implementations(self, family: str) -> tuple[str, ...]:
        return ()

    def supports_fused_act(
        self,
        family: str,
        *,
        activation: str,
        implementation: str = "triton",
    ) -> bool:
        return implementation in self.fused_act_implementations(family)

    def run_fused_act(
        self,
        row_state,
        family: str,
        *,
        implementation: str,
        activation: str,
        base_gateup: torch.Tensor,
        act_masked: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        raise NotImplementedError(
            f"{self.contract.key} has no {implementation!r} fused-act "
            f"implementation for {family!r}"
        )

    def mapped_down_lora_a_input(
        self,
        row_state,
        activation: torch.Tensor,
    ) -> MappedLoraAInput | None:
        """Give the provider's activation rows to a standalone grouped down-A.

        Return ``None`` when the activation rows have no stable pair-to-row
        map. The caller then keeps the pair-major bridge.
        """

        return None

    def fused_finalize_implementations(
        self, family: str, ownership: str
    ) -> tuple[str, ...]:
        return ()

    def supports_fused_finalize(
        self,
        family: str,
        ownership: str,
        *,
        implementation: str = "triton",
    ) -> bool:
        return implementation in self.fused_finalize_implementations(family, ownership)

    def supports_down_b_into_base(self) -> bool:
        """Report whether the down rows accept the down-B into-base add.

        Return ``True`` only with a stable pair-to-row map, as for
        ``mapped_down_lora_a_input``.
        """
        return False

    def run_down_b_into_base(
        self,
        row_state,
        *,
        down_out: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        config: Mapping[str, int],
    ) -> None:
        """Add each pair's unweighted down-B result into ``down_out``.

        Call this after the base down GEMM. It uses the provider's pair-to-row
        map. The later finalize must then run with no pair delta. ``bridge`` is
        the pair-major down-A output. ``b_down`` holds the down-B weights of
        every virtual expert.
        """
        raise NotImplementedError(
            f"{self.contract.key} has no down-B into-base epilogue"
        )

    def run_shared_rank_finalize(
        self,
        row_state,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, Mapping[str, int]],
    ) -> None:
        raise NotImplementedError(
            f"{self.contract.key} has no {implementation!r} shared-rank finalizer"
        )

    def run_shared_rank_reduce(
        self,
        row_state,
        *,
        implementation: str,
        bridge: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        """Launch the shared-rank reduction. It does not wait for the base W2 GEMM."""
        raise NotImplementedError(
            f"{self.contract.key} has no {implementation!r} shared-rank reduction"
        )

    def finish_shared_rank_finalize(
        self,
        row_state,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        """Finish the shared-rank path.

        Wait for the base W2 GEMM and the reduction. Then finalize the base
        rows and add the shared-B tail.
        """
        raise NotImplementedError(
            f"{self.contract.key} has no {implementation!r} shared B tail"
        )

    # The runner allocates every buffer, so it asks the provider for the shapes.
    def gateup_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError
