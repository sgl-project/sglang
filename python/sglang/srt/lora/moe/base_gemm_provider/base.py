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

    def run_fused_act(
        self,
        row_state,
        family: str,
        *,
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
            f"{self.contract.key} has no fused-act implementation for {family!r}"
        )

    def mapped_down_lora_a_input(
        self,
        row_state,
        activation: torch.Tensor,
    ) -> MappedLoraAInput:
        """Give the provider's activation rows to a standalone grouped down-A.

        ``src2dst`` already holds one activation row for each routed pair. A
        sentinel pair's entry stays uninitialized and no kernel reads it: the
        route puts every sentinel in a block labelled ``-1``, and the kernel
        skips those blocks. Both row domains carry ``src2dst``, so this works
        for either one.
        """
        expected = self.act_out_shape(row_state)
        if tuple(activation.shape) != expected:
            raise ValueError(
                f"mapped down-A activation must be {expected}, got "
                f"{tuple(activation.shape)}"
            )
        if activation.dtype != self.contract.lora_activation_dtype:
            raise TypeError(
                "mapped down-A activation dtype must match the provider "
                f"contract {self.contract.lora_activation_dtype}"
            )
        if not activation.is_contiguous():
            raise ValueError("mapped down-A activation rows must be contiguous")
        if (
            row_state.src2dst.ndim != 1
            or row_state.src2dst.dtype != torch.int32
            or row_state.src2dst.device != activation.device
            or not row_state.src2dst.is_contiguous()
        ):
            raise ValueError(
                "mapped down-A pair-to-row metadata must be contiguous 1-D "
                "int32 on the activation device"
            )
        return MappedLoraAInput(
            rows=activation.view(-1, activation.shape[-1]),
            pair_to_row=row_state.src2dst,
        )

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

        Call this after the base down GEMM. It reaches a row only through
        ``src2dst``, so both row domains work. The later finalize must then
        run with no pair delta. ``bridge`` is the pair-major down-A output.
        ``b_down`` holds the down-B weights of every virtual expert.
        """
        # Imported here so this module keeps to msgspec and torch.
        from sglang.srt.lora.moe.lora_b import invoke_down_b_into_base

        invoke_down_b_into_base(
            down_rows=down_out.view(-1, self.hidden_size),
            src2dst=row_state.src2dst,
            bridge=bridge,
            b_down=b_down,
            routing=routing,
            config=config,
        )

    def run_shared_rank_finalize(
        self,
        row_state,
        *,
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
        raise NotImplementedError(f"{self.contract.key} has no shared-rank finalizer")

    def run_shared_rank_reduce(
        self,
        row_state,
        *,
        bridge: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        """Launch the shared-rank reduction. It does not wait for the base W2 GEMM."""
        raise NotImplementedError(f"{self.contract.key} has no shared-rank reduction")

    def finish_shared_rank_finalize(
        self,
        row_state,
        *,
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
        raise NotImplementedError(f"{self.contract.key} has no shared B tail")

    # The runner allocates every buffer, so it asks the provider for the shapes.
    def gateup_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError
