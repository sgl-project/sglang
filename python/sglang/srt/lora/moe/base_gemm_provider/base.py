"""A provider owns the BASE model's half of the MoE — its physical row layout,
resident weight format, activation join, finalize, and workspace config — and
the runner owns the LoRA half. The only reason a provider is decomposed into
stages (prepare -> gateup -> act_with_delta -> down -> finalize) is so the
runner can inject LoRA between them, and every stage is launched from Python
so overlap topologies can place stream joins between them. No stock
``MoeRunner`` is involved.
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
    # Column order of the provider's own gate/up GEMM output. The LoRA delta is
    # always [gate | up]; these flags describe the BASE rows only.
    gate_first: bool
    interleaved: bool
    gate_up_output_dtype: torch.dtype
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype


class MappedLoraAInput(msgspec.Struct, frozen=True, kw_only=True):
    """``pair_to_row`` is a contiguous int32 tensor mapping each routed pair
    to one row; ``-1`` denotes an invalid pair.
    """

    rows: torch.Tensor
    pair_to_row: torch.Tensor


class MoeBaseProvider:
    """One instance per (layer, quant type), bound to quant_info.

    The semantic geometry below is on the seam rather than read off a
    provider-private payload because a packed FP8/NVFP4 provider cannot be
    asked for dimensions its resident tensors no longer carry.
    """

    contract: MoeBaseProviderContract

    @property
    def num_local_experts(self) -> int:
        raise NotImplementedError

    @property
    def intermediate_size(self) -> int:
        """Local (TP-sharded) intermediate width in semantic elements."""
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
        """Permute inputs into the provider's physical row domain.

        Returns an opaque provider-private workspace for the later stages.
        When the runner supplies its workspace, every per-forward tensor must
        come from that address-stable allocator so the provider is CUDA-graph
        capturable.
        """
        raise NotImplementedError

    def gateup(self, row_state, out: torch.Tensor) -> None:
        raise NotImplementedError

    def release_prepared_inputs(self, row_state) -> None:
        """Free whatever ``prepare`` allocated; the prepared inputs are dead
        once the gate/up GEMM is done.
        """
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
        """Fixed-order weighted top-k reduction into ``output``.

        Router coefficient and routed scaling are applied exactly once over
        ``base_pair + lora_delta``; ``lora_delta`` is the unweighted down-LoRA
        contribution in ``[T, K, H]`` pair order.
        """
        raise NotImplementedError

    # The implementation name is explicit so the config/benchmark can force
    # Triton versus an injected CuTe candidate.
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
        """Expose provider activation rows for standalone grouped down-A.

        ``None`` unless the activation row domain has a stable pair-to-row
        mapping; the default keeps the pair-major bridge.
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
        """Whether S4's output rows admit the down-B into-base epilogue; true
        only with a stable pair-to-row mapping, as for
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
        """Scatter-add each pair's unweighted down-B delta into ``down_out``.

        Runs AFTER the base down GEMM, through the provider's pair-to-row
        mapping, and the materialized finalize then runs in no-pair-delta
        mode.  ``bridge`` is the pair-major down-A output and ``b_down`` the
        flattened per-virtual-expert down-B groups.
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
        """Launch the shared rank reduction independently of base W2."""
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
        """Join base W2/reduce, then finalize base and add the shared B tail."""
        raise NotImplementedError(
            f"{self.contract.key} has no {implementation!r} shared B tail"
        )

    # Buffer shape helpers so the runner owns every allocation.
    def gateup_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, row_state) -> tuple[int, ...]:
        raise NotImplementedError
