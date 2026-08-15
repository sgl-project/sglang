"""Base-MoE provider seam for the MoE LoRA execution engine.

``MoeBaseProvider`` is the per-quant seam. The runner drives
prepare (S1: permute) -> gateup (S2) -> act_with_delta (S3, the gate/up LoRA
injection point) -> down (S4) -> finalize (S5, router weight and routed scaling
applied exactly once over base + LoRA delta), all launched from Python so
future overlap topologies can place stream joins between stages.

It provides the BASE model's half of the MoE — the LoRA half belongs to the
runner, which owns route views, LoRA kernels, and pipeline buffers. A provider
owns its physical row layout, resident weight format, activation join,
finalize, and workspace config; the only reason it is decomposed into stages
at all is so the runner can inject LoRA between them. No stock ``MoeRunner``
is involved.

Only fields that some code actually reads live on the contract. Descriptive
axes (resident weight format, row domain, activation family, coefficient
precision) belong to the campaign's case schema until a provider or a selector
consumes them; declaring them here without a reader would document intent that
execution does not honor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.lora.moe.routing import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MoeBaseProviderContract(msgspec.Struct, frozen=True, kw_only=True):
    """Semantics of one physical base provider that the runner must honor."""

    key: str
    # Column order of the provider's own gate/up GEMM output. The LoRA delta is
    # always canonical [gate | up]; these flags describe the BASE rows only.
    gate_first: bool
    interleaved: bool
    gate_up_output_dtype: torch.dtype
    lora_delta_dtype: torch.dtype
    lora_activation_dtype: torch.dtype
    supported_output_dtypes: tuple[torch.dtype, ...]

    def validate_output_dtype(self, dtype: torch.dtype) -> None:
        if dtype not in self.supported_output_dtypes:
            supported = ", ".join(str(item) for item in self.supported_output_dtypes)
            raise ValueError(
                f"{self.key} cannot write MoE LoRA output dtype {dtype}; "
                f"supported dtypes: {supported}"
            )


class MappedLoraAInput(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-owned activation rows exposed through a stable LoRA-A ABI.

    ``rows`` is a 2-D physical provider row domain. ``pair_to_row`` is a
    contiguous int32 tensor mapping each canonical routed pair to one row;
    ``-1`` denotes an invalid pair. Keeping this descriptor on the provider
    seam lets the runner use mapped grouped LoRA-A without learning the
    provider workspace's private field names or layout details.
    """

    rows: torch.Tensor
    pair_to_row: torch.Tensor


class MoeBaseProvider:
    """Interface. One instance per (layer, quant type), bound to quant_info.

    Subclasses must expose the semantic geometry below. The runner sizes its
    LoRA buffers from it, so it is part of the seam rather than something to
    read off a provider-private payload: a packed FP8/NVFP4 provider cannot be
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
        """One for non-gated activations, two for canonical gate/up."""
        raise NotImplementedError

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ):
        """Permute inputs into the provider's physical row domain.

        Returns a provider-private workspace object; the runner treats it as
        opaque and passes it back to the later stages.  When the runner
        supplies its workspace, every per-forward tensor must come from that
        address-stable allocator so the provider is CUDA-graph capturable.
        """
        raise NotImplementedError

    def configure_base_pdl(self, *, gateup_to_middle: bool) -> None:
        """Prepare requested base-GEMM producer variants before graph capture."""

        if gateup_to_middle:
            raise NotImplementedError(
                f"{self.contract.key} does not implement plan-local base-GEMM PDL"
            )

    def base_pdl_state(self) -> dict[str, object]:
        return {
            "provider": self.contract.key,
            "producer_signal_supported": False,
            "gateup_signal_compiled": False,
            "down_signal_compiled": False,
        }

    def gateup(
        self,
        ws,
        out: torch.Tensor,
        *,
        produce_pdl: bool = False,
    ) -> None:
        raise NotImplementedError

    def release_prepared_inputs(self, ws) -> None:
        """Free whatever ``prepare`` allocated once the gate/up GEMM is done.

        The provider owns its workspace members, so it performs the release;
        the runner only knows that the prepared inputs are dead after S2.
        """
        raise NotImplementedError

    def act_with_delta(
        self,
        ws,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu_mul",
        consume_base_pdl: bool = False,
    ) -> None:
        raise NotImplementedError

    def down(
        self,
        ws,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        ws,
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
        contribution in canonical ``[T, K, H]`` pair order.
        """
        raise NotImplementedError

    # Optional fused LoRA sites.  The implementation name is explicit so the
    # config/benchmark can force Triton versus an injected CuTe candidate;
    # unsupported providers fail before allocating or launching anything.
    def fused_middle_implementations(self, family: str) -> tuple[str, ...]:
        return ()

    def supports_fused_middle(
        self,
        family: str,
        *,
        activation: str,
        implementation: str = "triton",
    ) -> bool:
        return implementation in self.fused_middle_implementations(family)

    def run_fused_middle(
        self,
        ws,
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
            f"{self.contract.key} has no {implementation!r} fused-middle "
            f"implementation for {family!r}"
        )

    def mapped_down_lora_a_input(
        self,
        ws,
        activation: torch.Tensor,
    ) -> MappedLoraAInput | None:
        """Expose provider activation rows for standalone grouped down-A.

        Providers return ``None`` unless their activation row domain has a
        stable canonical-pair mapping.  The default preserves the pair-major
        bridge used by existing plans and future providers.
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

    def supports_down_b_scatter(self) -> bool:
        """Whether S4's output rows admit the down-B scatter-add epilogue.

        True only when the provider's physical row domain has a stable
        canonical-pair-to-row mapping it can hand the scatter launch, the
        same property behind ``mapped_down_lora_a_input``.
        """
        return False

    def run_down_b_scatter(
        self,
        ws,
        *,
        down_out: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        config: Mapping[str, int],
    ) -> None:
        """Scatter-add each pair's unweighted down-B delta into ``down_out``.

        Runs AFTER the base down GEMM: the SAME one-launch sliced down-B
        tiling targets ``down_out``'s physical rows through the provider's
        pair-to-row mapping instead of a dense pair-major delta buffer, and
        the materialized finalize then runs in no-pair-delta mode.
        ``bridge`` is the canonical pair-major down-A output and ``b_down``
        the flattened per-virtual-expert down-B groups.
        """
        raise NotImplementedError(
            f"{self.contract.key} has no down-B scatter-into-base epilogue"
        )

    def run_shared_rank_finalize(
        self,
        ws,
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
        ws,
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
        ws,
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
    def gateup_out_shape(self, ws) -> tuple[int, ...]:
        raise NotImplementedError

    def act_out_shape(self, ws) -> tuple[int, ...]:
        raise NotImplementedError

    def down_out_shape(self, ws) -> tuple[int, ...]:
        raise NotImplementedError

    def validate_runtime_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> None:
        """Validate the semantic boundary shared by every shipped provider."""
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                f"{self.contract.key} requires BF16 MoE/LoRA activations, got "
                f"{hidden_states.dtype}"
            )
        self.contract.validate_output_dtype(output_dtype)
