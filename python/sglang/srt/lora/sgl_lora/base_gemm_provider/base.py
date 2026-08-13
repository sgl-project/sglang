"""Base-MoE provider seam for the SGL LoRA execution engine.

``MoeBaseProvider`` is the per-quant seam. The runner drives
prepare (S1: permute) -> gateup (S2) -> act_with_delta (S3, the gate/up LoRA
injection point) -> down (S4) -> finalize (S5, router weight and routed scaling
applied exactly once over base + LoRA pair delta), all launched from Python so
future overlap topologies can place stream joins between stages.

It provides the BASE model's half of the MoE — the LoRA half belongs to the
runner, which owns route views, LoRA kernels, and pipeline buffers. A provider
owns its physical row layout, resident weight format, activation join,
finalize, and workspace policy; the only reason it is decomposed into stages
at all is so the runner can inject LoRA between them. No stock ``MoeRunner``
is involved.

Only fields that some code actually reads live on the contract. Descriptive
axes (resident weight format, row domain, activation family, coefficient
precision) belong to the campaign's case schema until a provider or a selector
consumes them; declaring them here without a reader would document intent that
execution does not honor.
"""

from __future__ import annotations

import msgspec
import torch


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
                f"{self.key} cannot write sgl_lora output dtype {dtype}; "
                f"supported dtypes: {supported}"
            )


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

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
    ):
        """Permute inputs into the provider's physical row domain.

        Returns a provider-private workspace object; the runner treats it as
        opaque and passes it back to the later stages.
        """
        raise NotImplementedError

    def gateup(self, ws, out: torch.Tensor) -> None:
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
    ) -> None:
        raise NotImplementedError

    def down(self, ws, act_out: torch.Tensor, out: torch.Tensor) -> None:
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
        pair_delta: torch.Tensor | None = None,
    ) -> None:
        """Fixed-order weighted top-k reduction into ``output``.

        Router coefficient and routed scaling are applied exactly once over
        ``base_pair + pair_delta``; ``pair_delta`` is the unweighted down-LoRA
        contribution in canonical ``[T, K, H]`` pair order.
        """
        raise NotImplementedError

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
