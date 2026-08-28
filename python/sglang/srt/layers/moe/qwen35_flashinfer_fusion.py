"""Qwen3.5 integration for FlashInfer MNNVL CuTe DSL AllReduce fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.layers.communicator import (
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


def is_supported_forward_mode(forward_mode: ForwardMode) -> bool:
    return forward_mode in (
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.TARGET_VERIFY,
    )


def resolve_max_m(model_runner) -> int:
    """Use framework token bounds as the workspace-capacity source of truth."""
    server_args = resolving_view(model_runner.server_args)
    decode_config = server_args.cuda_graph_config.decode
    prefill_config = server_args.cuda_graph_config.prefill
    candidates = [
        server_args.cutedsl_moe_max_num_tokens(),
        model_runner.max_running_requests,
        decode_config.max_bs,
        prefill_config.max_bs,
        *(decode_config.bs or []),
        *(prefill_config.bs or []),
    ]
    positive = [
        int(value) for value in candidates if value is not None and int(value) > 0
    ]
    if not positive:
        raise RuntimeError("framework reported no positive fusion workspace M bound")
    return max(positive)


@dataclass(frozen=True)
class Qwen35MoeFinalizeHandoff:
    """Unfinalized routed output plus the separately gated shared contribution."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor
    m: int

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor,
        m: int,
    ) -> Qwen35MoeFinalizeHandoff:
        top_k = int(deferred_output.top_k)
        return cls(
            routed_output=deferred_output.gemm2_out.view(
                -1, deferred_output.gemm2_out.shape[-1]
            ),
            expert_weights=deferred_output.expert_weights.view(-1, top_k)[:m],
            permuted_indices=deferred_output.expanded_idx_to_permuted_idx.view(
                -1, top_k
            )[:m],
            gated_shared_output=gated_shared_output,
            m=int(m),
        )


class Qwen35FlashInferFusionService:
    """A lightweight model handle for the process-local FlashInfer workspace."""

    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        rms_epsilon: float,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)
        self.max_m: int | None = None
        self._workspace = None

    @property
    def is_prepared(self) -> bool:
        return self._workspace is not None

    def prepare(self, *, max_m: int) -> None:
        if self._workspace is not None:
            assert self.max_m is not None
            if int(max_m) > self.max_m:
                raise RuntimeError(
                    f"fusion workspace is already prepared for M_max={self.max_m}; "
                    f"refusing M_max={max_m}"
                )
            return
        from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
            get_flashinfer_mnnvl_cutedsl_ar_fusion,
        )

        workspace = get_flashinfer_mnnvl_cutedsl_ar_fusion(
            hidden_size=self.hidden_size,
            top_k=self.top_k,
            max_m=int(max_m),
            rms_epsilon=self.rms_epsilon,
            # GemmaRMSNorm.gemma_weight is already checkpoint weight + 1.
            weight_bias=0.0,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m

    def supports(self, m: int) -> bool:
        if self._workspace is None or self.max_m is None:
            return False
        return 1 <= int(m) <= self.max_m and self._workspace.supports(m)

    def finalize(
        self,
        handoff: Qwen35MoeFinalizeHandoff,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_finalize(handoff, residual, gamma)
        assert self._workspace is not None
        return self._workspace.moe_finalize_all_reduce_rms_norm(
            routed_output=handoff.routed_output,
            expert_weights=handoff.expert_weights,
            permuted_indices=handoff.permuted_indices,
            gated_shared_output=handoff.gated_shared_output,
            residual=residual,
            gamma=gamma,
        )

    def all_reduce_residual_rms_norm(
        self,
        local_contribution: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_matrix(local_contribution, "local_contribution")
        self._validate_matrix(residual, "residual", m=local_contribution.shape[0])
        self._validate_gamma(gamma)
        if not self.supports(local_contribution.shape[0]):
            raise ValueError(f"unsupported M={local_contribution.shape[0]}")
        assert self._workspace is not None
        return self._workspace.all_reduce_residual_rms_norm(
            local_contribution=local_contribution,
            residual=residual,
            gamma=gamma,
        )

    def _validate_finalize(
        self,
        handoff: Qwen35MoeFinalizeHandoff,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> None:
        if not self.supports(handoff.m):
            raise ValueError(f"unsupported M={handoff.m}")
        self._validate_matrix(handoff.routed_output, "routed_output", exact_m=False)
        expected_metadata = (handoff.m, self.top_k)
        if tuple(handoff.expert_weights.shape) != expected_metadata:
            raise ValueError("expert_weights must have shape [M, top_k]")
        if tuple(handoff.permuted_indices.shape) != expected_metadata:
            raise ValueError("permuted_indices must have shape [M, top_k]")
        if handoff.expert_weights.dtype != torch.bfloat16:
            raise ValueError("expert_weights must be BF16")
        if handoff.permuted_indices.dtype != torch.int32:
            raise ValueError("permuted_indices must be Int32")
        self._validate_matrix(
            handoff.gated_shared_output, "gated_shared_output", m=handoff.m
        )
        self._validate_matrix(residual, "residual", m=handoff.m)
        self._validate_gamma(gamma)

    def _validate_matrix(
        self,
        tensor: torch.Tensor,
        name: str,
        *,
        m: int | None = None,
        exact_m: bool = True,
    ) -> None:
        if tensor.ndim != 2 or tensor.shape[1] != self.hidden_size:
            raise ValueError(f"{name} must have shape [M, hidden_size]")
        if m is not None and exact_m and tensor.shape[0] != int(m):
            raise ValueError(f"{name} has the wrong M dimension")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous BF16")

    def _validate_gamma(self, gamma: torch.Tensor) -> None:
        if (
            tuple(gamma.shape) != (self.hidden_size,)
            or gamma.dtype != torch.bfloat16
            or not gamma.is_contiguous()
        ):
            raise ValueError("gamma must be contiguous BF16 [hidden_size]")


class Qwen35FlashInferLayerCommunicator(LayerCommunicator):
    """Qwen-only hooks; generic LayerCommunicator remains backend agnostic."""

    fusion_service: Qwen35FlashInferFusionService | None = None

    def prepare_attn(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format: str = "",
        post_residual_addition=None,
    ):
        if isinstance(hidden_states, Qwen35MoeFinalizeHandoff):
            if not self.should_use_finalize(forward_batch, hidden_states.m):
                raise RuntimeError("received deferred MoE output on an ineligible path")
            if residual is None:
                raise RuntimeError("deferred MoE finalize requires residual input")
            if not hasattr(self.input_layernorm, "gemma_weight"):
                raise RuntimeError("deferred Qwen finalize requires GemmaRMSNorm")
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            return self.fusion_service.finalize(
                hidden_states, residual, self.input_layernorm.gemma_weight
            )
        return super().prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        if self.should_use_all_reduce_rms_norm(
            forward_batch, int(hidden_states.shape[0]), residual
        ):
            assert self.fusion_service is not None and residual is not None
            return self.fusion_service.all_reduce_residual_rms_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.gemma_weight,
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def should_use_all_reduce_rms_norm(
        self,
        forward_batch: ForwardBatch,
        m: int,
        residual: Optional[torch.Tensor],
    ) -> bool:
        communicate_fn = self._communicate_with_all_reduce_and_layer_norm_fn
        norm_fn = getattr(communicate_fn, "func", communicate_fn)
        residual_input_mode = getattr(communicate_fn, "keywords", {}).get(
            "residual_input_mode"
        )
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and residual is not None
            and hasattr(self.post_attention_layernorm, "gemma_weight")
            and norm_fn
            is CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
            and self._context.attn_dp_size == 1
            and parallel.attn_tp_size == parallel.tp_size
            and not get_exec().comm.enable_quant_communications
        )

    def should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and self.layer_scatter_modes.mlp_mode is not ScatterMode.SCATTERED
            and parallel.moe_ep_size == 1
        )

    def _common_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        parallel = get_parallel()
        return bool(
            self.fusion_service is not None
            and self.fusion_service.is_prepared
            and is_supported_forward_mode(forward_batch.forward_mode)
            and self.fusion_service.supports(m)
            and not is_dp_attention_enabled()
            and parallel.attn_cp_size == 1
            and not get_attn_tp_context().input_scattered
            and get_moe_a2a_backend().is_none()
            and self._context.tp_size > 1
        )

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        m = (
            int(forward_batch.input_ids.shape[0])
            if getattr(forward_batch, "input_ids", None) is not None
            else 0
        )
        if self.should_use_finalize(forward_batch, m):
            # The Qwen model consumes the final layer's handoff with its final
            # GemmaRMSNorm, so this is intentionally also true for that layer.
            return True
        return super().should_fuse_mlp_allreduce_with_next_layer(forward_batch)


def prepare_qwen35_flashinfer_fusion(model, model_runner) -> None:
    service = getattr(model, "flashinfer_mnnvl_cutedsl_fusion", None)
    if service is None:
        return
    if model_runner.server_args.enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(max_m=resolve_max_m(model_runner))
    logger.info(
        "Prepared Qwen3.5 FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d",
        service.max_m,
    )
