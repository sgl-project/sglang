"""FlashInfer MNNVL CuTe DSL AllReduce fusion, shared across architectures.

Two patterns share one workspace, both consumed at the next layer's input
RMSNorm: AR + residual + RMSNorm, and the same with the MoE finalize and the
shared-expert add folded in when the runner hands back a MoeFinalizeHandoff.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, Sequence

import msgspec
import torch

from sglang.srt.layers.communicator import (
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    ScatterMode,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_disagg,
    get_exec,
    get_flags,
    get_parallel,
)

_LayerPredicate = Callable[[torch.nn.Module], bool]

logger = logging.getLogger(__name__)


def _fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The multiplier as applied -- GemmaRMSNorm's pre-folded w + 1. None declines."""
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None


def _is_supported_forward_mode(forward_mode: ForwardMode) -> bool:
    return forward_mode in (
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.TARGET_VERIFY,
    )


def _resolve_max_m(*, max_running_requests: int | None) -> int:
    decode_config = get_exec().graph.cuda_graph_config.decode
    prefill_config = get_exec().graph.cuda_graph_config.prefill
    candidates = [
        cutedsl_moe_max_num_tokens(),
        max_running_requests,
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


class MoeFinalizeHandoff(msgspec.Struct, frozen=True):
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
    ) -> MoeFinalizeHandoff:
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


class CuteDSLFusionService:
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
            # _fused_norm_gamma() already returns the multiplier as applied.
            weight_bias=0.0,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m

    def supports(self, m: int) -> bool:
        """False before prepare(), so it doubles as the readiness check."""
        return self._workspace is not None and self._workspace.supports(m)

    def finalize(
        self,
        handoff: MoeFinalizeHandoff,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert self._workspace is not None
        return self._workspace.all_reduce_residual_rms_norm(
            local_contribution=local_contribution,
            residual=residual,
            gamma=gamma,
        )


class CuteDSLFusionLayerCommunicator(LayerCommunicator):
    fusion_service: CuteDSLFusionService | None = None

    # The runner can defer and a successor or the final norm consumes the handoff.
    may_defer_moe_finalize: bool = False
    # False on the last layer: the final norm does not all-reduce.
    successor_absorbs_all_reduce: bool = False
    # A replicated output follows the reduction; moving it would scale that by tp.
    owes_local_reduction: bool = False

    def prepare_attn(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format: str = "",
        post_residual_addition=None,
    ):
        if isinstance(hidden_states, MoeFinalizeHandoff):
            if not self._should_use_finalize(forward_batch, hidden_states.m):
                raise RuntimeError(
                    "received deferred MoE output on an ineligible path "
                    f"(M={hidden_states.m}, mode={forward_batch.forward_mode})"
                )
            if residual is None:
                raise RuntimeError("deferred MoE finalize requires residual input")
            gamma = _fused_norm_gamma(self.input_layernorm)
            if gamma is None:
                raise RuntimeError(
                    "deferred MoE finalize requires a fusable RMSNorm flavour"
                )
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            hidden_states, residual = self.fusion_service.finalize(
                handoff=hidden_states, residual=residual, gamma=gamma
            )
            return self._finish_prepare_attn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if (
            residual is not None
            and isinstance(hidden_states, UnreducedOutput)
            and self._can_consume_post_moe_all_reduce(
                forward_batch, int(hidden_states.partial.shape[0])
            )
        ):
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            hidden_states, residual = self.fusion_service.all_reduce_residual_rms_norm(
                local_contribution=hidden_states.partial,
                residual=residual,
                gamma=_fused_norm_gamma(self.input_layernorm),
            )
            return self._finish_prepare_attn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
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
        if self._should_use_all_reduce_rms_norm(
            forward_batch, int(hidden_states.shape[0]), residual
        ):
            assert self.fusion_service is not None and residual is not None
            return self.fusion_service.all_reduce_residual_rms_norm(
                local_contribution=hidden_states,
                residual=residual,
                gamma=_fused_norm_gamma(self.post_attention_layernorm),
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def _should_use_all_reduce_rms_norm(
        self,
        forward_batch: ForwardBatch,
        m: int,
        residual: Optional[torch.Tensor],
    ) -> bool:
        communicate_fn = self._communicate_with_all_reduce_and_layer_norm_fn
        if isinstance(communicate_fn, functools.partial):
            norm_fn = communicate_fn.func
            residual_input_mode = communicate_fn.keywords.get("residual_input_mode")
        else:
            norm_fn = communicate_fn
            residual_input_mode = None
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and residual is not None
            and _fused_norm_gamma(self.post_attention_layernorm) is not None
            and norm_fn
            is CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
            and self._context.attn_dp_size == 1
            and parallel.attn_tp_size == parallel.tp_size
            and not get_exec().comm.enable_quant_communications
        )

    def _should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        return (
            self._common_eligible(forward_batch, m) and get_parallel().moe_ep_size == 1
        )

    def _can_consume_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Incoming, and independent of this layer's own successor."""
        return (
            self._common_eligible(forward_batch, m)
            and _fused_norm_gamma(self.input_layernorm) is not None
            and not get_exec().comm.enable_quant_communications
        )

    def _can_absorb_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Outgoing: skip our own all-reduce because the next layer absorbs it."""
        return (
            self.successor_absorbs_all_reduce
            and not self.owes_local_reduction
            and self._can_consume_post_moe_all_reduce(forward_batch, m)
        )

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Deferring skips the post-experts all-reduce on the promise of a handoff."""
        if not self.may_defer_moe_finalize:
            return False
        if m is None:
            m = int(forward_batch.input_ids.shape[0])
        return self._should_use_finalize(forward_batch, m)

    def _common_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        parallel = get_parallel()
        return bool(
            self.fusion_service is not None
            and _is_supported_forward_mode(forward_batch.forward_mode)
            and self.fusion_service.supports(m)
            and not is_dp_attention_enabled()
            # Also forces moe_dp_size == 1, so a hybrid EP x MoE-TP reduction
            # always merges into the one TP reduction the workspace performs.
            and parallel.attn_cp_size == 1
            and not get_attn_tp_context().input_scattered
            and get_moe_a2a_backend().is_none()
            and self._context.tp_size > 1
            # Restates the base's moe-cp, MOE_FULL and SCATTERED refusals.
            and self.layer_scatter_modes.mlp_mode is ScatterMode.FULL
        )

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        m = int(forward_batch.input_ids.shape[0])
        if self.should_defer_moe_finalize(forward_batch, m):
            return True
        if self._can_absorb_post_moe_all_reduce(forward_batch, m):
            return True
        return super().should_fuse_mlp_allreduce_with_next_layer(forward_batch)


def install_cutedsl_fusion(
    layers: Sequence[torch.nn.Module],
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize: _LayerPredicate,
    requires_local_reduction: _LayerPredicate | None = None,
    final_norm_consumes_handoff: bool = False,
    label: str,
) -> CuteDSLFusionService | None:
    """One shared workspace handle per fusion-enabled layer, or None.

    Every entry of ``layers`` must carry a ``layer_communicator``.
    """
    if get_flags().moe.in_speculative_scope:
        # A draft shares the target's process, which holds one workspace.
        return None

    fusion_layers = [
        layer
        for layer in layers
        if isinstance(layer.layer_communicator, CuteDSLFusionLayerCommunicator)
    ]
    if not fusion_layers:
        return None

    # The workspace compiles one epsilon and cannot see a per-layer one.
    for layer in fusion_layers:
        for norm in (
            layer.layer_communicator.input_layernorm,
            layer.layer_communicator.post_attention_layernorm,
        ):
            if _fused_norm_gamma(norm) is None:
                continue
            if float(norm.variance_epsilon) != float(rms_epsilon):
                raise RuntimeError(
                    f"{label} CuTe DSL fusion compiles one workspace for "
                    f"rms_epsilon={rms_epsilon}, but a fused norm uses "
                    f"{norm.variance_epsilon}"
                )

    service = CuteDSLFusionService(
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
    )
    for index, layer in enumerate(layers):
        communicator = layer.layer_communicator
        if not isinstance(communicator, CuteDSLFusionLayerCommunicator):
            continue
        successor = layers[index + 1] if index + 1 < len(layers) else None
        if successor is None:
            has_consumer = final_norm_consumes_handoff
        else:
            has_consumer = isinstance(
                successor.layer_communicator, CuteDSLFusionLayerCommunicator
            )
        communicator.fusion_service = service
        communicator.may_defer_moe_finalize = (
            bool(can_defer_finalize(layer)) and has_consumer
        )
        communicator.successor_absorbs_all_reduce = successor is not None and (
            isinstance(successor.layer_communicator, CuteDSLFusionLayerCommunicator)
        )
        communicator.owes_local_reduction = (
            requires_local_reduction is not None and requires_local_reduction(layer)
        )
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        sum(layer.layer_communicator.may_defer_moe_finalize for layer in fusion_layers),
    )
    return service


def prepare_cutedsl_fusion(
    model: torch.nn.Module, *, max_running_requests: int | None
) -> None:
    """Build the workspace of every service installed anywhere in ``model``.

    Scans the module tree, so a wrapper holding the model as a submodule (a VLM's
    language_model) needs no hook of its own.
    """
    communicators = [
        communicator
        for module in model.modules()
        # Most modules carry no layer_communicator.
        if isinstance(
            communicator := module.__dict__.get("layer_communicator"),
            CuteDSLFusionLayerCommunicator,
        )
    ]
    if not communicators or any(c.fusion_service is None for c in communicators):
        raise ValueError(
            "--flashinfer-allreduce-fusion-backend cutedsl is set, but "
            f"{type(model).__name__} installed no CuTe DSL fusion service, so no "
            "allreduce fusion would run at all. Drop the flag, or choose 'auto', "
            "'trtllm' or 'mnnvl'."
        )
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    services = {id(c.fusion_service): c.fusion_service for c in communicators}
    max_m = _resolve_max_m(max_running_requests=max_running_requests)
    for service in services.values():
        service.prepare(max_m=max_m)
    logger.info(
        "Prepared the FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d "
        "(%d fused layers)",
        max_m,
        len(communicators),
    )
