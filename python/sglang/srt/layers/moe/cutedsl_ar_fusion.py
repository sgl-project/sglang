"""Model-agnostic FlashInfer MNNVL CuTe DSL AllReduce fusion.

Two fusion patterns share one workspace, both consumed at the *next* layer's
input RMSNorm:

* AllReduce + residual add + RMSNorm, replacing ``prepare_mlp``'s collective.
* MoE finalize + shared-expert add + AllReduce + residual add + RMSNorm, when
  the MoE runner hands back an unfinalized :class:`MoeFinalizeHandoff` instead
  of a tensor.

Nothing here is per-architecture. The only thing that differs between model
families is which attribute of their RMSNorm holds the gamma the fused kernel
reads, and :func:`fused_norm_gamma` answers that from the norm module itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.arg_groups.overrides import cutedsl_moe_max_num_tokens, resolving_view
from sglang.srt.layers.communicator import (
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


def fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The gamma the fused RMSNorm reads, or None for a norm it cannot serve.

    The kernel computes ``x * (gamma + weight_bias)`` with ``weight_bias=0``, so
    the gamma has to be the multiplier as applied. ``GemmaRMSNorm`` keeps that
    as ``gemma_weight`` (the checkpoint weight pre-folded to ``w + 1``) while a
    plain ``RMSNorm`` applies ``weight`` directly.

    Returning None rather than raising is load-bearing: the eligibility
    predicates call this to decline a layer whose norm is a flavour the kernel
    has no gamma for.
    """
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None


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
        cutedsl_moe_max_num_tokens(model_runner.server_args),
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
class MoeFinalizeHandoff:
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
            # fused_norm_gamma() hands over the multiplier as applied, so the
            # kernel's x * (gamma + weight_bias) wants no bias for any flavour.
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
        handoff: MoeFinalizeHandoff,
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
        handoff: MoeFinalizeHandoff,
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


class CuteDSLFusionLayerCommunicator(LayerCommunicator):
    """Fusion hooks; the generic LayerCommunicator stays backend agnostic."""

    fusion_service: CuteDSLFusionService | None = None

    # Whether this layer's MoE runner can hand back an unfinalized output.
    # Recorded by install_cutedsl_fusion(); the finalize pattern is unreachable
    # without one, while the plain AR+RMSNorm pattern stays available.
    experts_can_defer_finalize: bool = False

    # Whether a handoff produced here would actually be consumed: the next
    # layer takes one in prepare_attn, or -- for the last layer -- the model's
    # own final norm does. Recorded by install_cutedsl_fusion().
    handoff_has_consumer: bool = False

    def prepare_attn(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format: str = "",
        post_residual_addition=None,
    ):
        if isinstance(hidden_states, MoeFinalizeHandoff):
            if not self.should_use_finalize(forward_batch, hidden_states.m):
                raise RuntimeError(
                    "received deferred MoE output on an ineligible path "
                    f"(M={hidden_states.m}, mode={forward_batch.forward_mode})"
                )
            if residual is None:
                raise RuntimeError("deferred MoE finalize requires residual input")
            gamma = fused_norm_gamma(self.input_layernorm)
            if gamma is None:
                raise RuntimeError(
                    "deferred MoE finalize requires a fusable RMSNorm flavour"
                )
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            hidden_states, residual = self.fusion_service.finalize(
                hidden_states, residual, gamma
            )
            return self._finish_prepare_attn(hidden_states, residual, forward_batch)
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
                fused_norm_gamma(self.post_attention_layernorm),
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
            and fused_norm_gamma(self.post_attention_layernorm) is not None
            and norm_fn
            is CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
            and self._context.attn_dp_size == 1
            and parallel.attn_tp_size == parallel.tp_size
            and not get_exec().comm.enable_quant_communications
        )

    def should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        """Consumer side: whether prepare_attn here can absorb a handoff.

        Deliberately independent of this layer's own MoE runner -- consuming is
        a property of the fused kernel and the topology, not of who produced.
        """
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and self.layer_scatter_modes.mlp_mode is not ScatterMode.SCATTERED
            and parallel.moe_ep_size == 1
        )

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Producer side: whether this layer's MoE may hand one off.

        Beyond consumer eligibility this needs a runner that can produce a
        handoff and somewhere for it to land. Deferring skips the post-experts
        all-reduce on the promise of a handoff, so a layer whose successor
        cannot consume must not defer -- nothing would perform the collective.
        The producer's own eligibility stands in for the successor's; every
        fusion layer of a model shares its scatter modes and topology.
        """
        if not (self.experts_can_defer_finalize and self.handoff_has_consumer):
            return False
        if m is None:
            m = int(forward_batch.input_ids.shape[0])
        return self.should_use_finalize(forward_batch, m)

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
        if self.should_defer_moe_finalize(forward_batch):
            # True even on the last layer when the model's final norm consumes
            # the handoff; handoff_has_consumer is what decides that.
            return True
        return super().should_fuse_mlp_allreduce_with_next_layer(forward_batch)


def install_cutedsl_fusion(
    layers,
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize,
    final_norm_consumes_handoff: bool = False,
    label: str,
) -> CuteDSLFusionService | None:
    """Give every fusion-enabled layer in ``layers`` one shared workspace handle.

    ``layers`` are the decoder layers this rank actually built -- each must carry
    a ``layer_communicator``, so a PP-padded list is sliced to the local range
    before it gets here.

    The workspace compiles per ``(hidden_size, top_k, rms_epsilon)``, which every
    MoE layer of a model shares, so one handle serves the whole model. Returns it,
    or None when no layer got a fusion communicator.

    ``can_defer_finalize(layer) -> bool`` says whether that layer's MoE runner can
    hand back an unfinalized output; layers where it cannot keep the plain
    AR+RMSNorm fusion and never advertise the finalize pattern.

    ``final_norm_consumes_handoff`` says the model's own final norm closes out
    the last layer's handoff. Without it the last layer never defers, because
    nothing would perform the all-reduce that deferring skips.
    """
    fusion_layers = [
        layer
        for layer in layers
        if isinstance(layer.layer_communicator, CuteDSLFusionLayerCommunicator)
    ]
    if not fusion_layers:
        return None

    service = CuteDSLFusionService(
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
    )
    for index, layer in enumerate(layers):
        communicator = layer.layer_communicator
        if not isinstance(communicator, CuteDSLFusionLayerCommunicator):
            continue
        # A handoff is consumed by the next layer's prepare_attn, so the next
        # layer must have a communicator that can absorb one. The last layer
        # falls to the model's final norm.
        successor = layers[index + 1] if index + 1 < len(layers) else None
        if successor is None:
            has_consumer = final_norm_consumes_handoff
        else:
            has_consumer = isinstance(
                successor.layer_communicator, CuteDSLFusionLayerCommunicator
            )
        communicator.fusion_service = service
        communicator.experts_can_defer_finalize = bool(can_defer_finalize(layer))
        communicator.handoff_has_consumer = has_consumer
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        sum(
            layer.layer_communicator.experts_can_defer_finalize
            and layer.layer_communicator.handoff_has_consumer
            for layer in fusion_layers
        ),
    )
    return service


def prepare_cutedsl_fusion(
    service: CuteDSLFusionService | None, model_runner, *, label: str
) -> None:
    """Compile the workspace before graph capture; a no-op without a handle."""
    if service is None:
        return
    if model_runner.server_args.enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(max_m=resolve_max_m(model_runner))
    logger.info(
        "Prepared %s FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d",
        label,
        service.max_m,
    )
