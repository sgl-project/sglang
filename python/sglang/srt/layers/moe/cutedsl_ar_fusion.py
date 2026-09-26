"""FlashInfer MNNVL CuTe DSL AllReduce fusion, shared across architectures.

Two patterns share one workspace, both consumed at the next layer's input
RMSNorm: AR + residual + RMSNorm, and the same with the MoE finalize and the
shared-expert add folded in when the runner hands back a MoeFinalizeHandoff.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.layers.boundary_layout import SumGroup, TokenAxis
from sglang.srt.layers.communicator import (
    FfnExitFusion,
    FusedMlpInput,
    HandoffOutput,
    LayerCommunicator,
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


class MoeFinalizeHandoff(HandoffOutput, frozen=True):
    """Unfinalized routed output plus the separately gated shared contribution.
    The next layer's fused finalize + AR + add + norm takes it; ``finish`` is
    the MoE's own unfused tail, for any other reader."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor
    m: int
    finish: Callable[[], torch.Tensor]

    def complete(self) -> torch.Tensor:
        return self.finish()

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor,
        m: int,
        reduce: Callable[[torch.Tensor], torch.Tensor],
    ) -> MoeFinalizeHandoff:
        """``reduce`` is the MoE's own sum of its finalized output."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

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
            finish=lambda: reduce(
                finalize_flashinfer_trtllm_deferred_output(
                    deferred_output, gated_shared_output
                )
            ),
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
    # The FFN exit's CuteDSL kernels, which install() gives from what it knows
    # of this layer and the one after it.
    _cutedsl_exit_fusions: tuple = ()

    def install(
        self,
        service: CuteDSLFusionService,
        *,
        hands_off_finalize: bool,
        next_input_absorbs: bool,
        output_is_replicated: bool,
    ) -> None:
        """Take the model's fusion service and choose the FFN exit's CuteDSL
        kernels: the MoE may hand off its finalize when it can and something
        after it takes the handoff (``hands_off_finalize``); the FFN may leave
        its all-reduce to the next layer's AR + norm when that layer has one,
        unless a replicated output follows the reduction (moving it would scale
        that by tp)."""
        self.fusion_service = service
        self._cutedsl_exit_fusions = (
            (self._defer_moe_finalize_cutedsl,) if hands_off_finalize else ()
        ) + (
            (self._absorb_all_reduce_cutedsl,)
            if next_input_absorbs and not output_is_replicated
            else ()
        )
        self._ffn_exit_fusions = self._select_ffn_exit_fusions()

    def _select_attn_input_fusions(self):
        return (
            self._finalize_output_and_update_and_read_residual_cutedsl,
            self._reduce_output_and_update_and_read_residual_cutedsl,
            *super()._select_attn_input_fusions(),
        )

    def _finalize_output_and_update_and_read_residual_cutedsl(
        self, owed, residual, forward_batch, post_residual_addition
    ):
        """Finish a deferred MoE finalize with the all-reduce, residual add and
        input norm, when this layer's kernel takes the batch; otherwise the
        handoff is completed by its producer's own tail."""
        if not isinstance(owed, MoeFinalizeHandoff):
            return None
        gamma = _fused_norm_gamma(self.input_layernorm)
        if gamma is None or not self._should_use_finalize(forward_batch, owed.m):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        assert self.fusion_service is not None
        return self.fusion_service.finalize(
            handoff=owed, residual=residual, gamma=gamma
        )

    def _reduce_output_and_update_and_read_residual_cutedsl(
        self, owed, residual, forward_batch, post_residual_addition
    ):
        """Complete the all-reduce the previous layer left with the residual add
        and input norm."""
        if not isinstance(owed, UnreducedOutput) or not (
            self._can_consume_post_moe_all_reduce(
                forward_batch, int(owed.partial.shape[0])
            )
        ):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        assert self.fusion_service is not None
        return self.fusion_service.all_reduce_residual_rms_norm(
            local_contribution=owed.partial,
            residual=residual,
            gamma=_fused_norm_gamma(self.input_layernorm),
        )

    def _select_mlp_input_fusions(self):
        fusions = super()._select_mlp_input_fusions()
        parallel = get_parallel()
        if (
            TokenAxis.ATTN_TP_SCATTER not in self.input_rows.sharded
            # The workspace sums over TP, which is then the attention-TP group.
            and parallel.attn_tp_size == parallel.tp_size
            and _fused_norm_gamma(self.post_attention_layernorm) is not None
        ):
            return (
                FusedMlpInput(
                    completes=SumGroup.ATTN_TP,
                    run=self._mlp_input_reduce_output_and_update_and_read_residual_cutedsl,
                    may_return_new_residual=True,
                ),
                *fusions,
            )
        return fusions

    def _mlp_input_reduce_output_and_update_and_read_residual_cutedsl(
        self, hidden_states, residual, forward_batch
    ):
        """The attention output's all-reduce with the residual add and the
        post-attention norm."""
        if not (
            self._common_eligible(forward_batch, int(hidden_states.shape[0]))
            and residual is not None
            and not get_exec().comm.enable_quant_communications
        ):
            return None
        assert self.fusion_service is not None
        return self.fusion_service.all_reduce_residual_rms_norm(
            local_contribution=hidden_states,
            residual=residual,
            gamma=_fused_norm_gamma(self.post_attention_layernorm),
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

    def _select_ffn_exit_fusions(self):
        return (*self._cutedsl_exit_fusions, *super()._select_ffn_exit_fusions())

    def _defer_moe_finalize_cutedsl(
        self, forward_batch: ForwardBatch
    ) -> Optional[FfnExitFusion]:
        """The MoE hands its unfinalized output to the next layer's finalize +
        all-reduce + norm."""
        if self._should_use_finalize(
            forward_batch, int(forward_batch.input_ids.shape[0])
        ):
            return FfnExitFusion.DEFER_MOE_FINALIZE
        return None

    def _absorb_all_reduce_cutedsl(
        self, forward_batch: ForwardBatch
    ) -> Optional[FfnExitFusion]:
        """The next layer's AR + norm takes the post-experts all-reduce."""
        if self._can_consume_post_moe_all_reduce(
            forward_batch, int(forward_batch.input_ids.shape[0])
        ):
            return FfnExitFusion.NEXT_INPUT
        return None

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
            # The FFN runs on the full rows, not each rank's own slice.
            and TokenAxis.ATTN_TP_SCATTER
            not in self._batch_steps(forward_batch).ffn_input_rows.sharded
        )


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
    hands_off = 0
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
        hands_off_finalize = bool(can_defer_finalize(layer)) and has_consumer
        hands_off += hands_off_finalize
        communicator.install(
            service,
            hands_off_finalize=hands_off_finalize,
            # False on the last layer: the final norm does not all-reduce.
            next_input_absorbs=successor is not None
            and isinstance(
                successor.layer_communicator, CuteDSLFusionLayerCommunicator
            ),
            output_is_replicated=requires_local_reduction is not None
            and bool(requires_local_reduction(layer)),
        )
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        hands_off,
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
