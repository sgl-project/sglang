"""GLM-5-Next integration for FlashInfer MNNVL CuTe DSL AllReduce fusion.

Both patterns the CuTe DSL backend compiles end in ``residual + x`` followed by
RMSNorm, and mHC hyper-connections give GLM-5-Next neither boundary in that
shape: ``hc_post`` mixes the reduced value into ``hc_mult`` residual streams with
per-token weights, and ``hc_ffn_pre`` normalizes only after a Sinkhorn mix. The
workspace is therefore compiled without its residual add, and the model reads the
pre-norm output -- the cross-rank sum alone -- then combines it as it always did.
"""

from __future__ import annotations

import logging
from functools import partial

import torch

from sglang.srt.layers.communicator import ScatterMode, get_attn_tp_context
from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateSummableTensorPairFn,
    MHCCommunicateWithAllReduceAndLayerNormFn,
    MHCLayerCommunicator,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel, get_spec

logger = logging.getLogger(__name__)

# Decode and its speculative equivalent only. Admitting EXTEND measured neutral
# per forward and cost 9 ms of mean TTFT on GLM-5.3-Flash at 8192-token
# prefills, over 12 runs across 3 server processes.
_SUPPORTED_FORWARD_MODES = frozenset((ForwardMode.DECODE, ForwardMode.TARGET_VERIFY))

_LOGGED_DECISIONS: set[tuple[str, bool]] = set()


def _log_decision_once(site: str, engaged: bool, detail) -> bool:
    """Report each boundary's first verdict, so a silent fallback is visible."""
    key = (site, engaged)
    if key not in _LOGGED_DECISIONS:
        _LOGGED_DECISIONS.add(key)
        logger.info(
            "GLM-5-Next MNNVL CuTe DSL %s: %s (%s)",
            site,
            "engaged" if engaged else "declined",
            detail(),
        )
    return engaged


def resolve_max_m(model_runner) -> int:
    """Largest token count the fused path may serve, from the decode bounds.

    Prefill bounds are excluded because ``_SUPPORTED_FORWARD_MODES`` is. An
    under-estimate costs the optimization and nothing else -- ``supports()``
    declines and the ordinary path runs.
    """
    decode_config = get_exec().graph.cuda_graph_config.decode
    spec = get_spec()
    tokens_per_request = (
        (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    )
    requests = [
        int(value)
        for value in (
            model_runner.max_running_requests,
            decode_config.max_bs,
            *(decode_config.bs or []),
        )
        if value is not None and int(value) > 0
    ]
    if not requests:
        raise RuntimeError("framework reported no positive decode request bound")
    return max(requests) * tokens_per_request


class Glm5NextFlashInferFusionService:
    """A model handle for the process-local, residual-free FlashInfer workspace."""

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
        self._gamma: torch.Tensor | None = None
        self._norm_scratch: torch.Tensor | None = None

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
            weight_bias=0.0,
            fuse_residual=False,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m
        # The compiled kernel always writes a normalized output the model never
        # reads; a unit gamma keeps that half well-defined without a weight, and
        # one buffer absorbs it for every layer and both boundaries.
        self._gamma = torch.ones(
            self.hidden_size, dtype=torch.bfloat16, device=workspace.device
        )
        self._norm_scratch = torch.empty(
            (self.max_m, self.hidden_size),
            dtype=torch.bfloat16,
            device=workspace.device,
        )

    def supports(self, m: int) -> bool:
        if self._workspace is None or self.max_m is None:
            return False
        return 1 <= int(m) <= self.max_m and self._workspace.supports(m)

    def all_reduce(self, local_contribution: torch.Tensor) -> torch.Tensor:
        m = int(local_contribution.shape[0])
        if local_contribution.ndim != 2 or local_contribution.shape[1] != (
            self.hidden_size
        ):
            raise ValueError("local_contribution must have shape [M, hidden_size]")
        if (
            local_contribution.dtype != torch.bfloat16
            or not local_contribution.is_contiguous()
        ):
            raise ValueError("local_contribution must be contiguous BF16")
        if not self.supports(m):
            raise ValueError(f"unsupported M={m}")
        assert self._workspace is not None and self._gamma is not None
        assert self._norm_scratch is not None
        return self._workspace.all_reduce(
            local_contribution=local_contribution,
            gamma=self._gamma,
            norm_scratch=self._norm_scratch[:m],
        )


class Glm5NextFlashInferMHCLayerCommunicator(MHCLayerCommunicator):
    """GLM-5-Next-only hooks; MHCLayerCommunicator stays backend agnostic."""

    fusion_service: Glm5NextFlashInferFusionService | None = None

    def _post_init_communicate(self):
        super()._post_init_communicate()
        # Everything but the forward mode, M, and the scattered-input flag is
        # frozen once the communicate callables are chosen.
        parallel = get_parallel()
        communicate_fn = self._communicate_with_all_reduce_and_layer_norm_fn
        if isinstance(communicate_fn, partial):
            norm_fn = communicate_fn.func
            residual_input_mode = communicate_fn.keywords.get("residual_input_mode")
        else:
            norm_fn = communicate_fn
            residual_input_mode = None
        shape_eligible = (
            not is_dp_attention_enabled()
            and self._context.attn_dp_size == 1
            and self._context.tp_size > 1
            and parallel.attn_tp_size == parallel.tp_size
            and parallel.attn_cp_size == 1
            and get_moe_a2a_backend().is_none()
            and not get_exec().comm.enable_quant_communications
        )
        self._attn_output_eligible = (
            shape_eligible
            and norm_fn
            is MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
        )
        self._mlp_output_eligible = (
            shape_eligible
            and parallel.moe_ep_size == 1
            and self._communicate_summable_tensor_pair_fn
            is MHCCommunicateSummableTensorPairFn._trivial
        )
        # Published by should_defer_mlp_allreduce and consumed by
        # postprocess_layer, so one verdict drives both the MLP's skip and the
        # reduction that replaces it. One bool holds only because a layer runs
        # to completion between the two: two-batch overlap decomposes them into
        # separate operations and interleaves microbatches through one
        # communicator, so whoever adds a GLM-5-Next TBO strategy to
        # OperationsStrategy.init_new_tbo has to make this per-microbatch.
        self._mlp_allreduce_deferred = False

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        m = int(hidden_states.shape[0])
        if _log_decision_once(
            "attention-output all-reduce",
            self._attn_output_eligible and self._forward_eligible(forward_batch, m),
            lambda: f"m={m} static={self._attn_output_eligible}",
        ):
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.all_reduce(hidden_states)
            return self.mhc.attn_to_mlp(
                hidden_states, residual, out_norm=self.post_attention_layernorm
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def should_defer_mlp_allreduce(self, forward_batch: ForwardBatch) -> bool:
        m = int(forward_batch.input_ids.shape[0])
        self._mlp_allreduce_deferred = _log_decision_once(
            "MLP-output all-reduce",
            self._mlp_output_eligible and self._forward_eligible(forward_batch, m),
            lambda: f"m={m} static={self._mlp_output_eligible}",
        )
        return self._mlp_allreduce_deferred

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        if self._mlp_allreduce_deferred:
            self._mlp_allreduce_deferred = False
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.all_reduce(hidden_states)
        return super().postprocess_layer(hidden_states, residual, forward_batch)

    def _forward_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        return bool(
            self.fusion_service is not None
            and self.fusion_service.is_prepared
            and forward_batch.forward_mode in _SUPPORTED_FORWARD_MODES
            and not get_attn_tp_context().input_scattered
            and self.fusion_service.supports(m)
        )


def prepare_glm5_next_flashinfer_fusion(model, model_runner) -> None:
    service = model.flashinfer_mnnvl_cutedsl_fusion
    if service is None:
        return
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(max_m=resolve_max_m(model_runner))
    logger.info(
        "Prepared GLM-5-Next FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d",
        service.max_m,
    )
