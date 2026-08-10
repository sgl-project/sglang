"""FlashInfer MNNVL CuTe DSL all-reduce fusion backend.

Selected with ``--flashinfer-allreduce-fusion-backend cute-dsl``. Two fusion
patterns share one workspace:

* AllReduce + residual add + RMSNorm — the drop-in replacement for the
  ``mnnvl`` backend, dispatched from :mod:`sglang.srt.layers.layernorm`.
* MoE finalize + shared-expert add + AllReduce + residual add + RMSNorm —
  reached when the MoE runner defers its finalize (see
  :mod:`sglang.srt.layers.moe.moe_runner.flashinfer_trtllm`), which removes the
  separate finalize kernel and its round trip through HBM.

The workspace compiles CuTe DSL kernels for a single static shape and completes
a symmetric-memory rendezvous, so it is built once during model-runner warmup
(``pre_initialize_workspaces``) and never inside a forward pass. A shape that
the compiled workspace does not serve falls back to the unfused path.

FlashInfer ships routing profiles for GB300 H=8192 / top-k=10 only.
:func:`_build_config` re-targets those protocol boundaries and tuning presets at
the running model's hidden size and top-k; the LL/BT/HT crossover points are
inherited from the H=8192 measurements and are therefore approximate elsewhere.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

BACKEND_NAME = "cute-dsl"

# Warp size and the bf16 elements per 16-byte vector, mirroring the constants
# the FlashInfer HT device kernel derives its shard split from.
_WARP_SIZE = 32
_VEC_BF16 = 8

# The HT kernel dedicates two warps plus the reduction warps to non-consumer
# roles, so the consumer count has this much headroom under the 1024 limit.
_HT_MAX_CONSUMER_THREADS = 1024 - 3 * _WARP_SIZE

_unavailable = False


def _import_flashinfer():
    from flashinfer.comm.mnnvl_cutedsl import (
        KernelTarget,
        MNNVLCuteDSLConfig,
        MRangeDispatch,
        ProtocolKind,
        StaticProfile,
    )

    return KernelTarget, MNNVLCuteDSLConfig, MRangeDispatch, ProtocolKind, StaticProfile


def is_available() -> bool:
    """Whether the FlashInfer build exposes the CuTe DSL backend."""
    global _unavailable
    if _unavailable:
        return False
    try:
        from flashinfer.comm.mnnvl_cutedsl_ar import (  # noqa: F401
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        _import_flashinfer()
    except (ImportError, AttributeError) as e:
        _unavailable = True
        logger.warning(
            "FlashInfer MNNVL CuTe DSL allreduce fusion is not available (%s).", e
        )
        return False
    return True


def _ht_shard_split(hidden_size: int) -> Optional[tuple[int, int]]:
    """``(consumer_threads, vectors_per_thread)`` for the HT persistent kernel.

    The kernel splits a token into ``consumer_threads * 8 * vectors_per_thread``
    element shards and additionally requires ``consumer_threads`` to divide the
    token's 16-byte vector count. FlashInfer's H=8192 presets use 512 threads x 2
    vectors, i.e. one shard per token with two vectors loaded per consumer; this
    reproduces that shape for any hidden size by taking the widest consumer
    count that still leaves at least two vectors per thread.
    """
    packs = hidden_size // _VEC_BF16
    limit = min(_HT_MAX_CONSUMER_THREADS, packs // 2)
    for consumer_threads in range(
        limit - limit % _WARP_SIZE, _WARP_SIZE - 1, -_WARP_SIZE
    ):
        if packs % consumer_threads == 0:
            return consumer_threads, packs // consumer_threads
    return None


def _ht_reduction_warps(
    hidden_size: int, tp_size: int, preferred: int
) -> Optional[int]:
    """Reduction warps that evenly cover one rank's slice of a token.

    The HT reduction shard is ``hidden / 8 / tp`` vectors wide and must divide
    across ``reduction_warps * 32`` threads; ``preferred`` is FlashInfer's tuned
    value, which we only ever step down from.
    """
    packs_per_shard = (hidden_size // _VEC_BF16) // tp_size
    for warps in (8, 4, 2, 1):
        if warps <= preferred and packs_per_shard % (warps * _WARP_SIZE) == 0:
            return warps
    return None


def _ht_tunings(hidden_size: int, tp_size: int):
    """Re-target the GB300 HT presets at ``hidden_size``; None when unreachable."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
        HTAllReduceTuning,
        HTFinalizeTuning,
    )

    split = _ht_shard_split(hidden_size)
    if split is None:
        return None
    consumer_threads, vectors_per_thread = split

    finalize_warps = _ht_reduction_warps(
        hidden_size, tp_size, HTFinalizeTuning().reduction_warps
    )
    all_reduce_warps = _ht_reduction_warps(
        hidden_size, tp_size, HTAllReduceTuning().reduction_warps
    )
    if finalize_warps is None or all_reduce_warps is None:
        return None

    return (
        HTFinalizeTuning(
            consumer_threads=consumer_threads,
            vectors_per_thread=vectors_per_thread,
            reduction_warps=finalize_warps,
        ),
        HTAllReduceTuning(
            consumer_threads=consumer_threads,
            vectors_per_thread=vectors_per_thread,
            reduction_warps=all_reduce_warps,
        ),
    )


# Measured with autotune_mnnvl_cutedsl.py at this hidden size; the shipped
# bounds and presets are GB300 H=8192 numbers that do not transfer.
_MEASURED_BOUNDS: dict[tuple[int, int], tuple[tuple, tuple]] = {
    (8, 6144): ((12, 48, 703, None), (64, 192, 1024, None)),
    (4, 6144): ((32, 48, 703, None), (24, 192, 1024, None)),
}


def _measured_ll_all_reduce(tp_size: int, hidden_size: int):
    # publish_threads * elements_per_thread sets the publish block count, which
    # at small M is the only parallelism available: 8 gives H=8192 eight blocks
    # but H=6144 only six.
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll import LLAllReduceTuning
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll.protocol import LLCollectiveTuning

    if hidden_size != 6144 or tp_size not in (4, 8):
        return None
    ept = int(os.environ.get("SGLANG_TEST_CUTEDSL_LL_AR_EPT", "4"))
    return LLAllReduceTuning(
        publish_elements_per_thread=ept,
        publish_threads=128,
        collective=LLCollectiveTuning(cluster_size=8, rank_lanes=1, threads=128),
    )


def _build_config(tp_size: int, hidden_size: int, top_k: int, dtype: torch.dtype):
    """A single-profile routing config for this model's static shape."""
    (
        KernelTarget,
        MNNVLCuteDSLConfig,
        MRangeDispatch,
        ProtocolKind,
        StaticProfile,
    ) = _import_flashinfer()
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
        LL_ALL_REDUCE_GB300_TP8_H8192,
        LL_ALL_REDUCE_GB300_TP16_H8192,
        LL_FINALIZE_GB300_TP8_H8192_K10,
        LL_FINALIZE_GB300_TP16_H8192_K10,
    )

    ht = _ht_tunings(hidden_size, tp_size)
    if ht is None:
        logger.warning(
            "MNNVL CuTe DSL: hidden_size=%d does not admit an HT shard split "
            "at tp_size=%d; falling back to the mnnvl backend.",
            hidden_size,
            tp_size,
        )
        return None
    ht_finalize, ht_all_reduce = ht

    wide_tp = tp_size >= 16
    ll_finalize = (
        LL_FINALIZE_GB300_TP16_H8192_K10 if wide_tp else LL_FINALIZE_GB300_TP8_H8192_K10
    )
    ll_all_reduce = (
        LL_ALL_REDUCE_GB300_TP16_H8192 if wide_tp else LL_ALL_REDUCE_GB300_TP8_H8192
    )
    # FlashInfer's measured GB300 crossovers; TP16 shifts LL's window down
    # because each rank publishes a smaller slice.
    finalize_bounds = (7, 52, 703, None) if wide_tp else (23, 48, 703, None)
    all_reduce_bounds = (5, 512, 959, None) if wide_tp else (15, 256, 1024, None)
    stock = os.environ.get("SGLANG_TEST_CUTEDSL_STOCK") == "1"
    measured = None if stock else _MEASURED_BOUNDS.get((tp_size, hidden_size))
    if measured is not None:
        finalize_bounds, all_reduce_bounds = measured
    measured_ll_ar = None if stock else _measured_ll_all_reduce(tp_size, hidden_size)
    if measured_ll_ar is not None:
        ll_all_reduce = measured_ll_ar

    def target(protocol, preset):
        return KernelTarget(protocol=protocol, preset=preset)

    profile = StaticProfile(
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=dtype,
        finalize_routes=MRangeDispatch(
            upper_bounds=finalize_bounds,
            targets=(
                target(ProtocolKind.LL, ll_finalize),
                target(ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0),
                target(ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1),
                target(ProtocolKind.HT, ht_finalize),
            ),
        ),
        all_reduce_routes=MRangeDispatch(
            upper_bounds=all_reduce_bounds,
            targets=(
                target(ProtocolKind.LL, ll_all_reduce),
                target(ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0),
                target(ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1),
                target(ProtocolKind.HT, ht_all_reduce),
            ),
        ),
    )
    return MNNVLCuteDSLConfig(profiles=(profile,))


class CuteDSLFusionWorkspace:
    """One compiled FlashInfer workspace plus the shape contract it serves."""

    def __init__(
        self,
        *,
        workspace,
        max_token_num: int,
        hidden_size: int,
        top_k: int,
        rms_eps: float,
        supports_finalize: bool,
    ) -> None:
        self.workspace = workspace
        self.max_token_num = max_token_num
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.rms_eps = rms_eps
        self.supports_finalize = supports_finalize

    def serves(self, num_tokens: int, hidden_size: int, eps: float) -> bool:
        return (
            0 < num_tokens <= self.max_token_num
            and hidden_size == self.hidden_size
            and eps == self.rms_eps
        )

    def destroy(self) -> None:
        if self.workspace is not None:
            self.workspace.destroy()
            self.workspace = None


def create_workspace(
    *,
    tp_size: int,
    tp_rank: int,
    device_group: ProcessGroup,
    max_token_num: int,
    hidden_size: int,
    top_k: int,
    rms_eps: float,
    include_shared_expert: bool,
    dtype: torch.dtype,
) -> Optional[CuteDSLFusionWorkspace]:
    """Compile the LL/BT/HT kernels for this model. Returns None when the shape
    is out of contract, leaving the caller on its previous backend."""
    if not is_available():
        return None
    if dtype != torch.bfloat16:
        logger.warning(
            "MNNVL CuTe DSL allreduce fusion supports bfloat16 only (got %s).", dtype
        )
        return None
    if tp_size not in (2, 4, 8, 16):
        logger.warning(
            "MNNVL CuTe DSL allreduce fusion supports tp_size 2/4/8/16 (got %d).",
            tp_size,
        )
        return None

    config = _build_config(tp_size, hidden_size, top_k, dtype)
    if config is None:
        return None

    from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

    try:
        workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=tp_size,
            tp_rank=tp_rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=dtype,
            group=device_group,
            top_k=top_k,
            rms_eps=rms_eps,
            # The MoE runner folds routed_scaling_factor into the top-k weights
            # it hands back with the deferred finalize, so the kernel must not
            # apply it a second time.
            routed_scaling_factor=1.0,
            weight_bias=0.0,
            include_shared_expert=include_shared_expert,
            add_residual=True,
            write_residual_output=True,
            config=config,
        )
    except Exception as e:
        logger.warning(
            "MNNVL CuTe DSL workspace creation failed (%s); falling back to the "
            "mnnvl backend.",
            e,
        )
        return None

    logger.info(
        "MNNVL CuTe DSL allreduce fusion workspace ready: tp=%d hidden=%d "
        "top_k=%d capacity=%d shared_expert=%s",
        tp_size,
        hidden_size,
        top_k,
        max_token_num,
        include_shared_expert,
    )
    return CuteDSLFusionWorkspace(
        workspace=workspace,
        max_token_num=max_token_num,
        hidden_size=hidden_size,
        top_k=top_k,
        rms_eps=rms_eps,
        supports_finalize=True,
    )


# Set on the MoE layer's shared-expert output when its finalize was deferred for
# the fused collective; the next layer's input RMSNorm consumes it (see
# ``fuse_deferred_moe_finalize``). Living on the tensor mirrors the existing
# ``_sglang_needs_allreduce_fusion`` handoff between the same two points.
_DEFERRED_ATTR = "_sglang_cute_dsl_deferred_moe_finalize"


def can_defer_moe_finalize(num_tokens: int, hidden_size: int, top_k: int) -> bool:
    """Whether the MoE layer should hand its finalize to the fused collective.

    Checked in the MoE forward, before the deferred operands exist, so it only
    consults the workspace contract — the consumer re-checks and materializes
    the finalize itself if anything has changed.
    """
    from sglang.srt.layers.flashinfer_comm_fusion import get_cute_dsl_workspace

    handle = get_cute_dsl_workspace(use_attn_tp_group=False)
    return (
        handle is not None
        and handle.supports_finalize
        and handle.top_k == top_k
        and 0 < num_tokens <= handle.max_token_num
        and hidden_size == handle.hidden_size
    )


def attach_deferred_moe_finalize(shared_output: torch.Tensor, deferred) -> torch.Tensor:
    """Carry ``deferred`` on the shared-expert output to the next layer."""
    setattr(shared_output, _DEFERRED_ATTR, deferred)
    return shared_output


_logged_finalize_operands = False


def _finalize_operands(deferred, num_tokens: int):
    """``(expert_weights, expanded_idx_to_permuted_idx)`` in the layout the
    CuTe DSL kernel wants, or None when the deferred output cannot supply it.

    The TRT-LLM runner hands back a flat ``[num_tokens * top_k]`` routing map
    and top-k weights that may be fp32, while the kernel reads both as
    ``[num_tokens, top_k]`` with bf16 weights.
    """
    global _logged_finalize_operands

    top_k = deferred.top_k
    expert_weights = deferred.expert_weights
    expanded_idx = deferred.expanded_idx_to_permuted_idx
    if not _logged_finalize_operands:
        _logged_finalize_operands = True
        logger.info(
            "CuTe DSL fused MoE finalize operands: gemm2_out=%s expert_weights=%s/%s "
            "expanded_idx=%s/%s top_k=%d num_tokens=%d",
            tuple(deferred.gemm2_out.shape),
            tuple(expert_weights.shape),
            expert_weights.dtype,
            tuple(expanded_idx.shape),
            expanded_idx.dtype,
            top_k,
            num_tokens,
        )

    needed = num_tokens * top_k
    if expanded_idx.numel() < needed or not expanded_idx.is_contiguous():
        return None
    expanded_idx = expanded_idx.reshape(-1)[:needed].view(num_tokens, top_k)

    if expert_weights.numel() < needed or not expert_weights.is_contiguous():
        return None
    expert_weights = expert_weights.reshape(-1)[:needed].view(num_tokens, top_k)
    if expert_weights.dtype != torch.bfloat16:
        expert_weights = expert_weights.to(torch.bfloat16)
    return expert_weights, expanded_idx


def fuse_deferred_moe_finalize(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Consume a deferred MoE finalize riding on ``hidden_states``.

    Returns ``(norm_out, residual_out)`` when the finalize was folded into the
    fused collective, or ``None`` when there was nothing deferred. When a
    payload is present but the workspace cannot serve it, the finalize is
    materialized into ``hidden_states`` in place of the fusion and ``None`` is
    returned, so the caller continues down its ordinary all-reduce path.
    """
    deferred = getattr(hidden_states, _DEFERRED_ATTR, None)
    if deferred is None:
        return None
    delattr(hidden_states, _DEFERRED_ATTR)

    from sglang.srt.layers.flashinfer_comm_fusion import get_cute_dsl_workspace

    handle = get_cute_dsl_workspace(use_attn_tp_group=False)
    num_tokens = hidden_states.shape[0]
    if (
        handle is not None
        and handle.supports_finalize
        and handle.top_k == deferred.top_k
        and handle.serves(num_tokens, hidden_states.shape[-1], eps)
        and residual.is_contiguous()
        and hidden_states.is_contiguous()
    ):
        operands = _finalize_operands(deferred, num_tokens)
        if operands is not None:
            expert_weights, expanded_idx_to_permuted_idx = operands
            return moe_finalize_allreduce_residual_rmsnorm(
                handle,
                deferred.gemm2_out,
                expert_weights,
                expanded_idx_to_permuted_idx,
                hidden_states,
                residual,
                weight,
                eps,
            )

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        finalize_flashinfer_trtllm_deferred_output,
    )

    logger.debug("CuTe DSL cannot serve the deferred MoE finalize; materializing it")
    hidden_states.copy_(
        finalize_flashinfer_trtllm_deferred_output(deferred, hidden_states)
    )
    return None


def allreduce_residual_rmsnorm(
    handle: CuteDSLFusionWorkspace,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AllReduce + residual add + RMSNorm; returns ``(norm, residual)``."""
    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    allreduce_fusion(
        input=input_tensor,
        workspace=handle.workspace,
        pattern=AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=weight,
        rms_eps=eps,
    )
    return norm_out, residual_out


def moe_finalize_allreduce_residual_rmsnorm(
    handle: CuteDSLFusionWorkspace,
    routed_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE finalize + shared-expert add folded into the fused AllReduce.

    ``routed_output`` is the permuted GEMM2 output the MoE runner produced with
    its finalize deferred; the kernel gathers each token's top-k rows through
    ``expanded_idx_to_permuted_idx`` (``-1`` marks a dropped slot), scales them
    by ``expert_weights``, and adds ``shared_output`` before the collective.
    """
    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    allreduce_fusion(
        input=routed_output,
        workspace=handle.workspace,
        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        launch_with_pdl=True,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=weight,
        rms_eps=eps,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_scale_factor=expert_weights,
        shared_expert_output=shared_output,
    )
    return norm_out, residual_out
