# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark worker's low-latency verify paths: the block-wise rejection sampler, the
static verify epilogue with the Engram and side-stream context-KV commits, and the
sampling-accept graph with its startup precompile."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import torch

from sglang.srt.models.dsv41_low_latency.dspark.blockwise_rejection import (
    RejectionNoise,
    make_rejection_noise,
)
from sglang.srt.models.dsv41_low_latency.dspark.sampling_accept_graph import (
    DsparkSamplingAcceptGraph,
    precompile_sampling_accept,
)
from sglang.srt.models.dsv41_low_latency.dspark.static_epilogue import (
    ContextKvSideStream,
    StaticEpilogue,
)
from sglang.srt.models.dsv41_low_latency.switch import ROUTED_EXPERT_MAX_ROWS
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

logger = logging.getLogger(__name__)


class DsparkLowLatency:
    """Owned by the DSpark worker (DSparkWorkerV2.low_latency)."""

    def __init__(
        self,
        *,
        static_epilogue: Optional[StaticEpilogue],
        sampling_accept_graph: Optional[DsparkSamplingAcceptGraph],
        resolve_draft_sampler: Callable[[], Optional[object]],
        rejection_noise: RejectionNoise,
        capture_bs: list[int],
        stride: int,
        block_verification: bool,
    ) -> None:
        self._static = static_epilogue
        self._sampling_accept_graph = sampling_accept_graph
        self._resolve_draft_sampler = resolve_draft_sampler
        self._rejection_noise = rejection_noise
        self._capture_bs = sorted(capture_bs)
        self._stride = stride
        self._block_verification = block_verification

    def sampling_accept(
        self,
        *,
        bs: int,
        target_logits: torch.Tensor,
        proposal_folded: bool,
        planner_mode: str,
        sampling_info,
        simulate_bonus_sampling_info,
        has_grammar: bool,
    ) -> Optional[AcceptOuts]:
        """The accept of an all-sampled static step through the sampling-accept graph,
        or None for the eager accept. Every gate reads host state all TP ranks share."""
        if not (
            self._sampling_accept_graph is not None
            and proposal_folded
            and planner_mode == "static"
            and sampling_info is not None
            and not sampling_info.is_any_greedy
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and verify_logits_adjustments_are_noop(sampling_info)
            and simulate_bonus_sampling_info is None
            and not has_grammar
        ):
            return None
        return self._sampling_accept_graph.run(bs=bs, target_logits=target_logits)

    def committed_engram(
        self, *, folded_accept: bool, run_compact: bool, graph_accept: bool
    ) -> bool:
        """Whether this step's Engram history commit ran inside a graph: the static
        verify graph's folded accept, or the sampling-accept region (always commits)."""
        return (
            self._static is not None
            and self._static.folds_engram_commit
            and ((folded_accept and not run_compact) or graph_accept)
        )

    def precompile(self) -> None:
        """Compile the sampling and mixed accept kernels for every decode batch size,
        so the first temperature request does not stall on Triton compiles. Shapes come
        from the captured verify and draft graphs, so this runs after both captures."""
        static = self._static
        sampler = self._resolve_draft_sampler()
        if (
            static is None
            or not static.static_io
            or sampler is None
            or (not sampler.greedy_draft and sampler.step_logits is None)
        ):
            logger.warning(
                "The DSpark sampling-accept precompile needs the static verify and "
                "folded draft graphs; skipped."
            )
            return
        start = time.perf_counter()
        precompile_sampling_accept(
            capture_bs=self._capture_bs,
            stride=self._stride,
            target_logits=next(iter(static.static_io.values())).logits,
            draft_sampler=sampler,
            fused_argmax=static.fused_argmax,
            block_verification=self._block_verification,
            rejection_noise=self._rejection_noise,
        )
        logger.info(
            "Precompiled the DSpark sampling accept for batch sizes %s in %.1f s.",
            self._capture_bs,
            time.perf_counter() - start,
        )


def _check_draft_graph_rows(*, capture_bs: list[int], rows_per_request: int) -> bool:
    """Warn when the smallest draft CUDA graph has more rows than the draft's
    routed-expert GEMMs take, so the draft MoE never runs on them."""
    smallest = min(capture_bs) * rows_per_request if capture_bs else None
    if smallest is None or smallest <= ROUTED_EXPERT_MAX_ROWS:
        return True
    logger.warning(
        f"The smallest DSpark draft CUDA graph has {smallest} rows ({rows_per_request} "
        f"per request), above the {ROUTED_EXPERT_MAX_ROWS} the draft routed-expert "
        f"GEMMs take; add smaller decode CUDA graph batch sizes."
    )
    return False


def install(
    *,
    executor: TargetVerifyExecutor,
    epilogue: Optional[DsparkVerifyEpilogue],
    draft_is_moe: bool,
    resolve_engram_hasher: Callable[[], Optional[object]],
    resolve_draft_sampler: Callable[[], Optional[object]],
    tp_sync: SpecTpSync,
    fused_argmax: bool,
    random_seed: int,
    device,
    capture_bs: list[int],
    rows_per_request: int,
    stride: int,
    block_verification: bool,
) -> DsparkLowLatency:
    """Called at the end of the DSpark worker's construction, on every rank."""
    # Every rank holds the same seed, so the counter-based draws agree.
    rejection_noise = make_rejection_noise(random_seed, device)
    executor.rejection_noise = rejection_noise
    _check_draft_graph_rows(capture_bs=capture_bs, rows_per_request=rows_per_request)
    static_epilogue = sampling_accept_graph = None
    if epilogue is not None:

        def resolve_engram_history() -> Optional[torch.Tensor]:
            hasher = resolve_engram_hasher()
            return None if hasher is None else hasher.history

        static_epilogue = StaticEpilogue(
            epilogue=epilogue,
            tp_sync=tp_sync,
            fused_argmax=fused_argmax,
            resolve_engram_history=resolve_engram_history,
            context_kv_side_stream=ContextKvSideStream() if draft_is_moe else None,
        )
        epilogue.static_epilogue_override = static_epilogue
        sampling_accept_graph = DsparkSamplingAcceptGraph(
            epilogue=epilogue,
            static_epilogue=static_epilogue,
            resolve_draft_sampler=resolve_draft_sampler,
            tp_sync=tp_sync,
            fused_argmax=fused_argmax,
            block_verification=block_verification,
            device=device,
            rejection_noise=rejection_noise,
        )
    return DsparkLowLatency(
        static_epilogue=static_epilogue,
        sampling_accept_graph=sampling_accept_graph,
        resolve_draft_sampler=resolve_draft_sampler,
        rejection_noise=rejection_noise,
        capture_bs=capture_bs,
        stride=stride,
        block_verification=block_verification,
    )
