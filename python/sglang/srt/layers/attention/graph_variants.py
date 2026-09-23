from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: ClassVar[tuple[str, ...]]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
    capture_labels: ClassVar[tuple[str, ...]] = (DSA_DENSE, DSA_SPARSE)

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Plain decode maintains this host mirror without a D2H sync.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return DSA_SPARSE
        return DSA_DENSE if max_kv_len <= self.index_topk else DSA_SPARSE


def create_attention_graph_variants(hf_config) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        index_topk = get_dsa_index_topk(hf_config)
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            index_topk,
        )
        return DsaGraphVariants(index_topk)
    return None


DSV41_CANDIDATE_FILTERED = "candidate_filtered"


@dataclass(frozen=True)
class Dsv41CandidateGraphVariants:
    """Candidate-indexer graphs keyed by the batch's longest request; a variant
    below its limit skips low-ratio scoring or candidate filtering."""

    # (label, max_seq_len it serves), ascending; the last label is the fallback.
    graph_limits: tuple[tuple[str, int], ...]
    capture_labels: tuple[str, ...]
    verify_extra_tokens: int = 0

    def select(self, forward_batch: ForwardBatch) -> str:
        lengths = getattr(forward_batch, "seq_lens_cpu", None)
        max_seq_len = None
        if lengths is not None and lengths.device.type == "cpu" and lengths.numel() > 0:
            max_seq_len = int(lengths.max())
        if max_seq_len is None and self.verify_extra_tokens:
            # Includes acceptance still in flight, without a GPU-to-CPU copy.
            max_seq_len = getattr(
                getattr(forward_batch, "spec_info", None),
                "candidate_max_seq_len_upper_bound",
                None,
            )
        if max_seq_len is not None:
            max_seq_len += self.verify_extra_tokens
            for variant, limit in self.graph_limits:
                if max_seq_len <= limit:
                    return variant
        return DSV41_CANDIDATE_FILTERED


def create_dsv41_candidate_graph_variants(
    model_runner, capture_forward_mode, captured_req_width: int = 0
) -> Optional[Dsv41CandidateGraphVariants]:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.utils import is_hip

    text_config = model_runner.model_config.hf_text_config
    dspark_target_verify = (
        capture_forward_mode == ForwardMode.TARGET_VERIFY
        and model_runner.spec_algorithm.is_dspark()
        and not model_runner.is_draft_worker
        and captured_req_width > 0
    )
    if not (
        (capture_forward_mode == ForwardMode.DECODE or dspark_target_verify)
        and model_runner.device == "cuda"
        and (is_hip() or torch.cuda.get_device_capability(model_runner.gpu_id)[0] >= 10)
        and getattr(text_config, "model_type", None) == "deepseek_v41"
        and getattr(text_config, "candidate_source_layer_id", -1) >= 0
    ):
        return None
    span = text_config.candidate_topk_blocks * text_config.candidate_block_size
    if span <= 0:
        return None
    ratios = set(text_config.compress_ratios) & {1, 2}
    topk = text_config.index_topk
    variants = []
    # Verify needs per-query causal top-k, so it always keeps candidate filtering.
    if topk > 0 and ratios and not dspark_target_verify:
        variants.append(("candidate_all", topk * min(ratios)))
        if ratios == {1, 2}:
            variants.append(("candidate_c2_all", topk * 2))
    variants.append(("candidate_unfiltered", span))
    graph_limits = []
    for variant, limit in variants:
        graph_limits.append((variant, min(limit, span)))
        if limit >= span:
            break
    logger.info(
        "Candidate indexer graph limits: %s; use full filtering above %s.",
        graph_limits,
        span,
    )
    return Dsv41CandidateGraphVariants(
        graph_limits=tuple(graph_limits),
        capture_labels=tuple(v for v, _ in graph_limits) + (DSV41_CANDIDATE_FILTERED,),
        # The verify backend adds this width to committed CPU lengths.
        verify_extra_tokens=captured_req_width if dspark_target_verify else 0,
    )
