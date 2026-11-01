from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"
# Sparse, with the gfx950 indexer logits sharing KV pages across a request's
# MTP verify rows (paged_mqa_logits.cuh, logits_hist_slot).
DSA_SPARSE_SHARED_PAGES = "sparse_shared_pages"

# MTP verify batches of at most this many rows whose longest request holds at
# least this many tokens are faster with the shared-page logits kernel (MI355X).
DSA_SHARED_PAGES_MAX_ROWS = 18
_SHARED_PAGES_MIN_LEN = 98304


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: ClassVar[tuple[str, ...]]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


def dsa_max_kv_len(forward_batch: ForwardBatch) -> Optional[int]:
    """The batch's longest sequence, or None when the batch carries no lengths."""
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
        # Plain decode maintains this host mirror without a D2H sync.
        return int(seq_lens_cpu.max().item())
    if forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
        # Fallback: a single scalar reduction d2h (cheap, per-step).
        return int(forward_batch.seq_lens.max().item())
    return None


def dsa_share_logits_pages(rows: int, max_kv_len: int) -> bool:
    """Whether a verify batch takes the shared-page indexer logits kernel."""
    return rows <= DSA_SHARED_PAGES_MAX_ROWS and max_kv_len >= _SHARED_PAGES_MIN_LEN


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Draft tokens per request when these are MTP verify graphs on gfx950, so
    # the shared-page logits variant is captured too; 0 otherwise.
    verify_rows_per_req: int = 0

    @property
    def capture_labels(self) -> tuple[str, ...]:
        # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
        if self.verify_rows_per_req:
            return (DSA_DENSE, DSA_SPARSE, DSA_SPARSE_SHARED_PAGES)
        return (DSA_DENSE, DSA_SPARSE)

    def select(self, forward_batch: ForwardBatch) -> str:
        max_kv_len = dsa_max_kv_len(forward_batch)
        if max_kv_len is None:
            # No length info: be safe and use the correct-for-all sparse graph.
            return DSA_SPARSE
        if max_kv_len <= self.index_topk:
            return DSA_DENSE
        if self.verify_rows_per_req and dsa_share_logits_pages(
            forward_batch.batch_size * self.verify_rows_per_req, max_kv_len
        ):
            return DSA_SPARSE_SHARED_PAGES
        return DSA_SPARSE


def create_attention_graph_variants(
    hf_config, verify_rows_per_req: int = 0
) -> Optional[AttentionGraphVariants]:
    """`verify_rows_per_req`: draft tokens per request for MTP verify graphs."""
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        get_dsa_index_topk,
        is_deepseek_dsa,
    )
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        # KPool has no dense-skip path: both variants run the full indexer.
        # Capture it once, keeping the regular DSA dense/sparse split intact.
        if get_dsa_index_kpool(hf_config) > 1:
            return None

        index_topk = get_dsa_index_topk(hf_config)
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            index_topk,
        )
        from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import (
            supported_hardware,
        )

        return DsaGraphVariants(
            index_topk,
            verify_rows_per_req if supported_hardware() else 0,
        )
    return None


DSV41_CANDIDATE_FILTERED = "candidate_filtered"
DSV41_CANDIDATE_UNFILTERED = "candidate_unfiltered"
# SGLANG_OPT_DSV41_TARGET_METADATA_EARLY serves these target verify variants (requests
# up to the candidate span); longer ones keep the in-graph metadata upgrade.
DSV41_EARLY_VERIFY_VARIANTS = frozenset({DSV41_CANDIDATE_UNFILTERED})


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
        return self.select_for_max_seq_len(max_seq_len)

    def select_for_max_seq_len(self, max_seq_len: Optional[int]) -> str:
        """The variant for a batch whose longest request (before this step's verify
        tokens) is ``max_seq_len``; None, when unknown, takes the fallback."""
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
    from sglang.srt.utils import is_gfx95_supported, is_hip

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
        and (
            is_gfx95_supported()
            if is_hip()
            else torch.cuda.get_device_capability(model_runner.gpu_id)[0] >= 10
        )
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
    variants.append((DSV41_CANDIDATE_UNFILTERED, span))
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
