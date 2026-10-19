from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

import msgspec

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"

SPLIT_KV_STOCK = "split_kv_stock"
SPLIT_KV_LONG = "split_kv_long"
SPLIT_KV_MIN_LENGTH = 32768


PAGED_FP8_STOCK = "paged_fp8_stock"
PAGED_FP8_LONG = "paged_fp8_long"
PAGED_FP8_MIN_LENGTH = 2049


@dataclass(frozen=True)
class PagedFP8GraphVariants:
    """Use a host-only lower bound to select the eligible attention graph."""

    capture_labels: ClassVar[tuple[str, ...]] = (PAGED_FP8_STOCK, PAGED_FP8_LONG)

    def select(self, forward_batch: ForwardBatch) -> str:
        if forward_batch.batch_size != 1:
            return PAGED_FP8_STOCK
        length = forward_batch.attention_kv_len_lower_bound
        if length is None:
            lengths = forward_batch.seq_lens_cpu
            if lengths is not None and lengths.device.type == "cpu" and lengths.numel():
                length = int(lengths.min())
        return (
            PAGED_FP8_LONG
            if length is not None and length >= PAGED_FP8_MIN_LENGTH
            else PAGED_FP8_STOCK
        )


def create_paged_fp8_graph_variants(
    capture_forward_mode, captured_req_width, *, enabled=True
):
    from sglang.srt.environ import envs

    if enabled and envs.SGLANG_Q35MF_LL1_K1B.get() and captured_req_width in (1, 7):
        if (
            capture_forward_mode.is_decode_or_idle()
            or capture_forward_mode.is_target_verify()
            or capture_forward_mode.is_draft_extend_v2()
        ):
            return PagedFP8GraphVariants()
    return None


def use_paged_fp8_attention(forward_batch: ForwardBatch) -> bool:
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
        get_is_capture_mode,
    )

    variant = get_capture_attention_variant()
    if variant is not None:
        return variant == PAGED_FP8_LONG
    if get_is_capture_mode():
        return False
    return PagedFP8GraphVariants().select(forward_batch) == PAGED_FP8_LONG


def paged_fp8_engagement():
    """Host-only operator/captured-graph replay counters for engagement checks."""
    from sglang.srt.runtime_context import get_buffer

    return get_buffer("paged_fp8_engagement", lambda: {"calls": {}, "replays": {}})


def record_paged_fp8_call(qlen):
    counts = paged_fp8_engagement()["calls"]
    key = f"q{qlen}"
    counts[key] = counts.get(key, 0) + 1


Q1_STOCK = "q1_stock"
Q1_LONG = "q1_long"


class Q1GraphVariants(msgspec.Struct, frozen=True):
    """Draft q1 decode: the paged FP8 q1 kernel from a host-only KV length lower bound, trtllm-gen below it."""

    min_length: int
    capture_labels: ClassVar[tuple[str, ...]] = (Q1_STOCK, Q1_LONG)

    def select(self, forward_batch: ForwardBatch) -> str:
        if forward_batch.batch_size != 1:
            return Q1_STOCK
        length = forward_batch.attention_kv_len_lower_bound
        if length is None:
            lengths = forward_batch.seq_lens_cpu
            if (
                lengths is not None
                and lengths.device.type == "cpu"
                and lengths.numel() == 1
            ):
                length = int(lengths[0])
        return Q1_LONG if length is not None and length >= self.min_length else Q1_STOCK


def create_q1_graph_variants() -> Optional[Q1GraphVariants]:
    from sglang.srt.environ import envs

    if envs.SGLANG_Q35MF_MKA_Q1.get():
        return Q1GraphVariants(min_length=envs.SGLANG_Q35MF_MKA_Q1_MIN_LENGTH.get())
    return None


def use_q1_attention(forward_batch: ForwardBatch) -> bool:
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
        get_is_capture_mode,
    )

    variant = get_capture_attention_variant()
    if variant is not None:
        return variant == Q1_LONG
    if get_is_capture_mode():
        return False
    variants = Q1GraphVariants(min_length=envs.SGLANG_Q35MF_MKA_Q1_MIN_LENGTH.get())
    return variants.select(forward_batch) == Q1_LONG


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: ClassVar[tuple[str, ...]]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


@dataclass(frozen=True)
class SplitKVGraphVariants:
    """Keep short sequences on stock attention without a device-to-host copy."""

    min_length: int = SPLIT_KV_MIN_LENGTH
    capture_labels: ClassVar[tuple[str, ...]] = (SPLIT_KV_STOCK, SPLIT_KV_LONG)

    def select(self, forward_batch: ForwardBatch) -> str:
        if forward_batch.batch_size != 1:
            return SPLIT_KV_STOCK
        length = forward_batch.attention_kv_len_lower_bound
        if length is None:
            lengths = forward_batch.seq_lens_cpu
            if (
                lengths is not None
                and lengths.device.type == "cpu"
                and lengths.numel() == 1
            ):
                length = int(lengths[0])
        return (
            SPLIT_KV_LONG
            if length is not None and length >= self.min_length
            else SPLIT_KV_STOCK
        )


def _split_kv_min_length(is_target_verify: bool) -> int:
    from sglang.srt.environ import envs

    # The split-KV producer may also take target verify below 32K; the draft extend keeps the stock threshold.
    if is_target_verify and envs.SGLANG_Q35MF_MKA_SKV2.get():
        return envs.SGLANG_Q35MF_MKA_SKV2_VERIFY_MIN_LENGTH.get()
    return SPLIT_KV_MIN_LENGTH


def create_split_kv_graph_variants(
    capture_forward_mode, captured_req_width: int
) -> Optional[SplitKVGraphVariants]:
    from sglang.srt.environ import envs

    if (
        envs.SGLANG_Q35MF_ATTN_SPLIT_KV.get()
        and captured_req_width == 7
        and (
            capture_forward_mode.is_target_verify()
            or capture_forward_mode.is_draft_extend_v2()
        )
    ):
        return SplitKVGraphVariants(
            min_length=_split_kv_min_length(capture_forward_mode.is_target_verify())
        )
    return None


def use_split_kv_attention(forward_batch: ForwardBatch) -> bool:
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_attention_variant,
        get_is_capture_mode,
    )

    variant = get_capture_attention_variant()
    if variant is not None:
        if variant in (PAGED_FP8_STOCK, PAGED_FP8_LONG):
            return False
        return variant == SPLIT_KV_LONG
    if get_is_capture_mode():
        return False
    variants = SplitKVGraphVariants(
        min_length=_split_kv_min_length(forward_batch.forward_mode.is_target_verify())
    )
    return variants.select(forward_batch) == SPLIT_KV_LONG


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
