"""Compact target filtering for Ascend DFlash2 rejection sampling."""

import json
import logging
from collections.abc import Callable

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import _get_or_create_chain_verify_buffers

logger = logging.getLogger(__name__)


def filter_probs_with_row_fallback(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    reference_filter: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    *,
    fallback_mask: list[bool] | None = None,
) -> torch.Tensor:
    """Filter small top-k rows together and repair ambiguous rows with the reference."""
    limit = min(64, probs.shape[-1] - 1)
    max_top_k = max((k for k in top_ks.tolist() if 2 <= k <= limit), default=0)
    if max_top_k == 0:
        if fallback_mask is not None:
            fallback_mask.extend([True] * probs.shape[0])
        return reference_filter(probs, top_ks, top_ps)

    # One extra value detects ties across each row's top-k boundary.
    values, indices = probs.topk(max_top_k + 1, dim=-1, sorted=True)
    row_ks = top_ks[:, None].long()
    in_top_k = torch.arange(max_top_k, device=probs.device)[None, :] < row_ks
    head = values[:, :-1].masked_fill(~in_top_k, 0)
    weights = head / head.sum(dim=-1, keepdim=True)
    cumulative = weights.cumsum(dim=-1)
    thresholds = top_ps[:, None]
    keep = torch.cat(
        (torch.ones_like(in_top_k[:, :1]), cumulative[:, :-1] < thresholds), dim=-1
    )
    keep = (keep | (thresholds == 1)) & in_top_k

    boundary = row_ks.clamp(1, max_top_k)
    top_k_tie = values.gather(1, boundary - 1) == values.gather(1, boundary)
    top_p_tie = (
        (head[:, :-1] == head[:, 1:]) & (keep[:, :-1] != keep[:, 1:]) & in_top_k[:, 1:]
    ).any(dim=-1, keepdim=True)
    # Rounding in the compact reduction can change support near the threshold.
    near_top_p = (
        ((cumulative - thresholds).abs() <= 1e-5) & in_top_k & (thresholds < 1)
    ).any(dim=-1, keepdim=True)
    fallback = (
        (row_ks < 2)
        | (row_ks > max_top_k)
        | ~((thresholds > 0) & (thresholds <= 1))
        | top_k_tie
        | top_p_tie
        | near_top_p
        | ~torch.isfinite(weights).all(dim=-1, keepdim=True)
    )
    needs_fallback = fallback.flatten().tolist()
    if fallback_mask is not None:
        fallback_mask.extend(needs_fallback)
    fallback_rows = [row for row, needed in enumerate(needs_fallback) if needed]
    if len(fallback_rows) == probs.shape[0]:
        return reference_filter(probs, top_ks, top_ps)

    selected = head.masked_fill(~keep, 0)
    selected = selected / selected.sum(dim=-1, keepdim=True)
    result = torch.zeros_like(probs).scatter_(1, indices[:, :-1], selected)
    if fallback_rows:
        rows = torch.tensor(fallback_rows, dtype=torch.long, device=probs.device)
        repaired = reference_filter(
            probs.index_select(0, rows),
            top_ks.index_select(0, rows),
            top_ps.index_select(0, rows),
        )
        result.index_copy_(0, rows, repaired)
    return result


class NpuDFlashSelectorSampler:
    """Ascend target filtering and chain rejection sampling for one worker."""

    def __init__(self):
        from sgl_kernel_npu import sample

        self._top_k_renorm = sample.top_k_renorm_prob
        self._top_p_renorm = sample.top_p_renorm_prob
        self._chain_sampling = sample.chain_speculative_sampling_triton
        self._row_fallback = envs.SGLANG_OPT_DFLASH_NPU_ROW_FALLBACK.get()
        self._shadow = envs.SGLANG_DEBUG_DFLASH_NPU_ROW_FALLBACK_SHADOW.get()
        self._log_batches = envs.SGLANG_LOG_DFLASH_NPU_SAMPLING_BATCHES.get()
        self._seen_batches: set[int] = set()
        self._seen_compact_batches: set[int] = set()

    def log_batch(self, bs: int, rank: int) -> None:
        if (self._log_batches or self._shadow) and bs not in self._seen_batches:
            logger.info("DFLASH_DATASET_BATCH bs=%d rank=%d", bs, rank)
            self._seen_batches.add(bs)

    def _reference_filter(self, probs, top_ks, top_ps):
        probs = self._top_k_renorm(probs, top_ks.clamp(max=probs.shape[-1]))
        return self._top_p_renorm(probs, top_ps)

    def _target_probs(self, logits, sampling_info, bs, width):
        temperatures = sampling_info.temperatures.repeat_interleave(width, dim=0)
        probs = torch.softmax(logits.float() / temperatures, dim=-1)
        need_top_k = sampling_info.need_top_k_sampling
        need_top_p = sampling_info.need_top_p_sampling
        top_ks = sampling_info.top_ks.repeat_interleave(width) if need_top_k else None
        top_ps = sampling_info.top_ps.repeat_interleave(width) if need_top_p else None
        if need_top_k and need_top_p:
            if self._shadow or self._row_fallback:
                mask = [] if self._shadow or self._log_batches else None
                filtered = filter_probs_with_row_fallback(
                    probs.clone() if self._shadow else probs,
                    top_ks,
                    top_ps,
                    self._reference_filter,
                    fallback_mask=mask,
                )
                if self._shadow:
                    probs = self._reference_filter(probs, top_ks, top_ps)
                    if not torch.equal(filtered > 0, probs > 0):
                        raise ValueError(
                            "DFlash2 row fallback changed the target token support"
                        )
                    torch.testing.assert_close(filtered, probs, rtol=1e-5, atol=1e-7)
                    torch.testing.assert_close(
                        filtered.sum(-1),
                        torch.ones_like(filtered[:, 0]),
                        rtol=0,
                        atol=2e-6,
                    )
                    if ((filtered - probs).abs().sum(-1) > 2e-5).any().item():
                        raise ValueError(
                            "DFlash2 row fallback exceeded the target probability error limit"
                        )
                    logger.info("DFLASH_SPLIT_VERIFIED %s", json.dumps(mask))
                else:
                    probs = filtered
                    if (
                        mask is not None
                        and not all(mask)
                        and bs not in self._seen_compact_batches
                    ):
                        logger.info("DFLASH_COMPACT_BATCH bs=%d", bs)
                        self._seen_compact_batches.add(bs)
            else:
                probs = self._reference_filter(probs, top_ks, top_ps)
        elif need_top_k:
            probs = self._top_k_renorm(probs, top_ks.clamp(max=probs.shape[-1]))
        elif need_top_p:
            probs = self._top_p_renorm(probs, top_ps)

        if sampling_info.need_min_p_sampling:
            min_ps = sampling_info.min_ps.repeat_interleave(width)[:, None]
            probs.masked_fill_(probs < probs.amax(dim=-1, keepdim=True) * min_ps, 0)
            probs.div_(probs.sum(dim=-1, keepdim=True))
        if sampling_info.is_any_greedy:
            greedy = top_ks[:, None] <= 1
            greedy_tokens = logits.argmax(dim=-1, keepdim=True)
            greedy_mass = probs.gather(1, greedy_tokens)
            probs.masked_fill_(greedy, 0)
            probs.scatter_(1, greedy_tokens, torch.where(greedy, 1.0, greedy_mass))
        return probs.view(bs, width, -1).contiguous()

    def accept(self, candidates, target_logits, draft_probs, sampling_info):
        bs, width = candidates.shape
        device = candidates.device
        target_probs = self._target_probs(target_logits, sampling_info, bs, width)
        (
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            predicts,
            accept_indices,
            num_correct_drafts,
        ) = _get_or_create_chain_verify_buffers(
            bs=bs, draft_token_num=width, device=device
        )
        # The Ascend kernel strides its coins by the full candidate width.
        self._chain_sampling(
            predicts=predicts,
            accept_index=accept_indices,
            accept_token_num=num_correct_drafts,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=torch.rand((bs, width), dtype=torch.float32, device=device),
            uniform_samples_for_final_sampling=torch.rand(
                (bs,), dtype=torch.float32, device=device
            ),
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        bonus_positions = accept_indices.gather(1, num_correct_drafts.long()[:, None])
        bonus_tokens = predicts.gather(0, bonus_positions.flatten().long()).long()
        return num_correct_drafts, bonus_tokens
