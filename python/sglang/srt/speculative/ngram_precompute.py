import logging
from collections import deque
from collections.abc import Sequence
from typing import Any, Optional

import msgspec
import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_local_accept_path_nodes(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    """Convert sampler-global accept indices to one local node per request."""
    batch_size = accept_lens.shape[0]
    device = accept_index.device
    accept_index_2d = accept_index.reshape(batch_size, draft_token_num).to(torch.long)
    slot_indices = torch.arange(draft_token_num, device=device)
    valid_slots = (slot_indices[None, :] < accept_lens[:, None]) & (
        accept_index_2d >= 0
    )
    path_slots = torch.where(
        valid_slots,
        slot_indices[None, :],
        torch.full_like(accept_index_2d, -1),
    ).amax(dim=1)
    safe_path_slots = path_slots.clamp(min=0, max=draft_token_num - 1)
    row_indices = torch.arange(batch_size, device=device)
    path_nodes = (
        accept_index_2d[row_indices, safe_path_slots] - row_indices * draft_token_num
    )
    valid_path_nodes = (
        (path_slots >= 0) & (path_nodes >= 0) & (path_nodes < draft_token_num)
    )
    return torch.where(valid_path_nodes, path_nodes, -1)


def select_precomputed_drafts(
    *,
    cache_rows: torch.Tensor,
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select dense precomputed trees without inspecting results on the CPU.

    A miss returns a legal shallow tree rooted at the verified bonus token.
    """
    batch_size = accept_lens.shape[0]
    draft_token_num = cached_draft_tokens.shape[-1]
    device = accept_tokens.device

    has_precomputed_row = cache_rows >= 0
    safe_cache_rows = cache_rows.clamp(min=0)
    accept_lens_long = accept_lens.to(torch.long)
    last_slots = (accept_lens_long - 1).clamp(min=0, max=draft_token_num - 1)
    row_indices = torch.arange(batch_size, device=device)
    accept_tokens_2d = accept_tokens.reshape(batch_size, draft_token_num)
    bonus_tokens = accept_tokens_2d[row_indices, last_slots].to(torch.int32)

    path_nodes = accept_path_nodes.to(torch.long)
    valid_path = (
        has_precomputed_row
        & (accept_lens_long > 0)
        & (accept_lens_long <= draft_token_num)
        & (path_nodes >= 0)
        & (path_nodes < draft_token_num)
    )
    safe_path_nodes = path_nodes.clamp(min=0, max=draft_token_num - 1)

    bonus_candidates = cached_bonus_tokens[safe_cache_rows, safe_path_nodes]
    slot_matches = (bonus_candidates == bonus_tokens[:, None]) & valid_path[:, None]
    cache_hits = slot_matches.any(dim=1)
    bonus_slots = slot_matches.to(torch.int32).argmax(dim=1).to(torch.long)

    cached_drafts = cached_draft_tokens[safe_cache_rows, safe_path_nodes, bonus_slots]
    cached_masks = cached_tree_masks[safe_cache_rows, safe_path_nodes, bonus_slots]

    fallback_drafts = torch.zeros_like(cached_drafts)
    fallback_drafts[:, 0] = bonus_tokens.to(fallback_drafts.dtype)
    fallback_masks = fallback_tree_mask.expand(batch_size, -1, -1)

    selected_drafts = torch.where(cache_hits[:, None], cached_drafts, fallback_drafts)
    selected_masks = torch.where(
        cache_hits[:, None, None], cached_masks, fallback_masks
    )
    return selected_drafts, selected_masks, cache_hits


def select_precomputed_drafts_for_rows(
    *,
    cache_rows: Sequence[int],
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage request-to-cache rows, then run device-side draft selection."""
    cache_rows_cpu = torch.tensor(
        cache_rows,
        dtype=torch.long,
        device="cpu",
        pin_memory=accept_tokens.is_cuda,
    )
    cache_rows_device = cache_rows_cpu.to(
        accept_tokens.device, non_blocking=accept_tokens.is_cuda
    )
    return select_precomputed_drafts(
        cache_rows=cache_rows_device,
        accept_tokens=accept_tokens,
        accept_lens=accept_lens,
        accept_path_nodes=accept_path_nodes,
        cached_bonus_tokens=cached_bonus_tokens,
        cached_draft_tokens=cached_draft_tokens,
        cached_tree_masks=cached_tree_masks,
        fallback_tree_mask=fallback_tree_mask,
    )


def apply_precomputed_drafts_for_rows(
    *,
    cache_rows: Sequence[int],
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_path_nodes: torch.Tensor,
    cached_bonus_tokens: torch.Tensor,
    cached_draft_tokens: torch.Tensor,
    cached_tree_masks: torch.Tensor,
    fallback_tree_mask: torch.Tensor,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> torch.Tensor:
    """Select and install drafts into worker-owned device buffers."""
    selected_drafts, selected_masks, cache_hits = select_precomputed_drafts_for_rows(
        cache_rows=cache_rows,
        accept_tokens=accept_tokens,
        accept_lens=accept_lens,
        accept_path_nodes=accept_path_nodes,
        cached_bonus_tokens=cached_bonus_tokens,
        cached_draft_tokens=cached_draft_tokens,
        cached_tree_masks=cached_tree_masks,
        fallback_tree_mask=fallback_tree_mask,
    )
    draft_tokens.copy_(selected_drafts.reshape(-1), non_blocking=True)
    tree_mask.copy_(selected_masks.reshape(-1), non_blocking=True)
    return cache_hits


class NgramPrecomputeInputs(msgspec.Struct, frozen=True):
    """Host inputs consumed by one two-phase precompute call."""

    accept_tokens: list[int]
    accept_lens: list[int]
    draft_tokens: np.ndarray
    tree_mask: np.ndarray

    def validate(self, *, batch_size: int, draft_token_num: int) -> None:
        if len(self.accept_lens) != batch_size:
            raise RuntimeError("accept_lens does not match the precompute batch.")
        expected_draft_tokens = batch_size * draft_token_num
        if len(self.accept_tokens) != expected_draft_tokens:
            raise RuntimeError("accept_tokens does not match the precompute batch.")
        if self.draft_tokens.size != expected_draft_tokens:
            raise RuntimeError("draft_tokens does not match the precompute batch.")
        if self.tree_mask.size != expected_draft_tokens * draft_token_num:
            raise RuntimeError("tree_mask does not match the precompute batch.")


class _PendingHostCopy(msgspec.Struct, frozen=True):
    ready: Any
    tensors: tuple[torch.Tensor, ...]


class _GpuPrecomputedCache(msgspec.Struct, frozen=True):
    req_id_to_row: dict[str, int]
    bonus_tokens: torch.Tensor
    draft_tokens: torch.Tensor
    tree_masks: torch.Tensor
    copy_done: Any


class _HitRateTracker:
    def __init__(self) -> None:
        self.recent_hits = deque(maxlen=1000)
        self.reset()

    def reset(self) -> None:
        self.interval_hits = 0
        self.interval_total = 0
        self.total_hits = 0
        self.total = 0
        self.recent_hits.clear()

    def record(self, events: Sequence[bool]) -> None:
        hits = sum(events)
        total = len(events)
        self.interval_hits += hits
        self.interval_total += total
        self.total_hits += hits
        self.total += total
        self.recent_hits.extend(events)

    def reset_interval(self) -> None:
        self.interval_hits = 0
        self.interval_total = 0

    @staticmethod
    def _rate(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    @property
    def interval_rate(self) -> float:
        return self._rate(self.interval_hits, self.interval_total)

    @property
    def total_rate(self) -> float:
        return self._rate(self.total_hits, self.total)

    @property
    def recent_rate(self) -> float:
        return self._rate(sum(self.recent_hits), len(self.recent_hits))


class _NgramPrecomputeMetrics:
    def __init__(self, log_interval: int) -> None:
        self.log_interval = max(0, log_interval)
        self.bonus_prediction = _HitRateTracker()
        self.precomputed_cache = _HitRateTracker()
        self.forward_count = 0

    @property
    def enabled(self) -> bool:
        return self.log_interval > 0

    def reset(self) -> None:
        self.forward_count = 0
        self.bonus_prediction.reset()
        self.precomputed_cache.reset()

    def record(
        self,
        *,
        bonus_prediction_hits: list[bool],
        precomputed_cache_hits: list[bool],
    ) -> None:
        if not self.enabled:
            return
        self.forward_count += 1
        self.bonus_prediction.record(bonus_prediction_hits)
        self.precomputed_cache.record(precomputed_cache_hits)
        if self.forward_count % self.log_interval != 0:
            return

        bonus = self.bonus_prediction
        cache = self.precomputed_cache
        logger.info(
            "NGRAM precompute stats over %d forward steps: "
            "bonus_prediction_hit_rate=%.4f (%d/%d), "
            "precomputed_cache_hit_rate=%.4f (%d/%d), "
            "avg_bonus_prediction_hit_rate=%.4f (%d/%d), "
            "avg_precomputed_cache_hit_rate=%.4f (%d/%d), "
            "last1000_bonus_prediction_hit_rate=%.4f (%d samples), "
            "last1000_precomputed_cache_hit_rate=%.4f (%d samples)",
            self.log_interval,
            bonus.interval_rate,
            bonus.interval_hits,
            bonus.interval_total,
            cache.interval_rate,
            cache.interval_hits,
            cache.interval_total,
            bonus.total_rate,
            bonus.total_hits,
            bonus.total,
            cache.total_rate,
            cache.total_hits,
            cache.total,
            bonus.recent_rate,
            len(bonus.recent_hits),
            cache.recent_rate,
            len(cache.recent_hits),
        )
        bonus.reset_interval()
        cache.reset_interval()


class NgramPrecomputeState:
    """Own the cross-iteration cache and asynchronous host staging.

    The worker provides request-local tensors and remains responsible for the
    corpus lookup. This object only manages precompute data lifetime and stream
    dependencies.
    """

    def __init__(
        self,
        *,
        device: str,
        draft_token_num: int,
        stats_log_interval: int,
    ) -> None:
        self.device = device
        self.draft_token_num = draft_token_num
        self._metrics = _NgramPrecomputeMetrics(stats_log_interval)
        self._copy_stream = None
        self._cache: Optional[_GpuPrecomputedCache] = None
        self._host_inputs: Optional[NgramPrecomputeInputs] = None
        self._pending_accept_copy: Optional[_PendingHostCopy] = None
        self._pending_tree_copy: Optional[_PendingHostCopy] = None

        mask_indices = torch.arange(draft_token_num, device=device)
        self.fallback_tree_mask = (
            (mask_indices[:, None] == mask_indices[None, :])
            | (mask_indices[None, :] == 0)
        ).unsqueeze(0)

    def reset(self) -> None:
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        self._cache = None
        self._host_inputs = None
        self._pending_accept_copy = None
        self._pending_tree_copy = None
        self._metrics.reset()

    def discard_cache(self) -> None:
        self._cache = None

    def set_bootstrap_inputs(
        self,
        *,
        accept_tokens: list[int],
        accept_lens: list[int],
        draft_tokens: np.ndarray,
        tree_mask: np.ndarray,
    ) -> None:
        self._ensure_inputs_consumed()
        self._cache = None
        self._host_inputs = NgramPrecomputeInputs(
            accept_tokens=list(accept_tokens),
            accept_lens=list(accept_lens),
            # batch_get returns request-owned arrays. Retain them directly: the
            # GPU copy and phase-2 precompute only read these buffers.
            draft_tokens=draft_tokens,
            tree_mask=tree_mask,
        )

    def _ensure_inputs_consumed(self) -> None:
        if (
            self._host_inputs is not None
            or self._pending_accept_copy is not None
            or self._pending_tree_copy is not None
        ):
            raise RuntimeError("Previous NGRAM precompute inputs were not consumed.")

    def _copy_stream_and_event(self):
        device_module = torch.get_device_module(self.device)
        if self._copy_stream is None:
            self._copy_stream = device_module.Stream()
        return device_module, self._copy_stream, device_module.Event()

    def _stage_host_copy(self, *tensors: torch.Tensor) -> _PendingHostCopy:
        device_module, copy_stream, copy_event = self._copy_stream_and_event()
        cpu_tensors = tuple(
            torch.empty_like(tensor, device="cpu", pin_memory=True)
            for tensor in tensors
        )
        current_stream = device_module.current_stream()
        with device_module.stream(copy_stream):
            # Device-side dependency only; the scheduler thread remains async.
            copy_stream.wait_stream(current_stream)
            for cpu_tensor, tensor in zip(cpu_tensors, tensors):
                cpu_tensor.copy_(tensor, non_blocking=True)
            copy_event.record()
        return _PendingHostCopy(ready=copy_event, tensors=cpu_tensors)

    def consume_if_available(
        self,
        *,
        req_ids: Sequence[str],
        accept_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_path_nodes: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
    ) -> bool:
        cache = self._cache
        if cache is None or accept_path_nodes is None:
            return False
        if not (
            accept_tokens.is_cuda and accept_lens.is_cuda and accept_path_nodes.is_cuda
        ):
            return False
        self._ensure_inputs_consumed()

        cache_rows = [cache.req_id_to_row.get(req_id, -1) for req_id in req_ids]
        device_module = torch.get_device_module(self.device)
        device_module.current_stream().wait_event(cache.copy_done)

        # This small copy can overlap the device-side cache selection below.
        self._pending_accept_copy = self._stage_host_copy(accept_tokens, accept_lens)
        cache_hits = apply_precomputed_drafts_for_rows(
            cache_rows=cache_rows,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
            accept_path_nodes=accept_path_nodes,
            cached_bonus_tokens=cache.bonus_tokens,
            cached_draft_tokens=cache.draft_tokens,
            cached_tree_masks=cache.tree_masks,
            fallback_tree_mask=self.fallback_tree_mask,
            draft_tokens=draft_tokens,
            tree_mask=tree_mask,
        )
        selected_tensors = [draft_tokens, tree_mask]
        if self._metrics.enabled:
            selected_tensors.append(cache_hits)
        self._pending_tree_copy = self._stage_host_copy(*selected_tensors)

        self._cache = None
        self._host_inputs = None
        return True

    def resolve_inputs(self) -> NgramPrecomputeInputs:
        if self._host_inputs is not None:
            inputs = self._host_inputs
            self._host_inputs = None
            return inputs

        accept_copy = self._pending_accept_copy
        tree_copy = self._pending_tree_copy
        if accept_copy is None or tree_copy is None:
            raise RuntimeError("NGRAM precompute inputs were not staged.")

        # Both copies use one stream and the tree copy is enqueued second, so
        # its completion also guarantees the earlier accept copy is ready.
        tree_copy.ready.synchronize()
        cpu_accept_tokens, cpu_accept_lens = accept_copy.tensors
        cpu_draft_tokens, cpu_tree_mask, *optional_hits = tree_copy.tensors

        self._pending_accept_copy = None
        self._pending_tree_copy = None
        if optional_hits:
            cache_hits = optional_hits[0].bool().tolist()
            # A dense cache entry exists exactly when its bonus prediction hits.
            self._metrics.record(
                bonus_prediction_hits=cache_hits,
                precomputed_cache_hits=cache_hits,
            )

        return NgramPrecomputeInputs(
            accept_tokens=cpu_accept_tokens.tolist(),
            accept_lens=cpu_accept_lens.tolist(),
            draft_tokens=cpu_draft_tokens.numpy(),
            tree_mask=cpu_tree_mask.numpy(),
        )

    def stage_cache(
        self,
        *,
        req_ids: Sequence[str],
        bonus_tokens: np.ndarray,
        draft_tokens: np.ndarray,
        tree_masks: np.ndarray,
    ) -> None:
        if self._cache is not None:
            raise RuntimeError("Previous NGRAM precompute cache was not consumed.")
        batch_size = len(req_ids)
        draft_token_num = self.draft_token_num
        _validate_cache_shapes(
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            bonus_tokens=bonus_tokens,
            draft_tokens=draft_tokens,
            tree_masks=tree_masks,
        )
        bonus_tokens_cpu = torch.as_tensor(bonus_tokens, dtype=torch.int32).pin_memory()
        draft_tokens_cpu = torch.as_tensor(draft_tokens, dtype=torch.int64).pin_memory()
        tree_masks_cpu = torch.as_tensor(tree_masks, dtype=torch.bool).pin_memory()

        device_module, copy_stream, copy_done = self._copy_stream_and_event()
        with device_module.stream(copy_stream):
            bonus_tokens_gpu = bonus_tokens_cpu.to(self.device, non_blocking=True)
            draft_tokens_gpu = draft_tokens_cpu.to(self.device, non_blocking=True)
            tree_masks_gpu = tree_masks_cpu.to(self.device, non_blocking=True)
            copy_done.record()

        self._cache = _GpuPrecomputedCache(
            req_id_to_row={req_id: row for row, req_id in enumerate(req_ids)},
            bonus_tokens=bonus_tokens_gpu,
            draft_tokens=draft_tokens_gpu,
            tree_masks=tree_masks_gpu,
            copy_done=copy_done,
        )


def _validate_cache_shapes(
    *,
    batch_size: int,
    draft_token_num: int,
    bonus_tokens: np.ndarray,
    draft_tokens: np.ndarray,
    tree_masks: np.ndarray,
) -> None:
    if bonus_tokens.ndim != 3:
        raise ValueError("bonus_tokens must have shape [batch, path, bonus_topk].")
    bonus_topk = bonus_tokens.shape[2]
    expected_prefix = (batch_size, draft_token_num, bonus_topk)
    if bonus_tokens.shape != expected_prefix:
        raise ValueError("bonus_tokens shape does not match the request batch.")
    if draft_tokens.shape != (*expected_prefix, draft_token_num):
        raise ValueError("draft_tokens shape does not match bonus_tokens.")
    if tree_masks.shape != (
        *expected_prefix,
        draft_token_num,
        draft_token_num,
    ):
        raise ValueError("tree_masks shape does not match bonus_tokens.")
