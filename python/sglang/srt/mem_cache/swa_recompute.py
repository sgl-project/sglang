from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.mem_cache.allocation import (
    alloc_paged_token_slots_extend,
    alloc_req_slots,
    alloc_token_slots,
    update_extend_kv_bookkeeping,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class SWARecomputeConfig(msgspec.Struct, frozen=True):
    """Model-specific sizing parameters for SWA-window recompute."""

    window_size: int
    gate_threshold: int

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.gate_threshold < self.window_size:
            raise ValueError("gate_threshold must be at least window_size")

    @classmethod
    def from_dimensions(
        cls,
        *,
        sliding_window_size: int,
        num_swa_layers: int,
        page_size: int,
        gate_multiplier: float,
    ) -> SWARecomputeConfig:
        if sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        if num_swa_layers <= 0:
            raise ValueError("num_swa_layers must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if not math.isfinite(gate_multiplier) or gate_multiplier < 1:
            raise ValueError("gate_multiplier must be finite and at least 1")

        # The last SWA layer needs the trailing W tokens, rounded up to a
        # multiple of page_size. Each preceding SWA layer extends the dependency
        # range backward by W tokens:
        # W_r = align_up(align_up(W, page_size) + (N - 1) * W, page_size).
        commit_tail = _ceil_div(sliding_window_size, page_size) * page_size
        unaligned_window = commit_tail + (num_swa_layers - 1) * sliding_window_size
        window_size = _ceil_div(unaligned_window, page_size) * page_size
        min_prefix_len = math.ceil(window_size * gate_multiplier)
        gate_threshold = _ceil_div(min_prefix_len, page_size) * page_size
        return cls(window_size=window_size, gate_threshold=gate_threshold)

    def validate_chunked_prefill_size(
        self, chunked_prefill_size: Optional[int], page_size: int
    ) -> None:
        if chunked_prefill_size is None or chunked_prefill_size <= 0:
            return
        required = self.window_size + page_size
        if chunked_prefill_size < required:
            raise ValueError(
                "SGLANG_OPT_SWA_RECOMPUTE_WINDOW=1 requires "
                f"--chunked-prefill-size >= W_r + page_size = {required} "
                "for this model "
                f"(W_r = page-aligned recompute window = {self.window_size}, "
                f"page_size = {page_size}). Got "
                f"--chunked-prefill-size={chunked_prefill_size}. Either raise "
                "the chunked prefill size or disable the SWA-window recompute "
                "feature."
            )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def reset_request_to_cold_prefill(req: Req, root_node: object) -> None:
    """Drop a recompute prefix hit that cannot be represented by this batch."""
    req.prefix_indices = req.prefix_indices[:0]
    req.last_node = root_node
    req.last_host_node = root_node
    req.best_match_node = root_node
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.mamba_host_hit_length = 0
    req.storage_hit_length = 0
    req.num_matched_prefix_tokens = 0
    req.cache_protected_len = 0
    req.extra_compute_prefix_len = 0


def resolve_swa_recompute_prefix(
    req: Req, tree_cache: UnifiedRadixCache
) -> tuple[int, bool]:
    """Fall back to cold prefill if load-back cannot cover the recompute window."""
    recompute_len = max(0, req.extra_compute_prefix_len)
    if recompute_len <= len(req.prefix_indices):
        return recompute_len, False
    reset_request_to_cold_prefill(req, tree_cache.root_node)
    return 0, True


class _SWARecomputeTxn(msgspec.Struct):
    req: Req
    recompute_len: int
    full_indices: torch.Tensor
    fresh_swa_indices: torch.Tensor


class SWARecomputeBatchState(msgspec.Struct):
    """Own the private-page transaction for one SWA recompute forward."""

    extra_compute_lens: list[int]
    logical_prefix_lens: list[int]
    is_pending: bool = True
    out_cache_loc_override: Optional[torch.Tensor] = None
    txns: list[_SWARecomputeTxn] = []

    @classmethod
    def create_if_needed(cls, batch: ScheduleBatch) -> Optional[SWARecomputeBatchState]:
        extra_compute_lens = [
            max(0, req.extra_compute_prefix_len) for req in batch.reqs
        ]
        if not any(extra_compute_lens):
            return None
        if not isinstance(batch.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            raise AssertionError(
                "SWA-window recompute requires a SWATokenToKVPoolAllocator"
            )
        if batch.model_config.is_encoder_decoder:
            raise AssertionError(
                "SWA-window recompute does not support encoder-decoder models"
            )
        return cls(
            extra_compute_lens=extra_compute_lens,
            logical_prefix_lens=[len(req.prefix_indices) for req in batch.reqs],
        )

    def prepare_inputs(self, reqs: list[Req]) -> tuple[list[object], list[int]]:
        input_ids = [
            req.get_fill_ids()[prefix_len - recompute_len :]
            for req, prefix_len, recompute_len in zip(
                reqs, self.logical_prefix_lens, self.extra_compute_lens
            )
        ]
        prefix_lens = [
            prefix_len - recompute_len
            for prefix_len, recompute_len in zip(
                self.logical_prefix_lens, self.extra_compute_lens
            )
        ]
        return input_ids, prefix_lens

    def validate_request(self, req: Req, wants_input_logprobs: bool) -> None:
        if req.input_embeds is not None:
            raise AssertionError("SWA-window recompute does not support input_embeds")
        if req.positional_embed_overrides is not None:
            raise AssertionError(
                "SWA-window recompute does not support positional embed overrides"
            )
        if wants_input_logprobs:
            raise AssertionError(
                "SWA-window recompute supports OUTPUT logprobs only; use "
                "logprob_start_len = -1 (or prompt len) to score the first "
                "generated token."
            )

    def split_cached_tokens_by_source(
        self,
        req_index: int,
        *,
        pre_len: int,
        host_hit_length: int,
        storage_hit_length: int,
    ) -> tuple[int, int, int]:
        """Exclude the recomputed suffix from device/L2/L3 hit accounting."""
        storage_total = min(host_hit_length, storage_hit_length)
        host_only_total = host_hit_length - storage_total
        recomputed_host = min(host_hit_length, self.extra_compute_lens[req_index])

        # L3 hits are the suffix of the host-hit segment, so recompute consumes
        # storage attribution before host-only attribution.
        storage_recomputed = min(storage_total, recomputed_host)
        storage_portion = storage_total - storage_recomputed
        host_recomputed = recomputed_host - storage_recomputed
        host_portion = host_only_total - min(host_only_total, host_recomputed)
        device_portion = max(0, pre_len - host_portion - storage_portion)
        return device_portion, host_portion, storage_portion

    def allocate(
        self, batch: ScheduleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self._allocate_full_slots(batch)
        self._allocate_private_swa_slots(batch)
        update_extend_kv_bookkeeping(batch)
        return result

    def apply_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        forward_batch.swa_out_cache_loc_override = self.out_cache_loc_override
        forward_batch.swa_recompute_boundaries = self.logical_prefix_lens
        forward_batch.allow_prefill_cuda_graph = False

    def _allocate_full_slots(
        self, batch: ScheduleBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse FULL prefix slots and allocate FULL slots only for the new tail."""
        batch.maybe_evict_swa()
        req_pool_indices = alloc_req_slots(
            batch.req_to_token_pool, batch.reqs, batch.tree_cache
        )
        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        req_pool_indices_device = req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )

        out_cache_parts: list[torch.Tensor] = []
        page_size = batch.tree_cache.page_size
        for req_idx, req, recompute_len in zip(
            req_pool_indices, batch.reqs, self.extra_compute_lens
        ):
            prefix_len = len(req.prefix_indices)
            seq_len = req.extend_range.end
            new_tokens = seq_len - prefix_len
            assert recompute_len % page_size == 0, f"{recompute_len=} {page_size=}"
            assert recompute_len <= prefix_len, f"{recompute_len=} {prefix_len=}"
            assert new_tokens >= 0, f"{new_tokens=}"

            batch.req_to_token_pool.req_to_token[req_idx, :prefix_len] = (
                req.prefix_indices
            )
            recompute_full = req.prefix_indices[prefix_len - recompute_len : prefix_len]

            if new_tokens > 0:
                last_loc = (
                    req.prefix_indices[prefix_len - 1 : prefix_len]
                    if prefix_len > 0
                    else torch.tensor([-1], dtype=torch.int64, device=batch.device)
                )
                if page_size == 1:
                    tail = alloc_token_slots(batch.tree_cache, new_tokens)
                else:
                    tail = alloc_paged_token_slots_extend(
                        tree_cache=batch.tree_cache,
                        prefix_lens=torch.tensor(
                            [prefix_len], dtype=torch.int64, device=batch.device
                        ),
                        prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
                        seq_lens=torch.tensor(
                            [seq_len], dtype=torch.int64, device=batch.device
                        ),
                        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
                        last_loc=last_loc,
                        extend_num_tokens=new_tokens,
                    )
                batch.req_to_token_pool.req_to_token[req_idx, prefix_len:seq_len] = tail
                out_cache_parts.append(
                    torch.cat([recompute_full, tail]) if recompute_len > 0 else tail
                )
            else:
                assert recompute_len > 0, (
                    "SWA recompute batch contains a request with neither a "
                    f"recompute window nor new tokens, rid={req.rid}"
                )
                out_cache_parts.append(recompute_full)

        return (
            torch.cat(out_cache_parts),
            req_pool_indices_device,
            req_pool_indices_cpu,
        )

    def _allocate_private_swa_slots(self, batch: ScheduleBatch) -> None:
        allocator = batch.token_to_kv_pool_allocator
        assert isinstance(allocator, SWATokenToKVPoolAllocator)
        total_recompute_tokens = sum(self.extra_compute_lens)

        # Evict before allocating private pages. The global FULL-to-SWA mapping
        # remains unchanged until the forward succeeds and commit() runs.
        available_before_evict = allocator.swa_attn_allocator.available_size()
        if total_recompute_tokens > available_before_evict:
            batch.tree_cache.evict(
                EvictParams(
                    swa_num_tokens=total_recompute_tokens - available_before_evict
                )
            )

        out_parts: list[torch.Tensor] = []
        try:
            for req, recompute_len in zip(batch.reqs, self.extra_compute_lens):
                prefix_len = len(req.prefix_indices)
                seq_len = req.extend_range.end
                if recompute_len > 0:
                    assert req.swa_prefix_lock_released, (
                        "SWA-window recompute requires a FULL-only request lock "
                        f"before allocating private SWA pages, rid={req.rid}"
                    )
                    full_indices = req.prefix_indices[
                        prefix_len - recompute_len : prefix_len
                    ]
                    fresh_swa = allocator.alloc_fresh_swa_for_recompute_window(
                        full_indices
                    )
                    if fresh_swa is None:
                        raise AssertionError(
                            "SWA-window recompute fresh allocation failed after "
                            f"eviction: recompute_len={recompute_len}, "
                            f"total_recompute_tokens={total_recompute_tokens}, "
                            f"available_before_evict={available_before_evict}, "
                            "available_after_evict="
                            f"{allocator.swa_attn_allocator.available_size()}, "
                            f"swa_evictable={batch.tree_cache.swa_evictable_size()}, "
                            f"page_size={allocator.page_size}, rid={req.rid}"
                        )
                    self.txns.append(
                        _SWARecomputeTxn(
                            req=req,
                            recompute_len=recompute_len,
                            full_indices=full_indices,
                            fresh_swa_indices=fresh_swa,
                        )
                    )
                    out_parts.append(fresh_swa)

                if seq_len > prefix_len:
                    assert req.req_pool_idx is not None
                    tail_full = batch.req_to_token_pool.req_to_token[
                        req.req_pool_idx, prefix_len:seq_len
                    ]
                    out_parts.append(
                        allocator.translate_loc_from_full_to_swa(tail_full)
                    )

            self.out_cache_loc_override = torch.cat(out_parts).to(torch.int32)
            assert len(self.out_cache_loc_override) == batch.extend_num_tokens, (
                f"{len(self.out_cache_loc_override)=}, " f"{batch.extend_num_tokens=}"
            )
        except Exception:
            self.abort(batch)
            raise

    def commit(self, batch: ScheduleBatch) -> None:
        if not self.txns:
            self.is_pending = False
            return

        allocator = batch.token_to_kv_pool_allocator
        assert isinstance(allocator, SWATokenToKVPoolAllocator)
        sliding_window_size = batch.tree_cache.sliding_window_size
        assert sliding_window_size is not None and sliding_window_size > 0

        commit_lens: list[int] = []
        for txn in self.txns:
            commit_len = _recompute_commit_len(
                txn.recompute_len, sliding_window_size, allocator.page_size
            )
            commit_lens.append(commit_len)

            # The whole recompute window is private workspace, but only the
            # ordinary SWA tail is reusable cache state after the forward.
            stale_len = txn.recompute_len - commit_len
            if stale_len > 0:
                allocator.free_fresh_swa(txn.fresh_swa_indices[:stale_len])
            if commit_len > 0:
                allocator.commit_fresh_swa_for_recompute_window(
                    txn.full_indices[-commit_len:],
                    txn.fresh_swa_indices[-commit_len:],
                )

        for txn, commit_len in zip(self.txns, commit_lens):
            req = txn.req
            if commit_len > 0:
                batch.tree_cache.complete_swa_recompute_lock(req, commit_len)
                assert req.kv is not None
                req.kv.swa_evicted_seqlen = min(
                    req.kv.swa_evicted_seqlen,
                    len(req.prefix_indices) - commit_len,
                )
            req.extra_compute_prefix_len = 0

        self.txns.clear()
        self.out_cache_loc_override = None
        self.is_pending = False

    def abort(self, batch: ScheduleBatch) -> None:
        if self.txns:
            allocator = batch.token_to_kv_pool_allocator
            assert isinstance(allocator, SWATokenToKVPoolAllocator)
            for txn in self.txns:
                allocator.free_fresh_swa(txn.fresh_swa_indices)
            self.txns.clear()
        self.out_cache_loc_override = None
        self.is_pending = False


def _recompute_commit_len(
    recompute_len: int, sliding_window_size: int, page_size: int
) -> int:
    tail_len = _ceil_div(sliding_window_size, page_size) * page_size
    return min(recompute_len, tail_len)
