"""Utilities for updating LongCat ngram embedding token tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.speculative.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ForwardMode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class NgramEmbeddingManager:
    enabled: bool
    table: Optional[torch.Tensor]
    n: int
    # Draft runners have no hasher even when their config has engram layers.
    engram_hasher: Optional[torch.nn.Module] = None

    @classmethod
    def from_model(
        cls,
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        req_to_token_pool: ReqToTokenPool,
        max_running_requests: int,
        device: str,
    ):
        token_table = None
        use_ngram_embedding = model_config.use_ngram_embedding
        if use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            token_table = torch.empty(
                req_to_token_pool.req_to_token.shape[0],
                model_config.context_len,
                dtype=torch.int32,
                device=device,
            )
            chunked_prefill_size = get_schedule().chunked_prefill_size
            assert chunked_prefill_size is not None and chunked_prefill_size > 0, (
                "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            )
            for module in model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        max_running_requests, chunked_prefill_size, device
                    )
        engram_hasher = None
        if model_config.engram_ngram_size > 0:
            from sglang.srt.layers.engram import EngramHasher

            for module in model.modules():
                if isinstance(module, EngramHasher):
                    assert engram_hasher is None, "one engram hasher per model"
                    module.init_history(req_to_token_pool.req_to_token.shape[0], device)
                    engram_hasher = module
        return cls(
            enabled=use_ngram_embedding,
            table=token_table,
            n=model_config.ngram_context_size,
            engram_hasher=engram_hasher,
        )

    def update_after_decode(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        update_ngram_token_table_after_sampling(
            ngram_embedding_info=ngram_embedding_info,
            next_token_ids=next_token_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            batch_size=forward_batch.batch_size,
        )

    def update_after_verify(
        self,
        *,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        if self.engram_hasher is None:
            return
        self.engram_hasher.commit_after_verify(
            verify_ids_2d, req_pool_indices, commit_lens
        )

    def prepare_for_forward(
        self,
        batch: Optional[ScheduleBatch],
        *,
        chunked_req: Optional[Req],
    ) -> Optional[ScheduleBatch]:
        if batch is None:
            return batch
        if self.engram_hasher is not None:
            self._prepare_engram_history(batch)
        if not self.enabled:
            return batch
        batch.ne_token_table = self.table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_range.length
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.n + 1 : end]
                    column_starts.append(start - self.n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.table.dtype
            device = self.table.device
            update_token_table(
                ne_token_table=self.table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
            # Mark the chunked (not-yet-finished) prefill request so sample()
            # skips writing its pseudo next-token into the ngram token table.
            # Use self.chunked_req identity (not req.is_chunked) to avoid
            # overlap-scheduling timing issues.
            if chunked_req is not None:
                skip_token_table_update = [req is chunked_req for req in batch.reqs]
                batch.ne_skip_token_table_update = (
                    torch.tensor(
                        skip_token_table_update, dtype=torch.bool, device=device
                    )
                    if any(skip_token_table_update)
                    else None
                )
        return batch

    def _prepare_engram_history(self, batch: ScheduleBatch) -> None:
        """Refresh extend predecessors after prefix hits, retraction, or slot reuse,
        and seed the history row of a request whose prefill ran on another server."""
        n1 = self.engram_hasher.max_ngram_size - 1
        if batch.forward_mode.is_prebuilt():
            # PD decode runs no EXTEND for this request, so the row its first
            # DECODE reads is written here: the n - 1 tokens before the one
            # prefill sampled, which is the token decode feeds next.
            history = self.engram_hasher.history
            rows = []
            for req in batch.reqs:
                fill_ids = req.origin_input_ids + req.output_ids
                end = len(fill_ids) - 1
                ids = fill_ids[max(0, end - n1) : end]
                rows.append([0] * (n1 - len(ids)) + list(ids))
            slots = torch.tensor(
                [req.kv.req_pool_idx for req in batch.reqs],
                dtype=torch.int64,
                device=history.device,
            )
            history[slots] = torch.tensor(
                rows, dtype=history.dtype, device=history.device
            ).view(len(rows), n1)
            return
        if not batch.forward_mode.is_extend_without_speculative():
            return
        rows = []
        for req in batch.reqs:
            start = req.extend_range.start
            lo = max(0, start - n1)
            ids = req.full_untruncated_fill_ids[lo:start]
            rows.append([0] * (n1 - len(ids)) + list(ids))
        batch.ne_history = torch.tensor(
            rows, dtype=torch.int32, device=self.engram_hasher.history.device
        ).view(len(rows), n1)


def update_ngram_token_table_after_sampling(
    *,
    ngram_embedding_info,
    next_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
) -> bool:
    """Update the ngram token table with sampled tokens.

    Returns whether the token table was updated.
    """
    skip_token_table_update = ngram_embedding_info.skip_token_table_update
    if skip_token_table_update is not None:
        # Skip chunked (not-yet-finished) prefill requests: their sampled token
        # is a pseudo prediction and must not pollute the token table.
        indices = (~skip_token_table_update).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return False
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids[indices].to(torch.int32),
            row_indices=req_pool_indices[indices],
            column_starts=seq_lens[indices].to(torch.int32),
            req_lens=torch.ones(
                indices.numel(), dtype=torch.int32, device=next_token_ids.device
            ),
            ignore_tokens=None,
        )
        return True

    # NGRAM_BS_FIX: seq_lens / next_token_ids / req_pool_indices may be padded to the
    # cuda-graph batch size while batch_size is the real request count. Slice to
    # batch_size so padded rows don't pollute the token table (and shapes match).
    ngram_embedding_info.out_column_starts[:batch_size] = seq_lens[:batch_size]
    ngram_embedding_info.out_req_lens[:batch_size] = 1
    update_token_table(
        ne_token_table=ngram_embedding_info.token_table,
        tokens=next_token_ids[:batch_size].to(torch.int32),
        row_indices=req_pool_indices[:batch_size],
        column_starts=ngram_embedding_info.out_column_starts[:batch_size],
        req_lens=ngram_embedding_info.out_req_lens[:batch_size],
        ignore_tokens=None,
    )
    return True
