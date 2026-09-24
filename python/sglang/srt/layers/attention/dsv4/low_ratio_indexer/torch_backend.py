"""The candidate scheme on the portable torch path: bf16 scores materialized per
request and selected with ``torch.topk``, the candidates carried by boolean masks.
A source keeps a mask of its best blocks, a consumer scores densely and drops
everything outside it; it needs no page table, so it works wherever torch does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec
import torch

from .block_math import (
    mask_topk_scores,
    select_candidate_blocks,
)
from .inputs import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    Selection,
)
from .torch_utils import (
    decode_scores,
    prefill_requests,
    write_decode,
    write_prefill,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class CandidateMasks(CandidateMetadata, msgspec.Struct):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each

    def tail(self, rows_per_request: List[int]) -> CandidateMasks:
        assert self.request_masks is not None, "prefill masks missing"
        return CandidateMasks(
            request_masks=[
                mask[mask.shape[0] - t :]
                for mask, t in zip(self.request_masks, rows_per_request)
            ]
        )


class TorchCandidateBackend:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token

    def publish_prefill(self, inputs: PrefillInputs, out: Selection):
        indexer = inputs.indexer
        published = []
        for request, chunks in self._prefill_requests(inputs, out):
            if request.lc == 0:
                # Consumers address masks by request position, including empty requests.
                published.append(
                    torch.zeros(0, 0, dtype=torch.bool, device=inputs.positions.device)
                )
                continue
            masks = []
            for chunk in chunks:
                masks.append(
                    select_candidate_blocks(
                        chunk.scores,
                        chunk.lens[:, None],
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    )
                )
                idx = chunk.scores.topk(request.k, dim=-1, sorted=False).indices
                write_prefill(out, request, chunk, idx)
            published.append(torch.cat(masks) if len(masks) > 1 else masks[0])
        return CandidateMasks(request_masks=published)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[CandidateMasks],
        out: Selection,
    ) -> None:
        assert published is not None
        request_masks = published.request_masks
        for request, chunks in self._prefill_requests(inputs, out):
            for chunk in chunks:
                keep = request_masks[request.index][chunk.rows]
                s = chunk.scores.masked_fill(~keep, -torch.inf)
                idx = s.topk(request.k, dim=-1, sorted=False).indices
                idx = mask_topk_scores(s, idx)
                idx = idx.masked_fill(idx < 0, request.lc)
                write_prefill(out, request, chunk, idx)

    def publish_decode(self, inputs: DecodeInputs, out: Selection):
        indexer = inputs.indexer
        d = self._decode_scores(inputs, out)
        if d is None:
            return None
        published = CandidateMasks(
            mask=select_candidate_blocks(
                d.scores,
                d.lens[:, None],
                topk_blocks=indexer.candidate_topk_blocks,
                block_size=indexer.candidate_block_size,
            )
        )
        k = min(indexer.index_topk, d.lmax)
        idx = d.scores.topk(k, dim=-1, sorted=False).indices
        write_decode(out, d, idx)
        return published

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[CandidateMasks],
        out: Selection,
    ) -> None:
        d = self._decode_scores(inputs, out)
        if d is None:
            return
        assert published is not None
        consume = published.mask
        assert torch.is_tensor(consume) and consume.shape[0] == d.bs
        s = d.scores.masked_fill(~consume[:, : d.lmax], -torch.inf)
        k = min(inputs.indexer.index_topk, d.lmax)
        idx = s.topk(k, dim=-1, sorted=False).indices
        idx = mask_topk_scores(s, idx)
        idx = idx.masked_fill(idx < 0, d.lmax)
        write_decode(out, d, idx)

    def _prefill_requests(self, inputs: PrefillInputs, out: Selection):
        return prefill_requests(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )

    def _decode_scores(self, inputs: DecodeInputs, out: Selection):
        return decode_scores(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )
