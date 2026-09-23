from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateIndexer,
    CandidateMetadata,
    PrefillCandidateBlocks,
    TritonDecodeInputs,
    TritonPrefillInputs,
    select_candidate_block_ids,
)


class TritonDecodeCandidates(CandidateMetadata):
    def __init__(self, blocks):
        self.blocks = blocks  # [rows, budget], -1 padded logical block IDs


class TritonCandidateIndexer(CandidateIndexer):
    """Triton candidate scoring for prefill and decode/verify.

    Sources score their shared dense K matrix. Consumers always read their
    candidate K directly without a per-query gather tensor.
    The budget bounds per-tile working tensors, not the request K or outputs;
    at least one query row is processed even if it exceeds the budget.
    """

    def __init__(
        self,
        topk_blocks: int,
        block_size: int,
        *,
        budget_bytes: int = 1 << 30,
        decode_indexer: CandidateIndexer | None = None,
    ):
        if topk_blocks <= 0 or block_size <= 0 or budget_bytes <= 0:
            raise ValueError("candidate sizes and tile budget must be positive")
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.budget_bytes = budget_bytes
        # The explicit SM100 prefill override retains its DeepGEMM decode backend.
        self.decode_indexer = decode_indexer

    @staticmethod
    def _decode_scores(inputs, blocks=None, block_size=8):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            fp4_index_logits_decode,
        )

        return fp4_index_logits_decode(
            inputs.q,
            inputs.weights,
            inputs.slots,
            inputs.lens,
            inputs.table,
            inputs.page_size,
            candidate_blocks=blocks,
            candidate_block_size=block_size,
        )

    @staticmethod
    def _decode_topk(
        scores, inputs, page_indices, raw_indices, blocks=None, block_size=8
    ):
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        k = min(inputs.topk, scores.shape[1])
        if scores.shape[0] == 0 or k == 0 or inputs.slots.shape[1] == 0:
            return
        values, columns = scores.topk(k, dim=-1, sorted=False)
        logical = columns
        if blocks is not None:
            logical = (
                blocks.long().gather(1, columns // block_size) * block_size
                + columns % block_size
            )
        valid = (
            (values > -torch.inf)
            & (logical >= 0)
            & (logical < inputs.lens[:, None])
            & (logical < inputs.slots.shape[1])
        )
        sentinel = torch.iinfo(torch.int64).max
        logical = logical.masked_fill(~valid, sentinel).sort(dim=-1).values
        valid = logical != sentinel
        physical = inputs.slots.gather(1, logical.clamp(0, inputs.slots.shape[1] - 1))
        page_indices[: scores.shape[0], :k].copy_(
            torch.where(valid, physical, -1).int()
        )
        if raw_indices is not None:
            raw_indices[: scores.shape[0], :k].copy_(
                torch.where(valid, logical, -1).int()
            )

    def publish_decode(self, inputs, page_indices, raw_indices=None):
        if not isinstance(inputs, TritonDecodeInputs):
            # The SM100 prefill override keeps its existing decode implementation.
            if self.decode_indexer is None:
                raise TypeError("Triton decode requires unpacked inputs")
            return self.decode_indexer.publish_decode(inputs, page_indices, raw_indices)
        scores = self._decode_scores(inputs)
        blocks = torch.full(
            (inputs.q.shape[0], self.topk_blocks),
            -1,
            device=inputs.q.device,
            dtype=torch.int32,
        )
        if scores.shape[0] and scores.shape[1]:
            chosen = select_candidate_block_ids(
                scores, inputs.lens[:, None], self.topk_blocks, self.block_size
            )
            blocks[:, : chosen.shape[1]].copy_(chosen)
        self._decode_topk(scores, inputs, page_indices, raw_indices)
        return TritonDecodeCandidates(blocks)

    def select_decode(self, published, inputs, page_indices, raw_indices=None):
        if not isinstance(inputs, TritonDecodeInputs):
            if self.decode_indexer is None:
                raise TypeError("Triton decode requires unpacked inputs")
            return self.decode_indexer.select_decode(
                published, inputs, page_indices, raw_indices
            )
        if not isinstance(published, TritonDecodeCandidates):
            raise TypeError("Triton decode candidate blocks missing")
        if published.blocks.shape != (inputs.q.shape[0], self.topk_blocks):
            raise ValueError("decode candidate rows/budget do not match")
        scores = self._decode_scores(inputs, published.blocks, self.block_size)
        self._decode_topk(
            scores, inputs, page_indices, raw_indices, published.blocks, self.block_size
        )

    @staticmethod
    def plain_decode(inputs, page_indices, raw_indices=None):
        TritonCandidateIndexer._decode_topk(
            TritonCandidateIndexer._decode_scores(inputs),
            inputs,
            page_indices,
            raw_indices,
        )

    def _rows_per_tile(self, inputs, width: int) -> int:
        # Bound dense score intermediates and leave room for Top-K scratch.
        # Use the same conservative row budget for the fused sparse scorer.
        heads = inputs.q.shape[1]
        per_position = 3 * heads * inputs.q.element_size() + 32
        return max(1, self.budget_bytes // max(1, width * per_position))

    @staticmethod
    def _dense_scores(q, keys, weights):
        # Shared 2-D K only: source and plain index layers.
        scores = torch.einsum("rhd,nd->rhn", q, keys)
        return (scores.relu() * weights.unsqueeze(-1)).sum(dim=1).float()

    @staticmethod
    def _candidate_scores(q, keys, weights, blocks, lens):
        from sglang.kernels.ops.attention.dsv4.candidate_bf16_mqa import (
            candidate_bf16_mqa_logits,
        )

        return candidate_bf16_mqa_logits(q, keys, weights, blocks, lens)

    def _write_topk(self, scores, blocks, starts, out):
        k = min(out.shape[1], scores.shape[1])
        if k == 0:
            return
        values, selected = scores.topk(k, dim=-1, sorted=False)
        if blocks is not None:
            # Restore only the selected columns, not the full candidate matrix.
            selected = (
                blocks.gather(1, selected // self.block_size).long() * self.block_size
                + selected % self.block_size
            )
        selected = (selected + starts[:, None]).masked_fill(~(values > -torch.inf), -1)
        out[:, :k].copy_(selected)

    @staticmethod
    def _requests(inputs, out_positions):
        assert isinstance(inputs, TritonPrefillInputs)
        if (
            len(inputs.rows_per_request) != len(inputs.lens_per_request)
            or sum(inputs.rows_per_request) != inputs.num_rows
            or out_positions.shape[0] != inputs.num_rows
        ):
            raise ValueError("prefill request rows do not match operands")
        out_positions.fill_(-1)
        start = 0
        for b, (rows, length) in enumerate(
            zip(inputs.rows_per_request, inputs.lens_per_request)
        ):
            yield b, start, rows, length
            start += rows

    def publish_prefill(self, inputs, out_positions) -> PrefillCandidateBlocks:
        return self._dense(inputs, out_positions, publish=True)

    def plain_prefill(self, inputs, out_positions) -> None:
        """Score non-candidate layers against their shared dense K matrix."""
        self._dense(inputs, out_positions, publish=False)

    def _dense(self, inputs, out_positions, *, publish):
        request_blocks = []
        for b, start, rows, length in self._requests(inputs, out_positions):
            nb = min(
                self.topk_blocks, (length + self.block_size - 1) // self.block_size
            )
            blocks = torch.full(
                (rows, nb if publish else 0),
                -1,
                dtype=torch.int32,
                device=inputs.q.device,
            )
            request_blocks.append(blocks)
            if rows == 0 or length == 0:
                continue
            keys = inputs.get_keys(b)
            assert keys.shape == (length, inputs.q.shape[-1])
            columns = torch.arange(length, device=inputs.q.device)
            step = self._rows_per_tile(inputs, length)
            for offset in range(0, rows, step):
                local = slice(offset, min(rows, offset + step))
                tile = slice(start + local.start, start + local.stop)
                scores = self._dense_scores(inputs.q[tile], keys, inputs.weights[tile])
                scores.masked_fill_(
                    columns[None, :] >= inputs.compress_lens[tile, None], -torch.inf
                )
                if publish:
                    blocks[local].copy_(
                        select_candidate_block_ids(
                            scores,
                            inputs.compress_lens[tile, None],
                            self.topk_blocks,
                            self.block_size,
                        )
                    )
                self._write_topk(
                    scores, None, inputs.request_starts[tile], out_positions[tile]
                )
            del keys
        return PrefillCandidateBlocks(request_blocks=request_blocks)

    def select_prefill(self, published, inputs, out_positions) -> None:
        assert isinstance(published, PrefillCandidateBlocks), "candidate blocks missing"
        if len(published.request_blocks) != len(inputs.rows_per_request):
            raise ValueError("candidate requests do not match inputs")
        for b, start, rows, length in self._requests(inputs, out_positions):
            blocks = published.request_blocks[b]
            if blocks.shape[0] != rows:
                raise ValueError("candidate rows do not match query rows")
            width = blocks.shape[1] * self.block_size
            if rows == 0 or length == 0 or width == 0:
                continue
            keys = inputs.get_keys(b)
            assert keys.shape == (length, inputs.q.shape[-1])
            if self.block_size != 8:
                raise ValueError("Triton prefill candidates require block_size=8")
            step = self._rows_per_tile(inputs, width)
            for offset in range(0, rows, step):
                local = slice(offset, min(rows, offset + step))
                tile = slice(start + local.start, start + local.stop)
                scores = self._candidate_scores(
                    inputs.q[tile],
                    keys,
                    inputs.weights[tile],
                    blocks[local],
                    inputs.compress_lens[tile],
                )
                self._write_topk(
                    scores,
                    blocks[local],
                    inputs.request_starts[tile],
                    out_positions[tile],
                )
                del scores
            del keys

    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: list[int]
    ) -> PrefillCandidateBlocks:
        assert isinstance(published, PrefillCandidateBlocks), "candidate blocks missing"
        return published.tail(tail_lens)
