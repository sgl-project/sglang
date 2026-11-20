"""SynthID applied to sampling logits.

The transform rewrites the logits of watermarked rows so that a stock sampler,
which divides by the temperature and samples from the softmax, draws from the
SynthID law: on each row the ``num_candidates`` candidates get ``score * T`` and
every other token gets ``-inf``. ``score`` is the log-probability after
temperature, the request filters, and the tournament; a row whose context is
repeated (or among the first skipped calls) keeps the filtered base law inside
the cap.

Caller contract: the request top-k, top-p, and min-p run as ``filter_probs``,
before the tournament; the stock sampler then draws the watermarked rows with
top_p=1 and min_p=0.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang.kernels.ops.sampling.synthid import MAX_BLOCK_CANDIDATES
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    synthid_tournament_torch,
)
from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool

# Maps base probabilities ``[R, num_candidates]`` over each row's candidates to
# the filtered probabilities; a zero entry removes that candidate.
FilterProbs = Callable[[torch.Tensor], torch.Tensor]


class SynthIDWatermark:
    def __init__(
        self,
        config: SynthIDConfig,
        num_slots: int,
        device: str | torch.device,
    ):
        self.config = config
        self.keys = torch.tensor(config.keys, dtype=torch.int64, device=device)
        self.state = SynthIDStatePool(config, num_slots, device)

    def apply(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        slots: torch.Tensor,
        *,
        rows: Optional[torch.Tensor] = None,
        filter_probs: Optional[FilterProbs] = None,
    ) -> None:
        """Watermark the next-token logits of ordinary sampling, in place.

        ``logits`` is ``[B, V]`` float32. ``rows`` (int64, ``[N]``) picks the
        watermarked rows, all ``B`` when None; ``temperatures`` ``[N, 1]`` and
        ``slots`` ``[N]`` belong to those rows, and ``filter_probs`` sees them
        in that order. Other rows are left untouched.
        """
        context_hashes, apply_mask = self.state.current(slots)
        self._watermark_rows(
            logits, rows, temperatures, context_hashes, apply_mask, filter_probs
        )

    def apply_verify(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        slots: torch.Tensor,
        verify_tokens: torch.Tensor,
        *,
        request_rows: Optional[torch.Tensor] = None,
        filter_probs: Optional[FilterProbs] = None,
    ) -> None:
        """Watermark the target logits of a linear verify step, in place.

        ``logits`` is ``[B * W, V]`` float32, ``W`` rows per request.
        ``request_rows`` (int64, ``[N]``) picks the watermarked requests, all ``B``
        when None; ``temperatures`` ``[N, 1]``, ``slots`` ``[N]`` and
        ``verify_tokens`` ``[N, W]`` (see ``SynthIDStatePool.project``) belong
        to those requests, and ``filter_probs`` sees their ``N * W`` rows in
        row-major order. The state is not committed.
        """
        width = verify_tokens.shape[1]
        context_hashes, apply_mask = self.state.project(slots, verify_tokens)
        rows = None
        if request_rows is not None:
            offsets = torch.arange(width, device=request_rows.device)
            rows = (request_rows[:, None] * width + offsets).view(-1)
        self._watermark_rows(
            logits,
            rows,
            temperatures.repeat_interleave(width, dim=0),
            context_hashes,
            apply_mask,
            filter_probs,
        )

    def can_fuse_sample(self) -> bool:
        return (
            self.state.use_triton and self.config.num_candidates <= MAX_BLOCK_CANDIDATES
        )

    def sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Sample every row from the SynthID law in one kernel.

        For a batch whose rows are all watermarked and have no request filters,
        logprob, or sampling-mask outputs. ``logits`` ``[B, V]`` float32 is read
        only; returns int32 token ids ``[B]``, which the caller commits with
        ``state.commit``.
        """
        from sglang.kernels.ops.sampling.synthid_triton import synthid_sample_triton

        scores, token_ids = self._candidates(logits)
        exponential_noise = torch.empty_like(scores).exponential_(1.0)
        state = self.state
        return synthid_sample_triton(
            scores,
            token_ids,
            temperatures,
            slots,
            state.context,
            state.context_history,
            state.num_calls,
            self.keys,
            exponential_noise,
            hash_iv=self.config.hash_iv,
            skip_first_ngram_calls=self.config.skip_first_ngram_calls,
        )

    def _watermark_rows(
        self,
        logits: torch.Tensor,
        rows: Optional[torch.Tensor],
        temperatures: torch.Tensor,
        context_hashes: torch.Tensor,
        apply_mask: torch.Tensor,
        filter_probs: Optional[FilterProbs],
    ) -> None:
        selected = logits if rows is None else logits[rows]
        scores, token_ids = self._candidates(selected)
        scores.div_(temperatures)

        filtered_probs = None
        if filter_probs is not None:
            filtered_probs = filter_probs(torch.softmax(scores, dim=-1))
            scores = filtered_probs.log()

        scores = self._tournament(
            scores.contiguous(),
            token_ids.contiguous(),
            context_hashes.contiguous(),
            apply_mask.contiguous(),
        )
        if filtered_probs is not None:
            scores = scores.masked_fill(filtered_probs == 0, -float("inf"))
        # The stock sampler divides by the temperature again.
        scores.mul_(temperatures)

        if rows is None:
            logits.fill_(-float("inf"))
            logits.scatter_(1, token_ids, scores)
        else:
            logits.index_fill_(0, rows, -float("inf"))
            logits[rows[:, None], token_ids] = scores

    def _candidates(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(
            logits,
            k=min(self.config.num_candidates, logits.shape[-1]),
            dim=-1,
            sorted=False,
        )

    def _tournament(
        self,
        scores: torch.Tensor,
        token_ids: torch.Tensor,
        context_hashes: torch.Tensor,
        apply_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.state.use_triton:
            from sglang.kernels.ops.sampling.synthid_triton import (
                synthid_tournament_triton,
            )

            return synthid_tournament_triton(
                scores, token_ids, context_hashes, self.keys, apply_mask
            )
        return synthid_tournament_torch(
            scores, token_ids, context_hashes, self.keys, apply_mask
        )
