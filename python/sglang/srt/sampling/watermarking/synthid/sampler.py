# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SynthID applied to sampling logits.

The transform rewrites the logits of watermarked rows so that a stock sampler,
which divides by the temperature and samples from the softmax, draws from the
SynthID law: on each row the ``top_k`` candidates get ``score * T`` and every
other token gets ``-inf``. ``score`` is the log-probability after temperature,
the request filters, and the tournament; a row whose context is repeated (or
among the first skipped calls) keeps the filtered base law inside the cap.
Request filters (top-k, top-p, min-p) come from the caller as a callable and
run on the capped candidates before the watermark.

Caller contract: on watermarked rows the stock sampler runs with top_p=1 and
min_p=0; the request top-p and min-p belong in ``filter_probs``, before the
tournament. The request top-k may stay: after ``filter_probs`` a row's support
has at most top_k tokens, so the sampler's top-k keeps all of it. Without
``filter_probs`` the sampler's top-k would act after the tournament instead of
before it. The law holds up to rounding, since ``(score * T) / T`` is not
bit-equal to ``score``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang.kernels.ops.sampling.synthid import MAX_BLOCK_CANDIDATES
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    watermark_topk_scores_torch,
)
from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool

# Maps base probabilities ``[R, top_k]`` over each row's candidates to the
# filtered probabilities; a zero entry removes that candidate.
FilterProbs = Callable[[torch.Tensor], torch.Tensor]


def top_k_top_p_min_p_filter_probs(
    probs: torch.Tensor,
    *,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
) -> torch.Tensor:
    """The request top-k, top-p, and min-p of each row over ``probs`` ``[R, C]``,
    renormalized: a ``FilterProbs`` once the per-row values are bound. Keeps the
    same tokens as the pytorch sampling backend's filter (``min_ps`` None skips
    min-p)."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    rank = torch.arange(probs.shape[-1], device=probs.device).view(1, -1)
    probs_sort[rank >= top_ks.view(-1, 1)] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    if min_ps is not None:
        probs_sort[probs_sort < probs_sort[:, :1] * min_ps.view(-1, 1)] = 0.0
    filtered = torch.zeros_like(probs).scatter_(1, probs_idx, probs_sort)
    return filtered / filtered.sum(dim=-1, keepdim=True)


class SynthIDSampler:
    """SynthID keys, per-slot state, and the logits transforms."""

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
        in that order. Other rows are left untouched. The caller samples the
        watermarked rows under the module's caller contract (top_p=1, min_p=0).
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
        requests: Optional[torch.Tensor] = None,
        filter_probs: Optional[FilterProbs] = None,
    ) -> None:
        """Watermark the target logits of a linear verify step, in place.

        ``logits`` is ``[B * W, V]`` float32, ``W`` rows per request.
        ``requests`` (int64, ``[N]``) picks the watermarked requests, all ``B``
        when None; ``temperatures`` ``[N, 1]``, ``slots`` ``[N]`` and
        ``verify_tokens`` ``[N, W]`` (see ``SynthIDStatePool.project``) belong
        to those requests, and ``filter_probs`` sees their ``N * W`` rows in
        row-major order. The state is not committed.
        """
        width = verify_tokens.shape[1]
        context_hashes, apply_mask = self.state.project(slots, verify_tokens)
        rows = None
        if requests is not None:
            offsets = torch.arange(width, device=requests.device)
            rows = (requests[:, None] * width + offsets).view(-1)
        self._watermark_rows(
            logits,
            rows,
            temperatures.repeat_interleave(width, dim=0),
            context_hashes,
            apply_mask,
            filter_probs,
        )

    def can_fuse_sample(self, logits: torch.Tensor) -> bool:
        """Whether ``sample`` supports these logits and this config."""
        return (
            self.state.use_triton
            and logits.dtype == torch.float32
            and self.config.top_k <= MAX_BLOCK_CANDIDATES
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
        from sglang.kernels.ops.sampling.synthid.triton import synthid_sample_triton

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
            k=min(self.config.top_k, logits.shape[-1]),
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
        if scores.is_cuda:
            from sglang.kernels.ops.sampling.synthid.triton import (
                watermark_topk_scores_triton,
            )

            return watermark_topk_scores_triton(
                scores, token_ids, context_hashes, self.keys, apply_mask
            )
        return watermark_topk_scores_torch(
            scores, token_ids, context_hashes, self.keys, apply_mask
        )
