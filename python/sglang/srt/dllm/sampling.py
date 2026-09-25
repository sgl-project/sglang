"""Per-request sampling for diffusion-LLM block decoding.

A dLLM forward predicts every position of a block at once and commits the ones
whose prediction clears a confidence threshold. Sampling has to replace both
halves of that: the token is drawn from the tempered/filtered distribution, and
the confidence gating the commit becomes the draw's own, so an unlikely draw
defers the position to a later forward instead of freezing into the block. That
deferral is the quality mechanism, and LLaDA2's reference ``generate``
(``_sample_with_temperature_topk_topp`` in modeling_llada2_moe.py) gates on the
drawn token's probability for the same reason.

The commit confidence is the smaller of two quantities, both tested against the
threshold the algorithm already applies to its greedy confidence:

- the argmax probability, unchanged from greedy: has this position been decided
  at all;
- the drawn token's probability relative to the position's most likely token:
  is the draw a plausible token to put there.

Neither alone holds accuracy. The argmax probability alone lets an unlikely draw
freeze in; the relative confidence alone drops the "is it decided" test and
commits into flat distributions. A drawn token's *absolute* probability cannot
play the second role either, because it sits systematically below the argmax
token's and so almost never clears a bar greedy clears routinely, leaving every
sampled position to the algorithms' forced-top-1 fallback.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.sampler import (
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils.common import is_cuda, print_warning_once

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

SUPPORTED_PARAMS = "temperature, top_p, top_k and min_p"


def _warn_unsupported(sampling_info: SamplingBatchInfo) -> None:
    """Report, once per process, sampling params a dLLM batch silently drops."""
    unsupported = []
    if (
        sampling_info.penalizer_orchestrator is not None
        and sampling_info.penalizer_orchestrator.is_required
    ):
        unsupported.append("penalties")
    if sampling_info.logit_bias is not None:
        unsupported.append("logit_bias")
    if sampling_info.grammars is not None and any(sampling_info.grammars):
        unsupported.append("grammar-guided decoding")
    if sampling_info.has_custom_logit_processor:
        unsupported.append("custom logit processors")
    if unsupported:
        print_warning_once(
            f"Diffusion LLM decoding ignores {', '.join(unsupported)}; "
            f"only {SUPPORTED_PARAMS} are applied."
        )


def _flashinfer_sample(
    *,
    probs: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    req_ids: torch.Tensor,
) -> torch.Tensor:
    if sampling_info.need_min_p_sampling:
        probs = top_k_renorm_prob(probs, sampling_info.top_ks[req_ids])
        probs = top_p_renorm_prob(probs, sampling_info.top_ps[req_ids])
        return min_p_sampling_from_probs(probs, sampling_info.min_ps[req_ids])
    return top_k_top_p_sampling_from_probs(
        probs.contiguous(),
        sampling_info.top_ks[req_ids],
        sampling_info.top_ps[req_ids],
        filter_apply_order="joint",
    )


class DllmSamplingPlan(msgspec.Struct):
    """Sampling inputs for one denoise step, built only when a row is non-greedy.

    ``maybe_build`` returns None for an all-greedy batch, which is the signal for
    the algorithms to take their original argmax-only path unchanged.
    """

    sampling_info: SamplingBatchInfo
    # Request indices that sample, on device for the batched paths and as host
    # flags for the per-row loop.
    sampled_rows: torch.Tensor
    row_samples: List[bool]
    flashinfer_backend: bool

    @staticmethod
    def maybe_build(
        sampling_info: Optional[SamplingBatchInfo],
    ) -> Optional[DllmSamplingPlan]:
        if sampling_info is None or sampling_info.is_all_greedy:
            return None
        assert sampling_info.sampling_seed is None, (
            "Deterministic sampling is not supported for diffusion LLM decoding"
        )
        _warn_unsupported(sampling_info)
        # SamplingParams.normalize rewrites temperature == 0 to (1.0, top_k=1),
        # so top_k <= 1 is the whole greedy predicate.
        row_samples = sampling_info.top_ks > 1
        return DllmSamplingPlan(
            sampling_info=sampling_info,
            sampled_rows=row_samples.nonzero().squeeze(-1),
            row_samples=row_samples.tolist(),
            # The device check is the kernels' own precondition, and it keeps the
            # runtime context off the path for CPU-tensor callers (tests).
            flashinfer_backend=sampling_info.top_ks.is_cuda
            and get_exec().kernel.sampling_backend == "flashinfer",
        )

    def sample(
        self,
        *,
        logits: torch.Tensor,
        req_ids: torch.Tensor,
        argmax_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw one token per row of ``logits`` ([num_positions, vocab]).

        ``req_ids`` gives the request each row belongs to, indexing the
        per-request sampling params; ``argmax_probs`` is the greedy confidence for
        the same rows. Returns (drawn token ids, confidence to commit them on).
        """
        sampling_info = self.sampling_info
        # Out of place: the caller's logits are reused by the argmax path.
        probs = torch.softmax(
            logits / sampling_info.temperatures[req_ids], dim=-1, dtype=torch.float32
        )

        if (
            not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            tokens = sampling_from_probs_torch(probs)
        elif self.flashinfer_backend:
            # Rejection sampling rather than a full sort of the vocab. A dLLM step
            # draws block_size positions per request instead of one, so the sort
            # costs block_size times what it does for autoregressive decoding.
            tokens = _flashinfer_sample(
                probs=probs, sampling_info=sampling_info, req_ids=req_ids
            )
        else:
            tokens = top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                sampling_info.top_ks[req_ids],
                sampling_info.top_ps[req_ids],
                sampling_info.min_ps[req_ids],
                sampling_info.need_min_p_sampling,
                None,
                None,
            )
        drawn = self.confidence(logits=logits, req_ids=req_ids, token_ids=tokens)
        return tokens, torch.minimum(argmax_probs, drawn.to(argmax_probs.dtype))

    def confidence(
        self,
        *,
        logits: torch.Tensor,
        req_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """How likely each token is relative to its position's most likely token,
        under the request's tempered distribution: ``exp((z_t - z_max) / T)``.
        """
        # Cast the two reduced vectors, never the [num_positions, vocab] logits.
        top_logits = logits.max(dim=-1).values.float()
        token_logits = (
            torch.gather(logits, dim=-1, index=token_ids.long().view(-1, 1))
            .view(-1)
            .float()
        )
        temperatures = self.sampling_info.temperatures[req_ids].view(-1)
        return torch.exp((token_logits - top_logits) / temperatures)
