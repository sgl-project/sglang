from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dflash_utils import (
    build_speculative_verify_target_probs,
    is_dflash_sampling_verify_available,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def validate_spec_sampling_mask_request(
    req: Req, spec_algorithm: SpeculativeAlgorithm
) -> Optional[str]:
    """Reject configurations whose committed tokens are not exact samples from
    the captured target distribution."""
    is_ngram = spec_algorithm.is_ngram()
    if not (spec_algorithm.is_dflash_family() or is_ngram):
        return "return_sampling_mask is not supported with speculative decoding."

    if is_ngram and (
        not is_cuda() or envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
    ):
        return "return_sampling_mask with NGRAM requires the CUDA tree verifier."

    if req.sampling_params.min_p > 0:
        return "return_sampling_mask with speculative decoding does not support min_p."

    if envs.SGLANG_SIMULATE_ACC_LEN.get() > 0:
        return (
            "return_sampling_mask is not supported with simulated speculative "
            "acceptance."
        )

    if spec_algorithm.is_dflash() or is_ngram:
        if (
            get_spec().speculative_accept_threshold_single != 1.0
            or get_spec().speculative_accept_threshold_acc != 1.0
        ):
            return (
                f"return_sampling_mask with {spec_algorithm.name} requires acceptance thresholds "
                "of 1.0."
            )
        # Without the kernel DFlash verifies non-greedy rows by greedy argmax.
        if (
            spec_algorithm.is_dflash()
            and req.sampling_params.top_k > 1
            and not is_dflash_sampling_verify_available()
        ):
            return (
                "return_sampling_mask with non-greedy DFlash decoding requires "
                "sampling verification support."
            )

    return None


class SpeculativeSamplingMaskCapture(msgspec.Struct):
    target_probs: torch.Tensor | None
    batch_indices: torch.Tensor
    max_top_k: int
    greedy_mask: torch.Tensor | None = None
    support_capture_indices: torch.Tensor | None = None

    @classmethod
    def from_logits(
        cls,
        sampling_info,
        *,
        next_token_logits: torch.Tensor,
        draft_input,
        draft_token_num: int,
        bs: int,
        greedy_mask: torch.Tensor | None = None,
    ) -> SpeculativeSamplingMaskCapture | None:
        """Rebuild the verified target policy after acceptance; returns None
        when no request in the batch asked for sampling masks."""
        if sampling_info is None or sampling_info.sampling_mask_batch_indices is None:
            return None
        target_probs = None
        if not sampling_info.is_all_greedy:
            target_probs = build_speculative_verify_target_probs(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
                bs=bs,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
        return cls(
            target_probs=target_probs,
            batch_indices=sampling_info.sampling_mask_batch_indices,
            max_top_k=max(sampling_info.sampling_mask_top_ks),
            greedy_mask=greedy_mask,
            support_capture_indices=sampling_info.sampling_support_logprobs_capture_indices,
        )

    def build_output(
        self,
        *,
        out_tokens: torch.Tensor,
        commit_lens: torch.Tensor,
        accept_index: torch.Tensor | None = None,
    ) -> SamplingMaskOutput:
        batch_indices = self.batch_indices
        output_tokens = out_tokens.index_select(0, batch_indices).clone()
        num_accept_tokens = commit_lens.index_select(0, batch_indices).clone()
        greedy_mask = (
            None
            if self.greedy_mask is None
            else self.greedy_mask.index_select(0, batch_indices).clone()
        )

        support_logprobs = None
        if self.target_probs is None:
            support_tokens = output_tokens.unsqueeze(-1).to(torch.int32)
            support_lens = torch.ones_like(output_tokens, dtype=torch.int32)
            selected_logprobs = torch.zeros_like(output_tokens, dtype=torch.float32)
            if self.support_capture_indices is not None:
                support_logprobs = selected_logprobs.index_select(
                    0, self.support_capture_indices
                ).unsqueeze(-1)
        else:
            max_top_k = min(int(self.max_top_k), self.target_probs.shape[-1])
            if max_top_k <= 0:
                raise ValueError(
                    "Sampling-mask capture requires a positive finite top_k."
                )
            if accept_index is None:
                target_probs = self.target_probs.index_select(0, batch_indices)
            else:
                # NGRAM tree slots can skip siblings. Each accepted index identifies
                # both the committed token and the target row that emitted it.
                rows = accept_index.index_select(0, batch_indices)
                target_probs = self.target_probs.flatten(0, 1)[rows]
            # A tree verifier can keep cutoff ties beyond the packing limit.
            # One extra value detects overflow without scanning the whole row.
            pack_width = min(
                max_top_k + int(accept_index is not None), target_probs.shape[-1]
            )
            support_probs, support_tokens = torch.topk(
                target_probs, k=pack_width, dim=-1
            )
            support_lens = (support_probs > 0).sum(dim=-1, dtype=torch.int32)
            support_probs = support_probs[..., :max_top_k]
            support_tokens = support_tokens[..., :max_top_k].to(torch.int32)
            selected_logprobs = torch.log(
                target_probs.gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)
            )
            if self.support_capture_indices is not None:
                support_logprobs = torch.log(
                    support_probs.index_select(0, self.support_capture_indices)
                )
            if greedy_mask is not None:
                greedy_rows = greedy_mask[:, None]
                support_tokens = torch.where(
                    greedy_rows[:, :, None],
                    output_tokens[:, :, None],
                    support_tokens,
                )
                support_lens = torch.where(
                    greedy_rows,
                    torch.ones_like(support_lens),
                    support_lens,
                )
                selected_logprobs = torch.where(
                    greedy_rows,
                    torch.zeros_like(selected_logprobs),
                    selected_logprobs,
                )
                if support_logprobs is not None:
                    support_logprobs = torch.where(
                        greedy_rows.index_select(0, self.support_capture_indices)[
                            :, :, None
                        ],
                        torch.zeros_like(support_logprobs),
                        support_logprobs,
                    )

        statuses = torch.where(
            torch.isfinite(selected_logprobs),
            int(SamplingMaskStatus.OK),
            int(SamplingMaskStatus.INVALID),
        ).to(torch.int8)
        if accept_index is not None:
            statuses.masked_fill_(
                (statuses == SamplingMaskStatus.OK) & (support_lens > self.max_top_k),
                SamplingMaskStatus.OVERFLOW,
            )
        return SamplingMaskOutput(
            token_ids=support_tokens,
            lengths=support_lens,
            selected_logprobs=selected_logprobs,
            support_logprobs=support_logprobs,
            statuses=statuses,
            num_accept_tokens=num_accept_tokens,
        )
