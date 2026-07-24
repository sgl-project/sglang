from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.speculative.topk1 import (
    TargetVerifyTopk1Output,
    target_verify_topk1_postprocess,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


_is_cuda = is_cuda()


def prepare_eagle_verify_logits(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    vocab_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.utils.async_probe import sanitize_nan_logits

    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits
    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    if vocab_mask is not None:
        assert verify_input.grammar is not None
        verify_input.grammar.apply_vocab_mask(
            logits=next_token_logits, vocab_mask=vocab_mask
        )

    return next_token_logits


def maybe_eagle_sample_target_verify_topk1(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    vocab_mask: Optional[torch.Tensor] = None,
) -> Optional[TargetVerifyTopk1Output]:
    """Run the CUDA topk=1 verify fast path when its semantics apply."""
    from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

    if (
        not _is_cuda
        or batch.forward_mode.is_idle()
        or not batch.sampling_info.is_all_greedy
        or verify_input.spec_input_type != SpecInputType.EAGLE_VERIFY
        or verify_input.tree_topk != 1
        or verify_input.draft_token_num != verify_input.max_tree_depth
        or SIMULATE_ACC_LEN > 0
    ):
        return None

    batch_size = len(batch.seq_lens)
    draft_tokens = verify_input.draft_token
    if (
        not draft_tokens.is_contiguous()
        or draft_tokens.numel() != batch_size * verify_input.draft_token_num
    ):
        return None
    candidates = draft_tokens.view(batch_size, verify_input.draft_token_num)
    logits = logits_output.next_token_logits
    if (
        logits.ndim != 2
        or logits.device.type != "cuda"
        or logits.stride(1) != 1
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or logits.shape[0] != candidates.numel()
        or logits.shape[1] == 0
    ):
        return None

    logits = prepare_eagle_verify_logits(verify_input, batch, logits_output, vocab_mask)
    return target_verify_topk1_postprocess(
        logits,
        candidates,
        verify_input.retrieve_index,
        verify_input.retrieve_next_token,
        batch.seq_lens,
    )


__all__ = [
    "maybe_eagle_sample_target_verify_topk1",
    "prepare_eagle_verify_logits",
]
