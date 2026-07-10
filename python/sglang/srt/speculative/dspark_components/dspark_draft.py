from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens import (
    SampleStepTokens,
)


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None, out=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        if out is not None:
            assert out.shape == (int(max_bs) * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )

    def __call__(self, hidden_states, input_ids):
        draft_width = self.gamma + 1
        if hidden_states.shape[0] % draft_width != 0:
            raise RuntimeError(
                "DSpark folded draft sampler expects full blocks with "
                f"anchor + gamma tokens, got {hidden_states.shape[0]} rows "
                f"for gamma={self.gamma}."
            )
        bs = hidden_states.shape[0] // draft_width
        hidden_3d = hidden_states.view(bs, draft_width, -1)
        ids_2d = input_ids.view(bs, draft_width)
        anchor = ids_2d[:, 0]
        # Slot 0 conditions the block. Slots 1..gamma are the draft tokens
        # used by the verifier and confidence scheduler.
        draft_hidden = hidden_3d[:, 1:, :].contiguous()
        hidden_for_logits = draft_hidden.reshape(bs * self.gamma, -1)

        base_logits, confidence_tap = self.model.compute_base_logits(hidden_for_logits)
        base_logits = base_logits.view(bs, self.gamma, -1)
        draft_tokens, _ = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=draft_hidden,
            sampler=greedy_step_sampler,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=draft_hidden,
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
    draft_positions: torch.Tensor | None = None,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if fast_sampling:
                if sampling_info.sampling_seed is not None:
                    raise RuntimeError(
                        "SGLANG_DSPARK_FAST_SAMPLING does not support seeded "
                        "non-greedy sampling yet."
                    )
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                if sampling_info.sampling_seed is None:
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(
                        -1
                    )
                else:
                    if draft_positions is None:
                        raise RuntimeError(
                            "DSpark seeded non-greedy sampling needs draft positions."
                        )
                    logprobs = probs.to(torch.float64).log_()
                    sampled_tokens = multinomial_with_seed(
                        logprobs,
                        sampling_info.sampling_seed,
                        draft_positions[:, step_idx].to(torch.int64),
                    ).view(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )
