from typing import Any, List, Optional

import msgspec
import torch


class SpeculativeSamplingMaskOutput(msgspec.Struct):
    support_tokens: Optional[torch.Tensor]
    support_lens: Optional[torch.Tensor]
    output_tokens: Optional[torch.Tensor]
    output_lens: torch.Tensor
    selected_logprobs: Optional[torch.Tensor]
    greedy_mask: Optional[torch.Tensor]
    return_sampling_masks: List[bool]

    def map_device_tensors(self, copy_fn: Any) -> None:
        if self.support_tokens is not None:
            self.support_tokens = copy_fn(self.support_tokens)
        if self.support_lens is not None:
            self.support_lens = copy_fn(self.support_lens)
        if self.output_tokens is not None:
            self.output_tokens = copy_fn(self.output_tokens)
        self.output_lens = copy_fn(self.output_lens)
        if self.selected_logprobs is not None:
            self.selected_logprobs = copy_fn(self.selected_logprobs)
        if self.greedy_mask is not None:
            self.greedy_mask = copy_fn(self.greedy_mask)

    def finalize(
        self,
    ) -> tuple[
        List[Optional[List[List[int]]]],
        List[Optional[List[float]]],
    ]:
        output_lens = self.output_lens.cpu().tolist()
        output_tokens = (
            None if self.output_tokens is None else self.output_tokens.cpu().tolist()
        )
        selected_logprobs = (
            None
            if self.selected_logprobs is None
            else self.selected_logprobs.cpu().tolist()
        )
        support_tokens = (
            None if self.support_tokens is None else self.support_tokens.cpu().tolist()
        )
        support_lens = (
            None if self.support_lens is None else self.support_lens.cpu().tolist()
        )
        greedy_mask = (
            None if self.greedy_mask is None else self.greedy_mask.cpu().tolist()
        )

        masks: List[Optional[List[List[int]]]] = [None] * len(output_lens)
        logprobs: List[Optional[List[float]]] = [None] * len(output_lens)
        for request_idx, should_return in enumerate(self.return_sampling_masks):
            if not should_return:
                continue
            output_len = output_lens[request_idx]
            masks[request_idx], logprobs[request_idx] = [], []
            for token_idx in range(output_len):
                if support_tokens is None or (
                    greedy_mask is not None and greedy_mask[request_idx]
                ):
                    support_ids = [int(output_tokens[request_idx][token_idx])]
                    selected_logprob = 0.0
                else:
                    support_len = support_lens[request_idx][token_idx]
                    support_ids = support_tokens[request_idx][token_idx][:support_len]
                    selected_logprob = float(selected_logprobs[request_idx][token_idx])
                masks[request_idx].append(support_ids)
                logprobs[request_idx].append(selected_logprob)
        return masks, logprobs


class SpeculativeSamplingMaskCapture(msgspec.Struct):
    target_probs: Optional[torch.Tensor]
    return_sampling_masks: List[bool]
    max_top_k: int
    greedy_mask: Optional[torch.Tensor] = None

    def build_output(
        self,
        *,
        out_tokens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> SpeculativeSamplingMaskOutput:
        if self.target_probs is None:
            support_tokens = None
            support_lens = None
            selected_logprobs = None
        else:
            max_top_k = min(int(self.max_top_k), self.target_probs.shape[-1])
            if max_top_k <= 0:
                raise ValueError(
                    "Sampling-mask capture requires a positive finite top_k."
                )
            support_probs, support_tokens = torch.topk(
                self.target_probs, k=max_top_k, dim=-1
            )
            support_tokens = support_tokens.to(torch.int32)
            support_lens = (support_probs > 0).sum(dim=-1, dtype=torch.int32)
            selected_logprobs = torch.log(
                self.target_probs.gather(-1, out_tokens.unsqueeze(-1)).squeeze(-1)
            )

        return SpeculativeSamplingMaskOutput(
            support_tokens=support_tokens,
            support_lens=support_lens,
            output_tokens=(
                out_tokens.clone()
                if self.target_probs is None or self.greedy_mask is not None
                else None
            ),
            output_lens=commit_lens.clone(),
            selected_logprobs=selected_logprobs,
            greedy_mask=(
                None if self.greedy_mask is None else self.greedy_mask.clone()
            ),
            return_sampling_masks=list(self.return_sampling_masks),
        )
