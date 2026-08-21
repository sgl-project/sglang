import msgspec
import torch

from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus


class SpeculativeSamplingMaskCapture(msgspec.Struct):
    target_probs: torch.Tensor | None
    return_sampling_masks: list[bool]
    max_top_k: int
    greedy_mask: torch.Tensor | None = None

    def build_output(
        self,
        *,
        out_tokens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> SamplingMaskOutput:
        batch_indices = torch.tensor(
            [i for i, enabled in enumerate(self.return_sampling_masks) if enabled],
            device=out_tokens.device,
            dtype=torch.long,
        )
        output_tokens = out_tokens.index_select(0, batch_indices).clone()
        output_lens = commit_lens.index_select(0, batch_indices).clone()
        greedy_mask = (
            None
            if self.greedy_mask is None
            else self.greedy_mask.index_select(0, batch_indices).clone()
        )

        if self.target_probs is None:
            support_tokens = output_tokens.unsqueeze(-1).to(torch.int32)
            support_lens = torch.ones_like(output_tokens, dtype=torch.int32)
            selected_logprobs = torch.zeros_like(output_tokens, dtype=torch.float32)
        else:
            max_top_k = min(int(self.max_top_k), self.target_probs.shape[-1])
            if max_top_k <= 0:
                raise ValueError(
                    "Sampling-mask capture requires a positive finite top_k."
                )
            target_probs = self.target_probs.index_select(0, batch_indices)
            support_probs, support_tokens = torch.topk(
                target_probs, k=max_top_k, dim=-1
            )
            support_tokens = support_tokens.to(torch.int32)
            support_lens = (support_probs > 0).sum(dim=-1, dtype=torch.int32)
            selected_logprobs = torch.log(
                target_probs.gather(-1, output_tokens.unsqueeze(-1)).squeeze(-1)
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

        statuses = torch.where(
            torch.isfinite(selected_logprobs),
            int(SamplingMaskStatus.OK),
            int(SamplingMaskStatus.INVALID),
        ).to(torch.int8)
        return SamplingMaskOutput(
            token_ids=support_tokens,
            lengths=support_lens,
            selected_logprobs=selected_logprobs,
            statuses=statuses,
            output_lens=output_lens,
        )
