"""Logits processing."""

import dataclasses
from typing import List

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

from sglang.srt.managers.controller.model_runner import ForwardMode, InputMetadata


@dataclasses.dataclass
class LogitProcessorOutput:
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # The logprobs of the next tokens.     shape: [#seq, vocab_size]
    next_token_logprobs: torch.Tensor

    # The normlaized logprobs of prompts.  shape: [#seq]
    normalized_prompt_logprobs: torch.Tensor
    # The logprobs of prefill tokens.      shape: [#token, vocab_size]
    prefill_token_logprobs: torch.Tensor

    # The logprob and id of the top-k tokens in prefill positions.  shape [#seq, #token, k] of Tuple(logprob, token_id)
    prefill_top_logprobs: List
    # The logprob and id of the top-k tokens in decode positions.   shape [#seq, #token, k] of Tuple(logprob, token_id)
    decode_top_logprobs: List


class LogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()

    def _get_normalized_prompt_logprobs(
        self, prefill_token_logprobs, input_metadata: InputMetadata
    ):
        logprobs_cumsum = torch.cumsum(
            prefill_token_logprobs, dim=0, dtype=torch.float32
        )

        start = input_metadata.extend_start_loc.clone()
        end = start + input_metadata.extend_seq_lens - 2
        start.clamp_(min=0, max=prefill_token_logprobs.shape[0] - 1)
        end.clamp_(min=0, max=prefill_token_logprobs.shape[0] - 1)
        sum_logp = (
            logprobs_cumsum[end]
            - logprobs_cumsum[start]
            + prefill_token_logprobs[start]
        )
        normalized_prompt_logprobs = sum_logp / (
            (input_metadata.extend_seq_lens - 1).clamp(min=1)
        )

        return normalized_prompt_logprobs

    def _get_top_logprobs(self, all_logprobs, input_metadata: InputMetadata):
        # TODO: vectorize the code below
        if input_metadata.forward_mode == ForwardMode.DECODE:
            decode_top_logprobs = []
            for i in range(all_logprobs.shape[0]):
                k = input_metadata.top_logprobs_nums[i]
                t = all_logprobs[i].topk(k)
                v_cpu = t.values.tolist()
                p_cpu = t.indices.tolist()
                decode_top_logprobs.append(list(zip(v_cpu, p_cpu)))
            return None, decode_top_logprobs
        else:
            prefill_top_logprobs, decode_top_logprobs = [], []
            pt = 0
            extend_seq_lens_cpu = input_metadata.extend_seq_lens.tolist()
            for i, extend_seq_len in enumerate(extend_seq_lens_cpu):
                if extend_seq_len == 0:
                    prefill_top_logprobs.append([])
                    decode_top_logprobs.append([])
                    continue
                k = input_metadata.top_logprobs_nums[i]
                t = all_logprobs[pt : pt + extend_seq_len].topk(k)
                vs_cpu = t.values.tolist()
                ps_cpu = t.indices.tolist()
                prefill_top_logprobs.append(
                    [list(zip(vs_cpu[j], ps_cpu[j])) for j in range(len(vs_cpu) - 1)]
                )
                decode_top_logprobs.append(list(zip(vs_cpu[-1], ps_cpu[-1])))
                pt += extend_seq_len

            return prefill_top_logprobs, decode_top_logprobs

    def forward(self, input_ids, hidden_states, weight, input_metadata: InputMetadata):
        # Get the last hidden states and last logits for the next token prediction
        if input_metadata.forward_mode == ForwardMode.DECODE:
            last_index = None
            last_hidden = hidden_states
        else:
            last_index = (
                torch.cumsum(input_metadata.extend_seq_lens, dim=0, dtype=torch.long)
                - 1
            )
            last_hidden = hidden_states[last_index]

        last_logits = torch.matmul(last_hidden, weight.T)
        if self.tp_size > 1:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, : self.config.vocab_size]

        # Return only last_logits if logprob is not requested
        if not input_metadata.return_logprob:
            return LogitProcessorOutput(
                next_token_logits=last_logits,
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                prefill_token_logprobs=None,
                prefill_top_logprobs=None,
                decode_top_logprobs=None,
            )
        else:
            # When logprob is requested, compute the logits for all tokens.
            if input_metadata.forward_mode == ForwardMode.DECODE:
                all_logits = last_logits
            else:
                all_logits = torch.matmul(hidden_states, weight.T)
                if self.tp_size > 1:
                    all_logits = tensor_model_parallel_all_gather(all_logits)
                all_logits = all_logits[:, : self.config.vocab_size]

            all_logprobs = all_logits.float()
            del all_logits
            all_logprobs[:] = torch.nn.functional.log_softmax(all_logprobs, dim=-1)

            # Get the logprob of top-k tokens
            return_top_logprob = any(x > 0 for x in input_metadata.top_logprobs_nums)
            if return_top_logprob:
                prefill_top_logprobs, decode_top_logprobs = self._get_top_logprobs(
                    all_logprobs, input_metadata
                )
            else:
                prefill_top_logprobs = decode_top_logprobs = None

            if input_metadata.forward_mode == ForwardMode.DECODE:
                return LogitProcessorOutput(
                    next_token_logits=last_logits,
                    next_token_logprobs=all_logprobs,
                    normalized_prompt_logprobs=None,
                    prefill_token_logprobs=None,
                    prefill_top_logprobs=None,
                    decode_top_logprobs=decode_top_logprobs,
                )
            else:
                last_logprobs = all_logprobs[last_index]

                # Compute the logprobs and normalized logprobs for the prefill tokens.
                # Note that we pad a zero at the end of each sequence for easy computation.
                prefill_token_logprobs = all_logprobs[
                    torch.arange(all_logprobs.shape[0], device="cuda"),
                    torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
                ]

                normalized_prompt_logprobs = self._get_normalized_prompt_logprobs(
                    prefill_token_logprobs, input_metadata
                )

                return LogitProcessorOutput(
                    next_token_logits=last_logits,
                    next_token_logprobs=last_logprobs,
                    normalized_prompt_logprobs=normalized_prompt_logprobs,
                    prefill_token_logprobs=prefill_token_logprobs,
                    prefill_top_logprobs=prefill_top_logprobs,
                    decode_top_logprobs=decode_top_logprobs,
                )


def test():
    all_logprobs = torch.tensor(
        #       s                     s                s
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        dtype=torch.float32,
        device="cuda",
    )
    seq_lens = torch.tensor([2, 0, 3, 0], dtype=torch.int32, device="cuda")
    input_ids = torch.tensor([1, 2, 3, 0, 1], dtype=torch.int32, device="cuda")

    token_logprobs = all_logprobs[
        torch.arange(all_logprobs.shape[0], device="cuda"),
        torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
    ]
    logprobs_cumsum = torch.cumsum(token_logprobs, dim=0, dtype=torch.float32)

    len_cumsum = torch.cumsum(seq_lens, dim=0)
    start = torch.cat((torch.tensor([0], device="cuda"), len_cumsum[:-1]), 0)
    end = start + seq_lens - 2
    start.clamp_(min=0, max=token_logprobs.shape[0] - 1)
    end.clamp_(min=0, max=token_logprobs.shape[0] - 1)
    sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + token_logprobs[start]

    # assert logprobs == [2, _, 2, 4, _]
    print("token logprobs", token_logprobs)
    print("start", start)
    print("end", end)
    print("sum_logp", sum_logp)


if __name__ == "__main__":
    test()
