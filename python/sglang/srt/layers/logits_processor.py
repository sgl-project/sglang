"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Logits processing."""

import dataclasses
from typing import List, Optional, Union

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclasses.dataclass
class LogitsProcessorOutput:
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # The logprobs of the next tokens.     shape: [#seq, vocab_size]
    next_token_logprobs: torch.Tensor

    # The normlaized logprobs of prompts.  shape: [#seq]
    normalized_prompt_logprobs: torch.Tensor
    # The logprobs of input tokens.        shape: [#token, vocab_size]
    input_token_logprobs: torch.Tensor

    # The logprob and id of the top-k tokens in input positions.  shape [#seq, #token, k] of Tuple(logprob, token_id)
    input_top_logprobs: List
    # The logprob and id of the top-k tokens in output positions. shape [#seq, #token, k] of Tuple(logprob, token_id)
    output_top_logprobs: List


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    top_logprobs_nums: Optional[List[int]]

    return_logprob: bool = False
    return_top_logprob: bool = False

    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None

    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if forward_batch.return_logprob:
            return_top_logprob = any(x > 0 for x in forward_batch.top_logprobs_nums)
        else:
            return_top_logprob = False

        if forward_batch.forward_mode.is_extend():
            extend_logprob_pruned_lens_cpu = [
                extend_len - start_len
                for extend_len, start_len in zip(
                    forward_batch.extend_seq_lens,
                    forward_batch.extend_logprob_start_lens_cpu,
                )
            ]
        else:
            extend_logprob_pruned_lens_cpu = None
        return cls(
            forward_mode=forward_batch.forward_mode,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            return_logprob=forward_batch.return_logprob,
            return_top_logprob=return_top_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
        )


class LogitsProcessor(nn.Module):
    def __init__(self, config, skip_all_gather: bool = False):
        super().__init__()
        self.config = config
        self.do_tensor_parallel_all_gather = (
            not skip_all_gather and get_tensor_model_parallel_world_size() > 1
        )

    def _get_normalized_prompt_logprobs(
        self,
        input_token_logprobs: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ):
        logprobs_cumsum = torch.cumsum(input_token_logprobs, dim=0, dtype=torch.float32)
        pruned_lens = torch.tensor(
            logits_metadata.extend_logprob_pruned_lens_cpu, device="cuda"
        )

        start = torch.zeros_like(pruned_lens)
        start[1:] = torch.cumsum(pruned_lens[:-1], dim=0)
        end = torch.clamp(
            start + pruned_lens - 2, min=0, max=logprobs_cumsum.shape[0] - 1
        )
        sum_logp = (
            logprobs_cumsum[end] - logprobs_cumsum[start] + input_token_logprobs[start]
        )
        normalized_prompt_logprobs = sum_logp / (pruned_lens - 1).clamp(min=1)
        return normalized_prompt_logprobs

    @staticmethod
    def get_top_logprobs(all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()

        if logits_metadata.forward_mode.is_decode():
            output_top_logprobs = []
            for i, k in enumerate(logits_metadata.top_logprobs_nums):
                output_top_logprobs.append(list(zip(values[i][:k], indices[i][:k])))
            return None, output_top_logprobs
        else:
            input_top_logprobs, output_top_logprobs = [], []

            pt = 0
            for k, pruned_len in zip(
                logits_metadata.top_logprobs_nums,
                logits_metadata.extend_logprob_pruned_lens_cpu,
            ):
                if pruned_len <= 0:
                    input_top_logprobs.append([])
                    output_top_logprobs.append([])
                    continue

                input_top_logprobs.append(
                    [
                        list(zip(values[pt + j][:k], indices[pt + j][:k]))
                        for j in range(pruned_len - 1)
                    ]
                )
                output_top_logprobs.append(
                    list(
                        zip(
                            values[pt + pruned_len - 1][:k],
                            indices[pt + pruned_len - 1][:k],
                        )
                    )
                )
                pt += pruned_len

            return input_top_logprobs, output_top_logprobs

    def forward(
        self,
        input_ids,
        hidden_states,
        weight,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
    ):
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)
        assert isinstance(logits_metadata, LogitsMetadata)

        # Get the last hidden states and last logits for the next token prediction
        if logits_metadata.forward_mode.is_decode():
            last_index = None
            last_hidden = hidden_states
        else:
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            last_hidden = hidden_states[last_index]

        last_logits = torch.matmul(last_hidden, weight.T)
        if self.do_tensor_parallel_all_gather:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, : self.config.vocab_size].float()

        if hasattr(self.config, "final_logit_softcapping"):
            last_logits.div_(self.config.final_logit_softcapping)
            torch.tanh(last_logits, out=last_logits)
            last_logits.mul_(self.config.final_logit_softcapping)

        # Return only last_logits if logprob is not requested
        if not logits_metadata.return_logprob:
            return LogitsProcessorOutput(
                next_token_logits=last_logits,
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )
        else:
            last_logprobs = torch.nn.functional.log_softmax(last_logits, dim=-1)

            if logits_metadata.forward_mode.is_decode():
                if logits_metadata.return_top_logprob:
                    output_top_logprobs = self.get_top_logprobs(
                        last_logprobs, logits_metadata
                    )[1]
                else:
                    output_top_logprobs = None
                return LogitsProcessorOutput(
                    next_token_logits=last_logits,
                    next_token_logprobs=last_logprobs,
                    normalized_prompt_logprobs=None,
                    input_token_logprobs=None,
                    input_top_logprobs=None,
                    output_top_logprobs=output_top_logprobs,
                )
            else:
                # Slice the requested tokens to compute logprob
                pt, states, pruned_input_ids = 0, [], []
                for start_len, extend_len in zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                ):
                    states.append(hidden_states[pt + start_len : pt + extend_len])
                    pruned_input_ids.append(input_ids[pt + start_len : pt + extend_len])
                    pt += extend_len

                # Compute the logits and logprobs for all required tokens
                states = torch.cat(states, dim=0)
                all_logits = torch.matmul(states, weight.T)
                if self.do_tensor_parallel_all_gather:
                    all_logits = tensor_model_parallel_all_gather(all_logits)
                all_logits = all_logits[:, : self.config.vocab_size].float()

                if hasattr(self.config, "final_logit_softcapping"):
                    all_logits.div_(self.config.final_logit_softcapping)
                    torch.tanh(all_logits, out=all_logits)
                    all_logits.mul_(self.config.final_logit_softcapping)

                all_logprobs = all_logits
                del all_logits, hidden_states
                all_logprobs[:] = torch.nn.functional.log_softmax(all_logprobs, dim=-1)

                # Get the logprob of top-k tokens
                if logits_metadata.return_top_logprob:
                    input_top_logprobs, output_top_logprobs = self.get_top_logprobs(
                        all_logprobs, logits_metadata
                    )
                else:
                    input_top_logprobs = output_top_logprobs = None

                # Compute the normalized logprobs for the requested tokens.
                # Note that we pad a zero at the end for easy batching.
                input_token_logprobs = all_logprobs[
                    torch.arange(all_logprobs.shape[0], device="cuda"),
                    torch.cat(
                        [
                            torch.cat(pruned_input_ids)[1:],
                            torch.tensor([0], device="cuda"),
                        ]
                    ),
                ]
                normalized_prompt_logprobs = self._get_normalized_prompt_logprobs(
                    input_token_logprobs,
                    logits_metadata,
                )

                return LogitsProcessorOutput(
                    next_token_logits=last_logits,
                    next_token_logprobs=last_logprobs,
                    normalized_prompt_logprobs=normalized_prompt_logprobs,
                    input_token_logprobs=input_token_logprobs,
                    input_top_logprobs=input_top_logprobs,
                    output_top_logprobs=output_top_logprobs,
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
