# Copyright 2023-2024 SGLang Team
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
"""Logits processing."""

import dataclasses
from typing import List, Optional, Union

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)


@dataclasses.dataclass
class LogitsProcessorOutput:
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # The logprobs of the next tokens.     shape: [#seq, vocab_size]
    next_token_logprobs: torch.Tensor = None

    # The normlaized logprobs of prompts.  shape: [#seq]
    normalized_prompt_logprobs: torch.Tensor = None
    # The logprobs of input tokens.        shape: [#token, vocab_size]
    input_token_logprobs: torch.Tensor = None

    # The logprob and id of the top-k tokens in input positions.  shape [#seq, #token, k]
    input_top_logprobs_val: List = None
    input_top_logprobs_idx: List = None
    # The logprob and id of the top-k tokens in output positions. shape [#seq, #token, k]
    output_top_logprobs_val: List = None
    output_top_logprobs_idx: List = None

    # Used by speculative decoding (EAGLE)
    # The output of transformer layers
    hidden_states: Optional[torch.Tensor] = None


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

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        extend_logprob_pruned_lens_cpu = None

        if forward_batch.return_logprob:
            return_top_logprob = any(x > 0 for x in forward_batch.top_logprobs_nums)
            if forward_batch.forward_mode.is_extend():
                extend_logprob_pruned_lens_cpu = [
                    extend_len - start_len
                    for extend_len, start_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.extend_logprob_start_lens_cpu,
                    )
                ]
        else:
            return_top_logprob = False

        if forward_batch.spec_info:
            capture_hidden_mode = forward_batch.spec_info.capture_hidden_mode
        else:
            capture_hidden_mode = CaptureHiddenMode.NULL

        return cls(
            forward_mode=forward_batch.forward_mode,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            return_logprob=forward_batch.return_logprob,
            return_top_logprob=return_top_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            capture_hidden_mode=capture_hidden_mode,
        )


class LogitsProcessor(nn.Module):
    def __init__(
        self, config, skip_all_gather: bool = False, logit_scale: Optional[float] = None
    ):
        super().__init__()
        self.config = config
        self.logit_scale = logit_scale
        self.do_tensor_parallel_all_gather = (
            not skip_all_gather and get_tensor_model_parallel_world_size() > 1
        )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
    ):
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)
        assert isinstance(logits_metadata, LogitsMetadata)

        # Get the last hidden states and last logits for the next token prediction
        if (
            logits_metadata.forward_mode.is_decode()
            or logits_metadata.forward_mode.is_target_verify()
        ):
            last_index = None
            last_hidden = hidden_states
        else:
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            last_hidden = hidden_states[last_index]

        last_logits = self._get_logits(last_hidden, lm_head)
        if self.do_tensor_parallel_all_gather:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, : self.config.vocab_size].float()

        if self.final_logit_softcapping:
            last_logits.div_(self.final_logit_softcapping)
            torch.tanh(last_logits, out=last_logits)
            last_logits.mul_(self.final_logit_softcapping)

        # Return only last_logits if logprob is not requested
        if not logits_metadata.return_logprob:
            return LogitsProcessorOutput(
                next_token_logits=last_logits,
                hidden_states=(
                    hidden_states
                    if logits_metadata.capture_hidden_mode.is_full()
                    else (
                        last_hidden
                        if logits_metadata.capture_hidden_mode.is_last()
                        else None
                    )
                ),
            )
        else:
            last_logprobs = self.compute_temp_top_p_normalized_logprobs(
                last_logits, logits_metadata
            )

            if logits_metadata.forward_mode.is_decode():
                if logits_metadata.return_top_logprob:
                    output_top_logprobs_val, output_top_logprobs_idx = (
                        self.get_top_logprobs(last_logprobs, logits_metadata)[2:4]
                    )
                else:
                    output_top_logprobs_val = output_top_logprobs_idx = None
                return LogitsProcessorOutput(
                    next_token_logits=last_logits,
                    next_token_logprobs=last_logprobs,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
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
                all_logits = self._get_logits(states, lm_head)
                if self.do_tensor_parallel_all_gather:
                    all_logits = tensor_model_parallel_all_gather(all_logits)

                # The LM head's weights may be zero-padded for parallelism. Remove any
                # extra logits that this padding may have produced.
                all_logits = all_logits[:, : self.config.vocab_size].float()

                if self.final_logit_softcapping:
                    all_logits.div_(self.final_logit_softcapping)
                    torch.tanh(all_logits, out=all_logits)
                    all_logits.mul_(self.final_logit_softcapping)

                all_logprobs = all_logits
                del all_logits, hidden_states

                all_logprobs = self.compute_temp_top_p_normalized_logprobs(
                    all_logprobs, logits_metadata
                )

                # Get the logprob of top-k tokens
                if logits_metadata.return_top_logprob:
                    (
                        input_top_logprobs_val,
                        input_top_logprobs_idx,
                        output_top_logprobs_val,
                        output_top_logprobs_idx,
                    ) = self.get_top_logprobs(all_logprobs, logits_metadata)
                else:
                    input_top_logprobs_val = input_top_logprobs_idx = (
                        output_top_logprobs_val
                    ) = output_top_logprobs_idx = None

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
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(lm_head, "weight"):
            logits = torch.matmul(hidden_states, lm_head.weight.T)
        else:
            # GGUF models
            logits = lm_head.linear_method.apply(lm_head, hidden_states, embedding_bias)

        # Optional scaling factor
        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)  # In-place multiply
        return logits

    @staticmethod
    def _get_normalized_prompt_logprobs(
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
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            for i, k in enumerate(logits_metadata.top_logprobs_nums):
                output_top_logprobs_val.append(values[i][:k])
                output_top_logprobs_idx.append(indices[i][:k])
            return None, None, output_top_logprobs_val, output_top_logprobs_idx
        else:
            input_top_logprobs_val, input_top_logprobs_idx = [], []
            output_top_logprobs_val, output_top_logprobs_idx = [], []

            pt = 0
            for k, pruned_len in zip(
                logits_metadata.top_logprobs_nums,
                logits_metadata.extend_logprob_pruned_lens_cpu,
            ):
                if pruned_len <= 0:
                    input_top_logprobs_val.append([])
                    input_top_logprobs_idx.append([])
                    output_top_logprobs_val.append([])
                    output_top_logprobs_idx.append([])
                    continue

                input_top_logprobs_val.append(
                    [values[pt + j][:k] for j in range(pruned_len - 1)]
                )
                input_top_logprobs_idx.append(
                    [indices[pt + j][:k] for j in range(pruned_len - 1)]
                )
                output_top_logprobs_val.append(
                    list(
                        values[pt + pruned_len - 1][:k],
                    )
                )
                output_top_logprobs_idx.append(
                    list(
                        indices[pt + pruned_len - 1][:k],
                    )
                )
                pt += pruned_len

            return (
                input_top_logprobs_val,
                input_top_logprobs_idx,
                output_top_logprobs_val,
                output_top_logprobs_idx,
            )

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


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
