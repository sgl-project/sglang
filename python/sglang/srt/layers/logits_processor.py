import torch
from sglang.srt.managers.router.model_runner import ForwardMode, InputMetadata
from torch import nn
from vllm.model_executor.parallel_utils.communication_op import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


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

    def forward(self, input_ids, hidden_states, weight, input_metadata: InputMetadata):
        # Get last index for next token prediction, except for DECODE mode.
        last_index = None
        if input_metadata.forward_mode != ForwardMode.DECODE:
            last_index = (
                torch.cumsum(input_metadata.extend_seq_lens, dim=0, dtype=torch.long)
                - 1
            )

        # Get the last hidden states and last logits
        if input_metadata.forward_mode == ForwardMode.DECODE:
            last_hidden = hidden_states
        else:
            last_hidden = hidden_states[last_index]

        last_logits = torch.matmul(last_hidden, weight.T)
        if self.tp_size > 1:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, : self.config.vocab_size]

        # Return only last_logits if logprob is not requested
        if not input_metadata.return_logprob:
            hidden_states = None
            return last_logits, (None, None, None)
        else:
            # When logprob is requested, compute the logits for all tokens.
            if input_metadata.forward_mode == ForwardMode.DECODE:
                all_logits = last_logits
            else:
                all_logits = torch.matmul(hidden_states, weight.T)
                if self.tp_size > 1:
                    all_logits = tensor_model_parallel_all_gather(all_logits)
                all_logits = all_logits[:, : self.config.vocab_size]

            all_logprobs = torch.log(torch.softmax(all_logits.float(), dim=-1) + 1e-6)

            if input_metadata.forward_mode == ForwardMode.DECODE:
                last_logprobs = all_logprobs
                return last_logits, (None, None, last_logprobs)
            else:
                # Compute the logprobs for the last token of each request.
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
                return last_logits, (
                    prefill_token_logprobs,
                    normalized_prompt_logprobs,
                    last_logprobs,
                )


if __name__ == "__main__":
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
