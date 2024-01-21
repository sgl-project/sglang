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

    def forward(self, input_ids, hidden_states, weight, input_metadata):
        if not input_metadata.return_normalized_logprob:
            if input_metadata.forward_mode == ForwardMode.DECODE:
                last_hidden = hidden_states
            else:
                last_index = (
                    torch.cumsum(
                        input_metadata.seq_lens - input_metadata.prefix_lens,
                        dim=0,
                        dtype=torch.long,
                    )
                    - 1
                )
                last_hidden = hidden_states[last_index]
                hidden_states = None

            last_logits = torch.matmul(last_hidden, weight.T)
            if self.tp_size > 1:
                last_logits = tensor_model_parallel_all_gather(last_logits)
            last_logits = last_logits[:, : self.config.vocab_size]
            return last_logits, None
        else:
            assert input_metadata.forward_mode != ForwardMode.DECODE
            last_index = (
                torch.cumsum(
                    input_metadata.seq_lens - input_metadata.prefix_lens,
                    dim=0,
                    dtype=torch.long,
                )
                - 1
            )

            logits = torch.matmul(hidden_states, weight.T)
            if self.tp_size > 1:
                logits = tensor_model_parallel_all_gather(logits)
            logits = logits[:, : self.config.vocab_size]
            all_logprobs = torch.log(torch.softmax(logits.float(), dim=-1) + 1e-6)

            normalized_logprobs = compute_normalized_logprobs(
                all_logprobs,
                input_ids,
                input_metadata.extend_seq_lens,
                input_metadata.extend_start_loc,
            )

            last_logits = logits[last_index]
            return last_logits, normalized_logprobs


def compute_normalized_logprobs(all_logprobs, input_ids, seq_lens, start_loc):
    logprobs = all_logprobs[
        torch.arange(all_logprobs.shape[0], device="cuda"),
        torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
    ]
    logprobs_cumsum = torch.cumsum(logprobs, dim=0, dtype=torch.float32)

    start = start_loc.clone()
    end = start + seq_lens - 2
    start.clamp_(min=0, max=logprobs.shape[0] - 1)
    end.clamp_(min=0, max=logprobs.shape[0] - 1)
    sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + logprobs[start]
    return sum_logp / ((seq_lens - 1).clamp(min=1))


if __name__ == "__main__":
    all_logprobs = torch.tensor(
        #       s                     s                s
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        dtype=torch.float32,
        device="cuda",
    )
    seq_lens = torch.tensor([2, 0, 3, 0], dtype=torch.int32, device="cuda")
    input_ids = torch.tensor([1, 2, 3, 0, 1], dtype=torch.int32, device="cuda")
    logprobs = torch.zeros(5, dtype=torch.float32, device="cuda")

    logprobs = all_logprobs[
        torch.arange(all_logprobs.shape[0], device="cuda"),
        torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
    ]
    logprobs_cumsum = torch.cumsum(logprobs, dim=0, dtype=torch.float32)

    len_cumsum = torch.cumsum(seq_lens, dim=0)
    start = torch.cat((torch.tensor([0], device="cuda"), len_cumsum[:-1]), 0)
    end = start + seq_lens - 2
    start.clamp_(min=0, max=logprobs.shape[0] - 1)
    end.clamp_(min=0, max=logprobs.shape[0] - 1)
    sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + logprobs[start]

    # assert logprobs == [2, _, 2, 4, _]
    print("logprobs", logprobs)
    print("start", start)
    print("end", end)
    print("sum_logp", sum_logp)
