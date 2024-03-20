import torch
from sglang.srt.managers.router.model_runner import ForwardMode, InputMetadata
from torch import nn
from vllm.model_executor.parallel_utils.communication_op import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


class LogitsProcessor(nn.Module):
    # TODO: refractor all LogitsProcessor logic
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()

    def _get_normalized_logprobs(
        self, all_logprobs, input_ids, input_metadata: InputMetadata
    ):
        # Compute the logprobs and normalized logprobs for the prefill tokens.
        # Note that we pad a zero at the end of each sequence for easy computation.
        prefill_logprobs = all_logprobs[
            torch.arange(all_logprobs.shape[0], device="cuda"),
            torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
        ]
        logprobs_cumsum = torch.cumsum(prefill_logprobs, dim=0, dtype=torch.float32)

        start = input_metadata.extend_start_loc.clone()
        end = start + input_metadata.extend_seq_lens - 2
        start.clamp_(min=0, max=prefill_logprobs.shape[0] - 1)
        end.clamp_(min=0, max=prefill_logprobs.shape[0] - 1)
        sum_logp = (
            logprobs_cumsum[end] - logprobs_cumsum[start] + prefill_logprobs[start]
        )
        normalized_logprobs = sum_logp / (
            (input_metadata.extend_seq_lens - 1).clamp(min=1)
        )

        return prefill_logprobs, normalized_logprobs

    def forward(self, input_ids, hidden_states, weight, input_metadata: InputMetadata):
        last_index = None

        # Compute the last index (the first decode token) of each requeast
        if input_metadata.forward_mode == ForwardMode.VERIFY_WITH_DECODE:
            last_index = (
                torch.cumsum(input_metadata.b_qo_lens, dim=0, dtype=torch.long) - 1
            )
        elif input_metadata.forward_mode != ForwardMode.DECODE:
            last_index = (
                torch.cumsum(input_metadata.extend_seq_lens, dim=0, dtype=torch.long)
                - 1
            )

        return_all_logits = (
            input_metadata.tree_mask_flatten is not None
            and len(input_metadata.tree_mask_flatten) != 0
        )

        # Compute last hidden and last logits
        if input_metadata.forward_mode == ForwardMode.DECODE:
            last_hidden = hidden_states
        else:
            last_hidden = hidden_states[last_index]

        last_logits = torch.matmul(last_hidden, weight.T)
        if self.tp_size > 1:
            last_logits = tensor_model_parallel_all_gather(last_logits)
        last_logits = last_logits[:, : self.config.vocab_size]

        # When logprob is not requested, only compute the last logits.
        if not input_metadata.return_logprob and not return_all_logits:
            return last_logits, (None, None, None, None)
        else:
            # When logprob is requested, compute the logits for all tokens.
            if input_metadata.forward_mode == ForwardMode.DECODE:
                all_logits = last_logits
            else:
                all_logits = torch.matmul(hidden_states, weight.T)
                if self.tp_size > 1:
                    all_logits = tensor_model_parallel_all_gather(all_logits)
                all_logits = all_logits[:, : self.config.vocab_size]

            if not input_metadata.return_logprob:
                return last_logits, (all_logits.float(), None, None, None)

            all_logprobs = torch.log(torch.softmax(all_logits.float(), dim=-1) + 1e-6)

            # decode cases only return logits and last logprobs
            if input_metadata.forward_mode == ForwardMode.DECODE:
                return last_logits, (all_logits.float(), None, None, all_logprobs)
            elif input_metadata.forward_mode == ForwardMode.VERIFY_WITH_DECODE:
                # NOTE: not support return all_logprobs when VERIFY_WITH_DECODE
                return last_logits, (all_logits.float(), None, None, None)
            else:
                # Compute the logprobs for the last token of each request.
                last_logprobs = all_logprobs[last_index]

                prefill_logprobs, normalized_logprobs = self._get_normalized_logprobs(
                    all_logprobs, input_ids, input_metadata
                )

                return last_logits, (
                    all_logits.float(),
                    prefill_logprobs,
                    normalized_logprobs,
                    last_logprobs,
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


if __name__ == "__main__":
    test()
