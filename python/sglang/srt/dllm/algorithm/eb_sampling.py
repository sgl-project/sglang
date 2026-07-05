from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class EBSampling(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.gamma = config.algorithm_config.get("gamma", 0.0)
        if self.gamma < 0:
            raise ValueError("EBSampling requires a non-negative gamma threshold.")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _ in range(1, self.block_size + 1):
            mask_index = forward_batch.input_ids == self.mask_id
            if not mask_index.any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if not block_mask_index.any():
                    continue

                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]
                x = torch.argmax(curr_logits, dim=-1)

                masked_positions = torch.nonzero(
                    block_mask_index, as_tuple=False
                ).flatten()
                masked_logits = curr_logits[masked_positions]
                masked_log_probs = F.log_softmax(masked_logits, dim=-1)
                masked_probs = masked_log_probs.exp()
                masked_entropies = -(masked_probs * masked_log_probs).sum(dim=-1)

                sort_index = torch.argsort(masked_entropies, dim=0)
                sorted_positions = masked_positions[sort_index]
                sorted_entropies = masked_entropies[sort_index]

                reveal_count = 1
                entropy_budget = 0.0
                for entropy in sorted_entropies[:-1]:
                    entropy_budget += float(entropy.item())
                    if entropy_budget <= self.gamma:
                        reveal_count += 1
                    else:
                        break

                transfer_positions = sorted_positions[:reveal_count]
                transfer_index = torch.zeros_like(block_mask_index)
                transfer_index[transfer_positions] = True

                block_input_ids[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = EBSampling
