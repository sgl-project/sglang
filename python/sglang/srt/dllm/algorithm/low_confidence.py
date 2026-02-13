from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.kernels.post_process import dllm_post_process_fused
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.use_triton_post_process = config.algorithm_config.get(
            "use_triton_post_process", True
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _ in range(self.block_size):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                if self.use_triton_post_process:
                    # Fused Triton kernel: softmax + argmax + threshold + fallback
                    # all on-device, no GPU-CPU sync per block
                    dllm_post_process_fused(
                        curr_logits,
                        block_input_ids,
                        self.mask_id,
                        self.threshold,
                    )
                else:
                    # Original PyTorch implementation
                    block_mask_index = block_input_ids == self.mask_id
                    if torch.sum(block_mask_index).item() == 0:
                        continue
                    x = torch.argmax(curr_logits, dim=-1)
                    p = torch.squeeze(
                        torch.gather(
                            F.softmax(curr_logits, dim=-1),
                            dim=-1,
                            index=torch.unsqueeze(x, -1),
                        ),
                        -1,
                    )
                    x = torch.where(block_mask_index, x, block_input_ids)
                    confidence = torch.where(block_mask_index, p, -np.inf)

                    transfer_index = confidence > self.threshold

                    if transfer_index.sum().item() == 0:
                        _, select_index = torch.topk(confidence, k=1)
                        transfer_index[select_index] = True

                    block_input_ids[transfer_index] = x[transfer_index]

            # Check once per iteration (single sync) whether any masks remain
            mask_count = (forward_batch.input_ids == self.mask_id).sum().item()
            if mask_count == 0:
                break

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
