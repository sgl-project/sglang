from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton

try:
    from sglang.srt.dllm.kernels.low_confidence_utils import (
        calculate_low_confidence_score,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.disable_fused_triton_algorithm = config.algorithm_config.get(
            "disable_fused_triton_algorithm", False
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id
        attn_backend = get_global_server_args().attention_backend
        use_fused_kernel = (
            not self.disable_fused_triton_algorithm
            and _TRITON_AVAILABLE
            and support_triton(attn_backend)
        )

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

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

                if use_fused_kernel:
                    calculate_low_confidence_score(
                        curr_logits,
                        block_input_ids,
                        self.mask_id,
                        self.threshold,
                    )
                else:
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

            mask_count = (forward_batch.input_ids == self.mask_id).sum().item()
            if mask_count == 0:
                break

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
