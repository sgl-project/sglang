from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

_DIFFUSION_EPS = 1e-6


# Adapted from https://github.com/ML-GSAI/LLaDA/blob/d5634dace08aee47cc5cd68e67059994d5dab8c9/generate.py
class LowConfidence(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def add_gumbel_noise(
        self, logits: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a variant of Gumbel noise (or Gumbel max trick) for sampling. The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.

        Args:
            logits (torch.Tensor): Logits tensor with shape [batch_size, block_size, vocab_size].
            temperature (torch.Tensor): Temperature tensor with shape [batch_size, 1, 1],
                                        which will be broadcast across block_size and vocab_size.

        Returns:
            torch.Tensor: Logits modified by the temperature-scaled Gumbel noise.
        """

        # FIXME: handle the case when the temperature of a certain req is 0
        if torch.all(temperature < _DIFFUSION_EPS):
            return logits

        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)

        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

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
            # Check if all mask tokens have been filled across all blocks.
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            # full_logits shape: [batch_size * block_size, vocab_size] -> [batch_size, block_size, vocab_size]
            full_logits_reshaped = logits_output.full_logits.view(
                forward_batch.batch_size, self.block_size, -1
            )

            # input_ids shape: [batch_size * block_size] -> [batch_size, block_size]
            input_ids_reshaped = forward_batch.input_ids.view(
                forward_batch.batch_size, self.block_size
            )

            # mask_index shape: [batch_size, block_size]
            mask_index_reshaped = input_ids_reshaped == self.mask_id

            # Batched Gumbel noise
            # temperatures shape [batch_size] -> [batch_size, 1, 1] for broadcasting
            temperatures_expanded = forward_batch.sampling_info.temperatures.view(
                -1, 1, 1
            )

            logits_with_noise = self.add_gumbel_noise(
                full_logits_reshaped, temperatures_expanded
            )
            x = torch.argmax(logits_with_noise, dim=-1)

            softmax_logits = F.softmax(full_logits_reshaped, dim=-1)

            # Gather the probability P for the sampled token x (confidence before thresholding).
            # Expand x for gather index: [batch_size, block_size, 1]
            x_expanded = torch.unsqueeze(x, -1)
            # p shape: [batch_size, block_size]
            p = torch.squeeze(
                torch.gather(
                    softmax_logits,
                    dim=-1,
                    index=x_expanded,
                ),
                -1,
            )

            x = torch.where(mask_index_reshaped, x, input_ids_reshaped)
            confidence = torch.where(mask_index_reshaped, p, -np.inf)

            thresholds_expanded = forward_batch.sampling_info.thresholds.view(-1, 1)
            transfer_index = confidence > thresholds_expanded

            _, select_index = torch.topk(confidence, k=1, dim=-1)

            needs_topk_fallback = transfer_index.sum(dim=-1) == 0
            topk_mask = torch.zeros_like(transfer_index, dtype=torch.bool)

            # Use scatter to set the Top-1 index to True, but only for blocks needing fallback.
            # The source (src) is an expanded boolean mask indicating which batches need fallback.
            topk_mask.scatter_(
                -1,
                select_index,
                needs_topk_fallback.unsqueeze(-1).expand_as(select_index),
            )

            transfer_index = transfer_index | topk_mask
            input_ids_reshaped[transfer_index] = x[transfer_index]
            forward_batch.input_ids = input_ids_reshaped.view(-1)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        new_next_token_ids = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, new_next_token_ids, can_run_cuda_graph


Algorithm = LowConfidence
