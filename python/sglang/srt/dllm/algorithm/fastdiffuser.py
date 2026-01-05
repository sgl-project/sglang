"""
FastDiffuser algorithm for SGLang - LLADA2-style.

Simplified to match LLADA2's architecture: process one block at a time,
let SGLang's scheduler handle multi-block generation.
"""
from typing import Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Add Gumbel noise to logits for stochastic sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class FastDiffuser(DllmAlgorithm):
    """
    FastDiffuser algorithm - LLADA2-style single block processing.

    Processes one block of masks at a time using iterative denoising with
    threshold-based token selection.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.temperature = config.algorithm_config.get("temperature", 0.7)
        self.threshold = config.algorithm_config.get("threshold", 0.9)

        logger.info(f"FastDiffuser initialized: block_size={self.block_size}, "
                   f"max_steps={self.max_steps}, temperature={self.temperature}, "
                   f"threshold={self.threshold}")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """
        Run FastDiffuser - LLADA2-style single block processing.

        This matches LLADA2's low_confidence algorithm structure:
        - Find where masks start
        - Iteratively denoise using threshold-based selection
        - Let SGLang scheduler handle multi-block calls
        """
        # Find mask start position (like LLADA2)
        mask_index = forward_batch.input_ids == self.mask_id
        start = len(forward_batch.input_ids) - torch.sum(mask_index).item()

        # Iterative denoising loop (like LLADA2)
        total_iterations = 0
        for iteration in range(self.max_steps):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # Forward pass
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            total_iterations += 1

            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            # Get predictions with Gumbel noise
            logits_with_noise = add_gumbel_noise(logits_output.full_logits, self.temperature)
            x = torch.argmax(logits_with_noise, dim=-1)

            # Calculate confidence scores
            p = F.softmax(logits_output.full_logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1
            )

            # Only consider masked positions
            x = torch.where(mask_index, x, forward_batch.input_ids)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Transfer tokens based on threshold
            if self.threshold is None or self.threshold == "None":
                # When threshold is None, denoise exactly 1 token per step
                _, select_index = torch.topk(confidence, k=1)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index[select_index] = True
            else:
                # Original threshold-based logic
                transfer_index = (confidence > self.threshold) | (iteration == self.max_steps - 1)
                if transfer_index.sum().item() == 0:
                    # If no tokens above threshold, take the best one
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

            # Update input_ids
            forward_batch.input_ids[transfer_index] = x[transfer_index]
        logger.info("FastDiffuser finished with total block iterations: %d", total_iterations)
        # Final forward pass (like LLADA2)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        # Return generated tokens (like LLADA2)
        next_token_ids = forward_batch.input_ids[start:]
        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = FastDiffuser
