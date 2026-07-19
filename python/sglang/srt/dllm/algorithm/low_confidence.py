from typing import Any, List

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.sampling import sample_block_tokens
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LowConfidence(DllmAlgorithm):
    """Each step unmasks positions whose predicted-token confidence exceeds a
    threshold (falling back to the highest-confidence masked position).
    """

    supports_step_maps = True

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask_index = input_ids == self.mask_id
        done = block_mask_index.sum(dim=1) == 0

        # A mask token is a latent state rather than a valid sampled action.
        # Excluding it guarantees that each recorded transfer is a real
        # mask-to-token transition.
        logits[:, :, self.mask_id] = -float("inf")
        sampled_ids = []
        sampled_probs = []
        for batch_id in range(batch_size):
            block_start = batch_id * self.block_size
            block_end = block_start + self.block_size
            positions = (
                forward_batch.positions[block_start:block_end]
                if forward_batch.positions is not None
                else None
            )
            token_ids, token_probs = sample_block_tokens(
                logits[batch_id],
                forward_batch.sampling_info,
                batch_id,
                positions,
            )
            sampled_ids.append(token_ids)
            sampled_probs.append(token_probs)

        x = torch.stack(sampled_ids).to(dtype=input_ids.dtype)
        confidence = torch.stack(sampled_probs)
        confidence = torch.where(block_mask_index, confidence, -float("inf"))

        transfer_index = confidence > self.threshold
        has_transfer = transfer_index.sum(dim=1) > 0
        top1_indices = torch.argmax(confidence, dim=1)
        batch_indices = torch.arange(batch_size, device=top1_indices.device)
        top1_mask = torch.zeros_like(transfer_index, dtype=torch.bool)
        top1_mask[batch_indices, top1_indices] = True
        transfer_index = torch.where(
            has_transfer.unsqueeze(-1), transfer_index, top1_mask
        )

        x = torch.where(block_mask_index, x, input_ids)
        new_input_ids = torch.where(transfer_index, x, input_ids)
        # In-place to preserve the input_ids tensor identity (CUDA graph safe).
        forward_batch.input_ids.copy_(new_input_ids.view(-1))

        return done.tolist()


Algorithm = LowConfidence
