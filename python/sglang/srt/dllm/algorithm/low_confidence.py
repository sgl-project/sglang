from typing import Any, List

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LowConfidence(DllmAlgorithm):
    """Each step unmasks positions whose predicted-token confidence exceeds a
    threshold (falling back to the highest-confidence masked position).
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def prefill_tokens(self, batch):
        """Return input blocks only when a batch needs no denoising."""
        if batch is None or not batch.reqs:
            return None
        blocks = []
        for req in batch.reqs:
            if (
                not req.is_dllm_prefill()
                or req.dllm_incomplete_ids
                or req.dllm_algo_state is not None
                or req.extend_end > len(req.origin_input_ids)
            ):
                return None
            block = req.get_fill_ids()[req.prefix_len :]
            if len(block) != self.block_size or self.mask_id in block:
                return None
            blocks.append(list(block))
        return blocks

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
        prompt_mask = forward_batch.dllm_prompt_mask
        if prompt_mask is not None:
            block_mask_index = torch.where(prompt_mask, False, block_mask_index)
        done = block_mask_index.sum(dim=1) == 0

        x = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
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
