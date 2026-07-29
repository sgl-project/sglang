from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class EBSampling(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.gamma = config.algorithm_config.get("gamma", 0.15)
        if self.gamma < 0:
            raise ValueError("EBSampling requires a non-negative gamma threshold.")

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        _states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
        done = []

        for batch_id in range(batch_size):
            block_start = batch_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            if not block_mask_index.any():
                done.append(True)
                continue

            curr_logits = full_logits[block_start:block_end]
            # Never select or score the mask token itself: suppress it before
            # both candidate selection and the entropy computation.
            if 0 <= self.mask_id < curr_logits.shape[-1]:
                curr_logits = curr_logits.clone()
                curr_logits[..., self.mask_id] = torch.finfo(curr_logits.dtype).min
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

            # Always reveal the lowest-entropy token, then extend the prefix
            # while the running entropy budget stays within gamma.
            reveal_mask = torch.zeros_like(sorted_entropies, dtype=torch.bool)
            reveal_mask[0] = True
            reveal_mask[1:] = torch.cumsum(sorted_entropies[:-1], dim=0) <= self.gamma
            transfer_positions = sorted_positions[reveal_mask]
            transfer_index = torch.zeros_like(block_mask_index)
            transfer_index[transfer_positions] = True

            block_input_ids[transfer_index] = x[transfer_index]
            done.append(False)

        return done


Algorithm = EBSampling
