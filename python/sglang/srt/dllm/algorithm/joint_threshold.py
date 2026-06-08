from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class JointThreshold(DllmAlgorithm):
    """Joint-threshold denoising with a post-fill edit phase: mask-to-token (M2T)
    unmasking above ``threshold`` plus token-to-token (T2T) edits above
    ``edit_threshold``, finishing on no-change or an exhausted edit budget. Stateful
    (edit budget + prompt mask), carried across FDFO rounds via ``dllm_algo_state``.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def max_steps(self, block_size: int) -> int:
        # block_size fills + post-edit budget + the trailing completion step.
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        prompt_mask = input_ids != self.mask_id
        return [
            {
                "post_edit_steps": 0,
                "finished": False,
                # Prompt-carry positions, fixed for the block; CPU list for cheap carry.
                "prompt_mask": prompt_mask[i].tolist(),
            }
            for i in range(batch_size)
        ]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        done: List[bool] = []

        for i in range(batch_size):
            state = states[i]
            # Finished in a prior step => complete on entry, KV persisted here.
            if state["finished"]:
                done.append(True)
                continue
            done.append(False)

            block_start = i * self.block_size
            block_end = block_start + self.block_size
            curr_input_ids = forward_batch.input_ids[block_start:block_end]
            curr_logits = full_logits[block_start:block_end]
            curr_prompt_mask = torch.tensor(
                state["prompt_mask"], device=device, dtype=torch.bool
            )

            if self.penalty_lambda > 0:
                prev_ids = curr_input_ids[:-1]
                curr_logits[1:, :].scatter_(
                    1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                )

            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            mask_index = curr_input_ids == self.mask_id
            has_mask = mask_index.any()

            # Mask to token (M2T)
            mask_transfer_index = torch.zeros_like(mask_index)
            if has_mask:
                confidence = torch.where(mask_index, p, -np.inf)
                mask_transfer_index = confidence > self.threshold
                if not mask_transfer_index.any():
                    _, select_index = torch.topk(confidence, k=1)
                    mask_transfer_index[select_index] = True
            else:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    state["finished"] = True
                    continue

            # Token to token (T2T)
            edit_mask = ~mask_index & ~curr_prompt_mask
            edit_transfer_index = (
                (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index
            if not transfer_index.any():
                state["finished"] = True
                continue

            curr_input_ids[transfer_index] = x[transfer_index]

        return done


Algorithm = JointThreshold
