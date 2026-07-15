from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class JointThreshold(DllmAlgorithm):
    """Joint-threshold denoising: mask-to-token (M2T) unmasking plus token-to-token
    (T2T) edits, finishing on no-change or an exhausted edit budget. Stateful (edit
    budget + prompt mask), carried across FDFO rounds via ``dllm_algo_state``.
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
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        # Built once as a GPU tensor and reused across steps (no per-step
        # host/device transfer); the FDFO carry keeps it in-process.
        prompt_mask = input_ids != self.mask_id
        return [
            {
                "post_edit_steps": 0,
                "finished": False,
                "prompt_mask": prompt_mask[i],
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
        done: List[bool] = []

        for i in range(batch_size):
            state = states[i]
            if state["finished"]:
                done.append(True)
                continue

            block_start = i * self.block_size
            block_end = block_start + self.block_size
            curr_input_ids = forward_batch.input_ids[block_start:block_end]
            curr_logits = full_logits[block_start:block_end]
            curr_prompt_mask = state["prompt_mask"]

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
            budget_exhausted = False
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
                    budget_exhausted = True

            if not budget_exhausted:
                # Token to token (T2T)
                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )
                transfer_index = mask_transfer_index | edit_transfer_index
                if transfer_index.any():
                    curr_input_ids[transfer_index] = x[transfer_index]
                else:
                    state["finished"] = True

            # A terminating step changes nothing, so this forward already holds the
            # block's final KV: emit it now rather than after an extra forward.
            done.append(state["finished"])

        return done


Algorithm = JointThreshold
