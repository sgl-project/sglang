from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class JointThresholdInDel(DllmAlgorithm):
    """Joint-threshold denoising with InDel (insertion & deletion) support.

    Extends JointThreshold to handle edit tokens that modify block structure:
    DELETE removes a position (block shrinks, padded with mask), SPLIT (insert)
    expands a position into [mask, original_token].
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.delete_token_id = config.delete_token_id
        self.split_token_id = config.split_token_id
        if self.delete_token_id is None or self.split_token_id is None:
            raise RuntimeError(
                f"JointThresholdInDel requires delete_token_id and split_token_id, "
                f"got delete_token_id={self.delete_token_id}, "
                f"split_token_id={self.split_token_id}"
            )

    def max_steps(self, block_size: int) -> int:
        return block_size + self.max_post_edit_steps + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        return [
            {
                "prompt_mask": input_ids[i] != self.mask_id,
                "is_orig_mask": input_ids[i] == self.mask_id,
                "post_edit_steps": 0,
                "finished": False,
                "seen_input_keys": set(),
                "sampled_history": {},
            }
            for i in range(batch_size)
        ]

    def _apply_edit_operations(self, block_tokens, old_block_tokens, is_orig_mask):
        """Process DELETE and SPLIT tokens, tracking which positions were original masks.

        - DELETE: remove the position
        - SPLIT: expand to [mask_id, old_token]
        - Pad/truncate result to block_size
        """
        result_tokens = []
        result_tracking = []
        for i, token in enumerate(block_tokens):
            if token == self.delete_token_id:
                continue
            elif token == self.split_token_id:
                result_tokens.extend([self.mask_id, old_block_tokens[i]])
                result_tracking.extend([False, is_orig_mask[i]])
            else:
                result_tokens.append(token)
                result_tracking.append(is_orig_mask[i])

        if len(result_tokens) > self.block_size:
            result_tokens = result_tokens[: self.block_size]
            result_tracking = result_tracking[: self.block_size]
        elif len(result_tokens) < self.block_size:
            pad_count = self.block_size - len(result_tokens)
            result_tokens.extend([self.mask_id] * pad_count)
            result_tracking.extend([False] * pad_count)

        return result_tokens, result_tracking

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
            if state["finished"]:
                done.append(True)
                continue
            done.append(False)

            block_start = i * self.block_size
            block_end = block_start + self.block_size
            curr_input_ids = forward_batch.input_ids[block_start:block_end]
            curr_logits = full_logits[block_start:block_end]
            curr_prompt_mask = state["prompt_mask"]

            # Greedy decode: argmax tokens and their confidence
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

            # Track original vs new masks (introduced by DELETE/SPLIT)
            is_orig_mask = state["is_orig_mask"]
            original_mask_count = int((is_orig_mask & mask_index).sum().item())
            total_mask = mask_index.sum().item()
            new_mask_count = total_mask - original_mask_count

            # M2T: while original masks remain, fill at least (1 + new_mask_count)
            # per step by threshold or top-k. Once all original masks are filled,
            # fill all remaining masks at once.
            mask_transfer_index = torch.zeros_like(mask_index)
            if has_mask:
                confidence = torch.where(mask_index, p, float("-inf"))
                if original_mask_count > 0:
                    num_need = 1 + new_mask_count
                    high_conf = (confidence > self.threshold) & mask_index
                    if high_conf.sum().item() >= num_need:
                        mask_transfer_index = high_conf
                    else:
                        k_val = min(num_need, total_mask)
                        if k_val > 0:
                            _, select_index = torch.topk(confidence, k=k_val)
                            mask_transfer_index[select_index] = True
                else:
                    mask_transfer_index = mask_index

            # Post-edit budget: once all original masks filled, count steps.
            # After budget exhausted, scrub DELETE/SPLIT from predictions and
            # do a final decode.
            force_finish = False
            if original_mask_count == 0:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    force_finish = True
                    # Replace any DELETE/SPLIT in predictions with next-best token
                    indel_mask = (x == self.delete_token_id) | (
                        x == self.split_token_id
                    )
                    if indel_mask.any():
                        indel_indices = indel_mask.nonzero(as_tuple=False).view(-1)
                        scrub_logits = curr_logits[indel_indices].clone()
                        scrub_logits[:, self.delete_token_id] = float("-inf")
                        scrub_logits[:, self.split_token_id] = float("-inf")
                        x[indel_indices] = torch.argmax(scrub_logits, dim=-1)

            # T2T: edit non-mask, non-prompt positions where model disagrees
            edit_mask = ~mask_index & ~curr_prompt_mask
            edit_transfer_index = (
                (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index
            if not transfer_index.any():
                state["finished"] = True
                done[-1] = True
                continue

            # Anti-loop: if this block state was seen before, inject diversity
            # at the position with lowest confidence (deterministic).
            old_block = curr_input_ids.tolist()
            input_key = tuple(old_block)

            if input_key in state["seen_input_keys"] and not force_finish:
                changing = transfer_index & (x != curr_input_ids)
                changing_positions = changing.nonzero(as_tuple=False).view(-1).tolist()

                if changing_positions:
                    # Pick lowest-confidence position (minimizes quality impact)
                    pos_tensor = torch.tensor(changing_positions, device=device)
                    chosen_pos = changing_positions[p[pos_tensor].argmin().item()]
                    pos_logits = curr_logits[chosen_pos, :].clone()
                    # Exclude previously sampled tokens at this position
                    for prev_tok in state["sampled_history"].get(chosen_pos, ()):
                        pos_logits[prev_tok] = float("-inf")
                    x[chosen_pos] = torch.argmax(pos_logits)
            state["seen_input_keys"].add(input_key)

            # Record written tokens for anti-loop history
            write_positions = transfer_index.nonzero(as_tuple=False).view(-1)
            for pos in write_positions.tolist():
                state["sampled_history"].setdefault(pos, set()).add(x[pos].item())

            # Write tokens
            curr_input_ids[transfer_index] = x[transfer_index]

            # Apply DELETE/SPLIT operations if any edit tokens were written
            has_indel = (
                (curr_input_ids == self.delete_token_id)
                | (curr_input_ids == self.split_token_id)
            ).any()
            if has_indel:
                current_block = curr_input_ids.tolist()
                edited_block, new_is_orig_mask = self._apply_edit_operations(
                    current_block, old_block, is_orig_mask.tolist()
                )
                forward_batch.input_ids[block_start:block_end] = torch.tensor(
                    edited_block, device=device, dtype=torch.long
                )
                state["is_orig_mask"] = torch.tensor(
                    new_is_orig_mask, dtype=torch.bool, device=device
                )

            if force_finish:
                state["finished"] = True
                done[-1] = True

        return done


Algorithm = JointThresholdInDel
