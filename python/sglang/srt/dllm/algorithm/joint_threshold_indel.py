from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class JointThresholdInDel(DllmAlgorithm):
    """Joint-threshold denoising with InDel (insertion & deletion) support.

    Extends JointThreshold to handle edit tokens that modify block structure.
    DELETE removes a position and pads the shortened block with MASK. SPLIT
    inserts MASK before the previous token at that position.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.max_regular_update_steps = self.block_size + self.max_post_edit_steps
        self.delete_token_id = config.delete_token_id
        self.split_token_id = config.split_token_id
        if self.delete_token_id is None or self.split_token_id is None:
            raise RuntimeError(
                f"JointThresholdInDel requires delete_token_id and split_token_id, "
                f"got delete_token_id={self.delete_token_id}, "
                f"split_token_id={self.split_token_id}"
            )

    def max_steps(self, block_size: int) -> int:
        # Regular updates, one final cleanup update, and one forward that
        # persists the cleaned block in KV cache.
        return block_size + self.max_post_edit_steps + 2

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        # The prompt is the immutable contiguous prefix before the first mask.
        # is_orig_mask marks positions associated with masks in the initial
        # block. DELETE/SPLIT move the marker with the corresponding token.
        prompt_lens = (input_ids != self.mask_id).cumprod(dim=1).sum(dim=1).tolist()
        positions = torch.arange(self.block_size, device=input_ids.device)
        return [
            {
                "prompt_len": prompt_lens[i],
                "prompt_index": positions < prompt_lens[i],
                "is_orig_mask": input_ids[i] == self.mask_id,
                "post_edit_steps": 0,
                "num_update_steps": 0,
                "finished": False,
                "seen_input_keys": set(),
                "sampled_history": {},
            }
            for i in range(batch_size)
        ]

    def _apply_edit_operations(
        self,
        block_tokens,
        old_block_tokens,
        is_orig_mask,
        prompt_len,
    ):
        """Apply DELETE/SPLIT to the generated suffix.

        ``block_tokens`` contains this step's selected predictions, while
        ``old_block_tokens`` contains the input block. The prompt is copied
        unchanged from the input block. In the suffix, DELETE removes a
        position and SPLIT expands it to ``[mask_id, old_block_tokens[i]]``.
        The inserted mask is new; the old token retains its ``is_orig_mask``
        flag. The result is padded or truncated to ``block_size``.
        """
        result_tokens = old_block_tokens[:prompt_len]
        result_is_orig_mask = is_orig_mask[:prompt_len]
        for i in range(prompt_len, self.block_size):
            token = block_tokens[i]
            if token == self.delete_token_id:
                continue
            elif token == self.split_token_id:
                result_tokens.extend([self.mask_id, old_block_tokens[i]])
                result_is_orig_mask.extend([False, is_orig_mask[i]])
            else:
                result_tokens.append(token)
                result_is_orig_mask.append(is_orig_mask[i])

        if len(result_tokens) > self.block_size:
            result_tokens = result_tokens[: self.block_size]
            result_is_orig_mask = result_is_orig_mask[: self.block_size]
        elif len(result_tokens) < self.block_size:
            pad_count = self.block_size - len(result_tokens)
            result_tokens.extend([self.mask_id] * pad_count)
            result_is_orig_mask.extend([False] * pad_count)

        return result_tokens, result_is_orig_mask

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
            curr_prompt_index = state["prompt_index"]

            # Greedy decode: argmax tokens and their confidence
            x = torch.argmax(curr_logits, dim=-1)
            probabilities = F.softmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    probabilities,
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            mask_index = curr_input_ids == self.mask_id
            is_orig_mask = state["is_orig_mask"]
            original_mask_index = is_orig_mask & mask_index
            original_mask_count, total_mask_count = torch.stack(
                (original_mask_index.sum(), mask_index.sum())
            ).tolist()
            introduced_mask_count = total_mask_count - original_mask_count

            state["num_update_steps"] += 1
            if original_mask_count == 0:
                state["post_edit_steps"] += 1

            is_final_cleanup = (
                state["post_edit_steps"] > self.max_post_edit_steps
                or state["num_update_steps"] > self.max_regular_update_steps
            )

            # Replace MASK predictions at unresolved original masks before
            # selection. Otherwise, a selected position could write MASK back
            # without resolving the original mask.
            if not is_final_cleanup:
                fallback_positions = (
                    (original_mask_index & (x == self.mask_id))
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
                if fallback_positions.numel() > 0:
                    fallback_logits = curr_logits[fallback_positions].clone()
                    fallback_logits[:, self.mask_id] = float("-inf")
                    fallback_tokens = torch.argmax(fallback_logits, dim=-1)
                    x[fallback_positions] = fallback_tokens
                    p[fallback_positions] = probabilities[
                        fallback_positions, fallback_tokens
                    ]

            # M2T: request introduced_mask_count + 1 transfers while original
            # masks remain. Since only introduced_mask_count masks are new, at
            # least one selected position must be an original mask. Once no
            # original masks remain, transfer all masks.
            mask_transfer_index = torch.zeros_like(mask_index)
            if total_mask_count > 0:
                confidence = torch.where(mask_index, p, float("-inf"))
                if is_final_cleanup:
                    mask_transfer_index = mask_index & ~curr_prompt_index
                elif original_mask_count > 0:
                    required_mask_transfers = 1 + introduced_mask_count
                    high_confidence_index = (confidence > self.threshold) & mask_index
                    if high_confidence_index.sum().item() >= required_mask_transfers:
                        mask_transfer_index = high_confidence_index
                    else:
                        num_transfers = min(required_mask_transfers, total_mask_count)
                        if num_transfers > 0:
                            _, selected_positions = torch.topk(
                                confidence, k=num_transfers
                            )
                            mask_transfer_index[selected_positions] = True
                else:
                    mask_transfer_index = mask_index

            # T2T: edit non-mask, non-prompt positions where model disagrees
            edit_mask = ~mask_index & ~curr_prompt_index
            edit_transfer_index = (
                (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index
            write_positions = transfer_index.nonzero(as_tuple=False).view(-1)
            if write_positions.numel() == 0:
                state["finished"] = True
                done[-1] = True
                continue

            if is_final_cleanup:
                # Final cleanup must not emit MASK, DELETE, or SPLIT.
                write_tokens = x[write_positions]
                forbidden_write_mask = (
                    (write_tokens == self.mask_id)
                    | (write_tokens == self.delete_token_id)
                    | (write_tokens == self.split_token_id)
                )
                scrub_positions = write_positions[forbidden_write_mask]
                if scrub_positions.numel() > 0:
                    scrub_logits = curr_logits[scrub_positions].clone()
                    scrub_logits[:, self.mask_id] = float("-inf")
                    scrub_logits[:, self.delete_token_id] = float("-inf")
                    scrub_logits[:, self.split_token_id] = float("-inf")
                    x[scrub_positions] = torch.argmax(scrub_logits, dim=-1)

            # Anti-loop: if this block state was seen before, inject diversity
            # at the position with lowest confidence (deterministic).
            old_block = curr_input_ids.tolist()
            input_key = tuple(old_block)

            if input_key in state["seen_input_keys"] and not is_final_cleanup:
                changing_index = transfer_index & (x != curr_input_ids)
                changing_positions = changing_index.nonzero(as_tuple=False).view(-1)

                if changing_positions.numel() > 0:
                    # Pick lowest-confidence position (minimizes quality impact)
                    chosen_position = changing_positions[
                        p[changing_positions].argmin()
                    ].item()
                    chosen_position_logits = curr_logits[chosen_position, :].clone()
                    # Exclude previously sampled tokens at this position
                    for sampled_token in state["sampled_history"].get(
                        chosen_position, ()
                    ):
                        chosen_position_logits[sampled_token] = float("-inf")
                    if original_mask_index[chosen_position]:
                        chosen_position_logits[self.mask_id] = float("-inf")
                    x[chosen_position] = torch.argmax(chosen_position_logits)
            state["seen_input_keys"].add(input_key)

            write_pairs = torch.stack(
                (write_positions, x[write_positions]), dim=1
            ).tolist()
            block_tokens = old_block.copy()
            has_indel = False
            input_ids_changed = False
            # Record both M2T and T2T selections for anti-loop resampling.
            for pos, token in write_pairs:
                state["sampled_history"].setdefault(pos, set()).add(token)
                block_tokens[pos] = token
                has_indel |= token in (
                    self.delete_token_id,
                    self.split_token_id,
                )
                input_ids_changed |= token != old_block[pos]

            if has_indel:
                edited_block, new_is_orig_mask = self._apply_edit_operations(
                    block_tokens,
                    old_block,
                    is_orig_mask.tolist(),
                    state["prompt_len"],
                )
                forward_batch.input_ids[block_start:block_end] = torch.tensor(
                    edited_block, device=device, dtype=torch.long
                )
                state["is_orig_mask"] = torch.tensor(
                    new_is_orig_mask, dtype=torch.bool, device=device
                )
            else:
                curr_input_ids[write_positions] = x[write_positions]

            if is_final_cleanup:
                state["finished"] = True
                # If IDs changed, the next forward must persist their KV cache
                # before the request can be emitted.
                done[-1] = not input_ids_changed

        return done


Algorithm = JointThresholdInDel
