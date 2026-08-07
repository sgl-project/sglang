"""Joint-threshold denoising with insertion and deletion support."""

from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

_is_npu = is_npu()


def _apply_edit_operations_tensor(
    block_tokens: torch.Tensor,
    old_block_tokens: torch.Tensor,
    is_orig_mask: torch.Tensor,
    block_size: int,
    mask_id: int,
    delete_token_id: int,
    split_token_id: int,
):
    """Apply edits with fixed-shape tensor operations."""
    device = block_tokens.device
    is_delete = block_tokens == delete_token_id
    is_split = block_tokens == split_token_id

    counts = torch.ones_like(block_tokens)
    counts = torch.where(is_delete, torch.zeros_like(counts), counts)
    counts = torch.where(is_split, torch.full_like(counts, 2), counts)

    dest_start = torch.cumsum(counts, dim=0) - counts

    # The final slot absorbs writes that should not appear in the result.
    buf_size = 2 * block_size + 1
    dummy = buf_size - 1

    out_tokens = torch.full(
        (buf_size,), mask_id, dtype=block_tokens.dtype, device=device
    )
    out_tracking = torch.zeros((buf_size,), dtype=torch.bool, device=device)

    # SPLIT emits a mask first; DELETE writes only to the sink.
    idx1 = torch.where(
        is_delete, torch.full_like(dest_start, dummy), dest_start.clamp(0, dummy)
    )
    val1 = torch.where(is_split, torch.full_like(block_tokens, mask_id), block_tokens)
    track1 = torch.where(is_split, torch.zeros_like(is_orig_mask), is_orig_mask)
    out_tokens.scatter_(0, idx1, val1)
    out_tracking.scatter_(0, idx1, track1)

    # SPLIT preserves the previous token in its second emitted slot.
    idx2 = torch.where(
        is_split, (dest_start + 1).clamp(0, dummy), torch.full_like(dest_start, dummy)
    )
    out_tokens.scatter_(0, idx2, old_block_tokens)
    out_tracking.scatter_(0, idx2, is_orig_mask)

    return out_tokens[:block_size], out_tracking[:block_size]


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
        self.post_edit_threshold = config.algorithm_config.get(
            "post_edit_threshold", self.edit_threshold
        )
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.min_mask_transfer = config.algorithm_config.get("min_mask_transfer", 1)
        if self.min_mask_transfer < 1:
            raise ValueError("min_mask_transfer must be at least 1")
        self.min_indel_confidence = config.algorithm_config.get(
            "min_indel_confidence", 0.0
        )
        if not 0 <= self.min_indel_confidence <= 1:
            raise ValueError("min_indel_confidence must be between 0 and 1")
        self.done_check_interval = config.algorithm_config.get("done_check_interval", 1)
        if self.done_check_interval < 1:
            raise ValueError("done_check_interval must be at least 1")
        self.enable_graph = config.algorithm_config.get("enable_graph", False)
        if self.enable_graph and not _is_npu:
            raise ValueError("JointThresholdInDel graph mode requires an NPU device")
        self.delete_token_id = config.delete_token_id
        self.split_token_id = config.split_token_id
        self._npu_graph_runners = {}
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
                "sampled_mask": None,
            }
            for i in range(batch_size)
        ]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        if self.enable_graph:
            return self.step_graph(forward_batch, full_logits, states)
        return self.step_vectorized(forward_batch, full_logits, states)

    def step_graph(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Replay the complete fixed-width InDel update as an NPUGraph."""
        if not _is_npu or self.fdfo:
            # FDFO carries per-request state across changing batch
            # compositions; keep its established path until graph-state
            # packing is implemented.
            return self.step_vectorized(forward_batch, full_logits, states)

        batch_size = forward_batch.batch_size
        runner = states[0].get("_npu_graph_runner")
        if runner is None:
            from sglang.srt.dllm.algorithm.joint_threshold_indel_graph import (
                NPUJointThresholdGraphRunner,
            )

            cache_key = (
                batch_size,
                full_logits.shape[-1],
                full_logits.data_ptr(),
            )
            runner = self._npu_graph_runners.get(cache_key)
            if runner is None:
                runner = NPUJointThresholdGraphRunner(
                    full_logits=full_logits,
                    batch_size=batch_size,
                    block_size=self.block_size,
                    mask_id=self.mask_id,
                    delete_token_id=self.delete_token_id,
                    split_token_id=self.split_token_id,
                    threshold=self.threshold,
                    edit_threshold=self.edit_threshold,
                    post_edit_threshold=self.post_edit_threshold,
                    max_post_edit_steps=self.max_post_edit_steps,
                    min_mask_transfer=self.min_mask_transfer,
                    min_indel_confidence=self.min_indel_confidence,
                    done_check_interval=self.done_check_interval,
                    max_steps=self.max_steps(self.block_size),
                )
                self._npu_graph_runners[cache_key] = runner

            runner.reset(
                forward_batch.input_ids,
                prompt_mask=torch.stack([state["prompt_mask"] for state in states]),
                is_orig_mask=torch.stack([state["is_orig_mask"] for state in states]),
            )
            forward_batch.input_ids = runner.input_ids.view(-1)
            for state in states:
                state["_npu_graph_runner"] = runner
        elif not runner.matches_logits(full_logits):
            raise RuntimeError(
                "JointThresholdInDel NPUGraph logits buffer changed during "
                "a synchronous denoise loop"
            )

        done = runner.replay()
        for state, is_done in zip(states, done):
            state["finished"] = is_done
        return done

    def step_vectorized(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Run the fixed-shape implementation without graph capture."""
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

            if state["sampled_mask"] is None:
                state["sampled_mask"] = torch.zeros(
                    (self.block_size, curr_logits.shape[-1]),
                    dtype=torch.bool,
                    device=device,
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
            is_orig_mask = state["is_orig_mask"]
            if self.min_indel_confidence > 0:
                low_conf_indel = (
                    mask_index
                    & ((x == self.delete_token_id) | (x == self.split_token_id))
                    & (p <= self.min_indel_confidence)
                )
                scrub_logits = curr_logits.clone()
                scrub_logits[:, self.delete_token_id] = float("-inf")
                scrub_logits[:, self.split_token_id] = float("-inf")
                scrub_x = torch.argmax(scrub_logits, dim=-1)
                x = torch.where(low_conf_indel, scrub_x, x)

            confidence = torch.where(mask_index, p, torch.full_like(p, float("-inf")))
            high_conf = (confidence > self.threshold) & mask_index

            # Fetch the control-flow counters in one host transfer.
            stats = torch.stack(
                [
                    (is_orig_mask & mask_index).sum(),
                    mask_index.sum(),
                    high_conf.sum(),
                ]
            ).tolist()
            original_mask_count, total_mask, high_conf_count = stats
            new_mask_count = total_mask - original_mask_count

            mask_transfer_index = torch.zeros_like(mask_index)
            if original_mask_count > 0:
                num_need = self.min_mask_transfer + new_mask_count
                if high_conf_count >= num_need:
                    mask_transfer_index = high_conf
                else:
                    k_val = min(num_need, total_mask)
                    if k_val > 0:
                        _, select_index = torch.topk(confidence, k=k_val)
                        mask_transfer_index[select_index] = True
            else:
                mask_transfer_index = mask_index

            force_finish = False
            if original_mask_count == 0:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    force_finish = True
                    # Replace edit tokens before forcing completion.
                    if self.min_indel_confidence == 0:
                        scrub_logits = curr_logits.clone()
                        scrub_logits[:, self.delete_token_id] = float("-inf")
                        scrub_logits[:, self.split_token_id] = float("-inf")
                        scrub_x = torch.argmax(scrub_logits, dim=-1)
                    indel_mask = (x == self.delete_token_id) | (
                        x == self.split_token_id
                    )
                    x = torch.where(indel_mask, scrub_x, x)

            edit_mask = ~mask_index & ~curr_prompt_mask
            effective_edit_threshold = (
                self.post_edit_threshold
                if original_mask_count == 0
                else self.edit_threshold
            )
            edit_transfer_index = (
                (p > effective_edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index

            if not transfer_index.any().item():
                state["finished"] = True
                done[-1] = True
                continue

            pre_write_ids = curr_input_ids.clone()

            # The non-graph path keeps anti-loop history on the host.
            old_block = pre_write_ids.tolist()
            input_key = tuple(old_block)

            if input_key in state["seen_input_keys"] and not force_finish:
                changing = transfer_index & (x != curr_input_ids)
                if changing.any().item():
                    conf_for_choice = torch.where(
                        changing, p, torch.full_like(p, float("inf"))
                    )
                    chosen_pos = torch.argmin(conf_for_choice)
                    pos_logits = curr_logits[chosen_pos, :].clone()
                    pos_logits = torch.where(
                        state["sampled_mask"][chosen_pos],
                        torch.full_like(pos_logits, float("-inf")),
                        pos_logits,
                    )
                    new_tok = torch.argmax(pos_logits)
                    x = x.clone()
                    x[chosen_pos] = new_tok
            state["seen_input_keys"].add(input_key)

            # Scatter avoids data-dependent indexing during graph capture.
            selected_col = x.unsqueeze(-1)
            prev_bit = state["sampled_mask"].gather(1, selected_col)
            new_bit = prev_bit | transfer_index.unsqueeze(-1)
            state["sampled_mask"].scatter_(1, selected_col, new_bit)

            curr_input_ids.copy_(torch.where(transfer_index, x, curr_input_ids))

            # Always apply edits to preserve fixed tensor shapes.
            edited_tokens, new_is_orig_mask = _apply_edit_operations_tensor(
                curr_input_ids,
                pre_write_ids,
                is_orig_mask,
                self.block_size,
                self.mask_id,
                self.delete_token_id,
                self.split_token_id,
            )
            forward_batch.input_ids[block_start:block_end] = edited_tokens
            state["is_orig_mask"] = new_is_orig_mask

            if force_finish:
                state["finished"] = True
                done[-1] = True

        return done


Algorithm = JointThresholdInDel
