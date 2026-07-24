"""JointThresholdInDel: joint-threshold denoising with InDel (insertion &
deletion) support.

Two step() implementations are available, selected via the
SGLANG_NPU_JTI_IMPL environment variable (mirrors the
SGLANG_NPU_BLOCK_TOPK_IMPL env-selectable pattern used for the block-routing
MoE CANN-hybrid fix):

  - "original"   : the original PR #31773 step() -- Python-loop DELETE/SPLIT
                   compaction, dict-of-sets anti-loop history, multiple
                   separate .item()/.tolist() host syncs (10 host-sync call
                   sites total). Default on non-NPU platforms.
  - "vectorized" : rewritten step() -- fixed-shape tensor scatter for
                   DELETE/SPLIT compaction, tensor-mask anti-loop history via
                   gather/scatter_, batched mask-count fetch, and
                   torch.where/scatter_-based writes instead of boolean-mask
                   assignment (itself a hidden Ascend capture-breaking host
                   sync, found via capture bisection).
                   Verified bit-exact vs "original" on CPU (12 cases) and
                   real NPU hardware (7 cases). Motivation is NPU graph-capture
                   eligibility, not raw step speed: on an isolated,
                   edit-path-forced microbenchmark this measured 1.62x faster
                   (die 4, batch=4, block_size=32), but a rigorous real-weight
                   engine e2e test (real 100B-A6B model, TP=4, natural
                   decoding, n=429 steps) found the opposite for step time --
                   original median 2.351ms vs vectorized 2.605ms (~10% slower
                   per step) -- while aggregate e2e throughput was slightly
                   higher for vectorized (41.5 vs 39.6 tok/s, +4.9%), since
                   JTI's block-wise diffusion decode does not map steps to
                   tokens 1:1. Default on NPU because it remains the only
                   implementation of the two that is graph-capture-compatible
                   (fewer host syncs), not because it is faster.

Both implementations remain in this file so the CANN-hybrid (block-routing)
and vectorized (JointThresholdInDel) fixes can be toggled independently for
A/B engine profiling.
"""

import os
from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

_is_npu = is_npu()

# NPU host-sync minimization selector for JointThresholdInDel.step(): the
# original PR #31773 step() has 10 host-synchronizing call sites
# (.item()/.tolist()/host round-trip re-upload) that make NPUGraph capture
# impossible and cost real step-time on Ascend (measured 1.62x speedup from
# removing 8 of them). Set SGLANG_NPU_JTI_IMPL=original to force the
# original implementation for comparison.
_JTI_IMPL = os.environ.get(
    "SGLANG_NPU_JTI_IMPL", "vectorized" if _is_npu else "original"
)


def _apply_edit_operations_original(block_tokens, old_block_tokens, is_orig_mask, block_size, mask_id, delete_token_id, split_token_id):
    """Process DELETE and SPLIT tokens, tracking which positions were original masks.

    - DELETE: remove the position
    - SPLIT: expand to [mask_id, old_token]
    - Pad/truncate result to block_size
    """
    result_tokens = []
    result_tracking = []
    for i, token in enumerate(block_tokens):
        if token == delete_token_id:
            continue
        elif token == split_token_id:
            result_tokens.extend([mask_id, old_block_tokens[i]])
            result_tracking.extend([False, is_orig_mask[i]])
        else:
            result_tokens.append(token)
            result_tracking.append(is_orig_mask[i])

    if len(result_tokens) > block_size:
        result_tokens = result_tokens[:block_size]
        result_tracking = result_tracking[:block_size]
    elif len(result_tokens) < block_size:
        pad_count = block_size - len(result_tokens)
        result_tokens.extend([mask_id] * pad_count)
        result_tracking.extend([False] * pad_count)

    return result_tokens, result_tracking


def _apply_edit_operations_tensor(
    block_tokens: torch.Tensor,
    old_block_tokens: torch.Tensor,
    is_orig_mask: torch.Tensor,
    block_size: int,
    mask_id: int,
    delete_token_id: int,
    split_token_id: int,
):
    """Pure-tensor, fixed-shape replacement for _apply_edit_operations_original.
    No host syncs, no dynamic-size lists.

    Semantics identical to the original: DELETE removes a position, SPLIT
    expands a position into [mask_id, old_block_tokens[i]], result is
    padded/truncated to block_size. Implemented as a scatter into an
    oversized fixed buffer (2*block_size+1, generous upper bound since SPLIT
    at most doubles length) followed by a static-size slice -- this keeps
    every intermediate tensor a fixed, data-independent shape so no
    device->host synchronization is required to determine sizes.
    """
    device = block_tokens.device
    is_delete = block_tokens == delete_token_id
    is_split = block_tokens == split_token_id

    counts = torch.ones_like(block_tokens)
    counts = torch.where(is_delete, torch.zeros_like(counts), counts)
    counts = torch.where(is_split, torch.full_like(counts, 2), counts)

    dest_start = torch.cumsum(counts, dim=0) - counts  # exclusive cumsum

    buf_size = 2 * block_size + 1  # +1 dummy overflow/delete sink slot
    dummy = buf_size - 1

    out_tokens = torch.full((buf_size,), mask_id, dtype=block_tokens.dtype, device=device)
    out_tracking = torch.zeros((buf_size,), dtype=torch.bool, device=device)

    # First emitted slot per source position: for SPLIT -> (mask_id, False);
    # for a kept (non-delete) token -> (token, is_orig_mask[i]); DELETE emits
    # nothing so it is routed to the dummy sink index.
    idx1 = torch.where(is_delete, torch.full_like(dest_start, dummy), dest_start.clamp(0, dummy))
    val1 = torch.where(is_split, torch.full_like(block_tokens, mask_id), block_tokens)
    track1 = torch.where(is_split, torch.zeros_like(is_orig_mask), is_orig_mask)
    out_tokens.scatter_(0, idx1, val1)
    out_tracking.scatter_(0, idx1, track1)

    # Second emitted slot, SPLIT only: (old_block_tokens[i], is_orig_mask[i]).
    idx2 = torch.where(is_split, (dest_start + 1).clamp(0, dummy), torch.full_like(dest_start, dummy))
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
                "sampled_mask": None,
            }
            for i in range(batch_size)
        ]

    def _apply_edit_operations(self, block_tokens, old_block_tokens, is_orig_mask):
        return _apply_edit_operations_original(
            block_tokens, old_block_tokens, is_orig_mask,
            self.block_size, self.mask_id, self.delete_token_id, self.split_token_id,
        )

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Dispatches to step_original or step_vectorized based on
        SGLANG_NPU_JTI_IMPL / _JTI_IMPL (see module docstring)."""
        if _JTI_IMPL == "original":
            return self.step_original(forward_batch, full_logits, states)
        else:
            return self.step_vectorized(forward_batch, full_logits, states)

    def step_original(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Original PR #31773 step() -- unmodified algorithm/host-sync
        pattern, kept for A/B comparison against step_vectorized."""
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

            is_orig_mask = state["is_orig_mask"]
            original_mask_count = int((is_orig_mask & mask_index).sum().item())
            total_mask = mask_index.sum().item()
            new_mask_count = total_mask - original_mask_count

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

            force_finish = False
            if original_mask_count == 0:
                state["post_edit_steps"] += 1
                if state["post_edit_steps"] > self.max_post_edit_steps:
                    force_finish = True
                    indel_mask = (x == self.delete_token_id) | (
                        x == self.split_token_id
                    )
                    if indel_mask.any():
                        indel_indices = indel_mask.nonzero(as_tuple=False).view(-1)
                        scrub_logits = curr_logits[indel_indices].clone()
                        scrub_logits[:, self.delete_token_id] = float("-inf")
                        scrub_logits[:, self.split_token_id] = float("-inf")
                        x[indel_indices] = torch.argmax(scrub_logits, dim=-1)

            edit_mask = ~mask_index & ~curr_prompt_mask
            edit_transfer_index = (
                (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index
            if not transfer_index.any():
                state["finished"] = True
                done[-1] = True
                continue

            old_block = curr_input_ids.tolist()
            input_key = tuple(old_block)

            if input_key in state["seen_input_keys"] and not force_finish:
                changing = transfer_index & (x != curr_input_ids)
                changing_positions = changing.nonzero(as_tuple=False).view(-1).tolist()

                if changing_positions:
                    pos_tensor = torch.tensor(changing_positions, device=device)
                    chosen_pos = changing_positions[p[pos_tensor].argmin().item()]
                    pos_logits = curr_logits[chosen_pos, :].clone()
                    for prev_tok in state["sampled_history"].get(chosen_pos, ()):
                        pos_logits[prev_tok] = float("-inf")
                    x[chosen_pos] = torch.argmax(pos_logits)
            state["seen_input_keys"].add(input_key)

            write_positions = transfer_index.nonzero(as_tuple=False).view(-1)
            for pos in write_positions.tolist():
                state["sampled_history"].setdefault(pos, set()).add(x[pos].item())

            curr_input_ids[transfer_index] = x[transfer_index]

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

    def step_vectorized(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """Host-sync-minimized rewrite of step_original (see module
        docstring and JointThresholdInDel_NPU_rewrite_report.md, this
        project, for the full before/after and capture-bisection writeup).
        Bit-exact vs step_original, verified on CPU (12 cases) and real NPU
        hardware (7 cases)."""
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

            confidence = torch.where(mask_index, p, torch.full_like(p, float("-inf")))
            high_conf = (confidence > self.threshold) & mask_index

            # Batched host fetch: one round trip instead of 3 separate
            # .item() calls (original lines ~121-134).
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
                num_need = 1 + new_mask_count
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
                    # Vectorized scrub: always compute the DELETE/SPLIT-free
                    # argmax and select via torch.where -- no
                    # .any()/.nonzero().tolist() sync.
                    scrub_logits = curr_logits.clone()
                    scrub_logits[:, self.delete_token_id] = float("-inf")
                    scrub_logits[:, self.split_token_id] = float("-inf")
                    scrub_x = torch.argmax(scrub_logits, dim=-1)
                    indel_mask = (x == self.delete_token_id) | (
                        x == self.split_token_id
                    )
                    x = torch.where(indel_mask, scrub_x, x)

            edit_mask = ~mask_index & ~curr_prompt_mask
            edit_transfer_index = (
                (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
            )

            transfer_index = mask_transfer_index | edit_transfer_index

            if not transfer_index.any().item():
                state["finished"] = True
                done[-1] = True
                continue

            pre_write_ids = curr_input_ids.clone()

            # Anti-loop dedup: Python-set membership -- inherently host-side,
            # the primary remaining graph-capture blocker (see report).
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

            # Record written tokens for anti-loop history. NOTE: boolean-mask
            # ASSIGNMENT (tensor[bool_mask] = value) is itself a
            # capture-unsafe host sync on Ascend (lowers to an internal
            # nonzero call) -- found via capture bisection. Use
            # gather/scatter_ instead of boolean-mask assignment.
            selected_col = x.unsqueeze(-1)
            prev_bit = state["sampled_mask"].gather(1, selected_col)
            new_bit = prev_bit | transfer_index.unsqueeze(-1)
            state["sampled_mask"].scatter_(1, selected_col, new_bit)

            # Write tokens -- torch.where instead of boolean-mask assignment.
            curr_input_ids.copy_(torch.where(transfer_index, x, curr_input_ids))

            # Apply DELETE/SPLIT operations. Always run (fixed-shape, no-op
            # when there are no edit tokens) instead of guarding with
            # has_indel.any() -- removes that sync plus the
            # .tolist()/torch.tensor(..., device=device) re-upload pair.
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

