"""Joint-threshold denoising with insertion and deletion support."""

from typing import Any

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

_is_npu = is_npu()
argmax_softmax_prob_fused = None
if _is_npu:
    try:
        from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
    except (ImportError, OSError):
        pass
    from sglang.kernels.ops.llada2.indel_npu import scrub_argmax_fused


def _argmax_prob(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.argmax(logits, dim=-1)
    probabilities = torch.gather(
        torch.softmax(logits, dim=-1), -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_ids, probabilities


def _record_sampled(sampled_mask, token_ids, selected):
    selected_col = token_ids.unsqueeze(-1)
    sampled_mask.scatter_(
        -1,
        selected_col,
        sampled_mask.gather(-1, selected_col) | selected.unsqueeze(-1),
    )


def _init_graph_state(
    input_ids: torch.Tensor,
    *,
    mask_id: int,
    vocab_size: int,
    max_steps: int,
):
    batch_size, block_size = input_ids.shape
    return {
        "prompt_mask": (input_ids != mask_id).cumprod(dim=1).bool(),
        "is_orig_mask": input_ids == mask_id,
        "sampled_mask": torch.zeros(
            (batch_size, block_size, vocab_size),
            dtype=torch.bool,
            device=input_ids.device,
        ),
        "input_history": torch.full(
            (batch_size, max_steps, block_size),
            -1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        ),
        "history_pos": torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        ),
        "post_edit_steps": torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        ),
        "num_update_steps": torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        ),
        "persist_pending": torch.zeros(
            batch_size, dtype=torch.bool, device=input_ids.device
        ),
        "active": torch.ones(batch_size, dtype=torch.bool, device=input_ids.device),
    }


def _graph_step(
    input_ids: torch.Tensor,
    full_logits: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    mask_id: int,
    delete_token_id: int,
    split_token_id: int,
    threshold: float,
    edit_threshold: float,
    max_post_edit_steps: int,
    max_regular_update_steps: int,
):
    batch_size, block_size = input_ids.shape
    vocab_size = full_logits.shape[-1]
    device = input_ids.device

    if argmax_softmax_prob_fused is not None and full_logits.device.type == "npu":
        flat_logits = full_logits.view(-1, vocab_size)
        x_flat, p_flat = argmax_softmax_prob_fused(flat_logits)
        fallback_flat, scrub_flat = scrub_argmax_fused(
            flat_logits,
            mask_id,
            delete_token_id,
            split_token_id,
        )
        x = x_flat.view(batch_size, block_size)
        p = p_flat.view(batch_size, block_size)
        fallback_x = fallback_flat.view(batch_size, block_size)
        scrub_x = scrub_flat.view(batch_size, block_size)
    else:
        _, top_ids = torch.topk(full_logits, k=4, dim=-1)
        x, p = _argmax_prob(full_logits)
        fallback_x = torch.where(
            top_ids[..., 0] == mask_id,
            top_ids[..., 1],
            top_ids[..., 0],
        )
        valid_scrub = (
            (top_ids != mask_id)
            & (top_ids != delete_token_id)
            & (top_ids != split_token_id)
        )
        scrub_rank = valid_scrub.to(torch.int32).argmax(dim=-1, keepdim=True)
        scrub_x = top_ids.gather(-1, scrub_rank).squeeze(-1)

    mask_index = input_ids == mask_id
    is_orig_mask = state["is_orig_mask"]
    original_mask_index = is_orig_mask & mask_index
    needs_fallback = original_mask_index & (x == mask_id)
    original_logits = full_logits.gather(-1, x.unsqueeze(-1)).squeeze(-1)
    fallback_logits = full_logits.gather(-1, fallback_x.unsqueeze(-1)).squeeze(-1)
    fallback_p = p * torch.exp(fallback_logits.float() - original_logits.float())
    x = torch.where(needs_fallback, fallback_x, x)
    p = torch.where(needs_fallback, fallback_p.to(p.dtype), p)
    confidence = torch.where(mask_index, p, torch.full_like(p, float("-inf")))
    high_conf = (confidence > threshold) & mask_index

    original_mask_count = original_mask_index.sum(dim=1)
    total_mask = mask_index.sum(dim=1)
    high_conf_count = high_conf.sum(dim=1)
    new_mask_count = total_mask - original_mask_count
    num_need = 1 + new_mask_count
    select_count = torch.minimum(num_need, total_mask)

    # A full block_size top-k has static shape.  Convert ordering to ranks so
    # each request can select its tensor-valued select_count without dynamic k.
    sorted_index = torch.topk(confidence, k=block_size, dim=1).indices
    rank = torch.empty_like(sorted_index)
    rank.scatter_(
        1,
        sorted_index,
        torch.arange(block_size, device=device)
        .view(1, block_size)
        .expand(batch_size, block_size),
    )
    fallback_transfer = mask_index & (rank < select_count.unsqueeze(1))
    mask_transfer = torch.where(
        (original_mask_count > 0).unsqueeze(1),
        torch.where(
            (high_conf_count >= num_need).unsqueeze(1),
            high_conf,
            fallback_transfer,
        ),
        mask_index,
    )

    active = state["active"]
    persist_pending = state["persist_pending"]
    finish_after_persist = active & persist_pending
    work_active = active & ~persist_pending
    no_original_mask = original_mask_count == 0
    state["num_update_steps"].add_(work_active.to(state["num_update_steps"].dtype))
    state["post_edit_steps"].add_(
        (work_active & no_original_mask).to(state["post_edit_steps"].dtype)
    )
    post_edit_limit_reached = (
        (max_post_edit_steps > 0)
        & no_original_mask
        & (state["post_edit_steps"] >= max_post_edit_steps)
    )
    force_finish = work_active & (
        post_edit_limit_reached
        | ((max_post_edit_steps == 0) & no_original_mask)
        | (state["num_update_steps"] > max_regular_update_steps)
    )
    if max_post_edit_steps == 0:
        all_original_selected = (original_mask_index & ~mask_transfer).to(
            torch.int32
        ).sum(dim=1) == 0
        force_finish |= work_active & (original_mask_count > 0) & all_original_selected
    mask_transfer = torch.where(
        force_finish.unsqueeze(1),
        mask_index & ~state["prompt_mask"],
        mask_transfer,
    )
    forbidden_prediction = (
        (x == mask_id) | (x == delete_token_id) | (x == split_token_id)
    )
    x = torch.where(force_finish.unsqueeze(1) & forbidden_prediction, scrub_x, x)

    edit_mask = ~mask_index & ~state["prompt_mask"]
    edit_transfer = (p > edit_threshold) & (input_ids != x) & edit_mask
    transfer = mask_transfer | edit_transfer
    has_transfer = transfer.to(torch.int32).sum(dim=1) > 0
    do_update = work_active & has_transfer

    pre_write_ids = input_ids.clone()
    repeated = (
        (state["input_history"] == pre_write_ids.unsqueeze(1))
        .to(torch.int32)
        .sum(dim=2)
        == block_size
    ).to(torch.int32).sum(dim=1) > 0

    history_rows = (
        torch.arange(state["input_history"].shape[1], device=device).view(1, -1)
        == state["history_pos"].unsqueeze(1)
    ) & do_update.unsqueeze(1)
    state["input_history"].copy_(
        torch.where(
            history_rows.unsqueeze(2),
            pre_write_ids.unsqueeze(1),
            state["input_history"],
        )
    )
    state["history_pos"].add_(do_update.to(state["history_pos"].dtype))

    changing = transfer & (x != input_ids)
    chosen_pos = torch.where(changing, p, torch.full_like(p, float("inf"))).argmin(
        dim=1
    )
    batch_index = torch.arange(batch_size, device=device)
    chosen_logits = full_logits[batch_index, chosen_pos]
    chosen_sampled = state["sampled_mask"][batch_index, chosen_pos]
    chosen_original_mask = is_orig_mask[batch_index, chosen_pos]
    token_ids = torch.arange(vocab_size, device=device).view(1, vocab_size)
    alternative = chosen_logits.masked_fill(
        chosen_sampled | (chosen_original_mask.unsqueeze(1) & (token_ids == mask_id)),
        float("-inf"),
    ).argmax(dim=1)
    choose_alternative = (
        do_update & repeated & (changing.to(torch.int32).sum(dim=1) > 0) & ~force_finish
    )
    chosen_mask = torch.arange(block_size, device=device).view(
        1, block_size
    ) == chosen_pos.unsqueeze(1)
    x = torch.where(
        choose_alternative.unsqueeze(1) & chosen_mask,
        alternative.unsqueeze(1),
        x,
    )

    _record_sampled(state["sampled_mask"], x, transfer & do_update.unsqueeze(1))

    written_ids = torch.where(transfer & do_update.unsqueeze(1), x, input_ids)

    is_delete = written_ids == delete_token_id
    is_split = written_ids == split_token_id
    counts = torch.where(
        is_delete,
        torch.zeros_like(written_ids),
        torch.where(
            is_split,
            torch.full_like(written_ids, 2),
            torch.ones_like(written_ids),
        ),
    )
    dest_start = torch.cumsum(counts, dim=1) - counts
    # The matching matrix gives every valid destination one producer, avoiding
    # unordered scatter collisions during graph capture.
    destination = torch.arange(block_size, device=device).view(1, 1, block_size)
    source_match = (destination >= dest_start.unsqueeze(2)) & (
        destination < (dest_start + counts).unsqueeze(2)
    )
    source_index = source_match.to(torch.int32).argmax(dim=1)
    source_written = written_ids.gather(1, source_index)
    source_pre_write = pre_write_ids.gather(1, source_index)
    source_is_split = is_split.gather(1, source_index)
    source_dest_start = dest_start.gather(1, source_index)
    source_count = counts.gather(1, source_index)
    source_tracking = is_orig_mask.gather(1, source_index)
    flat_destination = destination.squeeze(1)
    has_emission = (flat_destination >= source_dest_start) & (
        flat_destination < source_dest_start + source_count
    )
    is_second_emission = source_is_split & (flat_destination == source_dest_start + 1)
    gathered_ids = torch.where(
        is_second_emission,
        source_pre_write,
        torch.where(
            source_is_split,
            torch.full_like(source_written, mask_id),
            source_written,
        ),
    )
    edited_ids = torch.where(
        has_emission,
        gathered_ids,
        torch.full_like(gathered_ids, mask_id),
    )
    edited_tracking = has_emission & torch.where(
        source_is_split,
        is_second_emission & source_tracking,
        source_tracking,
    )
    next_input_ids = torch.where(do_update.unsqueeze(1), edited_ids, input_ids)
    changed = (next_input_ids != input_ids).to(torch.int32).sum(dim=1) > 0
    input_ids.copy_(next_input_ids)
    state["is_orig_mask"].copy_(
        torch.where(
            do_update.unsqueeze(1),
            edited_tracking,
            state["is_orig_mask"],
        )
    )

    persist_next = force_finish & changed
    state["persist_pending"].copy_(persist_next)
    finished = work_active & (~has_transfer | (force_finish & ~changed))
    state["active"].logical_and_(~(finish_after_persist | finished))
    done = ~state["active"]
    return done


class _NPUJointThresholdGraphRunner:
    """Capture and replay one fixed-width JointThresholdInDel update.

    The model graph owns the logits buffer.  This runner captures directly
    against that stable buffer and owns a reusable fixed-width input/state
    buffer.  A new generation resets the state tensors without recapturing.
    """

    def __init__(
        self,
        *,
        full_logits: torch.Tensor,
        batch_size: int,
        block_size: int,
        mask_id: int,
        delete_token_id: int,
        split_token_id: int,
        threshold: float,
        edit_threshold: float,
        max_post_edit_steps: int,
        max_regular_update_steps: int,
        max_steps: int,
    ):
        self.full_logits = full_logits.view(
            batch_size, block_size, full_logits.shape[-1]
        )
        self.input_ids = torch.empty(
            (batch_size, block_size),
            dtype=torch.long,
            device=full_logits.device,
        )
        self.state = _init_graph_state(
            self.input_ids,
            mask_id=mask_id,
            vocab_size=full_logits.shape[-1],
            max_steps=max_steps,
        )
        self.graph = torch.npu.NPUGraph()
        self.stream = torch.npu.Stream()

        torch.npu.synchronize()
        with torch.npu.graph(
            self.graph,
            stream=self.stream,
            auto_dispatch_capture=False,
        ):
            self.done = _graph_step(
                self.input_ids,
                self.full_logits,
                self.state,
                mask_id=mask_id,
                delete_token_id=delete_token_id,
                split_token_id=split_token_id,
                threshold=threshold,
                edit_threshold=edit_threshold,
                max_post_edit_steps=max_post_edit_steps,
                max_regular_update_steps=max_regular_update_steps,
            )
        torch.npu.synchronize()

    def matches_logits(self, full_logits: torch.Tensor) -> bool:
        return (
            full_logits.data_ptr() == self.full_logits.data_ptr()
            and full_logits.shape[-1] == self.full_logits.shape[-1]
        )

    def reset(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_mask: torch.Tensor,
        is_orig_mask: torch.Tensor,
    ) -> None:
        self.input_ids.copy_(input_ids.view_as(self.input_ids))
        self.state["prompt_mask"].copy_(prompt_mask)
        self.state["is_orig_mask"].copy_(is_orig_mask)
        self.state["sampled_mask"].zero_()
        self.state["input_history"].fill_(-1)
        self.state["history_pos"].zero_()
        self.state["post_edit_steps"].zero_()
        self.state["num_update_steps"].zero_()
        self.state["persist_pending"].zero_()
        self.state["active"].fill_(True)

    def replay(self) -> list[bool]:
        self.graph.replay()
        return self.done.cpu().tolist()


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
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.max_regular_update_steps = self.block_size + self.max_post_edit_steps
        self.enable_graph = config.algorithm_config.get("enable_graph", False)
        if self.enable_graph and not _is_npu:
            raise ValueError("JointThresholdInDel graph mode requires an NPU device")
        self.delete_token_id = config.delete_token_id
        self.split_token_id = config.split_token_id
        self._npu_graph_runners = {}

    def max_steps(self, block_size: int) -> int:
        return block_size + self.max_post_edit_steps + 2

    def init_step_state(self, forward_batch: ForwardBatch) -> list[Any]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        prompt_lens = (input_ids != self.mask_id).cumprod(dim=1).sum(dim=1)
        positions = torch.arange(self.block_size, device=input_ids.device)
        return [
            {
                "prompt_mask": positions < prompt_lens[i],
                "is_orig_mask": input_ids[i] == self.mask_id,
                "post_edit_steps": 0,
                "num_update_steps": 0,
                "persist_pending": False,
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
        states: list[Any],
    ) -> list[bool]:
        if self.enable_graph:
            return self.step_graph(forward_batch, full_logits, states)
        return self.step_vectorized(forward_batch, full_logits, states)

    def step_graph(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: list[Any],
    ) -> list[bool]:
        """Replay the complete fixed-width InDel update as an NPUGraph."""
        if self.fdfo:
            # FDFO carries per-request state across changing batch
            # compositions; keep its established path until graph-state
            # packing is implemented.
            return self.step_vectorized(forward_batch, full_logits, states)

        batch_size = forward_batch.batch_size
        runner = states[0].get("_npu_graph_runner")
        if runner is None:
            cache_key = (
                batch_size,
                full_logits.shape[-1],
                full_logits.data_ptr(),
            )
            runner = self._npu_graph_runners.get(cache_key)
            if runner is None:
                runner = _NPUJointThresholdGraphRunner(
                    full_logits=full_logits,
                    batch_size=batch_size,
                    block_size=self.block_size,
                    mask_id=self.mask_id,
                    delete_token_id=self.delete_token_id,
                    split_token_id=self.split_token_id,
                    threshold=self.threshold,
                    edit_threshold=self.edit_threshold,
                    max_post_edit_steps=self.max_post_edit_steps,
                    max_regular_update_steps=self.max_regular_update_steps,
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
        states: list[Any],
    ) -> list[bool]:
        """Run the fixed-shape implementation without graph capture."""
        done: list[bool] = []

        for i in range(forward_batch.batch_size):
            state = states[i]
            if state["finished"]:
                done.append(True)
                continue
            if state["persist_pending"]:
                state["persist_pending"] = False
                state["finished"] = True
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
                    device=curr_logits.device,
                )

            x, p = _argmax_prob(curr_logits)

            mask_index = curr_input_ids == self.mask_id
            is_orig_mask = state["is_orig_mask"]
            original_mask_index = is_orig_mask & mask_index
            needs_fallback = original_mask_index & (x == self.mask_id)
            if needs_fallback.any().item():
                fallback_logits = curr_logits[needs_fallback].clone()
                fallback_logits[:, self.mask_id] = float("-inf")
                fallback_x = torch.argmax(fallback_logits, dim=-1)
                x[needs_fallback] = fallback_x
                p[needs_fallback] = (
                    F.softmax(curr_logits[needs_fallback], dim=-1)
                    .gather(1, fallback_x.unsqueeze(1))
                    .squeeze(1)
                )
            confidence = torch.where(mask_index, p, torch.full_like(p, float("-inf")))
            high_conf = (confidence > self.threshold) & mask_index

            # Fetch the control-flow counters in one host transfer.
            original_mask_count, total_mask, high_conf_count = torch.stack(
                [
                    original_mask_index.sum(),
                    mask_index.sum(),
                    high_conf.sum(),
                ]
            ).tolist()
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

            state["num_update_steps"] += 1
            if original_mask_count == 0:
                state["post_edit_steps"] += 1

            post_edit_limit_reached = (
                self.max_post_edit_steps > 0
                and state["post_edit_steps"] >= self.max_post_edit_steps
            )
            force_finish = (
                post_edit_limit_reached
                or (self.max_post_edit_steps == 0 and original_mask_count == 0)
                or state["num_update_steps"] > self.max_regular_update_steps
            )
            if (
                self.max_post_edit_steps == 0
                and original_mask_count > 0
                and not (original_mask_index & ~mask_transfer_index).any().item()
            ):
                force_finish = True

            if force_finish:
                mask_transfer_index = mask_index & ~curr_prompt_mask
                scrub_logits = curr_logits.clone()
                scrub_logits[:, self.mask_id] = float("-inf")
                scrub_logits[:, self.delete_token_id] = float("-inf")
                scrub_logits[:, self.split_token_id] = float("-inf")
                scrub_x = torch.argmax(scrub_logits, dim=-1)
                forbidden_mask = (
                    (x == self.mask_id)
                    | (x == self.delete_token_id)
                    | (x == self.split_token_id)
                )
                x = torch.where(forbidden_mask, scrub_x, x)

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

            # The non-graph path keeps anti-loop history on the host.
            input_key = tuple(pre_write_ids.tolist())

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
                    if original_mask_index[chosen_pos]:
                        pos_logits[self.mask_id] = float("-inf")
                    new_tok = torch.argmax(pos_logits)
                    x[chosen_pos] = new_tok
            state["seen_input_keys"].add(input_key)

            # Scatter keeps the update fixed-shape.
            _record_sampled(state["sampled_mask"], x, transfer_index)

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
                if torch.equal(edited_tokens, pre_write_ids):
                    state["finished"] = True
                    done[-1] = True
                else:
                    state["persist_pending"] = True

        return done


Algorithm = JointThresholdInDel
