import torch

try:
    from sglang.srt.hardware_backend.npu.norm.argmax_softmax_prob import (
        argmax_softmax_prob_fused,
        scrub_argmax_fused,
    )
except (ImportError, OSError):
    argmax_softmax_prob_fused = None
    scrub_argmax_fused = None


def init_graph_state(
    input_ids: torch.Tensor,
    *,
    mask_id: int,
    vocab_size: int,
    max_steps: int,
):
    batch_size, block_size = input_ids.shape
    history_capacity = 1 << (max_steps - 1).bit_length()
    return {
        "prompt_mask": input_ids != mask_id,
        "is_orig_mask": input_ids == mask_id,
        "sampled_mask": torch.zeros(
            (batch_size, block_size, vocab_size),
            dtype=torch.bool,
            device=input_ids.device,
        ),
        "input_history": torch.empty(
            (batch_size, history_capacity, block_size),
            dtype=input_ids.dtype,
            device=input_ids.device,
        ),
        "input_history_valid": torch.zeros(
            (batch_size, history_capacity),
            dtype=torch.bool,
            device=input_ids.device,
        ),
        "history_pos": torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        ),
        "post_edit_steps": torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        ),
        "active": torch.ones(batch_size, dtype=torch.bool, device=input_ids.device),
    }


def graph_step(
    input_ids: torch.Tensor,
    full_logits: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    mask_id: int,
    delete_token_id: int,
    split_token_id: int,
    threshold: float,
    edit_threshold: float,
    post_edit_threshold: float,
    max_post_edit_steps: int,
    min_mask_transfer: int = 1,
    min_indel_confidence: float = 0.0,
):
    batch_size, block_size = input_ids.shape
    vocab_size = full_logits.shape[-1]
    device = input_ids.device

    if (
        argmax_softmax_prob_fused is not None
        and scrub_argmax_fused is not None
        and full_logits.device.type == "npu"
    ):
        flat_logits = full_logits.view(-1, vocab_size)
        x_flat, p_flat = argmax_softmax_prob_fused(flat_logits)
        scrub_flat = scrub_argmax_fused(
            flat_logits,
            delete_token_id,
            split_token_id,
        )
        x = x_flat.view(batch_size, block_size)
        p = p_flat.view(batch_size, block_size)
        scrub_x = scrub_flat.view(batch_size, block_size)
    else:
        # Top-3 includes a non-edit candidate because only two edit IDs exist.
        _, top_ids = torch.topk(full_logits, k=3, dim=-1)
        x = torch.argmax(full_logits, dim=-1)
        p = torch.gather(
            torch.softmax(full_logits, dim=-1),
            dim=-1,
            index=x.unsqueeze(-1),
        ).squeeze(-1)
        valid_scrub = (top_ids != delete_token_id) & (top_ids != split_token_id)
        scrub_rank = valid_scrub.to(torch.int32).argmax(dim=-1, keepdim=True)
        scrub_x = top_ids.gather(-1, scrub_rank).squeeze(-1)

    mask_index = input_ids == mask_id
    if min_indel_confidence > 0:
        low_conf_indel = (
            mask_index
            & ((x == delete_token_id) | (x == split_token_id))
            & (p <= min_indel_confidence)
        )
        x = torch.where(low_conf_indel, scrub_x, x)
    is_orig_mask = state["is_orig_mask"]
    confidence = torch.where(mask_index, p, torch.full_like(p, float("-inf")))
    high_conf = (confidence > threshold) & mask_index

    original_mask_count = (is_orig_mask & mask_index).sum(dim=1)
    total_mask = mask_index.sum(dim=1)
    high_conf_count = high_conf.sum(dim=1)
    new_mask_count = total_mask - original_mask_count
    num_need = min_mask_transfer + new_mask_count
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
    no_original_mask = original_mask_count == 0
    state["post_edit_steps"].add_(
        (active & no_original_mask).to(state["post_edit_steps"].dtype)
    )
    force_finish = (
        active & no_original_mask & (state["post_edit_steps"] > max_post_edit_steps)
    )
    indel_prediction = (x == delete_token_id) | (x == split_token_id)
    x = torch.where(force_finish.unsqueeze(1) & indel_prediction, scrub_x, x)

    edit_mask = ~mask_index & ~state["prompt_mask"]
    effective_edit_threshold = torch.where(
        no_original_mask.unsqueeze(1),
        torch.full_like(p, post_edit_threshold),
        torch.full_like(p, edit_threshold),
    )
    edit_transfer = (p > effective_edit_threshold) & (input_ids != x) & edit_mask
    transfer = mask_transfer | edit_transfer
    has_transfer = transfer.to(torch.int32).sum(dim=1) > 0
    do_update = active & has_transfer

    pre_write_ids = input_ids.clone()
    repeated_rows = (
        (state["input_history"] == pre_write_ids.unsqueeze(1))
        .to(torch.int32)
        .sum(dim=2)
        == block_size
    ).logical_and(state["input_history_valid"])
    repeated = repeated_rows.to(torch.int32).sum(dim=1) > 0

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
    state["input_history_valid"].logical_or_(history_rows)
    state["history_pos"].add_(do_update.to(state["history_pos"].dtype))

    changing = transfer & (x != input_ids)
    chosen_pos = torch.where(changing, p, torch.full_like(p, float("inf"))).argmin(
        dim=1
    )
    batch_index = torch.arange(batch_size, device=device)
    chosen_logits = full_logits[batch_index, chosen_pos]
    chosen_sampled = state["sampled_mask"][batch_index, chosen_pos]
    alternative = chosen_logits.masked_fill(chosen_sampled, float("-inf")).argmax(dim=1)
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

    selected_col = x.unsqueeze(-1)
    previous_bit = state["sampled_mask"].gather(2, selected_col)
    new_bit = previous_bit | (transfer & do_update.unsqueeze(1)).unsqueeze(-1)
    state["sampled_mask"].scatter_(2, selected_col, new_bit)

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
    destination_values = torch.arange(block_size, device=device)
    compaction_size = destination_values.shape[0]
    destination = destination_values.view(1, 1, compaction_size)
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
    has_emission_full = (flat_destination >= source_dest_start) & (
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
    edited_ids_full = torch.where(
        has_emission_full,
        gathered_ids,
        torch.full_like(gathered_ids, mask_id),
    )
    edited_tracking_full = has_emission_full & torch.where(
        source_is_split,
        is_second_emission & source_tracking,
        source_tracking,
    )
    has_emission = has_emission_full
    edited_ids = edited_ids_full
    edited_tracking = edited_tracking_full
    next_input_ids = torch.where(do_update.unsqueeze(1), edited_ids, input_ids)
    input_ids.copy_(next_input_ids)
    state["is_orig_mask"].copy_(
        torch.where(
            do_update.unsqueeze(1),
            edited_tracking,
            state["is_orig_mask"],
        )
    )

    finished = active & (~has_transfer | force_finish)
    state["active"].logical_and_(~finished)
    done = ~state["active"]
    return done


class NPUJointThresholdGraphRunner:
    """Capture and replay one fixed-width JointThresholdInDel update.

    The model graph owns the logits buffer.  This runner captures directly
    against that stable buffer and owns a reusable 32-token input/state
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
        post_edit_threshold: float,
        max_post_edit_steps: int,
        min_mask_transfer: int,
        min_indel_confidence: float,
        done_check_interval: int,
        max_steps: int,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.mask_id = mask_id
        self.delete_token_id = delete_token_id
        self.split_token_id = split_token_id
        self.threshold = threshold
        self.edit_threshold = edit_threshold
        self.post_edit_threshold = post_edit_threshold
        self.max_post_edit_steps = max_post_edit_steps
        self.min_mask_transfer = min_mask_transfer
        self.min_indel_confidence = min_indel_confidence
        self.done_check_interval = done_check_interval
        self.max_steps = max_steps
        self.full_logits = full_logits.view(
            batch_size, block_size, full_logits.shape[-1]
        )
        self.logits_data_ptr = self.full_logits.data_ptr()
        self.input_ids = torch.empty(
            (batch_size, block_size),
            dtype=torch.long,
            device=full_logits.device,
        )
        self.state = init_graph_state(
            self.input_ids,
            mask_id=mask_id,
            vocab_size=full_logits.shape[-1],
            max_steps=max_steps,
        )
        self.graph = torch.npu.NPUGraph()
        self.stream = torch.npu.Stream()
        self.replay_count = 0

        torch.npu.synchronize()
        with torch.npu.graph(
            self.graph,
            stream=self.stream,
            auto_dispatch_capture=False,
        ):
            self.done = graph_step(
                self.input_ids,
                self.full_logits,
                self.state,
                mask_id=self.mask_id,
                delete_token_id=self.delete_token_id,
                split_token_id=self.split_token_id,
                threshold=self.threshold,
                edit_threshold=self.edit_threshold,
                post_edit_threshold=self.post_edit_threshold,
                max_post_edit_steps=self.max_post_edit_steps,
                min_mask_transfer=self.min_mask_transfer,
                min_indel_confidence=self.min_indel_confidence,
            )
        torch.npu.synchronize()

    def matches_logits(self, full_logits: torch.Tensor) -> bool:
        return (
            full_logits.data_ptr() == self.logits_data_ptr
            and full_logits.shape[-1] == self.full_logits.shape[-1]
        )

    def reset(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_mask: torch.Tensor,
        is_orig_mask: torch.Tensor,
    ) -> None:
        self.input_ids.copy_(input_ids.view(self.batch_size, self.block_size))
        self.state["prompt_mask"].copy_(prompt_mask)
        self.state["is_orig_mask"].copy_(is_orig_mask)
        self.state["sampled_mask"].zero_()
        self.state["input_history_valid"].zero_()
        self.state["history_pos"].zero_()
        self.state["post_edit_steps"].zero_()
        self.state["active"].fill_(True)
        self.replay_count = 0

    def replay(self) -> list[bool]:
        self.graph.replay()
        self.replay_count += 1
        if self.replay_count % self.done_check_interval:
            return [False] * self.batch_size
        # The synchronous algorithm loop needs one completion decision per
        # check interval.  Finished rows are inactive in the captured graph,
        # so delaying the check preserves their state while allowing the host
        # to enqueue subsequent model/algorithm replays without a D2H sync.
        return self.done.cpu().tolist()
