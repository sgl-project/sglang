"""Request-owned target-state lifecycle for DVR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclass(frozen=True, slots=True)
class DVRRollbackPlan:
    request_rows: torch.Tensor
    target_cache_slots: torch.Tensor
    tail_lens: torch.Tensor


class DVRStateLifecycle:
    """Own target boundaries and private self-draft state.

    Recurrent state is stored at the latest exact chunk boundary, while
    convolution state is stored at the accepted endpoint. DVR replays the short
    accepted transition tail between them.

    Radix publication is chunk-aligned. While a request is active, its accepted
    non-aligned tail remains in the request-owned transition window.
    On cache release, only tokens, KV, and recurrent state through the latest exact
    chunk boundary are donated to Radix; a later cache hit prefills that tail again.
    A boundary computed by verify first remains pending because CPU stop handling may
    trim that block. It is staged before the next verify, or at release when visible.
    Radix ping-pong slots never serve as verify inputs.
    """

    def __init__(self, *, server_args, model_runner):
        self.server_args = server_args
        self.model_runner = model_runner
        self.state_adapter = None
        self.target_boundary_lens = None
        self.radix_boundary_lens = None
        self.radix_boundary_slots = None
        self.pending_boundary_conv_steps = None

    def bind_state_adapter(self, state_adapter) -> None:
        self.state_adapter = state_adapter
        if state_adapter is None:
            if (
                getattr(self.model_runner.req_to_token_pool, "mamba_pool", None)
                is not None
            ):
                raise RuntimeError(
                    "DVR does not support this hybrid linear-state backend: no "
                    "target state adapter was initialized."
                )
            return

        if self.server_args.mamba_track_interval != state_adapter.chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_track_interval to match "
                f"the adapter chunk size {state_adapter.chunk_size}, got "
                f"{self.server_args.mamba_track_interval}."
            )
        if self.server_args.mamba_cache_chunk_size != state_adapter.chunk_size:
            raise ValueError(
                "DVR linear-state verify requires mamba_cache_chunk_size to "
                f"match the adapter chunk size {state_adapter.chunk_size}, got "
                f"{self.server_args.mamba_cache_chunk_size}."
            )

        request_capacity = state_adapter.recurrent_workspace.shape[1]
        device = state_adapter.recurrent_workspace.device
        self.target_boundary_lens = torch.full(
            (request_capacity,), -1, dtype=torch.int64, device=device
        )
        if not self.server_args.disable_radix_cache:
            self.pending_boundary_conv_steps = torch.full(
                (request_capacity,), -1, dtype=torch.int64, device=device
            )
            track_count = int(
                self.model_runner.req_to_token_pool.mamba_ping_pong_track_buffer_size
            )
            self.radix_boundary_lens = torch.full(
                (request_capacity, track_count),
                -1,
                dtype=torch.int64,
                device=device,
            )
            self.radix_boundary_slots = torch.full_like(self.radix_boundary_lens, -1)

    @property
    def chunk_size(self) -> int:
        if self.state_adapter is None:
            raise RuntimeError("DVR linear-state adapter is not initialized.")
        return self.state_adapter.chunk_size

    def clear_cache_state(self) -> None:
        if self.target_boundary_lens is not None:
            self.target_boundary_lens.fill_(-1)
        if self.radix_boundary_lens is not None:
            self.radix_boundary_lens.fill_(-1)
            self.radix_boundary_slots.fill_(-1)
        if self.pending_boundary_conv_steps is not None:
            self.pending_boundary_conv_steps.fill_(-1)

    def prepare_target_extend(self, batch: ScheduleBatch) -> None:
        if self.state_adapter is None or not batch.reqs:
            return
        if any(int(prefix_len) % self.chunk_size for prefix_len in batch.prefix_lens):
            raise ValueError(
                "DVR GDN target EXTEND must start from an exact chunk boundary."
            )
        request_rows = batch.req_pool_indices.to(
            device=self.target_boundary_lens.device, dtype=torch.long
        )
        self.target_boundary_lens.index_fill_(0, request_rows, -1)
        if self.radix_boundary_lens is not None:
            self.pending_boundary_conv_steps.index_fill_(0, request_rows, -1)
            self.radix_boundary_lens.index_fill_(0, request_rows, -1)
            self.radix_boundary_slots.index_fill_(0, request_rows, -1)

    def _invalidate_rebound_slots(
        self, request_rows: torch.Tensor, track_slots: torch.Tensor
    ) -> None:
        stored_slots = self.radix_boundary_slots[request_rows]
        valid = (track_slots >= 0) & stored_slots.eq(track_slots)
        self.radix_boundary_lens[request_rows] = torch.where(
            valid,
            self.radix_boundary_lens[request_rows],
            torch.full_like(self.radix_boundary_lens[request_rows], -1),
        )
        self.radix_boundary_slots[request_rows] = torch.where(
            valid, stored_slots, torch.full_like(stored_slots, -1)
        )

    def _stage_pending_boundaries(
        self,
        *,
        request_rows: torch.Tensor,
        target_cache_slots: torch.Tensor,
        track_slots: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.radix_boundary_lens is None:
            return

        self._invalidate_rebound_slots(request_rows, track_slots)
        pending_steps = self.pending_boundary_conv_steps[request_rows]
        stage_mask = pending_steps >= 0
        if eligible_mask is not None:
            stage_mask &= eligible_mask

        available_lanes = track_slots >= 0
        lane_scores = torch.where(
            available_lanes,
            self.radix_boundary_lens[request_rows],
            torch.full_like(
                self.radix_boundary_lens[request_rows], torch.iinfo(torch.int64).max
            ),
        )
        stage_lanes = lane_scores.argmin(dim=1)
        has_destination = available_lanes.any(dim=1)
        stage_mask &= has_destination
        destination_slots = track_slots.gather(1, stage_lanes.unsqueeze(1)).squeeze(1)
        source_steps = torch.where(
            stage_mask, pending_steps, torch.full_like(pending_steps, -1)
        )
        self.state_adapter.stage_boundary_state(
            request_rows=request_rows,
            source_slots=target_cache_slots,
            destination_slots=destination_slots,
            boundary_conv_steps=source_steps,
        )

        old_lens = self.radix_boundary_lens[request_rows, stage_lanes]
        old_slots = self.radix_boundary_slots[request_rows, stage_lanes]
        self.radix_boundary_lens[request_rows, stage_lanes] = torch.where(
            stage_mask, self.target_boundary_lens[request_rows], old_lens
        )
        self.radix_boundary_slots[request_rows, stage_lanes] = torch.where(
            stage_mask, destination_slots, old_slots
        )
        self.pending_boundary_conv_steps[request_rows] = torch.where(
            stage_mask, torch.full_like(pending_steps, -1), pending_steps
        )

    def prepare_for_cache_release(self, req) -> None:
        if self.state_adapter is None or req.req_pool_idx is None:
            return
        request_row = int(req.req_pool_idx)
        try:
            if self.radix_boundary_lens is None or req.skip_radix_cache_insert:
                return

            committed_len = req.effective_kv_committed_len()
            if req.finished_reason is not None:
                visible_kv_len = len(req.origin_input_ids) + len(
                    req.output_ids_through_stop
                )
                if req.sampling_params.max_new_tokens != 0:
                    # The final sampled token has not entered the KV cache yet.
                    visible_kv_len -= 1
                committed_len = min(committed_len, max(visible_kv_len, 0))
            # The Radix key and its recurrent checkpoint must describe the same
            # prefix. Do not donate the request-owned transition tail.
            publish_len = committed_len // self.chunk_size * self.chunk_size
            device = self.target_boundary_lens.device
            request_rows = torch.tensor([request_row], device=device, dtype=torch.long)
            target_cache_slots = req.mamba_pool_idx.to(
                device=device, dtype=torch.long
            ).reshape(1)
            track_slots = req.mamba_ping_pong_track_buffer.to(
                device=device, dtype=torch.long
            ).reshape(1, -1)
            self._stage_pending_boundaries(
                request_rows=request_rows,
                target_cache_slots=target_cache_slots,
                track_slots=track_slots,
                eligible_mask=(
                    (self.target_boundary_lens[request_rows] > 0)
                    & (self.target_boundary_lens[request_rows] <= publish_len)
                ),
            )
            candidates = [
                (length, lane)
                for lane, length in enumerate(
                    self.radix_boundary_lens[request_row].tolist()
                )
                if 0 < length <= publish_len
            ]
            if not candidates:
                # There is no newer complete chunk to add. An existing warm
                # prefix remains cached; an uncached partial tail is discarded.
                req.skip_radix_cache_insert = True
                return

            checkpoint_len, checkpoint_lane = max(candidates)
            req.mamba_last_track_seqlen = checkpoint_len
            if self.radix_boundary_lens.shape[1] == 2:
                req.mamba_next_track_idx = 1 - checkpoint_lane
            else:
                req.mamba_next_track_idx = checkpoint_lane
        finally:
            self.target_boundary_lens[request_row] = -1
            if self.radix_boundary_lens is not None:
                self.pending_boundary_conv_steps[request_row] = -1
                self.radix_boundary_lens[request_row].fill_(-1)
                self.radix_boundary_slots[request_row].fill_(-1)

    def prepare_rollback(self, batch) -> Optional[DVRRollbackPlan]:
        if self.state_adapter is None or batch.batch_size() == 0:
            return None

        request_rows, target_cache_slots = self.state_adapter.resolve_request_slots(
            batch=batch
        )
        tail_lens = batch.seq_lens.remainder(self.chunk_size).to(torch.long)
        expected_boundaries = batch.seq_lens - tail_lens
        torch._assert_async(
            self.target_boundary_lens[request_rows].eq(expected_boundaries).all(),
            "DVR draft started without the latest exact recurrent boundary.",
        )

        if self.radix_boundary_lens is not None:
            pool = batch.req_to_token_pool
            track_slot_mapping = pool.req_index_to_mamba_ping_pong_track_buffer_mapping
            track_slots = track_slot_mapping[batch.req_pool_indices].to(
                device=request_rows.device, dtype=torch.long
            )
            self._stage_pending_boundaries(
                request_rows=request_rows,
                target_cache_slots=target_cache_slots,
                track_slots=track_slots,
            )

        return DVRRollbackPlan(
            request_rows=request_rows,
            target_cache_slots=target_cache_slots,
            tail_lens=tail_lens,
        )

    def finish_target_extend(self, batch: ScheduleBatch) -> None:
        """Record the live boundary produced by target EXTEND."""

        if self.state_adapter is None or batch.batch_size() == 0:
            return
        request_rows, target_cache_slots = self.state_adapter.resolve_request_slots(
            batch=batch
        )
        seq_lens = batch.seq_lens.to(
            device=target_cache_slots.device, dtype=torch.int64
        )
        prefix_lens = torch.tensor(
            batch.prefix_lens, device=target_cache_slots.device, dtype=torch.int64
        )
        boundary_lens = seq_lens // self.chunk_size * self.chunk_size
        zero_mask = boundary_lens == 0
        self.state_adapter.zero_boundary_state(indices=target_cache_slots[zero_mask])
        self.target_boundary_lens[request_rows] = boundary_lens

        if self.radix_boundary_lens is not None:
            publish_lanes = []
            publish_slots = []
            for req in batch.reqs:
                lane = batch.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                publish_lanes.append(lane)
                publish_slots.append(req.mamba_ping_pong_track_buffer[lane])
            publish_lanes = torch.tensor(
                publish_lanes, device=target_cache_slots.device, dtype=torch.int64
            )
            publish_slots = torch.stack(publish_slots).to(
                device=target_cache_slots.device, dtype=torch.int64
            )
            # Only a boundary created by this EXTEND has matching convolution
            # state in the tracking slot. A warm partial-tail EXTEND reuses the
            # existing Radix node and has nothing new to publish.
            publish_mask = boundary_lens > prefix_lens
            self.state_adapter.publish_boundary_state(
                source_slots=target_cache_slots,
                destination_slots=publish_slots,
                publish_mask=publish_mask,
            )
            rows = request_rows[publish_mask]
            self.radix_boundary_lens[rows, publish_lanes[publish_mask]] = boundary_lens[
                publish_mask
            ]
            self.radix_boundary_slots[rows, publish_lanes[publish_mask]] = (
                publish_slots[publish_mask]
            )

        self.state_adapter.initialize_self_draft_state(
            request_rows=request_rows,
            target_cache_slots=target_cache_slots,
            tail_lens=seq_lens.remainder(self.chunk_size),
        )

    def rollback(
        self,
        *,
        batch: ScheduleBatch,
        plan: Optional[DVRRollbackPlan],
        accept_lens: torch.Tensor,
    ) -> None:
        if plan is None or accept_lens.numel() == 0:
            return

        crosses_boundary, boundary_conv_steps = (
            self.state_adapter.commit_accepted_state(
                request_rows=plan.request_rows,
                target_cache_slots=plan.target_cache_slots,
                tail_lens_before=plan.tail_lens,
                accepted_token_counts=accept_lens.to(torch.long),
            )
        )
        self.target_boundary_lens[plan.request_rows] = torch.where(
            crosses_boundary,
            self.target_boundary_lens[plan.request_rows] + self.chunk_size,
            self.target_boundary_lens[plan.request_rows],
        )
        if self.radix_boundary_lens is not None:
            self.pending_boundary_conv_steps[plan.request_rows] = torch.where(
                crosses_boundary,
                boundary_conv_steps,
                self.pending_boundary_conv_steps[plan.request_rows],
            )
