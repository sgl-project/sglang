from typing import Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
    hip_unified_kv_triton_enabled,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    BuildCommitInjectLayout,
    build_unified_commit_inject_layout,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if state_slot is not None:
            state_slot = state_slot.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if final_pos is not None:
            final_pos = final_pos.to(
                device=device, dtype=torch.int64, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
            return

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> None:
        if hip_unified_kv_triton_enabled():
            swa_loc = self._unified_inject_loc(
                pool=pool,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        else:
            swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
            if commit_lens is not None and cache_loc_2d is not None:
                bs, verify_len = cache_loc_2d.shape
                col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
                committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(
                    -1
                )
                swa_loc = torch.where(
                    committed_mask, swa_loc, torch.full_like(swa_loc, -1)
                )

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def _unified_inject_loc(
        self,
        *,
        pool,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor],
        final_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring row for target-hidden injection under unified_kv.

        loc = state_slot * ring + pos % ring, with two skip (-1) rules:
          * SWA window: only the last ``win`` tokens per req land in the ring;
            older tokens share a ring slot (pos % ring) and would race, so drop
            them (needed for long prefill chunks).
          * commit gate: uncommitted verify tokens (col >= commit_len) are dropped.
        """
        if state_slot is None:
            raise RuntimeError(
                "unified_kv target-hidden injection requires state_slot "
                "(per-token draft req_pool_indices)."
            )
        ring = pool.unified_swa_ring_size
        win = pool.unified_swa_window
        pos = positions.to(torch.int64)
        loc = state_slot.to(torch.int64) * ring + pos % ring
        if final_pos is not None:
            keep = pos > (final_pos.to(torch.int64) - win)
            loc = torch.where(keep, loc, torch.full_like(loc, -1))
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=positions.device).view(1, -1)
            committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            loc = torch.where(committed, loc, torch.full_like(loc, -1))
        return loc.to(torch.int32)

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            if hip_unified_kv_triton_enabled():
                inject_layout = build_unified_commit_inject_layout(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    commit_lens=commit_lens,
                    stride=stride,
                    ring_stride=pool.unified_swa_ring_size,
                )
            else:
                inject_layout = BuildCommitInjectLayout.execute(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    full_to_swa_mapping=pool.full_to_swa_index_mapping,
                    commit_lens=commit_lens,
                    stride=stride,
                )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                )
            return

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + stride,
            batch_size=bs,
            draft_token_num=stride,
            device=self.device,
        )
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
