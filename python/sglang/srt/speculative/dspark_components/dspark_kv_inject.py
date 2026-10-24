from typing import NamedTuple, Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import swa_ring_rows
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
    build_ring_commit_inject_layout,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.kv_loc_plan import Cols, KVLocPlan
from sglang.srt.model_executor.forward_batch_info import compute_position
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.utils import is_pin_memory_available


class PrefillPlacement(NamedTuple):
    """Per-token placement of a prefill's target hidden in the draft pool."""

    cache_loc: torch.Tensor
    positions: torch.Tensor
    state_slot: Optional[torch.Tensor]  # draft req slot, for a SWA ring
    final_pos: Optional[torch.Tensor]  # the request's last position, for a ring
    swa_loc: Optional[torch.Tensor]  # SWA rows, for a pool that takes them

    def select(self, token_indices: torch.Tensor) -> "PrefillPlacement":
        return PrefillPlacement(
            *(None if t is None else t[token_indices] for t in self)
        )


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

    @property
    def swa_ring_size(self) -> Optional[int]:
        """The ring length when the draft's SWA KV is a per-request ring."""
        return getattr(
            self.draft_model_runner.token_to_kv_pool, "swa_req_ring_size", None
        )

    def ids_for(self, plan: KVLocPlan, *, cols: Optional[Cols] = None) -> torch.Tensor:
        """The plan's write ids as the draft pool indexes them: the target's
        physical ids for a fused draft, the virtual ids for a private one."""
        return plan.write_ids(self.draft_model_runner.kv_index_translator, cols=cols)

    def place_prefill(self, batch: ScheduleBatch) -> PrefillPlacement:
        """Where a prefill's target hidden goes in the draft pool. It reads
        scheduler-shared tables, so it runs before the target forward."""
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = self.model_runner.device
        pin_memory = is_pin_memory_available(device)
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        # A SWA ring is keyed by (draft req slot, position); thread the per-token
        # state_slot + the req's final position so the injector keeps only the
        # last SWA window (older prefill tokens share a ring slot and would race).
        state_slot = final_pos = None
        if self.swa_ring_size is not None:
            repeats = ctx_lens.to(torch.int64)
            num_tokens = sum(batch.extend_lens)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64),
                repeats,
                output_size=num_tokens,
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64),
                repeats,
                output_size=num_tokens,
            )
        # The draft KV goes to the slots the target prefill writes.
        cache_loc = self.ids_for(self.model_runner.kv_index_translator.own_plan(batch))
        pool = self.draft_model_runner.token_to_kv_pool
        swa_loc = None
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            swa_loc = self.committed_swa_loc(
                pool=pool,
                cache_loc=cache_loc.to(torch.int64),
                positions=positions.to(torch.int64),
                cache_loc_2d=None,
                commit_lens=None,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        return PrefillPlacement(cache_loc, positions, state_slot, final_pos, swa_loc)

    def inject_prefill(
        self,
        placement: PrefillPlacement,
        *,
        target_hidden: torch.Tensor,
        token_indices: Optional[torch.Tensor],
        target_hidden_is_projected: bool,
    ) -> None:
        if token_indices is not None:
            placement = placement.select(token_indices)
        self._inject(
            target_hidden=target_hidden,
            cache_loc=placement.cache_loc,
            positions=placement.positions,
            state_slot=placement.state_slot,
            final_pos=placement.final_pos,
            target_hidden_is_projected=target_hidden_is_projected,
            swa_loc=placement.swa_loc,
        )

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
        target_hidden_is_projected: bool = False,
    ) -> None:
        self._inject(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            state_slot=state_slot,
            final_pos=final_pos,
            target_hidden_is_projected=target_hidden_is_projected,
        )

    def _inject(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
        swa_loc: Optional[torch.Tensor] = None,
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

        # `cache_loc` arrives in the draft pool's id space (`ids_for`).
        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if target_hidden_is_projected:
                raise RuntimeError(
                    "Pre-gather target-hidden projection is not supported by the "
                    "DSpark MLA KV injection path."
                )
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
                swa_loc=swa_loc,
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
                target_hidden_is_projected=target_hidden_is_projected,
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
        swa_loc: Optional[torch.Tensor] = None,
    ) -> None:
        if swa_loc is None:
            swa_loc = self.committed_swa_loc(
                pool=pool,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def committed_swa_loc(
        self,
        *,
        pool,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The draft SWA rows the committed context tokens store into, -1 elsewhere."""
        if pool.swa_req_ring_size is not None:
            swa_loc = self._ring_inject_loc(
                pool=pool,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        elif (
            cache_loc.is_cuda
            and cache_loc.is_contiguous()
            and commit_lens is not None
            and cache_loc_2d is not None
            and commit_lens.is_contiguous()
            and cache_loc.numel() == cache_loc_2d.numel()
        ):
            from sglang.kernels.ops.speculative.dspark.commit_swa import (
                committed_swa_locations,
            )

            swa_loc = committed_swa_locations(
                cache_loc,
                pool.full_to_swa_index_mapping,
                commit_lens,
                cache_loc_2d.shape[1],
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

        return swa_loc

    def _ring_inject_loc(
        self,
        *,
        pool,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor],
        final_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring row for target-hidden injection into a per-request SWA ring.

        loc = swa_ring_rows(state_slot, pos, ring), with two skip (-1) rules:
          * SWA window: only the last ``win`` tokens per req land in the ring;
            older tokens share a ring slot (pos % ring) and would race, so drop
            them (needed for long prefill chunks).
          * commit gate: uncommitted verify tokens (col >= commit_len) are dropped.
        """
        if state_slot is None:
            raise RuntimeError(
                "SWA ring target-hidden injection requires state_slot "
                "(per-token draft req_pool_indices)."
            )
        ring = pool.swa_req_ring_size
        win = pool.sliding_window
        pos = positions.to(torch.int64)
        loc = swa_ring_rows(state_slot, pos, ring)
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
        kv_loc_plan: KVLocPlan,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            if pool.swa_req_ring_size is not None:
                inject_layout = build_ring_commit_inject_layout(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    commit_lens=commit_lens,
                    stride=stride,
                    ring_stride=pool.swa_req_ring_size,
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
        verify_cache_loc = self.ids_for(kv_loc_plan)
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
