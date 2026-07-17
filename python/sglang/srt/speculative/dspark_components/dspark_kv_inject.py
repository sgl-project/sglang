from typing import Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    BuildCommitInjectLayout,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
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
        # Static staging buffers for the draft-graph inject prologue
        # (allocated at first graph capture, sized to the max capture bs).
        self._inject_bufs: Optional[dict] = None
        self._inject_max_bs = 0
        self._inject_pending = False
        self._inject_staged_bs = 0

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
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

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
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
    ) -> None:
        swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
            committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            swa_loc = torch.where(committed_mask, swa_loc, torch.full_like(swa_loc, -1))

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def _dense_pool(self):
        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            return None
        return pool

    def capture_prologue_hook(self, runner, forward_batch, num_tokens) -> None:
        """Recorded at the head of the draft TARGET_VERIFY graph: commit the
        staged target-hidden KV before the block-draft layers read it.
        Un-staged replays (idle participation / static verify) see zeroed
        commit_lens rows, which make every KV write a no-op."""
        if read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT:
            return
        if self._dense_pool() is None:
            return
        if runner.capture_forward_mode != ForwardMode.TARGET_VERIFY:
            return
        if self._inject_bufs is None:
            max_bs = int(runner.capture_bs[-1])
            # The staged hidden is the concatenated multi-layer context
            # features, not a single hidden vector.
            hidden_size = (
                self.draft_model.num_context_features
                * self.model_runner.model_config.hidden_size
            )
            dev = self.device
            self._inject_max_bs = max_bs
            self._inject_bufs = {
                "seq_lens": torch.zeros(max_bs, dtype=torch.int64, device=dev),
                "req_pool_indices": torch.zeros(max_bs, dtype=torch.int64, device=dev),
                "hidden": torch.zeros(
                    max_bs * self.verify_num_draft_tokens,
                    hidden_size,
                    dtype=self.model_runner.dtype,
                    device=dev,
                ),
                "commit_lens": torch.zeros(max_bs, dtype=torch.int32, device=dev),
            }
        bs = min(int(forward_batch.batch_size), self._inject_max_bs)
        self._inject_prologue_body(bs)

    def _inject_prologue_body(self, bs: int) -> None:
        stride = self.verify_num_draft_tokens
        bufs = self._inject_bufs
        prefix_lens = bufs["seq_lens"][:bs]
        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=bufs["req_pool_indices"][:bs],
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + stride,
            batch_size=bs,
            draft_token_num=stride,
            device=self.device,
        )
        self.inject_target_hidden(
            target_hidden=bufs["hidden"][: bs * stride],
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc.view(bs, stride),
            positions=positions_2d.reshape(-1),
            commit_lens=bufs["commit_lens"][:bs],
        )

    def stage_ragged(
        self,
        *,
        batch: ScheduleBatch,
        hidden_strided: Optional[torch.Tensor],
        commit_lens: torch.Tensor,
        bs: int,
    ) -> bool:
        """Stage the ragged inject inputs for the draft-graph prologue.
        Returns False when the caller must run the eager inject instead."""
        if self._inject_bufs is None or self._dense_pool() is None:
            return False
        if hidden_strided is None or hidden_strided.numel() == 0:
            return True
        if bs > self._inject_max_bs:
            return False
        stride = self.verify_num_draft_tokens
        bufs = self._inject_bufs
        bufs["seq_lens"][bs:].zero_()
        bufs["req_pool_indices"][bs:].zero_()
        bufs["commit_lens"][bs:].zero_()
        bufs["seq_lens"][:bs].copy_(batch.seq_lens[:bs])
        bufs["req_pool_indices"][:bs].copy_(batch.req_pool_indices[:bs])
        bufs["hidden"][: bs * stride].copy_(hidden_strided.view(bs * stride, -1))
        bufs["commit_lens"][:bs].copy_(commit_lens[:bs])
        self._inject_pending = True
        self._inject_staged_bs = bs
        return True

    def pre_draft_forward(self, forward_batch) -> None:
        """Called right before the draft block forward. Flushes the staged
        inject eagerly when the forward will not replay a graph, and clears
        stale commit_lens before un-staged graph replays (idle batches)."""
        if self._inject_bufs is None:
            return
        runner = getattr(self.draft_model_runner, "decode_cuda_graph_runner", None)
        will_graph = runner is not None and runner.can_run_graph(forward_batch)
        if self._inject_pending:
            self._inject_pending = False
            if not will_graph:
                self._inject_prologue_body(self._inject_staged_bs)
        elif will_graph:
            self._inject_bufs["commit_lens"].zero_()

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
