"""DFlash V2 worker with the ROCm (gfx950) fast paths.

Same algorithm as DFlashWorkerV2; overrides its platform hooks with HIP
kernels (packed draft argmax, fused draft-KV store, verify argmax and the
post-verify bookkeeping kernels) and replays the post-verify draft-KV append
from a captured graph.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_parallel, mamba_track_grid
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


class DFlashWorkerV2ROCm(DFlashWorkerV2):
    _packed_draft_argmax = True
    # Publish the new seq lens (the event the scheduler host-waits on) before
    # the mamba-state commit so the commit overlaps the scheduler's host work.
    _publish_before_mamba_commit = True

    def _init_tail_graph_state(self) -> None:
        # Lazily captured per input signature (see _tail_graph_run).
        self._tail_graph_on = bool(
            self.lilicorr is None and not get_parallel().attn_dp_enabled
        )
        self._tail_graph_states: dict = {}
        self._tail_graph_pool = None
        self._tail_graph_commit_buf: Optional[torch.Tensor] = None

    def _replay_tail_graph(self, **kwargs) -> bool:
        return self._tail_graph_on and self._tail_graph_run(**kwargs)

    def _tail_graph_run(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> bool:
        """Replay the post-verify draft-KV
        materialization (fc projection + per-layer K-norm/RoPE/KV-proj/
        prefix-valid KV write, ~26 launches) as one CUDA graph per batch size.

        Every input lives at a stable address for a given batch size (the
        shared target aux-hidden output, the static draft-block cache-loc /
        position buffers) except ``commit_lens`` (rotating accept buffers),
        which is staged into a static buffer. Captured lazily after a few eager
        steps with an unchanged input signature; any signature change (e.g. an
        eager-verify step) runs eager. Returns False when the caller must run
        the eager path."""
        if (
            commit_lens is None
            or cache_loc_2d is None
            or target_hidden.dim() != 2
            or not target_hidden.is_cuda
            or bs <= 0
            or bs > 64
        ):
            return False
        sig = (
            target_hidden.data_ptr(),
            tuple(target_hidden.shape),
            tuple(target_hidden.stride()),
            target_hidden.dtype,
            cache_loc.data_ptr(),
            cache_loc_2d.data_ptr(),
            tuple(cache_loc_2d.shape),
            positions.data_ptr(),
            positions.dtype,
            cache_loc.dtype,
            cache_loc_2d.dtype,
        )
        st = self._tail_graph_states.get(sig)
        if st is None:
            # One graph per input signature (normally one per batch size);
            # bounded so a flapping signature cannot recapture forever.
            if len(self._tail_graph_states) >= 64:
                return False
            st = {"warm": 0, "graph": None}
            self._tail_graph_states[sig] = st
        if self._tail_graph_commit_buf is None:
            self._tail_graph_commit_buf = torch.zeros(
                (64,), dtype=torch.int32, device=target_hidden.device
            )
        commit_static = self._tail_graph_commit_buf[:bs]
        if st["graph"] is None:
            st["warm"] += 1
            if st["warm"] <= 3 or st.get("failed"):
                return False
            if self._tail_graph_pool is None:
                self._tail_graph_pool = torch.cuda.graph_pool_handle()
            commit_static.copy_(commit_lens)
            graph = torch.cuda.CUDAGraph()
            try:
                torch.cuda.synchronize()
                with torch.cuda.graph(
                    graph, pool=self._tail_graph_pool, capture_error_mode="thread_local"
                ):
                    self._append_target_hidden_to_draft_kv_by_loc(
                        target_hidden=target_hidden,
                        cache_loc=cache_loc,
                        cache_loc_2d=cache_loc_2d,
                        positions=positions,
                        commit_lens=commit_static,
                    )
            except Exception:
                logger.warning(
                    "K3 DFlash tail graph capture failed for bs=%d; "
                    "running the eager draft-KV tail for it.",
                    bs,
                    exc_info=True,
                )
                st["failed"] = True
                return False
            st["graph"] = graph
            if get_parallel().tp_rank == 0:
                logger.info("Captured draft-KV tail bs=%d", bs)
            graph.replay()
            return True
        commit_static.copy_(commit_lens)
        st["graph"].replay()
        return True

    def _kv_fuse_store(self, attn, k, v, positions, cache_loc_2d, commit_lens) -> bool:
        """k-norm straight from the strided K
        slice (same AITER kernel, no .contiguous() copy) + one fused bf16-exact
        RoPE / prefix-valid KV write. Returns False (did nothing) if ineligible."""
        from sglang.srt.layers import layernorm as _ln

        rope = getattr(attn, "rotary_emb", None)
        pool = self.draft_model_runner.token_to_kv_pool
        hd = int(attn.head_dim)
        norm = attn.k_norm
        if not (
            getattr(_ln, "_use_aiter", False)
            and getattr(norm, "_forward_method", None) is not None
            and getattr(norm._forward_method, "__func__", None)
            is _ln.RMSNorm.forward_aiter
            and not getattr(norm, "_fused_pad_kernel", None)
            and norm.weight.dtype == torch.bfloat16
            and int(attn.num_kv_heads) == 1
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and k.dim() == 2
            and k.shape[1] == hd
            and k.stride(1) == 1
            and v.stride(1) == 1
            and rope is not None
            and getattr(rope, "use_fallback_kernel", False)
            and getattr(rope, "is_neox_style", False)
            and int(getattr(rope, "rotary_dim", -1)) == hd
            and int(getattr(rope, "head_size", -1)) == hd
            and hasattr(rope, "cos_sin_cache_as")
            and getattr(pool, "store_dtype", None) == torch.bfloat16
            and getattr(pool, "dtype", None) == torch.bfloat16
            and not getattr(pool, "use_hnd", False)
        ):
            return False
        from sglang.kernels.ops.speculative.k3_dflash_kv_store import (
            rope_store_prefix_valid,
        )

        if getattr(_ln, "is_batch_invariant_mode_enabled", lambda: False)():
            return False
        k_normed = _ln.rms_norm(k, norm.weight.data, norm.variance_epsilon)
        rope.cos_sin_cache = rope.cos_sin_cache.to(k.device)
        cos_sin = rope.cos_sin_cache_as(torch.bfloat16)
        layer_id = attn.attn.layer_id
        rope_store_prefix_valid(
            k_normed.view(-1, 1, hd),
            v.view(-1, 1, hd),
            positions,
            cos_sin,
            cache_loc_2d,
            commit_lens,
            pool.k_buffer[layer_id - pool.start_layer],
            pool.v_buffer[layer_id - pool.start_layer],
        )
        return True

    def _mamba_commit_steps(
        self,
        *,
        batch: ScheduleBatch,
        commit_lens: torch.Tensor,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not commit_lens.is_cuda:
            return super()._mamba_commit_steps(
                batch=batch,
                commit_lens=commit_lens,
                seq_lens_pre_verify=seq_lens_pre_verify,
                seq_lens_post_verify=seq_lens_post_verify,
            )
        from sglang.kernels.ops.speculative.k3_dflash_post import (
            mamba_track_steps,
        )

        has_track = batch.mamba_track_indices is not None
        last_correct_step_indices, mamba_steps_to_track = mamba_track_steps(
            commit_lens,
            seq_lens_pre_verify,
            seq_lens_post_verify,
            (mamba_track_grid(batch.tree_cache.page_size) if has_track else 1),
            has_track,
        )
        return last_correct_step_indices, mamba_steps_to_track

    def _verify_argmax(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        if (
            next_token_logits.is_cuda
            and next_token_logits.dtype == torch.float32
            and next_token_logits.dim() == 2
            and next_token_logits.stride(1) == 1
        ):
            from sglang.kernels.ops.speculative.k3_dflash_post import (
                verify_argmax,
            )

            return verify_argmax(next_token_logits)
        return super()._verify_argmax(next_token_logits)

    def _compact_draft_prefix_lens(
        self, prefix_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if prefix_lens.is_cuda and self._use_triton_compact_rebuild:
            from sglang.kernels.ops.speculative.k3_dflash_post import (
                compact_draft_lens,
            )

            return compact_draft_lens(
                prefix_lens,
                int(self.draft_window_size),
                self.page_size if self.page_size > 1 else 1,
            )
        return super()._compact_draft_prefix_lens(prefix_lens)
