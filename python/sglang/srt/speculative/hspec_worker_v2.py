"""H-Spec (mamba_attn_hybrid) speculative decoding worker for SGLang.

Reuses the DFLASH spec-v2 worker machinery (draft block construction,
target verify, accept/bonus, draft-KV lifecycle) and only overrides the
pieces where H-Spec semantics differ from DFLASH:

1. Context K/V materialization: H-Spec attention sub-layers were trained
   against the *target* layers' post-RoPE K/V (``attn_kv_layer_ids``), so the
   draft KV pool is populated by gathering from the target paged KV pool at
   the shared req_to_token slot mapping instead of projecting ctx hidden
   states with the draft's own K/V projections.
2. Mamba latent seed: the fused target hidden of each request's anchor token
   (the last accepted token) seeds the mixer states of the next draft block.
3. Draft sampling: the drafter has its own (possibly reduced) vocab head plus
   an optional sequential Markov bias; sampled ids are mapped into the target
   vocab via the checkpoint's ``d2t`` offsets.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


class HSpecWorkerV2(DFlashWorkerV2):
    """MAMBA_ATTN_HYBRID speculative decoding worker (spec-v2)."""

    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        draft_hf_config = self.draft_model_runner.model_config.hf_config
        self._attn_kv_layer_ids = list(
            getattr(draft_hf_config, "attn_kv_layer_ids", None) or []
        )
        if len(self._attn_kv_layer_ids) != len(
            getattr(self.draft_model, "attn_sublayers", [])
        ):
            raise ValueError(
                "MAMBA_ATTN_HYBRID: attn_kv_layer_ids does not match the draft "
                "model's attention sub-layers."
            )
        n_verifier_layers = int(
            getattr(
                self.target_worker.model_runner.model_config.hf_config,
                "num_hidden_layers",
                0,
            )
        )
        bad = [i for i in self._attn_kv_layer_ids if not 0 <= i < n_verifier_layers]
        if bad:
            raise ValueError(
                f"attn_kv_layer_ids {bad} out of range for a verifier with "
                f"{n_verifier_layers} layers."
            )
        self._pending_latent_seed: Optional[torch.Tensor] = None
        logger.info(
            "Initialized MAMBA_ATTN_HYBRID worker: draft=%s, attn_kv_layer_ids=%s, "
            "markov=%s, block_size=%s",
            self.draft_model.__class__.__name__,
            self._attn_kv_layer_ids,
            getattr(self.draft_model, "markov_head", None) is not None,
            self.block_size,
        )

    # ------------------------------------------------------------------
    # Draft forward with the mamba latent seed attached.
    # ------------------------------------------------------------------
    def _run_draft_forward(self, forward_batch):
        if getattr(self.draft_model, "set_latent_seed", None) is not None:
            self.draft_model.set_latent_seed(self._pending_latent_seed)
            try:
                return super()._run_draft_forward(forward_batch)
            finally:
                self.draft_model.set_latent_seed(None)
        return super()._run_draft_forward(forward_batch)

    # ------------------------------------------------------------------
    # Draft sampling through the draft vocab head (+ optional Markov bias).
    # ------------------------------------------------------------------
    def _sample_draft_next(
        self, draft_out, bs: int, draft_input, block_ids=None, sampling_info=None
    ):
        if self._draft_sampler is not None and draft_out.can_run_graph:
            # Captured-sampler path (CUDA graphs); uses the base machinery.
            return super()._sample_draft_next(draft_out, bs, draft_input)
        draft_logits_output = draft_out.logits_output
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("MAMBA_ATTN_HYBRID draft returned no hidden states.")
        block_size = int(self.block_size)
        draft_hidden = draft_hidden.view(bs, block_size, -1)
        # DFLASH infill convention: the drafted tokens are block positions
        # 1.., so the token at block position j is predicted by the hidden
        # state at row j-1 — sample from rows 1..K-1. (Rows 0..K-2 would
        # follow the SAMPLE_FROM_ANCHOR convention instead and break
        # first-position hits.)
        first_prev = draft_input.bonus_tokens.to(torch.long)
        return self.draft_model.sample_draft_ids(
            draft_hidden[:, 1:, :], first_prev
        ).view(bs, block_size - 1)

    # ------------------------------------------------------------------
    # Context K/V materialization from the target paged KV pool.
    # ------------------------------------------------------------------
    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        extend_lens: Optional[torch.Tensor] = None,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return

        # 1) Refresh the mamba latent seed from the anchor token's fused
        # target hidden before the next draft forward.
        with torch.inference_mode():
            if cache_loc_2d is not None and commit_lens is not None:
                bs = int(commit_lens.shape[0])
                block_hidden = target_hidden.view(bs, int(self.block_size), -1)
                anchor_idx = (commit_lens.to(torch.int64) - 1).clamp(min=0)
                self._pending_latent_seed = block_hidden[
                    torch.arange(bs, device=target_hidden.device), anchor_idx
                ].contiguous()
            elif extend_lens is not None:
                lens = torch.as_tensor(
                    extend_lens, device=target_hidden.device, dtype=torch.int64
                )
                last = torch.cumsum(lens, dim=0) - 1
                self._pending_latent_seed = target_hidden.index_select(
                    0, last
                ).contiguous()

        # 2) Gather the target layers' K/V into the draft pool.
        target_pool = self.target_worker.model_runner.token_to_kv_pool
        draft_pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(self.draft_model, "write_target_pool_kv"):
            with torch.inference_mode():
                self.draft_model.write_target_pool_kv(
                    target_token_to_kv_pool=target_pool,
                    draft_token_to_kv_pool=draft_pool,
                    cache_loc=cache_loc,
                    cache_loc_2d=cache_loc_2d,
                    commit_lens=commit_lens,
                )
            return

        # Fallback: generic per-layer gather (keeps the worker usable with
        # draft models that do not implement the fast path).
        with torch.inference_mode():
            loc = cache_loc.to(torch.int64)
            for sub_i, (_layer_idx, attn) in enumerate(self.draft_model.attn_sublayers):
                key_cache, value_cache = target_pool.get_kv_buffer(
                    self._attn_kv_layer_ids[sub_i]
                )
                if key_cache.dim() == 4:
                    key_cache = key_cache.reshape(-1, *key_cache.shape[2:])
                    value_cache = value_cache.reshape(-1, *value_cache.shape[2:])
                k = key_cache.index_select(0, loc).view(
                    -1, attn.num_kv_heads, attn.head_dim
                )
                v = value_cache.index_select(0, loc).view(
                    -1, attn.num_kv_heads, attn.head_dim
                )
                if v.dtype != k.dtype:
                    v = v.to(k.dtype)
                if cache_loc_2d is not None and commit_lens is not None:
                    draft_pool.set_kv_buffer_prefix_valid(
                        attn.attn,
                        cache_loc_2d,
                        commit_lens,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
                else:
                    draft_pool.set_kv_buffer(
                        attn.attn,
                        loc,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
