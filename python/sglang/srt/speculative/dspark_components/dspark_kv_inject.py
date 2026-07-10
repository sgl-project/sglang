import logging
import math
from typing import Callable, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_utils import can_dflash_use_fused_qkv_proj
from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    BuildCommitInjectLayout,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
    FusedKVMaterializeHelper,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)


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
        # Fused KV materialization (batched GEMM + single Triton norm/rope kernel)
        # replaces the per-layer eager loop in write_target_hidden_kv. None if the
        # draft is ineligible or we are off-CUDA; the eager path stays as fallback.
        self._fused_kv_helper = self._build_fused_kv_helper()

    def _build_fused_kv_helper(self) -> Optional[FusedKVMaterializeHelper]:
        """Build the fused KV helper, mirroring DFlashWorkerV2._init_fused_kv_helper.

        The Triton kernel handles both neox and interleaved RoPE, so unlike the
        DFlash worker this does not reject interleaved-rotary drafts. Returns
        None (eager write_target_hidden_kv fallback) on any failure.
        """
        try:
            if not (is_cuda() or is_hip()):
                return None
            layers = self.draft_model.layers
            if len(layers) == 0 or not self.draft_model.supports_fused_context_kv:
                return None

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    logger.info(
                        "DSpark fused KV disabled: %s (layer=%d).", reason, layer_idx
                    )
                    return None
                # The kernel writes V straight through; non-unit scales would need
                # to be folded in, which the fused path does not do.
                k_scale = attn.attn.k_scale
                v_scale = attn.attn.v_scale
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    logger.info(
                        "DSpark fused KV disabled: non-unit k_scale (layer=%d).",
                        layer_idx,
                    )
                    return None
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    logger.info(
                        "DSpark fused KV disabled: non-unit v_scale (layer=%d).",
                        layer_idx,
                    )
                    return None

            first_attn = layers[0].self_attn
            helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=first_attn.rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                max_position_hint=self.model_runner.model_config.context_len
                + int(self.verify_num_draft_tokens),
            )
            logger.info(
                "DSpark fused KV materialization enabled (n_layers=%d, neox=%s).",
                len(layers),
                helper.is_neox_style,
            )
            return helper
        except Exception as e:
            logger.warning(
                "DSpark fused KV init failed, using eager KV path: %s", e
            )
            return None

    def _inject_fused(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
    ) -> None:
        pool = self.draft_model_runner.token_to_kv_pool
        ctx_hidden = self.draft_model.project_target_hidden(target_hidden)

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = self.draft_model.layers[layer_idx].self_attn.attn
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn,
                    cache_loc_2d,
                    commit_lens,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn, cache_loc, cache_k, cache_v, attn.k_scale, attn.v_scale
                )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=positions,
            write_layer_kv=_write_layer_kv,
        )

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
            if self._fused_kv_helper is not None:
                try:
                    self._inject_fused(
                        target_hidden=target_hidden,
                        cache_loc=cache_loc,
                        positions=positions,
                        cache_loc_2d=cache_loc_2d,
                        commit_lens=commit_lens,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "DSpark fused KV append failed; falling back to the "
                        "per-layer eager path: %s",
                        e,
                    )
                    self._fused_kv_helper = None

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
