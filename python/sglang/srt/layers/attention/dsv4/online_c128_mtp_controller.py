from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from sglang.jit_kernel.dsv4.online_c128_mtp import (
    online_c128_mtp_lazy_commit,
    online_c128_mtp_prepare,
    online_c128_mtp_write_prefix_states,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@dataclass
class _OnlineC128LayerRuntime:
    head_dim: int
    state_width: int
    main_state: torch.Tensor
    temp_state_slot_offset: int


@dataclass
class _OnlineC128VerifyContext:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


class OnlineC128MTPController:
    def __init__(self, backend: Any):
        self.backend = backend
        self._verify_ctx: Optional[_OnlineC128VerifyContext] = None
        self._compressors: Optional[List[Tuple[int, Any]]] = None
        self._layer_runtimes: Optional[List[_OnlineC128LayerRuntime]] = None

    def enabled(self) -> bool:
        return (
            envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
            and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
            and self.backend.mtp_enabled
        )

    def state_slot_offset(self) -> int:
        if not self.enabled():
            return 0
        return self.backend.token_to_kv_pool.get_online_c128_mtp_state_slot_offset()

    def begin_verify(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if not self.enabled():
            self.clear()
            return

        self._verify_ctx = _OnlineC128VerifyContext(
            req_pool_indices=req_pool_indices.detach(),
            seq_lens=seq_lens.detach(),
        )
        if self._active_ctx() is not None:
            self._prepare_layers()

    def clear(self) -> None:
        self._verify_ctx = None

    def write_prefix_states(
        self,
        layer_id: int,
        compressor: Any,
        kv_score_input: torch.Tensor,
        logical_forward_mode: ForwardMode,
    ) -> None:
        if (
            not self.enabled()
            or logical_forward_mode != ForwardMode.TARGET_VERIFY
            or compressor.is_in_indexer
            or compressor.ratio != 128
            or kv_score_input.numel() == 0
        ):
            return

        ctx = self._active_ctx()
        num_verify_tokens = self._num_verify_tokens()
        if ctx is None or num_verify_tokens == 0:
            return

        token_to_kv_pool = self.backend.token_to_kv_pool
        head_dim = compressor.head_dim
        state_pool = token_to_kv_pool.get_attention_compress_states(layer_id)
        total_bs = kv_score_input.numel() // (num_verify_tokens * head_dim * 2)
        layer_bs = min(ctx.seq_lens.shape[0], ctx.req_pool_indices.shape[0], total_bs)
        if layer_bs <= 0:
            return

        online_c128_mtp_write_prefix_states(
            kv_score_input=kv_score_input,
            seq_lens=ctx.seq_lens,
            req_pool_indices=ctx.req_pool_indices,
            req_to_token=self.backend.req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
            ape=compressor.ape.reshape(128, head_dim),
            state=state_pool.kv_score_buffer.kv_score,
            layer_bs=layer_bs,
            swa_page_size=token_to_kv_pool.swa_page_size,
            num_verify_tokens=num_verify_tokens,
            state_slot_stride=state_pool.online_mtp_state_slot_offset,
            head_dim=head_dim,
        )

    def commit_pending(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if self._verify_ctx is None:
            return
        active_ctx = self._active_ctx()
        if not self.enabled() or active_ctx is None:
            self.clear()
            return
        if req_pool_indices.numel() == 0 or seq_lens.numel() == 0:
            return

        num_verify_tokens = self._num_verify_tokens()
        if num_verify_tokens == 0:
            self.clear()
            return

        backend = self.backend
        token_to_kv_pool = backend.token_to_kv_pool
        cur_req_pool_indices = req_pool_indices.to(active_ctx.req_pool_indices.device)
        cur_seq_lens = seq_lens.to(active_ctx.seq_lens.device)
        old_bs = min(
            active_ctx.seq_lens.shape[0],
            active_ctx.req_pool_indices.shape[0],
        )
        cur_bs = min(cur_seq_lens.shape[0], cur_req_pool_indices.shape[0])

        for runtime in self._iter_layer_runtimes():
            online_c128_mtp_lazy_commit(
                old_seq_lens=active_ctx.seq_lens,
                old_req_pool_indices=active_ctx.req_pool_indices,
                cur_seq_lens=cur_seq_lens,
                cur_req_pool_indices=cur_req_pool_indices,
                req_to_token=backend.req_to_token,
                full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
                state=runtime.main_state,
                old_bs=old_bs,
                cur_bs=cur_bs,
                swa_page_size=token_to_kv_pool.swa_page_size,
                num_verify_tokens=num_verify_tokens,
                head_dim=runtime.head_dim,
                state_slot_stride=runtime.temp_state_slot_offset,
            )

        self.clear()

    def _num_verify_tokens(self) -> int:
        if not self.enabled():
            return 0
        num_verify_tokens = int(self.backend.speculative_num_draft_tokens)
        max_draft_tokens = (
            self.backend.token_to_kv_pool.get_online_c128_mtp_max_draft_tokens()
        )
        return num_verify_tokens if 0 < num_verify_tokens <= max_draft_tokens else 0

    def _active_ctx(self) -> Optional[_OnlineC128VerifyContext]:
        ctx = self._verify_ctx
        if (
            ctx is None
            or ctx.seq_lens.numel() == 0
            or ctx.req_pool_indices.numel() == 0
        ):
            return None
        return ctx

    def _prepare_layers(self) -> None:
        ctx = self._active_ctx()
        if ctx is None:
            return

        backend = self.backend
        token_to_kv_pool = backend.token_to_kv_pool
        for runtime in self._iter_layer_runtimes():
            online_c128_mtp_prepare(
                seq_lens=ctx.seq_lens,
                req_pool_indices=ctx.req_pool_indices,
                req_to_token=backend.req_to_token,
                full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
                main_state=runtime.main_state,
                bs=ctx.seq_lens.shape[0],
                swa_page_size=token_to_kv_pool.swa_page_size,
                temp_state_slot_offset=runtime.temp_state_slot_offset,
                state_width=runtime.state_width,
            )

    def _iter_online_c128_layers(self):
        if self._compressors is None:
            compressors = []
            for layer in self.backend.model_runner.model.model.layers:
                attn = getattr(layer, "self_attn", None)
                compressor = getattr(attn, "compressor", None)
                if compressor is not None and compressor.ratio == 128:
                    compressors.append((compressor.layer_id, compressor))
            self._compressors = compressors
        return iter(self._compressors)

    def _iter_layer_runtimes(self):
        if self._layer_runtimes is None:
            runtimes = []
            token_to_kv_pool = self.backend.token_to_kv_pool
            for layer_id, compressor in self._iter_online_c128_layers():
                head_dim = compressor.head_dim
                main_pool = token_to_kv_pool.get_attention_compress_states(layer_id)
                runtimes.append(
                    _OnlineC128LayerRuntime(
                        head_dim=head_dim,
                        state_width=head_dim * 3,
                        main_state=main_pool.kv_score_buffer.kv_score,
                        temp_state_slot_offset=main_pool.online_mtp_state_slot_offset,
                    )
                )
            self._layer_runtimes = runtimes
        return iter(self._layer_runtimes)
