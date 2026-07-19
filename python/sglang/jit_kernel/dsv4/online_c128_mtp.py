from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.jit_kernel.dsv4.utils import make_name
from sglang.kernels.jit import cache_once, load_jit, make_cpp_args
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_online_c128_mtp_module(
    head_dim: int,
    seq_dtype: torch.dtype,
    req_dtype: torch.dtype,
    dtype_buffer: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, seq_dtype, req_dtype, dtype_buffer)
    return load_jit(
        make_name(f"online_c128_mtp_{head_dim}"),
        *args,
        cuda_files=["deepseek_v4/online_c128_mtp.cuh"],
        cuda_wrappers=[
            ("write_prefix_states", f"OnlineC128MTPWritePrefixKernel<{args}>::run"),
            ("mark_pending", f"OnlineC128MTPMarkPendingKernel<{args}>::run"),
            ("commit_pending", f"OnlineC128MTPCommitPendingKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@dataclass
class _OnlineC128LayerRuntime:
    head_dim: int
    main_state: torch.Tensor
    state_dtype: torch.dtype
    state_slot_offset: int


@dataclass
class _OnlineC128VerifyContext:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


class OnlineC128MTPController:
    def __init__(self, backend: Any):
        self.backend = backend
        self._verify_ctx: Optional[_OnlineC128VerifyContext] = None
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
        head_dim = self._head_dim()
        state_dtype = self._state_dtype()
        if head_dim is None or state_dtype is None or self._num_verify_tokens() == 0:
            return
        token_to_kv_pool = self.backend.token_to_kv_pool
        _jit_online_c128_mtp_module(
            head_dim, seq_lens.dtype, req_pool_indices.dtype, state_dtype
        ).mark_pending(
            seq_lens,
            req_pool_indices,
            token_to_kv_pool.get_online_c128_mtp_pending_seq_lens(),
            min(seq_lens.shape[0], req_pool_indices.shape[0]),
            token_to_kv_pool.get_online_c128_state_num_req_slots(),
        )

    def clear(self) -> None:
        self._verify_ctx = None

    def prepare_forward(
        self,
        logical_forward_mode,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        verify_bs: Optional[int] = None,
    ) -> int:
        if not self.enabled():
            self.clear()
            return 0
        if logical_forward_mode is None or logical_forward_mode.is_idle():
            self.clear()
            return 0

        active_req_pool_indices = req_pool_indices
        active_seq_lens = seq_lens
        if logical_forward_mode.is_target_verify():
            if verify_bs is None:
                verify_bs = req_pool_indices.shape[0]
            active_req_pool_indices = req_pool_indices[:verify_bs]
            active_seq_lens = seq_lens[:verify_bs]
            if verify_bs == 0:
                self.clear()
                return 0

        self.commit_pending(
            req_pool_indices=active_req_pool_indices,
            seq_lens=active_seq_lens,
        )
        if not logical_forward_mode.is_target_verify():
            return 0

        self.begin_verify(
            req_pool_indices=active_req_pool_indices,
            seq_lens=active_seq_lens,
        )
        return self.state_slot_offset()

    def write_prefix_states(
        self,
        layer_id: int,
        compressor: Any,
        kv_score_input: torch.Tensor,
        logical_forward_mode,
    ) -> None:
        if (
            not self.enabled()
            or logical_forward_mode is None
            or not logical_forward_mode.is_target_verify()
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
        state = state_pool.kv_score_buffer.kv_score
        total_bs = kv_score_input.numel() // (num_verify_tokens * head_dim * 2)
        layer_bs = min(ctx.seq_lens.shape[0], ctx.req_pool_indices.shape[0], total_bs)
        if layer_bs <= 0:
            return

        _jit_online_c128_mtp_module(
            head_dim, ctx.seq_lens.dtype, ctx.req_pool_indices.dtype, state.dtype
        ).write_prefix_states(
            kv_score_input,
            ctx.seq_lens,
            ctx.req_pool_indices,
            self.backend.req_to_token,
            compressor.ape.reshape(128, head_dim),
            state,
            layer_bs,
            num_verify_tokens,
            state_pool.online_mtp_state_slot_offset,
        )

    def commit_pending(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        if self._verify_ctx is None:
            return
        if not self.enabled():
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
        pending_seq_lens = token_to_kv_pool.get_online_c128_mtp_pending_seq_lens()
        cur_bs = min(seq_lens.shape[0], req_pool_indices.shape[0])

        for runtime in self._iter_layer_runtimes():
            _jit_online_c128_mtp_module(
                runtime.head_dim,
                seq_lens.dtype,
                req_pool_indices.dtype,
                runtime.state_dtype,
            ).commit_pending(
                seq_lens,
                req_pool_indices,
                backend.req_to_token,
                pending_seq_lens,
                runtime.main_state,
                cur_bs,
                num_verify_tokens,
                runtime.state_slot_offset,
                token_to_kv_pool.get_online_c128_state_num_req_slots(),
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

    def _head_dim(self) -> Optional[int]:
        for runtime in self._iter_layer_runtimes():
            return runtime.head_dim
        return None

    def _state_dtype(self) -> Optional[torch.dtype]:
        for runtime in self._iter_layer_runtimes():
            return runtime.state_dtype
        return None

    def _iter_layer_runtimes(self):
        if self._layer_runtimes is None:
            runtimes = []
            token_to_kv_pool = self.backend.token_to_kv_pool
            for layer in self.backend.model_runner.model.model.layers:
                attn = getattr(layer, "self_attn", None)
                compressor = getattr(attn, "compressor", None)
                if compressor is None or compressor.ratio != 128:
                    continue
                state_pool = token_to_kv_pool.get_attention_compress_states(
                    compressor.layer_id
                )
                runtimes.append(
                    _OnlineC128LayerRuntime(
                        head_dim=compressor.head_dim,
                        main_state=state_pool.kv_score_buffer.kv_score,
                        state_dtype=state_pool.kv_score_buffer.kv_score.dtype,
                        state_slot_offset=state_pool.online_mtp_state_slot_offset,
                    )
                )
            self._layer_runtimes = runtimes
        return iter(self._layer_runtimes)
