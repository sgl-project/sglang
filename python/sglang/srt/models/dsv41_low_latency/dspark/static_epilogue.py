# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The static DSpark verify epilogue with the low-latency commits, recorded in the
verify CUDA graph in place of DsparkVerifyEpilogue._static_epilogue: the greedy accept
and the target-hidden KV commit as there, plus the Engram history commit (gated per row)
and the draft context-KV projection on a side stream beside the accept. It also keeps
each captured batch size's static inputs and outputs for the sampling-accept graph."""

from __future__ import annotations

from typing import Callable, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite


class StaticVerifyIO(msgspec.Struct, frozen=True):
    """A static verify graph's input and output tensors for one captured batch size."""

    candidates: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor


class ContextKvSideStream:
    """Projects the draft context KV on a side stream beside the verify accept. Only
    under graph capture: eager steps keep the serial order, so no cross-stream reuse
    of allocator blocks."""

    def __init__(self) -> None:
        self._stream = torch.cuda.Stream()

    def fork(self, *, draft_model, pool, hidden: torch.Tensor):
        if not torch.cuda.is_current_stream_capturing():
            return None
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            return draft_model.project_context_kv(main_hidden=hidden, pool=pool)

    def join(self, context_kv: list[torch.Tensor]) -> list[torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self._stream)
        return context_kv


class StaticEpilogue:
    """Set as ``DsparkVerifyEpilogue.static_epilogue_override``."""

    def __init__(
        self,
        *,
        epilogue: DsparkVerifyEpilogue,
        tp_sync: SpecTpSync,
        fused_argmax: bool,
        resolve_engram_history: Callable[[], Optional[torch.Tensor]],
        context_kv_side_stream: Optional[ContextKvSideStream],
    ) -> None:
        self._epilogue = epilogue
        self._tp_sync = tp_sync
        self.fused_argmax = fused_argmax
        self._resolve_engram_history = resolve_engram_history
        self._context_kv_side_stream = context_kv_side_stream
        # Per captured static batch size; the sampling-accept graph replays read them.
        self.static_io: dict[int, StaticVerifyIO] = {}
        # Set when a captured epilogue recorded the Engram history commit.
        self.folds_engram_commit = False

    def __call__(self, out, forward_batch) -> None:
        bs = forward_batch.batch_size
        io = StaticVerifyIO(
            candidates=forward_batch.input_ids.view(bs, self._epilogue.stride),
            logits=out.next_token_logits,
            hidden=out.hidden_states,
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            out_cache_loc=forward_batch.out_cache_loc,
            positions=forward_batch.positions,
        )
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            self.static_io[bs] = io
        committed_engram = self.run(
            io, accept=self.accept_greedy, gate=self._epilogue.inject_gate_buf
        )
        if committed_engram and capturing:
            self.folds_engram_commit = True

    def run(self, io: StaticVerifyIO, *, accept, gate: torch.Tensor) -> bool:
        """``accept(io)`` returns the commit lengths; the Engram history and KV commits
        follow, gated by ``gate``. Returns whether the Engram history was committed."""
        epilogue = self._epilogue
        bs, stride = io.candidates.shape
        verify_lens = epilogue.verify_lens_buf[:bs]
        folds_commit = epilogue.folds_commit
        context_kv = None
        if self._context_kv_side_stream is not None and folds_commit:
            context_kv = self._context_kv_side_stream.fork(
                draft_model=epilogue.commit_ctx.draft_model,
                pool=epilogue.commit_ctx.resolve_pool(),
                hidden=io.hidden[: io.positions.shape[0]],
            )
        commit_lens = accept(io)
        history = self._resolve_engram_history()
        if history is not None:
            # Padded rows (verify_lens 0) and steps the graph does not commit (gate 0) skip.
            engram_commit_history(
                history,
                io.candidates,
                io.req_pool_indices,
                commit_lens,
                live_rows=verify_lens,
                gate=gate,
            )
        if not folds_commit:
            return history is not None
        # Same staged locations as target verify; padded and fallback rows skip KV.
        gated_commit_lens = (
            torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate
        )
        state_slot = None
        if is_unified_kv_triton():
            state_slot = io.req_pool_indices.view(-1, 1).expand(bs, stride).reshape(-1)
        epilogue.commit_ctx.kv_injector.inject_target_hidden(
            target_hidden=io.hidden,
            cache_loc=io.out_cache_loc,
            cache_loc_2d=io.out_cache_loc.view(bs, stride),
            positions=io.positions,
            commit_lens=gated_commit_lens,
            state_slot=state_slot,
            context_kv=(
                None
                if context_kv is None
                else self._context_kv_side_stream.join(context_kv)
            ),
        )
        return history is not None

    def accept_greedy(self, io: StaticVerifyIO) -> torch.Tensor:
        """DsparkVerifyEpilogue's in-graph greedy accept into its accept buffers."""
        epilogue = self._epilogue
        bs, stride = io.candidates.shape
        correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
            candidates=io.candidates,
            target_logits=io.logits,
            verify_num_draft_tokens=stride,
            fused_argmax=self.fused_argmax,
        )
        return build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=io.seq_lens[:bs],
            draft_tokens=io.candidates[:, 1:].contiguous(),
            gamma=stride - 1,
            verify_num_draft_tokens=stride,
            tp_sync=self._tp_sync,
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
            simulated_correct_len=epilogue.simulated_correct_len,
            out=epilogue.read_accept(bs),
        ).commit_lens
