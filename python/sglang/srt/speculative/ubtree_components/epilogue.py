# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""CUDA-graph folded target verification epilogue for UBTree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
    commit_kda_tree_after_verify,
)
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_commit_track_indices,
)
from sglang.kernels.ops.speculative.ubtree import (
    prepare_draft_commit_metadata,
    prepare_target_kv_move_metadata,
    prepare_unified_commit_metadata,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.ubtree_info import UBTreeVerifyInput


@dataclass(frozen=True)
class UBTreeVerifyResult:
    predict: torch.Tensor
    accept_lens: torch.Tensor
    accept_index: torch.Tensor
    hidden: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor


class UBTreeVerifyEpilogue:
    """Fold tree acceptance and accepted-path compaction into a CUDA Graph.

    Tree topology is dynamic, so runtime inputs are copied into stable buffers
    before replay. When target KV storage is configured, accepted-path metadata
    preparation and target KV movement are captured in the same graph tail.
    """

    def __init__(
        self,
        *,
        max_bs: int,
        node_count: int,
        max_depth: int,
        device,
        target_kvcache=None,
        req_to_token: Optional[torch.Tensor] = None,
        draft_commit=None,
        draft_req_to_token: Optional[torch.Tensor] = None,
    ) -> None:
        self.max_bs = int(max_bs)
        self.node_count = int(node_count)
        # A root-inclusive accepted path cannot contain more rows than the
        # verify tree itself.  Small forced-width buckets (for example M=9
        # with a B16 drafter) therefore need a narrower graph-tail buffer than
        # the checkpoint block depth.
        self.max_depth = min(int(max_depth), self.node_count)
        self.target_kvcache = target_kvcache
        self.req_to_token = req_to_token
        self.draft_commit = draft_commit
        self.draft_req_to_token = draft_req_to_token
        self._get_draft_kv_workspace: Optional[
            Callable[[], tuple[torch.Tensor, ...]]
        ] = None
        self.kda_spec_state = None
        self.mamba_index_mapping: Optional[torch.Tensor] = None
        self.mamba_track_interval = 0
        if (target_kvcache is None) != (req_to_token is None):
            raise ValueError(
                "target_kvcache and req_to_token must be configured together"
            )
        if (draft_commit is None) != (draft_req_to_token is None):
            raise ValueError(
                "draft_commit and draft_req_to_token must be configured together"
            )
        if self.max_bs <= 0 or self.node_count <= 0 or self.max_depth <= 0:
            raise ValueError("UBTree epilogue dimensions must be positive")

        shape = (self.max_bs, self.node_count)
        self.draft_token_buf = torch.zeros(shape, dtype=torch.int64, device=device)
        self.retrieve_index_buf = torch.arange(
            self.max_bs * self.node_count, dtype=torch.int64, device=device
        ).view(shape)
        self.retrieve_next_token_buf = torch.full(
            shape, -1, dtype=torch.int64, device=device
        )
        self.retrieve_next_sibling_buf = torch.full_like(
            self.retrieve_next_token_buf, -1
        )

        self.seq_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.target_predict_buf = torch.empty(shape, dtype=torch.int64, device=device)
        self.predict_buf = torch.zeros(shape, dtype=torch.int32, device=device)
        self.accept_index_buf = torch.full(
            (self.max_bs, self.max_depth),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.accept_lens_buf = torch.empty(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        # The target Graph includes the greedy epilogue. Sampling requests
        # disarm its persistent writes, then arm the separate sampling tail.
        # A device gate lets both paths reuse the same captured model.
        self.commit_gate_buf = torch.ones((1,), dtype=torch.int32, device=device)
        self.commit_lens_buf = torch.empty_like(self.accept_lens_buf)
        # Only the root-inclusive accepted path can be committed to the draft
        # cache.  Keeping a node_count-wide safe-index/hidden buffer gathered 65
        # rows per request even though UBTree's maximum accepted path is the
        # 16-row draft block.
        self.safe_accept_index_buf = torch.zeros(
            (self.max_bs, self.max_depth), dtype=torch.int64, device=device
        )
        self.compact_predict_buf = torch.zeros_like(self.predict_buf)
        self.path_predict_buf = torch.empty(
            (self.max_bs, self.max_depth), dtype=torch.int32, device=device
        )
        self.compact_hidden_buf: Optional[torch.Tensor] = None
        self.bonus_index_buf = torch.empty(
            (self.max_bs, 1), dtype=torch.int64, device=device
        )
        self.bonus_tokens_i32_buf = torch.empty(
            (self.max_bs, 1), dtype=torch.int32, device=device
        )
        self.bonus_tokens_buf = torch.empty(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.new_seq_lens_buf = torch.empty(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.target_cache_loc_buf = torch.empty(
            (self.max_bs, self.max_depth), dtype=torch.int64, device=device
        )
        self.source_cache_loc_buf = torch.empty_like(self.target_cache_loc_buf)
        self.draft_positions_buf = torch.empty(
            (self.max_bs, self.max_depth), dtype=torch.int64, device=device
        )
        self.draft_cache_loc_buf = torch.empty_like(self.draft_positions_buf)
        self.state_batch_indices_buf = torch.empty(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.factor_cache_indices_buf = torch.arange(
            self.max_bs, dtype=torch.int32, device=device
        )
        self.last_correct_step_buf = torch.empty(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.mamba_steps_to_track_buf = torch.empty_like(self.last_correct_step_buf)
        self._graph_metadata: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = {}
        self._graph_draft_kv_workspaces: dict[int, tuple[torch.Tensor, ...]] = {}

    def set_armed(self, armed: bool) -> None:
        """Enable persistent writes and sequence-length publication on replay."""
        self.commit_gate_buf.fill_(int(armed))

    def _stage_tree_inputs(
        self,
        *,
        draft_token: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
    ) -> None:
        if bs <= 0 or bs > self.max_bs:
            raise ValueError(f"UBTree epilogue batch size {bs} exceeds {self.max_bs}")
        expected = bs * self.node_count
        for name, value in (
            ("draft_token", draft_token),
            ("retrieve_index", retrieve_index),
            ("retrieve_next_token", retrieve_next_token),
            ("retrieve_next_sibling", retrieve_next_sibling),
        ):
            if value is None or value.numel() != expected:
                raise ValueError(f"UBTree epilogue received an invalid {name} shape")
        if seq_lens.numel() != bs:
            raise ValueError("UBTree epilogue received an invalid seq-lens shape")

        self.seq_lens_buf[:bs].copy_(seq_lens)
        self.draft_token_buf[:bs].copy_(draft_token.view(bs, self.node_count))
        self.retrieve_index_buf[:bs].copy_(retrieve_index.view(bs, self.node_count))
        self.retrieve_next_token_buf[:bs].copy_(
            retrieve_next_token.view(bs, self.node_count)
        )
        self.retrieve_next_sibling_buf[:bs].copy_(
            retrieve_next_sibling.view(bs, self.node_count)
        )

    def begin_tree_step(self, tree, seq_lens: torch.Tensor) -> None:
        """Stage graph-owned proposal tensors directly into target inputs."""

        self._stage_tree_inputs(
            draft_token=tree.tokens,
            retrieve_index=tree.retrieve_index,
            retrieve_next_token=tree.retrieve_next_token,
            retrieve_next_sibling=tree.retrieve_next_sibling,
            seq_lens=seq_lens,
            bs=tree.batch_size,
        )

    def begin_step(
        self, verify_input: UBTreeVerifyInput, seq_lens: torch.Tensor, bs: int
    ) -> None:
        """Copy one dynamic tree batch into graph-stable input buffers."""

        self._stage_tree_inputs(
            draft_token=verify_input.draft_token,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            seq_lens=seq_lens,
            bs=bs,
        )

    def capture_hook(self, runner, out, forward_batch, num_tokens: int) -> None:
        if runner.model_runner.is_draft_worker:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
            return
        bs = int(forward_batch.batch_size)
        if num_tokens != bs * self.node_count:
            return
        self._graph_metadata[bs] = (
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
            forward_batch.mamba_track_indices,
        )
        self(
            logits=out.next_token_logits,
            hidden=out.hidden_states,
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            out_cache_loc=forward_batch.out_cache_loc,
            mamba_track_indices=forward_batch.mamba_track_indices,
        )
        if torch.cuda.is_current_stream_capturing():
            # The folded target Graph keeps using these addresses after prefill
            # grows the materializer's workspace. Retain only captured buffers;
            # eager warmups and ordinary prefill growth need no extra ownership.
            self._graph_draft_kv_workspaces[bs] = self.draft_kv_workspace_tensors()

    def draft_kv_workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        """Snapshot the external materializer workspace used by this tail."""
        if self._get_draft_kv_workspace is None:
            return ()
        return self._get_draft_kv_workspace()

    def graph_metadata(
        self, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return the target Graph's stable metadata for the sampling tail."""
        try:
            return self._graph_metadata[bs]
        except KeyError as exc:
            raise RuntimeError(
                f"UBTree target Graph metadata is unavailable for BS={bs}"
            ) from exc

    def _ensure_hidden_buffer(self, hidden: torch.Tensor) -> torch.Tensor:
        buf = self.compact_hidden_buf
        if (
            buf is not None
            and buf.shape[1] == hidden.shape[1]
            and buf.dtype == hidden.dtype
        ):
            return buf
        assert not torch.cuda.is_current_stream_capturing(), (
            "UBTree epilogue hidden buffer must be allocated during graph warmup"
        )
        self.compact_hidden_buf = torch.empty(
            (self.max_bs * self.max_depth, hidden.shape[1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return self.compact_hidden_buf

    def attach_target_kv_commit(
        self, *, target_kvcache, req_to_token: torch.Tensor
    ) -> None:
        """Attach target KV storage after the worker allocates memory pools."""

        if target_kvcache is None or req_to_token is None:
            raise ValueError("target KV commit requires cache and req_to_token")
        self.target_kvcache = target_kvcache
        self.req_to_token = req_to_token

    def attach_draft_kv_commit(
        self,
        *,
        draft_commit,
        req_to_token: torch.Tensor,
        get_workspace_tensors: Optional[Callable[[], tuple[torch.Tensor, ...]]] = None,
    ) -> None:
        """Attach a graph-safe draft hidden/KV writer after pool allocation."""

        if draft_commit is None or req_to_token is None:
            raise ValueError("draft KV commit requires writer and req_to_token")
        self.draft_commit = draft_commit
        self.draft_req_to_token = req_to_token
        self._get_draft_kv_workspace = get_workspace_tensors

    def attach_kda_tree_commit(
        self,
        *,
        spec_state,
        mamba_index_mapping: torch.Tensor,
        mamba_track_interval: int,
    ) -> None:
        """Attach factorized KDA exact-fold and conv commit to the graph tail."""

        if spec_state is None or spec_state.tree_rawv is None:
            raise ValueError("KDA tree commit requires factorized tree state")
        if int(spec_state.tree_rawv.shape[-2]) < self.node_count:
            raise ValueError(
                "KDA tree cache is smaller than epilogue node count: "
                f"{spec_state.tree_rawv.shape[-2]} < {self.node_count}"
            )
        if mamba_index_mapping is None or mamba_index_mapping.dtype != torch.int32:
            raise ValueError("KDA tree commit requires the int32 mamba slot mapping")
        self.kda_spec_state = spec_state
        self.mamba_index_mapping = mamba_index_mapping
        self.mamba_track_interval = int(mamba_track_interval)

    @property
    def folds_target_kv(self) -> bool:
        return self.target_kvcache is not None

    @property
    def folds_draft_kv(self) -> bool:
        return self.draft_commit is not None

    @property
    def folds_kda_commit(self) -> bool:
        return self.kda_spec_state is not None

    def __call__(
        self,
        *,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        bs: int,
        req_pool_indices: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        mamba_track_indices: Optional[torch.Tensor] = None,
    ) -> None:
        rows = bs * self.node_count
        target_predict = self.target_predict_buf[:bs]
        torch.argmax(logits[:rows], dim=-1, out=target_predict.reshape(-1))
        self.run_from_target_predict(
            target_predict=target_predict,
            hidden=hidden,
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            mamba_track_indices=mamba_track_indices,
        )

    def run_from_target_predict(
        self,
        *,
        target_predict: torch.Tensor,
        hidden: torch.Tensor,
        bs: int,
        req_pool_indices: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        mamba_track_indices: Optional[torch.Tensor] = None,
        broadcast_outputs: bool = False,
    ) -> None:
        """Walk and commit a tree using one target prediction per node.

        Predictions may be argmax tokens or independent target samples. A
        matching child extends the sampled path; the first missing child ends
        it with a bonus token. Sampling therefore needs no draft probabilities.
        """
        rows = bs * self.node_count
        predict = self.predict_buf[:bs].reshape(-1)
        accept_index = self.accept_index_buf[:bs]
        accept_lens = self.accept_lens_buf[:bs]

        predict.zero_()
        accept_index.fill_(-1)
        verify_tree_greedy_func(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_lens,
            candidates=self.draft_token_buf[:bs],
            retrieve_index=self.retrieve_index_buf[:bs],
            retrieve_next_token=self.retrieve_next_token_buf[:bs],
            retrieve_next_sibling=self.retrieve_next_sibling_buf[:bs],
            target_predict=target_predict,
            topk=-1,
        )
        if broadcast_outputs:
            # All TP ranks must commit rank zero's sampled path, including its
            # hidden-state indices. Greedy requests retain the existing path.
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(accept_lens, src=0)
        # The verify kernel reports accepted drafts; UBTree's public accept_lens
        # includes the trailing bonus token.
        accept_lens.add_(1)
        commit_lens = self.commit_lens_buf[:bs]
        torch.mul(accept_lens, self.commit_gate_buf, out=commit_lens)

        safe_index = self.safe_accept_index_buf[:bs]
        safe_index.copy_(accept_index)
        safe_index.clamp_min_(0)

        compact_predict = self.compact_predict_buf[:bs]
        compact_predict.zero_()
        path_predict = self.path_predict_buf[:bs]
        torch.index_select(
            predict,
            0,
            safe_index.reshape(-1),
            out=path_predict.reshape(-1),
        )
        compact_predict[:, : self.max_depth].copy_(path_predict)

        compact_hidden = self._ensure_hidden_buffer(hidden)[: bs * self.max_depth]
        torch.index_select(
            hidden[:rows],
            0,
            safe_index.reshape(-1),
            out=compact_hidden,
        )

        bonus_index = self.bonus_index_buf[:bs]
        bonus_index.copy_(accept_lens.view(bs, 1))
        bonus_index.sub_(1)
        torch.gather(
            compact_predict,
            1,
            bonus_index,
            out=self.bonus_tokens_i32_buf[:bs],
        )
        self.bonus_tokens_buf[:bs].copy_(self.bonus_tokens_i32_buf[:bs].reshape(-1))
        if self.folds_target_kv or self.folds_draft_kv or self.folds_kda_commit:
            if req_pool_indices is None or seq_lens is None:
                raise ValueError(
                    "folded KV commit requires graph-stable request metadata"
                )

        target_cache_loc_2d = self.target_cache_loc_buf[:bs]
        source_cache_loc_2d = self.source_cache_loc_buf[:bs]
        positions_2d = self.draft_positions_buf[:bs]
        draft_cache_loc_2d = self.draft_cache_loc_buf[:bs]
        if self.folds_target_kv and self.folds_draft_kv:
            if out_cache_loc is None:
                raise ValueError(
                    "folded target KV commit requires graph-stable cache locations"
                )
            prepare_unified_commit_metadata(
                req_pool_indices=req_pool_indices[:bs],
                target_req_to_token=self.req_to_token,
                draft_req_to_token=self.draft_req_to_token,
                seq_lens=self.seq_lens_buf[:bs],
                accept_index=accept_index,
                accept_lens=commit_lens,
                out_cache_loc=out_cache_loc[:rows],
                target_cache_loc_out=target_cache_loc_2d,
                source_cache_loc_out=source_cache_loc_2d,
                draft_positions_out=positions_2d,
                draft_cache_loc_out=draft_cache_loc_2d,
            )
        else:
            if self.folds_target_kv:
                if out_cache_loc is None:
                    raise ValueError(
                        "folded target KV commit requires graph-stable cache locations"
                    )
                prepare_target_kv_move_metadata(
                    req_pool_indices=req_pool_indices[:bs],
                    req_to_token=self.req_to_token,
                    seq_lens=self.seq_lens_buf[:bs],
                    accept_index=accept_index,
                    accept_lens=commit_lens,
                    out_cache_loc=out_cache_loc[:rows],
                    target_cache_loc_out=target_cache_loc_2d.reshape(-1),
                    source_cache_loc_out=source_cache_loc_2d.reshape(-1),
                )
            if self.folds_draft_kv:
                prepare_draft_commit_metadata(
                    seq_lens=self.seq_lens_buf[:bs],
                    req_pool_indices=req_pool_indices[:bs],
                    req_to_token=self.draft_req_to_token,
                    positions_out=positions_2d,
                    cache_loc_out=draft_cache_loc_2d,
                )

        if self.folds_target_kv:
            self.target_kvcache.move_kv_cache(
                target_cache_loc_2d.reshape(-1),
                source_cache_loc_2d.reshape(-1),
            )

        if self.folds_draft_kv:
            self.draft_commit(
                target_hidden=compact_hidden,
                cache_loc=draft_cache_loc_2d.reshape(-1),
                cache_loc_2d=draft_cache_loc_2d,
                positions=positions_2d.reshape(-1),
                commit_lens=commit_lens,
            )

        if self.folds_kda_commit:
            state_batch_indices = self.state_batch_indices_buf[:bs]
            torch.index_select(
                self.mamba_index_mapping,
                0,
                req_pool_indices[:bs],
                out=state_batch_indices,
            )
            has_track = mamba_track_indices is not None
            mamba_steps_to_track = None
            if has_track:
                _, mamba_steps_to_track = fused_commit_track_indices(
                    accept_index,
                    # This helper requires a nonempty path. Only the commit
                    # kernels consume the gated length; they skip zero rows
                    # before reading or writing temporal/conv/track state.
                    accept_lens,
                    self.seq_lens_buf[:bs],
                    self.node_count,
                    self.mamba_track_interval,
                    last_correct_out=self.last_correct_step_buf,
                    track_steps_out=self.mamba_steps_to_track_buf,
                )
            commit_kda_tree_after_verify(
                spec_state=self.kda_spec_state,
                state_batch_indices=state_batch_indices,
                factor_cache_indices=self.factor_cache_indices_buf[:bs],
                accept_index=accept_index,
                accept_lens=commit_lens,
                tree_width=self.node_count,
                mamba_track_indices=(mamba_track_indices[:bs] if has_track else None),
                mamba_steps_to_track=mamba_steps_to_track,
                null_block_id=-1,
            )

        # Publish last: eager seq_lens may alias this previous step's output.
        # A disarmed greedy tail must preserve the staged prefix length.
        torch.add(
            self.seq_lens_buf[:bs],
            commit_lens,
            out=self.new_seq_lens_buf[:bs],
        )

    def read_result(self, bs: int) -> UBTreeVerifyResult:
        hidden_rows = bs * self.max_depth
        if self.compact_hidden_buf is None:
            raise RuntimeError("UBTree epilogue has not run yet")
        return UBTreeVerifyResult(
            predict=self.compact_predict_buf[:bs].reshape(-1),
            accept_lens=self.accept_lens_buf[:bs],
            accept_index=self.accept_index_buf[:bs],
            hidden=self.compact_hidden_buf[:hidden_rows],
            bonus_tokens=self.bonus_tokens_buf[:bs],
            new_seq_lens=self.new_seq_lens_buf[:bs],
        )


__all__ = ["UBTreeVerifyEpilogue", "UBTreeVerifyResult"]
