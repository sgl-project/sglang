"""DDTreeWorker: full-tree speculative decoding worker (extends DFlashWorkerV2).

DDTreeWorker subclasses DFlashWorkerV2 so draft-worker initialization, memory
pool setup, and CUDA graph handling stay aligned with DFLASH. DDTree keeps its
own non-overlap decode loop because its verification step builds and verifies a
tree instead of a linear DFLASH block:
- spine mode (ddtree_budget <= block_size-1): a linear chain, using the same
  target-verify pattern as DFLASH.
- full-tree mode (ddtree_budget > block_size-1): builds a branching tree from
  the draft distributions, converts it to EAGLE's LCRS format, and verifies it
  with EAGLE's tree-verify kernel + tree-aware Mamba scan (the only path that
  handles hybrid Mamba models). Accepts the longest matching path.

Only the non-overlap scheduler path is supported. See ddtree_utils.py for tree
construction and ddtree_info.py (DDTreeVerifyInput) for verification.
"""

import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.ddtree_info import DDTreeDraftInput, DDTreeVerifyInput
from sglang.srt.speculative.ddtree_utils import (
    build_ddtree_tree,
    build_eagle_tree_format,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DDTreeWorker(DFlashWorkerV2):
    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker,
    ):
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.tree_budget = getattr(server_args, "speculative_ddtree_budget", None)
        if self.tree_budget is None:
            self.tree_budget = self.block_size - 1

        self.max_tree_nodes = self.tree_budget + 1

        # Pre-allocated tree-build output buffers (reused across steps).
        _max_bs = getattr(server_args, "cuda_graph_max_bs", None) or 64
        _budget = self.tree_budget
        _mn = self.max_tree_nodes
        dev = self.device
        self._tree_node_token_ids_buf: torch.Tensor = torch.zeros(
            _max_bs, _budget, dtype=torch.long, device=dev
        )
        self._tree_node_depths_buf: torch.Tensor = torch.zeros(
            _max_bs, _budget, dtype=torch.long, device=dev
        )
        self._tree_parents_buf: torch.Tensor = torch.full(
            (_max_bs, _mn), -1, dtype=torch.long, device=dev
        )
        self._tree_visibility_buf: torch.Tensor = torch.zeros(
            _max_bs, _mn, _mn, dtype=torch.bool, device=dev
        )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DDTree worker. block_size=%s, tree_budget=%s, max_tree_nodes=%s",
                self.block_size,
                self.tree_budget,
                self.max_tree_nodes,
            )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DDTreeDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError(
                "DDTREE batch has grammar constraints, but scheduler should have rejected this request."
            )

        bs = batch.batch_size()

        # --- 1) Append target hidden to draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DDTREE requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block (reuse parent's shared implementation).
        draft_hidden, positions_2d, block_ids = self._run_draft_forward(
            batch, draft_input
        )

        # --- 3) Spine-mode fast path: a degenerate chain tree. ---
        if self.tree_budget <= self.block_size - 1:
            # Spine mode: linear chain. Use DDTreeVerifyInput with a chain LCRS
            # topology (each node has exactly one child = next index) so the V2
            # protocol path is uniform with full-tree mode. Reuse DFLASH's exact
            # token assembly (correct vocab-parallel sampler + the `positions_2d`
            # from `_run_draft_forward`) so AL matches.
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
            ).view(bs, self.block_size - 1)
            draft_tokens = self._draft_block_tokens_buf[:bs]
            draft_tokens[:, 0].copy_(block_ids[:, 0])
            draft_tokens[:, 1:].copy_(draft_next)

            # block_size == max_tree_nodes when budget == block_size - 1 (the
            # common spine case). For smaller budgets, truncate to max_tree_nodes.
            D = self.max_tree_nodes
            draft_token = draft_tokens[:, :D].reshape(-1)
            positions = positions_2d[:, :D].reshape(-1)

            # Chain LCRS encoding: node k's only child is node k+1; no siblings.
            # retrive_index is identity (local k -> global b*D + k); see
            # DDTreeVerifyInput.verify for indexing.
            arange_d = torch.arange(D, dtype=torch.long, device=batch.device)
            retrive_index = (
                torch.arange(bs, dtype=torch.long, device=batch.device).unsqueeze(1) * D
            ) + arange_d.unsqueeze(0)
            chain_next = torch.where(
                arange_d < D - 1, arange_d + 1, torch.full_like(arange_d, -1)
            )
            retrive_next_token = chain_next.unsqueeze(0).expand(bs, D).contiguous()
            retrive_next_sibling = torch.full(
                (bs, D), -1, dtype=torch.long, device=batch.device
            )

            # Per-req visibility for the flat tree mask: node k sees prefix + nodes
            # 0..k (its ancestors along the chain). actual_tree_sizes = full D.
            actual_tree_sizes = torch.full(
                (bs,), D, dtype=torch.long, device=batch.device
            )
            visibility = (
                torch.tril(torch.ones(D, D, dtype=torch.bool, device=batch.device))
                .unsqueeze(0)
                .expand(bs, D, D)
                .contiguous()
            )
            custom_mask = self._build_flat_tree_mask(
                batch=batch, visibility=visibility, actual_tree_sizes=actual_tree_sizes
            )

            # eff_topk=1 here: a degenerate chain doesn't need the tree-aware
            # mamba scan (linear processing already follows the chain), so leave
            # the mamba metadata path on the cheap linear kernel.
            verify_input = DDTreeVerifyInput(
                draft_token=draft_token,
                positions=positions,
                draft_token_num=D,
                tree_budget=self.tree_budget,
                child_maps=[{} for _ in range(bs)],  # unused (chain has no branching for diagnostics)
                actual_tree_sizes=actual_tree_sizes,
                custom_mask=custom_mask,
                tree_is_spine=True,
                topk=1,
                spec_steps=self.tree_budget,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
            )
            verify_input.prepare_for_verify(batch, self.page_size)

            batch.forward_mode = (
                ForwardMode.TARGET_VERIFY
                if not batch.forward_mode.is_idle()
                else ForwardMode.IDLE
            )
            batch.spec_info = verify_input
            batch.return_hidden_states = False
            return

        # --- 4) Full tree path (budget > L): compute logits, beam search, compile mask. ---
        draft_logits = self._compute_draft_logits(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1, -1)

        # Diagnostic (SGLANG_DDTREE_RANK_PROBE): per-position draft top-32 token
        # ids, so verify() can bucket where the target's correct token ranks in
        # the draft distribution (top1/2/4/8/16/32) — directly answers "does more
        # budget/width help, or is draft quality the ceiling".
        from sglang.srt.environ import envs

        _draft_pos_topk = None
        if envs.SGLANG_DDTREE_RANK_PROBE.get():
            _draft_pos_topk = torch.topk(
                draft_logits, k=min(32, draft_logits.shape[-1]), dim=-1
            ).indices  # [bs, block_size-1, 32]

        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility,
            actual_tree_sizes,
        ) = build_ddtree_tree(
            draft_logits=draft_logits,
            tree_budget=self.tree_budget,
            device=batch.device,
            _out_node_token_ids=self._tree_node_token_ids_buf,
            _out_node_depths=self._tree_node_depths_buf,
            _out_parents=self._tree_parents_buf,
            _out_visibility=self._tree_visibility_buf,
        )

        # Build verify input ids / positions (node 0 = verified root token).
        D = self.max_tree_nodes
        verify_input_ids = torch.zeros(bs, D, dtype=torch.long, device=batch.device)
        verify_input_ids[:, 0] = draft_input.verified_id
        verify_input_ids[:, 1:] = node_token_ids
        verify_position_ids = torch.zeros(bs, D, dtype=torch.long, device=batch.device)
        verify_position_ids[:, 0] = batch.seq_lens
        verify_position_ids[:, 1:] = batch.seq_lens.unsqueeze(1) + node_depths

        # Convert DDTree's parent encoding to EAGLE's LCRS tree format so the
        # verify kernel + tree-aware mamba scan can consume it unchanged.
        retrive_index, retrive_next_token, retrive_next_sibling = (
            build_eagle_tree_format(
                parents=parents,
                actual_tree_sizes=actual_tree_sizes,
                draft_token_num=D,
                device=batch.device,
            )
        )

        # Build the flat bool tree attention mask in the DFLASH/EAGLE layout
        # (per req: q_len x kv_len flat; prefix all visible, tree part = tree
        # ancestor visibility). This matches what generate_attn_arg_prefill and
        # the triton extend kernel expect (proven on this backend by spine).
        custom_mask = self._build_flat_tree_mask(
            batch=batch, visibility=visibility, actual_tree_sizes=actual_tree_sizes
        )

        tree_is_spine = all(
            all(len(children) <= 1 for children in cm.values()) for cm in child_maps
        )
        # Branching factor signal: topk>1 triggers the tree-aware mamba scan in
        # hybrid_linear_attn_backend (gated on spec_info.topk > 1). Use 2 for any
        # real tree; spine stays 1 to take the cheap chain path.
        # Always use topk=2 (tree-aware mamba scan) in the full-tree path, even
        # when beam search happens to produce a degenerate chain (tree_is_spine).
        # Reason: cudagraph captures the tree-scan kernel (topk=2). If a runtime
        # step flips to topk=1, the captured graph still executes the tree kernel
        # but _replay_metadata skips filling retrieve_parent_token → the kernel
        # reads stale parent indices → mamba state corruption → output collapse.
        # Tree-scan is correct for chains too (parent = previous node in chain),
        # so forcing topk=2 is safe and avoids the capture/replay mismatch.
        eff_topk = 2

        verify_input = DDTreeVerifyInput(
            draft_token=verify_input_ids.reshape(-1),
            positions=verify_position_ids.reshape(-1),
            draft_token_num=D,
            tree_budget=self.tree_budget,
            child_maps=child_maps,
            actual_tree_sizes=actual_tree_sizes,
            custom_mask=custom_mask,
            tree_is_spine=tree_is_spine,
            topk=eff_topk,
            spec_steps=self.tree_budget,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
        )
        verify_input._draft_pos_topk = _draft_pos_topk  # diagnostic (or None)
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _build_flat_tree_mask(self, *, batch, visibility, actual_tree_sizes):
        """Flat bool tree attention mask, DFLASH/EAGLE layout.

        Per request, a q_len x kv_len boolean block (q_len = max_tree_nodes,
        kv_len = prefix_len + q_len), flattened and concatenated across requests:
          - prefix columns (k < prefix_len): all True (every node sees history)
          - tree columns (k >= prefix_len): visibility[q, k-prefix_len]
        Padding nodes (>= actual_tree_size) stay fully masked.
        """
        device = batch.device
        D = self.max_tree_nodes
        chunks = []
        seq_lens_cpu = batch.seq_lens.detach().cpu().tolist()
        sizes_cpu = actual_tree_sizes.detach().cpu().tolist()
        for b, prefix_len in enumerate(seq_lens_cpu):
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + D
            m = torch.zeros(D, kv_len, dtype=torch.bool, device=device)
            actual = int(sizes_cpu[b])
            # Prefix visible for all actual nodes.
            m[:actual, :prefix_len_i] = True
            # Tree visibility for the actual nodes.
            m[:actual, prefix_len_i : prefix_len_i + actual] = visibility[
                b, :actual, :actual
            ].to(torch.bool)
            chunks.append(m.flatten())
        return (
            torch.cat(chunks, dim=0)
            if chunks
            else torch.empty((0,), dtype=torch.bool, device=device)
        )

    def forward_batch_generation(
        self, batch: ScheduleBatch, **kwargs
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "DDTREE batch requested return_logprob, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = ForwardBatch.init_new(
                batch, self.target_worker.model_runner
            )
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                None, forward_batch=model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DDTREE requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DDTREE expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DDTreeDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
                next_draft_input=draft_input,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DDTreeDraftInput):
            raise RuntimeError(
                "DDTREE decode requires DDTreeDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = ForwardBatch.init_new(
            batch, self.target_worker.model_runner
        )
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info

        # Mamba/linear-attention models need an explicit recurrent-state commit
        # after verify (the state is not committed during the verify forward).
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result = self.target_worker.forward_batch_generation(
            None, forward_batch=model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # Spine and full-tree paths both produce DDTreeVerifyInput now (spine
        # uses a chain LCRS topology). The verify() / mamba commit logic is
        # uniform — the tree-aware mamba scan is gated off via
        # ``spec_info.topk == 1`` for spine inside hybrid_linear_attn_backend.
        assert isinstance(verify_input, DDTreeVerifyInput)
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            model_runner=self.target_worker.model_runner,
        )
        if need_mamba_verify_commit:
            # Tree-aware mamba commit: the accepted path's last node is at an
            # arbitrary step in the flat verify buffer (not commit_len-1), so we
            # must use EAGLE's accepted_indices-based step mapping, NOT the
            # linear DFLASH commit.
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_tree(
                batch=batch,
                verify_input=verify_input,
                seq_lens_pre_verify=seq_lens_pre_verify,
            )

        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DDTREE verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_input.accepted_tokens,
            accept_lens=commit_lens.detach().cpu(),
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=verify_input.draft_token_num,
            new_seq_lens=batch.seq_lens.clone(),
            next_draft_input=draft_input,
        )

    def _update_target_mamba_state_tree(
        self, *, batch, verify_input, seq_lens_pre_verify
    ):
        """Tree-aware mamba state commit (mirrors EAGLE _mamba_verify_update).

        For a branching tree the last accepted node sits at an arbitrary step in
        the flat verify buffer, so accepted_steps must be derived from the
        accept_index layout, not commit_len-1. verify_input must have populated
        accept_index_flat (batch-global accepted slots, accept order) and
        accept_length_per_req ([bs] accepted-draft counts).
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        accept_index_flat = verify_input.accept_index_flat
        accept_len = verify_input.accept_length_per_req  # [bs], drafts only
        device = batch.seq_lens.device
        D = verify_input.draft_token_num

        accepted_length = accept_len.to(device=device, dtype=torch.int64) + 1
        cumulative = torch.cumsum(accepted_length, dim=0)
        # Per-request start offset into the flat accepted-slots buffer.
        accepted_indices_start = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=device), cumulative[:-1]]
        )
        offset = torch.arange(
            0, len(batch.seq_lens) * D, step=D, dtype=torch.int64, device=device
        )
        accept_index_flat = accept_index_flat.to(device)
        # Last accepted node's per-request step offset (0..D-1).
        accepted_steps = accept_index_flat[cumulative - 1] - offset

        # Prefix-cache state tracking: if a request crossed a mamba track
        # interval during this verify step, record the state at the crossing.
        # Tree-correct version of DFLASH's linear logic (uses accepted_indices).
        mamba_steps_to_track = None
        mamba_track_indices = getattr(batch, "mamba_track_indices", None)
        if mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            # The i-th committed token maps to the accepted node at
            # accepted_indices[i + start]; convert to a per-req step offset.
            can_track_mask = to_track_mask & (to_track_ith < accepted_length)
            track_flat_idx = torch.clamp(
                to_track_ith + accepted_indices_start, max=accept_index_flat.numel() - 1
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                accept_index_flat[track_flat_idx] - offset,
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accepted_steps,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def _greedy_sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,  # [total_tokens, hidden]
        lm_head,
    ) -> torch.Tensor:
        """Return argmax token IDs (long).  Used by spine fast path."""
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        weight = lm_head.weight
        weight_dtype = weight.dtype

        if hidden_states.dtype != weight_dtype:
            hidden_states = hidden_states.to(weight_dtype)

        local_logits = torch.mm(hidden_states, weight.t())

        if tp_size == 1:
            return local_logits.argmax(dim=-1)

        # TP > 1: per-rank argmax + all-gather of max/argmax
        shard = lm_head.shard_indices
        local_max, local_argmax = local_logits.max(dim=-1)

        gathered_max = torch.empty(
            tp_size,
            hidden_states.shape[0],
            dtype=local_max.dtype,
            device=local_max.device,
        )
        tp_group.all_gather_into_tensor(gathered_max, local_max)
        global_max_idx = gathered_max.argmax(dim=0)

        gathered_argmax = torch.empty(
            tp_size,
            hidden_states.shape[0],
            dtype=local_argmax.dtype,
            device=local_argmax.device,
        )
        tp_group.all_gather_into_tensor(gathered_argmax, local_argmax)
        global_argmax = gathered_argmax[
            global_max_idx,
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
        ]

        # Adjust for per-rank vocab shard offset
        num_org = int(shard.num_org_elements)
        return torch.where(
            global_argmax < num_org,
            global_argmax + int(shard.padding_offset),
            global_argmax - num_org + int(shard.org_vocab_start_index),
        )

    def _compute_draft_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
    ) -> torch.Tensor:
        """Compute global draft logits (raw, not log-softmax).

        Returns raw float logits.  build_ddtree_tree will compute log-probs
        itself via logsumexp, so we skip the exp→log chain here.
        """
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype

        if hidden_states.dtype != weight_dtype:
            hidden_states = hidden_states.to(weight_dtype)

        local_logits = torch.mm(hidden_states, weight.t())

        # Fast path: single-GPU — return raw logits directly.
        if tp_size == 1:
            return local_logits.float()

        # TP > 1: materialize logits in global vocab-id order.  The DDTree
        # builder uses torch.topk(...).indices as token ids, so concatenating
        # rank-local/padded shard columns would produce shard positions rather
        # than global vocab ids.
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        local_dim = local_logits.shape[-1]
        local_global_ids = torch.full(
            (local_dim,), -1, dtype=torch.int64, device=local_logits.device
        )
        if num_org > 0:
            local_global_ids[:num_org] = org_vocab_start + torch.arange(
                num_org, dtype=torch.int64, device=local_logits.device
            )
        if num_added > 0:
            local_global_ids[num_org_padded : num_org_padded + num_added] = (
                added_vocab_start
                + torch.arange(num_added, dtype=torch.int64, device=local_logits.device)
            )

        logits_for_gather = local_logits.float()
        valid_local = local_global_ids >= 0
        if not bool(valid_local.all()):
            logits_for_gather = logits_for_gather.masked_fill(
                ~valid_local.unsqueeze(0), float("-inf")
            )

        gathered_logits = torch.empty(
            tp_size,
            hidden_states.shape[0],
            local_dim,
            dtype=logits_for_gather.dtype,
            device=logits_for_gather.device,
        )
        gathered_ids = torch.empty(
            tp_size,
            local_dim,
            dtype=local_global_ids.dtype,
            device=local_global_ids.device,
        )
        tp_group.all_gather_into_tensor(gathered_logits, logits_for_gather.contiguous())
        tp_group.all_gather_into_tensor(gathered_ids, local_global_ids.contiguous())

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        global_logits = torch.full(
            (hidden_states.shape[0], vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        flat_ids = gathered_ids.reshape(1, -1).expand(hidden_states.shape[0], -1)
        flat_logits = gathered_logits.permute(1, 0, 2).reshape(hidden_states.shape[0], -1)
        valid = (flat_ids >= 0) & (flat_ids < vocab_size)
        global_logits.scatter_reduce_(
            dim=1,
            index=flat_ids.clamp(0, vocab_size - 1),
            src=flat_logits.masked_fill(~valid, float("-inf")),
            reduce="amax",
            include_self=True,
        )

        return global_logits

    # -----------------------------------------------------------------
    # DDTree uses DFlashWorkerV2 for common draft-worker setup, but keeps a
    # non-overlap decode loop with tree verification. These helpers provide the
    # two DDTree-specific operations that are not exposed by DFlashWorkerV2's
    # overlap-oriented API: running one non-causal draft block and materializing
    # newly committed target hidden states into the draft KV cache.
    # -----------------------------------------------------------------

    def _run_draft_forward(
        self, batch: ScheduleBatch, draft_input: DDTreeDraftInput
    ):
        """Run one non-causal draft block forward with the draft model.

        Returns:
            draft_hidden: [bs, block_size, hidden] draft hidden states.
            positions_2d: [bs, block_size] absolute positions (for RoPE).
            block_ids:    [bs, block_size] draft input token ids (slot 0 = current).
        """
        bs = batch.batch_size()
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # `target_prefix_lens` stay absolute for RoPE; `draft_prefix_lens` are the
        # logical resident lengths in the draft-local cache.
        target_prefix_lens = batch.seq_lens
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )
        positions = positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DDTREE draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            draft_spec_info = self._draft_block_spec_info
            seq_lens = draft_prefix_lens
            seq_lens_sum = int(draft_prefix_lens.sum().item())
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DDTREE draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        return draft_hidden, positions_2d, block_ids

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DDTreeDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        Must run before exposing new tokens to radix cache (prefix hits) so
        another request can't reuse target KV indices without having draft KV
        values. DFlashWorkerV2 exposes explicit-location materialization for
        overlap scheduling; DDTree's non-overlap path stores pending context in
        DDTreeDraftInput and materializes it here in one step.
        """
        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DDTREE draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DDTREE ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DDTREE draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)
            ctx_positions = pos2d.reshape(-1)
        else:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DDTREE invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]
            pos2d = ctx_start[:, None] + r
            mask = r < ctx_lens[:, None]

            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DDTREE target hidden KV append",
            )
            ctx_positions = pos2d[mask]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DDTREE ctx_hidden/cache_loc mismatch: "
                    f"{ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DDTREE fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        if self.use_compact_draft_cache:
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]
