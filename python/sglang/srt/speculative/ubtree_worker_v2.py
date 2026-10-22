# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See ubtree_components/LICENSE for the original license.

"""DFlash proposals with fixed-width irregular-tree verification and V2 relay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.kernels.ops.speculative.ubtree import (
    prepare_draft_commit_metadata,
    prepare_target_kv_move_metadata,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_exec, get_spec, mamba_track_grid
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
)
from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config
from sglang.srt.speculative.ubtree_components.draft_graph import (
    UBTreeDraftSegmentGraphRunner,
)
from sglang.srt.speculative.ubtree_components.epilogue import UBTreeVerifyEpilogue
from sglang.srt.speculative.ubtree_components.target_kv_graph import (
    UBTreeTargetKVMoveGraphRunner,
)
from sglang.srt.speculative.ubtree_components.tree import (
    UBTreeBatch,
    build_sglang_full_mask,
    build_ubtree_batch,
    compact_tree_values_to_front,
)
from sglang.srt.speculative.ubtree_components.tree_graph import (
    UBTreeProposalGraphRunner,
)
from sglang.srt.speculative.ubtree_components.verify_metadata import (
    UBTreeVerifyMetadataBuffers,
)
from sglang.srt.speculative.ubtree_components.vocab import vocab_parallel_topk
from sglang.srt.speculative.ubtree_info import UBTreeVerifyInput
from sglang.srt.utils.async_probe import maybe_detect_oob

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs


class UBTreeWorkerV2(DFlashWorkerV2):
    """DFlash draft plus one-forward irregular-tree target verification."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        # The target runner needs ``topk > 1`` to select SGLang's parent-aware
        # tree-verification attention path.  The DFlash draft is a linear
        # 16-slot block, however, and its ``custom_mask`` is intentionally
        # ``None``. Temporarily expose the linear geometry through SGLang's
        # published spec config while DFlashWorkerV2 constructs its draft
        # runner; the scoped override restores the target tree geometry.
        draft_block_size = get_spec().speculative_dflash_block_size
        if draft_block_size is None:
            raise ValueError("UBTREE requires a resolved DFlash block size")
        with get_spec().override(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=int(draft_block_size),
        ):
            super().__init__(server_args, gpu_id, nccl_port, target_worker)
        self.speculative_algorithm = SpeculativeAlgorithm.from_string("UBTREE")
        self.ubtree_config = parse_ubtree_config(
            self.draft_model_runner.model_config.hf_config
        )
        resolved_verify_node_count = int(get_spec().speculative_num_draft_tokens)
        if not 0 < resolved_verify_node_count <= self.ubtree_config.verify_node_count:
            raise ValueError(
                "UBTREE runtime verify width exceeds checkpoint tree budget: "
                f"{resolved_verify_node_count} > {self.ubtree_config.verify_node_count}"
            )

        if int(self.block_size) != self.ubtree_config.block_size:
            raise ValueError(
                "UBTREE draft runner block width was configured incorrectly"
            )
        # The inherited field is reported to the scheduler as target verify
        # width; the DFlash runner itself keeps ``block_size`` at 16.
        self.speculative_num_draft_tokens = resolved_verify_node_count

        selector = getattr(self.draft_model, "ubtree_selector", None)
        if selector is None:
            raise ValueError(
                "UBTREE draft checkpoint must use a registered UBTree architecture"
            )
        if selector.tied_token_factorizer is not None:
            target_model = self.target_worker.model_runner.model
            target_embedding = target_model.get_input_embeddings()
            vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
            embedding_weight = getattr(target_embedding, "weight", None)
            if embedding_weight is None:
                raise RuntimeError(
                    "UBTREE requires target input embeddings with weights"
                )
            embedding_tp_size = int(getattr(target_embedding, "tp_size", 1))
            if embedding_tp_size == 1 and int(embedding_weight.shape[0]) >= vocab_size:
                selector.materialize_cache(embedding_weight[:vocab_size])
            elif hasattr(target_embedding, "shard_indices"):
                selector.materialize_cache_from_vocab_parallel_embedding(
                    target_embedding,
                )
            else:
                selector.materialize_cache_from_embedding(
                    target_embedding,
                    device=self.device,
                )
        self._warned_seq_lens_copy = False
        self._fa3_tree_only_mask = envs.SGLANG_OPT_UBTREE_FA3_TREE_ONLY_MASK.get()
        self._verify_node_buckets = (resolved_verify_node_count,)
        self._verify_epilogues = {}
        self._verify_metadata_buffers = {}
        self._verify_positions_buf = None
        self._draft_segment_graph_runner = None
        self._proposal_graph_runner = None
        self._draft_commit_positions_buf = None
        self._draft_commit_cache_loc_buf = None
        self._target_move_tgt_cache_loc_buf = None
        self._target_move_src_cache_loc_buf = None
        self._target_kv_move_graph_runner = None
        exec_cfg = get_exec()
        if not exec_cfg.graph.disable_cuda_graph:
            decode_bs = exec_cfg.graph.cuda_graph_config.decode.bs or ()
            if decode_bs:
                max_decode_bs = max(decode_bs)
                self._ensure_draft_commit_buffers(max_decode_bs)
                self._ensure_target_move_buffers(
                    max_decode_bs, self.ubtree_config.block_size
                )
                self._ensure_verify_positions_buffer(max_decode_bs)
                self._verify_metadata_buffers = {
                    width: UBTreeVerifyMetadataBuffers(
                        batch_sizes=decode_bs,
                        num_nodes=width,
                        max_context_len=int(
                            self.target_worker.model_runner.model_config.context_len
                        ),
                        device=self.device,
                    )
                    for width in self._verify_node_buckets
                }
                self._verify_epilogues = {
                    width: UBTreeVerifyEpilogue(
                        max_bs=max(decode_bs),
                        node_count=width,
                        max_depth=self.ubtree_config.block_size,
                        device=self.device,
                    )
                    for width in self._verify_node_buckets
                }
                # This must be registered before Scheduler captures target graphs.
                self.model_runner.capture_tail_hooks.append(
                    self._capture_verify_epilogue
                )

    def _capture_verify_epilogue(
        self, runner, out, forward_batch, num_tokens: int
    ) -> None:
        bs = int(forward_batch.batch_size)
        if bs <= 0 or num_tokens % bs:
            return
        epilogue = self._verify_epilogues.get(num_tokens // bs)
        if epilogue is not None:
            epilogue.capture_hook(runner, out, forward_batch, num_tokens)

    def init_attention_backends(self):
        # SGLang initializes the draft attention backend only after the worker
        # constructor returns. Keep the draft backend on linear DFlash geometry
        # while it snapshots topk and verify width.
        with get_spec().override(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=int(self.block_size),
        ):
            return super().init_attention_backends()

    def _capture_proposal_tail(self, draft_output, input_ids, forward_batch):
        batch_size = int(forward_batch.batch_size)
        proposal_runner = self._proposal_graph_runner
        if proposal_runner is None:
            return None
        hidden = draft_output.hidden_states
        if hidden is None:
            raise RuntimeError("UBTREE draft graph tail received no hidden states")
        hidden = hidden.view(batch_size, self.block_size, -1)[:, 1:]
        bonus_tokens = input_ids.view(batch_size, self.block_size)[:, 0]
        tree = proposal_runner.build(hidden, bonus_tokens)

        # Fold fixed-address pre-verify work into the draft graph tail. The
        # dynamic-length mask itself stays eager so long-context requests do not
        # replay a max-context grid.
        self._ensure_verify_positions_buffer(batch_size)
        positions_2d = self._verify_positions_buf[:batch_size, : tree.node_count]
        torch.add(forward_batch.seq_lens[:, None], tree.depths, out=positions_2d)
        tree.verify_positions = positions_2d.reshape(-1)

        metadata_buffers = self._verify_metadata_buffers.get(tree.node_count)
        verify_epilogue = self._verify_epilogues.get(tree.node_count)
        if metadata_buffers is not None and verify_epilogue is not None:
            metadata_buffers.stage_prefix_lens(forward_batch.seq_lens, batch_size)
            verify_epilogue.begin_tree_step(tree, forward_batch.seq_lens)
            tree.preverify_staged = True
        return tree

    def init_cuda_graphs(self) -> None:
        """Initialize exact-shape target and segmented draft CUDA Graphs.

        FlashInfer attention remains eager so its plan follows live KV lengths.
        Fixed-shape draft work around attention, selector/top-k, and proposal-tree
        construction is captured separately. Target KV movement is captured in an
        exact-batch graph; dynamic mask packing and draft KV commit remain eager.
        """
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        exec_cfg = get_exec()
        if exec_cfg.graph.disable_cuda_graph:
            return
        decode_bs = exec_cfg.graph.cuda_graph_config.decode.bs or ()
        tp_group = get_tp_group()
        # The distributed capture context used by the draft graph enables
        # graph-safe TP collectives.  This lets proposal selection retain exact
        # global vocabulary Top-K while its all-gathers are captured as part of
        # the draft graph tail.
        proposal_graph_bs = decode_bs
        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        selector = self.draft_model.ubtree_selector
        logger.info("UBTREE capturing segmented draft CUDA Graphs")
        target_embedding = target_model.get_input_embeddings()
        embedding_tp_size = int(getattr(target_embedding, "tp_size", 1))
        draft_graph_bs = decode_bs
        if embedding_tp_size > 1:
            logger.info(
                "UBTREE capturing the TP%s vocab-parallel embedding and proposal "
                "collectives in the full draft CUDA Graph",
                embedding_tp_size,
            )
        if draft_graph_bs:
            self.draft_model_runner.attn_backend.init_cuda_graph_state(
                max(draft_graph_bs), max(draft_graph_bs) * self.block_size
            )
        self._draft_segment_graph_runner = UBTreeDraftSegmentGraphRunner(
            model=self.draft_model,
            input_embedding=target_embedding,
            attn_backend=(
                self.draft_model_runner.attn_backend if draft_graph_bs else None
            ),
            batch_sizes=draft_graph_bs,
            block_size=self.block_size,
            hidden_size=int(self.draft_model_runner.model_config.hidden_size),
            hidden_dtype=self.draft_model_runner.dtype,
            device=self.device,
        )
        self.draft_model.attach_segment_graph_runner(self._draft_segment_graph_runner)
        # Target KV movement is folded into the target verify graph tail for
        # greedy CUDA-graph decode. Keep the standalone runner as a fallback for
        # graph-ineligible paths.
        logger.info(
            "UBTREE target KV movement is folded into the target verify CUDA "
            "Graph tail; capturing standalone fallback graphs"
        )
        self._target_kv_move_graph_runner = UBTreeTargetKVMoveGraphRunner(
            kvcache=self.token_to_kv_pool_allocator.get_kvcache(),
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            batch_sizes=decode_bs,
            accept_width=self.ubtree_config.block_size,
            verify_node_counts=self._verify_node_buckets,
            device=self.device,
        )
        node_count = self.speculative_num_draft_tokens
        self._proposal_graph_runner = UBTreeProposalGraphRunner(
            batch_sizes=(() if draft_graph_bs else proposal_graph_bs),
            depth_count=self.ubtree_config.depth_count,
            hidden_size=int(selector.hidden_projection.in_features),
            hidden_dtype=selector.hidden_projection.weight.dtype,
            candidate_k=self.ubtree_config.candidate_k,
            vocab_size=int(self.target_worker.model_runner.model_config.vocab_size),
            lm_head=lm_head,
            selector=selector,
            budget=node_count - 1,
            supertree_width=self.ubtree_config.supertree_width,
            depth_bonus=self.ubtree_config.depth_bonus,
            device=self.device,
            tp_group=tp_group,
        )
        if draft_graph_bs:
            self._draft_segment_graph_runner.attach_capture_tail(
                self._capture_proposal_tail
            )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # DFlash only needs these objects while allocating its aliased draft
        # pool, whereas UBTree also consumes them later in the EAGLE verify
        # and accepted-path compaction helpers.
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        result = super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if req_to_token_pool is not None and token_to_kv_pool_allocator is not None:
            target_kvcache = token_to_kv_pool_allocator.get_kvcache()
            draft_pool = self.draft_model_runner.token_to_kv_pool
            can_fold_draft_kv = (
                self._use_fused_kv_materialize
                and self._fused_kv_helper is not None
                and hasattr(draft_pool, "set_kv_buffer_prefix_valid")
            )
            mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
            can_fold_kda_commit = bool(
                mamba_pool is not None
                and getattr(mamba_pool, "factorized_kda_tree", False)
            )
            tree_spec_state = (
                req_to_token_pool.get_speculative_mamba2_params_all_layers()
                if can_fold_kda_commit
                else None
            )
            mamba_mapping = getattr(
                req_to_token_pool,
                "req_index_to_mamba_index_mapping",
                None,
            )
            track_interval = mamba_track_grid(token_to_kv_pool_allocator.page_size)
            for epilogue in self._verify_epilogues.values():
                epilogue.attach_target_kv_commit(
                    target_kvcache=target_kvcache,
                    req_to_token=req_to_token_pool.req_to_token,
                )
                if can_fold_draft_kv:
                    epilogue.attach_draft_kv_commit(
                        draft_commit=self._append_target_hidden_to_draft_kv_by_loc,
                        req_to_token=req_to_token_pool.req_to_token,
                    )
                if (
                    can_fold_kda_commit
                    and mamba_mapping is not None
                    and int(tree_spec_state.tree_rawv.shape[-2]) >= epilogue.node_count
                ):
                    epilogue.attach_kda_tree_commit(
                        spec_state=tree_spec_state,
                        mamba_index_mapping=mamba_mapping,
                        mamba_track_interval=track_interval,
                    )
            if self._verify_epilogues and self._target_tp_rank == 0:
                if can_fold_draft_kv:
                    logger.info(
                        "UBTREE draft hidden/KV commit is folded into the target "
                        "verify CUDA Graph tail"
                    )
                else:
                    logger.info(
                        "UBTREE draft hidden/KV commit keeps the standalone "
                        "fallback because the draft pool/materializer is not "
                        "graph-foldable"
                    )
                if can_fold_kda_commit:
                    logger.info(
                        "UBTREE factorized KDA accepted-path commit is folded "
                        "into the target verify CUDA Graph tail"
                    )
        return result

    def _draft_tree(
        self, batch, draft_input: DFlashDraftInputV2
    ) -> tuple[UBTreeBatch, torch.Tensor, torch.Tensor]:
        """Run the DFlash block and turn its parallel hidden rows into trees."""

        bs = len(batch.seq_lens)
        block_size = int(self.block_size)
        device = self.device
        prefix_lens = batch.seq_lens
        self._ensure_draft_block_buffers(bs)
        block_ids = self._draft_block_ids_buf[:bs]
        positions_2d = self._draft_block_positions_buf[:bs]
        cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]

        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as error:
                self._use_triton_prepare_block = False
                logger.warning(
                    "UBTREE Triton prepare_block failed; falling back to eager path: %s",
                    error,
                )
        if not self._use_triton_prepare_block:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
            )
            draft_end = prefix_lens + block_size
            draft_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=draft_end,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            cache_loc_2d.copy_(draft_cache_loc.view(bs, block_size))
        draft_cache_loc = cache_loc_2d.reshape(-1)

        target_model = self.target_worker.model_runner.model
        positions = positions_2d.reshape(-1)
        use_draft_graph = (
            self._draft_segment_graph_runner is not None
            and self._draft_segment_graph_runner.can_run(bs)
        )
        input_embeds = None
        if not use_draft_graph:
            embed_module = target_model.get_input_embeddings()
            input_embeds = embed_module(block_ids).reshape(bs * block_size, -1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if batch.seq_lens_cpu is not None:
            seq_lens_cpu.copy_(batch.seq_lens_cpu)
            seq_lens_cpu.add_(block_size)
            seq_lens_sum = int(seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            # Overlap scheduling may clear batch.seq_lens_cpu while the
            # scheduler-owned DFlash state retains the page-aligned host
            # planning bound.  This matches the native DFlash/DSpARK path.
            seq_lens_cpu.copy_(draft_input.nxt_kv_lens_cpu)
            seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        else:
            seq_lens_cpu.copy_(prefix_lens.to(device="cpu", dtype=torch.int32))
            seq_lens_sum = int(seq_lens_cpu.sum())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_output = self.draft_model_runner.forward(forward_batch)
        hidden = draft_output.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("UBTREE DFlash draft returned no hidden states")
        hidden = hidden.view(bs, block_size, -1)[:, 1:]
        folded_tree = (
            self._draft_segment_graph_runner.tail_output(bs)
            if use_draft_graph and self._draft_segment_graph_runner is not None
            else None
        )
        if folded_tree is not None:
            tree = folded_tree
        elif self._proposal_graph_runner is not None:
            tree = self._proposal_graph_runner(
                hidden=hidden, bonus_tokens=draft_input.bonus_tokens
            )
        else:
            unary_values, candidate_ids = vocab_parallel_topk(
                hidden.reshape(-1, hidden.shape[-1]),
                target_model.lm_head,
                k=self.ubtree_config.candidate_k,
                vocab_size=int(self.target_worker.model_runner.model_config.vocab_size),
            )
            candidate_shape = (bs, block_size - 1, self.ubtree_config.candidate_k)
            candidate_ids = candidate_ids.view(candidate_shape)
            edge_log_probs = self.draft_model.ubtree_selector.build_edge_log_probs(
                hidden,
                draft_input.bonus_tokens,
                candidate_ids,
                unary_candidate_logits=unary_values.view(candidate_shape),
            )
            tree = build_ubtree_batch(
                anchor_tokens=draft_input.bonus_tokens,
                candidate_ids=candidate_ids,
                edge_log_probs=edge_log_probs,
                budget=self.speculative_num_draft_tokens - 1,
                supertree_width=self.ubtree_config.supertree_width,
                depth_bonus=self.ubtree_config.depth_bonus,
            )
        return tree, prefix_lens, positions

    def _ensure_verify_positions_buffer(self, batch_size: int) -> None:
        capacity = (
            0
            if self._verify_positions_buf is None
            else int(self._verify_positions_buf.shape[0])
        )
        if capacity >= int(batch_size):
            return
        new_capacity = max(int(batch_size), capacity * 2 if capacity else 1)
        self._verify_positions_buf = torch.empty(
            (new_capacity, self.ubtree_config.verify_node_count),
            dtype=torch.int64,
            device=self.device,
        )

    def _make_verify_input(self, batch, tree: UBTreeBatch) -> UBTreeVerifyInput:
        positions = tree.verify_positions
        if positions is None:
            self._ensure_verify_positions_buffer(tree.batch_size)
            positions_2d = self._verify_positions_buf[
                : tree.batch_size, : tree.node_count
            ]
            torch.add(batch.seq_lens[:, None], tree.depths, out=positions_2d)
            positions = positions_2d.reshape(-1)
        metadata_buffers = self._verify_metadata_buffers.get(tree.node_count)
        tree_mask_mode = TreeMaskMode.FULL_MASK
        if self._fa3_tree_only_mask:
            # FA3 only needs the tree suffix. Prefix tokens are handled by its
            # normal target-verify metadata, so avoid materializing and copying
            # node * prefix_len all-true cells every decode step.
            metadata_buffers = None
            custom_mask = tree.visibility.reshape(-1)
            tree_mask_mode = TreeMaskMode.QLEN_ONLY
        elif (
            metadata_buffers is not None
            and batch.seq_lens_sum is not None
            and metadata_buffers.can_prepare(tree.batch_size, int(batch.seq_lens_sum))
        ):
            custom_mask = metadata_buffers.prepare_custom_mask(
                tree.visibility,
                batch.seq_lens,
                prefix_lens_sum=int(batch.seq_lens_sum),
                prefix_prepared=tree.preverify_staged,
            )
        elif batch.seq_lens_sum is not None:
            metadata_buffers = None
            custom_mask = build_sglang_full_mask(
                tree.visibility,
                batch.seq_lens,
                prefix_lens_sum=int(batch.seq_lens_sum),
            )
        else:
            metadata_buffers = None
            if batch.seq_lens_cpu is None:
                if not self._warned_seq_lens_copy and self._target_tp_rank == 0:
                    logger.warning(
                        "UBTREE has no host prefix-length total; copying the small "
                        "length vector for the eager tree mask"
                    )
                    self._warned_seq_lens_copy = True
                prefix_lens = batch.seq_lens.to(device="cpu", dtype=torch.int64)
            else:
                prefix_lens = batch.seq_lens_cpu
            custom_mask = build_sglang_full_mask(tree.visibility, prefix_lens)
        return UBTreeVerifyInput(
            draft_token=tree.tokens.reshape(-1),
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=tree.retrieve_index,
            retrieve_next_token=tree.retrieve_next_token,
            retrieve_next_sibling=tree.retrieve_next_sibling,
            retrieve_parent_token=tree.parent_ids,
            tree_depths=tree.depths,
            retrieve_cum_len=None,
            spec_steps=self.ubtree_config.depth_count,
            topk=2,
            draft_token_num=tree.node_count,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
            logical_max_depth=self.ubtree_config.block_size,
            metadata_buffers=metadata_buffers,
            tree_mask_mode=tree_mask_mode,
        )

    def _ensure_draft_commit_buffers(self, batch_size: int) -> None:
        capacity = (
            0
            if self._draft_commit_positions_buf is None
            else int(self._draft_commit_positions_buf.shape[0])
        )
        if capacity >= int(batch_size):
            return
        new_capacity = max(int(batch_size), capacity * 2 if capacity else 1)
        shape = (new_capacity, int(self.ubtree_config.block_size))
        self._draft_commit_positions_buf = torch.empty(
            shape, dtype=torch.int64, device=self.device
        )
        self._draft_commit_cache_loc_buf = torch.empty(
            shape, dtype=torch.int64, device=self.device
        )

    def _ensure_target_move_buffers(self, batch_size: int, accept_width: int) -> None:
        size = int(batch_size) * int(accept_width)
        capacity = (
            0
            if self._target_move_tgt_cache_loc_buf is None
            else int(self._target_move_tgt_cache_loc_buf.numel())
        )
        if capacity >= size:
            return
        new_capacity = max(size, capacity * 2 if capacity else 1)
        self._target_move_tgt_cache_loc_buf = torch.empty(
            new_capacity, dtype=torch.int64, device=self.device
        )
        self._target_move_src_cache_loc_buf = torch.empty(
            new_capacity, dtype=torch.int64, device=self.device
        )

    def _move_accept_target_kv(
        self,
        *,
        batch,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        batch_size, accept_width = map(int, accept_index.shape)
        maybe_detect_oob(
            accept_index,
            -1,
            batch.out_cache_loc.size(0),
            "UBTree move accepted target KV cache index",
        )
        graph_runner = self._target_kv_move_graph_runner
        if graph_runner is not None and graph_runner(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            accept_index=accept_index,
            accept_lens=accept_lens,
            out_cache_loc=batch.out_cache_loc,
        ):
            return

        self._ensure_target_move_buffers(batch_size, accept_width)
        size = batch_size * accept_width
        target_cache_loc = self._target_move_tgt_cache_loc_buf[:size]
        source_cache_loc = self._target_move_src_cache_loc_buf[:size]
        prepare_target_kv_move_metadata(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            seq_lens=batch.seq_lens,
            accept_index=accept_index,
            accept_lens=accept_lens,
            out_cache_loc=batch.out_cache_loc,
            target_cache_loc_out=target_cache_loc,
            source_cache_loc_out=source_cache_loc,
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            target_cache_loc, source_cache_loc
        )

    def _commit_draft_kv(
        self,
        *,
        batch,
        hidden: torch.Tensor,
        accept_lens: torch.Tensor,
        node_count: int,
    ) -> None:
        batch_size = len(batch.seq_lens)
        hidden_stride = int(hidden.shape[0] // batch_size)
        if hidden_stride != self.ubtree_config.block_size:
            raise ValueError(
                "UBTREE draft KV commit requires compact accepted-path hidden "
                f"rows with stride {self.ubtree_config.block_size}, got {hidden_stride}"
            )
        self._ensure_draft_commit_buffers(batch_size)
        positions_2d = self._draft_commit_positions_buf[:batch_size, :hidden_stride]
        cache_loc_2d = self._draft_commit_cache_loc_buf[:batch_size, :hidden_stride]
        prepare_draft_commit_metadata(
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            positions_out=positions_2d,
            cache_loc_out=cache_loc_2d,
        )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden,
            cache_loc=cache_loc_2d.reshape(-1),
            cache_loc_2d=cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=accept_lens.to(torch.int32),
        )

    @staticmethod
    def _can_fold_verify_epilogue(batch) -> bool:
        sampling_info = batch.sampling_info
        if sampling_info is None or not sampling_info.is_all_greedy:
            return False
        if getattr(sampling_info, "acc_additive_penalties", None) is not None:
            return False
        if getattr(sampling_info, "acc_scaling_penalties", None) is not None:
            return False
        if getattr(sampling_info, "logit_bias", None) is not None:
            return False
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

        return SIMULATE_ACC_LEN <= 0

    def forward_batch_generation(
        self,
        batch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if getattr(batch, "return_logprob", False):
            raise ValueError("UBTREE does not support return_logprob yet")
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise ValueError(
                "UBTREE currently requires greedy sampling (temperature=0)"
            )
        if getattr(batch, "has_grammar", False):
            raise ValueError("UBTREE currently does not support constrained decoding")
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError("UBTREE expected DFlashDraftInputV2 batch state")
        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                bonus_tokens=empty_ids, new_seq_lens=empty_ids
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        tree, prefix_lens, _ = self._draft_tree(batch, draft_input)
        verify_input = self._make_verify_input(batch, tree)
        batch.spec_info = verify_input
        batch_size = tree.batch_size
        verify_epilogue = self._verify_epilogues.get(tree.node_count)
        if verify_epilogue is not None and not tree.preverify_staged:
            verify_epilogue.begin_step(verify_input, prefix_lens, batch_size)
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
        target_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_output.logits_output
        node_count = tree.node_count
        folded_epilogue = (
            verify_epilogue is not None
            and target_output.can_run_cuda_graph
            and self._can_fold_verify_epilogue(batch)
        )
        published = False
        if folded_epilogue:
            folded = verify_epilogue.read_result(batch_size)
            accept_index = folded.accept_index
            hidden = folded.hidden
            bonus_tokens = folded.bonus_tokens
            new_seq_lens = folded.new_seq_lens
            # The graph tail has already produced the committed lengths.
            # Publish them before output clones and Python-side bookkeeping,
            # matching DSpARK's early-publish ordering so scheduling can prepare
            # the next iteration while the current stream finishes its tail.
            if on_publish is not None:
                on_publish(new_seq_lens)
                published = True
            # The scheduler copies predict/accept_lens to host on a separate
            # stream which can overlap the next target replay, so those two
            # outputs need private storage. The remaining views are consumed on
            # the current forward stream before the next target replay:
            # accept_index/hidden by commit, and bonus/new_seq_lens by the
            # FutureMap publish/stash path (or as next-iteration inputs in the
            # synchronous path). Keep them graph-owned and avoid two redundant
            # D2D clones per decode.
            predict = folded.predict.clone()
            accept_lens = folded.accept_lens.clone()
        else:
            predict, accept_lens, accept_index = eagle_sample(
                verify_input, batch, logits_output
            )
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "UBTREE target verify returned no captured hidden states"
                )

        folded_kda_commit = folded_epilogue and verify_epilogue.folds_kda_commit
        if not folded_kda_commit:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                node_count,
            )
        folded_target_kv = folded_epilogue and verify_epilogue.folds_target_kv
        if not folded_target_kv:
            self._move_accept_target_kv(
                batch=batch,
                accept_index=accept_index,
                accept_lens=accept_lens,
            )
        if not folded_epilogue:
            predict = compact_tree_values_to_front(
                predict,
                accept_index,
                batch_size=batch_size,
                node_count=node_count,
            )
            hidden = compact_tree_values_to_front(
                hidden,
                accept_index,
                batch_size=batch_size,
                node_count=node_count,
            )
            hidden = hidden.view(batch_size, node_count, -1)[
                :, : self.ubtree_config.block_size
            ].reshape(-1, hidden.shape[-1])
        folded_draft_kv = folded_epilogue and verify_epilogue.folds_draft_kv
        if not folded_draft_kv:
            self._commit_draft_kv(
                batch=batch,
                hidden=hidden,
                accept_lens=accept_lens,
                node_count=node_count,
            )
        logits_output.hidden_states = None

        if not folded_epilogue:
            compact_predict = predict.view(batch_size, node_count)
            request_ids = torch.arange(batch_size, device=self.device)
            bonus_tokens = compact_predict[
                request_ids, accept_lens.to(torch.int64) - 1
            ].to(torch.int64)
            new_seq_lens = prefix_lens + accept_lens.to(prefix_lens.dtype)
        if on_publish is not None and not published:
            on_publish(new_seq_lens)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            accept_lens=accept_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=node_count,
            new_seq_lens=new_seq_lens,
            routed_experts_output=getattr(target_output, "routed_experts_output", None),
            indexer_topk_output=getattr(target_output, "indexer_topk_output", None),
            extra_keep_alive_refs=[verify_forward_batch],
        )


__all__ = ["UBTreeWorkerV2"]
