# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See ubtree_components/LICENSE for the original license.

"""DFlash proposals with fixed-width irregular-tree verification and V2 relay."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.distributed import get_tp_group
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
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config
from sglang.srt.speculative.ubtree_components.draft_graph import (
    UBTreeDraftSegmentGraphRunner,
)
from sglang.srt.speculative.ubtree_components.epilogue import UBTreeVerifyEpilogue
from sglang.srt.speculative.ubtree_components.tree import (
    UBTreeBatch,
)
from sglang.srt.speculative.ubtree_components.tree_graph import (
    build_ubtree_proposal,
)
from sglang.srt.speculative.ubtree_info import UBTreeVerifyInput

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
        target_model = self.target_worker.model_runner.model
        if selector.tied_token_factorizer is not None:
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
        self._proposal = partial(
            build_ubtree_proposal,
            config=self.ubtree_config,
            num_nodes=resolved_verify_node_count,
            lm_head=target_model.lm_head,
            selector=selector,
            tp_group=get_tp_group(),
        )
        self._verify_epilogue = UBTreeVerifyEpilogue(
            max_bs=1,
            node_count=resolved_verify_node_count,
            max_depth=self.ubtree_config.block_size,
            device=self.device,
        )
        self._verify_positions_buf = torch.empty(
            (1, resolved_verify_node_count), dtype=torch.int64, device=self.device
        )
        self._draft_segment_graph_runner = None
        if not get_exec().graph.disable_cuda_graph:
            self.model_runner.capture_tail_hooks.append(self._capture_verify_epilogue)

    def _defer_target_verify_gather(self):
        # The graph-folded argmax consumes global vocabulary IDs. Keep
        # DFlash's deferred local-logits optimization out of UBTree.
        return None

    def _capture_verify_epilogue(
        self, runner, out, forward_batch, num_tokens: int
    ) -> None:
        if (
            num_tokens
            == int(forward_batch.batch_size) * self.speculative_num_draft_tokens
        ):
            self._verify_epilogue.capture_hook(runner, out, forward_batch, num_tokens)

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
        hidden = draft_output.hidden_states
        if hidden is None:
            raise RuntimeError("UBTREE draft graph tail received no hidden states")
        tree = self._proposal(
            hidden=hidden.view(batch_size, self.block_size, -1)[:, 1:],
            bonus_tokens=input_ids.view(batch_size, self.block_size)[:, 0],
        )
        positions = self._verify_positions_buf[:batch_size]
        torch.add(forward_batch.seq_lens[:, None], tree.depths, out=positions)
        tree.verify_positions = positions.reshape(-1)
        self._verify_epilogue.begin_tree_step(tree, forward_batch.seq_lens)
        tree.preverify_staged = True
        return tree

    def init_cuda_graphs(self) -> None:
        """Capture the full draft, including proposal and pre-verify staging."""
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        exec_cfg = get_exec()
        if exec_cfg.graph.disable_cuda_graph:
            return
        decode_bs = exec_cfg.graph.cuda_graph_config.decode.bs
        backend = self.draft_model_runner.attn_backend
        backend.init_cuda_graph_state(max(decode_bs), max(decode_bs) * self.block_size)
        self._draft_segment_graph_runner = UBTreeDraftSegmentGraphRunner(
            model=self.draft_model,
            input_embedding=self.target_worker.model_runner.model.get_input_embeddings(),
            attn_backend=backend,
            batch_sizes=decode_bs,
            block_size=self.block_size,
            hidden_size=int(self.draft_model_runner.model_config.hidden_size),
            device=self.device,
        )
        self._draft_segment_graph_runner.attach_capture_tail(
            self._capture_proposal_tail
        )
        self.draft_model.attach_segment_graph_runner(self._draft_segment_graph_runner)
        logger.info(
            "UBTREE capturing full draft CUDA Graph with proposal and TP collectives"
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
            if not can_fold_draft_kv or not getattr(
                mamba_pool, "factorized_kda_tree", False
            ):
                raise ValueError(
                    "UBTREE requires fused draft KV materialization and factorized KDA state"
                )
            epilogue = self._verify_epilogue
            epilogue.attach_target_kv_commit(
                target_kvcache=target_kvcache,
                req_to_token=req_to_token_pool.req_to_token,
            )
            epilogue.attach_draft_kv_commit(
                draft_commit=self._append_target_hidden_to_draft_kv_by_loc,
                req_to_token=req_to_token_pool.req_to_token,
            )
            epilogue.attach_kda_tree_commit(
                spec_state=req_to_token_pool.get_speculative_mamba2_params_all_layers(),
                mamba_index_mapping=req_to_token_pool.req_index_to_mamba_index_mapping,
                mamba_track_interval=mamba_track_grid(
                    token_to_kv_pool_allocator.page_size
                ),
            )
            if self._target_tp_rank == 0:
                logger.info(
                    "UBTREE draft hidden/KV commit is folded into the target verify CUDA Graph tail"
                )
                logger.info(
                    "UBTREE factorized KDA accepted-path commit is folded into the target verify CUDA Graph tail"
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
        tree = (
            folded_tree
            if folded_tree is not None
            else self._proposal(hidden=hidden, bonus_tokens=draft_input.bonus_tokens)
        )
        return tree, prefix_lens, positions

    def _make_verify_input(self, batch, tree: UBTreeBatch) -> UBTreeVerifyInput:
        positions = tree.verify_positions
        if positions is None:
            positions = self._verify_positions_buf[: tree.batch_size]
            torch.add(batch.seq_lens[:, None], tree.depths, out=positions)
            positions = positions.reshape(-1)
        return UBTreeVerifyInput(
            draft_token=tree.tokens.reshape(-1),
            custom_mask=tree.visibility.reshape(-1),
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
            tree_mask_mode=TreeMaskMode.QLEN_ONLY,
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
        if getattr(batch, "return_logprob", False):
            raise ValueError("UBTREE does not support return_logprob yet")
        if batch.sampling_info is not None and not self._can_fold_verify_epilogue(
            batch
        ):
            raise ValueError(
                "UBTREE requires greedy sampling without penalties, logit bias, or simulated acceptance"
            )
        if getattr(batch, "has_grammar", False):
            raise ValueError("UBTREE currently does not support constrained decoding")
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
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

        if not self._can_fold_verify_epilogue(batch):
            raise ValueError(
                "UBTREE requires greedy sampling without penalties, logit bias, or simulated acceptance"
            )
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        tree, prefix_lens, _ = self._draft_tree(batch, draft_input)
        verify_input = self._make_verify_input(batch, tree)
        batch.spec_info = verify_input
        batch_size = tree.batch_size
        verify_epilogue = self._verify_epilogue
        if not tree.preverify_staged:
            verify_epilogue.begin_tree_step(tree, prefix_lens)
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
        if not target_output.can_run_cuda_graph:
            verify_epilogue(
                logits=logits_output.next_token_logits,
                hidden=logits_output.hidden_states,
                bs=batch_size,
                req_pool_indices=verify_forward_batch.req_pool_indices,
                seq_lens=verify_forward_batch.seq_lens,
                out_cache_loc=verify_forward_batch.out_cache_loc,
                mamba_track_indices=verify_forward_batch.mamba_track_indices,
            )
        folded = verify_epilogue.read_result(batch_size)
        bonus_tokens = folded.bonus_tokens
        new_seq_lens = folded.new_seq_lens
        # Publish before the scheduler's private output copies, preserving the
        # overlap lifetime of the graph-owned relay buffers.
        if on_publish is not None:
            on_publish(new_seq_lens)
        predict = folded.predict.clone()
        accept_lens = folded.accept_lens.clone()
        logits_output.hidden_states = None
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
