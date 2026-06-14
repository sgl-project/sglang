import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.ddtree_info import DDTreeVerifyInput
from sglang.srt.speculative.ddtree_utils import (
    build_ddtree_tree,
    compile_ddtree_tree,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_worker import DFlashWorker

logger = logging.getLogger(__name__)


class DDTreeWorker(DFlashWorker):
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
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
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

        # --- 3) Spine-mode fast path: delegate to DFLASH verify. ---
        if self.tree_budget <= self.block_size - 1:
            # Spine mode: linear chain, identical to DFLASH.  Skip all DDTree
            # overhead (tree building, child_maps, custom masks) and reuse
            # DFLASH's battle-tested verify path with CUDA graph support.
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput

            draft_tokens = self._greedy_sample_draft_tokens(
                hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
                lm_head=lm_head,
            ).view(bs, self.block_size - 1)

            # Build verify IDs: [bonus, draft_0, ..., draft_{budget-1}]
            draft_ids = torch.zeros(
                bs, self.max_tree_nodes, dtype=torch.long, device=batch.device
            )
            draft_ids[:, 0] = draft_input.bonus_tokens
            draft_ids[:, 1:] = draft_tokens[:, : self.tree_budget]

            positions = torch.zeros(
                bs, self.max_tree_nodes, dtype=torch.long, device=batch.device
            )
            positions[:, 0] = batch.seq_lens
            positions[:, 1:] = (
                batch.seq_lens.unsqueeze(1)
                + torch.arange(1, self.tree_budget + 1, device=batch.device)
            )

            verify_input = DFlashVerifyInput(
                draft_token=draft_ids.reshape(-1),
                positions=positions.reshape(-1),
                draft_token_num=self.max_tree_nodes,
                capture_hidden_mode=CaptureHiddenMode.FULL,
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

        (
            verify_input_ids,
            verify_position_ids,
            tree_attention_mask,
            actual_tree_sizes,
        ) = compile_ddtree_tree(
            root_token_ids=draft_input.bonus_tokens,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility=visibility,
            start_positions=batch.seq_lens,
            past_lengths=batch.seq_lens,
            tree_budget=self.tree_budget,
            actual_tree_sizes=actual_tree_sizes,
            dtype=torch.bfloat16,
            device=batch.device,
        )

        tree_is_spine = all(
            all(len(children) <= 1 for children in cm.values())
            for cm in child_maps
        )

        verify_input = DDTreeVerifyInput(
            draft_token=verify_input_ids.reshape(-1),
            positions=verify_position_ids.reshape(-1),
            draft_token_num=self.max_tree_nodes,
            tree_budget=self.tree_budget,
            child_maps=child_maps,
            actual_tree_sizes=actual_tree_sizes,
            custom_mask=tree_attention_mask,
            tree_is_spine=tree_is_spine,
        )
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self, batch: ScheduleBatch, **kwargs
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "DDTREE batch requested return_logprob, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(batch, **kwargs)
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DDTREE requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if batch.extend_lens is None or batch.prefix_lens is None:
                raise RuntimeError(
                    "DDTREE expected extend_lens / prefix_lens to be populated in extend mode, but got None."
                )

            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(batch.extend_lens)
            draft_input = DFlashDraftInput(
                bonus_tokens=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(batch.prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DDTREE decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        assert batch.forward_mode.is_target_verify()
        verify_input = batch.spec_info

        # Copy CUDA graph state from target worker BEFORE forward
        self.target_worker.capture_mode = getattr(
            self.target_worker.model_runner, "capture_mode", False
        )

        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # Spine path: delegate to DFLASH verify (CUDA-graph compatible, handles
        # KV, seq_lens, hidden states, etc. in one call).
        if isinstance(verify_input, DFlashVerifyInput):
            (
                new_bonus_tokens,
                commit_lens,
                next_target_hidden,
                num_correct_drafts_per_req_cpu,
            ) = verify_input.verify(
                batch=batch,
                logits_output=logits_output,
                page_size=self.page_size,
            )

            draft_input.bonus_tokens = new_bonus_tokens
            draft_input.target_hidden = next_target_hidden
            draft_input.ctx_lens = commit_lens
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            batch.forward_mode = ForwardMode.DECODE

            num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
            if not self._logged_first_verify and self.tp_rank == 0:
                logger.info(
                    "DDTREE spine verify completed. num_correct_drafts_per_req=%s",
                    num_correct_drafts_per_req_cpu,
                )
                self._logged_first_verify = True

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=new_bonus_tokens,
                num_correct_drafts=num_correct_drafts,
                num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        # Full tree path: use DDTree verify.
        assert isinstance(verify_input, DDTreeVerifyInput)
        (
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            model_runner=self.target_worker.model_runner,
        )

        draft_input.bonus_tokens = new_bonus_tokens
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DDTREE verify completed. num_correct_drafts_per_req=%s",
                num_correct_drafts_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_bonus_tokens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
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
            tp_size, hidden_states.shape[0],
            dtype=local_max.dtype, device=local_max.device,
        )
        tp_group.all_gather_into_tensor(gathered_max, local_max)
        global_max_idx = gathered_max.argmax(dim=0)

        gathered_argmax = torch.empty(
            tp_size, hidden_states.shape[0],
            dtype=local_argmax.dtype, device=local_argmax.device,
        )
        tp_group.all_gather_into_tensor(gathered_argmax, local_argmax)
        global_argmax = gathered_argmax[
            global_max_idx, torch.arange(hidden_states.shape[0], device=hidden_states.device)
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

        # TP > 1: compute global log-probs (the exp→log path is needed
        # here because vocab shard sizes may differ across ranks).
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)

        local_max = local_logits.amax(dim=-1, keepdim=True)
        gathered_max = torch.empty(
            tp_size, hidden_states.shape[0], dtype=local_max.dtype, device=local_max.device
        )
        tp_group.all_gather_into_tensor(gathered_max, local_max.squeeze(-1))
        global_max = gathered_max.amax(dim=0, keepdim=True)

        local_exp = torch.exp(local_logits - global_max.unsqueeze(-1))

        gathered_exp = torch.empty(
            tp_size, hidden_states.shape[0], local_logits.shape[-1],
            dtype=local_exp.dtype, device=local_exp.device,
        )
        tp_group.all_gather_into_tensor(gathered_exp, local_exp)
        global_sum = gathered_exp.sum(dim=(0, 2), keepdim=True)

        global_logits = torch.cat(
            [gathered_exp[i] for i in range(tp_size)], dim=-1
        )
        global_log_probs = torch.log(global_logits / global_sum.unsqueeze(-1))

        return global_log_probs.float()
