import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import GenerationBatchResult, ScheduleBatch
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.ddtree_info import DDTreeVerifyInput
from sglang.srt.speculative.ddtree_utils import (
    build_ddtree_tree,
    compile_ddtree_tree,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

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

        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
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

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

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
                spec_algorithm=SpeculativeAlgorithm.DDTREE,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DDTREE draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)

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
        ) = build_ddtree_tree(
            draft_logits=draft_logits,
            tree_budget=self.tree_budget,
            device=batch.device,
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
            dtype=torch.bfloat16,
            device=batch.device,
        )

        verify_input = DDTreeVerifyInput(
            draft_token=verify_input_ids.reshape(-1),
            positions=verify_position_ids.reshape(-1),
            draft_token_num=self.max_tree_nodes,
            tree_budget=self.tree_budget,
            child_maps=child_maps,
            actual_tree_sizes=actual_tree_sizes,
            custom_mask=tree_attention_mask,
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
        assert isinstance(verify_input, DDTreeVerifyInput)

        batch_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

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

    def _compute_draft_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
    ) -> torch.Tensor:
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        if hidden_states.dtype != weight_dtype:
            hidden_states = hidden_states.to(weight_dtype)

        local_logits = torch.mm(hidden_states, weight.t())

        if tp_size == 1:
            return local_logits.float()

        local_max = local_logits.amax(dim=-1, keepdim=True)
        local_shifted = local_logits - local_max

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
