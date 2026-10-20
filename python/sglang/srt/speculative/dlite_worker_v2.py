from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_input_v2,
)
from sglang.srt.speculative.frozen_kv_mtp_utils import select_last_extend_hidden
from sglang.srt.speculative.spec_sampling_mask import (
    SpeculativeSamplingMaskCapture,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_pp_context,
    draft_tp_context,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _commit_accept(
    candidates: torch.Tensor,
    accept_len: torch.Tensor,
    bonus_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    out_tokens[:, :-1].copy_(candidates[:, 1:])
    out_tokens[:, -1].fill_(0)
    out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus_tokens[:, None])
    return out_tokens, accept_len.to(torch.int32) + 1


class DLiteWorkerV2(BaseSpecWorker):
    """History-free DLite worker using SGLang's linear target-verify ABI."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        super().__init__()
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.page_size = get_schedule().page_size

        # DLite linears use the same TP group as the target. The model has no
        # paged attention and therefore owns no draft KV pool/backend.
        self.draft_owns_attention = False
        with draft_pp_context(), draft_tp_context(self.draft_owns_attention):
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DLITE",
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model

        self.block_size = int(self.draft_model.block_size)
        configured_width = int(get_spec().speculative_num_draft_tokens)
        if configured_width != self.block_size:
            raise ValueError(
                "DLite verify width must equal the checkpoint block_size, got "
                f"speculative_num_draft_tokens={configured_width}, "
                f"block_size={self.block_size}."
            )
        self.speculative_num_draft_tokens = self.block_size
        self._mask_token_id = int(self.draft_model.mask_token_id)
        self._block_pos_offsets = build_block_pos_offsets(
            length=self.block_size, device=self.device
        )

        self._draft_block_ids_buf: torch.Tensor | None = None
        self._draft_block_positions_buf: torch.Tensor | None = None
        self._draft_block_tokens_buf: torch.Tensor | None = None
        self._verify_out_cache_loc_buf: torch.Tensor | None = None
        self._buffer_capacity = 0
        self._target_logits_processor = self._defer_target_verify_gather()

        if get_parallel().tp_rank == 0:
            logger.info(
                "Initialized history-free DLite draft runner. model=%s, "
                "block_size=%d, target_layer_ids=%s.",
                self.draft_model.__class__.__name__,
                self.block_size,
                list(self.draft_model.target_layer_ids),
            )

    @property
    def draft_worker(self) -> TpModelWorker:
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self.target_worker.model_runner.attn_backend,)

    def _draft_model_runners(self) -> tuple:
        # DLite has weights but no paged draft KV to register with HiCache.
        return ()

    def weight_update_runners(self) -> list[tuple[str, ModelRunner]]:
        return [("draft", self.draft_model_runner)]

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        # The draft uses dense SDPA over one tiny in-memory block.
        return None

    def init_cuda_graphs(self) -> None:
        # Target verification still uses the target runner's CUDA graphs.
        if get_parallel().tp_rank == 0:
            logger.info(
                "DLite draft execution is eager; target verify graphs remain enabled."
            )

    def clear_cache_pool(self) -> None:
        return None

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _defer_target_verify_gather(self) -> LogitsProcessor | None:
        processors = [
            module
            for module in self.target_worker.model_runner.model.modules()
            if isinstance(module, LogitsProcessor)
        ]
        if len(processors) != 1:
            return None
        processors[0].defer_verify_gather = True
        return processors[0]

    @staticmethod
    def _can_skip_verify_logits(batch: ScheduleBatch, sampling_info) -> bool:
        if batch.has_grammar or batch.return_logprob or batch.return_sampling_mask:
            return False
        if sampling_info is None:
            return True
        penalizer = sampling_info.penalizer_orchestrator
        return (
            sampling_info.is_all_greedy
            and not sampling_info.has_custom_logit_processor
            and getattr(sampling_info, "acc_linear_penalties", None) is None
            and (penalizer is None or not penalizer.is_required)
            and getattr(sampling_info, "grammar_mask", None) is None
            and getattr(sampling_info, "logit_bias", None) is None
            and sampling_info.sampling_mask_batch_indices is None
        )

    def _ensure_buffers(self, batch_size: int) -> None:
        if self._buffer_capacity >= batch_size:
            return
        capacity = max(batch_size, self._buffer_capacity * 2)
        self._buffer_capacity = capacity
        shape = (capacity, self.block_size)
        self._draft_block_ids_buf = torch.empty(
            shape, dtype=torch.long, device=self.device
        )
        self._draft_block_positions_buf = torch.empty(
            shape, dtype=torch.int64, device=self.device
        )
        self._draft_block_tokens_buf = torch.empty(
            shape, dtype=torch.long, device=self.device
        )
        self._verify_out_cache_loc_buf = torch.empty(
            shape, dtype=torch.int64, device=self.device
        )

    @staticmethod
    def _state(
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> DFlashDraftInputV2:
        state = make_draft_input_v2(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
        )
        state.hidden_states = hidden_states
        return state

    def _prepare_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch.batch_size()
        self._ensure_buffers(batch_size)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._verify_out_cache_loc_buf is not None

        block_ids = self._draft_block_ids_buf[:batch_size]
        positions = self._draft_block_positions_buf[:batch_size]
        cache_locs = self._verify_out_cache_loc_buf[:batch_size]
        prefix_lens = batch.seq_lens

        block_ids.fill_(self._mask_token_id)
        block_ids[:, 0].copy_(draft_input.bonus_tokens)
        torch.add(
            prefix_lens.unsqueeze(1),
            self._block_pos_offsets,
            out=positions,
        )
        end_offsets = prefix_lens + self.block_size
        assigned = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offsets,
            batch_size=batch_size,
            draft_token_num=self.block_size,
            device=self.device,
        )
        cache_locs.copy_(assigned.view(batch_size, self.block_size))
        return block_ids, positions, cache_locs

    def _draft(
        self,
        *,
        block_ids: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        target_model = self.target_worker.model_runner.model
        embed_tokens = unwrap_lora_layer(target_model.get_input_embeddings())
        query_embeddings = embed_tokens(block_ids)
        with torch.inference_mode(), draft_tp_context(self.draft_owns_attention):
            draft_hidden = self.draft_model.forward_draft(
                query_embeddings=query_embeddings,
                target_hidden=target_hidden,
            )
            return self.draft_model.sample_greedy(
                draft_hidden=draft_hidden,
                anchor_token_ids=block_ids[:, 0],
            )

    def _validate_request_sampling(self, batch: ScheduleBatch) -> None:
        sampling_info = batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise ValueError(
                "DLite currently supports greedy decoding only. Use temperature=0 "
                "and do not enable stochastic sampling."
            )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        self._validate_request_sampling(batch)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
            logits_output = batch_output.logits_output
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DLite prefill requires target auxiliary hidden states."
                )
            pivot_hidden = select_last_extend_hidden(batch, logits_output.hidden_states)
            logits_output.hidden_states = None
            batch_output.new_seq_lens = batch.seq_lens
            batch_output.next_draft_input = self._state(
                bonus_tokens=batch_output.next_token_ids,
                new_seq_lens=batch.seq_lens,
                hidden_states=pivot_hidden,
            )
            if on_publish is not None:
                on_publish(batch.seq_lens)
            return batch_output

        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DLite expected DFlashDraftInputV2-compatible relay state."
            )

        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            state = self._state(
                bonus_tokens=empty_ids,
                new_seq_lens=empty_ids,
                hidden_states=torch.empty(
                    (0, 0), dtype=self.model_runner.dtype, device=self.device
                ),
            )
            if on_publish is not None:
                on_publish(state.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=state,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=self.block_size,
                new_seq_lens=state.new_seq_lens,
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        batch_size = batch.batch_size()
        prefix_lens = batch.seq_lens
        block_ids, positions_2d, cache_locs_2d = self._prepare_draft_block(
            batch=batch,
            draft_input=draft_input,
        )
        if draft_input.hidden_states is None or draft_input.hidden_states.numel() == 0:
            raise RuntimeError("DLite decode is missing relayed target hidden states.")

        draft_next = self._draft(
            block_ids=block_ids,
            target_hidden=draft_input.hidden_states,
        )
        assert self._draft_block_tokens_buf is not None
        draft_tokens = self._draft_block_tokens_buf[:batch_size]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        grammar_tree = (
            GrammarTree.from_linear_chain(draft_tokens) if batch.has_grammar else None
        )
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = cache_locs_2d.reshape(-1)
        kv_loc_plan = self.target_worker.model_runner.kv_index_translator.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
            write_virtual=verify_out_cache_loc,
            read_extent=self.block_size,
        )
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_loc_plan=kv_loc_plan,
        )
        batch.out_cache_loc = verify_out_cache_loc

        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_lens = seq_lens_cpu_backup + self.block_size
            batch.seq_lens_cpu = verify_host_lens
            batch.seq_lens_sum = int(verify_host_lens.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        sampling_info = batch.sampling_info
        target_predict = None
        target_lp = self._target_logits_processor
        if target_lp is not None and target_lp.is_deferred_shard(
            logits_output.next_token_logits
        ):
            if self._can_skip_verify_logits(batch, sampling_info):
                target_predict = target_lp.deferred_argmax(
                    logits_output.next_token_logits
                )
                logits_output.next_token_logits = None
            else:
                logits_output.next_token_logits = target_lp.finish_deferred_gather(
                    logits_output.next_token_logits
                )

        grammar_mask = None
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
        if sampling_info is not None and target_predict is None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=self.block_size,
            )
        if grammar_mask is not None:
            grammar_mask.apply(logits_output.next_token_logits)
        if target_predict is None:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.view(batch_size, self.block_size)

        accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=draft_tokens,
            target_predict=target_predict,
        )
        out_tokens, commit_lens = _commit_accept(draft_tokens, accept_len, bonus)
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
            sampling_info,
            next_token_logits=logits_output.next_token_logits,
            draft_input=draft_input,
            draft_token_num=self.block_size,
            bs=batch_size,
        )
        if sampling_mask_capture is not None:
            logits_output.sampling_mask_output = sampling_mask_capture.build_output(
                out_tokens=out_tokens,
                commit_lens=commit_lens,
            )
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=self.block_size,
            )

        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("DLite target verify requires auxiliary hidden states.")
        target_hidden = target_hidden[: batch_size * self.block_size].view(
            batch_size, self.block_size, -1
        )
        pivot_indices = commit_lens.to(torch.long) - 1
        next_hidden = target_hidden[
            torch.arange(batch_size, device=self.device), pivot_indices
        ]
        logits_output.hidden_states = None
        next_draft_input = self._state(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            hidden_states=next_hidden,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
        )


__all__ = ["DLiteWorkerV2"]
