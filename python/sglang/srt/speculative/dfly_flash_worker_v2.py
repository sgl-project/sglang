from typing import Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.runtime_context import get_exec, get_schedule, get_spec
from sglang.srt.sampling.verify_accept import VerifyAccept
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    commit_accept,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dfly_utils import DFlyRequestState
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.target_verify import TargetVerify, adjust_verify_logits


class DFlyFlashWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = None
        self.model_runner = target_worker.model_runner
        self.device = self.model_runner.device
        self.dfly = self.model_runner.model.dfly
        if self.dfly is None:
            raise ValueError("DFLY requires a target model with a bundled DFly draft")
        self.dfly_config = self.dfly.dfly_config
        self.block_size = int(get_spec().speculative_num_draft_tokens)
        if self.block_size != self.dfly_config.block_size:
            raise ValueError("DFly draft and verify block sizes disagree")
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._block_offsets = torch.arange(
            self.block_size, dtype=torch.int64, device=self.device
        )
        self._max_rows = self._max_verify_rows()
        self._topk_capture = self._bind_topk_capture()
        self.state: Optional[DFlyRequestState] = None

    def _max_verify_rows(self) -> int:
        graph = get_exec().graph.cuda_graph_config
        graph_bs = graph.decode.max_bs if graph is not None else None
        max_bs = max(graph_bs or 0, get_schedule().max_running_requests or 0, 1)
        return max_bs * self.block_size

    def _bind_topk_capture(self) -> torch.Tensor:
        config = self.dfly_config
        capture = torch.full(
            (config.num_layers, self._max_rows, config.target_index_topk),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        model = self.model_runner.model.model
        for i, layer_id in enumerate(config.target_index_layers):
            indexer = model.layers[layer_id].self_attn.indexer
            indexer.dfly_topk_capture = capture[i]
        hasher = model.engram_hasher
        if hasher is not None:
            hasher.verify_recent = torch.zeros(
                (self._max_rows, hasher.max_ngram_size - 1),
                dtype=torch.int64,
                device=self.device,
            )
        return capture

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        config = self.dfly_config
        self.state = DFlyRequestState(
            pool_size=req_to_token_pool.req_to_token.shape[0],
            hidden_size=self.model_runner.model_config.hidden_size,
            num_layers=config.num_layers,
            target_topk=config.target_index_topk,
            dtype=self.model_runner.dtype,
            device=self.device,
        )
        self.dfly.bind_buffers(
            state=self.state,
            draft_rows=torch.full(
                (config.num_layers, self._max_rows, config.draft_index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            draft_counts=torch.zeros(
                self._max_rows, dtype=torch.int32, device=self.device
            ),
            candidates_out=torch.zeros(
                self._max_rows, dtype=torch.int64, device=self.device
            ),
        )

    def init_attention_backends(self):
        if not isinstance(self.model_runner.attn_backend, DeepseekSparseAttnBackend):
            raise ValueError("DFLY requires the DSA attention backend")

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_prefill(batch, on_publish)
        if batch.forward_mode.is_idle():
            return self._forward_idle(on_publish)
        return self._forward_decode(batch, on_publish)

    def _forward_prefill(self, batch: ScheduleBatch, on_publish):
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.LAST
        )
        result.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        if batch.forward_mode.is_idle():
            result.next_draft_input = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )
            return result
        logits_output = result.logits_output
        bs = batch.batch_size()
        self.state.write(
            req_pool_indices=batch.req_pool_indices,
            hidden=logits_output.hidden_states[:bs],
            topk=self._topk_capture[:, :bs],
        )
        logits_output.hidden_states = None
        result.next_draft_input = make_draft_input_v2(
            bonus_tokens=result.next_token_ids, new_seq_lens=batch.seq_lens
        )
        return result

    def _forward_idle(self, on_publish):
        next_draft_input = make_draft_input_v2(
            bonus_tokens=torch.empty((0,), dtype=torch.int64, device=self.device),
            new_seq_lens=torch.empty((0,), dtype=torch.int64, device=self.device),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(self, batch: ScheduleBatch, on_publish):
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError("DFLY expects DFlashDraftInputV2 on the running batch")
        batch.seq_lens.record_stream(torch.cuda.current_stream())
        bs = len(batch.seq_lens)
        width = self.block_size
        prefix_lens = batch.seq_lens
        verify_forward_batch = self._prepare_verify(batch, draft_input)
        target_out = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=verify_forward_batch, is_verify=True
        )
        logits_output = target_out.logits_output
        candidates = self.dfly.candidates_out[: bs * width].view(bs, width)
        # Flash DFly admits neither grammar nor watermark requests.
        verify = TargetVerify.prepare(
            batch,
            logits_output=logits_output,
            grammar_tree=None,
            grammar_barrier=None,
            watermark_engine=None,
            watermark_full_mask=False,
        )
        num_correct_drafts, bonus = self._accept(
            batch, draft_input, candidates, logits_output.next_token_logits, verify
        )
        out_tokens, commit_lens = commit_accept(candidates, num_correct_drafts, bonus)
        verify.commit(
            batch,
            logits_output=logits_output,
            accept=VerifyAccept(tokens=out_tokens, accept_lens=commit_lens),
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)
        source_rows = torch.arange(bs, device=self.device) * width + num_correct_drafts
        self._commit_state(batch, logits_output.hidden_states, source_rows)
        logits_output.hidden_states = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=make_draft_input_v2(
                bonus_tokens=bonus, new_seq_lens=new_seq_lens
            ),
            speculative_num_draft_tokens=width,
            new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
            kv_weight_version_record=target_out.kv_weight_version_record,
        )

    def _prepare_verify(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInputV2
    ) -> ForwardBatch:
        bs = len(batch.seq_lens)
        width = self.block_size
        block_ids = torch.full(
            (bs, width),
            self.dfly_config.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        block_ids[:, 0] = draft_input.bonus_tokens
        positions = batch.seq_lens.to(torch.int64)[:, None] + self._block_offsets
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + width,
            batch_size=bs,
            draft_token_num=width,
            device=self.device,
        )
        verify_input = DFlashVerifyInput(
            draft_token=block_ids.view(-1),
            positions=positions.view(-1),
            draft_token_num=width,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        seq_lens_cpu, seq_lens_sum = batch.seq_lens_cpu, batch.seq_lens_sum
        if seq_lens_cpu is not None:
            batch.seq_lens_cpu = seq_lens_cpu + width
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker, defer_metadata_init=True
        )
        batch.seq_lens_cpu, batch.seq_lens_sum = seq_lens_cpu, seq_lens_sum
        return verify_forward_batch

    def _accept(self, batch, draft_input, candidates, next_token_logits, verify):
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            adjust_verify_logits(
                next_token_logits,
                sampling_info=sampling_info,
                grammar_mask=verify.grammar_mask,
                draft_tokens=candidates,
                has_custom_logit_processor=sampling_info.has_custom_logit_processor,
            )
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            num_correct_drafts, bonus = (
                compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=next_token_logits,
                    sampling_info=sampling_info,
                    max_top_k=draft_input.max_top_k,
                    uniform_top_k_value=draft_input.uniform_top_k_value,
                )
            )
            self._tp_sync.sync(SpecTpSyncSite.DFLY_ACCEPT_SAMPLE, num_correct_drafts)
            self._tp_sync.sync(SpecTpSyncSite.DFLY_ACCEPT_SAMPLE, bonus)
            return num_correct_drafts, bonus
        target_predict = torch.argmax(next_token_logits, dim=-1).view(candidates.shape)
        self._tp_sync.sync(SpecTpSyncSite.DFLY_ACCEPT_GREEDY, target_predict)
        return compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target_predict
        )

    def _commit_state(self, batch, hidden_states, source_rows) -> None:
        self.state.write(
            req_pool_indices=batch.req_pool_indices,
            hidden=hidden_states[source_rows],
            topk=self._topk_capture[:, source_rows],
        )
        hasher = self.model_runner.model.model.engram_hasher
        if hasher is not None:
            hasher.commit_verify(batch.req_pool_indices, source_rows)
