"""PARD-2 parallel-draft speculative decoding (https://arxiv.org/abs/2605.08632).

One decode step runs four forwards on one stream:

1. draft pass: the draft over ``[bonus, pard_token x (W - 1)]`` at positions
   ``s .. s + W - 1``, every row reading the projected target feature of
   position ``s - 1``; the argmax of rows ``0 .. W - 2`` are the K = W - 1 drafts;
2. target verify of ``[bonus, d1 .. dK]`` with the DFLASH chain accept;
3. target-dependent mode: the projected target features of the verify rows go
   to the per-slot buffer ``P_buf``;
4. context pass: the draft over the verify rows at their own positions, each
   row reading the feature of the position before it, which rewrites the draft
   KV of the window with the real tokens.

A prefill runs the target, writes ``P_buf`` for the chunk, then one draft
EXTEND over the chunk. ``P_buf`` and the draft KV share the target's slot ids,
so they live and die with the target KV.
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.configs.draft_parallel import DraftParallelSpec, DraftTPExecution
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.draft_parallel import resolve_draft_parallel_layout
from sglang.srt.speculative.draft_worker_common import (
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_input_v2,
)
from sglang.srt.speculative.pard2_utils import (
    draft_rows_to_tokens,
    fill_draft_block,
    gather_shifted_features,
    repeat_last_feature,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _commit_accept(
    candidates: torch.Tensor, accept_len: torch.Tensor, bonus: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The committed block (accepted drafts, then the bonus) and its lengths."""
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    out_tokens[:, :-1].copy_(candidates[:, 1:])
    out_tokens[:, -1].fill_(0)
    out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
    return out_tokens, accept_len.to(torch.int32) + 1


class Pard2WorkerV2(BaseSpecWorker):
    """PARD-2 speculative decoding worker (spec-v2).

    Drives both overlap and non-overlap scheduling: the scheduler runs it
    synchronously when overlap is disabled.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.page_size = get_schedule().page_size
        self.draft_parallel_layout = resolve_draft_parallel_layout(
            DraftParallelSpec(tp_execution=DraftTPExecution.TARGET_LAYOUT)
        )
        draft_backend = (
            get_spec().speculative_draft_attention_backend or attention_backends()[0]
        )
        with draft_tp_context(self.draft_parallel_layout):
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="PARD2",
                attention_backend_override=draft_backend,
                random_seed=target_worker.random_seed,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model

        draft_hf_config = self.draft_model_runner.model_config.hf_config
        self.block_size = int(get_spec().speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = self.block_size
        self.pard_token = int(draft_hf_config.pard_token)
        self.target_dependent = get_spec().speculative_pard2_mode == "target-dependent"
        if self.target_dependent != self.draft_model.target_dependent:
            raise RuntimeError(
                "PARD2 draft model mode does not match --speculative-pard2-mode "
                f"{get_spec().speculative_pard2_mode}."
            )
        self.p_buf: Optional[torch.Tensor] = None
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self.block_size, device=self.device
        )
        self._warned_sampling_fallback = False
        if get_parallel().tp_rank == 0:
            logger.info(
                "PARD2 draft worker: mode=%s, target_proj loaded=%s, "
                "draft dtype=%s, draft kv dtype=%s, attention_backend=%s, "
                "verify width=%d (K=%d), pard_token=%d",
                get_spec().speculative_pard2_mode,
                self.draft_model.target_proj is not None,
                self.draft_model_runner.dtype,
                self.draft_model_runner.kv_cache_dtype,
                bundle.resolved_attention_backend,
                self.block_size,
                self.block_size - 1,
                self.pard_token,
            )

    def weight_update_runners(self) -> List[Tuple[str, "ModelRunner"]]:
        return [("draft", self.draft_model_runner)]

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    @property
    def last_shared_read_runner(self):
        # The draft extend (prefill) and the context pass (decode) run last.
        return self.draft_model_runner

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # The draft pool is private but indexed by the target's slot ids.
        with draft_tp_context(self.draft_parallel_layout):
            self._draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        if self.target_dependent:
            num_slots = (
                int(self.draft_model_runner.token_to_kv_pool.size) + self.page_size
            )
            self.p_buf = torch.zeros(
                (num_slots, int(self.draft_model.config.hidden_size)),
                dtype=self.draft_model_runner.dtype,
                device=self.device,
            )
            if get_parallel().tp_rank == 0:
                logger.info(
                    "PARD2 feature buffer: %d slots x %d, %.2f GB",
                    num_slots,
                    self.p_buf.shape[1],
                    self.p_buf.numel() * self.p_buf.element_size() / (1 << 30),
                )

    def init_attention_backends(self):
        with draft_tp_context(self.draft_parallel_layout):
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        capture = get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
        with draft_tp_context(self.draft_parallel_layout):
            self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=capture)

    # ------------------------------------------------------------------ forward

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        self._warn_sampling_fallback(batch)
        if batch.forward_mode.is_extend():
            return self._forward_extend(batch, on_publish=on_publish)
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        if not isinstance(batch.spec_info, DFlashDraftInputV2):
            raise RuntimeError(
                "PARD2 expected DFlashDraftInputV2 state on the running batch."
            )
        if batch.forward_mode.is_idle():
            return self._forward_idle(on_publish=on_publish)
        return self._forward_decode(batch, on_publish=on_publish)

    def _warn_sampling_fallback(self, batch: ScheduleBatch) -> None:
        sampling_info = batch.sampling_info
        if (
            sampling_info is None
            or sampling_info.is_all_greedy
            or is_dflash_sampling_verify_available()
            or self._warned_sampling_fallback
        ):
            return
        self._warned_sampling_fallback = True
        logger.warning(
            "PARD2 sampled verification is unavailable on this build/device; "
            "falling back to greedy argmax verification."
        )

    def _forward_idle(self, *, on_publish) -> GenerationBatchResult:
        next_draft_input = make_draft_input_v2(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_extend(self, batch: ScheduleBatch, *, on_publish):
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_iteration_ids=True,
        )
        iteration_ids, batch_output.iteration_ids = batch_output.iteration_ids, None
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        logits_output = batch_output.logits_output
        if self.target_dependent:
            self._write_features(
                logits_output=logits_output,
                slots=iteration_ids.virtual_write_ids(),
            )
        self._draft_extend(batch, iteration_ids=iteration_ids)
        # Keep large hidden-state buffers off the host copy under overlap.
        logits_output.hidden_states = None
        logits_output.last_hidden_states = None
        batch_output.next_draft_input = make_draft_input_v2(
            bonus_tokens=batch_output.next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _write_features(
        self, *, logits_output: LogitsProcessorOutput, slots: torch.Tensor
    ) -> None:
        aux = logits_output.hidden_states
        final = logits_output.last_hidden_states
        if self.draft_model.num_target_aux_layers and aux is None:
            raise RuntimeError(
                "PARD2 target-dependent mode needs the target's aux hidden states."
            )
        if self.draft_model.uses_target_final and final is None:
            raise RuntimeError(
                "PARD2 target-dependent mode needs the target's post-norm final "
                "hidden states."
            )
        num_rows = slots.shape[0]
        projected = self.draft_model.project(
            None if aux is None else aux[:num_rows],
            None if final is None else final[:num_rows],
        )
        self.p_buf[slots] = projected.to(self.p_buf.dtype)

    def _draft_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embeds = self.draft_model.model.embed_tokens(input_ids.reshape(-1))
        if features is not None:
            embeds = embeds + features.reshape(embeds.shape).to(embeds.dtype)
        return embeds

    def _draft_extend(self, batch: ScheduleBatch, *, iteration_ids) -> None:
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_model_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
            iteration_ids=iteration_ids,
        )
        forward_batch.return_logprob = False
        positions = forward_batch.positions
        features = None
        if self.target_dependent:
            req_rows = torch.repeat_interleave(
                forward_batch.req_pool_indices,
                forward_batch.extend_seq_lens,
                output_size=positions.shape[0],
            )
            _, features = gather_shifted_features(
                self.p_buf,
                self.req_to_token_pool.req_to_token,
                req_rows,
                positions,
                forward_batch.input_ids,
            )
        forward_batch.input_embeds = self._draft_embeds(
            input_ids=forward_batch.input_ids, features=features
        )
        with torch.inference_mode(), draft_tp_context(self.draft_parallel_layout):
            self.draft_model_runner.forward(forward_batch)

    def _forward_decode(self, batch: ScheduleBatch, *, on_publish):
        draft_input: DFlashDraftInputV2 = batch.spec_info
        # Carried over from the previous overlap iteration, maybe on another stream.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        block_size = self.block_size
        prefix_lens = batch.seq_lens
        req_to_token = self.req_to_token_pool.req_to_token

        block_ids, positions_2d = fill_draft_block(
            draft_input.bonus_tokens.view(-1), self.pard_token, prefix_lens, block_size
        )
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + block_size,
            batch_size=bs,
            draft_token_num=block_size,
            device=self.device,
        )
        seq_lens_cpu, seq_lens_sum = self._draft_host_seq_lens(batch, draft_input)
        iteration_ids = self.target_worker.model_runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
            write_virtual=verify_out_cache_loc,
            read_extent=block_size,
        )

        # Draft pass: every row reads P(s - 1), as no newer target feature exists.
        block_features = None
        if self.target_dependent:
            block_features = repeat_last_feature(
                self.p_buf,
                req_to_token,
                batch.req_pool_indices,
                prefix_lens,
                block_size,
            )
        draft_hidden = self._run_draft_block(
            batch=batch,
            input_ids=block_ids,
            positions=positions_2d,
            features=block_features,
            iteration_ids=iteration_ids,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
        ).view(bs, block_size, -1)
        lm_head_weight = self._lm_head_weight()
        row_logits = torch.matmul(draft_hidden, lm_head_weight.T)
        draft_tokens = torch.cat(
            [block_ids[:, :1], draft_rows_to_tokens(row_logits)], dim=1
        )
        del row_logits

        logits_output, can_run_cuda_graph = self._verify(
            batch=batch,
            draft_input=draft_input,
            draft_tokens=draft_tokens,
            positions=positions_2d.reshape(-1),
            verify_out_cache_loc=verify_out_cache_loc,
            iteration_ids=iteration_ids,
        )
        accept_len, bonus = self._accept(
            batch=batch,
            draft_input=draft_input,
            candidates=draft_tokens,
            next_token_logits=logits_output.next_token_logits,
        )
        out_tokens, commit_lens = _commit_accept(draft_tokens, accept_len, bonus)
        if batch.return_logprob:
            compute_spec_logprobs(
                batch, logits_output, out_tokens.reshape(-1), chain_stride=block_size
            )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        if self.target_dependent:
            self._write_features(
                logits_output=logits_output, slots=verify_out_cache_loc
            )
        # Context pass: the verify rows at their own positions write the draft KV.
        context_features = None
        if self.target_dependent:
            _, context_features = gather_shifted_features(
                self.p_buf,
                req_to_token,
                batch.req_pool_indices,
                positions_2d,
                draft_tokens,
            )
        self._run_draft_block(
            batch=batch,
            input_ids=draft_tokens,
            positions=positions_2d,
            features=context_features,
            iteration_ids=iteration_ids,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
        )

        logits_output.hidden_states = None
        logits_output.last_hidden_states = None
        next_draft_input = make_draft_input_v2(
            bonus_tokens=bonus, new_seq_lens=new_seq_lens
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
        )

    def _lm_head_weight(self) -> torch.Tensor:
        vocab_size = int(self.draft_model_runner.model_config.vocab_size)
        return self.draft_model.lm_head.weight[:vocab_size]

    def _draft_host_seq_lens(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInputV2
    ) -> Tuple[torch.Tensor, int]:
        """Host upper bound of the KV the draft block reads: committed + one block."""
        if batch.seq_lens_cpu is not None:
            seq_lens_cpu = (batch.seq_lens_cpu + self.block_size).to(torch.int32)
            return seq_lens_cpu, int(seq_lens_cpu.sum())
        if draft_input.nxt_kv_lens_cpu is not None:
            return draft_input.nxt_kv_lens_cpu, int(draft_input.nxt_kv_lens_sum)
        seq_lens_cpu = torch.tensor(
            [req.kv.kv_allocated_len for req in batch.reqs], dtype=torch.int32
        )
        return seq_lens_cpu, int(seq_lens_cpu.sum())

    def _run_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        features: Optional[torch.Tensor],
        iteration_ids,
        seq_lens_cpu: torch.Tensor,
        seq_lens_sum: int,
    ) -> torch.Tensor:
        """One TARGET_VERIFY draft forward over a [bs, W] block; post-norm hidden."""
        bs, width = input_ids.shape
        num_tokens = bs * width
        input_embeds = self._draft_embeds(input_ids=input_ids, features=features)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids.reshape(-1),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=None,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions.reshape(-1),
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.PARD2,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=(
                torch.full((), num_tokens, dtype=torch.int32, device=self.device)
                if enable_num_token_non_padded()
                else None
            ),
            global_num_token_non_padded_cpu=num_tokens,
            can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
        )
        iteration_ids.bind(forward_batch, self.draft_model_runner.runner_ids)
        with torch.inference_mode(), draft_tp_context(self.draft_parallel_layout):
            draft_out = self.draft_model_runner.forward(forward_batch)
        hidden = draft_out.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("PARD2 draft model returned no hidden states.")
        return hidden[:num_tokens]

    def _verify(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        draft_tokens: torch.Tensor,
        positions: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        iteration_ids,
    ) -> Tuple[LogitsProcessorOutput, bool]:
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            iteration_ids=iteration_ids,
        )
        batch.out_cache_loc = verify_out_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_seq_lens = seq_lens_cpu_backup + self.block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
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
        return target_out.logits_output, target_out.can_run_cuda_graph

    def _accept(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        candidates: torch.Tensor,
        next_token_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampling_info = batch.sampling_info
        bs = candidates.shape[0]
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=self.block_size,
            )
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            return compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
        target_predict = torch.argmax(next_token_logits, dim=-1).view(
            bs, self.block_size
        )
        return compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target_predict
        )
