import logging
from contextlib import ExitStack, contextmanager
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    make_next_draft_input,
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        attn_tp_size = server_args.tp_size // max(server_args.dp_size, 1)
        if server_args.enable_dp_attention and self._draft_is_moe and attn_tp_size > 1:
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        with self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            target_vocab_size=int(
                self.target_worker.model_runner.model_config.vocab_size
            ),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                "markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.gamma), device=self.device
        )

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        self.draft_model.attach_shared_modules(
            embed_tokens=self._resolve_target_embed_tokens(target_model),
            lm_head=lm_head,
        )

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
        )
        self._verify_epilogue = None
        if (
            self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(server_args.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._verify_planner.carries_confidence

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    @contextmanager
    def _draft_context(self):
        with ExitStack() as stack:
            if self._draft_dp_context_enabled:
                stack.enter_context(draft_tp_context(get_attention_tp_group()))
            stack.enter_context(speculative_moe_backend_context())
            stack.enter_context(speculative_moe_a2a_backend_context())
            yield

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        with self._draft_context():
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                self._draft_sampler = self._maybe_build_draft_sampler()
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.tp_rank,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish)

        return self._forward_decode(batch, on_publish)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if self.server_args.enable_dp_attention:
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                self.target_worker.forward_batch_generation(batch)
            return self._decode_idle_result(on_publish=on_publish)

        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(batch)
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        if self.server_args.disaggregation_mode != "prefill":
            logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if self.server_args.enable_dp_attention:
                if self._draft_is_moe:
                    with self._draft_context():
                        self._proposer.run_idle_participation(batch)
                self._run_idle_verify_participation(batch)
            return self._decode_idle_result(on_publish=on_publish)

        self._maybe_inject_pd_prefill_tail(batch, draft_input)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._observers.begin_step()

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens

        confidence = proposal.confidence
        if confidence is None:
            confidence = self._verify_planner.compute_confidence_tensor(
                draft_hidden=proposal.draft_hidden,
                anchor_tokens=draft_block_ids[:, 0],
                draft_tokens=draft_tokens,
                confidence_tap=proposal.confidence_tap,
            )

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
        )
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    inject_gate=fold_eligible,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
        )

    def _maybe_inject_pd_prefill_tail(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
    ) -> None:
        tail_hidden = getattr(draft_input, "prefill_tail_hidden_states", None)
        tail_mask = getattr(draft_input, "prefill_tail_valid_mask", None)
        if tail_hidden is None or tail_mask is None or tail_hidden.numel() == 0:
            return

        tail_hidden = tail_hidden.to(device=self.device, non_blocking=True)
        tail_mask = tail_mask.to(device=self.device, non_blocking=True).bool()
        bs, tail_len = tail_mask.shape
        if bs == 0 or tail_len == 0:
            return

        device = torch.device(self.device)
        row = torch.arange(tail_len, device=device).view(1, tail_len)
        valid_counts = tail_mask.sum(dim=1).to(torch.int64)
        start_pos = batch.seq_lens.to(torch.int64).view(-1, 1) - valid_counts.view(
            -1, 1
        )
        positions_2d = start_pos + row

        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        cache_loc_2d = req_to_token[batch.req_pool_indices, positions_2d.clamp_min(0)]

        flat_mask = tail_mask.reshape(-1)
        if not bool(flat_mask.any()):
            return
        self._kv_injector.inject_target_hidden(
            target_hidden=tail_hidden.reshape(bs * tail_len, -1)[flat_mask],
            cache_loc=cache_loc_2d.reshape(-1)[flat_mask],
            positions=positions_2d.reshape(-1)[flat_mask],
        )
        draft_input.prefill_tail_hidden_states = None
        draft_input.prefill_tail_valid_mask = None

    def _simulated_correct_len(
        self, *, bs: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        buf = self._simulated_correct_drafts_buf
        if buf is None or buf.numel() < bs or buf.dtype != dtype:
            correct_target = int(
                round(min(max(self._simulate_acc_len - 1.0, 0.0), float(self.gamma)))
            )
            buf = torch.full(
                (max(bs, 512),), correct_target, dtype=dtype, device=device
            )
            self._simulated_correct_drafts_buf = buf
        return buf[:bs]

    def _maybe_record_sts_collect(
        self,
        *,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor,
        bs: int,
    ) -> None:
        collect_path = envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
        if not collect_path:
            return
        if not self._verify_planner.carries_confidence:
            return
        confidence_raw = self._verify_planner.last_confidence_raw
        if confidence_raw is None:
            return
        if self._sts_recorder is None:
            self._sts_recorder = StsDataRecorder(
                path_stem=collect_path,
                gamma=self.gamma,
                flush_every=_STS_COLLECT_FLUSH_EVERY,
            )
        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=verify_ids_2d,
            target_predict=target_predict,
        )
        self._sts_recorder.record(
            confidence_raw=confidence_raw,
            num_correct_drafts=num_correct_drafts,
        )

    def _resolve_verify_token_budget(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        confidence: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
    ) -> Optional[int]:
        if not self._verify_planner.schedules_verify_budget or confidence is None:
            return None
        if not self.server_args.disable_overlap_schedule:
            return draft_input.verify_token_budget
        return self._verify_planner.compute_budget_sync(
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
