import logging
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.environ import envs
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
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    draft_tp_context,
    prepare_mamba_track_for_verify,
)
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
        self._debug_prefill_trace_rids: dict[str, None] = {}
        self._debug_verify_trace_ct_by_rid: dict[str, int] = {}

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

    def _draft_context(self):
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

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
        # capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        capture_decode_cuda_graph = False

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
        if envs.SGLANG_DSPARK_DEBUG_TRACE.get():
            self.draft_model.reset_debug_target_hidden_kv_trace()
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None
        self._maybe_log_prefill_cache_mapping(
            batch=batch,
            positions=positions,
        )

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

    def _debug_trace_enabled(self) -> bool:
        return (
            envs.SGLANG_DSPARK_DEBUG_TRACE.get()
            and get_parallel().attn_tp_rank == 0
        )

    @staticmethod
    def _debug_batch_rids(batch: ScheduleBatch, bs: int) -> list[str]:
        reqs = getattr(batch, "reqs", None) or []
        return [
            str(getattr(reqs[row], "rid", f"unknown-row-{row}"))
            if row < len(reqs)
            else f"unknown-row-{row}"
            for row in range(bs)
        ]

    @staticmethod
    def _debug_remember_bounded(mapping: dict, key, value, limit: int = 64) -> None:
        if key not in mapping and len(mapping) >= limit:
            mapping.pop(next(iter(mapping)))
        mapping[key] = value

    def _maybe_log_prefill_cache_mapping(
        self,
        *,
        batch: ScheduleBatch,
        positions: torch.Tensor,
    ) -> None:
        if not self._debug_trace_enabled():
            return
        rids = self._debug_batch_rids(batch, len(batch.seq_lens))
        new_rids = [rid for rid in rids if rid not in self._debug_prefill_trace_rids]
        if not new_rids:
            return
        for rid in new_rids:
            self._debug_remember_bounded(self._debug_prefill_trace_rids, rid, None)
        try:
            extend_lens = torch.tensor(
                batch.extend_lens, dtype=torch.int64, device=positions.device
            )
            req_pool_indices = batch.req_pool_indices.to(
                device=positions.device, dtype=torch.int64
            )
            token_req_pool_indices = torch.repeat_interleave(
                req_pool_indices, extend_lens
            )
            req_to_token_pool = self.model_runner.req_to_token_pool
            mapped_cache_locs = req_to_token_pool.req_to_token[
                token_req_pool_indices, positions.to(torch.int64)
            ]
            out_cache_locs = batch.out_cache_loc.to(
                device=mapped_cache_locs.device, dtype=mapped_cache_locs.dtype
            )
            num_loc_mismatches = int(
                (mapped_cache_locs != out_cache_locs).sum().item()
            )
            logger.warning(
                "DSpark debug prefill mapping: dp_rank=%s rids=%s rows=%s "
                "num_loc_mismatches=%s shared_req_to_token_pool=%s "
                "shared_token_to_kv_pool=%s positions=[%s,%s] "
                "cache_locs=[%s,%s]",
                self.dp_rank,
                rids,
                int(out_cache_locs.numel()),
                num_loc_mismatches,
                self.draft_model_runner.req_to_token_pool is req_to_token_pool,
                self.draft_model_runner.token_to_kv_pool
                is self.model_runner.token_to_kv_pool,
                int(positions.min().item()),
                int(positions.max().item()),
                int(out_cache_locs.min().item()),
                int(out_cache_locs.max().item()),
            )
        except Exception:
            logger.exception(
                "DSpark debug prefill cache mapping failed; inference is unchanged."
            )

    def _maybe_log_verify_trace(
        self,
        *,
        batch: ScheduleBatch,
        proposal,
        draft_tokens: torch.Tensor,
        accept,
    ) -> None:
        if not self._debug_trace_enabled():
            return
        rids = self._debug_batch_rids(batch, int(draft_tokens.shape[0]))
        selected_rows = [
            row
            for row, rid in enumerate(rids)
            if self._debug_verify_trace_ct_by_rid.get(rid, 0) < 8
        ][:2]
        if not selected_rows:
            return
        selected_rids = [rids[row] for row in selected_rows]
        for rid in selected_rids:
            trace_ct = self._debug_verify_trace_ct_by_rid.get(rid, 0) + 1
            self._debug_remember_bounded(
                self._debug_verify_trace_ct_by_rid, rid, trace_ct
            )

        selected_indices = torch.tensor(
            selected_rows, dtype=torch.int64, device=draft_tokens.device
        )
        draft_tokens_host = (
            draft_tokens.index_select(0, selected_indices).detach().cpu().tolist()
        )
        bonus_tokens_host = (
            accept.bonus.index_select(0, selected_indices).detach().cpu().tolist()
        )
        correct_lens_host = (
            accept.correct_len.index_select(0, selected_indices)
            .detach()
            .cpu()
            .tolist()
        )
        commit_lens_host = (
            accept.commit_lens.index_select(0, selected_indices)
            .detach()
            .cpu()
            .tolist()
        )
        anchor_tokens_host = (
            proposal.draft_block_ids.index_select(0, selected_indices)[:, 0]
            .detach()
            .cpu()
            .tolist()
        )
        draft_top_tokens = None
        draft_top_values = None
        corrected_logits = proposal.draft_block.corrected_logits
        if corrected_logits is not None and corrected_logits.numel() > 0:
            selected_logits = corrected_logits.index_select(0, selected_indices)
            top_values, top_tokens = selected_logits[:, 0, :].float().topk(
                k=5, dim=-1
            )
            draft_top_tokens = top_tokens.detach().cpu().tolist()
            draft_top_values = top_values.detach().cpu().tolist()

        draft_block = proposal.draft_block
        base_top_tokens = (
            None
            if draft_block.debug_base_top_tokens is None
            else draft_block.debug_base_top_tokens.index_select(
                0, selected_indices
            )
            .detach()
            .cpu()
            .tolist()
        )
        base_top_values = (
            None
            if draft_block.debug_base_top_values is None
            else draft_block.debug_base_top_values.index_select(
                0, selected_indices
            )
            .detach()
            .cpu()
            .tolist()
        )
        markov_bias_max_abs = (
            None
            if draft_block.debug_markov_bias_max_abs is None
            else draft_block.debug_markov_bias_max_abs.index_select(
                0, selected_indices
            )
            .detach()
            .cpu()
            .tolist()
        )
        markov_bias_mean_abs = (
            None
            if draft_block.debug_markov_bias_mean_abs is None
            else draft_block.debug_markov_bias_mean_abs.index_select(
                0, selected_indices
            )
            .detach()
            .cpu()
            .tolist()
        )

        draft_hidden = proposal.draft_hidden.index_select(
            0, selected_indices
        ).float()
        logger.warning(
            "DSpark debug verify: dp_rank=%s forward_ct=%s rids=%s bs=%s "
            "folded=%s anchor_tokens=%s draft_tokens=%s bonus_tokens=%s "
            "num_correct_drafts=%s commit_lens=%s "
            "base_first_step_top_tokens=%s base_first_step_top_values=%s "
            "corrected_first_step_top_tokens=%s "
            "corrected_first_step_top_values=%s "
            "markov_bias_max_abs=%s markov_bias_mean_abs=%s "
            "draft_hidden_finite=%s "
            "draft_hidden_min=%.8g draft_hidden_max=%.8g "
            "draft_hidden_mean_abs=%.8g",
            self.dp_rank,
            int(batch.forward_iter),
            selected_rids,
            len(batch.seq_lens),
            proposal.folded,
            anchor_tokens_host,
            draft_tokens_host,
            bonus_tokens_host,
            correct_lens_host,
            commit_lens_host,
            base_top_tokens,
            base_top_values,
            draft_top_tokens,
            draft_top_values,
            markov_bias_max_abs,
            markov_bias_mean_abs,
            bool(torch.isfinite(draft_hidden).all().item()),
            float(draft_hidden.min().item()),
            float(draft_hidden.max().item()),
            float(draft_hidden.abs().mean().item()),
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
                    self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
            return self._decode_idle_result(on_publish=on_publish)

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
            # The in-graph accept epilogue is greedy-only.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
        )
        prepare_mamba_track_for_verify(batch)
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
        self._maybe_log_verify_trace(
            batch=batch,
            proposal=proposal,
            draft_tokens=draft_tokens,
            accept=accept,
        )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        # K3 KDA target verify writes per-step states into speculative scratch;
        # commit the last accepted chain step into the persistent cache.
        verify_stride = self.verify_num_draft_tokens
        chain_accept_index = torch.arange(
            bs * verify_stride, dtype=torch.int64, device=device
        ).view(bs, verify_stride)
        commit_mamba_states_after_verify(
            target_worker=self.target_worker,
            batch=batch,
            accept_lens=accept.commit_lens,
            accept_index=chain_accept_index,
            draft_token_num=verify_stride,
        )

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
            new_seq_lens=accept.new_seq_lens,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
