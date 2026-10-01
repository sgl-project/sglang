import logging
from contextlib import nullcontext
from dataclasses import replace
from typing import Optional

import msgspec
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    PPProxyTensors,
    compute_position,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_parallel,
    get_serving,
    get_spec,
)
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
    DraftProposal,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
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
    VerifyWindow,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    read_target_kv_draft_contract,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    TargetKVInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    TargetVerifyResult,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_tp_context,
    prepare_mamba_track_for_verify,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class DSparkDecodeStep(msgspec.Struct, kw_only=True):
    """Borrowed batch/graph buffers owned by one worker until commit completes."""

    batch: ScheduleBatch
    draft_input: DFlashDraftInputV2
    prefix_lens: torch.Tensor
    verify_window: VerifyWindow
    sampling_info: object
    proposal: DraftProposal
    confidence: torch.Tensor | None
    verify_token_budget: object
    layout: object
    run_compact: bool
    verify_ids_2d: torch.Tensor
    grammar_tree: GrammarTree | None
    fold_eligible: bool
    forward_started: bool = False
    accept_started: bool = False
    committed: bool = False
    target_verify: TargetVerifyResult | None = None
    hidden_strided: torch.Tensor | None = None
    acceptance: AcceptOuts | None = None


class _DSparkPrefillStep(msgspec.Struct):
    batch: ScheduleBatch
    result: GenerationBatchResult | None = None
    committed: bool = False


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self._pending_decode_step = None
        self._pending_prefill_step = None

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        self._is_pd_prefill = server_args.disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            not server_args.disable_cuda_graph and not self._is_pd_prefill
        )
        if (
            server_args.enable_dp_attention
            and self._draft_is_moe
            and ps.attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        with self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                ps=replace(ps, pp_rank=0),
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
        self._target_kv_contract = read_target_kv_draft_contract(
            self.draft_model_runner.model_config.hf_config
        )
        self._capture_hidden_mode = CaptureHiddenMode.FULL
        if self._target_kv_contract is not None:
            parallel = get_parallel()
            if (
                parallel.pp_size != 1
                or parallel.dp_size != 1
                or (
                    get_disagg().disaggregation_mode != "null"
                    and get_disagg().disaggregation_transfer_backend != "mooncake"
                )
                or get_lora().enable_lora
            ):
                raise ValueError(
                    "target-KV DSpark currently requires PP=DP=1, "
                    "Mooncake for disaggregation and no LoRA"
                )
            self._capture_hidden_mode = CaptureHiddenMode.NULL
        self._draft_sampler = None
        self._linear_accept_index_cache = None

        # The mask token is input-only (it is embedded, never sampled), so its
        # bound is the embedding-table row count: the PADDED vocab when the
        # target pads its embedding (e.g. Inkling true vocab 200058, padded
        # 201024, mask 200064), else the plain vocab size.
        target_model_config = self.target_worker.model_runner.model_config
        target_embed_rows = (
            getattr(target_model_config.hf_text_config, "padded_vocab_size", None)
            or target_model_config.vocab_size
        )
        # muP targets declare logits_mup_width_multiplier; the draft was
        # trained against the folded head, so compute_base_logits divides.
        self.draft_model.logits_mup_width_multiplier = getattr(
            target_model_config.hf_text_config, "logits_mup_width_multiplier", None
        )
        self._target_is_mambaish = mambaish_config(target_model_config) is not None
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        if self.ps.tp_rank == 0:
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
            tp_rank=self.ps.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        if self._target_kv_contract is not None:
            self._kv_injector = TargetKVInjector(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                model_runner=self.model_runner,
            )
        else:
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
            and self._decode_graph_allowed
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
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.ps.tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if (
            self._is_pd_prefill
            and not self._draft_is_moe
            and self._target_kv_contract is None
        ):
            self.draft_model.prune_to_ctx_kv_injection()

    @property
    def disaggregation_draft_kv_pool(self):
        # KV-input drafts project the received target prefix on D. Transferring
        # a P-side projection would couple the two draft versions and caches.
        if self._target_kv_contract is not None:
            return None
        return self.primary_draft_kv_pool

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._verify_planner.carries_confidence

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
        if self._target_kv_contract is not None:
            self._kv_injector.bind(
                tokenizer_path=get_serving().tokenizer_path,
                prediction_count=self.gamma,
                mask_token_id=self._mask_token_id,
            )
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = self._decode_graph_allowed
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
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
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
        if self._target_kv_contract is not None:
            self._kv_injector.invalidate_all()

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
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        self._check_no_pending_step()
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish)

        return self._forward_decode(batch, on_publish, grammar_barrier)

    @property
    def needs_cpu_seq_lens(self):
        return (
            self._target_kv_contract is not None
            or self.target_worker.training_capture is not None
        )

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if get_parallel().enable_dp_attention:
                self.target_worker.forward_batch_generation(
                    batch, capture_hidden_mode=self._capture_hidden_mode
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.forward_prefill_stage(batch)
        return self.commit_prefill_stage(batch, batch_output, on_publish=on_publish)

    def forward_prefill_stage(
        self, batch: ScheduleBatch, pp_proxy_tensors: PPProxyTensors | None = None
    ) -> GenerationBatchResult:
        """Produce local target KV and activations, deferring draft projection."""
        self._check_no_pending_step()
        step = _DSparkPrefillStep(batch)
        self._pending_prefill_step = step
        step.result = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=self._capture_hidden_mode,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return step.result

    def commit_prefill_stage(
        self,
        batch: ScheduleBatch,
        batch_output: GenerationBatchResult,
        *,
        next_token_ids: torch.Tensor | None = None,
        on_publish=None,
    ) -> GenerationBatchResult:
        step = self._pending_prefill_step
        if (
            step is None
            or step.batch is not batch
            or step.result is None
            or step.result is not batch_output
            or step.committed
        ):
            raise RuntimeError("DSpark prefill step is stale or already committed")
        if next_token_ids is not None:
            batch_output.next_token_ids = next_token_ids
        if batch_output.next_token_ids is None:
            raise RuntimeError(
                "DSpark prefill commit requires the final stage's sample"
            )
        step.committed = True
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if self._target_kv_contract is not None:
            # Include cached target prefixes whose draft projection may have
            # been evicted or may belong to a previous request.
            if not self._is_pd_prefill:
                self._kv_injector.ensure_context(batch)
            batch_output.next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=batch.seq_lens,
            )
            self._pending_prefill_step = None
            return batch_output

        if logits_output is None or logits_output.hidden_states is None:
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

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        # unified_kv injects into the SWA ring keyed by (draft req slot, position);
        # thread the per-token state_slot + the req's final position so the
        # injector keeps only the last SWA window (older prefill tokens share a
        # ring slot and would race). Cheap; only consumed under unified_kv.
        state_slot = final_pos = None
        if is_unified_kv_triton():
            repeats = ctx_lens.to(torch.int64)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64), repeats
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64), repeats
            )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
            state_slot=state_slot,
            final_pos=final_pos,
        )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        self._pending_prefill_step = None
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
            and get_parallel().enable_dp_attention
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
        self, batch: ScheduleBatch, on_publish, grammar_barrier=None
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
            if get_parallel().enable_dp_attention:
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
            return self._decode_idle_result(on_publish=on_publish)

        step = self.prepare_decode_step(batch)
        self.forward_decode_stage(step)
        self.accept_decode_step(step, grammar_barrier=grammar_barrier)
        return self.commit_decode_step(step, on_publish=on_publish)

    def prepare_decode_step(self, batch: ScheduleBatch) -> DSparkDecodeStep:
        self._check_no_pending_step()
        if not batch.forward_mode.is_decode() or not isinstance(
            batch.spec_info, DFlashDraftInputV2
        ):
            raise RuntimeError("DSpark decode preparation requires a live draft batch")
        # Preparation also borrows graph buffers and reserves KV. On failure the
        # worker must restart instead of reusing a partially prepared window.
        self._pending_decode_step = object()
        draft_input = batch.spec_info
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens
        if self.target_worker.training_capture is not None:
            self.target_worker.training_capture.before_forward(batch.reqs)

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
            if self._target_kv_contract is not None:
                self._kv_injector.ensure_context(batch)
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
            and get_parallel().enable_dp_attention
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

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (accept_greedy_triton);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
            and not batch.has_grammar
        )
        step = DSparkDecodeStep(
            batch=batch,
            draft_input=draft_input,
            prefix_lens=prefix_lens,
            verify_window=verify_window,
            sampling_info=sampling_info,
            proposal=proposal,
            confidence=confidence,
            verify_token_budget=verify_token_budget,
            layout=layout,
            run_compact=run_compact,
            verify_ids_2d=verify_ids_2d,
            grammar_tree=grammar_tree,
            fold_eligible=fold_eligible,
        )
        self._pending_decode_step = step
        return step

    def _check_no_pending_step(self):
        if (
            self._pending_decode_step is not None
            or self._pending_prefill_step is not None
        ):
            raise RuntimeError("DSpark must commit its pending step first")

    def _check_decode_step(self, step):
        if step is not self._pending_decode_step or step.committed:
            raise RuntimeError("DSpark decode step is stale or already committed")

    def forward_decode_stage(
        self, step: DSparkDecodeStep, pp_proxy_tensors: PPProxyTensors | None = None
    ) -> TargetVerifyResult:
        self._check_decode_step(step)
        if step.forward_started:
            raise RuntimeError("DSpark decode step has already launched target verify")
        if step.run_compact and pp_proxy_tensors is not None:
            raise RuntimeError("DSpark pipeline stages require static verify")
        step.forward_started = True
        batch = step.batch
        bs = len(step.prefix_lens)
        prepare_mamba_track_for_verify(batch)
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if step.run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=step.layout,
                    draft_block_ids=step.proposal.draft_block_ids,
                    draft_tokens=step.proposal.draft_block.draft_tokens,
                    bs=bs,
                    device=self.device,
                    sampling_info=step.sampling_info,
                    inject_gate=step.fold_eligible,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=step.draft_input,
                    verify_ids_2d=step.verify_ids_2d,
                    verify_window=step.verify_window,
                    sampling_info=step.sampling_info,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                hidden_strided = None
        step.target_verify = target_verify
        step.hidden_strided = hidden_strided
        return target_verify

    def accept_decode_step(
        self, step: DSparkDecodeStep, *, grammar_barrier=None
    ) -> AcceptOuts:
        self._check_decode_step(step)
        target_verify = step.target_verify
        if target_verify is None or target_verify.logits_output is None:
            raise RuntimeError("DSpark accept requires logits from the final stage")
        if step.accept_started:
            raise RuntimeError("DSpark decode step has already launched acceptance")
        step.accept_started = True
        batch = step.batch
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        if batch.has_grammar:
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=step.grammar_tree,
                sampling_info=step.sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        folded_accept = step.fold_eligible and step.run_compact and can_run_cuda_graph
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=len(step.prefix_lens),
            verify_ids_2d=step.verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=step.proposal.draft_block,
            sampling_info=step.sampling_info,
            draft_input=step.draft_input,
            layout=step.layout,
            prefix_lens=step.prefix_lens,
            draft_tokens=step.proposal.draft_block.draft_tokens,
        )
        step.acceptance = accept
        return accept

    def commit_decode_step(
        self,
        step: DSparkDecodeStep,
        *,
        acceptance: AcceptOuts | None = None,
        on_publish=None,
    ) -> GenerationBatchResult:
        """Commit local or caller-validated remote acceptance after target forward."""
        self._check_decode_step(step)
        if (
            acceptance is not None
            and step.accept_started
            and acceptance is not step.acceptance
        ):
            raise RuntimeError("DSpark cannot replace local acceptance")
        target_verify = step.target_verify
        accept = step.acceptance if acceptance is None else acceptance
        if target_verify is None or accept is None:
            raise RuntimeError(
                "DSpark commit requires target forward and final acceptance"
            )
        # A failed commit cannot be retried against potentially reused KV or graph
        # buffers. Keep the step pending until the whole commit succeeds.
        step.committed = True
        batch = step.batch
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        confidence = step.confidence
        bs = len(step.prefix_lens)
        training_capture = None
        if self.target_worker.training_capture is not None:
            training_capture = self.target_worker.training_capture.after_verify_accept(
                target_verify.training_capture,
                commit_lens=accept.commit_lens,
                out_tokens=accept.out_tokens,
            )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=step.prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_accept = step.fold_eligible and step.run_compact and can_run_cuda_graph
        epilogue = self._verify_executor.verify_epilogue
        folded_commit = folded_accept and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_target_context(
                batch=batch,
                layout=step.layout,
                hidden_strided=step.hidden_strided,
                verify_window=step.verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=step.run_compact,
            )
        if logits_output is not None:
            logits_output.hidden_states = None
            self._observers.observe_verify_step(
                forward_ct=int(batch.forward_iter),
                reqs=batch.reqs,
                bs=bs,
                proposal_folded=step.proposal.folded,
                verify_ids_2d=step.verify_ids_2d,
                target_logits=logits_output.next_token_logits,
                layout=step.layout,
                confidence=confidence,
                prefix_lens=step.prefix_lens,
                draft_tokens=step.proposal.draft_block.draft_tokens,
                draft_block=step.proposal.draft_block,
                sampling_info=step.sampling_info,
                correct_len=accept.correct_len,
                cap_trim_lens=accept.cap_trim_lens,
                bonus=accept.bonus,
                commit_lens=accept.commit_lens,
                verify_token_budget=step.verify_token_budget,
                req_pool_indices=batch.req_pool_indices,
                verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
                dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        result = GenerationBatchResult(
            logits_output=logits_output,
            training_capture=training_capture,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                step.layout.verify_lens.to(torch.int32)
                if step.layout is not None
                else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )
        self._pending_decode_step = None
        return result

    def _commit_target_mamba_states_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the last accepted verify step's KDA/mamba state (chain
        layout: step index = commit_lens - 1) into the persistent caches."""
        if not self._need_mamba_verify_commit:
            return
        # Chain layout only: step index = commit_lens - 1. A tree (topk > 1)
        # layout would need the accept-index mapping the shared spec_utils
        # commit helper does.
        assert get_spec().speculative_eagle_topk in (None, 1)
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = get_exec().mamba.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
