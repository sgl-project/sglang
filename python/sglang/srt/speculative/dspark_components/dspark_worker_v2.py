import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import torch

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    make_rejection_noise,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import get_pp_stage_load_group
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.draft_parallel import (
    model_draft_parallel_spec,
    resolve_draft_parallel_layout,
)
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
from sglang.srt.speculative.dspark_components.dspark_deferred_commit import (
    DeferredContextCommit,
    defer_context_kv_refusal,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
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
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph import (
    DsparkSamplingAcceptGraph,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_sampling_mask import (
    SpeculativeSamplingMaskCapture,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_pp_context,
    draft_tp_context,
    prepare_mamba_track_for_verify,
)
from sglang.srt.utils import (
    is_cuda,
    is_cuda_alike,
    is_npu,
)
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_is_npu = is_npu()


@runtime_checkable
class _SupportsDSparkTargetHiddenProjection(Protocol):
    def set_dspark_target_hidden_projector(
        self,
        projector: Callable[[torch.Tensor], torch.Tensor],
        *,
        num_context_features: int,
    ) -> bool: ...

    def should_project_dspark_target_hidden(
        self,
        *,
        forward_mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode,
    ) -> bool: ...


def _configure_target_hidden_projection(
    *, target_model, draft_model, is_deepseek_v4_draft: bool
) -> bool:
    """Install the optional token-major projection before the target SP gather."""
    if is_deepseek_v4_draft:
        return False
    if not draft_model.supports_pre_gather_target_hidden_projection:
        return False
    if not isinstance(target_model, _SupportsDSparkTargetHiddenProjection):
        return False
    return bool(
        target_model.set_dspark_target_hidden_projector(
            draft_model.project_target_hidden,
            num_context_features=int(draft_model.num_context_features),
        )
    )


def sim_draw_ahead_enabled(simulated_correct_len: Optional[torch.Tensor]) -> bool:
    """SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD, only under simulated acceptance: with
    real acceptance there is no draw, so the step order stays as it was."""
    return (
        envs.SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD.get()
        and simulated_correct_len is not None
    )


class DSparkWorkerV2(BaseSpecWorker):
    """Non-last PP stages run only the target; draft state belongs to the last stage."""

    _sampling_accept_graph = None
    _rejection_noise = None
    _sim_draw_ahead = False
    num_sim_draws_ahead = 0
    _prepare_inputs_once = False
    num_folded_verify_ids_steps = 0
    num_relayed_in_graph_steps = 0
    _verify_inputs_ready = None

    def weight_update_runners(self) -> list[tuple[str, "ModelRunner"]]:
        return [("draft", self.draft_model_runner)] if self._hosts_draft else []

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        draft_worker_cls: type[TpModelWorker] = TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = get_schedule().page_size
        self.device = target_worker.device
        self._draft_worker = None
        self.enable_dp_spec_prefill_coordination = (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        )
        self._hosts_draft = get_parallel().pp_group.is_last_rank
        if not self._hosts_draft:
            return

        self._draft_is_moe = draft_is_deepseek_v4()
        self.draft_parallel_layout = resolve_draft_parallel_layout(
            model_draft_parallel_spec(
                target_worker.model_runner.spec_aux_config.draft_model_config,
                spec_algorithm=target_worker.model_runner.spec_algorithm,
                dspark_is_moe=self._draft_is_moe,
            )
        )
        self._is_pd_prefill = get_disagg().disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self._is_pd_prefill
        )
        if (
            get_parallel().attn_dp_enabled
            and not _is_npu
            and self._draft_is_moe
            and get_parallel().attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        # Inside the draft scope the context answers the draft's narrowed rank.
        self._target_tp_rank = get_parallel().tp_rank
        load_group = get_pp_stage_load_group()
        with draft_pp_context(), self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
                draft_worker_cls=draft_worker_cls,
                random_seed=target_worker.random_seed,
                load_group=load_group,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

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
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.sample_from_anchor = bool(self.draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        parallel = get_parallel()
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._draft_graph_group = (
            self.draft_parallel_layout.tp_group or parallel.tp_group
        )

        if get_parallel().tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, query_token_num=%s, "
                "sample_from_anchor=%s, mask_token_id=%s, markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self.query_token_num,
                self.sample_from_anchor,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        # Verify-window positions, cache locs and the draft anchor column in one launch.
        self._prepare_inputs_once = (
            envs.SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE.get()
        )
        self.num_folded_verify_ids_steps = 0
        self.num_relayed_in_graph_steps = 0
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.query_token_num), device=self.device
        )

        if getattr(self.draft_model, "uses_own_vocab_modules", False):
            if get_parallel().tp_rank == 0:
                logger.info(
                    "DSpark draft uses its checkpoint-local embedding and LM head."
                )
        else:
            target_model = self.target_worker.model_runner.model
            lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError(
                    "DSpark requires the target model to expose `lm_head` with `weight`."
                )
            self.draft_model.attach_shared_modules(
                embed_tokens=unwrap_lora_layer(
                    self._resolve_target_embed_tokens(target_model)
                ),
                lm_head=lm_head,
            )
        self._target_hidden_projection_enabled = False

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=get_parallel().tp_rank,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
        )
        if (
            get_parallel().attn_dp_enabled
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under attention DP does not "
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
        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        # The step's simulated correct length, read by the eager and graph accepts.
        self._simulated_correct_len = (
            torch.zeros((1,), dtype=torch.int64, device=self.device)
            if self._simulate_acc_len > 0
            else None
        )
        # The simulated acceptance draw (and sim bonus seed) before the draft launch,
        # where the host waits on the previous verify anyway; off with real acceptance.
        self._sim_draw_ahead = sim_draw_ahead_enabled(self._simulated_correct_len)
        self.num_sim_draws_ahead = 0
        self._greedy_draft = envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            random_seed=target_worker.random_seed,
            greedy_draft=self._greedy_draft,
            dp_moe_sync=self._draft_is_moe and get_parallel().attn_dp_enabled,
        )
        self._verify_epilogue = None
        self._sampling_accept_graph: Optional[DsparkSamplingAcceptGraph] = None
        # Counter-based draws from the server seed: every TP rank draws the same coins,
        # so no accept broadcast is needed.
        self._rejection_noise = (
            make_rejection_noise(self.target_worker.random_seed, self.device)
            if is_cuda() and envs.SGLANG_OPT_DSV41_BLOCKWISE_REJECTION.get()
            else None
        )
        target_is_dsv41 = (
            getattr(
                self.target_worker.model_runner.model_config.hf_text_config,
                "model_type",
                None,
            )
            == "deepseek_v41"
        )
        # SGLANG_OPT_DSV41_TARGET_METADATA_EARLY: the static verify's inputs are final
        # after the head; the target runner stages their metadata beside the draft.
        self._verify_inputs_ready: Optional[torch.cuda.Event] = (
            torch.cuda.Event()
            if envs.SGLANG_OPT_DSV41_TARGET_METADATA_EARLY.get()
            and self._verify_planner.mode_value == "static"
            and not get_parallel().attn_dp_enabled
            and torch.device(self.device).type == "cuda"
            else None
        )
        static_epilogue_supported = (
            target_is_dsv41
            and self._verify_planner.mode_value == "static"
            and self._draft_is_moe
            and not get_parallel().attn_dp_enabled
            and get_parallel().pp_size == 1
        )
        # ROCm (V4.1 target only): inside a HIP graph the accept-site TP broadcasts
        # need the group's pynccl communicator
        if (
            (self._verify_planner.is_compact_mode or static_epilogue_supported)
            and self._decode_graph_allowed
            and (is_cuda() or (is_cuda_alike() and target_is_dsv41))
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                fused_argmax=target_is_dsv41,
                simulated_correct_len=self._simulated_correct_len,
                simulate_fold=envs.SGLANG_DSPARK_SIMULATE_FOLD.get(),
                fused_accept_tail=envs.SGLANG_DSPARK_FUSED_ACCEPT_TAIL.get(),
                accept_tail_min_bs=envs.SGLANG_DSPARK_ACCEPT_TAIL_MIN_BS.get(),
                accept_tail_max_bs=envs.SGLANG_DSPARK_ACCEPT_TAIL_MAX_BS.get(),
                deferred_commit=self._maybe_build_deferred_commit(
                    static_epilogue_supported
                ),
                post_fold=(
                    static_epilogue_supported
                    and envs.SGLANG_OPT_DSV41_DSPARK_POST_FOLD.get()
                ),
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                    kv_injector=self._kv_injector,
                    resolve_engram_hasher=(
                        (
                            lambda: (
                                self.model_runner.ngram_embedding_manager.engram_hasher
                            )
                        )
                        if envs.SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH.get()
                        else None
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )
            if (
                envs.SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH.get()
                and static_epilogue_supported
                and is_cuda()
            ):
                self._sampling_accept_graph = DsparkSamplingAcceptGraph(
                    epilogue=self._verify_epilogue,
                    resolve_draft_sampler=lambda: self._draft_sampler,
                    tp_sync=self._tp_sync,
                    fused_argmax=target_is_dsv41,
                    block_verification=get_spec().speculative_use_block_verification,
                    device=self.device,
                    rejection_noise=self._rejection_noise,
                )

        self._simulate_acc_greedy = (
            self._simulate_acc_len > 0 and envs.SGLANG_SIMULATE_ACC_GREEDY.get()
        )
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
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
            simulated_correct_len=self._simulated_correct_len,
            block_verification=get_spec().speculative_use_block_verification,
            rejection_noise=self._rejection_noise,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=get_parallel().tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if self._is_pd_prefill and not self._draft_is_moe:
            self.draft_model.prune_to_ctx_kv_injection()

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._hosts_draft and self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self._hosts_draft:
            return super().spec_v2_attn_backends
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        return draft_tp_context(self.draft_parallel_layout)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if not self._hosts_draft:
            return
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if not self._hosts_draft:
            return
        with draft_pp_context(), self._draft_context():
            self._draft_worker.init_attention_backends()
        self._target_hidden_projection_enabled = _configure_target_hidden_projection(
            target_model=self.target_worker.model_runner.model,
            draft_model=self.draft_model,
            is_deepseek_v4_draft=self._draft_is_moe,
        )
        if self._target_hidden_projection_enabled and get_parallel().tp_rank == 0:
            logger.info(
                "DSpark prefill target-hidden projection runs before "
                "sequence-parallel gather."
            )
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        if not self._hosts_draft:
            return
        capture_decode_cuda_graph = self._decode_graph_allowed
        available_mem = self._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM,
            self.device,
            self.gpu_id,
            group=self._draft_graph_group,
        )
        if is_cuda_alike() and capture_decode_cuda_graph:
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with draft_pp_context(), self._draft_context():
            if capture_decode_cuda_graph:
                # Keep the draft model graph enabled when folded proposal is
                # disabled, but do not capture the proposal head as a tail
                # hook. The proposer will compute base logits and the Markov
                # block eagerly from the graph's hidden states instead. Apart
                # from being the intended precision fallback, skipping the
                # unused hook avoids paying for two proposal computations.
                if envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
                    self._draft_sampler = self._maybe_build_draft_sampler()
                    if self._draft_sampler is not None:
                        self.draft_model_runner.capture_tail_hooks.append(
                            make_draft_sampler_capture_hook(self._draft_sampler)
                        )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            commit = self._deferred_commit()
            if commit is not None:
                self._wire_deferred_commit(commit)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
            if commit is not None:
                commit.disarm()

    def _defer_context_kv_eligible(self, static_epilogue_supported: bool) -> bool:
        """SGLANG_DSPARK_DEFER_CONTEXT_KV: only where nothing moves or reads draft KV
        between a verify and the next draft forward but this worker's forwards."""
        if not envs.SGLANG_DSPARK_DEFER_CONTEXT_KV.get():
            return False
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )

        memory = get_memory()
        refusal = defer_context_kv_refusal(
            static_verify=static_epilogue_supported,
            draft_is_moe=self._draft_is_moe,
            unified_kv=is_unified_kv_triton(),
            unified_memory=memory.enable_unified_memory,
            hicache=memory.enable_hierarchical_cache
            or memory.hicache_storage_backend is not None,
            disaggregated=get_disagg().disaggregation_mode != "null",
            attn_dp=get_parallel().attn_dp_enabled,
        )
        if refusal:
            logger.warning(
                "SGLANG_DSPARK_DEFER_CONTEXT_KV refused, the commit stays in the "
                "verify graph: %s",
                refusal,
            )
            return False
        return True

    def _maybe_build_deferred_commit(
        self, static_epilogue_supported: bool
    ) -> Optional[DeferredContextCommit]:
        if not self._defer_context_kv_eligible(static_epilogue_supported):
            return None
        commit = DeferredContextCommit(
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            stride=self.verify_num_draft_tokens,
            device=self.device,
            kv_injector=self._kv_injector,
        )
        # The target's logits processor: SGLang's forward and a megakernel's head
        # both call it, with the DSpark hidden, once the hidden is final.
        self.model_runner.model.logits_processor.register_forward_pre_hook(
            commit.logits_pre_hook, with_kwargs=True
        )
        return commit

    def _deferred_commit(self):
        epilogue = self._verify_epilogue
        return None if epilogue is None else epilogue.deferred_commit

    def _wire_deferred_commit(self, commit) -> None:
        """Draft side of SGLANG_DSPARK_DEFER_CONTEXT_KV: the draft graph forks the
        staged write at its head and its first KV store joins it."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
            DeepseekV4MultiStepBackend,
        )

        self._proposer.deferred_commit = commit
        self.draft_model_runner.capture_head_hooks.append(commit.capture_head_hook)
        backend = self.draft_model_runner.attn_backend
        backends = [backend]
        if isinstance(backend, DeepseekV4MultiStepBackend):
            backends.extend(backend.attn_backends)
        for b in backends:
            if isinstance(b, DeepseekV4AttnBackend):
                b.pre_kv_store_hook = commit.join
        logger.info(
            "SGLANG_DSPARK_DEFER_CONTEXT_KV: the draft context-KV commit runs at the "
            "head of the next draft graph, joined at its first KV store."
        )

    def _maybe_build_draft_sampler(self):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self._target_tp_rank,
            noise=self._proposer.noise,
            greedy_draft=self._greedy_draft,
            tp_sync=self._tp_sync,
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        if not self._hosts_draft:
            return
        # The staged rows' slots were just freed.
        commit = self._deferred_commit()
        if commit is not None:
            commit.disarm()

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        if not self._hosts_draft:
            return
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        if not self._hosts_draft:
            return None
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        if not self._hosts_draft:
            return
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if not self._hosts_draft:
            return None
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if not self._hosts_draft:
            return
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        if not self._hosts_draft:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if batch.is_extend_in_batch and self.enable_dp_spec_prefill_coordination:
                plan = DPSpecPrefillCoordinationPlan(
                    *batch.dp_spec_prefill_coordination_metadata,
                    draft_width=self._proposer.query_token_num,
                    verify_width=self.verify_num_draft_tokens,
                )
                if plan.heterogeneous:
                    return self._forward_dp_spec_prefill_coordination(
                        batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
                    )
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, grammar_barrier)

    def _forward_dp_spec_prefill_coordination(
        self, batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
    ):
        """Pair local prefill with peer verification after the shared draft pass."""
        if not batch.forward_mode.is_extend():
            return self._forward_decode(
                batch, on_publish, grammar_barrier, coordination_plan=plan
            )

        rank = get_parallel().attn_dp_rank
        target_local_only = len(batch.global_num_tokens) == 1
        plan.apply(
            batch,
            "draft",
            rank,
            local_only=target_local_only or self.draft_parallel_layout.local_dp,
        )
        with self._draft_context():
            self._proposer.run_idle_participation(batch)
        plan.apply(batch, "target", rank, local_only=target_local_only)
        self._verify_planner.note_non_decode_step()
        self._observers.note_prefill_step()
        return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

    def _forward_prefill(
        self,
        batch: ScheduleBatch,
        on_publish,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if get_parallel().attn_dp_enabled:
                self.target_worker.forward_batch_generation(
                    batch, capture_hidden_mode=CaptureHiddenMode.FULL
                )
            return self._decode_idle_result(on_publish=on_publish)

        # The prefill's draft KV may land in slots a pending commit targets.
        commit = self._deferred_commit()
        if commit is not None:
            commit.flush()
        # Placing the draft KV reads scheduler-shared tables, which prefill
        # read-done releases once the target forward's own reads end.
        placement = self._kv_injector.place_prefill(batch)
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        # BCG replay skips model-side Python, so re-evaluate the same pure predicate.
        target_hidden_is_projected = (
            self._target_hidden_projection_enabled
            and self.target_worker.model_runner.model.should_project_dspark_target_hidden(
                forward_mode=batch.forward_mode,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = batch.seq_lens
        batch_output.new_seq_lens = new_seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = placement.positions.device
        draft_pool = self._kv_injector.draft_model_runner.token_to_kv_pool
        if (
            getattr(draft_pool, "swa_valid_from", None) is not None
            and batch.encoder_swa_reset is not None
        ):
            # A newly placed request's ring holds nothing below its cached prefix.
            placed = [i for i, reset in enumerate(batch.encoder_swa_reset) if reset]
            if placed:
                # Pinned copies and a device gather: a list index or a pageable
                # copy here would wait for the whole target forward.
                placed_rows = async_h2d(placed, dtype=torch.int64, device=device)
                valid_from = async_h2d(
                    [batch.prefix_lens[i] for i in placed],
                    dtype=torch.int64,
                    device=device,
                )
                draft_pool.set_swa_valid_from(
                    batch.req_pool_indices.index_select(0, placed_rows), valid_from
                )
        self._kv_injector.inject_prefill(
            placement,
            target_hidden=logits_output.hidden_states,
            token_indices=logits_output.hidden_states_token_indices,
            target_hidden_is_projected=target_hidden_is_projected,
        )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None
        logits_output.hidden_states_token_indices = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=new_seq_lens,
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
            and get_parallel().attn_dp_enabled
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

    def _stage_target_verify_metadata(
        self, *, batch: ScheduleBatch, verify_window, seq_lens_cpu
    ) -> None:
        """Right after the draft graph's launch; the sources are the tensors the verify
        batch uses, and the side stream waits only for the head (_verify_inputs_ready)."""
        target_runner = self.target_worker.model_runner.decode_cuda_graph_runner
        if target_runner is None:
            return
        target_runner.stage_verify_metadata_early(
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            out_cache_loc=verify_window.verify_cache_loc,
            positions=verify_window.positions_2d.reshape(-1),
            max_seq_len=(
                int(seq_lens_cpu.max())
                if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0
                else None
            ),
            after=self._verify_inputs_ready,
        )

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
        grammar_barrier=None,
        coordination_plan=None,
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and get_parallel().attn_dp_enabled
            and batch.global_num_tokens is not None
            else None
        )
        if coordination_plan is not None:
            rank = get_parallel().attn_dp_rank
            target_local_only = len(batch.global_num_tokens) == 1
            coordination_plan.apply(
                batch,
                "draft",
                rank,
                local_only=target_local_only or self.draft_parallel_layout.local_dp,
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if get_parallel().attn_dp_enabled:
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(batch)
                if coordination_plan is not None:
                    coordination_plan.apply(
                        batch, "target", rank, local_only=target_local_only
                    )
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
        anchor_ids = (
            self._proposer.draft_block_ids_view(bs=bs, device=prefix_lens.device)
            if self._prepare_inputs_once
            else None
        )
        window_seq_lens_cpu = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else draft_input.nxt_kv_lens_cpu
        )
        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
            seq_lens_cpu=window_seq_lens_cpu,
            bonus=draft_input.bonus_tokens.view(-1) if anchor_ids is not None else None,
            anchor_ids=anchor_ids,
        )
        if self._verify_inputs_ready is not None:
            # The verify's inputs are final here; the early metadata waits for this.
            self._verify_inputs_ready.record()

        sampling_info = batch.sampling_info
        # Simulated acceptance overrides correct_len, so only the bonus token needs the
        # request's temperature: draft + accept run greedy, the bonus is still sampled.
        simulate_bonus_sampling_info = None
        if (
            self._simulate_acc_greedy
            and sampling_info is not None
            and not sampling_info.is_all_greedy
            and not batch.has_grammar
            and verify_logits_adjustments_are_noop(sampling_info)
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            simulate_bonus_sampling_info = sampling_info
            sampling_info = None
        if self._sim_draw_ahead:
            self._begin_simulated_step_ahead(
                bs=bs, simulate_bonus_sampling_info=simulate_bonus_sampling_info
            )
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
                anchor_prepared=anchor_ids is not None,
            )
        if self._verify_inputs_ready is not None:
            self._stage_target_verify_metadata(
                batch=batch,
                verify_window=verify_window,
                seq_lens_cpu=window_seq_lens_cpu,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens
        if coordination_plan is not None:
            coordination_plan.apply(batch, "target", rank, local_only=target_local_only)

        confidence = proposal.confidence

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
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

        verify_ids_2d = proposal.verify_ids
        if verify_ids_2d is None:
            verify_ids_2d = torch.cat(
                [draft_block_ids[:, :1], draft_tokens], dim=1
            ).contiguous()
        else:
            self.num_folded_verify_ids_steps += 1
            if self.num_folded_verify_ids_steps == 1:
                logger.info("Verify ids read from the draft graph output (first step).")

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (AcceptGreedy);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            # The graph accept is greedy: a sampled simulated bonus stays in the graph
            # only when the epilogue draws it there (SGLANG_DSPARK_SIMULATE_FOLD).
            and (
                simulate_bonus_sampling_info is None
                or self._verify_executor.verify_epilogue.simulate_fold
            )
            and not batch.has_grammar
        )
        if not self._sim_draw_ahead:
            self._verify_executor.begin_simulated_step()
            if fold_eligible and self._verify_executor.verify_epilogue.simulate_fold:
                self._verify_executor.verify_epilogue.begin_sim_bonus(
                    bs,
                    (
                        None
                        if simulate_bonus_sampling_info is None
                        else simulate_bonus_sampling_info.temperatures[:bs]
                    ),
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
                    verify_window=verify_window,
                    inject_gate=fold_eligible,
                )
            else:
                if (
                    self._verify_epilogue is not None
                    and self._verify_planner.mode_value == "static"
                ):
                    self._verify_epilogue.begin_static_step(bs, fold_eligible)
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
        if batch.has_grammar:
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = (
            fold_eligible
            and can_run_cuda_graph
            and (run_compact or self._verify_planner.mode_value == "static")
        )
        region_accept = None
        if (
            not folded_accept
            and can_run_cuda_graph
            and not run_compact
            and self._sampling_graph_eligible(
                batch=batch,
                proposal_folded=proposal.folded,
                sampling_info=sampling_info,
                simulate_bonus_sampling_info=simulate_bonus_sampling_info,
            )
        ):
            region_accept = self._sampling_accept_graph.run(
                bs=bs, target_logits=logits_output.next_token_logits
            )
        # The accept and its commits ran in a graph: the verify graph's fold or the
        # sampling-accept region replayed right after it.
        graph_accept = folded_accept or region_accept is not None
        if region_accept is not None:
            epilogue.note_static_verify(bs)
            accept = region_accept
        else:
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
                simulate_bonus_sampling_info=simulate_bonus_sampling_info,
            )
        engram_in_graph = (
            folded_accept and not run_compact and epilogue.folds_engram_commit
        ) or (
            region_accept is not None and self._sampling_accept_graph.committed_engram
        )
        # The static verify graph committed the engram history when it folded the accept.
        if not engram_in_graph:
            self.model_runner.ngram_embedding_manager.update_after_verify(
                verify_ids_2d=verify_ids_2d,
                req_pool_indices=batch.req_pool_indices,
                commit_lens=accept.commit_lens,
            )
        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
            sampling_info,
            next_token_logits=logits_output.next_token_logits,
            draft_input=draft_input,
            draft_token_num=self.verify_num_draft_tokens,
            bs=bs,
            greedy_mask=draft_block.greedy_mask,
        )
        if sampling_mask_capture is not None:
            logits_output.sampling_mask_output = sampling_mask_capture.build_output(
                out_tokens=accept.out_tokens,
                commit_lens=accept.commit_lens,
            )
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                accept.out_tokens.reshape(-1),
                chain_stride=self.verify_num_draft_tokens,
            )

        relay = self._future_relay_in_graph(
            run_compact=run_compact, folded_accept=graph_accept
        )
        if relay is not None:
            relay.published_seq_lens = accept.new_seq_lens
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_commit = graph_accept and epilogue.folds_commit
        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
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
            confidence_raw=proposal.confidence_raw,
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
        if relay is not None:
            relay.published_bonus = next_draft_input.bonus_tokens
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=self._block_accept_lens(
                accept=accept, folded_accept=graph_accept, bs=bs
            ),
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def _sampling_graph_eligible(
        self,
        *,
        batch: ScheduleBatch,
        proposal_folded: bool,
        sampling_info,
        simulate_bonus_sampling_info,
    ) -> bool:
        """An all-sampled static step the sampling-accept region reproduces exactly."""
        return (
            self._sampling_accept_graph is not None
            and proposal_folded
            and self._verify_planner.mode_value == "static"
            and sampling_info is not None
            and not sampling_info.is_any_greedy
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and verify_logits_adjustments_are_noop(sampling_info)
            and simulate_bonus_sampling_info is None
            and not batch.has_grammar
        )

    def _future_relay_in_graph(self, *, run_compact: bool, folded_accept: bool):
        """The FutureMap relay this step's static graph already scattered into
        (SGLANG_OPT_DSV41_DSPARK_POST_FOLD), or None for the host scatter."""
        epilogue = self._verify_executor.verify_epilogue
        if epilogue is None or run_compact or not folded_accept:
            return None
        relay = epilogue.future_relay
        if relay is None or not relay.adopted:
            return None
        self.num_relayed_in_graph_steps += 1
        if self.num_relayed_in_graph_steps == 1:
            logger.info("Seq lens and bonus relayed by the verify graph (first step).")
        return relay

    def _begin_simulated_step_ahead(self, *, bs: int, simulate_bonus_sampling_info):
        """Issued before the draft launch (neither write depends on it); a step that
        turns out not fold-eligible ignores the sim bonus buffers."""
        executor = self._verify_executor
        executor.begin_simulated_step()
        epilogue = executor.verify_epilogue
        if epilogue is not None and epilogue.simulate_fold:
            epilogue.begin_sim_bonus(
                bs,
                (
                    None
                    if simulate_bonus_sampling_info is None
                    else simulate_bonus_sampling_info.temperatures[:bs]
                ),
            )
        self.num_sim_draws_ahead += 1
        if self.num_sim_draws_ahead == 1:
            logger.info(
                "Simulated acceptance drawn ahead of the draft launch (first step)."
            )

    def _block_accept_lens(self, *, accept, folded_accept: bool, bs: int):
        in_graph = (
            self._verify_epilogue.read_block_accept_lens(bs)
            if folded_accept and self._verify_epilogue is not None
            else None
        )
        if in_graph is not None:
            return in_graph
        return accept.commit_lens + accept.cap_trim_lens

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
        mamba_track_indices = batch.mamba_track_indices

        if mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            seq_lens_cpu = batch.seq_lens_cpu
            if (
                _is_npu
                and seq_lens_cpu is not None
                and seq_lens_cpu.device.type == "cpu"
                and seq_lens_cpu.ndim == 1
                and seq_lens_cpu.numel() == seq_lens_pre_verify.numel()
                and seq_lens_cpu.dtype in (torch.int32, torch.int64)
            ):
                # Verify restores the CPU prefix lengths before the forward.
                # Acceptance can commit at most this many tokens, so this
                # check needs no device readback. Passing None also avoids
                # the NPU backend's conv-state self-copy for untracked rows.
                if all(
                    seq_len >= 0
                    and seq_len // mamba_track_interval
                    == (seq_len + self.verify_num_draft_tokens) // mamba_track_interval
                    for seq_len in seq_lens_cpu.tolist()
                ):
                    mamba_track_indices = None

        if mamba_track_indices is not None:
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
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        if not self._hosts_draft:
            return None
        return self._verify_planner.confidence_budget_prepare()
