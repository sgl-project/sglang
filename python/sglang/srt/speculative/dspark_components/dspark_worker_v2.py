import copy
import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import torch

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    make_rejection_noise,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
    is_unified_kv_triton,
)
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    PPProxyTensors,
    compute_position,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_loader.weight_utils import get_pp_stage_load_group
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
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
    draft_uses_compressed_kv_attention,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_deferred_draft_kv import (
    DeferredDraftKVStore,
    maybe_build_deferred_draft_kv,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
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
from sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph import (
    DsparkSamplingAcceptGraph,
)
from sglang.srt.speculative.dspark_components.dspark_pp import (
    PPDSparkCandidate,
    PPDSparkIdentity,
    draft_owner,
    owned_token_rows,
    validate_identities,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    target_model_type,
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
    is_hip,
    is_npu,
)
from sglang.srt.utils.device import async_h2d, is_pin_memory_available

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
@dataclass(frozen=True)
class PPDSparkCommitState:
    rids: tuple[str, ...]
    cache_loc: torch.Tensor
    cache_loc_2d: Optional[torch.Tensor]
    positions: torch.Tensor
    state_slot: Optional[torch.Tensor]
    final_pos: Optional[torch.Tensor] = None
    identities: tuple[PPDSparkIdentity, ...] = ()
    token_counts: tuple[int, ...] = ()


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
        self._draft_is_moe = draft_uses_compressed_kv_attention()
        self.draft_parallel_layout = resolve_draft_parallel_layout(
            model_draft_parallel_spec(
                target_worker.model_runner.spec_aux_config.draft_model_config,
                spec_algorithm=target_worker.model_runner.spec_algorithm,
                dspark_is_moe=self._draft_is_moe,
            )
        )
        parallel = get_parallel()
        self.ps = parallel
        self._draft_dp_context_enabled = (
            parallel.attn_dp_enabled and not self._draft_is_moe
        )
        disaggregation_mode = get_disagg().disaggregation_mode
        self._is_pd_prefill = disaggregation_mode == "prefill"
        self._replicated_pp_draft = (
            get_spec().speculative_dspark_pp_replicated_draft and parallel.pp_size > 1
        )
        self._hosts_draft = self._replicated_pp_draft or parallel.pp_group.is_last_rank
        self._pp_draft_dp_enabled = (
            self._replicated_pp_draft and parallel.attn_dp_enabled
        )
        self._replicated_pp_decode = (
            self._replicated_pp_draft and disaggregation_mode == "decode"
        )
        self._replicate_pp_prefill_context = (
            self._replicated_pp_draft
            and self._is_pd_prefill
            and not get_memory().disable_radix_cache
        )
        self._pp_candidates = weakref.WeakKeyDictionary()
        if not self._hosts_draft:
            return

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
        markov_head = self.draft_model.markov_head
        self._sharded_accept = (
            is_hip()
            and not self._greedy_draft
            and markov_head is not None
            and markov_head.vocab_shard is not None
        )
        if self._sharded_accept and self._target_tp_rank == 0:
            logger.info("DSpark sampling accept runs on the vocab shards.")
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            random_seed=target_worker.random_seed,
            greedy_draft=self._greedy_draft,
            sharded_accept=self._sharded_accept,
            dp_moe_sync=self._draft_is_moe and get_parallel().attn_dp_enabled,
            force_draft_embedding=self._replicated_pp_draft,
        )
        self._verify_epilogue = None
        self._deferred_draft_kv: Optional[DeferredDraftKVStore] = None
        self._sampling_accept_graph: Optional[DsparkSamplingAcceptGraph] = None
        # Counter-based draws from the server seed: every TP rank draws the same coins,
        # so no accept broadcast is needed.
        self._rejection_noise = (
            make_rejection_noise(self.target_worker.random_seed, self.device)
            if is_cuda() and envs.SGLANG_OPT_DSV41_BLOCKWISE_REJECTION.get()
            else None
        )
        target_is_dsv41 = target_model_type(self.target_worker) == "deepseek_v41"
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
            and (is_cuda() or _is_npu or (is_cuda_alike() and target_is_dsv41))
        ):
            max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)
            self._deferred_draft_kv = maybe_build_deferred_draft_kv(
                static_verify=static_epilogue_supported,
                draft_is_moe=self._draft_is_moe,
                max_bs=max_bs,
                stride=self.verify_num_draft_tokens,
                device=self.device,
                kv_injector=self._kv_injector,
                target_model=self.model_runner.model,
            )
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max_bs,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                simulated_correct_len=self._simulated_correct_len,
                simulate_fold=envs.SGLANG_DSPARK_SIMULATE_FOLD.get(),
                fused_accept_tail=envs.SGLANG_DSPARK_FUSED_ACCEPT_TAIL.get(),
                accept_tail_min_bs=envs.SGLANG_DSPARK_ACCEPT_TAIL_MIN_BS.get(),
                accept_tail_max_bs=envs.SGLANG_DSPARK_ACCEPT_TAIL_MAX_BS.get(),
                deferred_draft_kv=self._deferred_draft_kv,
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
        if not self._hosts_draft:
            return False
        return self._verify_planner.carries_confidence

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
            deferred_draft_kv = self._deferred_draft_kv
            if deferred_draft_kv is not None:
                deferred_draft_kv.attach_draft(self.draft_model_runner, self._proposer)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )
            if deferred_draft_kv is not None:
                deferred_draft_kv.disarm()

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
            sharded_accept=self._sharded_accept,
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
        if self._deferred_draft_kv is not None:
            self._deferred_draft_kv.disarm()

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

        return self._forward_decode(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def _forward_dp_spec_prefill_coordination(
        self, batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
    ):
        """Pair local prefill with peer verification after the shared draft pass."""
        if not batch.forward_mode.is_extend():
            return self._forward_decode(
                batch,
                on_publish,
                grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
                coordination_plan=plan,
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
            return self._forward_idle_prefill(
                batch=batch,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

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
        output_pp_proxy_tensors = batch_output.pp_hidden_states_proxy_tensors
        target_hidden = (
            logits_output.hidden_states
            if logits_output is not None
            else (
                output_pp_proxy_tensors.tensors.get("dspark_aux_hidden_states")
                if output_pp_proxy_tensors is not None
                else None
            )
        )
        next_token_ids = batch_output.next_token_ids
        if next_token_ids is not None:
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = batch.seq_lens
        batch_output.new_seq_lens = new_seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if target_hidden is None and self.ps.pp_size <= 1:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        has_local_target_hidden = (
            target_hidden is not None and target_hidden.numel() > 0
        )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = self.device
        pin_memory = is_pin_memory_available(device)
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        state_slot = final_pos = None
        if is_unified_kv_triton():
            repeats = ctx_lens.to(torch.int64)
            num_tokens = sum(batch.extend_lens)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64),
                repeats,
                output_size=num_tokens,
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64),
                repeats,
                output_size=num_tokens,
            )
        cache_loc = batch.out_cache_loc
        token_counts = tuple(batch.extend_lens)
        token_indices = (
            logits_output.hidden_states_token_indices
            if logits_output is not None
            else None
        )
        if token_indices is not None:
            cache_loc = cache_loc[token_indices]
            positions = positions[token_indices]
            if state_slot is not None:
                state_slot = state_slot[token_indices]
                final_pos = final_pos[token_indices]
            boundaries = torch.tensor(
                batch.extend_lens, dtype=torch.int64, device=device
            ).cumsum(0)
            token_counts = tuple(
                torch.bincount(
                    torch.bucketize(token_indices, boundaries, right=True),
                    minlength=len(batch.extend_lens),
                ).tolist()
            )
        pp_commit_state = (
            PPDSparkCommitState(
                rids=tuple(req.rid for req in batch.reqs),
                cache_loc=cache_loc,
                cache_loc_2d=None,
                positions=positions,
                state_slot=state_slot,
                final_pos=final_pos,
                identities=tuple(PPDSparkIdentity.from_req(req) for req in batch.reqs),
                token_counts=token_counts,
            )
            if self._replicated_pp_draft
            else None
        )
        pp_projected_context = None
        incoming_ctx = (
            pp_proxy_tensors.tensors.get("dspark_ctx_acc")
            if pp_proxy_tensors is not None
            else None
        )
        local_ctx = None
        if self._draft_is_moe and has_local_target_hidden:
            local_ctx = self.draft_model.project_target_hidden_for_transfer(
                target_hidden
            )
        if output_pp_proxy_tensors is not None:
            output_pp_proxy_tensors.tensors.pop("dspark_aux_hidden_states", None)
        ctx_acc = None
        if incoming_ctx is not None and local_ctx is not None:
            ctx_acc = (
                incoming_ctx.to(device=local_ctx.device, dtype=local_ctx.dtype)
                + local_ctx
            )
        elif incoming_ctx is not None:
            ctx_acc = incoming_ctx.to(device=self.device, non_blocking=True)
        elif local_ctx is not None:
            ctx_acc = local_ctx
        if output_pp_proxy_tensors is not None:
            if ctx_acc is not None:
                output_pp_proxy_tensors.tensors["dspark_ctx_acc"] = ctx_acc
        elif ctx_acc is not None:
            if self._replicated_pp_draft:
                pp_projected_context = ctx_acc
            else:
                self._kv_injector.inject_projected_context(
                    projected_context=ctx_acc,
                    cache_loc=cache_loc,
                    positions=positions,
                    state_slot=state_slot,
                    final_pos=final_pos,
                )
        elif has_local_target_hidden and not (
            self.ps.pp_size > 1 and not self._draft_is_moe
        ):
            self._kv_injector.inject_target_hidden(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                state_slot=state_slot,
                final_pos=final_pos,
                target_hidden_is_projected=target_hidden_is_projected,
            )
        if (
            self._replicated_pp_draft
            and output_pp_proxy_tensors is None
            and pp_projected_context is None
        ):
            raise RuntimeError(
                "Replicated PP DSpark prefill produced no projected context on "
                "the final PP stage."
            )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        if logits_output is not None:
            logits_output.hidden_states = None
            logits_output.hidden_states_token_indices = None

        if next_token_ids is not None:
            batch_output.next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=new_seq_lens,
            )
        batch_output.pp_dspark_commit_state = pp_commit_state
        batch_output.pp_dspark_projected_context = pp_projected_context
        return batch_output

    def _forward_idle_prefill(
        self,
        *,
        batch: ScheduleBatch,
        on_publish,
        pp_proxy_tensors,
        capture_hidden_mode: CaptureHiddenMode,
    ) -> GenerationBatchResult:
        if get_parallel().attn_dp_enabled:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=capture_hidden_mode,
            )
            if self.ps.pp_rank < self.ps.pp_size - 1:
                return batch_output
        return self._decode_idle_result(on_publish=on_publish)

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
        draft_idle: bool = False,
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
            pp_dspark_draft_idle=draft_idle,
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
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
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
                idle_layout = self._idle_verify_ragged_layout(batch)
                if self._replicated_pp_decode:
                    idle_verify_result = self._verify_executor.run_idle_participation(
                        batch=batch,
                        idle_layout=idle_layout,
                        pp_proxy_tensors=pp_proxy_tensors,
                    )
                    if self.ps.pp_rank < self.ps.pp_size - 1:
                        return idle_verify_result
                else:
                    if self._draft_is_moe:
                        self._proposer.run_idle_participation(batch)
                    if coordination_plan is not None:
                        coordination_plan.apply(
                            batch, "target", rank, local_only=target_local_only
                        )
                    self._verify_executor.run_idle_participation(
                        batch=batch,
                        idle_layout=idle_layout,
                        pp_proxy_tensors=pp_proxy_tensors,
                    )
            return self._decode_idle_result(
                on_publish=on_publish,
                draft_idle=(
                    self._replicated_pp_decode
                    and self._draft_is_moe
                    and get_parallel().attn_dp_enabled
                ),
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

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
        sampling_info = batch.sampling_info
        simulate_bonus_sampling_info = None
        if (
            self._replicated_pp_decode
            and sampling_info is not None
            and not sampling_info.is_all_greedy
        ):
            raise ValueError(
                "Replicated PP DSpark currently supports greedy sampling only."
            )
        self._observers.begin_step()
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

        if self._replicated_pp_decode:
            proposal = self.consume_pp_draft(batch)
        else:
            if (
                self._simulated_correct_len is not None
                and sampling_info is not None
                and not sampling_info.is_all_greedy
            ):
                simulate_bonus_sampling_info = sampling_info
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
        pp_commit_state = (
            self._build_pp_commit_state(batch, verify_window)
            if self._replicated_pp_decode
            else None
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
                    verify_window=verify_window,
                    inject_gate=fold_eligible,
                    pp_proxy_tensors=pp_proxy_tensors,
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
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        if self._replicated_pp_decode and logits_output is None:
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=(
                    target_verify.pp_hidden_states_proxy_tensors
                ),
                can_run_cuda_graph=can_run_cuda_graph,
                next_draft_input=make_next_draft_input(
                    bonus_tokens=draft_input.bonus_tokens,
                    new_seq_lens=prefix_lens,
                ),
                speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
                pp_dspark_commit_state=pp_commit_state,
            )
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

        target_logits, watermark_rows = self._force_watermark(
            batch=batch,
            logits_output=logits_output,
            verify_ids_2d=verify_ids_2d,
            positions_2d=verify_window.positions_2d,
            sampling_info=sampling_info,
        )
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
            if simulate_bonus_sampling_info is not None:
                epilogue.begin_sim_bonus(
                    bs, simulate_bonus_sampling_info.temperatures[:bs]
                )
            region_accept = self._sampling_accept_graph.run(
                bs=bs, target_logits=target_logits
            )
        # The accept and its commits ran in a graph: the verify graph's fold or the
        # sampling-accept region replayed right after it.
        graph_accept = folded_accept or region_accept is not None
        if region_accept is not None:
            accept = region_accept
        else:
            accept = self._verify_executor.accept_and_finalize(
                folded_accept=folded_accept,
                bs=bs,
                verify_ids_2d=verify_ids_2d,
                target_logits=target_logits,
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=draft_input,
                layout=layout,
                prefix_lens=prefix_lens,
                draft_tokens=draft_tokens,
                simulate_bonus_sampling_info=simulate_bonus_sampling_info,
            )
        if graph_accept and self._deferred_draft_kv is not None:
            self._deferred_draft_kv.note_staged(batch.reqs)
        if watermark_rows is not None:
            self.target_worker.model_runner.watermark_state.commit_speculative_chain(
                req_pool_indices=batch.req_pool_indices,
                context_hashes=watermark_rows[0],
                selected=watermark_rows[1],
                accept_tokens=accept.out_tokens,
                accept_lens=accept.commit_lens,
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
            with self._draft_context():
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
        folded_commit = folded_accept and epilogue.folds_commit
        pp_projected_context = None
        if self._replicated_pp_decode:
            if run_compact:
                raise RuntimeError(
                    "Replicated PP DSpark does not support compact verify."
                )
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "Replicated PP DSpark requires target hidden states on the "
                    "final PP stage."
                )
            pp_projected_context = self.draft_model.project_target_hidden_for_transfer(
                hidden
            )
        elif not folded_commit:
            # Every decode graph size shares one aux output, which the next
            # target forward overwrites.
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
        pp_next_proposal = None
        if self._replicated_pp_decode:
            self.commit_pp_draft_context(
                state=pp_commit_state,
                rids=pp_commit_state.rids,
                projected_context=pp_projected_context,
                commit_lens=accept.commit_lens,
            )
            if get_spec().speculative_draft_scheduling_policy == "tail":
                pp_next_proposal = self.prepare_pp_draft(batch, next_draft_input)
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
            pp_dspark_commit_state=pp_commit_state,
            pp_dspark_projected_context=pp_projected_context,
            pp_dspark_next_proposal=pp_next_proposal,
        )

    def prepare_pp_draft(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInputV2
    ) -> dict:
        bs = len(batch.seq_lens)
        prefix_lens = draft_input.new_seq_lens
        next_batch = copy.copy(batch)
        next_batch.seq_lens = prefix_lens
        verify_window = alloc_verify_window(
            batch=next_batch,
            bs=bs,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
            seq_lens_cpu=draft_input.nxt_kv_lens_cpu,
        )
        identities = tuple(
            PPDSparkIdentity.from_req(req).next_round() for req in batch.reqs
        )
        rows = [
            i
            for i, req in enumerate(batch.reqs)
            if draft_owner(req.rid, self.ps.pp_size) == self.ps.pp_rank
        ]
        payload = {"identities": [identities[i].to_wire() for i in rows]}
        draft_batch = self._pp_draft_sync_batch(next_batch, local_bs=len(rows))
        if rows:
            from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

            index = torch.tensor(rows, dtype=torch.int64, device=self.device)
            owned = draft_batch
            owned.forward_mode = ForwardMode.DECODE
            owned.reqs = [batch.reqs[i] for i in rows]
            owned.req_pool_indices = batch.req_pool_indices[index]
            owned.req_pool_indices_cpu = batch.req_pool_indices_cpu[rows]
            owned.seq_lens = prefix_lens[index]
            owned.seq_lens_cpu = owned.seq_lens.to("cpu")
            owned.orig_seq_lens = batch.orig_seq_lens[index]
            owned.seq_lens_sum = None
            owned.spec_info = make_next_draft_input(
                bonus_tokens=draft_input.bonus_tokens[index],
                new_seq_lens=owned.seq_lens,
            )
            owned.sampling_info = SamplingBatchInfo.from_schedule_batch(
                owned, self.model_runner.model_config.vocab_size
            ).copy_for_forward()
            owned_window = VerifyWindow(
                positions_2d=verify_window.positions_2d[index],
                verify_cache_loc=verify_window.verify_cache_loc_2d[index].reshape(-1),
                verify_cache_loc_2d=verify_window.verify_cache_loc_2d[index],
                iteration_ids=verify_window.iteration_ids.take_rows(index),
            )
            with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
                proposal = self._proposer.propose(
                    batch=owned,
                    draft_input=owned.spec_info,
                    verify_window=owned_window,
                    bs=len(rows),
                    device=self.device,
                    target_model=self.target_worker.model_runner.model,
                    sampling_info=owned.sampling_info,
                )
                confidence = proposal.confidence
            payload["draft_block_ids"] = proposal.draft_block_ids.clone()
            payload["draft_tokens"] = proposal.draft_block.draft_tokens.clone()
            if confidence is not None:
                payload["confidence"] = confidence.clone()
        elif self._pp_draft_dp_enabled:
            with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
                self._proposer.run_idle_participation(draft_batch)
        return payload

    def prepare_pp_idle_draft(self, batch: ScheduleBatch) -> None:
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            self._proposer.run_idle_participation(
                self._pp_draft_sync_batch(batch, local_bs=0)
            )

    def _pp_draft_sync_batch(
        self, batch: ScheduleBatch, *, local_bs: int
    ) -> ScheduleBatch:
        if not self._pp_draft_dp_enabled:
            return batch
        global_counts = batch.draft_global_num_tokens
        if global_counts is None:
            raise RuntimeError(
                "PP DSpark requires owner counts from the scheduler DP sync."
            )
        if global_counts[self.ps.attn_dp_rank] != local_bs:
            raise RuntimeError(
                "PP DSpark owner count does not match the local filtered batch: "
                f"expected={global_counts[self.ps.attn_dp_rank]}, actual={local_bs}."
            )
        draft_batch = copy.copy(batch)
        draft_batch.global_num_tokens = global_counts
        draft_batch.global_num_tokens_for_logprob = global_counts
        return draft_batch

    def install_pp_draft(self, batch: ScheduleBatch, owner: int, payload: dict) -> None:
        expected = [
            PPDSparkIdentity.from_req(req).next_round()
            for req in batch.reqs
            if draft_owner(req.rid, self.ps.pp_size) == owner
        ]
        validate_identities(expected, payload["identities"])
        row = 0
        for req in batch.reqs:
            if draft_owner(req.rid, self.ps.pp_size) != owner:
                continue
            self._pp_candidates[req] = PPDSparkCandidate(
                identity=expected[row],
                draft_block_ids=payload["draft_block_ids"][row],
                draft_tokens=payload["draft_tokens"][row],
                confidence=payload["confidence"][row]
                if "confidence" in payload
                else None,
            )
            row += 1

    def consume_pp_draft(self, batch: ScheduleBatch) -> DraftProposal:
        blocks, tokens, confidences = [], [], []
        stream = torch.get_device_module(self.device).current_stream()
        for i, req in enumerate(batch.reqs):
            candidate = self._pp_candidates.pop(req, None)
            if candidate is None:
                # Bootstrap/retraction: a deterministic warmup verify seeds the
                # owner pipeline without an extra reverse-PP communication round.
                anchor = batch.spec_info.bonus_tokens[i]
                blocks.append(anchor.expand(self.query_token_num))
                tokens.append(anchor.expand(self.gamma))
                continue
            validate_identities(
                [PPDSparkIdentity.from_req(req)], [candidate.identity.to_wire()]
            )
            candidate.draft_block_ids.record_stream(stream)
            candidate.draft_tokens.record_stream(stream)
            blocks.append(candidate.draft_block_ids)
            tokens.append(candidate.draft_tokens)
            if candidate.confidence is not None:
                candidate.confidence.record_stream(stream)
                confidences.append(candidate.confidence)
        return DraftProposal(
            draft_block_ids=torch.stack(blocks),
            draft_block=DraftBlockResult(
                draft_tokens=torch.stack(tokens),
                corrected_logits=None,
                greedy_draft=self._greedy_draft,
                greedy_mask=torch.ones(
                    len(batch.reqs), dtype=torch.bool, device=self.device
                ),
                temperatures=torch.ones(len(batch.reqs), device=self.device),
            ),
            confidence=torch.stack(confidences)
            if len(confidences) == len(batch.reqs)
            else None,
        )

    def _build_pp_commit_state(
        self, batch: ScheduleBatch, verify_window
    ) -> PPDSparkCommitState:
        state_slot = None
        if is_unified_kv_triton():
            verify_len = verify_window.verify_cache_loc_2d.shape[1]
            state_slot = (
                batch.req_pool_indices.view(-1, 1)
                .expand(len(batch.reqs), verify_len)
                .reshape(-1)
            )
        return PPDSparkCommitState(
            rids=tuple(req.rid for req in batch.reqs),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            state_slot=state_slot,
            identities=tuple(PPDSparkIdentity.from_req(req) for req in batch.reqs),
            token_counts=(verify_window.positions_2d.shape[1],) * len(batch.reqs),
        )

    def commit_pp_draft_context(
        self,
        *,
        state: PPDSparkCommitState,
        rids: tuple[str, ...],
        projected_context: torch.Tensor,
        commit_lens: Optional[torch.Tensor],
        context_token_counts: Optional[tuple[int, ...]] = None,
    ) -> None:
        if state.rids != rids:
            raise RuntimeError(
                "Replicated PP DSpark result does not match the forward batch: "
                f"forward_rids={state.rids}, result_rids={rids}."
            )
        context_token_counts = (
            state.token_counts
            if context_token_counts is None
            else tuple(context_token_counts)
        )
        if len(context_token_counts) != len(state.token_counts):
            raise RuntimeError(
                "Replicated PP DSpark context counts do not match request rows: "
                f"context_counts={context_token_counts}, "
                f"stage_counts={state.token_counts}."
            )
        if any(
            context_count < 0 or context_count > stage_count
            for context_count, stage_count in zip(
                context_token_counts, state.token_counts
            )
        ):
            raise RuntimeError(
                "Replicated PP DSpark context rows exceed the stage-local chunk: "
                f"context_counts={context_token_counts}, "
                f"stage_counts={state.token_counts}."
            )
        context_rows = sum(context_token_counts)
        if projected_context.shape[0] < context_rows:
            raise RuntimeError(
                "Replicated PP DSpark projected context is shorter than its row "
                f"metadata: projected_rows={projected_context.shape[0]}, "
                f"context_rows={context_rows}."
            )
        if self._replicate_pp_prefill_context:
            # A later radix hit can be assigned to either rid-based draft owner.
            # Keep both Prefill replicas complete while the target token slots
            # (and the draft slots indexed by them) are retained by the tree.
            rows = list(range(len(rids)))
        else:
            rows, _ = owned_token_rows(
                rids, state.token_counts, self.ps.pp_rank, self.ps.pp_size
            )
        if not rows:
            return
        selected_rows = set(rows)
        stage_token_rows = []
        context_token_rows = []
        stage_offset = context_offset = 0
        for row, (stage_count, context_count) in enumerate(
            zip(state.token_counts, context_token_counts)
        ):
            if row in selected_rows:
                stage_token_rows.extend(
                    range(
                        stage_offset + stage_count - context_count,
                        stage_offset + stage_count,
                    )
                )
                context_token_rows.extend(
                    range(context_offset, context_offset + context_count)
                )
            stage_offset += stage_count
            context_offset += context_count
        stage_index = torch.tensor(
            stage_token_rows, dtype=torch.int64, device=self.device
        )
        context_index = torch.tensor(
            context_token_rows, dtype=torch.int64, device=self.device
        )
        inject_kwargs = dict(
            projected_context=projected_context[context_index],
            cache_loc=state.cache_loc[stage_index],
            positions=state.positions[stage_index],
            commit_lens=commit_lens[rows] if commit_lens is not None else None,
        )
        self._kv_injector.inject_projected_context(
            **inject_kwargs,
            cache_loc_2d=(
                state.cache_loc_2d[rows] if state.cache_loc_2d is not None else None
            ),
            state_slot=state.state_slot[stage_index]
            if state.state_slot is not None
            else None,
            final_pos=(
                state.final_pos[stage_index] if state.final_pos is not None else None
            ),
        )

    def _force_watermark(
        self,
        *,
        batch: ScheduleBatch,
        logits_output,
        verify_ids_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        sampling_info,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        target_logits = logits_output.next_token_logits
        watermark_state = self.target_worker.model_runner.watermark_state
        if (
            watermark_state is None
            or sampling_info is None
            or not sampling_info.has_watermark_candidates
        ):
            return target_logits, None
        if batch.return_logprob:
            # compute_spec_logprobs reads logits_output after acceptance.
            target_logits = target_logits.clone()
        watermark_rows = watermark_state.force_speculative_chain(
            logits=target_logits,
            req_pool_indices=batch.req_pool_indices,
            chain_tokens=verify_ids_2d,
            positions=positions_2d,
            sampling_info=sampling_info,
        )
        return target_logits, watermark_rows

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
            # A simulated bonus is drawn in the region from this batch's temperatures.
            and (
                simulate_bonus_sampling_info is None
                or simulate_bonus_sampling_info is sampling_info
            )
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
