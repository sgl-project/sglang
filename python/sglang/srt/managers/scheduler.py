# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

import dataclasses
import faulthandler
import functools
import logging
import math
import os
import signal
import sys
import time
from array import array
from contextlib import ExitStack, nullcontext
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.runtime_context import (
    SpawnRanks,
    attention_backends,
    get_context,
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_mm,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    hicache_storage_enabled,
    publish,
    spawn_world_rank,
)

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.distributed import barrier

if TYPE_CHECKING:
    from torch.cuda import Stream as CudaStream

try:
    from sglang.kernels.ops.mamba.triton_ops import (
        initialize_mamba_selective_state_update_backend,
    )
except ImportError:
    initialize_mamba_selective_state_update_backend = None
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.configs.model_config import (
    ModelConfig,
    ModelImpl,
    is_deepseek_v4,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation import role_switch
from sglang.srt.disaggregation.decode import DisaggDecode
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encoder.receiver import (
    MMSchedulerSide,
    create_mm_receiver,
)
from sglang.srt.disaggregation.kv_events import uses_dynamo_format
from sglang.srt.disaggregation.pd_pp_consensus import (
    _DisaggDecodeRing,
    _DisaggPrefillRing,
)
from sglang.srt.disaggregation.prefill import (
    PREFILL_INPUT_POLL_INTERVAL_S,
    DisaggPrefill,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import bootstrap
from sglang.srt.distributed.parallel_state import (
    abort_distributed_environment,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.manager import DllmManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BeginWeightUpdateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    EndWeightUpdateReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    MMInputsProcessError,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    PdRoleSwitchReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    ShutdownReq,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
    sock_send,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer
from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.srt.managers.mm_utils import duplicate_pad_mm_input_ids
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    decide_needs_confidence_relay,
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.request_lists import CoreLists, RequestList
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
    ScheduleBatch,
    check_decode_mem,
    commit_step_counters,
    retract_all,
    retract_decode,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.bundles import (
    KvCapacity,
    KvResources,
    SchedulerServices,
    Streams,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.dynamic_chunk_sizer import (
    DynamicChunkSizer,
)
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import (
    IdleSleeper,
    RustServerIdleSleeper,
)
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    SchedulerWatchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.load_publisher import (
    SchedulerLoadPublisher,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.memory_usage import (
    combine_graph_memory_usage,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    MetricsSink,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.mode_stages import (
    ModeStages,
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.pp_channel import PPChannel
from sglang.srt.managers.scheduler_components.pp_ring import PPRing
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.request_registry import (
    RequestRegistry,
)
from sglang.srt.managers.scheduler_components.request_retirer import (
    RequestRetirer,
    make_abort_req,
)
from sglang.srt.managers.scheduler_components.step_builder import StepBuilder
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    allocate_distinct_stream,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.common import (
    checkpoint_kv_cache,
    release_kv_cache,
)
from sglang.srt.mem_cache.events import LoRANameTable
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_utils.pool import prewarm_graph_pool_borrow
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    flush_trace_batch,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_IDLE,
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_RUN_BATCH,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    scheduler_stage_method,
)
from sglang.srt.observability.startup_time import build_scheduler_startup_time
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.forward_batch_view import worker_forward_batch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import (
    validate_spec_sampling_mask_request,
)
from sglang.srt.speculative.uno_validation import validate_uno_request
from sglang.srt.state_capturer.indexer_topk import destroy_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import destroy_global_experts_capturer
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_mps,
    kill_itself_when_parent_died,
    rank_consensus_checker,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
    triton_load_watch,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
    resolve_image_processor_backend,
)
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_versions import (
    record_weight_version_events,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
else:
    from torch.cuda import StreamContext as CudaStreamContext


logger = logging.getLogger(__name__)


def _prewarm_hccl_group(device, group, device_module):
    warmup_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(warmup_tensor, group=group)
    device_module.synchronize()


# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()


STEP_MAX_US = 2_000_000

# Min wall-clock between load publishes on the stalled no-batch path, which
# spins on_idle without sleeping. Bounds the O(queue) get_loads for both the
# DP-balancing writer and the router-facing socket.
LOAD_STALL_REFRESH_S = 0.05


@dataclasses.dataclass(frozen=True)
class _MultimodalInputBroadcast:
    inputs: Optional[MultimodalInputs] = None
    error: Optional[str] = None


class _MultimodalInputProcessingError(RuntimeError):
    pass


def _accumulate_decode_moment(
    totals: list[float],
    batch_size: int,
    step_us: int,
    generated: int,
) -> None:
    if batch_size <= 0 or step_us <= 0:
        return
    b = float(batch_size)
    t = float(step_us)
    g = float(generated)
    totals[0] += 1.0
    totals[1] += b
    totals[2] += t
    totals[3] += b * b
    totals[4] += b * t
    totals[5] += g


_is_npu = is_npu()
_is_hip = is_hip()


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

    # Logical-to-physical KV capacity ratio; defaults to 1 before pool init.
    kv_shard_widening: int = 1

    # Class-level default so on_idle's stall gate works even if a fork
    # overrides init_load_publisher (which would otherwise not set it).
    _last_stall_publish_ts: float = float("-inf")

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # NOTE: KEEP THE FOLLOWING CODE STYLE for this function:
        # Keep __init__ as an orchestrator: sequence init_* and maybe_init_* calls
        # with minimal glue. Move substantial component-specific logic into
        # dedicated methods instead of adding inline blocks here.
        self.init_startup_timing_begin()
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        self.counters = LaunchCounters()
        self.flags = EngineFlags()
        self.init_soft_watchdog()

        # Parse args
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        # Read once: the tokenizer init and the IPC channels gate on the same
        # value and must not disagree.
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        reject_modes_without_pipeline_support()

        # Init model configs
        self.init_model_config()

        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        bootstrap.init_parallel_runtime(
            server_args=server_args,
            device=get_device().device,
            dist_port=self.nccl_port,
        )
        bootstrap.init_layer_runtime(model_config=self.model_config)

        # Init metrics stats
        self.init_metrics_collector()

        # Init inter-process communication
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        self.init_kv_event_lora_names()
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=get_schedule().page_size,
            spec_algorithm=self.spec_algorithm,
            enable_metrics=get_observability().enable_metrics,
            enable_kv_cache_events=bool(
                get_observability().kv_events_config
                and get_parallel().pp_rank == 0
                and get_parallel().attn_tp_rank == 0
                and get_parallel().attn_cp_rank == 0
            ),
            dynamo_kv_event_format=uses_dynamo_format(
                get_observability().kv_events_config
            ),
            kv_event_lora_names=self.kv_event_lora_names,
            enable_hierarchical_cache=get_memory().enable_hierarchical_cache,
            hicache_draft_plan=(
                self.draft_worker.hicache_draft_plan
                if self.draft_worker is not None
                else None
            ),
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.penalty_store = PenaltyStore(
            self.model_config.vocab_size, self.req_to_token_pool.device
        )
        self.req_to_token_pool.register_on_free_slot(self.penalty_store.free)
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.kv_shard_widening = page_interleave_shard_size(
            self.token_to_kv_pool_allocator
        )
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache
        if get_memory().enable_hierarchical_cache:
            cache_controller = self.tree_cache.cache_controller
            if cache_controller is not None:
                cache_controller.load_fence_stream = (
                    self.tp_worker.model_runner.forward_stream
                )
                if get_memory().enable_unified_memory:
                    # Keep device rows stable until host transfers are acknowledged.
                    # Queue reads and relocation both run on the scheduler thread.
                    self.token_to_kv_pool_allocator.set_host_transfer_move_gate(
                        lambda c=cache_controller: not c.has_inflight_device_transfers()
                    )
        self.emit_metrics_constants()
        self.maybe_init_hccl_dp_prewarm()
        self.init_kv_bundles()

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        self.init_hisparse_coordinator()

        if (
            get_disagg().disaggregation_mode == "decode"
            and get_disagg().disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=self.dp_tp_cpu_group,
                tree_cache=self.tree_cache,
            )
        else:
            self.decode_offload_manager = None

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()
        self.maybe_init_dynamic_chunk_sizer()

        self.init_metrics_sink()
        self.init_metrics_reporter()
        self.scheduler_stage_metrics = self.metrics_sink.scheduler_stage_metrics
        self.enable_fpm = self.metrics_sink.enable_fpm

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Start the embedded Rust frontend (rank 0). Must precede
        # init_disaggregation: on PD prefill the rust api listener also serves
        # the KV bootstrap registry, and the KVManagers built there register to
        # it synchronously. (The listener is bound synchronously inside launch,
        # so the registry is accepting once this returns.) Must also precede
        # the request receiver, which reads self.recv_from_tokenizer to pick
        # its ingress transport.
        self.maybe_init_rust_server()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()
        self.maybe_init_mm_receiver()

        # Init overlap schedule
        self.init_overlap()
        self.finalize_schedule_stream()
        self.init_streams()

        # Init the launch of a step: its forward and its result's copy to host
        self.init_forward_launch()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()
        self.init_dsa_kpool_truncation_align()
        self.check_truncation_align_fits_chunk()

        self.init_weight_updater()

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA drainer for fair scheduling
        self.init_lora_drainer()

        # Init LoRA overlap loader
        self.init_lora_overlap_loader()

        # Init the grammar backend for constrained generation
        self.init_grammar_manager()

        self.maybe_init_scripted_scheduler_hook()

        self.init_request_receiver()

        self.init_step_builder()

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()
        self.init_watchdog_diagnostics()

        self.init_kv_events_publisher()

        self.init_load_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()
        self.init_services()

        self.init_batch_result_processor()

        self.init_request_retirer()

        self.maybe_init_dllm()

        self.maybe_init_disagg_prefill()

        self.maybe_init_disagg_decode()

        self.init_request_registry()

        self.maybe_init_disagg_move_gate()

        self.init_mode_stages()

        self.init_internal_state()

        self.init_rank_consensus_checker()

        self.flags.is_initializing = False
        self.init_startup_timing_summary()

    def init_startup_timing_begin(self) -> None:
        self.scheduler_startup_begin = time.perf_counter()

    def init_startup_timing_summary(self) -> None:
        self.startup_time = build_scheduler_startup_time(
            target_load_weight=self.tp_worker.weight_load_time,
            draft_load_weight=(
                0.0 if self.draft_worker is None else self.draft_worker.weight_load_time
            ),
            kv_cache_allocation=self.kv_cache_allocation_time,
            scheduler_e2e=time.perf_counter() - self.scheduler_startup_begin,
            target_cuda_graph=self.tp_worker.graph_time_usage,
            draft_cuda_graph=(
                None
                if self.draft_worker is None
                else self.draft_worker.graph_time_usage
            ),
        )

    def maybe_init_hccl_dp_prewarm(self) -> None:
        if not (
            _is_npu
            and is_deepseek_v4(self.tp_worker.model_runner.model_config.hf_config)
        ):
            return

        parallel = get_parallel()
        tp_group = parallel.tp_group
        rank = (
            parallel.dp_rank if parallel.dp_rank is not None else tp_group.rank_in_group
        )
        logger.info("HCCL DP prewarm start: rank=%s", rank)
        _prewarm_hccl_group(
            device=tp_group.device,
            group=tp_group.device_group,
            device_module=tp_group.device_module,
        )
        logger.info("HCCL DP prewarm done: rank=%s", rank)

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if get_parallel().pp_size > 1:
                logger.error("only zbal mix mode support pp_size > 1!")
            init_zbal(
                get_parallel().tp_size, get_device().gpu_id, get_parallel().tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)

    def init_metrics_collector(self) -> None:
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            enable_priority_scheduling=get_schedule().enable_priority_scheduling,
            enable_lora=get_lora().enable_lora,
            enable_hierarchical_cache=get_memory().enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            get_parallel().pp_rank == 0
            and get_parallel().attn_tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            # The snapshot taken at construction, not a second bag read: this
            # scheduler gates its tokenizer init on the same value, and the two
            # must not be able to disagree.
            skip_tokenizer_init=self.skip_tokenizer_init,
            metrics_enabled=get_observability().enable_metrics
            and (
                get_parallel().attn_tp_rank == 0
                or get_observability().enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        self.recv_from_tokenizer = None

        if not is_rank_zero:
            return

        self.recv_from_tokenizer = self.ipc_channels.recv_from_tokenizer
        dp_rank = get_parallel().dp_rank if get_parallel().dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                port_args,
                get_parallel().num_dp_ranks,
                dp_rank,
                publish_interval=get_observability().load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        if (
            get_parallel().pp_rank == 0
            and get_parallel().attn_tp_rank == 0
            and get_parallel().attn_cp_rank == 0
            and get_device().sleep_on_idle
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
                flags=self.flags,
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        """Returns the LoadSnapshot it published, or None when disabled,
        throttled, or failed — so co-located sinks (the router-facing load
        publisher) can reuse it instead of walking the queues again."""
        writer = self.load_snapshot_writer
        if writer is None:
            return None
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return None
        writer.publish_counter = 0
        try:
            load = self.load_inquirer.get_loads()
            writer.write(load)
            return load
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)
            return None

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if self.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    image_processor_backend=resolve_image_processor_backend(get_mm()),
                    tokenizer_backend=get_serving().tokenizer_backend,
                    model_name=get_model().model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if (
            self.model_config.is_multimodal
            and self.processor is not None
            and not get_disagg().language_model_only
        ):
            try:
                import_processors("sglang.srt.multimodal.processors")
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        if get_serving().reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=get_serving().reasoning_parser,
                stream_reasoning=False,
                tokenizer=self.tokenizer,
            )
            think_end_ids = reasoning_parser.detector.get_think_end_token_ids(
                self.tokenizer
            )
            if think_end_ids:
                self.model_config.think_end_ids = think_end_ids
            else:
                logger.warning(
                    "Reasoning parser think_end_token %r could not be encoded; "
                    "grammar-gated reasoning is disabled.",
                    reasoning_parser.detector.think_end_token,
                )

            selectable_tokens = getattr(
                reasoning_parser.detector,
                "request_selectable_think_end_tokens",
                (),
            )
            if selectable_tokens:
                selectable_sequences = []
                for end_token in selectable_tokens:
                    token_ids = self.tokenizer.encode(
                        end_token, add_special_tokens=False
                    )
                    if not token_ids:
                        raise ValueError(
                            f"Request-selectable reasoning terminator {end_token!r} "
                            "could not be encoded"
                        )
                    selectable_sequences.append(token_ids)
                self.model_config.request_selectable_think_end_id_sequences = [
                    list(sequence)
                    for sequence in dict.fromkeys(
                        tuple(sequence) for sequence in selectable_sequences
                    )
                ]

    def init_mamba_backend(self) -> None:
        if initialize_mamba_selective_state_update_backend is not None:
            initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # Use the language config already normalized by ModelConfig. Multimodal
        # wrappers expose it under different attributes (for example,
        # ``text_config`` or ``llm_config``).
        config_to_check = self.model_config.hf_text_config

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``,
        # LongCat-2.0 uses ``moe_topk``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
            "moe_topk",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config()

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        initialize_bf16_gemm_config()

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync()

    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=get_device().gpu_id,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        if (
            envs.SGLANG_ENABLE_PP_SPEC.get()
            and get_parallel().pp_size > 1
            and get_parallel().pp_rank != get_parallel().pp_size - 1
        ):
            # PP+spec: the draft model (MTP layer) needs final hidden states and
            # the lm_head, both of which live on the last PP stage only.
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding. It builds its draft
        # from this process's own config: what differs for the draft — the
        # target's context length, the draft load format, its attention backend
        # — is resolved per runner, not on a config copy.
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=get_device().gpu_id,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
        )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.ipc_channels.send_to_tokenizer.send_output,
            )
        else:
            self.external_corpus_manager = None

    def maybe_init_dllm_worker(self):
        if get_exec().dllm.dllm_algorithm is None:
            self.dllm_worker = None
            return
        from sglang.srt.dllm.worker import DllmWorker

        self.dllm_worker = DllmWorker(self.tp_worker, self.server_args)

    def init_target_memory_pool(self):
        """Allocate target KV cache pools if they have not been allocated yet."""
        if (
            self.tp_worker.model_runner.memory_pool_config is not None
            and self.tp_worker.model_runner.req_to_token_pool is not None
            and self.tp_worker.model_runner.token_to_kv_pool_allocator is not None
        ):
            return
        preloaded_weights_bytes = self.tp_worker.preloaded_weights_bytes
        if self.draft_worker is not None:
            preloaded_weights_bytes += self.draft_worker.preloaded_weights_bytes
        self.tp_worker.model_runner.account_preloaded_weights(preloaded_weights_bytes)
        self.tp_worker.alloc_memory_pool()

    def init_memory_pools(self):
        """Allocate KV cache pools for target and draft workers."""
        self.init_target_memory_pool()
        # Lands the retraction backend on the disagg bag before the draft
        # worker's HiCache plan reads it.
        kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.tp_worker)
        if self.draft_worker is not None:
            pool, allocator = self.tp_worker.get_memory_pool()
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=self.tp_worker.model_runner.memory_pool_config,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
            )
            self.draft_worker.init_hicache_draft_plan()

    def init_all_attention_backends(self):
        """Initialize attention backends for all workers."""
        self.tp_worker.init_attention_backends()
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    # Not DynamicGradMode: tensors created under inference mode here could not
    # be written in place by the no_grad disagg event loops.
    @torch.no_grad()
    def init_all_cuda_graphs(self):
        """Capture cuda graphs for all workers."""
        self.tp_worker.init_cuda_graphs()
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def init_model_worker(self):
        # Load model weights.
        self.init_tp_model_worker()
        if get_model().is_startup_weight_load_overlap:
            self.tp_worker.start_startup_weight_load()
        self.maybe_init_draft_worker()
        self.maybe_init_dllm_worker()

        # Prepare KV cache pools for all workers
        tic = time.perf_counter()
        self.init_memory_pools()
        self.kv_cache_allocation_time = time.perf_counter() - tic

        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        model_runner = self.tp_worker.model_runner
        if model_runner.token_to_kv_pool.post_capture_active:
            kv_cache_builder.prepare_hicache_staging(
                tp_worker=self.tp_worker,
                draft_plan=(
                    self.draft_worker.hicache_draft_plan
                    if self.draft_worker is not None
                    else None
                ),
            )
        device_module = torch.get_device_module(model_runner.device)
        self.schedule_stream = device_module.Stream(priority=0)
        # Match run_batch / PPRing.run_slot_batch so warmup allocations stay reusable.
        forward_stream = (
            model_runner.forward_stream
            if (not get_schedule().disable_overlap_schedule)
            or get_parallel().pp_size > 1
            else self.schedule_stream
        )
        with device_module.stream(forward_stream):
            prewarm_graph_pool_borrow()
            if self.draft_worker is None:
                model_runner.prewarm_sampling()
            else:
                self.draft_worker.prewarm_sampling()
        if model_runner.token_to_kv_pool.post_capture_active:
            tic = time.perf_counter()
            model_runner.post_capture_resize_kv_pool(
                draft_runners=(
                    self.draft_worker._draft_model_runners()
                    if self.draft_worker is not None
                    else ()
                )
            )
            self.kv_cache_allocation_time += time.perf_counter() - tic

        if get_model().is_startup_weight_load_overlap:
            self.tp_worker.finalize_startup_weight_load()

        # Adaptive/speculative graphs and post-capture KV sizing can consume
        # the headroom seen by the initial DeepGEMM layout budget. Refresh it
        # after these allocations, before elastic EP rejoins healthy ranks
        # that do not participate in this startup collective.
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            refresh_deep_gemm_layout_memory_budget,
        )

        refresh_deep_gemm_layout_memory_budget(model_runner, only_if_initialized=True)

        if (
            get_exec().moe.elastic_ep_backend is not None
            and get_exec().moe.ep_join_mode == "recover"
        ):
            model_runner.post_capture_elastic_ep_recover()

        # Dispatch the model worker
        if self.dllm_worker is not None:
            self.model_worker = self.dllm_worker
        elif self.spec_algorithm.is_none() or self.draft_worker is None:
            # PP+spec: non-last stages have no draft worker; they run the
            # verify-shaped target forward through the plain tp_worker.
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = (self.dllm_worker or self.tp_worker).get_worker_info()
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
        min_free_slots = resolve_min_free_slots(
            get_schedule().min_free_slots_delay,
            self.max_running_requests,
            is_dflash_family=self.spec_algorithm.is_dflash_family(),
        )
        if min_free_slots is not None:
            self.min_free_slots_delayer = MinFreeSlotsDelayer(
                min_free_slots=min_free_slots
            )
        if not get_parallel().pp_max_micro_batch_size:
            get_context().override(
                "scheduler.pp_max_micro_batch_size_default",
                pp_max_micro_batch_size=max(
                    self.max_running_requests // get_parallel().pp_size, 1
                ),
            )

        self.tp_group = get_parallel().tp_group
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_parallel().attn_tp_group
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_parallel().attn_cp_group
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_parallel().pp_group
        self.world_group = get_parallel().world_group

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # Entry rank is the local rank 0 in that group. Use the CPU (gloo) group to
        # broadcast VLM Python objects and avoid CUDA stream/device coupling (#11910).
        self.dp_tp_group = get_dp_tp_group()
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        self.startup_available_gpu_memory_gb = get_available_gpu_memory(
            self.device, get_device().gpu_id, empty_cache=False
        )
        if get_parallel().tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={get_schedule().chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}="
                f"{self.startup_available_gpu_memory_gb:.2f} GB"
            )

    def emit_metrics_constants(self) -> None:
        if not get_observability().enable_metrics:
            return

        self.metrics_collector.emit_constants(
            max_total_num_tokens=self.max_total_num_tokens,
            max_total_num_tokens_swa=self.swa_tokens_per_layer,
            weight_memory_usage_gb=self.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_memory_usage_gb=(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage
            ),
            graph_memory_usage_gb=combine_graph_memory_usage(
                self.tp_worker.graph_memory_usage,
                (
                    None
                    if self.draft_worker is None
                    else self.draft_worker.graph_memory_usage
                ),
            ),
            # TODO: max_running_requests_under_SLO has no setter — dead chain.
            max_running_requests_under_SLO=None,
            page_size=get_schedule().page_size,
            num_pages=self.max_total_num_tokens
            // self.token_to_kv_pool_allocator.page_size,
            context_len=self.model_config.context_len,
            startup_available_gpu_memory_gb=self.startup_available_gpu_memory_gb,
        )

    def init_kv_bundles(self) -> None:
        model_runner = self.tp_worker.model_runner
        self.kv = KvResources(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            penalty_store=self.penalty_store,
            draft_token_to_kv_pool=(
                self.draft_worker.primary_draft_kv_pool
                if self.draft_worker is not None
                else None
            ),
        )
        self.capacity = KvCapacity(
            max_total_num_tokens=self.max_total_num_tokens,
            kv_shard_widening=self.kv_shard_widening,
            effective_max_total_num_tokens=(
                model_runner.effective_logical_max_total_num_tokens
            ),
            swa_max_total_num_tokens=(
                model_runner.swa_max_total_num_tokens
                if self.tp_worker.is_hybrid_swa
                else None
            ),
            max_token_pool_size=model_runner.max_token_pool_size,
            max_running_requests=self.max_running_requests,
        )

    def init_hisparse_coordinator(self) -> None:
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None
        if not get_memory().enable_hisparse:
            return

        # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

    def init_running_status(self):
        self.tp_size = get_parallel().tp_size
        # Set by a runtime PD role switch to break out of the current event loop.
        self._event_loop_should_restart = False
        # Guards against concurrent/re-entrant PD role switches.
        self._pd_role_switch_in_progress = False
        # Set if a role switch tore down the old role but failed to rebuild
        # either the new or the old role; the instance can no longer serve.
        self._pd_role_switch_unhealthy = False
        self.lists = CoreLists()
        # The flat driver's schedule slot: every decoding request, and the
        # admission flag (the PP ring keeps one per microbatch slot).
        self.flat_slot = ScheduleSlot()
        # The launched steps whose results are not committed yet.
        self.pending = PendingSteps()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self._last_logged_elastic_radix_namespace: Optional[str] = None
        self.session_controller = SessionController(self.tree_cache)
        self._deferred_input_requests: List = []

    def init_chunked_prefill(self):
        self.chunked_prefill_size = get_schedule().chunked_prefill_size
        self.prefill_decode_interval = get_schedule().prefill_decode_interval or 0
        self._prefill_decode_interval_remaining = 0
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
        )

    def maybe_init_dynamic_chunk_sizer(self) -> None:
        """Profile a PP prefill latency model that sizes chunks per stage."""
        self.dynamic_chunk_sizer: Optional[DynamicChunkSizer] = None
        if not (get_schedule().enable_dynamic_chunking and get_parallel().pp_size > 1):
            return
        sizer = DynamicChunkSizer(
            model_runner=self.tp_worker.model_runner,
            model_config=self.model_config,
            tree_cache=self.tree_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            chunked_prefill_size=self.chunked_prefill_size,
            max_prefill_tokens=self.max_prefill_tokens,
            page_size=get_schedule().page_size,
            device=self.device,
        )
        if sizer.profile_and_fit():
            self.dynamic_chunk_sizer = sizer

    def _should_defer_prefill(self) -> bool:
        if self._prefill_decode_interval_remaining == 0:
            return False

        self._prefill_decode_interval_remaining -= 1
        return True

    def _arm_prefill_decode_interval(self, batch: Optional[ScheduleBatch]) -> None:
        if self.prefill_decode_interval == 0 or batch is None:
            return

        # DP attention synchronizes this flag across ranks. This keeps every
        # rank on the same prefill/decode cadence even when only one rank has
        # local prefill work. Non-DP scheduling can use the local mode directly.
        is_extend = (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        )
        if is_extend:
            self._prefill_decode_interval_remaining = self.prefill_decode_interval

    def init_metrics_sink(self) -> None:
        self.metrics_sink = MetricsSink(
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
            device=self.device,
        )

    def init_metrics_reporter(self) -> None:
        # Override point for deployments that need a specialized reporter.
        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self, metrics_sink=self.metrics_sink
        )

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            get_schedule().schedule_policy,
            self.tree_cache,
            get_memory().enable_hierarchical_cache,
            get_schedule().enable_priority_scheduling,
            get_schedule().schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker(
            window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
        )
        self.max_prefill_bs: int = 0
        if get_schedule().enable_prefill_delayer:
            if get_disagg().disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_sink.enable_metrics
                        else None
                    ),
                    max_delay_passes=get_schedule().prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=get_schedule().prefill_delayer_token_usage_low_watermark,
                    debug_log_enabled=get_parallel().attn_tp_rank == 0,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            get_schedule().enable_priority_scheduling
            and not get_schedule().disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_config()

    def init_soft_watchdog(self):
        self.soft_watchdog: Optional[SchedulerWatchdog] = None
        if (x := get_device().soft_watchdog_timeout) is not None:
            self.soft_watchdog = SchedulerWatchdog(
                counters=self.counters,
                flags=self.flags,
                watchdog_timeout=x,
                soft=True,
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = SchedulerWatchdog(
            counters=self.counters,
            flags=self.flags,
            watchdog_timeout=get_device().watchdog_timeout,
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create()
        self.input_blocker = (
            SchedulerInputBlocker(noop=get_parallel().attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        self.transfer_backend = TransferBackend(
            get_disagg().disaggregation_transfer_backend
        )

        # In rust-server mode the KV bootstrap registry is already serving on
        # the rust api listener (maybe_init_rust_server runs before this
        # method — the PrefillBootstrapQueue's KVManager below registers to it
        # synchronously, and a failed registration only retries ~60s then logs,
        # leaving every PD request unroutable). Only the ascend config store,
        # which start_disagg_service would otherwise create, is left to do.
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and self._hosts_rust_server()
        ):
            maybe_create_ascend_config_store(
                transfer_backend=self.transfer_backend,
            )

    def maybe_init_mm_receiver(self) -> None:
        """The receiver of encoder embeddings in EPD language-only mode."""
        self.mm_receiver = None
        if get_disagg().language_only and get_disagg().encoder_transfer_backend in [
            "zmq_to_scheduler",
            "mooncake",
        ]:
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                scheduler_side=MMSchedulerSide(
                    model_config=self.model_config,
                    tokenizer=self.tokenizer,
                    metrics_sink=self.metrics_sink,
                ),
            )

    def maybe_init_disagg_prefill(self) -> None:
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            self.disagg_prefill = None
            return
        self.disagg_prefill = DisaggPrefill(
            server_args=self.server_args,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            kv=self.kv,
            capacity=self.capacity,
            streams=self.streams,
            services=self.services,
            kv_cache_dtype_str=self.tp_worker.model_runner.kv_cache_dtype_str,
            rust_http_port=(
                self.rust_server.http_port if self.rust_server is not None else None
            ),
            transfer_backend=self.transfer_backend,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            pp_group=self.pp_group,
            tp_size=self.tp_size,
            tp_cpu_group=self.tp_cpu_group,
            sliding_window_size=self.sliding_window_size,
            enable_overlap=(not get_schedule().disable_overlap_schedule),
            metrics_collector=self.metrics_collector,
            batch_result_processor=self.batch_result_processor,
            counters=self.counters,
            request_retirer=self.request_retirer,
        )

    def maybe_init_disagg_decode(self) -> None:
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            self.disagg_decode = None
            return
        self.disagg_decode = DisaggDecode(
            server_args=self.server_args,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            kv=self.kv,
            capacity=self.capacity,
            streams=self.streams,
            services=self.services,
            is_hybrid_swa=self.tp_worker.is_hybrid_swa,
            kv_cache_dtype_str=self.tp_worker.model_runner.kv_cache_dtype_str,
            sliding_window_size=self.sliding_window_size,
            hisparse_coordinator=self.hisparse_coordinator,
            can_schedule_lora_req=self.can_schedule_lora_req,
            pool_stats_observer=self.pool_stats_observer,
            decode_offload_manager=self.decode_offload_manager,
            request_retirer=self.request_retirer,
            beam_coordinator=self.beam_coordinator,
            batch_result_processor=self.batch_result_processor,
            flags=self.flags,
            core_lists=self.lists,
            pending=self.pending,
            policy=self.policy,
            future_map=self.future_map,
            ngram_embedding_manager=self.ngram_embedding_manager,
        )

    def init_request_registry(self) -> None:
        """Built after the mode owners; a PD role switch rebuilds it."""
        self.request_registry = RequestRegistry(
            core_lists=self.lists,
            grammar_manager=self.grammar_manager,
            request_retirer=self.request_retirer,
            beam_coordinator=self.beam_coordinator,
            ipc_channels=self.ipc_channels,
            kv=self.kv,
            mm_receiver=self.mm_receiver,
            pending=self.pending,
            decode_offload_manager=self.decode_offload_manager,
            disagg_prefill=self.disagg_prefill,
            disagg_decode=self.disagg_decode,
            dllm=self.dllm,
            hisparse_coordinator=self.hisparse_coordinator,
        )

    def maybe_init_disagg_move_gate(self) -> None:
        """Unified-memory compaction on a PD node waits for the owner's KV
        transfers; every page mover runs on the scheduler thread."""
        owner = self.disagg_prefill or self.disagg_decode
        if get_memory().enable_unified_memory and owner is not None:
            self.token_to_kv_pool_allocator.set_disagg_move_gate(
                owner.unified_memory_move_gate
            )

    def init_mode_stages(self) -> None:
        """Choose the mode's stages, once; a PD role switch rebuilds them."""
        hisparse = self.hisparse_coordinator
        if self.dllm is not None:
            stages = dict(
                schedule=self.get_next_batch_to_run,
                admit=self._admit_dllm_step,
                on_launch=self.dllm.on_launch,
            )
        elif (owner := self.disagg_prefill) is not None:
            stages = dict(
                schedule=self.get_next_disagg_prefill_batch_to_run,
                intake=owner.intake,
                before_forward=owner.before_forward,
                on_prefilled=owner.on_launch,
                after_commit=owner.after_commit,
                # Read only by the overlap flat driver.
                is_result_ready=(
                    owner.is_result_ready
                    if (not get_schedule().disable_overlap_schedule)
                    and get_parallel().pp_size == 1
                    and self._is_continuous_input_polling_enabled()
                    else None
                ),
                ring_consensus=functools.partial(
                    _DisaggPrefillRing,
                    owner,
                    intake=self.process_bootstrapped_queue,
                    schedule=self.get_next_pp_disagg_prefill_batch_to_run,
                    release=self.process_prefill_transfer_queue,
                ),
            )
        elif (owner := self.disagg_decode) is not None:
            stages = dict(
                schedule=self.get_next_disagg_decode_batch_to_run,
                prefetch=owner.prefetch_prefill_dp_rank_queries,
                intake=owner.intake,
                hold_finished=(
                    owner.hold_finished
                    if self.decode_offload_manager is not None
                    else None
                ),
                ring_consensus=functools.partial(
                    _DisaggDecodeRing,
                    owner,
                    schedule=self.get_next_disagg_decode_batch_to_run,
                ),
            )
        else:
            stages = dict(schedule=self.get_next_batch_to_run)
        if hisparse is not None:
            # Launched prefills stage until their host copy lands (PD prefill
            # hands them over instead); the normal schedule then admits them.
            stages.setdefault("on_prefilled", hisparse.on_launch)
            stages["leave_finished"] = hisparse.leave_finished
            if self.disagg_prefill is None and self.disagg_decode is None:
                stages["collect_ready"] = hisparse.collect_ready_reqs
        self.stages = ModeStages(**stages)

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(attn_backends)
        needs_confidence_relay = decide_needs_confidence_relay()
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
            needs_confidence_relay=needs_confidence_relay,
        )
        self.model_worker.attach_future_map(
            self.future_map,
            enable_overlap=(not get_schedule().disable_overlap_schedule),
        )

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via the PP ring; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

    def finalize_schedule_stream(self):
        """Settle the schedule stream the event loop runs on, before any
        component that waits on it is built; it does not change afterwards."""
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        elif (is_cuda() or _is_hip) and (
            (not get_schedule().disable_overlap_schedule) or get_parallel().pp_size > 1
        ):
            # CUDA/HIP streams come from a fixed round-robin pool. Redraw if this
            # stream aliases forward_stream, which would eliminate scheduler
            # overlap. Only CUDA/HIP streams expose a ``cuda_stream`` handle;
            # other accelerators (e.g. NPU/XPU) skip the alias check.
            if self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream:
                self.schedule_stream = allocate_distinct_stream(
                    self.device_module, (self.forward_stream,)
                )

    def init_streams(self) -> None:
        self.streams = Streams(
            device=self.device,
            schedule_stream=self.schedule_stream,
            forward_stream=self.forward_stream,
            copy_stream=self.copy_stream,
        )

    def maybe_init_ngram_embedding(self):
        self.ngram_embedding_manager = (
            self.tp_worker.model_runner.ngram_embedding_manager
        )
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not get_exec().deterministic.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        # Only flashinfer needs chunks on its prefill split tiles; triton tiles
        # keys at absolute positions (align_window_kv_to_tiles).
        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
        }
        prefill_backend, _ = attention_backends()
        env_var, default_size = backend_sizes.get(prefill_backend, (None, None))
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_dsa_kpool_truncation_align(self):
        """Kpool compress-write asserts chunked extends start on pool boundaries.
        Use the LCM to preserve any existing deterministic-inference alignment."""
        from sglang.srt.configs.model_config import (
            get_dsa_index_kpool,
            is_deepseek_dsa,
        )

        if not is_deepseek_dsa(self.model_config.hf_config):
            return

        dsa_index_kpool = get_dsa_index_kpool(self.model_config.hf_config)
        if dsa_index_kpool <= 1:
            return

        if self.truncation_align_size is None:
            self.truncation_align_size = dsa_index_kpool
        else:
            self.truncation_align_size = math.lcm(
                self.truncation_align_size, dsa_index_kpool
            )

    def check_truncation_align_fits_chunk(self):
        """A chunk shorter than the alignment truncates to zero tokens, so a prompt
        longer than the chunk would wait in the queue forever."""
        align = self.truncation_align_size
        chunk = self.chunked_prefill_size
        if align is None or chunk is None or chunk >= align:
            return
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # A decode server never prefills a prompt.
            return
        attn_dp_size = get_parallel().attn_dp_size
        dp_note = (
            f" (--chunked-prefill-size is split across {attn_dp_size} DP attention ranks)"
            if attn_dp_size > 1
            else ""
        )
        raise ValueError(
            f"Chunked prefill is aligned to {align} tokens, but each rank's "
            f"chunked prefill size is {chunk}{dp_note}; a prompt longer than "
            f"{chunk} tokens could never be scheduled. Raise --chunked-prefill-size "
            f"to at least {align * attn_dp_size}, or, with the flashinfer backend "
            f"under deterministic inference, lower "
            f"SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE."
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.weight_updater.destroy_weights_update_group,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    BeginWeightUpdateReqInput,
                    self.weight_updater.begin_weight_update,
                ),
                (
                    EndWeightUpdateReqInput,
                    self.weight_updater.end_weight_update,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.weight_updater.update_weights_from_ipc,
                ),
                (
                    UpdateWeightVersionReqInput,
                    self.handle_update_weight_version,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (PdRoleSwitchReqInput, self.handle_pd_role_switch),
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (ScaleElasticEPReqInput, self.handle_scale_elastic_ep),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
            "startup_time": self.startup_time,
        }

        if get_serving().grpc_port is not None and not (
            get_serving().smg_grpc_mode or get_serving().grpc_mode
        ):
            result_dict["kv_event_sources"] = (
                self.kv_events_publisher.local_kv_event_sources(
                    get_schedule().page_size * get_parallel().dcp_size
                )
            )
        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()
        self.tree_cache.release_host_resources()
        if self.decode_offload_manager is not None:
            self.decode_offload_manager.release_host_resources()
        destroy_global_experts_capturer()
        destroy_global_indexer_capturer()

        rank_consensus_checker.shutdown()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        The event loop blocks until shutdown.
        """
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
        triton_load_watch.install()
        triton_load_watch.mark_serving_started()

        # The global WAR barrier fences the scheduler's next shared-buffer write
        # on the previous forward's read of the unified memory pool.
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.device_module.StreamContext(self.schedule_stream):
            self.metrics_sink.start_scheduler_time_accounting()
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # WAR: keep later schedule_stream writes behind this forward's shared reads.
        # Clearing matters: a phase that skips the publish then falls back to coarse.
        if not self._war_barrier_enabled:
            return
        runner = self.model_worker.last_shared_read_runner
        ev = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            self.schedule_stream.wait_event(ev)
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop: a step commits in the iteration launching it."""
        self.run_flat_loop(overlap=False)

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation:
        a step commits in the next iteration, after that iteration's launch."""
        self.flags.enable_skip_finishing_decode = True
        self.run_flat_loop(overlap=True)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        self.run_flat_loop(overlap=False)

    def _is_continuous_input_polling_enabled(self) -> bool:
        """Check support once, when the PD prefill stages are built."""
        if not envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.get():
            return False
        parallel = get_parallel()
        enabled = (
            parallel.attn_dp_size == 1
            and parallel.attn_cp_size == 1
            and self.spec_algorithm.is_none()
            and self.is_generation
        )
        if not enabled:
            logger.warning(
                "Continuous prefill input polling requires attention DP=CP=1 and "
                "non-speculative autoregressive generation; using the regular overlap loop."
            )
        return enabled

    def _yield_gil_if_needed(self) -> None:
        if self.tp_size == 1:
            # No collective parks this thread at TP=1. A bounded sleep lets
            # bootstrap, transfer and cache workers reacquire the GIL.
            time.sleep(PREFILL_INPUT_POLL_INTERVAL_S)

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self) -> None:
        """An overlap scheduler loop for prefill worker in disaggregation mode;
        with continuous input polling, intake runs while a result is pending."""
        self.run_flat_loop(overlap=True)

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self):
        """A normal scheduler loop for decode worker in disaggregation mode."""
        self.run_flat_loop(overlap=False)

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self):
        """An overlap scheduler loop for decode worker in disaggregation mode."""
        self.run_flat_loop(overlap=True)

    @DynamicGradMode()
    def event_loop_pp(self):
        """A scheduler loop for pipeline parallelism."""
        self.init_pp_ring()
        self.pp_ring.run_ring_loop()

    def init_pp_ring(self):
        """Build the ring driver and its PP channel, once per ring loop."""
        channel = PPChannel(
            pp_group=self.pp_group,
            world_group=self.world_group,
            attn_tp_group=self.attn_tp_group,
            device=self.device,
            device_module=self.device_module,
            schedule_stream=self.schedule_stream,
            forward_stream=self.forward_stream,
            copy_stream=self.copy_stream,
        )
        self.pp_ring = PPRing(
            channel=channel,
            stages=self.stages,
            pending=self.pending,
            recv_skipper=self.recv_skipper,
            ingest_requests=self.ingest_requests,
            launch_step=self.launch_step,
            commit_step=self.commit_step,
            run_batch=self.run_batch,
            on_idle=self.on_idle,
            counters=self.counters,
            mark_idle=self.mark_sched_idled,
            pp_group=self.pp_group,
            device=self.device,
            device_module=self.device_module,
            spec_algorithm=self.spec_algorithm,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tp_worker=self.tp_worker,
            model_worker=self.model_worker,
            future_map=self.future_map,
            schedule_stream=self.schedule_stream,
            copy_stream=self.copy_stream,
            copy_stream_ctx=self.copy_stream_ctx,
            forward_stream_ctx=self.forward_stream_ctx,
        )

    def mark_sched_idled(self):
        self._sched_idled = True

    def run_flat_loop(self, overlap: bool):
        """The loop of every mode without PP. Each pass takes intake, schedules
        and launches a step, then commits: without overlap every pending step,
        with overlap the oldest step launched before this pass. With a commit
        readiness gate (PD prefill continuous polling), a pass whose oldest
        result is not ready on every rank returns to intake instead."""
        stages = self.stages
        prefetch, intake, schedule = stages.prefetch, stages.intake, stages.schedule
        after_commit = stages.after_commit
        is_result_ready = stages.is_result_ready if overlap else None
        slot = self.flat_slot
        # Fixed for the loop, so read once: an env read costs more than the
        # check it gates.
        may_sync_overlap = overlap and self.may_disable_overlap()
        busy_self_check = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        # True while passes repeat intake for the oldest pending result.
        waiting_for_result = False
        # A schedule that found nothing is not retried until new input.
        skip_schedule = False
        # The newest step built and the launched one not sampled yet; under
        # the gate they carry across the passes that wait.
        step = launched = None
        # The step the last full pass built, and whether the chunk it
        # launched was processed: once per launched step, not per pass.
        last_step, chunk_processed = None, False
        while True:
            if self.flags.gracefully_exit:
                break

            # Receive requests
            if prefetch is not None and not self.flags._engine_paused:
                # Pending rooms from the prior cycle can overlap request intake
                # and the tail of the in-flight decode graph.
                prefetch()
            received = self.ingest_requests(waiting_for_result)
            if self.flags._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                self._yield_gil_if_needed()
                continue
            joined = intake() if intake is not None else None
            if joined:
                self.lists.waiting_queue.extend(joined)

            # Schedule the step
            scheduled = False
            if is_result_ready is None:
                if not self.flags.gracefully_exit:
                    step = schedule(slot)
                    scheduled = True
            else:
                if not waiting_for_result or received or joined:
                    skip_schedule = False
                if (
                    len(self.pending) < 2
                    and not self.flags.gracefully_exit
                    and not self._deferred_input_requests
                    and not skip_schedule
                ):
                    step = schedule(
                        slot,
                        # Storage retry deadlines count schedules, not polls.
                        should_retry_storage_prefetch=not waiting_for_result,
                        process_chunk=last_step is None or not chunk_processed,
                    )
                    chunk_processed = last_step is not None
                    skip_schedule = step is None
                    scheduled = True

            # Launch it; a step that must not overlap the pending ones commits
            # them first.
            record = None
            if scheduled:
                self.counters.cur_batch_for_debug = step
                if may_sync_overlap and self.is_disable_overlap_for_batch(step):
                    self.commit_pending_steps()
                    # Opportunistic flush at the disable_overlap sync boundary:
                    # forward_stream is idle (prev forward drained, next not launched),
                    # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                    if get_memory().enable_unified_memory:
                        try:
                            self.token_to_kv_pool_allocator.flush_opportunistic()
                        except Exception:
                            pass
                if step:
                    launched = record = self.launch_step(step, slot, overlap=overlap)

            # Commit: without overlap every pending step; with overlap the
            # oldest launched before this pass, once its result is ready.
            if not overlap:
                num_committed = self.commit_pending_steps()
            elif len(self.pending) > (1 if record is not None else 0):
                if (
                    is_result_ready is not None
                    and not self.flags.gracefully_exit
                    and not is_result_ready(next(iter(self.pending)).result)
                ):
                    waiting_for_result = True
                    self._yield_gil_if_needed()
                    continue
                self.commit_step(self.pending.popleft())
                num_committed = 1
            else:
                num_committed = 0
            if step is None and num_committed == 0:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()
            if after_commit is not None:
                # Poll the KV transfers of the requests the commits handed over.
                self.lists.waiting_queue = after_commit(self.lists.waiting_queue)

            # Run sample of the launched step. It depends on the result of the
            # pending one (e.g., grammar), so it runs after that commit.
            if launched is not None:
                if overlap and self.is_generation:
                    self.launch_batch_sample_if_needed(launched.result, launched.step)
                launched = None

            if step is None:
                # A launch clears it.
                self._sched_idled = True
            if self.recv_skipper is not None:
                self.recv_skipper.on_launch(step)
            if is_result_ready is not None:
                if step is not last_step:
                    last_step, chunk_processed = step, False
                waiting_for_result = False

            if busy_self_check:
                self.invariant_checker.self_check_during_busy()

    def launch_step(
        self,
        step: ScheduleBatch,
        slot: ScheduleSlot,
        overlap: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> PendingStep:
        """Run a step, move the requests it launches, and keep it pending. A
        ring slot's step runs behind the PP fences, until the ring brings its
        outputs back."""
        launch_ts = time.monotonic()
        after_idle_gap = self._sched_idled
        self._sched_idled = False
        if slot.index is not None:
            result = self.pp_ring.run_slot_batch(slot, step, pp_proxy_tensors)
        else:
            result = self.run_batch(step)
            if overlap:
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
        self.join_launched_step(step, slot)
        record = PendingStep(
            step=step,
            result=result,
            slot=slot,
            launch_ts=launch_ts,
            after_idle_gap=after_idle_gap,
        )
        self.pending.push(record)
        return record

    def join_launched_step(self, step: ScheduleBatch, slot: ScheduleSlot) -> None:
        """Move the requests a launched step prefills: out of the waiting queue,
        into the chunked holder for a middle chunk, into decoding once their
        last prefill chunk runs (on a PD prefill node, into prefilling until
        the commit hands their KV over; with HiSparse, into staging until their
        KV is copied to host; in dLLM mode, into dLLM staging until the
        commit)."""
        stages = self.stages
        if stages.on_launch is not None:
            stages.on_launch(step)
            return
        prefill_reqs = step.prefill_reqs()
        if not prefill_reqs:
            return
        admitted = set(prefill_reqs)
        self.lists.waiting_queue = [
            r for r in self.lists.waiting_queue if r not in admitted
        ]
        self.lists.chunked_req = step.chunked_req
        launched = [r for r in prefill_reqs if r is not step.chunked_req]
        if stages.on_prefilled is not None:
            stages.on_prefilled(launched)
        else:
            self.lists.join_decoding(launched, slot)

    def commit_pending_steps(self, keep: int = 0) -> int:
        """Commit the oldest pending steps until keep remain; return how many."""
        num_committed = 0
        while len(self.pending) > keep:
            self.commit_step(self.pending.popleft())
            num_committed += 1
        return num_committed

    def commit_step(self, record: PendingStep) -> None:
        """The host commit of one launched step, popped from pending."""
        self.process_batch_result(record.step, record.result, record.slot)
        self._record_step_counters(record)

    def leave_finished_reqs(self, slot: ScheduleSlot) -> None:
        """Drop the requests a commit finished from decoding (PD decode holds
        those still releasing KV), and from the mode's own lists (HiSparse
        staging); the committed slot's admission reopens if decoding shrank."""
        num_decoding = len(self.lists.decoding)
        if (hold_finished := self.stages.hold_finished) is not None:
            hold_finished([r for r in self.lists.decoding if r.finished()])
        self.lists.decoding = [r for r in self.lists.decoding if not r.finished()]
        if len(self.lists.decoding) < num_decoding:
            slot.batch_is_full = False
        if (leave_finished := self.stages.leave_finished) is not None:
            leave_finished()

    def may_disable_overlap(self) -> bool:
        """Whether is_disable_overlap_for_batch can hold for any step: only
        with consecutive prefill overlap disabled or a speculative algorithm."""
        return (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            or not self.spec_algorithm.is_none()
        )

    def is_disable_overlap_for_batch(self, batch: Optional[ScheduleBatch]) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        # The step launched last, still pending under overlap.
        last = self.pending.newest()
        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(last.step if last is not None else None)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # Sync so the FSM advance lands before the next batch's bitmask, for
        # algorithms whose verify does not run the pending grammar commit
        # (the forward's pending_grammar). Permanent path for host-draft
        # algorithms.
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.grammar_needs_sync()
            and batch.forward_mode.is_decode()
            and len(self.pending) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        """Receive and broadcast input, then dispatch queued requests in order.

        With stop_at_pause, retain the first pause and all following inputs
        until the caller reaches a scheduling boundary. Receiving and broadcasting
        continue each pass; all ranks retain the same suffix.

        The one place a new per-iteration input source belongs; the return
        value exists for the pipeline stages that relay requests onward.
        """
        local_reqs = []
        # Avoid re-enqueuing timeout aborts while dispatch is stopped at a pause.
        if (
            not self._deferred_input_requests
            and get_parallel().pp_rank == 0
            and get_parallel().attn_tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        ):
            local_reqs = self.request_registry.poll_timeout_aborts()
        recv_start_ns = time.monotonic_ns()
        self._deferred_input_requests.extend(
            self.request_receiver.recv_requests(local_reqs=local_reqs)
        )

        num_requests_to_process = len(self._deferred_input_requests)
        if stop_at_pause:
            for index, req in enumerate(self._deferred_input_requests):
                if isinstance(req, PauseGenerationReqInput):
                    num_requests_to_process = index
                    break
        recv_reqs = self._deferred_input_requests[:num_requests_to_process]
        del self._deferred_input_requests[:num_requests_to_process]

        if recv_reqs:
            # Count successful receive/broadcast time as active; empty polls stay idle.
            self.metrics_sink.record_scheduler_active(recv_start_ns)
        self.process_input_requests(recv_reqs)
        return recv_reqs

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_REQUESTS)
    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.session_controller.maybe_reap(now)

        for recv_req in recv_reqs:
            vmm_errors = None
            if get_mm().mm_feature_transport == "cuda_vmm":
                vmm_errors = self._materialize_cuda_vmm_inputs(recv_req)

            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(
                recv_req
            ) and not self.request_registry.is_fully_idle(for_health_check=True):
                self.output_streamer.defer_health_check(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            if vmm_errors is not None and any(vmm_errors):
                self._dispatch_tokenized_mm_requests(recv_req, vmm_errors)
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if self.rust_server is not None:
                    # Embedded Rust server: every control-request response goes
                    # back through the egress ring (the zmq tokenizer socket is
                    # not consumed); the Rust api_server shapes it per-endpoint.
                    self.rust_server.push_control_output(recv_req, output)
                elif isinstance(output, RpcReqOutput):
                    if self.ipc_channels.recv_from_rpc is not None:
                        sock_send(self.ipc_channels.recv_from_rpc, output)
                else:
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()

        # A runtime PD role switch rebuilt the disaggregation structures for a new
        # role. The response has already been sent above; now break out of the
        # current (old-role) event loop so the supervisor can re-dispatch.
        if get_disagg().enable_pd_role_switch and self._event_loop_should_restart:
            self._event_loop_should_restart = False
            raise role_switch.PdRoleSwitchRestart()

    @staticmethod
    def _tokenized_requests(recv_req):
        if isinstance(
            recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            return (recv_req,)
        if isinstance(
            recv_req,
            (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
        ):
            return tuple(recv_req)
        return ()

    def _gather_vmm_materialization_errors(
        self, local_error: Optional[str]
    ) -> List[Optional[str]]:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.dp_tp_cpu_group is not None
        ):
            return [local_error]

        world_size = torch.distributed.get_world_size(group=self.dp_tp_cpu_group)
        errors = [None] * world_size
        torch.distributed.all_gather_object(
            errors,
            local_error,
            group=self.dp_tp_cpu_group,
        )
        return errors

    def _materialize_cuda_vmm_inputs(self, recv_req) -> Optional[List[Optional[str]]]:
        """Materialize each request and agree on failures across TP ranks."""
        tokenized_reqs = self._tokenized_requests(recv_req)
        if not tokenized_reqs:
            return None

        request_errors = []
        for tokenized_req in tokenized_reqs:
            # The request broadcast makes this skip consistent across ranks.
            if tokenized_req.mm_inputs is None:
                request_errors.append(None)
                continue

            local_error = None
            try:
                if not isinstance(tokenized_req.mm_inputs, MultimodalInputs):
                    tokenized_req.mm_inputs = MultimodalInputs.from_processor_output(
                        tokenized_req.mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"

            rank_errors = self._gather_vmm_materialization_errors(local_error)
            failed_ranks = [
                rank for rank, error in enumerate(rank_errors) if error is not None
            ]
            if failed_ranks:
                details = "; ".join(
                    f"rank {rank}: {rank_errors[rank]}" for rank in failed_ranks
                )
                error_msg = f"Multimodal feature reconstruction failed ({details})"
                logger.error(error_msg)
                if isinstance(
                    tokenized_req.mm_inputs,
                    (MultimodalInputs, MultimodalProcessorOutput),
                ):
                    for item in tokenized_req.mm_inputs.mm_items:
                        item.release_transport_proxies()
                tokenized_req.mm_inputs = None
                request_errors.append(error_msg)
            else:
                request_errors.append(None)
        return request_errors

    def _dispatch_tokenized_mm_requests(
        self, recv_req, errors: List[Optional[str]]
    ) -> None:
        tokenized_reqs = self._tokenized_requests(recv_req)
        if len(tokenized_reqs) != len(errors):
            raise RuntimeError("VMM materialization results do not match requests")
        for tokenized_req, error in zip(tokenized_reqs, errors, strict=True):
            if isinstance(tokenized_req, TokenizedGenerateReqInput):
                self.handle_generate_request(tokenized_req, mm_input_error=error)
            elif isinstance(tokenized_req, TokenizedEmbeddingReqInput):
                self.handle_embedding_request(tokenized_req, mm_input_error=error)
            else:
                raise TypeError(
                    f"Unsupported tokenized request type: {type(tokenized_req).__name__}"
                )

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            counters=self.counters,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            flags=self.flags,
            record_weight_version_change=self.record_weight_version_change,
            get_disagg_owners=lambda: (self.disagg_decode, self.disagg_prefill),
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if get_lora().lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                get_lora().max_loras_per_batch,
                get_lora().lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if get_lora().enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(
            server_args=self.server_args,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            dp_tp_group=self.dp_tp_group,
            pp_group=self.pp_group,
        )

    def maybe_init_scripted_scheduler_hook(self) -> None:
        if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
            from sglang.test.scripted_runtime.scheduler_hook import (
                ScriptedSchedulerHook,
            )

            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                scheduler=self,
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def _hosts_rust_server(self) -> bool:
        """Whether this scheduler rank embeds the Rust server (rank 0 only) —
        and with it the server-process duties a Python ``TokenizerManager``
        would otherwise own (e.g. serving the PD KV bootstrap registry)."""
        return envs.SGLANG_RUST_SERVER.get() and (
            get_parallel().pp_rank == 0
            and get_parallel().attn_tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        )

    def maybe_init_rust_server(self) -> None:
        """Start the embedded Rust server (rank 0) if ``SGLANG_RUST_SERVER`` is
        set, and point the ingress receiver at it. All the plumbing lives in
        ``RustServer`` (scheduler_components/rust_scheduler.py)."""

        if not self._hosts_rust_server():
            # Always define the attribute: init_output_streamer and the
            # process_input_requests hook read self.rust_server unconditionally.
            self.rust_server = None
            return

        rust_server = self.get_rust_server_class().launch(
            server_args=self.server_args,
            model_config=self.model_config,
            processor=self.processor,
            tokenizer_path=self.rust_server_tokenizer_path(),
            max_total_num_tokens=self.max_total_num_tokens,
        )
        self.rust_server = rust_server
        # The rust server *is* the ingress source: SchedulerRequestReceiver
        # drains its request ring (rust_server_mode) instead of a zmq socket.
        self.recv_from_tokenizer = rust_server
        # Park the idle loop on the request ring within the rank-0 rust-server
        self.idle_sleeper = RustServerIdleSleeper(rust_server, flags=self.flags)

    def get_rust_server_class(self) -> type[RustServer]:
        return RustServer

    def rust_server_tokenizer_path(self) -> str:
        return get_serving().tokenizer_path

    def init_request_receiver(self) -> None:
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get(),
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            scripted_scheduler_hook=self.scripted_scheduler_hook,
            scheduler_stage_metrics=self.scheduler_stage_metrics,
        )

    def init_step_builder(self) -> None:
        self.step_builder = StepBuilder(
            kv=self.kv,
            future_map=self.future_map,
            counters=self.counters,
            tp_worker=self.tp_worker,
            model_worker=self.model_worker,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            device=self.device,
        )

    def init_dp_attn_adapter(self) -> None:
        # Spec workers have no .model_runner of their own; the prefill graph
        # runner that votes belongs to the target model.
        target_worker = (
            self.tp_worker.target_worker
            if isinstance(self.tp_worker, BaseSpecWorker)
            else self.tp_worker
        )
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            model_runner=target_worker.model_runner,
            flags=self.flags,
            spec_algorithm=self.spec_algorithm,
            step_builder=self.step_builder,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )

    def init_pool_stats_observer(self) -> None:
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=get_memory().enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            # Match the allocator and radix counters' logical units.
            max_total_num_tokens=self.capacity.widened_max_total_num_tokens,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            page_size=get_schedule().page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.capacity.widened_max_total_num_tokens,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            counters=self.counters,
            core_lists=self.lists,
            scheduler_stage_metrics=self.scheduler_stage_metrics,
        )

    def init_watchdog_diagnostics(self) -> None:
        """The watchdogs dump the pools once the invariant checker exists."""
        self.watchdog.invariant_checker = self.invariant_checker
        if self.soft_watchdog is not None:
            self.soft_watchdog.invariant_checker = self.invariant_checker

    def init_rank_consensus_checker(self) -> None:
        groups = []
        if self.attn_cp_group is not None and self.attn_tp_group is not None:
            groups += [self.attn_cp_group, self.attn_tp_group]
        else:
            groups += [self.tp_group]
        if self.pp_group is not None:
            groups += [self.pp_group]
        rank_consensus_checker.configure(groups)

    def init_kv_event_lora_names(self) -> None:
        """Init the table that maps lora_id to adapter name for KV events."""
        self.kv_event_lora_names = LoRANameTable.from_lora_refs(get_lora().lora_paths)

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=get_observability().kv_events_config,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            metrics_sink=self.metrics_sink,
        )

    def init_load_publisher(self) -> None:
        # Router-facing load reporting; rank gating and no-op fallback live
        # inside the component. Same interval as the DP-balancing writer so
        # the two fire in phase and the load sink always reuses that snapshot
        # instead of walking the queues itself.
        self.load_publisher = SchedulerLoadPublisher(
            kv_events_config=get_observability().kv_events_config,
            load_publish_endpoint=get_observability().load_publish_endpoint,
            publish_interval=get_observability().load_snapshot_publish_interval,
        )

    def init_load_inquirer(self) -> None:
        self.total_prefill_uncached_tokens = 0
        self.total_prefill_busy_us = 0
        self.decode_moment_totals: list[float] = [0.0] * 6
        self._prev_step: Optional[Tuple[int, float, bool]] = None
        self._prev_prefill_end_ts: Optional[float] = None
        self._sched_idled = False
        self.load_inquirer = SchedulerLoadInquirer(
            # The worker reports logical DCP capacity; add KV shard widening.
            max_total_num_tokens=self.capacity.widened_max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            policy=self.policy,
            metrics_sink=self.metrics_sink,
            get_request_lists=self.request_lists,
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode.prealloc_queue,
            get_total_prefill_uncached_tokens=lambda: (
                self.total_prefill_uncached_tokens
            ),
            get_total_prefill_busy_us=lambda: self.total_prefill_busy_us,
            get_decode_moment_totals=lambda: self.decode_moment_totals,
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = self.get_output_streamer_class()(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            send_to_tokenizer=self.ipc_channels.send_to_tokenizer,
            tree_cache=self.tree_cache,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            rust_server=self.rust_server,
        )

    def init_services(self) -> None:
        self.services = SchedulerServices(
            metrics_sink=self.metrics_sink,
            output_streamer=self.output_streamer,
            scheduler_stage_metrics=self.scheduler_stage_metrics,
        )

    def get_output_streamer_class(self) -> type[SchedulerOutputStreamer]:
        return SchedulerOutputStreamer

    def init_beam_coordinator(self) -> None:
        self.beam_coordinator = BeamCoordinator(
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            max_req_len=self.max_req_len,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            future_map=self.future_map,
            core_lists=self.lists,
        )

    def init_batch_result_processor(self) -> None:
        self.init_beam_coordinator()
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            enable_overlap=(not get_schedule().disable_overlap_schedule),
            model_config=self.model_config,
            kv=self.kv,
            services=self.services,
            hisparse_coordinator=self.hisparse_coordinator,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                model_config=self.model_config
            ),
            beam_coordinator=self.beam_coordinator,
            pending=self.pending,
        )

    def init_request_retirer(self) -> None:
        self.request_retirer = RequestRetirer(
            kv=self.kv,
            beam_coordinator=self.beam_coordinator,
            output_streamer=self.output_streamer,
            ipc_channels=self.ipc_channels,
        )

    def maybe_init_dllm(self) -> None:
        if get_exec().dllm.dllm_algorithm is None:
            self.dllm = None
            return
        self.dllm = DllmManager(
            dllm_config=DllmConfig.from_server_args(self.server_args),
            kv=self.kv,
            services=self.services,
            policy=self.policy,
            new_token_ratio_tracker=self.new_token_ratio_tracker,
            page_size=get_schedule().page_size,
            max_prefill_tokens=self.max_prefill_tokens,
            chunked_prefill_size=self.chunked_prefill_size,
            priority_scheduling_preemption_threshold=(
                get_schedule().priority_scheduling_preemption_threshold
            ),
            enable_priority_preemption=self.enable_priority_preemption,
            enable_priority_scheduling=get_schedule().enable_priority_scheduling,
            truncation_align_size=self.truncation_align_size,
            context_len=self.model_config.context_len,
            beam_coordinator=self.beam_coordinator,
            step_builder=self.step_builder,
            request_retirer=self.request_retirer,
        )

    def init_internal_state(self) -> None:
        self.internal_state = SchedulerInternalStateController(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            startup_available_gpu_memory_gb=self.startup_available_gpu_memory_gb,
            metrics_sink=self.metrics_sink,
            rust_server=self.rust_server,
            get_request_lists=self.request_lists,
            get_startup_time=lambda: self.startup_time,
            record_step_time=RECORD_STEP_TIME,
        )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        max_new_tokens = (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        )
        if (
            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get() is not None
            and envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get() > 0
        ):
            if max_new_tokens > envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get():
                logger.warning(
                    f"Capping max_new_tokens of request {req.rid} to "
                    f"SGLANG_MAX_NEW_TOKENS_LIMIT={envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()} "
                    f"(requested: {req.sampling_params.max_new_tokens})."
                )
            max_new_tokens = min(max_new_tokens, envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get())

        # Keep this bound consistent with PrefillAdder's admission budget.
        max_new_tokens = max(
            0,
            min(
                max_new_tokens,
                self.max_req_len - input_len - 1,
            ),
        )
        # PrefillAdder reserves one page per shard; the allocator reserves one.
        # Subtract the other N - 1 pages to keep queue admission schedulable.
        token_capacity = (
            self.capacity.widened_max_total_num_tokens
            - get_schedule().page_size * (self.capacity.kv_shard_widening - 1)
        )
        max_new_tokens = self.token_to_kv_pool_allocator.max_new_tokens_for_memory(
            input_len,
            max_new_tokens,
            token_capacity=token_capacity,
            sliding_window_size=self.sliding_window_size,
            chunk_size=self.chunked_prefill_size,
        )
        if max_new_tokens is None:
            req.set_finish_with_abort(
                f"Request prompt exceeds the KV memory budget: input_len={input_len}."
            )
            max_new_tokens = 0

        req.sampling_params.max_new_tokens = max(0, max_new_tokens)
        # Clipping above can push max_new_tokens below min_new_tokens, which
        # would suppress EOS for the whole generation. Restore the invariant.
        if req.sampling_params.min_new_tokens > req.sampling_params.max_new_tokens:
            req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None

        Raises:
            _MultimodalInputProcessingError: The entry rank could not build the
                request's multimodal inputs. The same error is broadcast to all
                ranks before it is raised.
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        result = None
        if self.dp_tp_group.rank_in_group == 0:
            try:
                result = _MultimodalInputBroadcast(
                    inputs=MultimodalInputs.from_processor_output(
                        raw_mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
                )
            except Exception as error:
                result = _MultimodalInputBroadcast(
                    error=(
                        "Multimodal input processing failed on the TP entry rank: "
                        f"{type(error).__name__}: {error}"
                    )
                )

            # Broadcast either the prepared inputs or the request-local error.
            if group_world_size > 1:
                obj_list = [result]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                result = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                result = obj_list[0]
            else:
                result = _MultimodalInputBroadcast(
                    inputs=MultimodalInputs.from_processor_output(
                        raw_mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
                )

        if result.error is not None:
            raise _MultimodalInputProcessingError(result.error)
        return result.inputs

    def _get_multimodal_inputs(self, mm_inputs):
        if isinstance(mm_inputs, MMInputsProcessError):
            raise _MultimodalInputProcessingError(mm_inputs.message)
        if isinstance(mm_inputs, MultimodalInputs):
            return mm_inputs

        if get_mm().enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs)
        return MultimodalInputs.from_processor_output(
            mm_inputs,
            requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
        )

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_namespace_elastic_radix_cache(self, req: Req) -> None:
        if (
            get_exec().moe.elastic_ep_backend is None
            or not self.tree_cache.supports_prefix_sharing()
        ):
            return

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is None:
            return

        namespace = f"elastic_ep_size={ElasticEPStateManager.get_effective_ep_size()}"
        if req.extra_key:
            req.extra_key = f"{req.extra_key}|{namespace}"
        else:
            req.extra_key = namespace

        if self._last_logged_elastic_radix_namespace != namespace:
            self._last_logged_elastic_radix_namespace = namespace
            logger.debug(
                "[Elastic EP][scale] radix cache namespace is now %s",
                namespace,
            )

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                if isinstance(req.finished_reason, FINISH_ABORT):
                    req.release_mm_inputs_on_abort()
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ):
        with ExitStack() as mm_cleanup:
            if isinstance(
                recv_req.mm_inputs, (MultimodalInputs, MultimodalProcessorOutput)
            ):
                for item in recv_req.mm_inputs.mm_items:
                    mm_cleanup.callback(item.release_transport_proxies)
            self._handle_generate_request(
                recv_req, mm_cleanup, mm_input_error=mm_input_error
            )

    def _handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        mm_cleanup: ExitStack,
        *,
        mm_input_error: Optional[str] = None,
    ):
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )
        if recv_req.bootstrap_port is None:
            recv_req.bootstrap_port = get_disagg().disaggregation_bootstrap_port
        # Radix-native sessions use only the top-level session_id.
        radix_native_session = (
            recv_req.session_id is not None and get_memory().enable_session_radix_cache
        )

        if session_id is None or radix_native_session:
            # Normal non-session request, or a radix-native session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            is_beam = BeamCoordinator.request_beam_width(recv_req) > 1
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                return_sampling_mask=recv_req.return_sampling_mask,
                sampling_logprobs_mode=recv_req.sampling_logprobs_mode,
                return_flat_raw_top_logprobs=recv_req.return_flat_raw_top_logprobs,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                kv_hints=recv_req.kv_hints,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.metrics_sink.enable_metrics else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                cache_salt=recv_req.cache_salt,
                http_worker_ipc=recv_req.http_worker_ipc,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
                token_indices_to_pool=recv_req.token_indices_to_pool,
            )
            req.tokenizer = self.tokenizer

            if radix_native_session:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    recv_req.session_id
                )

            if is_beam:
                error_msg = self.beam_coordinator.validate_and_init(req, recv_req)
                if error_msg:
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    if not envs.SGLANG_RUST_SERVER.get():
                        recv_req.time_stats.trace_ctx.abort(
                            abort_info={"reason": error_msg}
                        )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
                disagg_mode=self.disaggregation_mode,
            )
            if get_memory().enable_session_radix_cache:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    session_id
                )
            # TODO: set trace context
            if self.metrics_sink.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if recv_req.pp_prefetch_ticketed is True:
            self.tree_cache.bind_prefetch_ticket(req.rid)
        self._maybe_namespace_elastic_radix_cache(req)

        if mm_input_error is not None:
            req.set_finish_with_abort(
                mm_input_error,
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type="InternalServerError",
            )
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if self.dllm is not None and (error_msg := self.dllm.validate_request(req)):
            abort_req = make_abort_req(
                req,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.BAD_REQUEST,
                    "message": error_msg,
                },
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return

        if self.spec_algorithm.is_dflash_family():
            error_msg = validate_dflash_request(
                req, (not get_schedule().disable_overlap_schedule)
            )
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        if self.spec_algorithm.is_uno():
            error_msg = validate_uno_request(req)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        if req.return_sampling_mask:
            if (
                self.disaggregation_mode != DisaggregationMode.NULL
                and not (
                    self.disagg_prefill or self.disagg_decode
                ).metadata_buffers.enable_sampling_mask
            ):
                self.request_retirer.reject_sampling_mask_request(
                    req,
                    "return_sampling_mask requires "
                    "SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1 on both prefill and "
                    "decode servers when using disaggregated serving.",
                )
                return
            top_k = req.sampling_params.top_k
            sampling_mask_cap = self.server_args.sampling_mask_max_tokens
            if top_k != 1 and not (1 < top_k <= sampling_mask_cap):
                error_msg = (
                    "return_sampling_mask requires top_k=1 for greedy sampling "
                    f"or finite 1 < top_k <= {sampling_mask_cap}; got top_k="
                    f"{top_k}. Lower top_k or increase "
                    "--sampling-mask-max-tokens."
                )
                self.request_retirer.reject_sampling_mask_request(req, error_msg)
                return

        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            error_msg = validate_spec_sampling_mask_request(req, self.spec_algorithm)
            if error_msg is not None:
                self.request_retirer.reject_sampling_mask_request(req, error_msg)
                return

        if req.return_sampling_mask and get_exec().kernel.sampling_backend == "ascend":
            # The ascend backend samples from logits directly and never builds the
            # top-k/top-p support, so it cannot produce a sampling mask.
            error_msg = (
                "return_sampling_mask is not supported with the ascend "
                "sampling backend."
            )
            self.request_retirer.reject_sampling_mask_request(req, error_msg)
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            try:
                image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            except _MultimodalInputProcessingError as error:
                req.set_finish_with_abort(
                    str(error),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type="InternalServerError",
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = duplicate_pad_mm_input_ids(
                    req.origin_input_ids, image_inputs, self.pad_input_ids_func
                )
            req.extend_image_inputs(image_inputs)
            # the request now owns cleanup, including rejection after padding
            mm_cleanup.pop_all()
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                req.set_finish_with_abort(error_msg)
                self._add_request_to_queue(req)
                return

        if get_parallel().pp_rank == 0 and getattr(
            self.tree_cache.cache_controller, "pp_prefetch_command_group", None
        ):
            recv_req.pp_prefetch_ticketed = bool(self._prefetch_kvcache(req))
            self.tree_cache.bind_prefetch_ticket(req.rid, recv_req.pp_prefetch_ticketed)

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req, storage_hit_end: Optional[int] = None):
        if hicache_storage_enabled() or get_memory().enable_lmcache:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            tree_cache = self.tree_cache
            buffer_mode = get_memory().hicache_host_memory_mode == "buffer_only"
            last_host_node = req.last_host_node
            # Buffer mode host-backups nothing, so match_prefix anchors at
            # root; re-anchor on the deepest device node. The anchor is only
            # read for hash/extra-key context here (never locked), so a device
            # node serves. Cache mode keeps the is_backuped gate below: its
            # write-through prefix is contiguous from root, so an unbacked
            # anchor means a guaranteed storage miss.
            if (
                buffer_mode
                and tree_cache.is_root(last_host_node)
                and not tree_cache.is_root(req.last_node)
            ):
                last_host_node = req.last_node

            matched_len = req.prefix_len + req.host_hit_length
            req.storage_prefetch_last_match_len = matched_len

            if (
                tree_cache.is_backuped(last_host_node)
                or tree_cache.is_root(last_host_node)
                or (
                    buffer_mode
                    and tree_cache.get_last_hash_value(last_host_node) is not None
                )
            ):
                match_end = req._compute_max_prefix_len(
                    len(req.full_untruncated_fill_ids)
                )
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]
                prefix_keys = (
                    tree_cache.get_prefix_hash_values(last_host_node)
                    if tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                return tree_cache.prefetch_from_storage(
                    req.cache_request_handle,
                    last_host_node,
                    new_input_tokens,
                    tree_cache.get_last_hash_value(last_host_node),
                    prefix_keys,
                    matched_prefix_tokens=req.full_untruncated_fill_ids[:matched_len],
                    extra_key=req.extra_key,
                    cache_salt=req.cache_salt,
                    storage_hit_end=storage_hit_end,
                )

    def _process_storage_prefetch_retries(self):
        """Issue due L3 attempts in the current waiting-queue order."""
        retries = self.tree_cache.storage_prefetch_retries
        if retries is None:
            return
        memory = get_memory()
        for req, storage_hit_end in retries.pop_ready(
            self.lists.waiting_queue,
            memory.hicache_storage_prefetch_retry_poll_interval,
            memory.hicache_storage_prefetch_retry_max_attempts,
        ):
            self._retry_storage_prefetch(req, storage_hit_end)

    def _retry_storage_prefetch(
        self, req: Req, storage_hit_end: Optional[int] = None
    ) -> None:
        req.storage_prefetch_retry_attempts += 1
        max_attempts = get_memory().hicache_storage_prefetch_retry_max_attempts
        if req.storage_prefetch_retry_attempts >= max_attempts:
            logger.warning(
                "HiCache storage prefetch reissue cap reached req=%s attempts=%d; "
                "the request is admitted without further L3 lookups",
                req.rid,
                req.storage_prefetch_retry_attempts,
            )
        else:
            logger.debug(
                "HiCache storage prefetch re-issue req=%s attempt=%d",
                req.rid,
                req.storage_prefetch_retry_attempts,
            )
        self._prefetch_kvcache(req, storage_hit_end)

    def _prefetch_after_device_hit_loss(self, req: Req) -> bool:
        """Re-query an L3 range newly exposed by queue-time device eviction."""
        previous_match_len = req.storage_prefetch_last_match_len
        buffer_pipeline = self.tree_cache.buffer_pipeline
        if not previous_match_len or (
            buffer_pipeline is not None
            and buffer_pipeline.has_staged(req.cache_request_handle)
        ):
            return False
        current_match_len = req.prefix_len + req.host_hit_length
        if current_match_len >= previous_match_len:
            return False
        if (
            req.storage_prefetch_retry_attempts
            >= get_memory().hicache_storage_prefetch_retry_max_attempts
        ):
            # Past the re-issue cap the shorter live match is admitted as is.
            req.storage_prefetch_last_match_len = current_match_len
            return False
        logger.warning(
            "HiCache device prefix shrank before admission req=%s "
            "lookup_match=%d current_match=%d; reissuing storage lookup",
            req.rid,
            previous_match_len,
            current_match_len,
        )
        self._retry_storage_prefetch(req)
        return True

    def _add_request_to_queue(
        self,
        req: Req,
        is_retracted: bool = False,
        admitted: AbstractSet[Req] = frozenset(),
    ):
        if not self._set_or_validate_priority(req):
            self.request_retirer.release_aborted_request(req)
            return
        if is_retracted:
            req.storage_prefetch_retry_attempts = 0
            req.storage_prefetch_last_match_len = None
            req.staged_prefetch_plan = None
        if self.disagg_prefill is not None:
            self._prefetch_kvcache(req)
            self.disagg_prefill.enqueue(req)
        elif self.disagg_decode is not None:
            self.disagg_decode.enqueue(req, is_retracted=is_retracted)
        else:
            if self._abort_on_queued_limit(req, admitted):
                self.request_retirer.release_aborted_request(req)
                return
            self._prefetch_kvcache(req)
            self.lists.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
            req.arrival_processed_tokens = self.counters.processed_tokens_counter

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if get_schedule().enable_priority_scheduling and req.priority is None:
            if get_schedule().schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not get_schedule().enable_priority_scheduling
            and req.priority is not None
            and get_schedule().abort_on_priority_when_disabled
        ):
            abort_req = make_abort_req(
                req,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(
        self, recv_req: Req, admitted: AbstractSet[Req] = frozenset()
    ) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted.

        The requests in ``admitted`` belong to the step being built: they
        leave the queue at its launch, so they neither count nor are aborted.
        """
        waiting = self.lists.waiting_queue
        if admitted:
            waiting = [r for r in waiting if r not in admitted]
        if (
            self.max_queued_requests is None
            or len(waiting) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if get_schedule().enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if get_schedule().schedule_low_priority_values_first else -1
            key_fn = lambda req: (
                direction * req.priority,
                req.time_stats.wait_queue_entry_time,
            )
            candidate_req = max(waiting, key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                self.request_retirer.release_aborted_request(candidate_req)
                self.lists.waiting_queue.remove(candidate_req)
                self.beam_coordinator.retire_group(candidate_req)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            make_abort_req(
                req_to_abort,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ):
        with ExitStack() as mm_cleanup:
            if isinstance(
                recv_req.mm_inputs, (MultimodalInputs, MultimodalProcessorOutput)
            ):
                for item in recv_req.mm_inputs.mm_items:
                    mm_cleanup.callback(item.release_transport_proxies)
            self._handle_embedding_request(
                recv_req, mm_cleanup, mm_input_error=mm_input_error
            )

    def _handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
        mm_cleanup: ExitStack,
        *,
        mm_input_error: Optional[str] = None,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            token_indices_to_pool=recv_req.token_indices_to_pool,
            decision_layout=recv_req.decision_layout,
        )
        req.tokenizer = self.tokenizer
        self._maybe_namespace_elastic_radix_cache(req)

        if mm_input_error is not None:
            req.set_finish_with_abort(
                mm_input_error,
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type="InternalServerError",
            )
            self._add_request_to_queue(req)
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            try:
                image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            except _MultimodalInputProcessingError as error:
                req.set_finish_with_abort(
                    str(error),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type="InternalServerError",
                )
                self._add_request_to_queue(req)
                return
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = duplicate_pad_mm_input_ids(
                    req.origin_input_ids, image_inputs, self.pad_input_ids_func
                )

            req.extend_image_inputs(image_inputs)
            mm_cleanup.pop_all()
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        if self.disagg_prefill is not None:
            self.disagg_prefill.checkpoint(req)
        else:
            checkpoint_kv_cache(req, self.tree_cache)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later; that step still names the request as its
        ``chunked_req``, so the batch-result processors skip the aborted chunk:
        ``DisaggPrefill.commit_extend`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self.lists._pending_chunked_abort_req
        if req is None:
            return
        if self.lists.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or not req.kv.holds_kv:
                self.lists._pending_chunked_abort_req = None
                return
            # The request moved to another scheduler queue after abort_request
            # deferred it, so retry against its current location.
            self.lists._pending_chunked_abort_req = None
            self.request_registry.abort_request(AbortReq(rid=req.rid))
            return

        reason = req.discard_output_reason or FINISH_ABORT("Aborted")
        prepare_abort(req, reason.message, reason.status_code)
        req.time_stats.trace_ctx.abort(abort_info={"reason": reason.message})
        req.to_finish = None
        if self.disagg_prefill is not None:
            self.disagg_prefill.release_aborted_chunked_req(req)
        self.request_retirer.release_aborted_request(req)
        release_kv_cache(req, self.tree_cache, checkpoint=False)

        self.lists.chunked_req = None
        self.lists._pending_chunked_abort_req = None
        # A client abort leaves the message to the tokenizer.
        finished_reason = req.discard_output_reason and reason.to_json()
        self.ipc_channels.send_to_tokenizer.send_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _admit_hisparse_ready_reqs(self, reqs: List[Req], slot: ScheduleSlot) -> None:
        """Move requests whose HiSparse staging is done into decoding."""
        req_pool_indices = torch.tensor(
            [r.kv.req_pool_idx for r in reqs], dtype=torch.int64, device=self.device
        )
        # The next decode step gathers its input token from the relay.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=self.device
        )
        self.future_map.stash(req_pool_indices, RelayPayload(bonus_tokens=last_tokens))
        self.lists.join_decoding(reqs, slot)

    def _process_hicache_events(
        self, should_retry_storage_prefetch: bool = True
    ) -> None:
        # The HiCache drain is TP-wide consensus; run it before rank-local
        # decisions (_should_defer_prefill) or ranks enter different collectives.
        if (
            get_memory().enable_hierarchical_cache
            or get_memory().enable_flexkv
            or get_memory().enable_unified_cache_external_linker
            or get_memory().enable_lmcache
        ):
            self.tree_cache.check_hicache_events()
            if hicache_storage_enabled() and should_retry_storage_prefetch:
                self._process_storage_prefetch_retries()

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_batch_to_run(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        """Schedule this iteration and build its step for slot from the
        request lists.

        The step is built fresh from the lists and the requests admitted now;
        launching it moves them (see ``join_launched_step``)."""
        self.process_pending_chunked_abort()
        self._process_hicache_events()
        self._start_fpm_clock()

        if self.lists.chunked_req is not None:
            # A parked chunk (add_chunked_req hybrid-SWA early-return) leaves
            # extend_end at prefix_len: it computed no new KV, so nothing to stash.
            if self.lists.chunked_req.extend_end > self.lists.chunked_req.prefix_len:
                self.stash_chunked_request(self.lists.chunked_req)

        stages = self.stages
        if stages.collect_ready is not None:
            ready_reqs = stages.collect_ready()
            if len(ready_reqs) > 0:
                self._admit_hisparse_ready_reqs(ready_reqs, slot)
            # Reset batch_is_full so the scheduler can schedule more prefills.
            slot.batch_is_full = False

        if stages.admit is not None:
            new_batch = stages.admit(slot)
        elif self._should_defer_prefill():
            new_batch = None
        else:
            new_batch = self.get_new_batch_prefill(slot)

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not get_spec().speculative_skip_dp_mlp_sync
            and not envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before building the decode step:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        # Run prefill first if possible
        ret = new_batch if new_batch is not None else self.get_decode_step(slot)
        return self.finalize_step(ret, slot, need_mlp_sync)

    def _admit_dllm_step(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        """The dLLM owner takes the requests it runs; the rest stay waiting."""
        step, self.lists.waiting_queue = self.dllm.schedule(self.lists.waiting_queue)
        return step

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_disagg_prefill_batch_to_run(
        self,
        slot: ScheduleSlot,
        should_retry_storage_prefetch: bool = True,
        process_chunk: bool = True,
    ) -> Optional[ScheduleBatch]:
        self.process_pending_chunked_abort()
        self._process_hicache_events(
            should_retry_storage_prefetch=should_retry_storage_prefetch
        )
        self._start_fpm_clock()
        self.lists.waiting_queue = self.disagg_prefill.resolve_waiting_bootstrap(
            self.lists.waiting_queue
        )
        if process_chunk:
            self.lists.chunked_req, self.lists.waiting_queue = (
                self.disagg_prefill.process_prefill_chunk(
                    self.lists.chunked_req, self.lists.waiting_queue, self.pending
                )
            )

        return self.finalize_step(self.get_new_batch_prefill(slot), slot)

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def process_bootstrapped_queue(self):
        self.lists.waiting_queue.extend(self.disagg_prefill.intake())

    def get_next_pp_disagg_prefill_batch_to_run(
        self, slot: ScheduleSlot
    ) -> Optional[ScheduleBatch]:
        self.lists.chunked_req, self.lists.waiting_queue = (
            self.disagg_prefill.process_prefill_chunk(
                self.lists.chunked_req, self.lists.waiting_queue, self.pending
            )
        )
        self._process_hicache_events()
        self._start_fpm_clock()
        return self.finalize_step(self.get_new_batch_prefill(slot), slot)

    def process_prefill_transfer_queue(self, release_rids: Optional[List[str]]) -> None:
        self.lists.waiting_queue = self.disagg_prefill.after_commit(
            self.lists.waiting_queue, release_rids
        )

    def get_next_disagg_decode_batch_to_run(
        self, slot: ScheduleSlot
    ) -> Optional[ScheduleBatch]:
        """Admit transferred requests into decoding, then build the decode step."""
        self._start_fpm_clock()
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)
        reqs = self.disagg_decode.pop_reqs_to_admit(slot)
        if reqs:
            self.disagg_decode.commit_prefill_handoff(reqs)
            self.lists.join_decoding([r for r in reqs if not r.finished()], slot)

        return self.finalize_step(self.get_decode_step(slot), slot)

    def finalize_step(
        self,
        step: Optional[ScheduleBatch],
        slot: ScheduleSlot,
        need_mlp_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """The tail of every schedule, on the step it built: the DP attention
        sync and the step's forward inputs; nothing writes the step after it
        until its commit. need_mlp_sync overrides whether the sync runs."""
        step = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            step, need_sync=need_mlp_sync
        )
        # The extend view keeps a heterogeneous dp step replayable.
        self.dp_attn_adapter.maybe_set_extend_view(step)
        self._arm_prefill_decode_interval(step)

        # Handle ngram embedding
        step = self.ngram_embedding_manager.prepare_for_forward(
            step, chunked_req=step.chunked_req if step is not None else None
        )

        if step:
            set_schedule_time_batch(step)
            if self.enable_fpm:
                step.fpm_start_time = self._fpm_batch_t0

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_schedule(step, slot)
        return step

    def _start_fpm_clock(self) -> None:
        """Start the forward pass metrics clock of the step being scheduled."""
        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()

    def get_new_batch_prefill(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret = self._get_new_batch_prefill_raw(
            slot,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
        )

        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )

        return ret

    def _get_new_batch_prefill_raw(
        self,
        slot: ScheduleSlot,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
    ) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            slot.batch_is_full = False
        # Under PP only the scheduled slot's requests: a slot's batch size
        # is capped, and preemption must not pick a request in flight.
        running_reqs = slot.filter_decoding(self.lists.decoding)
        if not running_reqs:
            # With nothing decoding no decode step clears the flag, so
            # admission would stay shut.
            slot.batch_is_full = False

        if (
            slot.batch_is_full or len(self.lists.waiting_queue) == 0
        ) and self.lists.chunked_req is None:
            return None

        running_bs = len(running_reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.lists.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.beam_coordinator.get_num_allocatable_reqs(
                    running_bs
                ),
            )
        ):
            return None

        # Ignore the check if self.lists.chunked_req is not None.
        # In the non-PP case, when self.lists.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.beam_coordinator.get_num_allocatable_reqs(running_bs) <= 0
            and self.lists.chunked_req is None
            and not self.enable_priority_preemption
        ):
            slot.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(
            self.lists.waiting_queue,
            running_reqs,
            processed_tokens=self.counters.processed_tokens_counter,
        )

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the decoding batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.lists.chunked_req is not None and self.dynamic_chunk_sizer is not None:
            history_len = self.lists.chunked_req.prefix_len
            dynamic_size = self.dynamic_chunk_sizer.predict(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        # Get BLOCK_M from the backend for tile-budget admission logic
        attn_backend = self.tp_worker.model_runner.attn_backend
        if hasattr(attn_backend, "extend_attention_block_m"):
            prefill_tile_block_m = attn_backend.extend_attention_block_m
        else:
            prefill_tile_block_m = 64  # Fallback for non-Triton backends

        adder = PrefillAdder(
            get_schedule().page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_reqs,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            get_schedule().priority_scheduling_preemption_threshold,
            max_prefill_bs=int(self.max_prefill_bs),
            max_running_requests=self.max_running_requests,
            prefill_max_requests=get_schedule().prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            waiting_queue_len=len(self.lists.waiting_queue),
            prefill_tile_block_m=prefill_tile_block_m,
            req_to_token_pool=self.req_to_token_pool,
            hisparse_coordinator=self.hisparse_coordinator,
        )

        # The chunked request after this step; the launch moves it into the
        # chunked holder (see join_launched_step).
        chunked_req = self.lists.chunked_req
        if chunked_req is not None:
            chunked_req.init_next_round_input()
            adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
                chunked_req,
                self.lists.waiting_queue,
                adder.rem_chunk_tokens or 0,
                get_schedule().page_size,
            )
            chunked_req = adder.add_chunked_req(chunked_req)

        if get_lora().enable_lora:
            # The adapters of the decoding requests and of the steps in flight
            # (a request finished at the last commit may still run in one).
            running_loras = {req.lora_id for req in self.lists.decoding}
            running_loras.update(self.pending.lora_ids())
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.lists.waiting_queue,
                    self.lists.decoding,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.lists.waiting_queue))
        buffer_pipeline = self.tree_cache.buffer_pipeline
        # Get requests from the waiting queue to a new prefill batch
        for req in self.lists.waiting_queue:
            if get_lora().enable_lora and not self.can_schedule_lora_req(
                req, running_loras
            ):
                continue

            # A forward batch runs one pooling mode, so setwise readout requests
            # (token_indices_to_pool) cannot share a batch with last-token ones.
            # Admit only the first request's mode; the other stays queued.
            if adder.can_run_list and (
                (req.token_indices_to_pool is not None)
                != (adder.can_run_list[0].token_indices_to_pool is not None)
            ):
                continue

            # Preemption below shrinks the requests still decoding.
            running_bs = len(running_reqs) - len(adder.preempt_list)
            candidate_beam_width = (
                req.beam_group.beam_width if req.beam_group is not None else None
            )
            if len(
                adder.can_run_list
            ) >= self.beam_coordinator.get_num_allocatable_reqs(
                running_bs, candidate_beam_width
            ):
                slot.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    slot.batch_is_full = True

            if slot.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            if hicache_storage_enabled() or get_memory().enable_lmcache:
                prefetch_done = self.tree_cache.check_prefetch_progress(
                    req.cache_request_handle
                )
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the L3-loaded span. Unified cache exposes its absolute
                # start so cache-mode L2/L3 attribution survives L3-tail eviction.
                loaded_tokens, loaded_start = self.tree_cache.pop_prefetch_loaded_span(
                    req.cache_request_handle
                )
                if loaded_tokens > 0:
                    req.storage_hit_length = loaded_tokens
                    req.storage_hit_start = loaded_start
                    # Cache-mode host memory is a resident L2 tier. Buffer mode
                    # marks the staged span below once it is surfaced.
                    req.host_hit_is_storage = False

            req.init_next_round_input(self.tree_cache)
            if hicache_storage_enabled() and (
                self._prefetch_after_device_hit_loss(req)
            ):
                continue
            if (
                hicache_storage_enabled()
                and buffer_pipeline is not None
                and not buffer_pipeline.prepare_staged_prefetch(req)
            ):
                continue
            res = adder.add_one_req(
                req,
                has_chunked_req=(chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if get_lora().enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    # Set batch_is_full after making sure there are requests that can be served
                    slot.batch_is_full = len(adder.can_run_list) > 0 or (
                        len(running_reqs) > len(adder.preempt_list)
                    )
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if not added:
                    # init_next_round_input() may stage deferred Mamba COW/clear
                    # metadata before add_one_req() rejects the request.
                    req.kv.mamba_cow_src_index = None
                    req.kv.mamba_needs_clear = False
                    if req.kv.holds_mamba and not getattr(req, "session", None):
                        self.tree_cache.req_to_token_pool.mamba_allocator.free(
                            req.kv.mamba_pool_idx.unsqueeze(-1)
                        )
                        req.kv.mamba_pool_idx = None
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        # Preemption already released these requests; requeue them even when
        # the request they made room for was not admitted.
        if adder.preempt_list:
            preempted = set(adder.preempt_list)
            self.lists.decoding = [r for r in self.lists.decoding if r not in preempted]
            running_reqs = slot.filter_decoding(self.lists.decoding)
            admitted = set(adder.can_run_list)
            for req in adder.preempt_list:
                self._add_request_to_queue(req, admitted=admitted)

        # The admitted requests stay in the waiting queue until the launch.
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        retries = self.tree_cache.storage_prefetch_retries
        if hicache_storage_enabled() and retries is not None:
            for req in can_run_list:
                retries.cancel(req.rid)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert chunked_req is None
            chunked_req = adder.new_chunked_req

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = self.step_builder.build_extend_step(
            can_run_list, chunked_req=chunked_req
        )

        # Record prefill stats for logging after forward. The tokens pending
        # prefill once this step launches: the waiting requests it does not
        # admit, and the chunked request's tokens past this chunk.
        admitted = set(can_run_list)
        num_pending_tokens = sum(
            req.seqlen for req in self.lists.waiting_queue if req not in admitted
        )
        if chunked_req is not None:
            num_pending_tokens += (
                chunked_req.seqlen - chunked_req.prefix_len - chunked_req.extend_len
            )
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_reqs,
            get_schedule().enable_priority_scheduling,
            num_pending_tokens=num_pending_tokens,
        )

        # Mixed-style chunked prefill: the decoding requests ride this step
        # as one-token extend rows after the prefill rows.
        if (
            self.is_mixed_chunk
            and running_reqs
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            and not new_batch.return_logprob
            and not any(r.return_logprob for r in running_reqs)
            # The decode rows carry no input_embeds; shapes would mismatch.
            and new_batch.input_embeds is None
            # Beam member rows are not supported inside a mixed extend batch.
            and all(r.beam_group is None for r in running_reqs)
        ):
            decode_reqs = self._select_decode_reqs(slot)
            # The prefill admitted above may leave no KV room for the decode
            # rows. They then sit out this step and decode on the next decode
            # step, which fits them; a mixed step does not retract them.
            if decode_reqs and check_decode_mem(
                decode_reqs,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                spec_algorithm=self.spec_algorithm,
            ):
                new_batch.mix_decode_rows(
                    self.step_builder.new_decode_step(decode_reqs)
                )

        new_batch.sampling_info = self.step_builder.build_sampling_info(new_batch.reqs)
        return new_batch

    def can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if get_lora().enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )

    def _select_decode_reqs(self, slot: ScheduleSlot) -> List[Req]:
        """The decoding requests one decode step (or a mixed step's decode
        rows) runs: not prefill-only, and not already reaching max_new_tokens
        in the steps still in flight. Skipping one reopens admission: it
        leaves decoding at its commit."""
        candidates = [
            req
            for req in slot.filter_decoding(self.lists.decoding)
            if not req.is_prefill_only
        ]
        num_pending = len(self.pending)
        if not self.flags.enable_skip_finishing_decode or num_pending == 0:
            return candidates
        # Each pending step commits at most one output to a request, so only a
        # request within num_pending outputs of max_new_tokens can finish in
        # flight; the rest skip the exact count.
        reqs = [
            req
            for req in candidates
            if len(req.output_ids) + num_pending < req.sampling_params.max_new_tokens
            or not self._finishes_inflight(req)
        ]
        if len(reqs) < len(candidates):
            slot.batch_is_full = False
        return reqs

    def _finishes_inflight(self, req: Req) -> bool:
        """Whether the uncommitted outputs of req reach max_new_tokens."""
        if req.beam_group is not None:
            return False
        num_inflight = self.pending.num_inflight_outputs(req)
        return (
            num_inflight > 0
            and len(req.output_ids) + num_inflight >= req.sampling_params.max_new_tokens
        )

    def get_decode_step(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        """Build the decode step of the decoding requests, retracting some when
        the step does not fit the KV pool; None skips decode this step."""
        reqs = self._select_decode_reqs(slot)
        if not reqs:
            return None

        kv_full_retract_flag = not check_decode_mem(
            reqs,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            spec_algorithm=self.spec_algorithm,
        )
        if kv_full_retract_flag and len(reqs) < sum(
            not req.is_prefill_only for req in slot.filter_decoding(self.lists.decoding)
        ):
            # The pending result frees the finishing requests' KV at the end of
            # this step; decode the rest next step instead of retracting them.
            return None

        # Check if decode out of memory
        if kv_full_retract_flag or (
            TEST_RETRACT and self.counters.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            reqs = self._retract_decode_reqs(reqs, kv_full_retract_flag, slot)
        else:
            self.new_token_ratio_tracker.decay_step()

        if not reqs:
            return None
        return self.step_builder.build_decode_step(reqs)

    def _retract_decode_reqs(
        self, reqs: List[Req], kv_full_retract_flag: bool, slot: ScheduleSlot
    ) -> List[Req]:
        """Move decoding requests back to the waiting queue until the rest fit
        one decode step; return the requests that stay."""
        if self.disagg_decode is not None and self.decode_offload_manager is not None:
            # Pending offload copies may still read the device KV that retraction frees.
            self.disagg_decode.drain_offloads_before_retraction()
        old_available_tokens = self.token_to_kv_pool_allocator.available_size()
        old_ratio = self.new_token_ratio_tracker.current
        mamba_allocator = getattr(
            self.tree_cache.req_to_token_pool, "mamba_allocator", None
        )
        old_mamba_available = (
            mamba_allocator.available_size() if mamba_allocator is not None else None
        )
        kept_reqs, retracted_reqs, reqs_to_abort = retract_decode(
            reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            spec_algorithm=self.spec_algorithm,
        )
        new_token_ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(
            kept_reqs
        )
        # The kept requests decode in the retraction order from now on.
        kept = set(kept_reqs)
        left = set(retracted_reqs) | set(reqs_to_abort)
        self.lists.decoding = kept_reqs + [
            r for r in self.lists.decoding if r not in kept and r not in left
        ]
        if left:
            slot.batch_is_full = False
        new_available_tokens = self.token_to_kv_pool_allocator.available_size()
        new_token_gained = new_available_tokens - old_available_tokens
        mamba_num_gained = (
            mamba_allocator.available_size() - old_mamba_available
            if mamba_allocator is not None
            else None
        )

        self.metrics_sink.num_retracted_reqs = len(retracted_reqs)
        if self.metrics_sink.enable_metrics and len(retracted_reqs) > 0:
            self.metrics_sink.metrics_collector.increment_retracted_reqs(
                num_retracted_reqs=len(retracted_reqs),
                num_retracted_input_tokens=sum(
                    len(r.origin_input_ids) for r in retracted_reqs
                ),
                num_retracted_output_tokens=sum(
                    len(r.output_ids) for r in retracted_reqs
                ),
            )
        self.new_token_ratio_tracker.current = new_token_ratio
        for req in reqs_to_abort:
            abort_reason: FINISH_ABORT = req.to_finish
            self.ipc_channels.send_to_tokenizer.send_output(
                make_abort_req(req, finished_reason=abort_reason.to_json()),
                req,
            )
        for req in reqs_to_abort:
            # Member rows were freed inside retract_decode; the group only
            # has to leave the live set.
            self.beam_coordinator.retire_group(req)

        msg_prefix = (
            "KV cache pool is full. Retract requests. "
            if kv_full_retract_flag
            else "Testing retraction. "
        )
        msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
        if mamba_num_gained is not None:
            msg_details += f", #mamba_num_gained: {mamba_num_gained}"
        if kv_full_retract_flag:
            msg_details += (
                f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
            )
        logger.warning(msg_prefix + msg_details)

        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
        return kept_reqs

    @scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.metrics_sink.record_scheduler_active(time.monotonic_ns())
        self.counters.forward_ct += 1

        # Accumulate the prefill-token counter used by the HRRN scheduling policy. Decode / prebuilt batches contribute 0.
        if batch.extend_num_tokens:
            self.counters.processed_tokens_counter += batch.extend_num_tokens

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.flags.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.flags.forward_sleep_time}s")
            time.sleep(self.flags.forward_sleep_time)

        if (before_forward := self.stages.before_forward) is not None:
            before_forward(batch)

        with self.launch_stream_ctx:
            if not get_schedule().disable_overlap_schedule:
                self.forward_stream.wait_stream(self.schedule_stream)
            # The worker resolves the step's inputs on a view of it and writes
            # its device commit; the launch then starts the result's copy to
            # host.
            ret = self.forward_step(
                batch, pp_proxy_tensors=pp_proxy_tensors, pending_grammar=self.pending
            )
            self.finish_launch(batch, ret)

        if get_memory().enable_unified_cache_external_linker:
            failed = self.tree_cache.finish_external_linker_loads(batch.reqs)
            if batch.chunked_req in failed:
                # Result processing finishes the others; a chunked one is
                # released like a client abort so its next chunk never runs.
                self.lists._pending_chunked_abort_req = batch.chunked_req

        self._maybe_report_active_ranks()

        return ret

    def init_forward_launch(self) -> None:
        """Pick how a step launches, once: what runs its forward and how its
        result starts toward the host differ by mode, not by step."""
        self.launch_stream_ctx = (
            self.forward_stream_ctx
            if (not get_schedule().disable_overlap_schedule)
            else nullcontext()
        )
        # The beam coordinator relays tokens of non-speculative steps only.
        self.relays_beam_tokens = self.spec_algorithm.is_none()
        if not self.is_generation:
            self.forward_step = self._forward_embedding
            self.finish_launch = self._finish_embedding_launch
        elif not get_schedule().disable_overlap_schedule:
            # FIXME: pp is not compatible with overlap
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_overlap_launch
        elif get_parallel().pp_size > 1 and not self.spec_algorithm.is_none():
            self.forward_step = self._pp_spec_forward
            # The ring brings the result back from the last stage.
            self.finish_launch = lambda step, result: None
        elif self.spec_algorithm.is_none():
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_sync_launch
        else:
            # Non-overlap: drive the V2 worker synchronously; it keys its next
            # draft input into FutureMap.
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_sync_spec_launch

    def _pp_spec_forward(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[PendingSteps] = None,
    ) -> GenerationBatchResult:
        """A speculative step under pipeline parallelism: the ring verifies
        the trees its relay keeps in the rows' slots."""
        return self.pp_ring.spec_forward(
            step, pp_proxy_tensors=pp_proxy_tensors, pending_grammar=pending_grammar
        )

    def _forward_embedding(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[PendingSteps] = None,
    ) -> EmbeddingBatchResult:
        """An embedding or reward step: no relay, no grammar."""
        # The forward fills a view of the step; the step stays as built.
        forward_batch = worker_forward_batch(step)
        resolve_forward_inputs(forward_batch, self.future_map)
        pooler_output, can_run_cuda_graph = self.tp_worker.forward_batch_embedding(
            forward_batch
        )
        return EmbeddingBatchResult(
            embeddings=pooler_output.embeddings,
            pooled_hidden_states=pooler_output.pooled_hidden_states,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _finish_embedding_launch(
        self, batch: ScheduleBatch, result: EmbeddingBatchResult
    ) -> None:
        if not get_schedule().disable_overlap_schedule:
            result.copy_to_cpu()

    def _finish_overlap_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        """Start the result's copy to host on the copy stream; a delayed sample
        starts it once it runs (launch_batch_sample_if_needed)."""
        if get_memory().enable_unified_memory:
            # Record a `forward_done` event after the forward (before
            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
            # it. Only the unified pool's allocator exposes these hooks.
            allocator = self.token_to_kv_pool_allocator
            forward_done = self.device_module.Event()
            forward_done.record(stream=self.forward_stream)
            allocator.set_latest_forward_done_event(forward_done)
            # Write-set classification: hand the allocator this
            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
            allocator.set_inflight_forward(
                forward_done,
                batch.out_cache_loc,
            )
        # FIXME(lsyin): maybe move this to forward_batch_generation
        batch_result.copy_done = self.device_module.Event()
        if batch_result.delay_sample_func is None:
            if self.relays_beam_tokens:
                self._relay_beam_tokens(batch, batch_result)
            if _is_hip:
                # Cross-stream sync costs more than the tiny D2H it
                # overlaps.
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                # Result D2H on copy_stream overlaps the next forward
                # instead of serializing on forward_stream; it's a leaf
                # gated by copy_done, so nothing on forward_stream waits.
                self.copy_stream.wait_stream(self.forward_stream)
                with self.copy_stream_ctx:
                    batch_result.copy_to_cpu(
                        return_logprob=batch.return_logprob,
                        return_hidden_states=batch.return_hidden_states,
                    )

    def _finish_sync_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        self._relay_beam_tokens(batch, batch_result)
        self.update_cache_from_scheduler(batch, batch_result)
        self._copy_auxiliary_output_to_cpu(batch, batch_result)

    def _finish_sync_spec_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        self.update_cache_from_scheduler(batch, batch_result)
        batch_result.copy_done = self.device_module.Event()
        if batch_result.has_sampled_token_ids:
            batch_result.copy_to_cpu(
                return_logprob=batch.return_logprob,
                return_hidden_states=batch.return_hidden_states,
            )

    def _maybe_report_active_ranks(self) -> None:
        if not (
            get_parallel().attn_dp_enabled
            and get_exec().moe.elastic_ep_backend is not None
        ):
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is not None and inst.active_ranks_cpu is not None:
            self.ipc_channels.send_to_tokenizer.send_output(
                ActiveRanksOutput(
                    status=[bool(x) for x in inst.active_ranks_cpu.tolist()]
                )
            )
        else:
            logger.debug("[Elastic EP] active rank state is unavailable")
            return

        model_runner = self.tp_worker.model_runner
        pending = model_runner._pending_elastic_scale_update
        if pending is not None:
            self.ipc_channels.send_to_tokenizer.send_output(pending)
            model_runner._pending_elastic_scale_update = None

    def _relay_beam_tokens(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        """Overwrite the beam rows' relayed tokens with the joint-selected
        ones, after the worker stashed the step's sampled tokens."""
        if not batch_result.has_sampled_token_ids:
            return
        self.beam_coordinator.maybe_select_and_relay(
            batch, batch_result, chunked_req=batch.chunked_req
        )

    def _copy_auxiliary_output_to_cpu(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        logits_output = result.logits_output
        if (
            logits_output is None
            or logits_output.auxiliary_device_output is None
            or result.auxiliary_host_output is not None
        ):
            return
        # PP transports the device output to the first rank before copying it.
        if get_parallel().pp_size > 1:
            return
        if result.copy_done is not None:
            raise RuntimeError(
                "generation result has an uncopied auxiliary output after its "
                "device-to-host copy was scheduled"
            )
        result.copy_done = self.device_module.Event()
        result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult, cur_batch: ScheduleBatch
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            # Delay-sample is non-spec only; the worker's sample stashes the
            # sampled bonus tokens.
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self._relay_beam_tokens(cur_batch, batch_result)

        # Run device-to-host copy on a separate stream to avoid blocking the
        # forward stream. The copy waits for the sampled result and can overlap
        # with subsequent forward computation.
        self.copy_stream.wait_stream(self.forward_stream)
        with self.copy_stream_ctx:
            batch_result.copy_to_cpu(
                return_logprob=cur_batch.return_logprob,
                return_hidden_states=cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # the pending result until its commit in the next iteration,
        # causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_BATCH_RESULT)
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        slot: ScheduleSlot,
    ):
        # Flush async trace ops here: in overlap mode this CPU work runs while
        # the next batch's GPU forward is in flight, giving free overlap.
        flush_trace_batch(batch.reqs)
        snapshot = self.publish_load_snapshot(force=batch.forward_mode.is_extend())
        # Router-facing gauge on the dedicated PUB socket, reusing the
        # snapshot above rather than walking the queues again.
        self.load_publisher.publish_load_stat(
            self.load_inquirer.get_loads,
            force=batch.forward_mode.is_extend(),
            snapshot=snapshot,
        )

        commit_step_counters(batch)
        if batch.forward_mode.is_decode():
            num_generated_tokens = (
                self.batch_result_processor.process_batch_result_decode(batch, result)
            )
            self.metrics_reporter.report_decode_stats(
                result.can_run_cuda_graph,
                batch=batch,
                num_generated_tokens=num_generated_tokens,
            )
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.free_remapped_decode_slots()
        elif batch.forward_mode.is_extend():
            if self.dllm is not None:
                self.dllm.commit_extend(batch, result)
            elif self.disagg_prefill is not None:
                self.lists.waiting_queue = self.disagg_prefill.commit_extend(
                    batch, result, self.lists.waiting_queue, self.lists.chunked_req
                )
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
            # None on decode->extend converted batches; they are decode work and
            # have no prefill stats to report.
            if batch.prefill_stats is not None:
                self.metrics_reporter.report_prefill_stats(
                    batch=batch,
                    prefill_stats=batch.prefill_stats,
                    can_run_cuda_graph=result.can_run_cuda_graph,
                    dp_cooperation_info=batch.dp_cooperation_info,
                )
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)
        self.leave_finished_reqs(slot)

        # Submit this batch's queued host backups before the next scheduler step.
        self.tree_cache.flush_pending_backups()

        self.metrics_sink.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.output_streamer.maybe_reply_health_check()
        self.metrics_sink.update_device_timer()

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_process_result(batch)

    def _record_step_counters(self, record: PendingStep) -> None:
        batch = record.step
        mode = batch.forward_mode
        is_prefill = mode.is_extend_without_speculative()
        if not (is_prefill or mode.is_decode() or mode.is_target_verify()):
            return
        if all(is_health_check_generate_req(req) for req in batch.reqs):
            return
        start_iter, start_ts = batch.forward_iter, record.launch_ts
        prev = self._prev_step
        self._prev_step = (batch.forward_iter, record.launch_ts, is_prefill)
        # An idle pass keeps forward_iter contiguous (forward_ct advances in run_batch).
        contiguous = False
        if prev is not None:
            prev_iter, prev_launch_ts, prev_is_prefill = prev
            contiguous = (
                not record.after_idle_gap
                and prev_iter + 1 == start_iter
                and prev_is_prefill == is_prefill
            )
        if is_prefill:
            # Completion boundaries include the burst tail and scheduling overhead
            # without double-counting overlapping launch-to-result spans.
            end_ts = time.monotonic()
            prev_end_ts = self._prev_prefill_end_ts
            self._prev_prefill_end_ts = end_ts
            if prev_end_ts is not None:
                # A new burst or mode can still overlap an earlier prefill.
                start_ts = prev_end_ts if contiguous else max(start_ts, prev_end_ts)
            step_us = int((end_ts - start_ts) * 1e6)
        else:
            # Decode regression uses launch cadence and skips boundaries.
            if not contiguous:
                return
            step_us = int((record.launch_ts - prev_launch_ts) * 1e6)
        if not 0 < step_us < STEP_MAX_US:
            return
        if is_prefill:
            self.total_prefill_busy_us += step_us
            self.total_prefill_uncached_tokens += batch.extend_num_tokens
        else:
            batch_size = len(batch.reqs)
            _accumulate_decode_moment(
                self.decode_moment_totals,
                batch_size,
                step_us,
                record.result.get_num_generated_tokens(batch_size),
            )

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if get_memory().enable_hierarchical_cache or get_memory().enable_lmcache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache or LMCache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache or LMCache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    @scheduler_stage_method(SCHEDULER_STAGE_IDLE)
    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        self.dp_attn_adapter.drop_sync_wait_carry()
        # Flush any health-check signal deferred while the engine was busy.
        self.output_streamer.maybe_reply_health_check()

        # Publish before the fully-idle gate: a no-batch-but-not-idle stall
        # (queues parked under KV pressure / disagg transfer) has no
        # process_batch_result to publish the growing gauge, and gating here
        # froze /get_loads, DP balancing, and the LoadStat for the stall. This
        # path is polled repeatedly, so a wall-clock floor bounds the
        # O(queue) get_loads for both sinks; the fully-idle publish runs
        # post-flush below.
        fully_idle = self.request_registry.is_fully_idle()
        if not fully_idle:
            self.metrics_sink.record_scheduler_active(time.monotonic_ns())
            now = time.monotonic()
            if now - self._last_stall_publish_ts >= LOAD_STALL_REFRESH_S:
                self._last_stall_publish_ts = now
                snapshot = self.publish_load_snapshot(force=True)
                self.load_publisher.publish_load_stat(
                    self.load_inquirer.get_loads, force=True, snapshot=snapshot
                )
            if (
                hicache_storage_enabled()
                or self.disaggregation_mode != DisaggregationMode.NULL
                or get_memory().enable_lmcache
            ):
                # Storage and transfer workers need the GIL between I/O calls.
                # Singleton PD polls no longer yield through a collective.
                time.sleep(0)
            return
        self.metrics_sink.record_scheduler_idle()

        if get_memory().enable_unified_memory:
            try:
                self.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp). Requests
        # holding KV until an ack lands keep the scheduler from fully idle.
        with self.scheduler_stage_metrics.record(SCHEDULER_STAGE_SANITY_CHECK_CACHE):
            if not get_memory().enable_hisparse:
                has_leak, messages = self.invariant_checker._check_all_pools(
                    self.pool_stats_observer.get_pool_stats(),
                )
                if has_leak:
                    self.invariant_checker._report_leak("pool", "\n".join(messages))
                self.invariant_checker._check_req_pool()
                # Byte-conservation diagnostic (allocator-owned; static pools
                # return [] — the token identity above can't see byte leaks).
                byte_violations = (
                    self.token_to_kv_pool_allocator.verify_byte_accounting()
                )
                if byte_violations:
                    self.invariant_checker._report_leak(
                        "pool-bytes", "\n".join(byte_violations)
                    )

            # tree cache sanity check
            self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # Nothing decodes: release the requests the last decode step's inputs
        # still reference.
        self.step_builder.clear_decode_rows()

        # Fully-idle publish, post-flush so the gauge reflects compacted KV.
        # Forced (immediate) so the busy->idle transition is never delayed.
        snapshot = self.publish_load_snapshot(force=True)
        self.load_publisher.publish_load_stat(
            self.load_inquirer.get_loads, force=True, snapshot=snapshot
        )

        # sleep until next event
        self.maybe_sleep_on_idle()
        self.metrics_sink.record_scheduler_idle()

    def _record_scheduler_state_for_paused_engine(self) -> None:
        if self.request_registry.is_fully_idle():
            self.metrics_sink.record_scheduler_idle()
        else:
            self.metrics_sink.record_scheduler_active(time.monotonic_ns())

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not get_memory().enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.request_registry.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.request_registry.queued_reqs())} "
                    f"#running-req={len(self.lists.decoding)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=get_serving().served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            hicache_fields = {
                "hicache_storage_backend": recv_req.hicache_storage_backend
            }
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = recv_req.hicache_write_policy
            get_context().override("scheduler.attach_hicache", **hicache_fields)
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not get_memory().enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.request_registry.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.request_registry.queued_reqs())} "
                    f"#running-req={len(self.lists.decoding)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not hicache_storage_enabled()):
            # Treat "already disabled / nothing to do" as success for idempotence.
            get_context().override(
                "scheduler.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.request_registry.is_fully_idle(
            ignore_waiting=self.flags._engine_paused
        ):
            self.counters.cur_batch_for_debug = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.req_to_token_pool.reset_aux_cache_allocator()
            self.grammar_manager.clear()
            self.metrics_sink.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_sink.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.request_registry.queued_reqs())}, "
                f"#running-req: {len(self.lists.decoding)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        return self.internal_state.get_internal_state(recv_req)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        return self.internal_state.set_internal_state(recv_req)

    def save_remote_model(self, **kwargs):
        self.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier(group=self.tp_group.cpu_group)
        return RpcReqOutput(success=success, message="" if not exec else str(exec))

    def handle_update_weight_version(
        self, recv_req: UpdateWeightVersionReqInput
    ) -> UpdateWeightVersionReqOutput:
        self.record_weight_version_change(new_version=recv_req.new_version)
        return UpdateWeightVersionReqOutput()

    def record_weight_version_change(self, new_version: Optional[str]) -> None:
        if new_version is None or new_version == get_serving().weight_version:
            return

        old_version = get_serving().weight_version
        get_context().override("scheduler.weight_version", weight_version=new_version)

        live_reqs = [
            req
            for entry in self.request_registry.request_lists().values()
            for req in entry.reqs
        ]
        num_recorded = record_weight_version_events(live_reqs, old_version=old_version)
        logger.info(
            f"Weight version changed. {old_version=} {new_version=} {num_recorded=}"
        )

    # The abort RPC handler, and the callbacks of the components built before
    # the registry (a PD role switch rebuilds it).
    def request_lists(self) -> Dict[str, RequestList]:
        return self.request_registry.request_lists()

    def is_fully_idle(
        self, for_health_check=False, ignore_waiting=False, ignore_releasing=False
    ) -> bool:
        return self.request_registry.is_fully_idle(
            for_health_check=for_health_check,
            ignore_waiting=ignore_waiting,
            ignore_releasing=ignore_releasing,
        )

    def abort_request(self, recv_req: AbortReq):
        self.request_registry.abort_request(recv_req)

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        assert recv_req.mode in ("in_place", "retract")
        self.flags._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # The request lists, chunked_req and the pending steps are left
            # untouched. On resume, the event loop commits the pending steps
            # and schedules from the lists through the standard code paths.
            return

        # Under PP the ring commits its pending steps once their outputs come
        # back over the relay; there is nothing to commit here.
        if self.pending and get_parallel().pp_size == 1:
            self.commit_pending_steps()

        lists = self.request_registry.request_lists()
        retract_reqs = [
            req
            for entry in lists.values()
            if entry.retract_on_pause
            for req in entry.reqs
            if not req.finished()
        ]

        self.counters.cur_batch_for_debug = None

        if self.disagg_decode is not None and self.decode_offload_manager is not None:
            # Pending offload copies may still read the device KV that retraction frees,
            # and the paused decode loop never drains them.
            self.disagg_decode.drain_offloads_before_retraction()
        if retract_reqs:
            # Decode-side retract always rebootstraps (recomputes the KV from
            # the prefill), so skip the device->host KV offload that release_req
            # would otherwise do; the offloaded copy would be immediately
            # discarded. Non-decode modes ignore offload_kv (they never offload).
            retract_all(
                reqs=retract_reqs,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                hisparse_coordinator=self.hisparse_coordinator,
                offload_kv=False,
            )
        self.lists.decoding = []
        if self.disagg_decode is not None:
            self.disagg_decode.hold_for_rebootstrap(retract_reqs)
        else:
            for req in retract_reqs:
                self._add_request_to_queue(req)
        # Under PP the ring slots keep their flags: as before, a slot whose
        # decoding requests all left reopens at its next schedule.
        self.flat_slot.batch_is_full = False
        # In disagg-PREFILL, keep a live mid-chunk chunked_req rather than retract it:
        # freeing its KV under a live disagg KV-sender crashes pop_bootstrapped or
        # sends freed/reused KV to decode. Kept, it resumes prefill after the pause.
        # TODO(disagg-prefill-retract): tear the sender down (abort + release metadata
        # buffer + reset pending_bootstrap) before freeing KV, then retract for real.
        # Until then a weight-update pause leaves stale-weight prefix KV (off-policy).
        if lists["chunked"].retract_on_pause:
            self.lists.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_sink.last_gen_throughput = 0.0
        if self.metrics_sink.current_scheduler_metrics_enabled:
            self.metrics_sink.metrics_collector.last_log_time = 0.0
            self.metrics_reporter._maybe_log_idle_metrics()
        self.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        # Enqueue any rebootstrap requests that were staged during a
        # retract-mode pause. Deferring until resume keeps the preallocation
        # queue empty during the pause window (so an intervening weight update
        # can flush the cache) and recomputes the prefix KV under the updated
        # weights.
        if self.disagg_decode is not None:
            self.disagg_decode.prealloc_queue.enqueue_held_rebootstrap()
        self.flags._engine_paused = False

    def handle_scale_elastic_ep(
        self, recv_req: ScaleElasticEPReqInput
    ) -> ScaleElasticEPReqOutput:
        """Begin a pending elastic EP scale-up request."""
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        old_ep_size = ElasticEPStateManager.get_effective_ep_size()
        new_ep_size = recv_req.new_ep_size
        max_ep_size = get_parallel().max_world_size

        logger.debug(
            "[Elastic EP][scale] request received: new_ep_size=%d "
            "old_ep_size=%d max_ep_size=%d",
            new_ep_size,
            old_ep_size,
            max_ep_size,
        )

        if new_ep_size <= old_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) must be greater than current "
                    f"effective_ep_size ({old_ep_size})."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if new_ep_size > max_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) exceeds --max-ep-size "
                    f"({max_ep_size}). Restart with a larger --max-ep-size."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if ElasticEPStateManager.is_scaling():
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "A previous scale operation has not completed yet. Wait until "
                    "all pending ranks have joined before issuing another scale."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )

        if not ElasticEPStateManager.request_scale(new_ep_size):
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "Failed to queue elastic EP scale: no elastic state or "
                    "scale already pending."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )
        if (eplb_manager := self.tp_worker.model_runner.eplb_manager) is not None:
            eplb_manager.disable_rebalance("elastic EP scale-up is pending")
        logger.debug(
            "[Elastic EP][scale] scale requested: target_ep_size=%d; "
            "waiting for a joining cohort",
            new_ep_size,
        )

        return ScaleElasticEPReqOutput(
            success=True,
            message=f"Scaling initiated from {old_ep_size} to {new_ep_size}",
            old_ep_size=old_ep_size,
            new_ep_size=new_ep_size,
            pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
            scale_phase=ElasticEPStateManager.get_scale_phase(),
        )

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        if result.success:
            self.kv_event_lora_names.register(
                lora_id=recv_req.lora_id, lora_name=recv_req.lora_name
            )
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        if result.success:
            self.kv_event_lora_names.register(
                lora_id=recv_req.lora_id, lora_name=recv_req.lora_name
            )
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(
            success=success, message=message
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success=success, message=message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.flags.forward_sleep_time = t
        return SlowDownReqOutput()

    def handle_pd_role_switch(self, recv_req: PdRoleSwitchReqInput):
        return role_switch.handle_pd_role_switch(self, recv_req)

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.session_controller.open(recv_req)
        if output.success and get_memory().enable_session_radix_cache:
            self.tree_cache.open_radix_session(recv_req.session_id)
        if (
            get_parallel().pp_rank == 0
            and get_parallel().tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        ):
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        if get_memory().enable_session_radix_cache:
            self.tree_cache.release_radix_session(recv_req.session_id)
        if (
            recv_req.session_id in self.session_controller
            or not get_memory().enable_session_radix_cache
        ):
            self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        self.flags.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


def reject_modes_without_pipeline_support():
    """Fail at startup in the modes the scheduling pipeline does not run yet."""
    if use_mlx():
        raise ValueError(
            "MLX is not supported: its scheduler is being rewritten for the "
            "scheduling pipeline."
        )
    unsupported = [
        name
        for name, enabled in (
            ("PD multiplexing (--enable-pdmux)", get_disagg().enable_pdmux),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(
            "The scheduling pipeline does not support these modes yet: "
            + ", ".join(unsupported)
        )


def dispatch_event_loop(scheduler: Scheduler):
    if scheduler.server_args.enable_pd_role_switch:
        return role_switch.run_event_loop_supervisor(
            scheduler,
            _dispatch_event_loop_once,
        )
    return _dispatch_event_loop_once(scheduler)


def _dispatch_event_loop_once(scheduler: Scheduler):
    # A PD role switch can select a different loop on the same scheduler.
    scheduler.flags.enable_skip_finishing_decode = False
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if get_parallel().pp_size > 1:
        # The ring takes each mode's stages, PD consensus included.
        scheduler.event_loop_pp()
    elif disaggregation_mode == DisaggregationMode.NULL:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def resolve_spawn_dp_rank(dp_rank: Optional[int]) -> Optional[int]:
    """Resolve the launcher DP rank, falling back to ``SGLANG_DP_RANK``."""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        return int(os.environ["SGLANG_DP_RANK"])
    return dp_rank


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
) -> None:
    """Configure scheduler worker logging and process title.

    Runs after `publish`, so every rank it labels the process with comes from
    the context. display_* ranks are cosmetic; runtime ranks stay local.
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    parallel = get_parallel()
    shown_dp = display_dp_rank if display_dp_rank is not None else parallel.dp_rank
    shown_tp = display_tp_rank if display_tp_rank is not None else parallel.tp_rank
    shown_moe_ep = (
        display_moe_ep_rank if display_moe_ep_rank is not None else parallel.moe_ep_rank
    )

    prefix = ""
    if shown_dp is not None:
        prefix += f" DP{shown_dp}"
    if parallel.pp_size > 1:
        prefix += f" PP{parallel.pp_rank}"
    if parallel.attn_cp_size > 1:
        prefix += f" ATTN_CP{parallel.attn_cp_rank}"
    if parallel.moe_dp_size > 1:
        prefix += f" MOE_DP{parallel.moe_dp_rank}"
    if parallel.tp_size > 1:
        prefix += f" TP{shown_tp}"
    if parallel.ep_size > 1:
        prefix += f" EP{shown_moe_ep}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(gpu_id)
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = resolve_spawn_dp_rank(dp_rank)
    # Placement is needed before publish(): TP/PP select a WORLD rank;
    # ordinary DP replicas have separate WORLDs and need an explicit DP rank.
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=pp_rank),
            dp_rank=dp_rank,
            gpu_id=gpu_id,
        ),
    )
    configure_scheduler_process(
        server_args,
        gpu_id,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            get_observability().otlp_service_name,
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "Scheduler"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(server_args, port_args)

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_sink._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.flags.gracefully_exit:
                scheduler.release_host_resources()
                # Last: anything above may still need a working communicator.
                abort_distributed_environment()
