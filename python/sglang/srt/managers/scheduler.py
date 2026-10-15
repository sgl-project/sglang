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

import faulthandler
import functools
import logging
import os
import signal
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.runtime_context import (
    SpawnRanks,
    get_context,
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_mm,
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
from sglang.srt.disaggregation.kv_events import (
    is_kv_publisher_rank,
    uses_dynamo_format,
)
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
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
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
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer_or_none
from sglang.srt.managers.min_free_slots_delayer import create_min_free_slots_delayer
from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    create_future_map,
)
from sglang.srt.managers.prefill_delayer import (
    RecentPrefillBatchSizeTracker,
    create_prefill_delayer,
)
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Req,
    ScheduleBatch,
    commit_step_counters,
)
from sglang.srt.managers.schedule_policy import (
    SchedulePolicy,
    priority_preemption_enabled,
)
from sglang.srt.managers.scheduler_components import worker_setup
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.bundles import (
    ComponentClasses,
    KvCapacity,
    KvResources,
    SchedulerServices,
    Streams,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.dynamic_chunk_sizer import (
    DynamicChunkSizer,
    fit_dynamic_chunk_sizer,
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
    CHECK_REQUEST_LISTS,
    SchedulerInvariantChecker,
    SchedulerWatchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
    is_entry_rank,
)
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
from sglang.srt.managers.scheduler_components.lora_admission import LoraAdmission
from sglang.srt.managers.scheduler_components.memory_usage import emit_startup_metrics
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    MetricsSink,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import (
    DriverOps,
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
from sglang.srt.managers.scheduler_components.pp_spec_forward import PPSpecForward
from sglang.srt.managers.scheduler_components.prefill_chunking import (
    resolve_chunked_prefill_size,
    resolve_truncation_align_size,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.managers.scheduler_components.request_intake import RequestIntake
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
from sglang.srt.managers.scheduler_components.step_counters import StepCounters
from sglang.srt.managers.scheduler_components.step_launcher import StepLauncher
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.managers.scheduler_components.tokenizer_setup import load_tokenizer
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    allocate_distinct_stream,
    is_health_check_generate_req,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.events import LoRANameTable
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    flush_trace_batch,
    set_schedule_time_batch,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_IDLE,
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    scheduler_stage_method,
)
from sglang.srt.observability.startup_time import build_scheduler_startup_time
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.plugins import load_plugins
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.indexer_topk import destroy_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import destroy_global_experts_capturer
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    is_cuda,
    is_hip,
    kill_itself_when_parent_died,
    rank_consensus_checker,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
    triton_load_watch,
)
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


# Min wall-clock between load publishes on the stalled no-batch path, which
# spins on_idle without sleeping. Bounds the O(queue) get_loads for both the
# DP-balancing writer and the router-facing socket.
LOAD_STALL_REFRESH_S = 0.05


_is_hip = is_hip()


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

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
        self.init_engine_state()
        self.init_soft_watchdog()

        # Config, and the process's distributed runtime
        self.init_server_args(server_args, port_args)
        self.init_component_classes()
        self.init_model_config()
        # ZBAL switches the allocator, so it precedes any torch allocation.
        self.init_zbal_on_npu()
        self.init_parallel_runtime()

        # Metrics, IPC, tokenizer and kernel configs
        self.init_metrics_collector()
        self.init_ipc_channels(port_args)
        self.init_load_snapshot_writer(port_args)
        self.init_idle_sleeper()
        self.init_tokenizer()
        self.init_moe_gemm_config()
        self.init_mamba_backend()

        # Workers. PR-fix reverts patch targets (e.g. _init_pools) that run
        # inside init_model_worker, so they apply first.
        maybe_revert_pr_fix()
        self.init_model_worker()
        self.init_worker_info()
        self.init_process_groups()
        self.init_startup_memory()
        self.maybe_stall_for_test()

        # KV cache and the mode owners over it
        self.init_kv_event_lora_names()
        self.init_kv_cache()
        self.init_penalty_store()
        self.emit_metrics_constants()
        self.maybe_init_hccl_dp_prewarm()
        self.init_kv_bundles()
        self.init_kv_canary()
        self.init_mode_owners()
        self.init_hisparse_coordinator()
        self.maybe_init_decode_offload_manager()

        # Scheduling state and policy
        self.init_running_status()
        self.init_session_controller()
        self.init_chunked_prefill()
        self.maybe_init_dynamic_chunk_sizer()
        self.init_metrics_sink()
        self.init_schedule_policy()
        self.maybe_init_prefill_delayer()
        self.init_new_token_ratio_tracker()
        self.init_watchdog()
        self.init_memory_saver_adapter()
        self.init_recv_skipper()
        self.init_gc_logger()
        self.init_profiler()

        # Start the embedded Rust frontend (rank 0). Must precede
        # init_disaggregation: on PD prefill the rust api listener also serves
        # the KV bootstrap registry, and the KVManagers built there register to
        # it synchronously. (The listener is bound synchronously inside launch,
        # so the registry is accepting once this returns.) Must also precede
        # the request receiver, which takes the rust server as its ingress.
        self.maybe_init_rust_server()
        self.init_disaggregation()
        self.maybe_init_mm_receiver()

        # Streams: the schedule stream settles before anything that waits on it.
        self.init_overlap()
        self.init_copy_stream()
        self.finalize_schedule_stream()
        self.init_streams()
        self.maybe_init_pp_spec_forward()
        self.maybe_init_ngram_embedding()
        self.init_truncation_align()

        # Components, each built after the ones it takes
        self.init_lora_admission()
        self.init_grammar_manager()
        self.maybe_init_scripted_scheduler_hook()
        self.init_output_streamer()
        self.init_request_receiver()
        self.init_step_builder()
        self.init_dp_attn_adapter()
        self.init_pool_stats_observer()
        self.init_invariant_checker()
        self.init_watchdog_diagnostics()
        self.init_kv_events_publisher()
        self.init_load_publisher()
        self.init_services()
        self.init_beam_coordinator()
        self.init_request_retirer()
        self.init_batch_result_processor()
        self.init_request_registry()
        self.maybe_init_dllm()
        self.maybe_init_disagg_prefill()
        self.maybe_init_disagg_decode()
        self.init_request_intake()
        self.init_step_planner()
        self.maybe_init_disagg_move_gate()
        self.init_mode_stages()
        self.init_step_launcher()
        # Built after the request registry and the mode owners, which they read.
        self.init_weight_updater()
        self.init_flush_wrapper()
        self.init_request_dispatcher()
        self.init_step_counters()
        self.init_load_inquirer()
        self.init_metrics_reporter()
        self.init_rank_consensus_checker()

        self.flags.is_initializing = False
        self.init_startup_timing_summary()
        # Reads the startup time.
        self.init_internal_state()
        if (hook := self.scripted_scheduler_hook) is not None:
            hook.attach(self)

    def init_engine_state(self) -> None:
        self.counters = LaunchCounters()
        self.flags = EngineFlags()

    def init_server_args(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        # Read once: the tokenizer init and the IPC channels gate on the same
        # value and must not disagree.
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        reject_modes_without_pipeline_support()

    def init_parallel_runtime(self) -> None:
        bootstrap.init_parallel_runtime(
            server_args=self.server_args,
            device=get_device().device,
            dist_port=self.nccl_port,
        )
        bootstrap.init_layer_runtime(model_config=self.model_config)

    def maybe_stall_for_test(self) -> None:
        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

    def init_component_classes(self) -> None:
        self.component_classes = ComponentClasses(
            rust_server=self.get_rust_server_class(),
            output_streamer=self.get_output_streamer_class(),
        )

    def get_rust_server_class(self) -> type[RustServer]:
        return RustServer

    def get_output_streamer_class(self) -> type[SchedulerOutputStreamer]:
        return SchedulerOutputStreamer

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
        worker_setup.maybe_prewarm_hccl_dp(
            self.tp_worker.model_runner.model_config.hf_config
        )

    def init_zbal_on_npu(self):
        worker_setup.init_zbal_on_npu()

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
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_entry_rank(),
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

    def init_load_snapshot_writer(self, port_args: PortArgs) -> None:
        self.load_snapshot_writer = (
            create_load_snapshot_writer_or_none(port_args) if is_entry_rank() else None
        )

    def init_idle_sleeper(self) -> None:
        if is_entry_rank() and get_device().sleep_on_idle:
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
        self.is_generation = self.model_config.is_generation
        loaded = load_tokenizer(
            model_config=self.model_config,
            server_args=self.server_args,
            skip_tokenizer_init=self.skip_tokenizer_init,
        )
        self.tokenizer = loaded.tokenizer
        self.processor = loaded.processor
        self._mm_processor = loaded.mm_processor

    def init_mamba_backend(self) -> None:
        if initialize_mamba_selective_state_update_backend is not None:
            initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        worker_setup.init_moe_gemm_config(self.model_config.hf_text_config)
        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync()

    def init_model_worker(self):
        started = worker_setup.start_workers(
            server_args=self.server_args,
            nccl_port=self.nccl_port,
            spec_algorithm=self.spec_algorithm,
        )
        self.tp_worker = started.workers.tp_worker
        self.draft_worker = started.workers.draft_worker
        self.model_worker = started.workers.model_worker
        self.dllm_worker = started.dllm_worker
        self.schedule_stream = started.schedule_stream
        self.kv_cache_allocation_time = started.kv_cache_allocation_time
        self.maybe_init_external_corpus_manager()

    def maybe_init_external_corpus_manager(self):
        """The external corpus of ngram speculative decoding, over its draft
        worker."""
        if self.draft_worker is None or not self.spec_algorithm.is_ngram():
            self.external_corpus_manager = None
            return
        from sglang.srt.speculative.external_corpus_manager import (
            ExternalCorpusManager,
        )

        self.external_corpus_manager = ExternalCorpusManager(
            self.draft_worker,
            self.ipc_channels.send_to_tokenizer.send_output,
        )

    def init_worker_info(self):
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
        worker_setup.set_default_pp_max_micro_batch_size(self.max_running_requests)
        set_random_seed(self.random_seed)

    def init_process_groups(self):
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

    def init_startup_memory(self):
        self.startup_available_gpu_memory_gb = worker_setup.measure_startup_memory(
            device=self.device,
            max_total_num_tokens=self.max_total_num_tokens,
            max_prefill_tokens=self.max_prefill_tokens,
            max_running_requests=self.max_running_requests,
            context_len=self.model_config.context_len,
        )

    def init_kv_cache(self) -> None:
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=get_schedule().page_size,
            spec_algorithm=self.spec_algorithm,
            enable_metrics=get_observability().enable_metrics,
            enable_kv_cache_events=is_kv_publisher_rank(
                get_observability().kv_events_config
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
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache

    def init_penalty_store(self) -> None:
        self.penalty_store = PenaltyStore(
            self.model_config.vocab_size,
            self.req_to_token_pool.req_to_token.shape[0],
            self.req_to_token_pool.device,
        )

    def init_kv_canary(self) -> None:
        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

    def maybe_init_decode_offload_manager(self) -> None:
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

    def emit_metrics_constants(self) -> None:
        if not get_observability().enable_metrics:
            return

        emit_startup_metrics(
            self.metrics_collector,
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            max_total_num_tokens=self.max_total_num_tokens,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
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
            kv_shard_widening=page_interleave_shard_size(
                self.token_to_kv_pool_allocator
            ),
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

    def init_mode_owners(self) -> None:
        self.mode_owners = ModeOwners()

    # The mode owners, read through the holder a PD role switch updates.
    @property
    def disagg_prefill(self) -> Optional[DisaggPrefill]:
        return self.mode_owners.disagg_prefill

    @property
    def disagg_decode(self) -> Optional[DisaggDecode]:
        return self.mode_owners.disagg_decode

    @property
    def dllm(self) -> Optional[DllmManager]:
        return self.mode_owners.dllm

    @property
    def hisparse_coordinator(self) -> Optional[HiSparseCoordinator]:
        return self.mode_owners.hisparse_coordinator

    def init_hisparse_coordinator(self) -> None:
        self.mode_owners.hisparse_coordinator = None
        if not get_memory().enable_hisparse:
            return

        # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
        self.mode_owners.hisparse_coordinator = (
            self.tp_worker.model_runner.hisparse_coordinator
        )
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
        # The flat driver's schedule slot: every decoding request (the PP ring
        # keeps one per microbatch slot).
        self.flat_slot = ScheduleSlot()
        # The launched steps whose results are not committed yet.
        self.pending = PendingSteps()
        self._deferred_input_requests: List = []

    def init_session_controller(self) -> None:
        self.session_controller = SessionController(self.tree_cache)

    def init_chunked_prefill(self):
        self.chunked_prefill_size = resolve_chunked_prefill_size(self.model_config)
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
        )

    def maybe_init_dynamic_chunk_sizer(self) -> None:
        """Profile a PP prefill latency model that sizes chunks per stage."""
        self.dynamic_chunk_sizer: Optional[DynamicChunkSizer] = None
        if not (get_schedule().enable_dynamic_chunking and get_parallel().pp_size > 1):
            return
        self.dynamic_chunk_sizer = fit_dynamic_chunk_sizer(
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

    def init_metrics_sink(self) -> None:
        self.metrics_sink = MetricsSink(
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
            device=self.device,
        )
        self.scheduler_stage_metrics = self.metrics_sink.scheduler_stage_metrics
        self.enable_fpm = self.metrics_sink.enable_fpm

    def init_metrics_reporter(self) -> None:
        # Override point for deployments that need a specialized reporter.
        self.metrics_reporter = SchedulerMetricsReporter(
            metrics_sink=self.metrics_sink,
            request_registry=self.request_registry,
            core_lists=self.lists,
            counters=self.counters,
            pool_stats_observer=self.pool_stats_observer,
            grammar_manager=self.grammar_manager,
            mm_receiver=self.mm_receiver,
            kv_events_publisher=self.kv_events_publisher,
            mode_owners=self.mode_owners,
            kv=self.kv,
            capacity=self.capacity,
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            device=self.device,
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

    def maybe_init_prefill_delayer(self) -> None:
        self.prefill_delayer = create_prefill_delayer(
            metrics_collector=(
                self.metrics_collector if self.metrics_sink.enable_metrics else None
            ),
        )

    def init_new_token_ratio_tracker(self) -> None:
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

    def init_watchdog(self) -> None:
        self.watchdog = SchedulerWatchdog(
            counters=self.counters,
            flags=self.flags,
            watchdog_timeout=get_device().watchdog_timeout,
        )

    def init_memory_saver_adapter(self) -> None:
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )

    def init_recv_skipper(self) -> None:
        self.recv_skipper = SchedulerRecvSkipper.maybe_create()

    def init_gc_logger(self) -> None:
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
            self.mode_owners.disagg_prefill = None
            return
        self.mode_owners.disagg_prefill = DisaggPrefill(
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
            batch_result_processor=self.batch_result_processor,
            counters=self.counters,
            request_retirer=self.request_retirer,
        )

    def maybe_init_disagg_decode(self) -> None:
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            self.mode_owners.disagg_decode = None
            return
        self.mode_owners.disagg_decode = DisaggDecode(
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
            lora_admission=self.lora_admission,
            pool_stats_observer=self.pool_stats_observer,
            decode_offload_manager=self.decode_offload_manager,
            request_retirer=self.request_retirer,
            flags=self.flags,
            core_lists=self.lists,
            pending=self.pending,
            policy=self.policy,
            future_map=self.future_map,
            ngram_embedding_manager=self.ngram_embedding_manager,
        )

    def init_request_registry(self) -> None:
        self.request_registry = RequestRegistry(
            core_lists=self.lists,
            grammar_manager=self.grammar_manager,
            request_retirer=self.request_retirer,
            ipc_channels=self.ipc_channels,
            kv=self.kv,
            mm_receiver=self.mm_receiver,
            pending=self.pending,
            decode_offload_manager=self.decode_offload_manager,
            mode_owners=self.mode_owners,
        )

    def init_request_intake(self) -> None:
        self.request_intake = RequestIntake(
            core_lists=self.lists,
            counters=self.counters,
            kv=self.kv,
            capacity=self.capacity,
            ipc_channels=self.ipc_channels,
            metrics_sink=self.metrics_sink,
            output_streamer=self.output_streamer,
            beam_coordinator=self.beam_coordinator,
            session_controller=self.session_controller,
            grammar_manager=self.grammar_manager,
            request_retirer=self.request_retirer,
            mode_owners=self.mode_owners,
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            transfer_backend=self.transfer_backend,
            tokenizer=self.tokenizer,
            mm_processor=self._mm_processor,
            pad_input_ids_func=self.tp_worker.get_pad_input_ids_func(),
            dp_tp_group=self.dp_tp_group,
            max_req_len=self.max_req_len,
            max_req_input_len=self.max_req_input_len,
            max_queued_requests=self.max_queued_requests,
            sliding_window_size=self.sliding_window_size,
            chunked_prefill_size=self.chunked_prefill_size,
            sampling_mask_max_tokens=self.server_args.sampling_mask_max_tokens,
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

        self.future_map = create_future_map(
            spec_algorithm=self.spec_algorithm,
            device=self.device,
            req_to_token_pool=self.req_to_token_pool,
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
        )
        self.model_worker.attach_future_map(
            self.future_map,
            enable_overlap=(not get_schedule().disable_overlap_schedule),
        )

    def init_copy_stream(self) -> None:
        # copy_stream is also used by PP (non-overlap) via the PP ring; init
        # unconditionally to match main.
        self.copy_stream: CudaStream = self.device_module.Stream()

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

    def init_step_launcher(self) -> None:
        self.step_launcher = StepLauncher(
            kv=self.kv,
            streams=self.streams,
            future_map=self.future_map,
            counters=self.counters,
            flags=self.flags,
            pending=self.pending,
            core_lists=self.lists,
            ipc_channels=self.ipc_channels,
            metrics_sink=self.metrics_sink,
            scheduler_stage_metrics=self.scheduler_stage_metrics,
            profiler_manager=self.profiler_manager,
            beam_coordinator=self.beam_coordinator,
            pp_spec_forward=self.pp_spec_forward,
            stages=self.stages,
            scripted_scheduler_hook=self.scripted_scheduler_hook,
            tp_worker=self.tp_worker,
            model_worker=self.model_worker,
            spec_algorithm=self.spec_algorithm,
            is_generation=self.is_generation,
            device_module=self.device_module,
        )

    def maybe_init_pp_spec_forward(self) -> None:
        """The forward a speculative step runs under pipeline parallelism."""
        self.pp_spec_forward = (
            PPSpecForward(
                kv=self.kv,
                streams=self.streams,
                future_map=self.future_map,
                tp_worker=self.tp_worker,
                model_worker=self.model_worker,
                pp_group=self.pp_group,
                spec_algorithm=self.spec_algorithm,
            )
            if get_parallel().pp_size > 1 and not self.spec_algorithm.is_none()
            else None
        )

    def maybe_init_ngram_embedding(self):
        self.ngram_embedding_manager = (
            self.tp_worker.model_runner.ngram_embedding_manager
        )

    def init_truncation_align(self):
        self.truncation_align_size = resolve_truncation_align_size(
            hf_config=self.model_config.hf_config,
            chunked_prefill_size=self.chunked_prefill_size,
            disaggregation_mode=self.disaggregation_mode,
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
            device_module=self.device_module,
            streams=self.streams,
        )
        self.pp_ring = PPRing(
            channel=channel,
            stages=self.stages,
            pending=self.pending,
            recv_skipper=self.recv_skipper,
            ops=DriverOps(
                ingest_requests=self.ingest_requests,
                launch_step=self.launch_step,
                commit_step=self.commit_step,
                run_batch=self.step_launcher.run_batch,
                on_idle=self.on_idle,
                mark_idle=self.mark_sched_idled,
            ),
            counters=self.counters,
            pp_spec_forward=self.pp_spec_forward,
            pp_group=self.pp_group,
            device_module=self.device_module,
            kv=self.kv,
            streams=self.streams,
            tp_worker=self.tp_worker,
            future_map=self.future_map,
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
                    self.step_launcher.launch_batch_sample_if_needed(
                        launched.result, launched.step
                    )
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
            result = self.step_launcher.run_batch(step)
            if overlap:
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
        self.join_launched_step(step, slot)
        record = PendingStep(
            step=step,
            result=result,
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
        self.process_batch_result(record.step, record.result)
        # Its forward has finished: what released requests left to it is free.
        self.request_retirer.release_held(record.step)
        self.step_counters.record(record)
        if CHECK_REQUEST_LISTS:
            self.invariant_checker.check_no_finished_reqs(
                self.request_registry.request_lists()
            )

    def leave_finished_reqs(self) -> None:
        """Drop the requests a commit finished from decoding (PD decode holds
        those still releasing KV), and from the mode's own lists (HiSparse
        staging)."""
        if (hold_finished := self.stages.hold_finished) is not None:
            hold_finished([r for r in self.lists.decoding if r.finished()])
        self.lists.decoding = [r for r in self.lists.decoding if not r.finished()]
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
                vmm_errors = self.request_intake.materialize_cuda_vmm_inputs(recv_req)

            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(
                recv_req
            ) and not self.request_registry.is_fully_idle(for_health_check=True):
                self.output_streamer.defer_health_check(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            if vmm_errors is not None and any(vmm_errors):
                self.request_intake.dispatch_tokenized_mm_requests(recv_req, vmm_errors)
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

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            counters=self.counters,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.step_planner.flush_cache,
            request_registry=self.request_registry,
            flags=self.flags,
            mode_owners=self.mode_owners,
            metrics_collector=self.metrics_collector,
        )

    def init_flush_wrapper(self) -> None:
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.step_planner.flush_cache,
            request_registry=self.request_registry,
            ipc_channels=self.ipc_channels,
        )

    def init_lora_admission(self) -> None:
        self.lora_admission = LoraAdmission(tp_worker=self.tp_worker)

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

            # The scheduler attaches itself once it is built.
            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def _hosts_rust_server(self) -> bool:
        """Whether this scheduler rank embeds the Rust server (rank 0 only) —
        and with it the server-process duties a Python ``TokenizerManager``
        would otherwise own (e.g. serving the PD KV bootstrap registry)."""
        return envs.SGLANG_RUST_SERVER.get() and is_entry_rank()

    def maybe_init_rust_server(self) -> None:
        """Start the embedded Rust server (rank 0) if ``SGLANG_RUST_SERVER`` is
        set, and point the ingress receiver at it. All the plumbing lives in
        ``RustServer`` (scheduler_components/rust_scheduler.py)."""

        if not self._hosts_rust_server():
            # Always define the attribute: init_output_streamer,
            # init_request_receiver and the process_input_requests hook read
            # self.rust_server unconditionally.
            self.rust_server = None
            return

        rust_server = self.component_classes.rust_server.launch(
            server_args=self.server_args,
            model_config=self.model_config,
            processor=self.processor,
            tokenizer_path=self.rust_server_tokenizer_path(),
            max_total_num_tokens=self.max_total_num_tokens,
        )
        self.rust_server = rust_server
        # Park the idle loop on the request ring within the rank-0 rust-server
        self.idle_sleeper = RustServerIdleSleeper(rust_server, flags=self.flags)

    def rust_server_tokenizer_path(self) -> str:
        return get_serving().tokenizer_path

    def init_request_receiver(self) -> None:
        self.request_receiver = SchedulerRequestReceiver(
            # The rust server *is* the ingress source: SchedulerRequestReceiver
            # drains its request ring (rust_server_mode) instead of a zmq socket.
            recv_from_tokenizer=(
                self.rust_server
                if self.rust_server is not None
                else self.ipc_channels.recv_from_tokenizer
            ),
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=SchedulerInputBlocker.maybe_create(),
            mm_receiver=self.mm_receiver,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get(),
            output_streamer=self.output_streamer,
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

    def init_step_planner(self) -> None:
        self.step_planner = StepPlanner(
            core_lists=self.lists,
            pending=self.pending,
            flags=self.flags,
            counters=self.counters,
            kv=self.kv,
            capacity=self.capacity,
            policy=self.policy,
            new_token_ratio_tracker=self.new_token_ratio_tracker,
            pool_stats_observer=self.pool_stats_observer,
            lora_admission=self.lora_admission,
            request_registry=self.request_registry,
            beam_coordinator=self.beam_coordinator,
            step_builder=self.step_builder,
            hisparse_coordinator=self.hisparse_coordinator,
            request_retirer=self.request_retirer,
            grammar_manager=self.grammar_manager,
            metrics_sink=self.metrics_sink,
            ipc_channels=self.ipc_channels,
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            decode_offload_manager=self.decode_offload_manager,
            mode_owners=self.mode_owners,
            request_intake=self.request_intake,
            prefill_delayer=self.prefill_delayer,
            prefill_bs_tracker=RecentPrefillBatchSizeTracker(
                window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
            ),
            # DFlash auto-enables the legacy formula; other workloads opt in via
            # --min-free-slots-delay. Built independently of the prefill delayer.
            min_free_slots_delayer=create_min_free_slots_delayer(
                get_schedule().min_free_slots_delay,
                self.max_running_requests,
                is_dflash_family=self.spec_algorithm.is_dflash_family(),
            ),
            dynamic_chunk_sizer=self.dynamic_chunk_sizer,
            spec_algorithm=self.spec_algorithm,
            is_hybrid_swa=self.is_hybrid_swa,
            require_mlp_sync=self.require_mlp_sync,
            chunked_prefill_size=self.chunked_prefill_size,
            is_mixed_chunk=self.is_mixed_chunk,
            prefill_decode_interval=get_schedule().prefill_decode_interval or 0,
            max_prefill_tokens=self.max_prefill_tokens,
            enable_priority_preemption=priority_preemption_enabled(),
            truncation_align_size=self.truncation_align_size,
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
            pending=self.pending,
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

    def init_step_counters(self) -> None:
        self.step_counters = StepCounters()

    def init_load_inquirer(self) -> None:
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
            request_registry=self.request_registry,
            mode_owners=self.mode_owners,
            step_counters=self.step_counters,
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = self.component_classes.output_streamer(
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
            request_retirer=self.request_retirer,
        )

    def init_request_retirer(self) -> None:
        self.request_retirer = RequestRetirer(
            kv=self.kv,
            beam_coordinator=self.beam_coordinator,
            hisparse_coordinator=self.hisparse_coordinator,
            decode_offload_manager=self.decode_offload_manager,
            pending=self.pending,
            output_streamer=self.output_streamer,
            ipc_channels=self.ipc_channels,
        )

    def maybe_init_dllm(self) -> None:
        if get_exec().dllm.dllm_algorithm is None:
            self.mode_owners.dllm = None
            return
        self.mode_owners.dllm = DllmManager(
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
            enable_priority_preemption=priority_preemption_enabled(),
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
            request_registry=self.request_registry,
            startup_time=self.startup_time,
            record_step_time=RECORD_STEP_TIME,
        )

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        self.request_intake.handle_generate_request(recv_req)

    def handle_batch_generate_request(self, recv_req: BatchTokenizedGenerateReqInput):
        self.request_intake.handle_batch_generate_request(recv_req)

    def handle_embedding_request(self, recv_req: TokenizedEmbeddingReqInput):
        self.request_intake.handle_embedding_request(recv_req)

    def handle_batch_embedding_request(self, recv_req: BatchTokenizedEmbeddingReqInput):
        self.request_intake.handle_batch_embedding_request(recv_req)

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
        self.request_retirer.retire_aborted(req)

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
                self.request_intake.process_storage_prefetch_retries()

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

        if stages.admit is not None:
            new_batch = stages.admit(slot)
        elif self.step_planner.should_defer_prefill():
            new_batch = None
        else:
            new_batch = self.step_planner.get_new_batch_prefill(slot)

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
        ret = (
            new_batch
            if new_batch is not None
            else self.step_planner.get_decode_step(slot)
        )
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

        return self.finalize_step(self.step_planner.get_new_batch_prefill(slot), slot)

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
        return self.finalize_step(self.step_planner.get_new_batch_prefill(slot), slot)

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
                self.request_intake.add_request_to_queue(req)
        reqs = self.disagg_decode.pop_reqs_to_admit(slot)
        if reqs:
            self.disagg_decode.commit_prefill_handoff(reqs)
            self.lists.join_decoding([r for r in reqs if not r.finished()], slot)

        return self.finalize_step(self.step_planner.get_decode_step(slot), slot)

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
        self.step_planner.arm_prefill_decode_interval(step)

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

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_BATCH_RESULT)
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
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
        self.leave_finished_reqs()

        # Submit this batch's queued host backups before the next scheduler step.
        self.tree_cache.flush_pending_backups()

        self.metrics_sink.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self.request_retirer.clear_finished_media(batch.reqs)
        self.output_streamer.maybe_reply_health_check()
        self.metrics_sink.update_device_timer()

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_process_result(batch)

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
        return self.step_planner.flush_cache(empty_cache)

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
        self.weight_updater.record_weight_version_change(
            new_version=recv_req.new_version
        )
        return UpdateWeightVersionReqOutput()

    # The abort RPC handler.
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

        self.step_planner.retract_for_pause()

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
