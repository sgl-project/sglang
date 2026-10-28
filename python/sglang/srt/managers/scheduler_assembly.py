"""Builds the scheduler's components, in order, into three parts:

- `SchedulerParts`: everything below the mode owners, which the loop runs on
  and the mode is built from;
- `SchedulerMode`: what a PD role switch replaces as one, built from the parts;
- `SchedulerHub`: the components over the mode owners.

`Scheduler` holds the three and runs the loop over them. Nothing here takes
the scheduler: the loop's own stages come in as `LoopSchedules`.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Set, Tuple

import torch

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
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
from sglang.srt.disaggregation.prefill import DisaggPrefill
from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.distributed import bootstrap
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.manager import DllmManager
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer_or_none
from sglang.srt.managers.overlap_utils import create_future_map
from sglang.srt.managers.prefill_delayer import (
    create_prefill_delayer,
)
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_policy import (
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components import worker_setup
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.bundles import (
    ComponentClasses,
    KvCapacity,
    KvResources,
    RequestLimits,
    SchedulerServices,
    Streams,
    Workers,
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
    LoopSchedules,
    ModeStages,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
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
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.managers.scheduler_components.step_builder import StepBuilder
from sglang.srt.managers.scheduler_components.step_counters import StepCounters
from sglang.srt.managers.scheduler_components.step_launcher import StepLauncher
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.managers.scheduler_components.tokenizer_setup import load_tokenizer
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.utils import allocate_distinct_stream
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.startup_time import build_scheduler_startup_time
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
)
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_gc_logger,
    is_cuda,
    is_hip,
    rank_consensus_checker,
    require_mlp_sync,
    set_random_seed,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

try:
    from sglang.kernels.ops.mamba.triton_ops import (
        initialize_mamba_selective_state_update_backend,
    )
except ImportError:
    initialize_mamba_selective_state_update_backend = None

if TYPE_CHECKING:
    from torch.cuda import Stream as CudaStream

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

_is_hip = is_hip()


@dataclass(frozen=True, kw_only=True, slots=True)
class SchedulerParts:
    """The scheduler's parts below the mode owners: what the loop runs on and
    what the mode is built from. None of them holds `ModeOwners` but
    `mode_owners` itself, which a PD role switch updates in place."""

    # The engine state, shared with the components that read it.
    counters: LaunchCounters
    flags: EngineFlags
    lists: CoreLists
    pending: PendingSteps
    # Config.
    server_args: ServerArgs
    model_config: ModelConfig
    spec_algorithm: SpeculativeAlgorithm
    is_generation: bool
    require_mlp_sync: bool
    # Process IO.
    ipc_channels: SchedulerIpcChannels
    load_snapshot_writer: Any
    idle_sleeper: Any
    rust_server: Optional[RustServer]
    # The workers and their KV.
    workers: Workers
    external_corpus_manager: Any
    limits: RequestLimits
    kv_event_lora_names: KvEventLoraNames
    kv: KvResources
    capacity: KvCapacity
    disable_radix_cache: bool
    mode_owners: ModeOwners
    decode_offload_manager: Optional[DecodeKVCacheOffloadManager]
    # Scheduling state and policy.
    session_controller: SessionController
    metrics_sink: MetricsSink
    policy: SchedulePolicy
    new_token_ratio_tracker: NewTokenRatioTracker
    # The watchdogs run on their own threads; held for their lifetime.
    soft_watchdog: Optional[SchedulerWatchdog]
    watchdog: SchedulerWatchdog
    recv_skipper: Optional[SchedulerRecvSkipper]
    profiler_manager: SchedulerProfilerManager
    # Streams.
    future_map: Any
    streams: Streams
    pp_spec_forward: Optional[PPSpecForward]
    ngram_embedding_manager: Any
    # Components.
    lora_admission: LoraAdmission
    grammar_manager: GrammarManager
    scripted_scheduler_hook: Any
    output_streamer: SchedulerOutputStreamer
    request_receiver: SchedulerRequestReceiver
    step_builder: StepBuilder
    dp_attn_adapter: SchedulerDPAttnAdapter
    pool_stats_observer: SchedulerPoolStatsObserver
    kv_events_publisher: SchedulerKvEventsPublisher
    load_publisher: SchedulerLoadPublisher
    services: SchedulerServices
    beam_coordinator: BeamCoordinator
    batch_result_processor: SchedulerBatchResultProcessor
    request_retirer: RequestRetirer
    step_counters: StepCounters


@dataclass(frozen=True, kw_only=True, slots=True)
class SchedulerMode:
    """What a PD role switch replaces, as one: the disaggregation mode, the
    stages the drivers run for it and the step launcher that holds them. The
    PD owners themselves are replaced in `ModeOwners`."""

    disaggregation_mode: DisaggregationMode
    stages: ModeStages
    step_launcher: StepLauncher


@dataclass(frozen=True, kw_only=True, slots=True)
class SchedulerHub:
    """The components over the mode owners: each holds `ModeOwners`, or a
    component that does, so nothing in the parts or the mode takes one."""

    request_registry: RequestRegistry
    invariant_checker: SchedulerInvariantChecker
    request_intake: RequestIntake
    step_planner: StepPlanner
    weight_updater: SchedulerWeightUpdaterManager
    flush_wrapper: SchedulerFlushWrapper
    load_inquirer: SchedulerLoadInquirer
    metrics_reporter: SchedulerMetricsReporter
    internal_state: SchedulerInternalStateController


def default_component_classes() -> ComponentClasses:
    """The component classes of a plain `Scheduler`."""
    return ComponentClasses(
        rust_server=RustServer,
        output_streamer=SchedulerOutputStreamer,
        prefill_adder=PrefillAdder,
        request_intake=RequestIntake,
    )


def assemble_scheduler(
    server_args: ServerArgs,
    port_args: PortArgs,
    schedules: LoopSchedules,
    component_classes: ComponentClasses,
) -> Tuple[SchedulerParts, SchedulerMode, SchedulerHub]:
    """Build the scheduler's parts, its mode and its hub, in that order."""
    scheduler_startup_begin = time.perf_counter()
    # init_soft_watchdog starts a daemon thread that reads these on its first tick.
    counters = LaunchCounters()
    flags = EngineFlags()
    soft_watchdog = init_soft_watchdog(counters=counters, flags=flags)

    # Config, and the process's distributed runtime
    nccl_port = port_args.nccl_port
    # Read once: the tokenizer init and the IPC channels gate on the same
    # value and must not disagree.
    skip_tokenizer_init = get_serving().skip_tokenizer_init
    spec_algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    reject_modes_without_pipeline_support()
    model_config = ModelConfig.from_server_args(server_args)
    # ZBAL switches the allocator, so it precedes any torch allocation.
    worker_setup.init_zbal_on_npu()
    bootstrap.init_parallel_runtime(
        server_args=server_args,
        device=get_device().device,
        dist_port=nccl_port,
    )
    bootstrap.init_layer_runtime(model_config=model_config)

    # Metrics, IPC, tokenizer and kernel configs
    metrics_collector_context = SchedulerMetricsCollector.init_new(
        server_args=server_args,
        enable_priority_scheduling=get_schedule().enable_priority_scheduling,
        enable_lora=get_lora().enable_lora,
        enable_hierarchical_cache=get_memory().enable_hierarchical_cache,
    )
    metrics_collector = metrics_collector_context.collector
    ipc_channels = SchedulerIpcChannels.create(
        port_args=port_args,
        is_rank_zero=is_entry_rank(),
        # The snapshot taken at construction, not a second bag read: this
        # scheduler gates its tokenizer init on the same value, and the two
        # must not be able to disagree.
        skip_tokenizer_init=skip_tokenizer_init,
        metrics_enabled=get_observability().enable_metrics
        and (
            get_parallel().attn_tp_rank == 0
            or get_observability().enable_metrics_for_all_schedulers
        ),
        enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
    )
    load_snapshot_writer = (
        create_load_snapshot_writer_or_none(port_args) if is_entry_rank() else None
    )
    idle_sleeper = init_idle_sleeper(ipc_channels=ipc_channels, flags=flags)
    is_generation = model_config.is_generation
    loaded = load_tokenizer(
        model_config=model_config,
        server_args=server_args,
        skip_tokenizer_init=skip_tokenizer_init,
    )
    tokenizer = loaded.tokenizer
    processor = loaded.processor
    mm_processor = loaded.mm_processor
    worker_setup.init_moe_gemm_config(model_config.hf_text_config)
    # This must be called after initialize_moe_config
    requires_mlp_sync = require_mlp_sync()
    if initialize_mamba_selective_state_update_backend is not None:
        initialize_mamba_selective_state_update_backend(server_args)

    # Workers. PR-fix reverts patch targets (e.g. _init_pools) that run
    # inside start_workers, so they apply first.
    maybe_revert_pr_fix()
    started = worker_setup.start_workers(
        server_args=server_args, nccl_port=nccl_port, spec_algorithm=spec_algorithm
    )
    workers = started.workers
    tp_worker = workers.tp_worker
    draft_worker = workers.draft_worker
    model_worker = workers.model_worker
    schedule_stream = started.schedule_stream
    kv_cache_allocation_time = started.kv_cache_allocation_time
    external_corpus_manager = maybe_init_external_corpus_manager(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        ipc_channels=ipc_channels,
    )
    # Get token and memory info from the model worker
    (
        max_total_num_tokens,
        max_prefill_tokens,
        max_running_requests,
        max_queued_requests,
        max_req_len,
        max_req_input_len,
        random_seed,
        device,
        forward_stream,
        _,
        _,
        _,
    ) = (started.dllm_worker or tp_worker).get_worker_info()
    worker_setup.set_default_pp_max_micro_batch_size(max_running_requests)
    set_random_seed(random_seed)
    tp_group = get_parallel().tp_group
    attn_tp_group = get_parallel().attn_tp_group
    attn_cp_group = get_parallel().attn_cp_group
    pp_group = get_parallel().pp_group
    # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
    # Entry rank is the local rank 0 in that group. Use the CPU (gloo) group to
    # broadcast VLM Python objects and avoid CUDA stream/device coupling (#11910).
    dp_tp_group = get_dp_tp_group()
    dp_tp_cpu_group = dp_tp_group.cpu_group
    startup_available_gpu_memory_gb = worker_setup.measure_startup_memory(
        device=device,
        max_total_num_tokens=max_total_num_tokens,
        max_prefill_tokens=max_prefill_tokens,
        max_running_requests=max_running_requests,
        context_len=model_config.context_len,
    )
    if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
        time.sleep(t)

    # KV cache and the mode owners over it
    # The table that maps lora_id to adapter name for KV events.
    kv_event_lora_names = KvEventLoraNames.from_lora_refs(get_lora().lora_paths)
    result = kv_cache_builder.build_kv_cache(
        server_args=server_args,
        model_config=model_config,
        tp_worker=tp_worker,
        page_size=get_schedule().page_size,
        spec_algorithm=spec_algorithm,
        enable_metrics=get_observability().enable_metrics,
        enable_kv_cache_events=is_kv_publisher_rank(
            get_observability().kv_events_config
        ),
        dynamo_kv_event_format=uses_dynamo_format(get_observability().kv_events_config),
        kv_event_lora_names=kv_event_lora_names,
        enable_hierarchical_cache=get_memory().enable_hierarchical_cache,
        hicache_draft_plan=(
            draft_worker.hicache_draft_plan if draft_worker is not None else None
        ),
    )
    is_hybrid_swa = result.is_hybrid_swa
    is_hybrid_ssm = result.is_hybrid_ssm
    sliding_window_size = result.sliding_window_size
    full_tokens_per_layer = result.full_tokens_per_layer
    swa_tokens_per_layer = result.swa_tokens_per_layer
    req_to_token_pool = result.req_to_token_pool
    token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
    disable_radix_cache = result.disable_radix_cache
    tree_cache = result.tree_cache
    penalty_store = PenaltyStore(
        model_config.vocab_size,
        req_to_token_pool.req_to_token.shape[0],
        req_to_token_pool.device,
    )
    if get_observability().enable_metrics:
        emit_startup_metrics(
            metrics_collector,
            tp_worker=tp_worker,
            draft_worker=draft_worker,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            max_total_num_tokens=max_total_num_tokens,
            swa_tokens_per_layer=swa_tokens_per_layer,
            context_len=model_config.context_len,
            startup_available_gpu_memory_gb=startup_available_gpu_memory_gb,
        )
    worker_setup.maybe_prewarm_hccl_dp(tp_worker.model_runner.model_config.hf_config)
    model_runner = tp_worker.model_runner
    kv = KvResources(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        penalty_store=penalty_store,
        draft_token_to_kv_pool=(
            draft_worker.primary_draft_kv_pool if draft_worker is not None else None
        ),
    )
    capacity = KvCapacity(
        max_total_num_tokens=max_total_num_tokens,
        kv_shard_widening=page_interleave_shard_size(token_to_kv_pool_allocator),
        effective_max_total_num_tokens=(
            model_runner.effective_logical_max_total_num_tokens
        ),
        swa_max_total_num_tokens=(
            model_runner.swa_max_total_num_tokens if tp_worker.is_hybrid_swa else None
        ),
        max_token_pool_size=model_runner.max_token_pool_size,
        max_running_requests=max_running_requests,
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        sliding_window_size=sliding_window_size,
        full_tokens_per_layer=full_tokens_per_layer,
        swa_tokens_per_layer=swa_tokens_per_layer,
    )
    if (c := tp_worker.model_runner.canary_manager) is not None:
        c.attach_radix_cache(tree_cache)
    mode_owners = ModeOwners()
    init_hisparse_coordinator(
        mode_owners=mode_owners, tp_worker=tp_worker, forward_stream=forward_stream
    )
    if (
        get_disagg().disaggregation_mode == "decode"
        and get_disagg().disaggregation_decode_enable_offload_kvcache
    ):
        decode_offload_manager = DecodeKVCacheOffloadManager(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tp_group=dp_tp_cpu_group,
            tree_cache=tree_cache,
        )
    else:
        decode_offload_manager = None

    # Scheduling state and policy
    lists = CoreLists()
    # The launched steps whose results are not committed yet.
    pending = PendingSteps()
    session_controller = SessionController(tree_cache)
    chunked_prefill_size = resolve_chunked_prefill_size(model_config)
    is_mixed_chunk = (
        chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
    )
    dynamic_chunk_sizer = maybe_init_dynamic_chunk_sizer(
        model_runner=tp_worker.model_runner,
        model_config=model_config,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        spec_algorithm=spec_algorithm,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        device=device,
    )
    metrics_sink = MetricsSink(
        metrics_collector_context=metrics_collector_context,
        metrics_collector=metrics_collector,
        device=device,
    )
    scheduler_stage_metrics = metrics_sink.scheduler_stage_metrics
    # Init schedule policy and new token estimation
    policy = SchedulePolicy(
        get_schedule().schedule_policy,
        tree_cache,
        get_memory().enable_hierarchical_cache,
        get_schedule().enable_priority_scheduling,
        get_schedule().schedule_low_priority_values_first,
    )
    prefill_delayer = create_prefill_delayer(
        metrics_collector=(metrics_collector if metrics_sink.enable_metrics else None),
    )
    new_token_ratio_tracker = NewTokenRatioTracker.from_config()
    watchdog = SchedulerWatchdog(
        counters=counters,
        flags=flags,
        watchdog_timeout=get_device().watchdog_timeout,
    )
    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=get_exec().features.enable_memory_saver
    )
    recv_skipper = SchedulerRecvSkipper.maybe_create()
    if envs.SGLANG_LOG_GC.get():
        configure_gc_logger()
    profiler_manager = SchedulerProfilerManager(
        counters=counters,
    )

    # Start the embedded Rust frontend (rank 0). Must precede
    # init_disaggregation: on PD prefill the rust api listener also serves
    # the KV bootstrap registry, and the KVManagers built there register to
    # it synchronously. (The listener is bound synchronously inside launch,
    # so the registry is accepting once this returns.) Must also precede
    # the request receiver, which takes the rust server as its ingress.
    rust_server = None
    if _hosts_rust_server():
        rust_server = component_classes.rust_server.launch(
            server_args=server_args,
            model_config=model_config,
            processor=processor,
            tokenizer_path=get_serving().tokenizer_path,
            max_total_num_tokens=max_total_num_tokens,
            load_snapshot_writer=load_snapshot_writer,
        )
        # Park the idle loop on the request ring within the rank-0 rust-server
        idle_sleeper = RustServerIdleSleeper(rust_server, flags=flags)
    disaggregation_mode = init_disaggregation()
    # The receiver of encoder embeddings in EPD language-only mode.
    mm_receiver = None
    if get_disagg().language_only and get_disagg().encoder_transfer_backend in [
        "zmq_to_scheduler",
        "mooncake",
    ]:
        mm_receiver = create_mm_receiver(
            server_args,
            dtype=model_config.dtype,
            hf_config=model_config.hf_config,
            scheduler_side=MMSchedulerSide(
                model_config=model_config,
                tokenizer=tokenizer,
                metrics_sink=metrics_sink,
            ),
        )

    # Streams: the schedule stream settles before anything that waits on it.
    device_module = torch.get_device_module(device)
    future_map = create_future_map(
        spec_algorithm=spec_algorithm,
        device=device,
        req_to_token_pool=req_to_token_pool,
        tp_worker=tp_worker,
        draft_worker=draft_worker,
    )
    model_worker.attach_future_map(
        future_map,
        enable_overlap=(not get_schedule().disable_overlap_schedule),
    )
    # copy_stream is also used by PP (non-overlap) via the PP ring; init
    # unconditionally to match main.
    copy_stream: CudaStream = device_module.Stream()
    schedule_stream = finalize_schedule_stream(
        schedule_stream=schedule_stream,
        forward_stream=forward_stream,
        device=device,
        device_module=device_module,
    )
    streams = Streams(
        device=device,
        device_module=device_module,
        schedule_stream=schedule_stream,
        forward_stream=forward_stream,
        copy_stream=copy_stream,
    )
    # The forward a speculative step runs under pipeline parallelism.
    pp_spec_forward = (
        PPSpecForward(
            kv=kv,
            streams=streams,
            future_map=future_map,
            workers=workers,
        )
        if get_parallel().pp_size > 1 and not spec_algorithm.is_none()
        else None
    )
    ngram_embedding_manager = tp_worker.model_runner.ngram_embedding_manager
    truncation_align_size = resolve_truncation_align_size(
        hf_config=model_config.hf_config,
        chunked_prefill_size=chunked_prefill_size,
        disaggregation_mode=disaggregation_mode,
    )
    limits = RequestLimits(
        max_req_len=max_req_len,
        max_req_input_len=max_req_input_len,
        max_queued_requests=max_queued_requests,
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=chunked_prefill_size,
        is_mixed_chunk=is_mixed_chunk,
        truncation_align_size=truncation_align_size,
    )

    # Components, each built after the ones it takes
    lora_admission = LoraAdmission(workers=workers)
    grammar_manager = GrammarManager(
        server_args=server_args,
        tokenizer=tokenizer,
        model_config=model_config,
        dp_tp_group=dp_tp_group,
        pp_group=pp_group,
    )
    scripted_scheduler_hook = None
    if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
        from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook

        # The scheduler attaches itself once it holds its parts.
        scripted_scheduler_hook = ScriptedSchedulerHook(
            tokenizer_recv_proxy=ipc_channels.recv_from_tokenizer,
        )
    output_streamer = component_classes.output_streamer(
        send_to_detokenizer=ipc_channels.send_to_detokenizer,
        send_to_tokenizer=ipc_channels.send_to_tokenizer,
        tree_cache=tree_cache,
        server_args=server_args,
        is_generation=is_generation,
        spec_algorithm=spec_algorithm,
        rust_server=rust_server,
    )
    request_receiver = SchedulerRequestReceiver(
        # The rust server *is* the ingress source: SchedulerRequestReceiver
        # drains its request ring (rust_server_mode) instead of a zmq socket.
        recv_from_tokenizer=(
            rust_server if rust_server is not None else ipc_channels.recv_from_tokenizer
        ),
        recv_from_rpc=ipc_channels.recv_from_rpc,
        recv_skipper=recv_skipper,
        input_blocker=SchedulerInputBlocker.maybe_create(),
        mm_receiver=mm_receiver,
        server_args=server_args,
        model_config=model_config,
        max_recv_per_poll=envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get(),
        output_streamer=output_streamer,
        scripted_scheduler_hook=scripted_scheduler_hook,
        scheduler_stage_metrics=scheduler_stage_metrics,
    )
    step_builder = StepBuilder(
        kv=kv,
        future_map=future_map,
        counters=counters,
        workers=workers,
        hisparse_coordinator=mode_owners.hisparse_coordinator,
        model_config=model_config,
        device=device,
    )
    # Spec workers have no .model_runner of their own; the prefill graph
    # runner that votes belongs to the target model.
    target_worker = (
        tp_worker.target_worker if isinstance(tp_worker, BaseSpecWorker) else tp_worker
    )
    dp_attn_adapter = SchedulerDPAttnAdapter(
        model_runner=target_worker.model_runner,
        flags=flags,
        spec_algorithm=spec_algorithm,
        step_builder=step_builder,
    )
    pool_stats_observer = SchedulerPoolStatsObserver(
        kv=kv,
        session_controller=session_controller,
        hisparse_coordinator=mode_owners.hisparse_coordinator,
        capacity=capacity,
    )
    kv_events_publisher = SchedulerKvEventsPublisher(
        kv=kv,
        send_metrics_from_scheduler=ipc_channels.send_metrics_from_scheduler,
        capacity=capacity,
        metrics_sink=metrics_sink,
    )
    # Router-facing load reporting; rank gating and no-op fallback live
    # inside the component. Same interval as the DP-balancing writer so
    # the two fire in phase and the load sink always reuses that snapshot
    # instead of walking the queues itself.
    load_publisher = SchedulerLoadPublisher(
        kv_events_config=get_observability().kv_events_config,
        load_publish_endpoint=get_observability().load_publish_endpoint,
        publish_interval=get_observability().load_snapshot_publish_interval,
    )
    services = SchedulerServices(
        metrics_sink=metrics_sink,
        output_streamer=output_streamer,
        scheduler_stage_metrics=scheduler_stage_metrics,
    )
    beam_coordinator = BeamCoordinator(
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        max_req_len=max_req_len,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        future_map=future_map,
        core_lists=lists,
    )
    request_retirer = RequestRetirer(
        kv=kv,
        beam_coordinator=beam_coordinator,
        hisparse_coordinator=mode_owners.hisparse_coordinator,
        decode_offload_manager=decode_offload_manager,
        pending=pending,
        output_streamer=output_streamer,
        ipc_channels=ipc_channels,
    )
    batch_result_processor = SchedulerBatchResultProcessor(
        is_generation=is_generation,
        model_config=model_config,
        kv=kv,
        services=services,
        decode_offload_manager=decode_offload_manager,
        metrics_collector=metrics_collector,
        workers=workers,
        logprob_result_processor=SchedulerLogprobResultProcessor(
            model_config=model_config
        ),
        beam_coordinator=beam_coordinator,
        pending=pending,
        request_retirer=request_retirer,
    )
    request_registry = RequestRegistry(
        core_lists=lists,
        grammar_manager=grammar_manager,
        request_retirer=request_retirer,
        ipc_channels=ipc_channels,
        kv=kv,
        mm_receiver=mm_receiver,
        pending=pending,
        decode_offload_manager=decode_offload_manager,
        mode_owners=mode_owners,
    )
    invariant_checker = SchedulerInvariantChecker(
        capacity=capacity,
        kv=kv,
        pool_stats_observer=pool_stats_observer,
        counters=counters,
        request_registry=request_registry,
        pending=pending,
        scheduler_stage_metrics=scheduler_stage_metrics,
    )
    # The watchdogs dump the pools once the invariant checker exists.
    watchdog.invariant_checker = invariant_checker
    if soft_watchdog is not None:
        soft_watchdog.invariant_checker = invariant_checker
    if get_exec().dllm.dllm_algorithm is None:
        mode_owners.dllm = None
    else:
        mode_owners.dllm = DllmManager(
            dllm_config=DllmConfig.from_server_args(server_args),
            kv=kv,
            services=services,
            policy=policy,
            new_token_ratio_tracker=new_token_ratio_tracker,
            limits=limits,
            context_len=model_config.context_len,
            beam_coordinator=beam_coordinator,
            step_builder=step_builder,
            request_retirer=request_retirer,
        )
    step_counters = StepCounters()
    parts = SchedulerParts(
        counters=counters,
        flags=flags,
        lists=lists,
        pending=pending,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_generation=is_generation,
        require_mlp_sync=requires_mlp_sync,
        ipc_channels=ipc_channels,
        load_snapshot_writer=load_snapshot_writer,
        idle_sleeper=idle_sleeper,
        rust_server=rust_server,
        workers=workers,
        external_corpus_manager=external_corpus_manager,
        limits=limits,
        kv_event_lora_names=kv_event_lora_names,
        kv=kv,
        capacity=capacity,
        disable_radix_cache=disable_radix_cache,
        mode_owners=mode_owners,
        decode_offload_manager=decode_offload_manager,
        session_controller=session_controller,
        metrics_sink=metrics_sink,
        policy=policy,
        new_token_ratio_tracker=new_token_ratio_tracker,
        soft_watchdog=soft_watchdog,
        watchdog=watchdog,
        recv_skipper=recv_skipper,
        profiler_manager=profiler_manager,
        future_map=future_map,
        streams=streams,
        pp_spec_forward=pp_spec_forward,
        ngram_embedding_manager=ngram_embedding_manager,
        lora_admission=lora_admission,
        grammar_manager=grammar_manager,
        scripted_scheduler_hook=scripted_scheduler_hook,
        output_streamer=output_streamer,
        request_receiver=request_receiver,
        step_builder=step_builder,
        dp_attn_adapter=dp_attn_adapter,
        pool_stats_observer=pool_stats_observer,
        kv_events_publisher=kv_events_publisher,
        load_publisher=load_publisher,
        services=services,
        beam_coordinator=beam_coordinator,
        batch_result_processor=batch_result_processor,
        request_retirer=request_retirer,
        step_counters=step_counters,
    )

    mode = build_mode(
        parts,
        schedules,
        request_registry.lora_ids_in_use,
        disaggregation_mode=disaggregation_mode,
    )

    # The rest of the hub over the mode owners
    request_intake = component_classes.request_intake(
        core_lists=lists,
        counters=counters,
        kv=kv,
        capacity=capacity,
        ipc_channels=ipc_channels,
        services=services,
        beam_coordinator=beam_coordinator,
        session_controller=session_controller,
        grammar_manager=grammar_manager,
        request_retirer=request_retirer,
        mode_owners=mode_owners,
        model_config=model_config,
        tokenizer=tokenizer,
        mm_processor=mm_processor,
        workers=workers,
        dp_tp_group=dp_tp_group,
        limits=limits,
        kv_event_lora_names=kv_event_lora_names,
    )
    step_planner = StepPlanner(
        core_lists=lists,
        pending=pending,
        flags=flags,
        counters=counters,
        kv=kv,
        capacity=capacity,
        policy=policy,
        new_token_ratio_tracker=new_token_ratio_tracker,
        pool_stats_observer=pool_stats_observer,
        lora_admission=lora_admission,
        request_registry=request_registry,
        beam_coordinator=beam_coordinator,
        step_builder=step_builder,
        request_retirer=request_retirer,
        grammar_manager=grammar_manager,
        metrics_sink=metrics_sink,
        ipc_channels=ipc_channels,
        workers=workers,
        decode_offload_manager=decode_offload_manager,
        mode_owners=mode_owners,
        request_intake=request_intake,
        prefill_delayer=prefill_delayer,
        dynamic_chunk_sizer=dynamic_chunk_sizer,
        limits=limits,
        prefill_adder_class=component_classes.prefill_adder,
    )
    # Built after the request registry and the mode owners, which they read.
    weight_updater = SchedulerWeightUpdaterManager(
        workers=workers,
        memory_saver_adapter=memory_saver_adapter,
        flush_cache=step_planner.flush_cache,
        request_registry=request_registry,
        flags=flags,
        mode_owners=mode_owners,
        metrics_collector=metrics_collector,
    )
    flush_wrapper = SchedulerFlushWrapper(
        flush_cache=step_planner.flush_cache,
        request_registry=request_registry,
        ipc_channels=ipc_channels,
    )
    load_inquirer = SchedulerLoadInquirer(
        capacity=capacity,
        pool_stats_observer=pool_stats_observer,
        workers=workers,
        kv=kv,
        policy=policy,
        metrics_sink=metrics_sink,
        request_registry=request_registry,
        mode_owners=mode_owners,
        step_counters=step_counters,
    )
    metrics_reporter = SchedulerMetricsReporter(
        metrics_sink=metrics_sink,
        request_registry=request_registry,
        counters=counters,
        pool_stats_observer=pool_stats_observer,
        grammar_manager=grammar_manager,
        mm_receiver=mm_receiver,
        kv_events_publisher=kv_events_publisher,
        mode_owners=mode_owners,
        kv=kv,
        capacity=capacity,
        workers=workers,
        model_config=model_config,
        device=device,
    )
    groups = []
    if attn_cp_group is not None and attn_tp_group is not None:
        groups += [attn_cp_group, attn_tp_group]
    else:
        groups += [tp_group]
    if pp_group is not None:
        groups += [pp_group]
    rank_consensus_checker.configure(groups)

    flags.is_initializing = False
    startup_time = build_scheduler_startup_time(
        target_load_weight=tp_worker.weight_load_time,
        draft_load_weight=(
            0.0 if draft_worker is None else draft_worker.weight_load_time
        ),
        kv_cache_allocation=kv_cache_allocation_time,
        scheduler_e2e=time.perf_counter() - scheduler_startup_begin,
        target_cuda_graph=tp_worker.graph_time_usage,
        draft_cuda_graph=(
            None if draft_worker is None else draft_worker.graph_time_usage
        ),
    )
    # Reads the startup time.
    internal_state = SchedulerInternalStateController(
        workers=workers,
        kv=kv,
        capacity=capacity,
        startup_available_gpu_memory_gb=startup_available_gpu_memory_gb,
        metrics_sink=metrics_sink,
        rust_server=rust_server,
        request_registry=request_registry,
        startup_time=startup_time,
        record_step_time=RECORD_STEP_TIME,
    )
    hub = SchedulerHub(
        request_registry=request_registry,
        invariant_checker=invariant_checker,
        request_intake=request_intake,
        step_planner=step_planner,
        weight_updater=weight_updater,
        flush_wrapper=flush_wrapper,
        load_inquirer=load_inquirer,
        metrics_reporter=metrics_reporter,
        internal_state=internal_state,
    )
    return parts, mode, hub


def build_mode(
    parts: SchedulerParts,
    schedules: LoopSchedules,
    lora_ids_in_use: Callable[[], Set[Optional[str]]],
    *,
    disaggregation_mode: DisaggregationMode,
) -> SchedulerMode:
    """Build the parts of a disaggregation mode, in order: its PD owner (into
    `ModeOwners`), the unified-memory move gate, its stages and the step
    launcher that holds them."""
    maybe_init_disagg_prefill(parts, disaggregation_mode=disaggregation_mode)
    maybe_init_disagg_decode(parts, disaggregation_mode=disaggregation_mode)
    maybe_init_disagg_move_gate(parts)
    stages = init_mode_stages(parts, schedules, lora_ids_in_use)
    return SchedulerMode(
        disaggregation_mode=disaggregation_mode,
        stages=stages,
        step_launcher=init_step_launcher(parts, stages),
    )


def rebuild_mode(
    parts: SchedulerParts,
    schedules: LoopSchedules,
    lora_ids_in_use: Callable[[], Set[Optional[str]]],
) -> SchedulerMode:
    """Build the mode again for the role a PD role switch set in config; the
    old PD owner is torn down already."""
    return build_mode(
        parts,
        schedules,
        lora_ids_in_use,
        disaggregation_mode=init_disaggregation(),
    )


def build_pp_ring(parts: SchedulerParts, stages: ModeStages, ops: DriverOps) -> PPRing:
    """Build the ring driver and its PP channel, once per ring loop."""
    channel = PPChannel(streams=parts.streams)
    return PPRing(
        channel=channel,
        stages=stages,
        pending=parts.pending,
        recv_skipper=parts.recv_skipper,
        ops=ops,
        counters=parts.counters,
        flags=parts.flags,
        pp_spec_forward=parts.pp_spec_forward,
        kv=parts.kv,
        streams=parts.streams,
        workers=parts.workers,
        future_map=parts.future_map,
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


def init_soft_watchdog(
    *, counters: LaunchCounters, flags: EngineFlags
) -> Optional[SchedulerWatchdog]:
    soft_watchdog: Optional[SchedulerWatchdog] = None
    if (x := get_device().soft_watchdog_timeout) is not None:
        soft_watchdog = SchedulerWatchdog(
            counters=counters,
            flags=flags,
            watchdog_timeout=x,
            soft=True,
        )
    return soft_watchdog


def init_idle_sleeper(*, ipc_channels: SchedulerIpcChannels, flags: EngineFlags):
    if is_entry_rank() and get_device().sleep_on_idle:
        return IdleSleeper(
            sockets=[
                ipc_channels.recv_from_tokenizer,
                ipc_channels.recv_from_rpc,
            ],
            flags=flags,
        )
    else:
        return None


def maybe_init_external_corpus_manager(
    *, draft_worker, spec_algorithm: SpeculativeAlgorithm, ipc_channels
):
    """The external corpus of ngram speculative decoding, over its draft worker."""
    if draft_worker is None or not spec_algorithm.is_ngram():
        return None
    from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager

    return ExternalCorpusManager(
        draft_worker,
        ipc_channels.send_to_tokenizer.send_output,
    )


def init_hisparse_coordinator(
    *, mode_owners: ModeOwners, tp_worker, forward_stream
) -> None:
    mode_owners.hisparse_coordinator = None
    if not get_memory().enable_hisparse:
        return

    # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
    mode_owners.hisparse_coordinator = tp_worker.model_runner.hisparse_coordinator
    mode_owners.hisparse_coordinator.set_decode_producer_stream(forward_stream)


def maybe_init_dynamic_chunk_sizer(
    *,
    model_runner,
    model_config: ModelConfig,
    tree_cache,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    spec_algorithm: SpeculativeAlgorithm,
    chunked_prefill_size: Optional[int],
    max_prefill_tokens: int,
    device: str,
) -> Optional[DynamicChunkSizer]:
    """Profile a PP prefill latency model that sizes chunks per stage."""
    if not (get_schedule().enable_dynamic_chunking and get_parallel().pp_size > 1):
        return None
    return fit_dynamic_chunk_sizer(
        model_runner=model_runner,
        model_config=model_config,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        spec_algorithm=spec_algorithm,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        page_size=get_schedule().page_size,
        device=device,
    )


def _hosts_rust_server() -> bool:
    """Whether this scheduler rank embeds the Rust server (rank 0 only) —
    and with it the server-process duties a Python ``TokenizerManager``
    would otherwise own (e.g. serving the PD KV bootstrap registry)."""
    return envs.SGLANG_RUST_SERVER.get() and is_entry_rank()


def init_disaggregation() -> DisaggregationMode:
    """The configured disaggregation mode."""
    disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)

    # In rust-server mode the KV bootstrap registry is already serving on
    # the rust api listener (the rust server launches before this runs —
    # the PrefillBootstrapQueue's KVManager below registers to it
    # synchronously, and a failed registration only retries ~60s then logs,
    # leaving every PD request unroutable). Only the ascend config store,
    # which start_disagg_service would otherwise create, is left to do.
    if disaggregation_mode == DisaggregationMode.PREFILL and _hosts_rust_server():
        maybe_create_ascend_config_store(
            transfer_backend=TransferBackend(
                get_disagg().disaggregation_transfer_backend
            ),
        )
    return disaggregation_mode


def finalize_schedule_stream(*, schedule_stream, forward_stream, device, device_module):
    """Settle the schedule stream the event loop runs on, before any
    component that waits on it is built; it does not change afterwards."""
    if device == "cpu":
        schedule_stream.synchronize = lambda: None  # No-op for CPU
    elif (is_cuda() or _is_hip) and (
        (not get_schedule().disable_overlap_schedule) or get_parallel().pp_size > 1
    ):
        # CUDA/HIP streams come from a fixed round-robin pool. Redraw if this
        # stream aliases forward_stream, which would eliminate scheduler
        # overlap. Only CUDA/HIP streams expose a ``cuda_stream`` handle;
        # other accelerators (e.g. NPU/XPU) skip the alias check.
        if schedule_stream.cuda_stream == forward_stream.cuda_stream:
            schedule_stream = allocate_distinct_stream(device_module, (forward_stream,))
    return schedule_stream


def maybe_init_disagg_prefill(
    parts: SchedulerParts, *, disaggregation_mode: DisaggregationMode
) -> None:
    if disaggregation_mode != DisaggregationMode.PREFILL:
        parts.mode_owners.disagg_prefill = None
        return
    parts.mode_owners.disagg_prefill = DisaggPrefill(
        server_args=parts.server_args,
        model_config=parts.model_config,
        kv=parts.kv,
        capacity=parts.capacity,
        streams=parts.streams,
        services=parts.services,
        workers=parts.workers,
        rust_http_port=(
            parts.rust_server.http_port if parts.rust_server is not None else None
        ),
        batch_result_processor=parts.batch_result_processor,
        counters=parts.counters,
        request_retirer=parts.request_retirer,
        core_lists=parts.lists,
    )


def maybe_init_disagg_decode(
    parts: SchedulerParts, *, disaggregation_mode: DisaggregationMode
) -> None:
    if disaggregation_mode != DisaggregationMode.DECODE:
        parts.mode_owners.disagg_decode = None
        return
    parts.mode_owners.disagg_decode = DisaggDecode(
        server_args=parts.server_args,
        model_config=parts.model_config,
        kv=parts.kv,
        capacity=parts.capacity,
        streams=parts.streams,
        services=parts.services,
        workers=parts.workers,
        hisparse_coordinator=parts.mode_owners.hisparse_coordinator,
        lora_admission=parts.lora_admission,
        pool_stats_observer=parts.pool_stats_observer,
        decode_offload_manager=parts.decode_offload_manager,
        request_retirer=parts.request_retirer,
        flags=parts.flags,
        core_lists=parts.lists,
        policy=parts.policy,
        future_map=parts.future_map,
        ngram_embedding_manager=parts.ngram_embedding_manager,
    )


def maybe_init_disagg_move_gate(parts: SchedulerParts) -> None:
    """Unified-memory compaction on a PD node waits for the owner's KV
    transfers; every page mover runs on the scheduler thread."""
    owner = parts.mode_owners.disagg_prefill or parts.mode_owners.disagg_decode
    if get_memory().enable_unified_memory and owner is not None:
        parts.kv.token_to_kv_pool_allocator.set_disagg_move_gate(
            owner.unified_memory_move_gate
        )


def init_mode_stages(
    parts: SchedulerParts,
    schedules: LoopSchedules,
    lora_ids_in_use: Callable[[], Set[Optional[str]]],
) -> ModeStages:
    """Choose the mode's stages, once; a PD role switch rebuilds them. The PP
    ring and the flat driver differ only where pp_size decides here. PD
    decode admission takes the adapters the scheduler's requests pin."""
    owners = parts.mode_owners
    hisparse = owners.hisparse_coordinator
    if owners.dllm is not None:
        stages = dict(
            commit_extend=owners.dllm.commit_extend,
            admit=schedules.admit_dllm,
            select_decode=schedules.select_decode,
            on_launch=owners.dllm.on_launch,
        )
    elif (owner := owners.disagg_prefill) is not None:
        under_pp = get_parallel().pp_size > 1
        stages = dict(
            commit_extend=owner.commit_extend,
            intake=owner.intake,
            # The resolve poll reduces over TP only, so under PP it could
            # split the stages; optimistic prefill is off under PP anyway.
            resolve_waiting=None if under_pp else owner.resolve_waiting_bootstrap,
            advance_chunked=owner.process_prefill_chunk,
            admit=schedules.admit_prefill,
            before_forward=owner.before_forward,
            on_prefilled=owner.on_launch,
            after_commit=owner.after_commit,
            # Read only by the overlap flat driver.
            is_result_ready=(
                owner.is_result_ready
                if (not get_schedule().disable_overlap_schedule)
                and not under_pp
                and continuous_input_polling_enabled(
                    spec_algorithm=parts.spec_algorithm,
                    is_generation=parts.is_generation,
                )
                else None
            ),
            ring_consensus=functools.partial(
                _DisaggPrefillRing,
                owner,
                intake=schedules.ring_intake_disagg_prefill,
                release=schedules.ring_release_disagg_prefill,
            ),
        )
    elif (owner := owners.disagg_decode) is not None:
        under_pp = get_parallel().pp_size > 1
        stages = dict(
            commit_extend=_commit_prefill(parts, hisparse),
            prefetch=owner.prefetch_prefill_dp_rank_queries,
            drain_hicache=owner.check_hicache_events,
            # Under PP the ring's consensus messages advance the lists instead.
            before_admission=(
                None
                if under_pp
                else functools.partial(owner.intake, lora_ids_in_use=lora_ids_in_use)
            ),
            join_ready=owner.join_ready,
            select_decode=schedules.select_decode,
            hold_finished=(
                owner.hold_finished
                if parts.decode_offload_manager is not None
                else None
            ),
            ring_consensus=functools.partial(
                _DisaggDecodeRing,
                owner,
                lora_ids_in_use=lora_ids_in_use,
            ),
        )
    else:
        stages = dict(
            commit_extend=_commit_prefill(parts, hisparse),
            admit=schedules.admit_prefill_after_interval,
            select_decode=schedules.select_decode,
        )
    if hisparse is not None:
        # Launched prefills stage until their host copy lands (PD prefill
        # hands them over instead); the normal schedule then joins them.
        stages.setdefault("on_prefilled", hisparse.on_launch)
        stages["leave_finished"] = hisparse.leave_finished
        stages["after_decode_commit"] = hisparse.free_remapped_decode_slots
        if owners.disagg_prefill is None and owners.disagg_decode is None:
            stages["join_ready"] = functools.partial(
                hisparse.join_ready,
                future_map=parts.future_map,
                device=parts.streams.device,
            )
    return ModeStages(**stages)


def _commit_prefill(
    parts: SchedulerParts, hisparse: Optional[HiSparseCoordinator]
) -> Callable:
    """The prefill result processor's extend commit; with HiSparse a concluded
    request stages until its host copy lands."""
    commit = parts.batch_result_processor.process_batch_result_prefill
    if hisparse is None:
        return commit
    return functools.partial(commit, on_checkpoint=hisparse.admit_request_into_staging)


def continuous_input_polling_enabled(
    *, spec_algorithm: SpeculativeAlgorithm, is_generation: bool
) -> bool:
    """Check support once, when the PD prefill stages are built."""
    if not envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.get():
        return False
    parallel = get_parallel()
    enabled = (
        parallel.attn_dp_size == 1
        and parallel.attn_cp_size == 1
        and spec_algorithm.is_none()
        and is_generation
    )
    if not enabled:
        logger.warning(
            "Continuous prefill input polling requires attention DP=CP=1 and "
            "non-speculative autoregressive generation; using the regular overlap loop."
        )
    return enabled


def init_step_launcher(parts: SchedulerParts, stages: ModeStages) -> StepLauncher:
    return StepLauncher(
        kv=parts.kv,
        streams=parts.streams,
        future_map=parts.future_map,
        counters=parts.counters,
        flags=parts.flags,
        pending=parts.pending,
        core_lists=parts.lists,
        ipc_channels=parts.ipc_channels,
        services=parts.services,
        profiler_manager=parts.profiler_manager,
        beam_coordinator=parts.beam_coordinator,
        pp_spec_forward=parts.pp_spec_forward,
        stages=stages,
        scripted_scheduler_hook=parts.scripted_scheduler_hook,
        workers=parts.workers,
        is_generation=parts.is_generation,
    )
