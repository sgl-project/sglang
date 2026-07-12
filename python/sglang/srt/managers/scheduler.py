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
import hashlib
import logging
import numbers
import signal
import sys
import time
from array import array
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation.decode import (
    DecodeRequest,
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.flip_state_machine import (
    ClusterSnapshot,
    FlipDecision,
    FlipDirection,
    FlipEvent,
    FlipState,
    FlipStateMachine,
    FlipTransition,
    SLOThresholdFlipEvaluator,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
    get_attention_cp_group,
    get_attention_tp_group,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
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
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
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
    PDFlipMigrationAbortReq,
    PDFlipMigrationReqOutput,
    PDFlipMigrationSourceDeltaReq,
    PDFlipMigrationSourceFinishReq,
    PDFlipMigrationSourceStartReq,
    PDFlipMigrationStatusReq,
    PDFlipMigrationTargetActivateReq,
    PDFlipMigrationTargetAbortReq,
    PDFlipMigrationTargetCommitReq,
    PDFlipMigrationTargetDeltaPrepareReq,
    PDFlipMigrationTargetPrepareReq,
    PDRuntimeRoleAdmissionReq,
    PDRuntimeRoleReqOutput,
    PDRuntimeRoleSetReq,
    PDRuntimeRoleStatusReq,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
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
)
from sglang.srt.managers.load_snapshot import LoadSnapshot, create_load_snapshot_writer
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MIGRATED,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.dflash_utils import (
    resolve_dflash_prefill_refill_target,
    should_delay_dflash_prefill_for_batching,
    validate_dflash_request,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_mps,
    kill_itself_when_parent_died,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin
else:
    from torch.cuda import StreamContext as CudaStreamContext

    class SchedulerMlxOverlapMixin:
        pass


logger = logging.getLogger(__name__)
PD_FLIP_MIN_METADATA_BUFFER_SIZE = 1024

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()


class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerPPMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        self.forward_ct: int = 0
        self.cur_batch: Optional[ScheduleBatch] = None
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
        self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.enable_decode_hicache = (
            server_args.disaggregation_decode_enable_radix_cache
            and self.enable_hierarchical_cache
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.enable_hisparse = server_args.enable_hisparse
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None

        # Distributed rank info
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                tp_rank,
                server_args.tp_size,
                server_args.dp_size,
                server_args.attn_cp_size,
            )
        )
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            dp_rank=dp_rank,
            dp_size=server_args.dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=server_args.attn_cp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=server_args.moe_dp_size,
            gpu_id=gpu_id,
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

        # Init inter-process communication
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

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
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=self.server_args.enable_metrics,
            enable_kv_cache_events=bool(
                self.server_args.kv_events_config
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
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

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        if self.enable_hisparse:
            # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture
            self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

        if (
            self.server_args.disaggregation_mode == "decode"
            and self.server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=(
                    self.attn_tp_cpu_group
                    if self.server_args.enable_dp_attention
                    else self.tp_cpu_group
                ),
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        # Register draft KV pool (when spec + HiCache co-enabled).
        kv_cache_builder.maybe_register_hicache_draft(
            tree_cache=self.tree_cache,
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            page_size=self.page_size,
        )

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()
        self.init_pd_flip_state_machine()

        # Init overlap schedule
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

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

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()

        self.init_kv_events_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()

        self.init_batch_result_processor()

        self.is_initializing = False

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if self.ps.pp_size > 1:
                logger.error(f"only zbal mix mode support pp_size > 1!")
            init_zbal(
                self.ps.tp_size, self.ps.gpu_id, self.ps.tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )
            if self.dllm_config:
                if self.dllm_config.block_size < self.page_size:
                    logger.warning(
                        "WARNING: "
                        f"The page size {self.page_size} should not be larger than dllm block size {self.dllm_config.block_size}."
                        f"Page size now falls back to {self.dllm_config.block_size}"
                    )
                    self.page_size = self.dllm_config.block_size

    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            skip_tokenizer_init=self.server_args.skip_tokenizer_init,
            metrics_enabled=self.server_args.enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or self.server_args.enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        if not is_rank_zero:
            return

        dp_rank = self.ps.dp_rank if self.ps.dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                self.server_args,
                port_args,
                self.ps.dp_size,
                dp_rank,
                publish_interval=self.server_args.load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        if (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
            and self.server_args.sleep_on_idle
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        writer = self.load_snapshot_writer
        if writer is None:
            return
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return
        writer.publish_counter = 0
        try:
            result = self.load_inquirer.get_loads(GetLoadsReqInput(include=["all"]))
            writer.write(LoadSnapshot.from_get_loads_output(result))
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)

    def handle_get_loads_req(self, req: GetLoadsReqInput):
        return self.load_inquirer.get_loads(req)

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if self.model_config.is_multimodal and self.processor is not None:
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

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.model_config.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            pp_rank=self.ps.pp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
            dp_rank=self.ps.dp_rank,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.ps.dp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
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

    def init_model_worker(self):
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
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
        ) = self.tp_worker.get_worker_info()
        self.dflash_prefill_refill_target = (
            resolve_dflash_prefill_refill_target(self.max_running_requests)
            if self.spec_algorithm.is_dflash()
            else 1
        )
        if not get_global_server_args().pp_max_micro_batch_size:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.ps.pp_size, 1
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # TODO(Jialin): Migrate pad_input_ids implementations to return array.
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        avail_mem = get_available_gpu_memory(
            self.device, self.ps.gpu_id, empty_cache=False
        )
        if self.ps.tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.server_args.enable_metrics:
            self.metrics_collector.emit_constants(
                max_total_num_tokens=self.max_total_num_tokens,
                # TODO: max_running_requests_under_SLO has no setter — dead chain.
                max_running_requests_under_SLO=getattr(
                    self, "max_running_requests_under_SLO", None
                ),
                engine_startup_time=0.0,
                engine_load_weights_time=0.0,
                page_size=self.page_size,
                num_pages=self.max_total_num_tokens // self.page_size,
                context_len=self.model_config.context_len,
                startup_available_gpu_memory_gb=avail_mem,
            )

    def init_running_status(self):
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False
        self.pd_flip_quiesce_requested = False
        self.pd_flip_batch_quiesced = False
        self.pd_flip_quiesce_rids: Tuple[str, ...] = ()
        self.pd_flip_quiesce_session_id: Optional[str] = None

    def init_chunked_prefill(self):
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
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
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.ps.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: int = 0
        if self.server_args.enable_prefill_delayer:
            if self.server_args.disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.ps.dp_size,
                    attn_tp_size=self.ps.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    device_group=self.tp_group.device_group,
                    server_args=self.server_args,
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_reporter.enable_metrics
                        else None
                    ),
                    max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                    device=self.tp_group.device,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_server_args(
            self.server_args
        )

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
        draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )
        self.disagg_draft_token_to_kv_pool = draft_token_to_kv_pool
        # Default to the target model_config so the MetadataBuffers branches
        # below can always access it; overridden by the draft model_config
        # when this node runs a spec module.
        if model_config is None:
            model_config = self.model_config

        if self.pd_runtime_role_switch_enabled():
            self._init_hybrid_disaggregation_metadata(model_config)
            self._init_decode_disaggregation(draft_token_to_kv_pool, model_config)
            self._init_prefill_disaggregation(draft_token_to_kv_pool, model_config)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._init_decode_disaggregation(draft_token_to_kv_pool, model_config)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._init_prefill_disaggregation(draft_token_to_kv_pool, model_config)

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def _init_pd_metadata_buffers(
        self, buffer_size: int, hidden_size: int, hidden_states_dtype
    ) -> None:
        current_allocator = getattr(self, "req_to_metadata_buffer_idx_allocator", None)
        current_size = int(getattr(current_allocator, "size", 0) or 0)
        if current_allocator is not None:
            if current_size >= buffer_size:
                return
            available_size = (
                current_allocator.available_size()
                if hasattr(current_allocator, "available_size")
                else current_size
            )
            if available_size != current_size:
                raise RuntimeError(
                    "cannot grow PD metadata buffers while metadata entries are allocated"
                )

        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
            buffer_size
        )
        custom_mem_pool = None
        kv_pool_allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if kv_pool_allocator is not None:
            custom_mem_pool = (
                kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool()
            )
        self.disagg_metadata_buffers = MetadataBuffers(
            buffer_size,
            hidden_size=hidden_size,
            hidden_states_dtype=hidden_states_dtype,
            custom_mem_pool=custom_mem_pool,
        )

    def _init_hybrid_disaggregation_metadata(self, model_config) -> None:
        buffer_size = max(
            max(self.req_to_token_pool.size, self.max_running_requests) * 2,
            PD_FLIP_MIN_METADATA_BUFFER_SIZE,
        )
        hidden_size = (
            model_config.spec_hidden_size
            if self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            else 16  # minimal padding size for RDMA
        )
        hidden_states_dtype = (
            model_config.dtype
            if self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            else torch.float32
        )
        self._init_pd_metadata_buffers(buffer_size, hidden_size, hidden_states_dtype)

    def _init_decode_disaggregation(self, draft_token_to_kv_pool, model_config) -> None:
        buffer_size = self.req_to_token_pool.size * 2
        hidden_size = (
            model_config.spec_hidden_size
            if self.spec_algorithm.is_eagle()
            else 16  # minimal padding size for RDMA
        )
        hidden_states_dtype = (
            model_config.dtype if self.spec_algorithm.is_eagle() else torch.float32
        )
        self._init_pd_metadata_buffers(buffer_size, hidden_size, hidden_states_dtype)

        # The decode requests polling kv cache
        self.disagg_decode_transfer_queue = DecodeTransferQueue(
            gloo_group=self.attn_tp_cpu_group,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            tp_rank=self.ps.tp_rank,
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            tree_cache=self.tree_cache,
        )

        # The decode requests pending for pre-allocation
        self.disagg_decode_prealloc_queue = DecodePreallocQueue(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            scheduler=self,
            transfer_queue=self.disagg_decode_transfer_queue,
            tree_cache=self.tree_cache,
            gloo_group=self.attn_tp_cpu_group,
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            dp_size=self.server_args.dp_size,
            gpu_id=self.ps.gpu_id,
            bootstrap_port=self.server_args.disaggregation_bootstrap_port,
            max_total_num_tokens=self.max_total_num_tokens,
            pp_rank=self.ps.pp_rank,
            num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
            transfer_backend=self.transfer_backend,
        )

    def _init_prefill_disaggregation(
        self, draft_token_to_kv_pool, model_config
    ) -> None:
        buffer_size = self.max_running_requests * 2
        hidden_size = (
            model_config.spec_hidden_size
            if self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            else 16  # minimal padding size for RDMA
        )
        hidden_states_dtype = (
            model_config.dtype
            if self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            else torch.float32
        )
        self._init_pd_metadata_buffers(buffer_size, hidden_size, hidden_states_dtype)

        self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
            token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
            metadata_buffers=self.disagg_metadata_buffers,
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            gpu_id=self.ps.gpu_id,
            bootstrap_port=self.server_args.disaggregation_bootstrap_port,
            gloo_group=self.attn_tp_cpu_group,
            max_total_num_tokens=self.max_total_num_tokens,
            scheduler=self,
            pp_rank=self.ps.pp_rank,
            pp_size=self.ps.pp_size,
            transfer_backend=self.transfer_backend,
        )
        # The prefill requests that are in the middle of kv sending
        self.disagg_prefill_inflight_queue: List[Req] = []

    def pd_runtime_role(self) -> str:
        return DisaggregationMode.to_engine_type(self.disaggregation_mode.value)

    def _pd_role_loop_should_exit(self, expected: DisaggregationMode) -> bool:
        return self.disaggregation_mode != expected or bool(
            getattr(self, "_shutdown_requested", False)
        )

    def pd_runtime_role_switch_enabled(self) -> bool:
        server_args = getattr(self, "server_args", None)
        return bool(getattr(server_args, "enable_pd_runtime_role_switch", False))

    def _pd_flip_capacity_status(self) -> Dict[str, Any]:
        running = list(getattr(getattr(self, "running_batch", None), "reqs", []))
        free_slots = min(
            max(0, self.max_running_requests - len(running)),
            int(self.req_to_token_pool.available_size()),
        )
        return {
            "free_request_slots": free_slots,
            "available_kv_tokens": int(
                self.token_to_kv_pool_allocator.available_size()
            ),
            "max_running_requests_per_dp": int(self.max_running_requests),
            "reserved_decode_tokens_per_req": int(
                self.server_args.num_reserved_decode_tokens
            ),
            "running_requests": [
                {
                    "rid": str(req.rid),
                    "kv_committed_len": int(req.kv_committed_len),
                }
                for req in running
                if not req.finished()
            ],
        }

    def _pd_runtime_role_status_dict(self) -> Dict[str, Any]:
        is_idle = False
        try:
            is_idle = bool(self.is_fully_idle())
        except Exception:
            logger.exception("Failed to compute PD runtime role idle status.")

        status = {
            "dp_rank": self.ps.dp_rank,
            "tp_rank": self.ps.tp_rank,
            "role": self.pd_runtime_role(),
            "active_event_loop_role": getattr(
                self, "active_pd_event_loop_role", None
            ),
            "runtime_role_switch_enabled": self.pd_runtime_role_switch_enabled(),
            "admission_paused": bool(
                getattr(self, "pd_runtime_admission_paused", False)
            ),
            "pd_flip_admission_paused": self.pd_flip_should_reject_new_work(),
            "is_idle": is_idle,
            "dual_queues_initialized": bool(
                hasattr(self, "disagg_decode_prealloc_queue")
                and hasattr(self, "disagg_decode_transfer_queue")
                and hasattr(self, "disagg_prefill_bootstrap_queue")
            ),
            "event_loop_dynamic": True,
        }
        status.update(self._pd_flip_capacity_status())
        return status

    def get_pd_runtime_role_status(
        self, recv_req: PDRuntimeRoleStatusReq
    ) -> PDRuntimeRoleReqOutput:
        status = self._pd_runtime_role_status_dict()
        return PDRuntimeRoleReqOutput(
            success=True,
            message="ok",
            role=status["role"],
            status=status,
        )

    def set_pd_runtime_admission(
        self, recv_req: PDRuntimeRoleAdmissionReq
    ) -> PDRuntimeRoleReqOutput:
        self.pd_runtime_admission_paused = bool(recv_req.paused)
        status = self._pd_runtime_role_status_dict()
        return PDRuntimeRoleReqOutput(
            success=True,
            message=(
                "PD runtime admission paused"
                if recv_req.paused
                else "PD runtime admission resumed"
            ),
            role=status["role"],
            status=status,
        )

    def set_pd_runtime_role(
        self, recv_req: PDRuntimeRoleSetReq
    ) -> PDRuntimeRoleReqOutput:
        target_mode = DisaggregationMode(recv_req.role)
        if target_mode not in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            status = self._pd_runtime_role_status_dict()
            return PDRuntimeRoleReqOutput(
                success=False,
                message=f"Invalid PD runtime role: {recv_req.role}",
                role=status["role"],
                status=status,
            )

        if not self.pd_runtime_role_switch_enabled():
            status = self._pd_runtime_role_status_dict()
            return PDRuntimeRoleReqOutput(
                success=False,
                message="PD runtime role switch is disabled.",
                role=status["role"],
                status=status,
            )

        status = self._pd_runtime_role_status_dict()
        if not recv_req.force and not status["is_idle"]:
            return PDRuntimeRoleReqOutput(
                success=False,
                message="PD runtime role switch requires an idle scheduler.",
                role=status["role"],
                status=status,
            )

        self.disaggregation_mode = target_mode
        self.server_args.disaggregation_mode = target_mode.value
        get_global_server_args().disaggregation_mode = target_mode.value
        if hasattr(self, "output_streamer"):
            self.output_streamer.disaggregation_mode = target_mode

        status = self._pd_runtime_role_status_dict()
        return PDRuntimeRoleReqOutput(
            success=True,
            message=f"PD runtime role switched to {target_mode.value}.",
            role=status["role"],
            status=status,
        )

    def init_pd_flip_state_machine(self):
        if not self.server_args.enable_pd_flip_state_machine:
            self.pd_flip_state_machine: Optional[FlipStateMachine] = None
            return

        self.pd_flip_state_machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(
                slo_threshold=self.server_args.pd_flip_slo_threshold
            ),
            prepare_flip=self.prepare_pd_flip,
            commit_flip=self.commit_pd_flip,
            min_window_seconds=self.server_args.pd_flip_window_seconds,
        )
        if self.ps.tp_rank == 0:
            logger.info(
                "PD flip state machine enabled: window=%.3fs, slo_threshold=%.3f, role=%s",
                self.server_args.pd_flip_window_seconds,
                self.server_args.pd_flip_slo_threshold,
                DisaggregationMode.to_engine_type(self.disaggregation_mode.value),
            )

    def maybe_tick_pd_flip_state_machine(self):
        if self.pd_flip_state_machine is None:
            return

        self.refresh_pd_flip_runtime_config()
        snapshot = self.build_pd_flip_snapshot()
        server_args = get_global_server_args()
        if getattr(server_args, "pd_flip_abort", False):
            event = self.pd_flip_state_machine.abort("external orchestrator abort")
            setattr(server_args, "pd_flip_abort", False)
        else:
            event = self.pd_flip_state_machine.tick(snapshot)
        self.log_pd_flip_event(event, snapshot)

    def refresh_pd_flip_runtime_config(self):
        server_args = get_global_server_args()
        self.pd_flip_state_machine.min_window_seconds = (
            server_args.pd_flip_window_seconds
        )
        evaluator = self.pd_flip_state_machine.evaluator
        if hasattr(evaluator, "slo_threshold"):
            evaluator.slo_threshold = server_args.pd_flip_slo_threshold

    def pd_flip_should_reject_new_work(self) -> bool:
        if getattr(self, "pd_runtime_admission_paused", False):
            return True
        machine = getattr(self, "pd_flip_state_machine", None)
        return machine is not None and machine.state in (
            FlipState.PREPARING,
            FlipState.FLIPPING,
        )

    def reject_pd_flip_admission(self, req: Req):
        error_message = (
            "PD role flip is draining this worker; retry through the router "
            "or another worker."
        )
        prepare_abort(
            req,
            error_message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
        self.output_streamer.stream_output(
            [req], getattr(req, "return_logprob", False)
        )

    def pd_flip_is_idle_for_commit(self, snapshot: ClusterSnapshot) -> bool:
        if self._pd_flip_source_migration_blocks_idle():
            return False
        snapshot_idle = (
            snapshot.waiting_reqs == 0
            and snapshot.running_reqs == 0
            and snapshot.prefill_bootstrap_reqs == 0
            and snapshot.prefill_inflight_reqs == 0
            and snapshot.decode_prealloc_reqs == 0
            and snapshot.decode_transfer_reqs == 0
        )
        if not snapshot_idle:
            return False

        is_fully_idle = getattr(self, "is_fully_idle", None)
        if not callable(is_fully_idle):
            return True
        return bool(is_fully_idle())

    def get_pd_flip_internal_state(self) -> Dict[str, Any]:
        role = DisaggregationMode.to_engine_type(self.disaggregation_mode.value)
        if self.pd_flip_state_machine is None:
            return {
                "enabled": False,
                "current_role": role,
                "can_hot_switch_in_process": False,
            }

        status = self.pd_flip_state_machine.status()
        status["enabled"] = True
        status["current_role"] = role
        status["implementation_stage"] = "decision_and_observability"
        status["role_mutation_wired"] = False
        status["active_decode_migration_wired"] = False
        status["requires_process_restart"] = status["direction"] != "none"
        status["drain_to_idle_required"] = status["direction"] != "none"
        status["admission_paused"] = self.pd_flip_should_reject_new_work()
        try:
            snapshot = self.pd_flip_state_machine.last_snapshot
            if snapshot is None:
                snapshot = self.build_pd_flip_snapshot()
            status["is_idle_for_flip"] = self.pd_flip_is_idle_for_commit(snapshot)
        except Exception:
            status["is_idle_for_flip"] = False
        server_args = get_global_server_args()
        status["external_prepare_ack"] = getattr(
            server_args, "pd_flip_prepare_ack", False
        )
        status["external_commit_ack"] = getattr(server_args, "pd_flip_commit_ack", False)
        status["external_abort_requested"] = getattr(server_args, "pd_flip_abort", False)
        migration_status = self._pd_flip_migration_status_dict()
        status["migration_enabled"] = migration_status["enabled"]
        status["migration_role"] = migration_status["role"]
        status["migration_state"] = migration_status["state"]
        status["migration_session_id"] = migration_status["session_id"]
        status["migration_pending_reqs"] = migration_status["pending_reqs"]
        status["migration_transferred_reqs"] = migration_status["transferred_reqs"]
        status["migration_released_reqs"] = migration_status["released_reqs"]
        status["migration_failed_reqs"] = migration_status["failed_reqs"]
        status["migration_held_reqs"] = migration_status["held_reqs"]
        status["migration_last_error"] = migration_status["last_error"]
        status["migration_dry_run"] = migration_status["dry_run"]
        status["migration_prepare_only"] = migration_status["prepare_only"]
        if migration_status["enabled"] and status.get("direction") == "d_to_p":
            status["active_request_migration_strategy"] = (
                "decode_to_decode_kv_transfer"
            )
        return status

    def build_pd_flip_snapshot(self) -> ClusterSnapshot:
        role = DisaggregationMode.to_engine_type(self.disaggregation_mode.value)
        server_args = get_global_server_args()
        prefill_nodes = server_args.pd_flip_prefill_nodes
        decode_nodes = server_args.pd_flip_decode_nodes
        if prefill_nodes is None:
            prefill_nodes = 1 if role == "prefill" else 0
        if decode_nodes is None:
            decode_nodes = 1 if role == "decode" else 0

        kv_total_tokens = getattr(self, "max_total_num_tokens", None)
        kv_used_tokens = None
        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if kv_total_tokens is not None and allocator is not None:
            try:
                kv_used_tokens = kv_total_tokens - allocator.available_size()
            except Exception:
                kv_used_tokens = None

        return ClusterSnapshot(
            timestamp=time.monotonic(),
            role=role,
            prefill_nodes=prefill_nodes,
            decode_nodes=decode_nodes,
            waiting_reqs=len(self.waiting_queue),
            running_reqs=len(getattr(self.running_batch, "reqs", [])),
            prefill_bootstrap_reqs=self._pd_flip_queue_len(
                self.disagg_prefill_bootstrap_queue
            ),
            prefill_inflight_reqs=len(self.disagg_prefill_inflight_queue or []),
            decode_prealloc_reqs=self._pd_flip_queue_len(
                self.disagg_decode_prealloc_queue
            ),
            decode_transfer_reqs=self._pd_flip_queue_len(
                self.disagg_decode_transfer_queue
            ),
            kv_used_tokens=kv_used_tokens,
            kv_total_tokens=kv_total_tokens,
            prefill_slo_attainment=server_args.pd_flip_prefill_slo_attainment,
            decode_slo_attainment=server_args.pd_flip_decode_slo_attainment,
        )

    def prepare_pd_flip(
        self, snapshot: ClusterSnapshot, decision: FlipDecision
    ) -> bool:
        if self.ps.tp_rank == 0:
            logger.debug(
                "PD flip preparing waits for migration: direction=%s reason=%s "
                "role=%s waiting=%d running=%d prefill_bootstrap=%d "
                "prefill_inflight=%d decode_prealloc=%d decode_transfer=%d",
                decision.direction.value,
                decision.reason,
                snapshot.role,
                snapshot.waiting_reqs,
                snapshot.running_reqs,
                snapshot.prefill_bootstrap_reqs,
                snapshot.prefill_inflight_reqs,
                snapshot.decode_prealloc_reqs,
                snapshot.decode_transfer_reqs,
            )
        if (
            decision.direction == FlipDirection.D_TO_P
            and self._pd_flip_migration_is_active()
            and not self._pd_flip_migration_is_released()
        ):
            return False

        if not self.pd_flip_is_idle_for_commit(snapshot):
            return False

        if self.pd_runtime_role_switch_enabled():
            return True

        server_args = get_global_server_args()
        ready = bool(getattr(server_args, "pd_flip_prepare_ack", False))
        if ready:
            setattr(server_args, "pd_flip_prepare_ack", False)
        return ready

    def commit_pd_flip(self, snapshot: ClusterSnapshot, decision: FlipDecision) -> bool:
        if self.ps.tp_rank == 0:
            logger.debug(
                "PD flip commit waits for role mutation: direction=%s "
                "target_prefill_nodes=%s target_decode_nodes=%s.",
                decision.direction.value,
                decision.target_prefill_nodes,
                decision.target_decode_nodes,
            )
        if not self.pd_flip_is_idle_for_commit(snapshot):
            return False

        if self.pd_runtime_role_switch_enabled():
            target = (
                "prefill"
                if decision.direction == FlipDirection.D_TO_P
                else "decode"
            )
            out = self.set_pd_runtime_role(PDRuntimeRoleSetReq(role=target))
            return out.success

        server_args = get_global_server_args()
        ready = bool(getattr(server_args, "pd_flip_commit_ack", False))
        if ready:
            setattr(server_args, "pd_flip_commit_ack", False)
        return ready

    def log_pd_flip_event(self, event: FlipEvent, snapshot: ClusterSnapshot):
        if self.ps.tp_rank != 0 or event.transition == FlipTransition.NONE:
            return
        log_fn = (
            logger.debug
            if event.transition
            in (FlipTransition.PREPARING_NOT_READY, FlipTransition.FLIPPING_NOT_READY)
            else logger.info
        )
        log_fn(
            "PD flip state transition: %s -> %s transition=%s direction=%s "
            "reason=%s role=%s prefill_nodes=%d decode_nodes=%d kv=%s/%s",
            event.from_state.value,
            event.to_state.value,
            event.transition.value,
            event.direction.value,
            event.reason,
            snapshot.role,
            snapshot.prefill_nodes,
            snapshot.decode_nodes,
            snapshot.kv_used_tokens,
            snapshot.kv_total_tokens,
        )

    @staticmethod
    def _pd_flip_queue_len(queue) -> int:
        if queue is None:
            return 0
        inner_queue = getattr(queue, "queue", None)
        if inner_queue is not None:
            return len(inner_queue)
        try:
            return len(queue)
        except TypeError:
            return 0

    def _pd_flip_waiting_req_skip_reason(self, req: Req) -> str:
        finished = getattr(req, "finished", None)
        if callable(finished) and finished():
            return "finished"
        if getattr(req, "req_pool_idx", None) is None:
            return "missing_req_pool_idx"
        kv_committed_len = getattr(req, "kv_committed_len", None)
        if kv_committed_len is None:
            kv_committed_len = len(getattr(req, "origin_input_ids", []) or []) + max(
                0, len(getattr(req, "output_ids", []) or []) - 1
            )
        try:
            committed_len = int(kv_committed_len or 0)
        except (TypeError, ValueError):
            committed_len = 0
        if committed_len <= 0:
            return "missing_committed_kv"
        return ""

    def _pd_flip_classify_waiting_reqs(
        self, waiting_reqs: List[Req]
    ) -> Tuple[List[Tuple[int, Req]], List[Dict[str, Any]]]:
        selected: List[Tuple[int, Req]] = []
        skipped: List[Dict[str, Any]] = []
        for index, req in enumerate(waiting_reqs):
            reason = self._pd_flip_waiting_req_skip_reason(req)
            if reason:
                skipped.append(
                    {
                        "rid": getattr(req, "rid", ""),
                        "queue_index": index,
                        "reason": reason,
                    }
                )
                continue
            selected.append((index, req))
        return selected, skipped

    def _pd_flip_select_source_batch(
        self,
        recv_req: PDFlipMigrationSourceStartReq,
        waiting_reqs: Optional[List[Req]] = None,
        waiting_skipped_out: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Req]:
        def raise_for_duplicate_rid(rids: List[str]) -> None:
            seen = set()
            for rid in rids:
                if rid in seen:
                    raise ValueError(f"duplicate rid in source selection: {rid}")
                seen.add(rid)

        running_reqs = [
            req
            for req in getattr(getattr(self, "running_batch", None), "reqs", [])
            if not req.finished()
        ]
        if recv_req.rids is None:
            selected = list(running_reqs)
        else:
            requested_rids = [str(rid) for rid in recv_req.rids]
            raise_for_duplicate_rid(requested_rids)
            selected = running_reqs[: len(requested_rids)]
            if [str(req.rid) for req in selected] != requested_rids:
                raise ValueError("selected rids must be a running-batch prefix")

        if recv_req.include_waiting:
            if waiting_reqs is None:
                waiting_reqs = list(getattr(self, "waiting_queue", []))
            waiting_selected, waiting_skipped = self._pd_flip_classify_waiting_reqs(
                waiting_reqs
            )
            if waiting_skipped_out is not None:
                waiting_skipped_out.extend(waiting_skipped)
            live_waiting_skipped = [
                item for item in waiting_skipped if item.get("reason") != "finished"
            ]
            if live_waiting_skipped:
                skipped_summary = ", ".join(
                    f"{item.get('rid', '')}:{item.get('reason', '')}"
                    for item in live_waiting_skipped
                )
                raise ValueError(
                    "remaining waiting requests are not migratable: "
                    f"{skipped_summary}"
                )
            selected.extend(req for _, req in waiting_selected)

        raise_for_duplicate_rid([str(req.rid) for req in selected])
        return selected

    def start_pd_flip_migration_source(
        self, recv_req: PDFlipMigrationSourceStartReq
    ) -> PDFlipMigrationReqOutput:
        timing_debug = {}
        role = DisaggregationMode.to_engine_type(self.disaggregation_mode.value)
        if role != "decode":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"source migration requires decode role, got {role}",
                status=self._pd_flip_migration_status_dict(),
            )

        waiting_scan_started = time.monotonic()
        waiting_reqs = list(getattr(self, "waiting_queue", []))
        waiting_skipped = []
        scan_started = time.monotonic()
        try:
            selected_reqs = self._pd_flip_select_source_batch(
                recv_req,
                waiting_reqs=waiting_reqs,
                waiting_skipped_out=waiting_skipped,
            )
        except ValueError as exc:
            return PDFlipMigrationReqOutput(
                success=False,
                message=str(exc),
                status=self._pd_flip_migration_status_dict(),
            )
        running_candidates = [
            req
            for req in getattr(getattr(self, "running_batch", None), "reqs", [])
            if not req.finished()
        ]
        running_count = (
            len(running_candidates)
            if recv_req.rids is None
            else len(recv_req.rids)
        )
        running_reqs = selected_reqs[:running_count]
        timing_debug["scan_running_reqs_s"] = time.monotonic() - scan_started
        timing_debug["running_reqs"] = len(running_reqs)

        if recv_req.include_waiting:
            waiting_indexes = {id(req): index for index, req in enumerate(waiting_reqs)}
            waiting_selected = [
                (waiting_indexes[id(req)], req)
                for req in selected_reqs[running_count:]
            ]
        else:
            waiting_selected = []
        timing_debug["scan_waiting_reqs_s"] = time.monotonic() - waiting_scan_started
        timing_debug["waiting_reqs"] = len(waiting_reqs)
        timing_debug["waiting_skipped_count"] = len(waiting_skipped)
        timing_debug["waiting_skipped"] = waiting_skipped

        session_id = recv_req.session_id or f"pd-flip-{int(time.time() * 1000)}"
        manifest_started = time.monotonic()
        manifests = []
        migration_reqs = []
        source_waiting_reqs = []
        for req in running_reqs:
            req.pd_flip_migration_session_id = session_id
            manifest = self._pd_flip_build_migration_manifest(req)
            manifest["pd_flip_source_queue"] = "running"
            manifest["migration_bootstrap_room"] = self._pd_flip_migration_room_for_req(
                req
            )
            manifest["source_bootstrap_port"] = (
                getattr(
                    getattr(self, "server_args", None),
                    "disaggregation_bootstrap_port",
                    8998,
                )
            )
            manifests.append(manifest)
            migration_reqs.append(req)
        timing_debug["running_manifest_count"] = len(manifests)
        waiting_manifest_started = time.monotonic()
        for queue_index, req in waiting_selected:
            req.pd_flip_migration_session_id = session_id
            manifest = self._pd_flip_build_migration_manifest(req)
            manifest["pd_flip_source_queue"] = "waiting"
            manifest["pd_flip_waiting_queue_index"] = queue_index
            manifest["migration_bootstrap_room"] = self._pd_flip_migration_room_for_req(
                req
            )
            manifest["source_bootstrap_port"] = (
                getattr(
                    getattr(self, "server_args", None),
                    "disaggregation_bootstrap_port",
                    8998,
                )
            )
            manifests.append(manifest)
            migration_reqs.append(req)
            source_waiting_reqs.append(
                {
                    "rid": str(manifest.get("rid") or getattr(req, "rid", "")),
                    "req": req,
                    "original_index": queue_index,
                }
            )
        timing_debug["build_waiting_manifests_s"] = (
            time.monotonic() - waiting_manifest_started
        )
        timing_debug["waiting_manifest_count"] = len(source_waiting_reqs)
        timing_debug["build_manifests_s"] = time.monotonic() - manifest_started
        timing_debug["manifest_count"] = len(manifests)
        entry_started = time.monotonic()
        real_entries, real_error = self._pd_flip_start_source_entries(
            migration_reqs, manifests
        )
        timing_debug["start_source_entries_s"] = time.monotonic() - entry_started
        session = {
            "session_id": session_id,
            "role": "source",
            "state": "source_started" if not real_error else "source_failed",
            "target_url": recv_req.target_url,
            "manifests": manifests,
            "pending_reqs": len(manifests),
            "transferred_reqs": 0,
            "released_reqs": 0,
            "failed_reqs": len(manifests) if real_error else 0,
            "last_error": real_error,
            "dry_run": not bool(real_entries),
            "source_entries": real_entries,
            "source_waiting_reqs": source_waiting_reqs,
            "timing_debug": timing_debug,
        }
        self.pd_flip_migration_session = session
        if real_entries:
            freeze_started = time.monotonic()
            self._pd_flip_freeze_waiting_source_requests(session)
            timing_debug["freeze_waiting_reqs_s"] = time.monotonic() - freeze_started
            self._pd_flip_source_pump_transfer(self.pd_flip_migration_session)
        return PDFlipMigrationReqOutput(
            success=not bool(real_error),
            message=real_error or "source migration session started",
            status=self._pd_flip_migration_status_dict(),
            manifests=manifests,
        )

    def prepare_pd_flip_migration_target(
        self, recv_req: PDFlipMigrationTargetPrepareReq
    ) -> PDFlipMigrationReqOutput:
        role = DisaggregationMode.to_engine_type(self.disaggregation_mode.value)
        if role != "decode":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"target migration requires decode role, got {role}",
                status=self._pd_flip_migration_status_dict(),
            )

        session_id = recv_req.session_id or f"pd-flip-target-{int(time.time() * 1000)}"
        manifests = list(recv_req.manifests or [])
        for manifest in manifests:
            manifest_session_id = manifest.get("pd_flip_session_id")
            if manifest_session_id is not None and str(manifest_session_id) != str(
                session_id
            ):
                return PDFlipMigrationReqOutput(
                    success=False,
                    message="target manifest session id does not match",
                    status=self._pd_flip_migration_status_dict(),
                    manifests=manifests,
                )
            manifest["pd_flip_session_id"] = session_id
        timing_debug = {"manifest_count": len(manifests)}
        entry_started = time.monotonic()
        target_entries, real_error = self._pd_flip_prepare_target_entries(
            manifests, recv_req.source_url
        )
        timing_debug["prepare_target_entries_s"] = time.monotonic() - entry_started
        self.pd_flip_migration_session = {
            "session_id": session_id,
            "role": "target",
            "state": "target_prepared" if not real_error else "target_failed",
            "source_url": recv_req.source_url,
            "manifests": manifests,
            "pending_reqs": len(manifests),
            "transferred_reqs": 0,
            "released_reqs": 0,
            "failed_reqs": len(manifests) if real_error else 0,
            "last_error": real_error,
            "dry_run": not bool(target_entries),
            "adopt_on_success": recv_req.adopt_on_success,
            "prepare_only": recv_req.prepare_only,
            "adopt_on_commit": recv_req.adopt_on_commit,
            "held_reqs": 0,
            "target_entries": target_entries,
            "timing_debug": timing_debug,
        }
        if target_entries:
            self._pd_flip_target_pump_transfer(self.pd_flip_migration_session)
        return PDFlipMigrationReqOutput(
            success=not bool(real_error),
            message=real_error or "target migration session prepared",
            status=self._pd_flip_migration_status_dict(),
            manifests=manifests,
        )

    def prepare_pd_flip_migration_target_delta(
        self, recv_req: PDFlipMigrationTargetDeltaPrepareReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return PDFlipMigrationReqOutput(
                success=False,
                message="no target migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if session.get("role") != "target":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"local migration role is {session.get('role')}, not target",
                status=self._pd_flip_migration_status_dict(),
            )
        if recv_req.session_id and recv_req.session_id != session.get("session_id"):
            return PDFlipMigrationReqOutput(
                success=False,
                message="target migration session id does not match",
                status=self._pd_flip_migration_status_dict(),
            )

        self._pd_flip_target_pump_transfer(session)
        entries = session.get("target_entries") or {}
        manifests = list(recv_req.manifests or [])
        real_error = ""
        delta_entries = 0
        for manifest in manifests:
            manifest_session_id = manifest.get("pd_flip_session_id")
            if manifest_session_id is not None and str(manifest_session_id) != str(
                session.get("session_id")
            ):
                real_error = "target delta manifest session id does not match"
                break
            manifest["pd_flip_session_id"] = session.get("session_id")
            rid = str(manifest.get("rid", ""))
            entry = entries.get(rid)
            if entry is None:
                real_error = f"target delta references unknown rid={rid}"
                break
            if entry.get("phase") != "transferred_held":
                real_error = f"target delta requires held base transfer for rid={rid}"
                break
            try:
                if manifest.get("pd_flip_drop_target"):
                    entry["drop_on_commit"] = True
                    self._pd_flip_apply_delta_manifest_to_target(entry, manifest)
                    self._pd_flip_release_target_request(entry)
                    entry["held"] = False
                    continue
                if manifest.get("delta_noop"):
                    self._pd_flip_apply_delta_manifest_to_target(entry, manifest)
                    continue
                self._pd_flip_start_target_delta_entry(
                    entry, manifest, recv_req.source_url
                )
                delta_entries += 1
            except Exception as exc:
                real_error = str(exc)
                entry["phase"] = "failed"
                break

        session["delta_manifests"] = manifests
        session["delta_pending_reqs"] = delta_entries
        session["pending_reqs"] = delta_entries
        if real_error:
            session["state"] = "target_failed"
            session["last_error"] = real_error
            session["failed_reqs"] = int(session.get("failed_reqs", 0)) + 1
        else:
            session["state"] = (
                "target_delta_started" if delta_entries else "target_delta_transferred"
            )
        if delta_entries:
            self._pd_flip_target_pump_transfer(session)

        return PDFlipMigrationReqOutput(
            success=not bool(real_error),
            message=real_error or "target migration delta prepared",
            status=self._pd_flip_migration_status_dict(),
            manifests=manifests,
        )

    def commit_pd_flip_migration_target(
        self, recv_req: PDFlipMigrationTargetCommitReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return PDFlipMigrationReqOutput(
                success=False,
                message="no target migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if session.get("role") != "target":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"local migration role is {session.get('role')}, not target",
                status=self._pd_flip_migration_status_dict(),
            )
        if recv_req.session_id and recv_req.session_id != session.get("session_id"):
            return PDFlipMigrationReqOutput(
                success=False,
                message="target migration session id does not match",
                status=self._pd_flip_migration_status_dict(),
            )

        self._pd_flip_note_timing(session, "commit_received")
        self._pd_flip_target_pump_transfer(session)
        entries = session.get("target_entries") or {}
        requested_rids = [
            str(rid)
            for rid in (entries.keys() if recv_req.rids is None else recv_req.rids)
        ]
        requested = set(requested_rids)
        if (
            not requested_rids
            or len(requested) != len(requested_rids)
            or requested != set(entries)
            or any(entries[rid].get("phase") != "transferred_held" for rid in requested)
        ):
            message = "target migration batch is not atomically ready"
            self._pd_flip_abort_target_session(session, message)
            return PDFlipMigrationReqOutput(
                success=False,
                message=message,
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("manifests", [])),
            )

        try:
            for rid in requested_rids:
                entry = entries[rid]
                if not entry.get("drop_on_commit"):
                    self._pd_flip_target_commit_hicache_restore(entry["decode_req"])
        except Exception as exc:
            message = f"target migration commit failed: {exc}"
            self._pd_flip_abort_target_session(session, message)
            return PDFlipMigrationReqOutput(
                success=False,
                message=message,
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("manifests", [])),
            )

        for rid in requested_rids:
            entries[rid]["phase"] = "ready_to_activate"
        session["state"] = "ready_to_activate"
        return PDFlipMigrationReqOutput(
            success=True,
            message="target batch ready to activate",
            status=self._pd_flip_migration_status_dict(),
            manifests=list(session.get("manifests", [])),
        )

    def activate_pd_flip_migration_target(
        self, recv_req: PDFlipMigrationTargetActivateReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session or session.get("role") != "target":
            return PDFlipMigrationReqOutput(
                success=False,
                message="no target migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if recv_req.session_id and recv_req.session_id != session.get("session_id"):
            return PDFlipMigrationReqOutput(
                success=False,
                message="target migration session id does not match",
                status=self._pd_flip_migration_status_dict(),
            )

        entries = session.get("target_entries") or {}
        requested_rids = [
            str(rid)
            for rid in (entries.keys() if recv_req.rids is None else recv_req.rids)
        ]
        requested = set(requested_rids)
        if (
            not requested_rids
            or len(requested) != len(requested_rids)
            or requested != set(entries)
            or any(entries[rid].get("phase") != "ready_to_activate" for rid in requested)
        ):
            return PDFlipMigrationReqOutput(
                success=False,
                message="target batch is not committed",
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("manifests", [])),
            )

        adopt_entries = [
            entries[rid]
            for rid in requested_rids
            if not entries[rid].get("drop_on_commit")
        ]
        requests = [
            getattr(entry.get("decode_req"), "req", None) for entry in adopt_entries
        ]
        if any(
            req is None
            or entry.get("request_released")
            or entry.get("request_adopted")
            or any(req is queued for queued in self.waiting_queue)
            for entry, req in zip(adopt_entries, requests)
        ):
            return PDFlipMigrationReqOutput(
                success=False,
                message="target batch activation preflight failed",
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("manifests", [])),
            )

        previous_waiting_queue = self.waiting_queue
        previous_entry_state = {
            rid: (
                entries[rid].get("phase"),
                entries[rid].get("held"),
                entries[rid].get("request_adopted"),
            )
            for rid in requested_rids
        }
        try:
            self.waiting_queue = [*previous_waiting_queue, *requests]
            for rid in requested_rids:
                entry = entries[rid]
                if not entry.get("drop_on_commit"):
                    entry["request_adopted"] = True
                    self._pd_flip_note_timing(entry, "target_adopted")
                entry["held"] = False
                entry["phase"] = "active"
        except Exception as exc:
            self.waiting_queue = previous_waiting_queue
            for rid, (phase, held, request_adopted) in previous_entry_state.items():
                entry = entries[rid]
                entry["phase"] = phase
                entry["held"] = held
                if request_adopted is None:
                    entry.pop("request_adopted", None)
                else:
                    entry["request_adopted"] = request_adopted
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"target batch activation failed: {exc}",
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("manifests", [])),
            )
        session["state"] = "active"
        session["held_reqs"] = 0
        session["released_reqs"] = int(session.get("released_reqs", 0)) + len(
            requested_rids
        )
        for req in requests:
            try:
                if hasattr(req, "time_stats") and hasattr(
                    req.time_stats, "set_wait_queue_entry_time"
                ):
                    req.time_stats.set_wait_queue_entry_time()
            except Exception:
                logger.warning(
                    "Failed to record target activation wait-queue timing for %s",
                    getattr(req, "rid", ""),
                    exc_info=True,
                )
        return PDFlipMigrationReqOutput(
            success=True,
            message="target batch activated",
            status=self._pd_flip_migration_status_dict(),
            manifests=list(session.get("manifests", [])),
        )

    def abort_pd_flip_migration_target(
        self, recv_req: PDFlipMigrationTargetAbortReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return PDFlipMigrationReqOutput(
                success=True,
                message="no target migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if session.get("role") != "target":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"local migration role is {session.get('role')}, not target",
                status=self._pd_flip_migration_status_dict(),
            )

        self._pd_flip_abort_target_session(
            session, recv_req.reason or "target migration aborted"
        )
        return PDFlipMigrationReqOutput(
            success=True,
            message="target migration session aborted",
            status=self._pd_flip_migration_status_dict(),
            manifests=list(session.get("manifests", [])),
        )

    def get_pd_flip_migration_status(
        self, recv_req: PDFlipMigrationStatusReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None) or {}
        if session.get("role") == "source":
            self._pd_flip_source_pump_transfer(session)
        elif session.get("role") == "target":
            self._pd_flip_target_pump_transfer(session)
        return PDFlipMigrationReqOutput(
            success=True,
            message="",
            status=self._pd_flip_migration_status_dict(),
            manifests=list(session.get("manifests", [])),
        )

    def finish_pd_flip_migration_source(
        self, recv_req: PDFlipMigrationSourceFinishReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return PDFlipMigrationReqOutput(
                success=False,
                message="no source migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if session.get("role") != "source":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"local migration role is {session.get('role')}, not source",
                status=self._pd_flip_migration_status_dict(),
            )
        if recv_req.session_id and recv_req.session_id != session.get("session_id"):
            return PDFlipMigrationReqOutput(
                success=False,
                message="source migration session id does not match",
                status=self._pd_flip_migration_status_dict(),
            )

        self._pd_flip_note_timing(session, "finish_received")
        self._pd_flip_source_pump_transfer(session)

        manifests = list(session.get("manifests", []))
        released_rids = recv_req.released_rids
        if released_rids is not None:
            released = {
                manifest.get("rid")
                for manifest in manifests
                if manifest.get("rid") in set(released_rids)
            }
        else:
            released = {manifest.get("rid") for manifest in manifests}
        transferred = set(session.get("transferred_rids", set()))
        not_ready = sorted(rid for rid in released if rid not in transferred)
        if not_ready and not session.get("dry_run", False):
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"migration transfer still pending for rids={not_ready}",
                status=self._pd_flip_migration_status_dict(),
                manifests=manifests,
            )

        entries = session.get("source_entries") or {}
        advanced_snapshots = []
        for rid in released:
            entry = entries.get(rid)
            if not entry:
                continue
            source_queue = entry.get("source_queue") or (
                entry.get("manifest") or {}
            ).get("pd_flip_source_queue")
            if source_queue != "running":
                continue
            try:
                snapshot_len = int(entry.get("committed_len") or 0)
                current_len = int(
                    getattr(entry.get("req"), "kv_committed_len", snapshot_len) or 0
                )
            except (TypeError, ValueError):
                continue
            if current_len > snapshot_len:
                advanced_snapshots.append(f"{rid}: {snapshot_len}->{current_len}")
        if advanced_snapshots and not session.get("dry_run", False):
            message = (
                "running requests advanced after migration snapshot; "
                "delta KV transfer is required before source release: "
                + ", ".join(sorted(advanced_snapshots))
            )
            session["state"] = "source_failed"
            session["last_error"] = message
            session["failed_reqs"] = len(advanced_snapshots)
            return PDFlipMigrationReqOutput(
                success=False,
                message=message,
                status=self._pd_flip_migration_status_dict(),
                manifests=manifests,
            )

        self._pd_flip_release_source_requests(session, released)
        running_released = {
            rid
            for rid in released
            if (
                (entries.get(rid) or {}).get("source_queue")
                or ((entries.get(rid) or {}).get("manifest") or {}).get(
                    "pd_flip_source_queue"
                )
            )
            == "running"
        }
        running_batch = getattr(self, "running_batch", None)
        if running_released and running_batch is not None:
            running_batch.filter_batch()
        self._pd_flip_resume_batch_after_cutover()
        session["state"] = "source_released"
        session["pending_reqs"] = max(0, len(manifests) - len(released))
        session["released_reqs"] = len(released)
        session["transferred_reqs"] = len(released)
        return PDFlipMigrationReqOutput(
            success=True,
            message="source migration session released",
            status=self._pd_flip_migration_status_dict(),
            manifests=manifests,
        )

    def start_pd_flip_migration_source_delta(
        self, recv_req: PDFlipMigrationSourceDeltaReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return PDFlipMigrationReqOutput(
                success=False,
                message="no source migration session exists",
                status=self._pd_flip_migration_status_dict(),
            )
        if session.get("role") != "source":
            return PDFlipMigrationReqOutput(
                success=False,
                message=f"local migration role is {session.get('role')}, not source",
                status=self._pd_flip_migration_status_dict(),
            )
        if recv_req.session_id and recv_req.session_id != session.get("session_id"):
            return PDFlipMigrationReqOutput(
                success=False,
                message="source migration session id does not match",
                status=self._pd_flip_migration_status_dict(),
            )

        self._pd_flip_source_pump_transfer(session)
        entries = session.get("source_entries") or {}
        requested_rids = [
            str(rid)
            for rid in (entries.keys() if recv_req.rids is None else recv_req.rids)
        ]
        requested = set(requested_rids)
        if (
            not requested_rids
            or len(requested) != len(requested_rids)
            or not requested.issubset(entries)
        ):
            return PDFlipMigrationReqOutput(
                success=False,
                message="source delta batch is empty, duplicated, or unknown",
                status=self._pd_flip_migration_status_dict(),
                manifests=[],
            )
        requested_key = tuple(sorted(requested))
        captured_rids = set(session.get("delta_request_rids") or ())
        if captured_rids:
            if requested != captured_rids:
                return PDFlipMigrationReqOutput(
                    success=False,
                    message="source delta rids do not match the captured batch",
                    status=self._pd_flip_migration_status_dict(),
                    manifests=[],
                )
            return PDFlipMigrationReqOutput(
                success=session.get("state") != "source_failed",
                message=session.get("last_error")
                or "source migration delta already captured",
                status=self._pd_flip_migration_status_dict(),
                manifests=list(session.get("delta_manifests") or []),
            )
        if not getattr(self, "pd_flip_batch_quiesced", False):
            resolved_session_id = str(session.get("session_id") or "")
            frozen_session_id = getattr(self, "pd_flip_quiesce_session_id", None)
            frozen_rids = tuple(getattr(self, "pd_flip_quiesce_rids", ()))
            if getattr(self, "pd_flip_quiesce_requested", False):
                if (
                    frozen_session_id != resolved_session_id
                    or frozen_rids != requested_key
                ):
                    return PDFlipMigrationReqOutput(
                        success=False,
                        message="source batch quiesce key does not match pending request",
                        status=self._pd_flip_migration_status_dict(),
                        manifests=[],
                    )
            else:
                self._pd_flip_request_batch_quiesce(resolved_session_id, requested_key)
            session["state"] = "source_quiesce_requested"
            return PDFlipMigrationReqOutput(
                success=False,
                message="source batch quiesce pending; retry delta after quiesce",
                status=self._pd_flip_migration_status_dict(),
                manifests=[],
            )
        if str(session.get("session_id") or "") != getattr(
            self, "pd_flip_quiesce_session_id", None
        ) or requested_key != tuple(getattr(self, "pd_flip_quiesce_rids", ())):
            return PDFlipMigrationReqOutput(
                success=False,
                message="source delta rids do not match the quiesced batch",
                status=self._pd_flip_migration_status_dict(),
                manifests=[],
            )
        delta_index = int(session.get("delta_generation", 0)) + 1
        session["delta_generation"] = delta_index
        session["delta_request_rids"] = tuple(sorted(requested))
        delta_manifests: List[Dict[str, Any]] = []
        delta_entries = 0
        real_error = ""

        for rid in sorted(requested):
            entry = entries.get(rid)
            if not entry:
                continue
            source_queue = entry.get("source_queue") or (
                entry.get("manifest") or {}
            ).get("pd_flip_source_queue")
            if source_queue != "running":
                continue
            manifest = self._pd_flip_build_delta_manifest(entry, delta_index)
            delta_manifests.append(manifest)
            if manifest.get("pd_flip_drop_target") or manifest.get("delta_noop"):
                self._pd_flip_mark_source_delta_applied(entry, manifest)
                continue
            try:
                self._pd_flip_start_source_delta_entry(entry, manifest)
                delta_entries += 1
            except Exception as exc:
                real_error = str(exc)
                entry["failed"] = True
                session.setdefault("failed_rids", set()).add(rid)
                break

        session["delta_manifests"] = delta_manifests
        session["delta_pending_reqs"] = delta_entries
        session["pending_reqs"] = delta_entries
        session["failed_reqs"] = int(session.get("failed_reqs", 0)) + (
            1 if real_error else 0
        )
        session["state"] = (
            "source_delta_started" if delta_entries else "source_delta_transferred"
        )
        if real_error:
            session["state"] = "source_failed"
            session["last_error"] = real_error
        if delta_entries:
            self._pd_flip_source_pump_transfer(session)

        return PDFlipMigrationReqOutput(
            success=not bool(real_error),
            message=real_error or "source migration delta started",
            status=self._pd_flip_migration_status_dict(),
            manifests=delta_manifests,
        )

    def _pd_flip_request_batch_quiesce(self, session_id: str, rids) -> None:
        self.pd_flip_quiesce_session_id = session_id
        self.pd_flip_quiesce_rids = tuple(str(rid) for rid in rids)
        self.pd_flip_quiesce_requested = True
        self.pd_flip_batch_quiesced = False

    def _pd_flip_maybe_enter_batch_quiesce(self) -> bool:
        if not getattr(self, "pd_flip_quiesce_requested", False):
            return False
        if getattr(self, "result_queue", None):
            return False
        self.pd_flip_batch_quiesced = True
        return True

    def _pd_flip_resume_batch_after_cutover(self) -> None:
        self.pd_flip_quiesce_requested = False
        self.pd_flip_batch_quiesced = False
        self.pd_flip_quiesce_rids = ()
        self.pd_flip_quiesce_session_id = None

    def abort_pd_flip_migration(
        self, recv_req: PDFlipMigrationAbortReq
    ) -> PDFlipMigrationReqOutput:
        session = getattr(self, "pd_flip_migration_session", None)
        if session:
            reason = recv_req.reason or "migration aborted"
            if session.get("role") == "target":
                self._pd_flip_abort_target_session(session, reason)
            elif session.get("role") == "source":
                self._pd_flip_abort_source_session(session, reason)
            else:
                session["state"] = "aborted"
                session["last_error"] = reason
                session["pending_reqs"] = 0
                session["failed_reqs"] = len(session.get("manifests", []))
        return PDFlipMigrationReqOutput(
            success=True,
            message="migration aborted",
            status=self._pd_flip_migration_status_dict(),
        )

    def _pd_flip_abort_source_session(
        self, session: Dict[str, Any], reason: str
    ) -> None:
        for entry in (session.get("source_entries") or {}).values():
            sender = entry.get("sender")
            if sender is not None and hasattr(sender, "abort"):
                try:
                    sender.abort()
                except Exception:
                    pass
            delta = entry.get("delta")
            if isinstance(delta, dict):
                delta_sender = delta.get("sender")
                if delta_sender is not None and hasattr(delta_sender, "abort"):
                    try:
                        delta_sender.abort()
                    except Exception:
                        pass
                self._pd_flip_free_source_delta_metadata(delta)
            self._pd_flip_finish_source_kv_release_defer(entry)
            self._pd_flip_free_source_metadata(entry)
        restore_started = time.monotonic()
        self._pd_flip_restore_waiting_source_requests(session)
        self._pd_flip_resume_batch_after_cutover()
        session.setdefault("timing_debug", {})["restore_waiting_reqs_s"] = (
            time.monotonic() - restore_started
        )
        session["state"] = "source_aborted"
        session["last_error"] = reason
        session["pending_reqs"] = 0
        session["failed_reqs"] = 0

    def _pd_flip_abort_target_session(
        self, session: Dict[str, Any], reason: str
    ) -> None:
        released = 0
        for entry in (session.get("target_entries") or {}).values():
            decode_req = entry.get("decode_req")
            receiver = getattr(decode_req, "kv_receiver", None)
            if receiver is not None and hasattr(receiver, "abort"):
                try:
                    receiver.abort()
                except Exception:
                    pass
            if not entry.get("request_adopted"):
                self._pd_flip_release_target_request(entry)
                released += 1
            self._pd_flip_free_target_metadata(entry)
            entry["phase"] = "aborted"
            entry["held"] = False
        session["state"] = "target_aborted"
        session["last_error"] = reason
        session["pending_reqs"] = 0
        session["held_reqs"] = 0
        session["released_reqs"] = int(session.get("released_reqs", 0)) + released
        session["failed_reqs"] = 0

    def _pd_flip_can_use_real_migration(self) -> bool:
        return (
            getattr(self, "disaggregation_mode", None) == DisaggregationMode.DECODE
            and getattr(self, "disagg_decode_prealloc_queue", None) is not None
            and getattr(self, "disagg_metadata_buffers", None) is not None
            and getattr(self, "req_to_metadata_buffer_idx_allocator", None) is not None
            and getattr(self, "req_to_token_pool", None) is not None
            and getattr(self, "token_to_kv_pool_allocator", None) is not None
        )

    def _pd_flip_migration_room_for_req(self, req: Req) -> int:
        server_args = getattr(self, "server_args", None)
        dp_size = max(1, int(getattr(server_args, "dp_size", 1)))
        ps = getattr(self, "ps", None)
        dp_rank = getattr(ps, "attn_dp_rank", None)
        if dp_rank is None:
            dp_rank = getattr(ps, "dp_rank", 0) or 0
        digest = hashlib.sha256(str(getattr(req, "rid", "")).encode()).digest()
        base = int.from_bytes(digest[:8], "big") % (2**30)
        return (base // dp_size) * dp_size + int(dp_rank)

    def _pd_flip_delta_room_for_req(self, req: Req, delta_index: int) -> int:
        server_args = getattr(self, "server_args", None)
        dp_size = max(1, int(getattr(server_args, "dp_size", 1)))
        return self._pd_flip_migration_room_for_req(req) + dp_size * max(
            1, int(delta_index)
        )

    def _pd_flip_delta_page_start(self, committed_len: int) -> int:
        page_size = int(getattr(self.token_to_kv_pool_allocator, "page_size", 1) or 1)
        return (max(0, int(committed_len)) // page_size) * page_size

    @staticmethod
    def _pd_flip_stitch_boundary(
        storage_hit: int, prompt_len: int, page_size: int
    ) -> Tuple[int, str]:
        page_size = max(1, int(page_size))
        page_aligned_prompt = (max(0, int(prompt_len)) // page_size) * page_size
        hit_len = min(max(0, int(storage_hit)), page_aligned_prompt)
        if hit_len == 0:
            return 0, "source_decode_full_fallback"
        if hit_len == page_aligned_prompt:
            return hit_len, "full_prefix_stitch"
        return hit_len, "partial_prefix_stitch"

    def _pd_flip_source_page_indices_range(
        self, req: Req, start_len: int, end_len: int
    ):
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} req_pool_idx was released "
                "before migration delta transfer"
            )
        if hasattr(req_pool_idx, "item"):
            req_pool_idx = req_pool_idx.item()
        page_start = self._pd_flip_delta_page_start(start_len)
        kv_indices = (
            self.req_to_token_pool.req_to_token[int(req_pool_idx), page_start:end_len]
            .cpu()
            .numpy()
        )
        if kv_indices.ndim != 1:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} produced non-1D delta KV indices "
                f"shape={kv_indices.shape}"
            )
        return kv_to_page_indices(
            kv_indices,
            self.token_to_kv_pool_allocator.page_size,
        )

    def _pd_flip_stitch_page_indices_range(
        self, req: Req, hit_len: int, committed_len: int
    ):
        hit_len = int(hit_len)
        committed_len = int(committed_len)
        if not 0 <= hit_len <= committed_len:
            raise ValueError(
                f"invalid PD flip stitch page range H={hit_len}, C0={committed_len}"
            )
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} req_pool_idx was released "
                "before migration stitch transfer"
            )
        if hasattr(req_pool_idx, "item"):
            req_pool_idx = req_pool_idx.item()
        page_size = int(self.token_to_kv_pool_allocator.page_size)
        mapping_start = (hit_len // page_size) * page_size
        kv_indices = (
            self.req_to_token_pool.req_to_token[
                int(req_pool_idx), mapping_start:committed_len
            ]
            .cpu()
            .numpy()
        )
        expected_mapping_len = committed_len - mapping_start
        if kv_indices.ndim != 1 or len(kv_indices) != expected_mapping_len:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} has incomplete stitch KV "
                f"mapping shape={kv_indices.shape}, expected_tokens={expected_mapping_len}"
            )
        return kv_to_page_indices(kv_indices, page_size)

    def _pd_flip_build_delta_manifest(
        self, entry: Dict[str, Any], delta_index: int
    ) -> Dict[str, Any]:
        req = entry.get("req")
        base_manifest = dict(entry.get("manifest") or {})
        if req is None:
            manifest = base_manifest
            manifest["pd_flip_drop_target"] = True
            manifest["delta_noop"] = True
            return manifest

        old_len = int(entry.get("committed_len") or 0)
        current_len = int(getattr(req, "kv_committed_len", old_len) or old_len)
        manifest = self._pd_flip_build_migration_manifest(req)
        migration_room = self._pd_flip_delta_room_for_req(req, delta_index)
        manifest.update(
            {
                "pd_flip_delta": True,
                "pd_flip_source_queue": "running",
                "delta_from_len": old_len,
                "delta_page_start_len": self._pd_flip_delta_page_start(old_len),
                "migration_bootstrap_room": migration_room,
                "source_bootstrap_port": getattr(
                    getattr(self, "server_args", None),
                    "disaggregation_bootstrap_port",
                    8998,
                ),
            }
        )
        if req.finished():
            manifest["pd_flip_drop_target"] = True
            manifest["delta_noop"] = True
        elif current_len <= old_len:
            manifest["delta_noop"] = True
        else:
            manifest["delta_noop"] = False
        return manifest

    def _pd_flip_mark_source_delta_applied(
        self, entry: Dict[str, Any], manifest: Dict[str, Any]
    ) -> None:
        entry["committed_len"] = int(
            manifest.get("kv_committed_len") or entry.get("committed_len") or 0
        )
        entry["manifest"] = manifest
        entry["delta"] = {
            "phase": "transferred",
            "from_len": int(manifest.get("delta_from_len") or 0),
            "to_len": int(manifest.get("kv_committed_len") or 0),
            "noop": True,
            "drop_target": bool(manifest.get("pd_flip_drop_target")),
        }

    def _pd_flip_start_source_entries(
        self, running_reqs: List[Req], manifests: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        if not self._pd_flip_can_use_real_migration():
            return {}, ""
        if getattr(self, "enable_hisparse", False):
            return {}, "PD flip migration does not support HiSparse decode yet"

        try:
            manager_started = time.monotonic()
            kv_manager = self._pd_flip_get_source_kv_manager()
            manager_s = time.monotonic() - manager_started
            sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
            bootstrap_addr = self._pd_flip_local_bootstrap_addr(kv_manager)
            entries: Dict[str, Dict[str, Any]] = {}

            def fail(message: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
                for entry in entries.values():
                    sender = entry.get("sender")
                    if hasattr(sender, "abort"):
                        try:
                            sender.abort()
                        except Exception:
                            pass
                    self._pd_flip_clear_source_kv_release_defer(entry.get("req"))
                    self._pd_flip_free_source_metadata(entry)
                return {}, message

            for req, manifest in zip(running_reqs, manifests):
                entry_started = time.monotonic()
                timing_debug = {"get_source_kv_manager_s": manager_s}
                rid = str(manifest.get("rid") or getattr(req, "rid", ""))
                committed_len = int(manifest.get("kv_committed_len") or 0)
                if committed_len <= 0:
                    return fail(f"request {rid} has no committed KV to migrate")
                if getattr(req, "req_pool_idx", None) is None:
                    return fail(f"request {rid} has no req_pool_idx")

                alloc_started = time.monotonic()
                metadata_index = self.req_to_metadata_buffer_idx_allocator.alloc()
                timing_debug["metadata_alloc_s"] = time.monotonic() - alloc_started
                if metadata_index is None:
                    return fail("no metadata buffer available for source migration")

                migration_room = int(manifest["migration_bootstrap_room"])
                sender_started = time.monotonic()
                sender = sender_class(
                    mgr=kv_manager,
                    bootstrap_addr=bootstrap_addr,
                    bootstrap_room=migration_room,
                    dest_tp_ranks=[self.ps.tp_rank],
                    pp_rank=self.ps.pp_rank,
                )
                timing_debug["sender_create_s"] = time.monotonic() - sender_started
                self._pd_flip_defer_source_kv_release(req)
                entries[rid] = {
                    "req": req,
                    "sender": sender,
                    "manifest": manifest,
                    "source_queue": manifest.get("pd_flip_source_queue", "running"),
                    "metadata_index": metadata_index,
                    "migration_bootstrap_room": migration_room,
                    "committed_len": committed_len,
                    "sent": False,
                    "transferred": False,
                    "failed": False,
                    "timing_debug": timing_debug,
                }
                timing_debug["source_entry_init_total_s"] = (
                    time.monotonic() - entry_started
                )
            return entries, ""
        except Exception as exc:
            try:
                return fail(str(exc))
            except Exception:
                return {}, str(exc)

    def _pd_flip_start_source_delta_entry(
        self, entry: Dict[str, Any], manifest: Dict[str, Any]
    ) -> None:
        if not self._pd_flip_can_use_real_migration():
            self._pd_flip_mark_source_delta_applied(entry, manifest)
            return
        req = entry.get("req")
        if req is None:
            raise ValueError("source delta entry has no request")
        if getattr(req, "req_pool_idx", None) is None:
            raise ValueError(f"request {getattr(req, 'rid', '')} has no req_pool_idx")

        metadata_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        if metadata_index is None:
            raise RuntimeError("no metadata buffer available for source delta migration")

        kv_manager = self._pd_flip_get_source_kv_manager()
        sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)
        migration_room = int(manifest["migration_bootstrap_room"])
        sender = sender_class(
            mgr=kv_manager,
            bootstrap_addr=self._pd_flip_local_bootstrap_addr(kv_manager),
            bootstrap_room=migration_room,
            dest_tp_ranks=[self.ps.tp_rank],
            pp_rank=self.ps.pp_rank,
        )
        from_len = int(manifest.get("delta_from_len") or entry.get("committed_len") or 0)
        to_len = int(manifest.get("kv_committed_len") or from_len)
        page_indices = self._pd_flip_source_page_indices_range(req, from_len, to_len)
        sender.init(len(page_indices), metadata_index)
        entry["delta"] = {
            "phase": "new",
            "sender": sender,
            "metadata_index": metadata_index,
            "migration_bootstrap_room": migration_room,
            "from_len": from_len,
            "to_len": to_len,
            "page_start_len": int(
                manifest.get("delta_page_start_len")
                or self._pd_flip_delta_page_start(from_len)
            ),
            "sent": False,
            "transferred": False,
            "failed": False,
            "manifest": manifest,
        }
        entry["manifest"] = manifest

    def _pd_flip_freeze_waiting_source_requests(
        self, session: Dict[str, Any]
    ) -> None:
        waiting_specs = session.get("source_waiting_reqs") or []
        if not waiting_specs:
            session.setdefault("timing_debug", {})["waiting_frozen_count"] = 0
            return

        by_id = {
            id(spec.get("req")): spec
            for spec in waiting_specs
            if spec.get("req") is not None
        }
        if not by_id:
            session.setdefault("timing_debug", {})["waiting_frozen_count"] = 0
            return

        new_queue = []
        frozen_count = 0
        entries = session.get("source_entries") or {}
        for index, req in enumerate(getattr(self, "waiting_queue", [])):
            spec = by_id.get(id(req))
            if spec is None:
                new_queue.append(req)
                continue
            spec.setdefault("original_index", index)
            spec["frozen"] = True
            spec["restored"] = False
            frozen_count += 1
            rid = str(spec.get("rid") or getattr(req, "rid", ""))
            entry = entries.get(rid)
            if entry is not None:
                self._pd_flip_note_timing(entry, "source_waiting_frozen")

        self.waiting_queue = new_queue
        session.setdefault("timing_debug", {})["waiting_frozen_count"] = frozen_count

    def _pd_flip_restore_waiting_source_requests(
        self, session: Dict[str, Any]
    ) -> None:
        waiting_specs = [
            spec
            for spec in (session.get("source_waiting_reqs") or [])
            if spec.get("frozen") and not spec.get("restored")
        ]
        if not waiting_specs:
            session.setdefault("timing_debug", {})["waiting_restored_count"] = 0
            return

        entries = session.get("source_entries") or {}
        restored_count = 0
        for spec in sorted(
            waiting_specs, key=lambda item: int(item.get("original_index", 0))
        ):
            req = spec.get("req")
            if req is None:
                continue
            if any(existing is req for existing in self.waiting_queue):
                spec["restored"] = True
                continue
            insert_at = min(
                max(0, int(spec.get("original_index", len(self.waiting_queue)))),
                len(self.waiting_queue),
            )
            self.waiting_queue.insert(insert_at, req)
            spec["restored"] = True
            restored_count += 1
            rid = str(spec.get("rid") or getattr(req, "rid", ""))
            entry = entries.get(rid)
            if entry is not None:
                self._pd_flip_note_timing(entry, "source_waiting_restored")

        session.setdefault("timing_debug", {})[
            "waiting_restored_count"
        ] = restored_count

    def _pd_flip_get_source_kv_manager(self):
        manager = getattr(self, "pd_flip_source_kv_manager", None)
        if manager is not None:
            return manager

        base_manager = self.disagg_decode_prealloc_queue.kv_manager
        kv_args = base_manager.kv_args
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        kv_args.prefill_start_layer = getattr(kv_pool, "start_layer", 0)
        kv_args.prefill_end_layer = getattr(kv_pool, "end_layer", None)
        kv_args.page_size = getattr(kv_pool, "page_size", kv_args.page_size)
        kv_args.ib_device = self.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.ps.gpu_id
        if getattr(kv_pool, "compression_ratios", None) is not None:
            kv_args.mla_compression_ratios = list(kv_pool.compression_ratios)
        is_mla = is_mla_backend(kv_pool)
        if not is_mla:
            if hasattr(kv_pool, "head_num"):
                kv_args.kv_head_num = kv_pool.head_num
            kv_args.total_kv_head_num = self.model_config.get_total_num_kv_heads()

        manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        manager = manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.server_args,
            is_mla,
        )
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(manager, "set_kv_buffer_tensors")
            and not is_mla
        ):
            kv_pool_for_tensors = kv_pool.full_kv_pool if hasattr(kv_pool, "full_kv_pool") else kv_pool
            if hasattr(kv_pool_for_tensors, "k_buffer") and hasattr(
                kv_pool_for_tensors, "v_buffer"
            ):
                manager.set_kv_buffer_tensors(
                    kv_pool_for_tensors.k_buffer,
                    kv_pool_for_tensors.v_buffer,
                    kv_pool_for_tensors.page_size,
                )
        self.pd_flip_source_kv_manager = manager
        return manager

    def _pd_flip_local_bootstrap_addr(self, kv_manager) -> str:
        host = getattr(self.server_args, "host", "127.0.0.1")
        if host in ("0.0.0.0", "::"):
            host = getattr(kv_manager, "local_ip", "127.0.0.1")
        return f"{host}:{self.server_args.disaggregation_bootstrap_port}"

    def _pd_flip_source_page_indices(self, req: Req, committed_len: int):
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} req_pool_idx was released "
                "before migration transfer"
            )
        non_scalar_error = (
            f"request {getattr(req, 'rid', '')} has non-scalar req_pool_idx "
            f"{req_pool_idx!r}"
        )
        if hasattr(req_pool_idx, "numel") and req_pool_idx.numel() != 1:
            raise ValueError(non_scalar_error)
        if hasattr(req_pool_idx, "item"):
            try:
                req_pool_idx = req_pool_idx.item()
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(non_scalar_error) from exc
        if not isinstance(req_pool_idx, numbers.Integral):
            raise ValueError(non_scalar_error)
        kv_indices = (
            self.req_to_token_pool.req_to_token[
                int(req_pool_idx), :committed_len
            ]
            .cpu()
            .numpy()
        )
        if kv_indices.ndim != 1:
            raise ValueError(
                f"request {getattr(req, 'rid', '')} produced non-1D KV indices "
                f"shape={kv_indices.shape}"
            )
        return kv_to_page_indices(
            kv_indices,
            self.token_to_kv_pool_allocator.page_size,
        )

    def _pd_flip_source_state_indices(
        self, req: Req, committed_len: int, kv_manager
    ) -> Optional[List]:
        page_size = self.token_to_kv_pool_allocator.page_size

        def _mamba_payload():
            return [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]

        def _swa_payload():
            window_size = self.sliding_window_size
            window_start = max(0, committed_len - window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:committed_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            return kv_to_page_indices(window_kv_indices_swa.cpu().numpy(), page_size)

        def _dsa_payload():
            kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :committed_len
            ]
            return kv_to_page_indices(kv_indices_full.cpu().numpy(), page_size)

        state_indices: Optional[List] = []
        for st in getattr(kv_manager.kv_args, "state_types", []):
            if st == StateType.MAMBA:
                state_indices.append(_mamba_payload())
            elif st == StateType.SWA:
                state_indices.append(_swa_payload())
            elif st == StateType.DSA:
                state_indices.append(_dsa_payload())
            else:
                state_indices.append(None)
        return state_indices

    def _pd_flip_source_send_initial(self, entry: Dict[str, Any]) -> None:
        sender = entry["sender"]
        req = entry["req"]
        committed_len = int(entry["committed_len"])
        if getattr(
            getattr(self, "server_args", None),
            "enable_pd_flip_hicache_stitch",
            False,
        ):
            hit_len = int(sender.pop_decode_prefix_len())
        else:
            hit_len = 0
        if not 0 <= hit_len <= committed_len:
            raise ValueError(
                f"invalid PD flip stitch boundary H={hit_len}, C0={committed_len}"
            )
        page_indices = self._pd_flip_stitch_page_indices_range(
            req, hit_len, committed_len
        )
        sender.init(len(page_indices), entry["metadata_index"])
        sender.send(
            page_indices,
            self._pd_flip_source_state_indices(req, committed_len, sender.kv_mgr),
        )
        entry.update(
            mooncake_hit_len=hit_len,
            source_transfer_start=hit_len,
            source_transfer_end=committed_len,
            stitch_mode=(
                "source_decode_full_fallback" if hit_len == 0 else "prefix_stitch"
            ),
            source_index_shape=list(page_indices.shape)
            if hasattr(page_indices, "shape")
            else [len(page_indices)],
            source_index_size=int(page_indices.size)
            if hasattr(page_indices, "size")
            else len(page_indices),
        )

    def _pd_flip_set_source_metadata(
        self, req: Req, metadata_index: int, migration_room: int
    ) -> None:
        buffers = self.disagg_metadata_buffers
        buffers.output_ids[metadata_index].zero_()
        buffers.cached_tokens[metadata_index].zero_()
        buffers.output_token_logprobs_val[metadata_index].zero_()
        buffers.output_token_logprobs_idx[metadata_index].zero_()
        buffers.output_top_logprobs_val[metadata_index].zero_()
        buffers.output_top_logprobs_idx[metadata_index].zero_()
        buffers.output_topk_p[metadata_index].zero_()
        buffers.output_topk_index[metadata_index].zero_()
        buffers.output_hidden_states[metadata_index].zero_()
        buffers.bootstrap_room[metadata_index].zero_()

        output_ids = list(getattr(req, "output_ids", []) or [])
        buffers.output_ids[metadata_index][0] = output_ids[-1]
        buffers.cached_tokens[metadata_index][0] = getattr(req, "cached_tokens", 0)
        buffers.cached_tokens[metadata_index][1] = getattr(
            req, "cached_tokens_device", 0
        )
        buffers.cached_tokens[metadata_index][2] = getattr(req, "cached_tokens_host", 0)
        buffers.cached_tokens[metadata_index][3] = getattr(
            req, "cached_tokens_storage", 0
        )
        if getattr(req, "return_logprob", False):
            vals = getattr(req.logprob, "output_token_logprobs_val", None)
            idxs = getattr(req.logprob, "output_token_logprobs_idx", None)
            top_vals = getattr(req.logprob, "output_top_logprobs_val", None)
            top_idxs = getattr(req.logprob, "output_top_logprobs_idx", None)
            if vals:
                buffers.output_token_logprobs_val[metadata_index][0] = vals[-1]
            if idxs:
                buffers.output_token_logprobs_idx[metadata_index][0] = idxs[-1]
            if top_vals:
                buffers.output_top_logprobs_val[metadata_index][
                    : len(top_vals[-1])
                ] = torch.tensor(top_vals[-1], dtype=torch.float32, device="cpu")
            if top_idxs:
                buffers.output_top_logprobs_idx[metadata_index][
                    : len(top_idxs[-1])
                ] = torch.tensor(top_idxs[-1], dtype=torch.int32, device="cpu")
        if getattr(req, "hidden_states_tensor", None) is not None:
            topk = req.output_topk_p.size(0)
            buffers.output_topk_p[metadata_index, :topk].copy_(req.output_topk_p)
            buffers.output_topk_index[metadata_index, :topk].copy_(
                req.output_topk_index
            )
            buffers.output_hidden_states[metadata_index].copy_(
                req.hidden_states_tensor
            )
        buffers.bootstrap_room[metadata_index, 0] = migration_room

    def _pd_flip_source_pump_transfer(self, session: Dict[str, Any]) -> None:
        entries = session.get("source_entries") or {}
        if not entries:
            return

        transferred = set(session.get("transferred_rids", set()))
        failed = set(session.get("failed_rids", set()))
        for rid, entry in entries.items():
            if entry.get("transferred") or entry.get("failed"):
                continue
            sender = entry["sender"]
            poll = sender.poll()
            if poll == KVPoll.Failed:
                entry["failed"] = True
                failed.add(rid)
                try:
                    sender.failure_exception()
                except Exception as exc:
                    session["last_error"] = str(exc)
                continue
            if poll == KVPoll.WaitingForInput and not entry.get("sent"):
                self._pd_flip_note_timing(entry, "source_waiting_for_input")
                req = entry["req"]
                old_metadata_index = getattr(req, "metadata_buffer_index", -1)
                old_bootstrap_room = getattr(req, "bootstrap_room", None)
                old_sender = getattr(req, "disagg_kv_sender", None)
                try:
                    req.metadata_buffer_index = entry["metadata_index"]
                    req.bootstrap_room = entry["migration_bootstrap_room"]
                    req.disagg_kv_sender = sender
                    metadata_started = time.monotonic()
                    self._pd_flip_set_source_metadata(
                        req,
                        entry["metadata_index"],
                        entry["migration_bootstrap_room"],
                    )
                    self._pd_flip_note_timing(
                        entry, "source_set_metadata", metadata_started
                    )
                    send_started = time.monotonic()
                    self._pd_flip_source_send_initial(entry)
                    self._pd_flip_note_timing(entry, "source_send", send_started)
                    self._pd_flip_note_timing(entry, "source_sent")
                    entry["sent"] = True
                except Exception as exc:
                    entry["failed"] = True
                    failed.add(rid)
                    session["last_error"] = str(exc)
                    if hasattr(sender, "abort"):
                        sender.abort()
                finally:
                    req.metadata_buffer_index = old_metadata_index
                    req.bootstrap_room = old_bootstrap_room
                    req.disagg_kv_sender = old_sender
            poll = sender.poll()
            if poll == KVPoll.Success:
                entry["transferred"] = True
                transferred.add(rid)
                self._pd_flip_note_timing(entry, "source_transferred")
                self._pd_flip_free_source_metadata(entry)

        session["transferred_rids"] = transferred
        session["failed_rids"] = failed
        session["transferred_reqs"] = len(transferred)
        session["failed_reqs"] = len(failed)
        session["pending_reqs"] = max(
            0, len(session.get("manifests", [])) - len(transferred) - len(failed)
        )
        if failed:
            session["state"] = "source_failed"
        elif session["pending_reqs"] == 0 and session.get("state") == "source_started":
            session["state"] = "source_transferred"
        self._pd_flip_source_pump_delta_transfer(session)

    def _pd_flip_free_source_delta_metadata(self, delta: Dict[str, Any]) -> None:
        if delta.get("metadata_freed"):
            return
        metadata_index = delta.get("metadata_index")
        if metadata_index is not None and metadata_index >= 0:
            self.disagg_metadata_buffers.bootstrap_room[metadata_index] = 0
            self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
        delta["metadata_freed"] = True

    def _pd_flip_source_pump_delta_transfer(self, session: Dict[str, Any]) -> None:
        entries = session.get("source_entries") or {}
        delta_entries = {
            rid: entry
            for rid, entry in entries.items()
            if isinstance(entry.get("delta"), dict)
            and not entry["delta"].get("noop")
        }
        if not delta_entries:
            return

        transferred = set(session.get("delta_transferred_rids", set()))
        failed = set(session.get("delta_failed_rids", set()))
        for rid, entry in delta_entries.items():
            delta = entry["delta"]
            if delta.get("transferred") or delta.get("failed"):
                continue
            sender = delta["sender"]
            poll = sender.poll()
            if poll == KVPoll.Failed:
                delta["failed"] = True
                failed.add(rid)
                try:
                    sender.failure_exception()
                except Exception as exc:
                    session["last_error"] = str(exc)
                continue
            if poll == KVPoll.WaitingForInput and not delta.get("sent"):
                req = entry["req"]
                old_metadata_index = getattr(req, "metadata_buffer_index", -1)
                old_bootstrap_room = getattr(req, "bootstrap_room", None)
                old_sender = getattr(req, "disagg_kv_sender", None)
                try:
                    req.metadata_buffer_index = delta["metadata_index"]
                    req.bootstrap_room = delta["migration_bootstrap_room"]
                    req.disagg_kv_sender = sender
                    self._pd_flip_set_source_metadata(
                        req,
                        delta["metadata_index"],
                        delta["migration_bootstrap_room"],
                    )
                    page_indices = self._pd_flip_source_page_indices_range(
                        req, delta["from_len"], delta["to_len"]
                    )
                    delta["source_index_shape"] = list(page_indices.shape)
                    delta["source_index_size"] = int(page_indices.size)
                    state_indices = self._pd_flip_source_state_indices(
                        req, delta["to_len"], sender.kv_mgr
                    )
                    sender.send(page_indices, state_indices)
                    delta["sent"] = True
                    self._pd_flip_note_timing(entry, "source_delta_sent")
                except Exception as exc:
                    delta["failed"] = True
                    failed.add(rid)
                    session["last_error"] = str(exc)
                    if hasattr(sender, "abort"):
                        sender.abort()
                finally:
                    req.metadata_buffer_index = old_metadata_index
                    req.bootstrap_room = old_bootstrap_room
                    req.disagg_kv_sender = old_sender
            poll = sender.poll()
            if poll == KVPoll.Success:
                delta["transferred"] = True
                transferred.add(rid)
                entry["committed_len"] = int(delta["to_len"])
                entry["manifest"] = delta.get("manifest") or entry.get("manifest")
                self._pd_flip_note_timing(entry, "source_delta_transferred")
                self._pd_flip_free_source_delta_metadata(delta)

        session["delta_transferred_rids"] = transferred
        session["delta_failed_rids"] = failed
        session["delta_transferred_reqs"] = len(transferred)
        session["delta_failed_reqs"] = len(failed)
        session["pending_reqs"] = max(
            0, len(delta_entries) - len(transferred) - len(failed)
        )
        session["delta_pending_reqs"] = session["pending_reqs"]
        if failed:
            session["state"] = "source_failed"
            session["failed_reqs"] = max(
                int(session.get("failed_reqs", 0) or 0), len(failed)
            )
        elif session["pending_reqs"] == 0:
            session["state"] = "source_delta_transferred"

    def _pd_flip_free_source_metadata(self, entry: Dict[str, Any]) -> None:
        if entry.get("metadata_freed"):
            return
        metadata_index = entry.get("metadata_index")
        if metadata_index is not None and metadata_index >= 0:
            self.disagg_metadata_buffers.bootstrap_room[metadata_index] = 0
            self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
        entry["metadata_freed"] = True

    @staticmethod
    def _pd_flip_defer_source_kv_release(req: Req) -> None:
        req.pd_flip_defer_kv_release = True
        req.pd_flip_force_kv_release = False
        req.pd_flip_kv_release_deferred = False
        req.pd_flip_deferred_kv_release_is_insert = False

    @staticmethod
    def _pd_flip_clear_source_kv_release_defer(req: Optional[Req]) -> None:
        if req is None:
            return
        req.pd_flip_defer_kv_release = False
        req.pd_flip_force_kv_release = False

    def _pd_flip_finish_source_kv_release_defer(
        self, entry: Dict[str, Any]
    ) -> None:
        req = entry.get("req")
        if req is None:
            return
        was_deferred = bool(getattr(req, "pd_flip_kv_release_deferred", False))
        is_insert = bool(getattr(req, "pd_flip_deferred_kv_release_is_insert", False))
        self._pd_flip_clear_source_kv_release_defer(req)
        if was_deferred and getattr(req, "req_pool_idx", None) is not None:
            req.pd_flip_force_kv_release = True
            try:
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)
            finally:
                req.pd_flip_force_kv_release = False
        req.pd_flip_kv_release_deferred = False
        req.pd_flip_deferred_kv_release_is_insert = False

    def _pd_flip_release_source_requests(
        self, session: Dict[str, Any], released_rids: set
    ) -> None:
        for rid in released_rids:
            entry = (session.get("source_entries") or {}).get(rid)
            if not entry:
                continue
            req = entry.get("req")
            source_queue = entry.get("source_queue") or (
                entry.get("manifest") or {}
            ).get("pd_flip_source_queue")
            if source_queue == "waiting":
                if req is not None:
                    self._pd_flip_clear_source_kv_release_defer(req)
                    req.pd_flip_force_kv_release = True
                    try:
                        release_kv_cache(req, self.tree_cache)
                    finally:
                        req.pd_flip_force_kv_release = False
                    req.pd_flip_kv_release_deferred = False
                    req.pd_flip_deferred_kv_release_is_insert = False
                entry["request_released"] = True
                self._pd_flip_note_timing(entry, "source_waiting_released")
                self._pd_flip_free_source_metadata(entry)
                continue
            if req is not None and not req.finished():
                self._pd_flip_clear_source_kv_release_defer(req)
                req.pd_flip_migrated_to_target = True
                req.pd_flip_waiting_for_relay_output = True
                req.to_finish = FINISH_MIGRATED()
                self._pd_flip_note_timing(entry, "source_finish_migrated")
            else:
                self._pd_flip_finish_source_kv_release_defer(entry)
                self._pd_flip_note_timing(entry, "source_finish_released")
            self._pd_flip_free_source_metadata(entry)

    def _pd_flip_prepare_target_entries(
        self, manifests: List[Dict[str, Any]], source_url: Optional[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        if not self._pd_flip_can_use_real_migration():
            return {}, ""
        if getattr(self, "enable_hisparse", False):
            return {}, "PD flip migration does not support HiSparse decode yet"
        try:
            source_host = self._pd_flip_source_host_from_url(source_url)
            entries: Dict[str, Dict[str, Any]] = {}
            receiver_class = get_kv_class(self.transfer_backend, KVClassType.RECEIVER)
            kv_manager = self.disagg_decode_prealloc_queue.kv_manager
            for manifest in manifests:
                entry_started = time.monotonic()
                timing_debug = {}
                req_started = time.monotonic()
                req = self._pd_flip_manifest_to_req(manifest, source_host)
                timing_debug["manifest_to_req_s"] = time.monotonic() - req_started
                receiver_started = time.monotonic()
                receiver = receiver_class(
                    mgr=kv_manager,
                    bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
                    bootstrap_room=req.bootstrap_room,
                )
                timing_debug["receiver_create_s"] = time.monotonic() - receiver_started
                decode_req = DecodeRequest(req=req, kv_receiver=receiver)
                entries[req.rid] = {
                    "decode_req": decode_req,
                    "phase": "new",
                    "manifest": manifest,
                    "source_queue": manifest.get("pd_flip_source_queue", "running"),
                    "metadata_index": -1,
                    "timing_debug": timing_debug,
                }
                timing_debug["target_entry_prepare_total_s"] = (
                    time.monotonic() - entry_started
                )
            return entries, ""
        except Exception as exc:
            return {}, str(exc)

    def _pd_flip_source_host_from_url(self, source_url: Optional[str]) -> str:
        if not source_url:
            raise ValueError("source_url is required for real migration target prepare")
        parsed = urlparse(source_url)
        if parsed.hostname:
            return parsed.hostname
        return parsed.path.split(":", 1)[0]

    def _pd_flip_manifest_to_req(
        self, manifest: Dict[str, Any], source_host: str
    ) -> Req:
        sampling_params = self._pd_flip_deserialize_sampling_params(
            manifest.get("sampling_params") or {}
        )
        try:
            sampling_params.normalize(getattr(self, "tokenizer", None))
            sampling_params.verify(self.model_config.vocab_size)
        except Exception:
            logger.debug("Skipping sampling param normalize/verify for migration", exc_info=True)

        req = Req(
            str(manifest.get("rid", "")),
            "",
            array("q", list(manifest.get("origin_input_ids") or [])),
            sampling_params,
            return_logprob=bool(manifest.get("return_logprob", False)),
            stream=bool(manifest.get("stream", False)),
            eos_token_ids=self.model_config.hf_eos_token_id,
            bootstrap_host=source_host,
            bootstrap_port=int(
                manifest.get("source_bootstrap_port")
                or getattr(self.server_args, "disaggregation_bootstrap_port", 8998)
            ),
            bootstrap_room=int(
                manifest.get("migration_bootstrap_room")
                or manifest.get("bootstrap_room")
                or 0
            ),
            disagg_mode=self.disaggregation_mode,
            vocab_size=self.model_config.vocab_size,
            priority=manifest.get("priority"),
            metrics_collector=(
                self.metrics_collector
                if getattr(self.metrics_reporter, "enable_metrics", False)
                else None
            ),
            routing_key=manifest.get("routing_key"),
            extra_key=manifest.get("extra_key"),
            http_worker_ipc=manifest.get("http_worker_ipc"),
            time_stats=self._pd_flip_deserialize_time_stats(
                manifest.get("time_stats")
            ),
        )
        req.output_ids = array("q", list(manifest.get("output_ids") or []))
        req.send_token_offset = len(req.output_ids)
        req.send_decode_id_offset = len(req.output_ids)
        req.logprob_start_len = int(manifest.get("logprob_start_len", -1))
        req.kv_committed_len = int(
            manifest.get("kv_committed_len")
            or len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        )
        req.pd_flip_migration_session_id = manifest.get("pd_flip_session_id")
        req.pd_flip_last_emitted_output_seq = int(
            manifest.get("last_emitted_output_seq", 0) or 0
        )
        return req

    @staticmethod
    def _pd_flip_serialize_time_stats(time_stats) -> Dict[str, Any]:
        if time_stats is None:
            return {}
        values = getattr(time_stats, "__dict__", {})
        return {
            key: value
            for key, value in values.items()
            if key not in ("metrics_collector", "trace_ctx", "disagg_mode")
            and (
                isinstance(value, (str, int, float, bool))
                or value is None
                or (
                    isinstance(value, (list, tuple))
                    and all(
                        isinstance(item, (str, int, float, bool)) or item is None
                        for item in value
                    )
                )
            )
        }

    @staticmethod
    def _pd_flip_deserialize_time_stats(values: Optional[Dict[str, Any]]):
        if not values or not isinstance(values, dict):
            return None
        return SimpleNamespace(**values)

    @staticmethod
    def _pd_flip_deserialize_sampling_params(values: Dict[str, Any]) -> SamplingParams:
        allowed = set(SamplingParams.__init__.__code__.co_varnames)
        allowed.discard("self")
        kwargs = {key: value for key, value in dict(values).items() if key in allowed}
        return SamplingParams(**kwargs)

    def _pd_flip_apply_delta_manifest_to_target(
        self, entry: Dict[str, Any], manifest: Dict[str, Any]
    ) -> None:
        decode_req = entry.get("decode_req")
        req = getattr(decode_req, "req", None)
        if req is None:
            return
        req.output_ids = array("q", list(manifest.get("output_ids") or []))
        req.send_token_offset = len(req.output_ids)
        req.send_decode_id_offset = len(req.output_ids)
        req.logprob_start_len = int(manifest.get("logprob_start_len", -1))
        committed_len = int(
            manifest.get("kv_committed_len")
            or len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        )
        req.kv_committed_len = committed_len
        req.kv_allocated_len = max(int(getattr(req, "kv_allocated_len", 0) or 0), committed_len)
        req.pd_flip_migration_session_id = manifest.get("pd_flip_session_id")
        req.pd_flip_last_emitted_output_seq = int(
            manifest.get("last_emitted_output_seq", 0) or 0
        )
        entry["committed_len"] = committed_len
        entry["manifest"] = manifest
        self._pd_flip_note_timing(entry, "target_delta_state_applied")

    def _pd_flip_start_target_delta_entry(
        self,
        entry: Dict[str, Any],
        manifest: Dict[str, Any],
        source_url: Optional[str],
    ) -> None:
        decode_req = entry.get("decode_req")
        req = getattr(decode_req, "req", None)
        if req is None:
            raise ValueError("target delta entry has no request")
        receiver_class = get_kv_class(self.transfer_backend, KVClassType.RECEIVER)
        source_host = self._pd_flip_source_host_from_url(
            source_url or (entry.get("manifest") or {}).get("source_url") or ""
        )
        migration_room = int(manifest["migration_bootstrap_room"])
        receiver = receiver_class(
            mgr=self.disagg_decode_prealloc_queue.kv_manager,
            bootstrap_addr=f"{source_host}:{int(manifest.get('source_bootstrap_port') or getattr(self.server_args, 'disaggregation_bootstrap_port', 8998))}",
            bootstrap_room=migration_room,
        )
        req.bootstrap_host = source_host
        req.bootstrap_port = int(
            manifest.get("source_bootstrap_port")
            or getattr(self.server_args, "disaggregation_bootstrap_port", 8998)
        )
        req.bootstrap_room = migration_room
        decode_req.kv_receiver = receiver
        entry["delta"] = {
            "phase": "new",
            "manifest": manifest,
            "from_len": int(manifest.get("delta_from_len") or entry.get("committed_len") or 0),
            "to_len": int(manifest.get("kv_committed_len") or 0),
            "metadata_index": -1,
            "transferred": False,
            "failed": False,
        }

    def _pd_flip_target_delta_prealloc_and_send_metadata(
        self, entry: Dict[str, Any]
    ) -> None:
        delta = entry["delta"]
        manifest = delta["manifest"]
        decode_req: DecodeRequest = entry["decode_req"]
        req = decode_req.req
        from_len = int(delta["from_len"])
        to_len = int(delta["to_len"])
        if to_len < from_len:
            raise RuntimeError(
                f"target delta regressed committed length for {req.rid}: {from_len}->{to_len}"
            )

        metadata_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        if metadata_index is None:
            raise RuntimeError("no metadata buffer available for target delta migration")
        delta["metadata_index"] = metadata_index
        entry["metadata_index"] = metadata_index
        decode_req.metadata_buffer_index = metadata_index

        delta_len = to_len - from_len
        if delta_len > 0:
            device = self.token_to_kv_pool_allocator.device
            if from_len > 0:
                last_loc = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, from_len - 1
                ].reshape(1).to(dtype=torch.int64, device=device)
            else:
                last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

            if self.token_to_kv_pool_allocator.page_size == 1:
                kv_loc = self.token_to_kv_pool_allocator.alloc(delta_len)
            else:
                kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                    prefix_lens=torch.tensor([from_len], dtype=torch.int64, device=device),
                    prefix_lens_cpu=torch.tensor([from_len], dtype=torch.int64),
                    seq_lens=torch.tensor([to_len], dtype=torch.int64, device=device),
                    seq_lens_cpu=torch.tensor([to_len], dtype=torch.int64),
                    last_loc=last_loc,
                    extend_num_tokens=delta_len,
                )
            if kv_loc is None:
                raise RuntimeError("KV cache is full during target delta migration")
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(from_len, from_len + len(kv_loc))),
                kv_loc,
            )

        req.kv_committed_len = to_len
        req.kv_allocated_len = max(int(getattr(req, "kv_allocated_len", 0) or 0), to_len)
        page_start = int(
            manifest.get("delta_page_start_len")
            or self._pd_flip_delta_page_start(from_len)
        )
        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, page_start:to_len]
            .cpu()
            .numpy()
        )
        page_indices = kv_to_page_indices(
            kv_indices,
            self.token_to_kv_pool_allocator.page_size,
        )
        delta["target_index_shape"] = list(page_indices.shape)
        delta["target_index_size"] = int(page_indices.size)
        state_indices = self._pd_flip_target_state_indices(req, to_len)
        decode_req.kv_receiver.send_metadata(
            page_indices,
            metadata_index,
            state_indices,
            decode_prefix_len=page_start,
        )
        if self.transfer_backend == TransferBackend.FAKE:
            self.disagg_metadata_buffers.bootstrap_room[metadata_index, 0] = (
                req.bootstrap_room
            )

    def _pd_flip_target_pump_delta_transfer(self, session: Dict[str, Any]) -> None:
        entries = session.get("target_entries") or {}
        delta_entries = {
            rid: entry
            for rid, entry in entries.items()
            if isinstance(entry.get("delta"), dict)
            and not entry["delta"].get("noop")
        }
        if not delta_entries:
            return

        transferred = set(session.get("delta_transferred_rids", set()))
        failed = set(session.get("delta_failed_rids", set()))
        for rid, entry in delta_entries.items():
            delta = entry["delta"]
            if delta.get("transferred") or delta.get("failed"):
                continue
            decode_req = entry["decode_req"]
            try:
                if delta.get("phase") == "new":
                    if not self._pd_flip_target_init_receiver(decode_req):
                        continue
                    delta["phase"] = "waiting_for_input"

                if delta.get("phase") == "waiting_for_input":
                    poll = decode_req.kv_receiver.poll()
                    if poll == KVPoll.Failed:
                        raise RuntimeError("migration target delta bootstrap failed")
                    if poll != KVPoll.WaitingForInput:
                        continue
                    self._pd_flip_target_delta_prealloc_and_send_metadata(entry)
                    delta["phase"] = "transferring"

                if delta.get("phase") == "transferring":
                    poll = decode_req.kv_receiver.poll()
                    if poll == KVPoll.Failed:
                        raise RuntimeError("migration target delta transfer failed")
                    if poll == KVPoll.Success and self._pd_flip_target_metadata_ready(
                        entry
                    ):
                        self._pd_flip_apply_delta_manifest_to_target(
                            entry, delta["manifest"]
                        )
                        transferred.add(rid)
                        delta["transferred"] = True
                        delta["phase"] = "transferred"
                        entry["phase"] = "transferred_held"
                        entry["held"] = not bool(entry.get("drop_on_commit"))
                        if getattr(decode_req, "kv_receiver", None) is not None:
                            decode_req.kv_receiver.clear()
                            decode_req.kv_receiver = None
                        self._pd_flip_free_target_metadata(entry)
            except Exception as exc:
                failed.add(rid)
                delta["failed"] = True
                delta["phase"] = "failed"
                entry["phase"] = "failed"
                session["last_error"] = str(exc)
                if getattr(decode_req, "kv_receiver", None) is not None:
                    decode_req.kv_receiver.abort()
                self._pd_flip_free_target_metadata(entry)

        session["delta_transferred_rids"] = transferred
        session["delta_failed_rids"] = failed
        session["delta_transferred_reqs"] = len(transferred)
        session["delta_failed_reqs"] = len(failed)
        session["pending_reqs"] = max(
            0, len(delta_entries) - len(transferred) - len(failed)
        )
        session["delta_pending_reqs"] = session["pending_reqs"]
        if failed:
            session["state"] = "target_failed"
            session["failed_reqs"] = max(
                int(session.get("failed_reqs", 0) or 0), len(failed)
            )
        elif session["pending_reqs"] == 0:
            session["state"] = "target_delta_transferred"

    def _pd_flip_target_pump_transfer(self, session: Dict[str, Any]) -> None:
        entries = session.get("target_entries") or {}
        if not entries:
            return

        transferred = set(session.get("transferred_rids", set()))
        failed = set(session.get("failed_rids", set()))
        for rid, entry in entries.items():
            if rid in transferred or rid in failed:
                continue
            decode_req = entry["decode_req"]
            phase = entry.get("phase")
            try:
                if phase == "new":
                    init_started = time.monotonic()
                    if not self._pd_flip_target_init_receiver(decode_req):
                        continue
                    self._pd_flip_note_timing(entry, "target_init_receiver", init_started)
                    entry["phase"] = "waiting_for_input"
                    self._pd_flip_note_timing(entry, "target_waiting_for_input")
                    phase = "waiting_for_input"

                if phase == "waiting_for_input":
                    poll = decode_req.kv_receiver.poll()
                    if poll == KVPoll.Failed:
                        raise RuntimeError("migration target bootstrap failed")
                    if poll != KVPoll.WaitingForInput:
                        continue
                    metadata_started = time.monotonic()
                    self._pd_flip_target_prealloc_and_send_metadata(entry)
                    self._pd_flip_note_timing(
                        entry, "target_prealloc_send_metadata", metadata_started
                    )
                    entry["phase"] = "transferring"
                    self._pd_flip_note_timing(entry, "target_transferring")

                if entry.get("phase") == "transferring":
                    if self._pd_flip_target_hicache_restore_pending(decode_req):
                        continue
                    poll = decode_req.kv_receiver.poll()
                    if poll == KVPoll.Failed:
                        raise RuntimeError("migration target transfer failed")
                    if (
                        poll == KVPoll.Success
                        and self._pd_flip_target_metadata_ready(entry)
                        and self._pd_flip_target_stitch_ready(entry)
                    ):
                        if not session.get("prepare_only", False):
                            self._pd_flip_target_commit_hicache_restore(decode_req)
                        self._pd_flip_note_timing(entry, "target_transfer_success")
                        transferred.add(rid)
                        entry["phase"] = (
                            "transferred_held"
                            if session.get("prepare_only", False)
                            else "transferred"
                        )
                        if getattr(decode_req, "kv_receiver", None) is not None:
                            decode_req.kv_receiver.clear()
                            decode_req.kv_receiver = None
                        if session.get("prepare_only", False):
                            entry["held"] = True
                            self._pd_flip_note_timing(entry, "target_held")
                        elif session.get("adopt_on_success", False):
                            self._pd_flip_adopt_target_request(entry)
                        else:
                            self._pd_flip_release_target_request(entry)
                        self._pd_flip_free_target_metadata(entry)
            except Exception as exc:
                failed.add(rid)
                entry["phase"] = "failed"
                session["last_error"] = str(exc)
                if getattr(decode_req, "kv_receiver", None) is not None:
                    decode_req.kv_receiver.abort()
                self._pd_flip_release_target_request(entry)
                self._pd_flip_free_target_metadata(entry)

        session["transferred_rids"] = transferred
        session["failed_rids"] = failed
        session["transferred_reqs"] = len(transferred)
        session["failed_reqs"] = len(failed)
        session["held_reqs"] = sum(
            1
            for entry in entries.values()
            if entry.get("phase") == "transferred_held" and entry.get("held")
        )
        session["pending_reqs"] = max(
            0, len(session.get("manifests", [])) - len(transferred) - len(failed)
        )
        if failed:
            session["state"] = "target_failed"
        elif session.get("held_reqs", 0) > 0:
            session["state"] = "target_transferred_held"
        elif session["pending_reqs"] == 0:
            session["state"] = "target_transferred"
        self._pd_flip_target_pump_delta_transfer(session)

    def _pd_flip_target_hicache_restore_pending(
        self, decode_req: DecodeRequest
    ) -> bool:
        prefix_match = getattr(decode_req, "prefix_match", None)
        if prefix_match is None or not getattr(
            prefix_match, "needs_local_restore", False
        ):
            return False

        if not getattr(self, "enable_decode_hicache", False):
            raise RuntimeError(
                "migration target requires HiCache restore but decode HiCache is disabled"
            )

        transfer_queue = getattr(self, "disagg_decode_transfer_queue", None)
        if transfer_queue is None or not hasattr(
            transfer_queue, "_process_hicache_local_restores"
        ):
            raise RuntimeError(
                "migration target requires HiCache restore but no restore processor is available"
            )
        transfer_queue._process_hicache_local_restores([decode_req])
        status_value = getattr(
            getattr(decode_req, "hicache_restore_status", None), "value", None
        )
        if status_value == "failed":
            raise RuntimeError("migration target HiCache restore failed")
        if status_value == "pending":
            return True
        if status_value == "ready":
            return False
        raise RuntimeError(
            f"migration target HiCache restore returned invalid status {status_value!r}"
        )

    def _pd_flip_target_stitch_ready(self, entry: Dict[str, Any]) -> bool:
        if not getattr(
            getattr(self, "server_args", None),
            "enable_pd_flip_hicache_stitch",
            False,
        ):
            return True
        try:
            hit_len = int(entry["mooncake_hit_len"])
            prompt_len = int(entry["target_prompt_len"])
            committed_len = int(entry["target_committed_len"])
            suffix_start = int(entry["target_received_suffix_start"])
            suffix_end = int(entry["target_received_suffix_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "migration target stitch metadata H/P/C0/coverage is missing or invalid"
            ) from exc
        if not 0 <= hit_len <= prompt_len <= committed_len:
            raise RuntimeError(
                f"migration target stitch boundary is invalid: "
                f"H={hit_len}, P={prompt_len}, C0={committed_len}"
            )
        if suffix_start != hit_len or suffix_end != committed_len:
            raise RuntimeError(
                f"migration target suffix coverage mismatch: "
                f"received=[{suffix_start},{suffix_end}), "
                f"expected=[{hit_len},{committed_len})"
            )
        prefix_match = getattr(entry["decode_req"], "prefix_match", None)
        if prefix_match is None or not getattr(
            prefix_match, "needs_local_restore", False
        ):
            return True
        status_value = getattr(
            getattr(entry["decode_req"], "hicache_restore_status", None),
            "value",
            None,
        )
        if status_value != "ready":
            raise RuntimeError(
                f"migration target HiCache restore is not ready: {status_value!r}"
            )
        return True

    def _pd_flip_target_commit_hicache_restore(self, decode_req: DecodeRequest) -> None:
        if not getattr(self, "enable_decode_hicache", False):
            return
        prefix_match = getattr(decode_req, "prefix_match", None)
        if prefix_match is None or not getattr(
            prefix_match, "needs_local_restore", False
        ):
            return

        status_value = getattr(
            getattr(decode_req, "hicache_restore_status", None), "value", None
        )
        if status_value == "failed":
            raise RuntimeError("migration target HiCache restore failed")
        if status_value == "pending":
            raise RuntimeError("migration target HiCache restore is still pending")

        transfer_queue = getattr(self, "disagg_decode_transfer_queue", None)
        if transfer_queue is None or not hasattr(
            transfer_queue, "_commit_hicache_local_restore_to_req"
        ):
            raise RuntimeError("migration target HiCache restore cannot be committed")
        transfer_queue._commit_hicache_local_restore_to_req(decode_req)

    def _pd_flip_target_init_receiver(self, decode_req: DecodeRequest) -> bool:
        queue = self.disagg_decode_prealloc_queue
        if self.transfer_backend == TransferBackend.FAKE:
            decode_req.kv_receiver.init(0)
            return True

        addr = f"{decode_req.req.bootstrap_host}:{decode_req.req.bootstrap_port}"
        if not queue.kv_manager.try_ensure_parallel_info(addr):
            return False
        prefill_dp_rank = queue._resolve_prefill_dp_rank(decode_req.req)
        if prefill_dp_rank is None:
            room_to_rank = decode_req.kv_receiver.query_prefill_dp_ranks(
                addr, [decode_req.req.bootstrap_room]
            )
            prefill_dp_rank = room_to_rank.get(str(decode_req.req.bootstrap_room))
        if prefill_dp_rank is None:
            return False
        decode_req.kv_receiver.init(int(prefill_dp_rank))
        return True

    def _pd_flip_target_prealloc_and_send_metadata(
        self, entry: Dict[str, Any]
    ) -> None:
        decode_req: DecodeRequest = entry["decode_req"]
        req = decode_req.req
        queue = self.disagg_decode_prealloc_queue
        metadata_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        if metadata_index is None:
            raise RuntimeError("no metadata buffer available for target migration")
        entry["metadata_index"] = metadata_index
        decode_req.metadata_buffer_index = metadata_index

        committed_len = int(req.kv_committed_len)
        prefix_match = None
        prefix_indices = None
        prefix_len = 0
        total_prefix_len = 0
        if getattr(self.server_args, "enable_pd_flip_hicache_stitch", False):
            prefix_started = time.monotonic()
            prefix_match = queue._match_prefix_and_lock(req)
            raw_prefix_len = int(prefix_match.decode_prefix_len)
            prompt_len = len(req.origin_input_ids)
            total_prefix_len, stitch_mode = self._pd_flip_stitch_boundary(
                raw_prefix_len,
                prompt_len,
                self.token_to_kv_pool_allocator.page_size,
            )
            prefix_len = min(int(prefix_match.l1_prefix_len), total_prefix_len)
            prefix_match.prefix_indices = prefix_match.prefix_indices[:prefix_len]
            remaining_prefix = total_prefix_len - prefix_len
            if hasattr(prefix_match, "l2_host_hit_length"):
                prefix_match.l2_host_hit_length = min(
                    int(prefix_match.l2_host_hit_length), remaining_prefix
                )
                remaining_prefix -= prefix_match.l2_host_hit_length
            if hasattr(prefix_match, "l3_storage_hit_length"):
                prefix_match.l3_storage_hit_length = min(
                    int(prefix_match.l3_storage_hit_length), remaining_prefix
                )
            prefix_indices = prefix_match.prefix_indices
            entry["target_hicache_prefix_match_s"] = time.monotonic() - prefix_started
            entry["target_hicache_l1_prefix_len"] = prefix_len
            entry["target_hicache_prefix_len"] = total_prefix_len
            entry["target_hicache_restore_tokens"] = max(
                0, total_prefix_len - prefix_len
            )
            decode_req.prefix_match = prefix_match
            entry.update(
                mooncake_hit_len=total_prefix_len,
                target_prompt_len=prompt_len,
                target_committed_len=committed_len,
                target_received_suffix_start=total_prefix_len,
                target_received_suffix_end=committed_len,
                stitch_mode=stitch_mode,
            )

        if prefix_indices is None:
            dst_kv_indices = queue._pre_alloc(
                req, prefix_len=prefix_len, total_prefix_len=total_prefix_len
            )
        else:
            dst_kv_indices = queue._pre_alloc(
                req,
                prefix_indices=prefix_indices,
                prefix_len=prefix_len,
                total_prefix_len=total_prefix_len,
            )
        if prefix_match is not None and getattr(self, "enable_decode_hicache", False):
            queue._start_hicache_prefetch(req, prefix_match)

        if getattr(self.server_args, "enable_pd_flip_hicache_stitch", False):
            page_indices = self._pd_flip_stitch_page_indices_range(
                req, total_prefix_len, committed_len
            )
        elif self.server_args.disaggregation_decode_enable_radix_cache:
            kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                total_prefix_len:committed_len
            ].cpu().numpy()
            page_indices = kv_to_page_indices(
                kv_indices,
                self.token_to_kv_pool_allocator.page_size,
            )
        else:
            kv_indices = dst_kv_indices.cpu().numpy()
            page_indices = kv_to_page_indices(
                kv_indices,
                self.token_to_kv_pool_allocator.page_size,
            )
        entry["target_index_shape"] = list(page_indices.shape)
        entry["target_index_size"] = int(page_indices.size)
        state_indices = self._pd_flip_target_state_indices(req, committed_len)
        decode_req.kv_receiver.send_metadata(
            page_indices,
            metadata_index,
            state_indices,
            decode_prefix_len=total_prefix_len,
        )
        if self.transfer_backend == TransferBackend.FAKE:
            self.disagg_metadata_buffers.bootstrap_room[metadata_index, 0] = (
                req.bootstrap_room
            )

    def _pd_flip_target_state_indices(self, req: Req, committed_len: int) -> List:
        page_size = self.token_to_kv_pool_allocator.page_size

        def _mamba_payload():
            return [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]

        def _swa_payload():
            window_size = self.sliding_window_size
            window_start = max(0, committed_len - window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:committed_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            return kv_to_page_indices(window_kv_indices_swa.cpu().numpy(), page_size)

        def _dsa_payload():
            kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :committed_len
            ]
            return kv_to_page_indices(kv_indices_full.cpu().numpy(), page_size)

        state_indices: List = []
        for st in getattr(
            self.disagg_decode_prealloc_queue.kv_manager.kv_args, "state_types", []
        ):
            if st == StateType.MAMBA:
                state_indices.append(_mamba_payload())
            elif st == StateType.SWA:
                state_indices.append(_swa_payload())
            elif st == StateType.DSA:
                state_indices.append(_dsa_payload())
            else:
                state_indices.append(None)
        return state_indices

    def _pd_flip_target_metadata_ready(self, entry: Dict[str, Any]) -> bool:
        metadata_index = entry.get("metadata_index", -1)
        if metadata_index is None or metadata_index < 0:
            return True
        expected_room = entry["decode_req"].req.bootstrap_room
        actual_room = self.disagg_metadata_buffers.bootstrap_room[
            metadata_index, 0
        ].item()
        if actual_room == 0:
            return False
        if actual_room != expected_room:
            raise RuntimeError(
                f"migration metadata room mismatch: expected {expected_room}, got {actual_room}"
            )
        return True

    def _pd_flip_free_target_metadata(self, entry: Dict[str, Any]) -> None:
        metadata_index = entry.get("metadata_index", -1)
        if metadata_index is not None and metadata_index >= 0:
            self.disagg_metadata_buffers.bootstrap_room[metadata_index] = 0
            self.req_to_metadata_buffer_idx_allocator.free(metadata_index)
            entry["metadata_index"] = -1

    def _pd_flip_release_target_request(self, entry: Dict[str, Any]) -> None:
        if entry.get("request_released"):
            return
        decode_req = entry.get("decode_req")
        req = getattr(decode_req, "req", None)
        if req is None or getattr(req, "req_pool_idx", None) is None:
            entry["request_released"] = True
            self._pd_flip_note_timing(entry, "target_released")
            return
        release_kv_cache(req, self.tree_cache, is_insert=False)
        entry["request_released"] = True
        self._pd_flip_note_timing(entry, "target_released")

    def _pd_flip_adopt_target_request(self, entry: Dict[str, Any]) -> None:
        if entry.get("request_adopted"):
            return
        decode_req = entry.get("decode_req")
        req = getattr(decode_req, "req", None)
        if req is None:
            entry["request_adopted"] = True
            self._pd_flip_note_timing(entry, "target_adopted")
            return
        req.init_next_round_input(self.tree_cache)
        if hasattr(req, "time_stats") and hasattr(
            req.time_stats, "set_wait_queue_entry_time"
        ):
            req.time_stats.set_wait_queue_entry_time()
        self.waiting_queue.append(req)
        entry["request_adopted"] = True
        self._pd_flip_note_timing(entry, "target_adopted")

    def _pd_flip_build_migration_manifest(self, req: Req) -> Dict[str, Any]:
        origin_input_ids = list(getattr(req, "origin_input_ids", []) or [])
        output_ids = list(getattr(req, "output_ids", []) or [])
        kv_committed_len = getattr(req, "kv_committed_len", None)
        if kv_committed_len is None:
            kv_committed_len = len(origin_input_ids) + max(0, len(output_ids) - 1)
        return {
            "rid": getattr(req, "rid", ""),
            "origin_input_ids": origin_input_ids,
            "output_ids": output_ids,
            "bootstrap_room": getattr(req, "bootstrap_room", None),
            "priority": getattr(req, "priority", None),
            "routing_key": getattr(req, "routing_key", None),
            "extra_key": getattr(req, "extra_key", None),
            "http_worker_ipc": getattr(req, "http_worker_ipc", None),
            "stream": bool(getattr(req, "stream", False)),
            "return_logprob": bool(getattr(req, "return_logprob", False)),
            "logprob_start_len": getattr(req, "logprob_start_len", -1),
            "time_stats": self._pd_flip_serialize_time_stats(
                getattr(req, "time_stats", None)
            ),
            "req_pool_idx": getattr(req, "req_pool_idx", None),
            "kv_committed_len": int(kv_committed_len),
            "sampling_params": self._pd_flip_serialize_sampling_params(
                getattr(req, "sampling_params", None)
            ),
            "last_emitted_output_seq": int(
                getattr(req, "pd_flip_last_emitted_output_seq", 0) or 0
            ),
            "pd_flip_session_id": getattr(req, "pd_flip_migration_session_id", None),
        }

    @staticmethod
    def _pd_flip_serialize_sampling_params(sampling_params) -> Dict[str, Any]:
        if sampling_params is None:
            return {}
        for method_name in ("to_json", "to_dict"):
            method = getattr(sampling_params, method_name, None)
            if callable(method):
                value = method()
                if isinstance(value, dict):
                    return Scheduler._pd_flip_json_safe_dict(value)
        return Scheduler._pd_flip_json_safe_dict(
            getattr(sampling_params, "__dict__", {})
        )

    @staticmethod
    def _pd_flip_json_safe_dict(values: Dict[str, Any]) -> Dict[str, Any]:
        safe = {}
        for key, value in dict(values).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, (list, tuple)):
                if all(
                    isinstance(item, (str, int, float, bool)) or item is None
                    for item in value
                ):
                    safe[key] = list(value)
        return safe

    def _pd_flip_migration_status_dict(self) -> Dict[str, Any]:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session:
            return {
                "enabled": False,
                "role": "none",
                "state": "none",
                "session_id": None,
                "pending_reqs": 0,
                "transferred_reqs": 0,
                "released_reqs": 0,
                "failed_reqs": 0,
                "held_reqs": 0,
                "last_error": "",
                "dry_run": False,
                "prepare_only": False,
                "waiting_reqs": 0,
                "waiting_manifest_count": 0,
                "waiting_skipped_count": 0,
                "waiting_skipped": [],
                "delta_pending_reqs": 0,
                "delta_transferred_reqs": 0,
                "delta_failed_reqs": 0,
            }
        session_timing = session.get("timing_debug") or {}
        return {
            "enabled": True,
            "role": session.get("role", "none"),
            "state": session.get("state", "none"),
            "session_id": session.get("session_id"),
            "pending_reqs": int(session.get("pending_reqs", 0)),
            "transferred_reqs": int(session.get("transferred_reqs", 0)),
            "released_reqs": int(session.get("released_reqs", 0)),
            "failed_reqs": int(session.get("failed_reqs", 0)),
            "held_reqs": int(session.get("held_reqs", 0)),
            "last_error": session.get("last_error", ""),
            "dry_run": bool(session.get("dry_run", False)),
            "prepare_only": bool(session.get("prepare_only", False)),
            "waiting_reqs": int(session_timing.get("waiting_reqs", 0) or 0),
            "waiting_manifest_count": int(
                session_timing.get("waiting_manifest_count", 0) or 0
            ),
            "waiting_skipped_count": int(
                session_timing.get("waiting_skipped_count", 0) or 0
            ),
            "waiting_skipped": list(session_timing.get("waiting_skipped") or []),
            "delta_pending_reqs": int(session.get("delta_pending_reqs", 0) or 0),
            "delta_transferred_reqs": int(
                session.get("delta_transferred_reqs", 0) or 0
            ),
            "delta_failed_reqs": int(session.get("delta_failed_reqs", 0) or 0),
            "index_debug": self._pd_flip_migration_index_debug(session),
            "timing_debug": self._pd_flip_migration_timing_debug(session),
        }

    @staticmethod
    def _pd_flip_note_timing(
        container: Dict[str, Any], name: str, started: Optional[float] = None
    ) -> None:
        timing = container.setdefault("timing_debug", {})
        now = time.monotonic()
        if started is None:
            timing.setdefault(f"{name}_mono", now)
        else:
            timing[f"{name}_s"] = now - started
            timing.setdefault(f"{name}_mono", now)

    @staticmethod
    def _pd_flip_json_safe_timing(values: Dict[str, Any]) -> Dict[str, Any]:
        safe = {}
        for key, value in dict(values or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
        return safe

    def _pd_flip_migration_timing_debug(
        self, session: Dict[str, Any]
    ) -> Dict[str, Any]:
        entries = session.get("source_entries") or session.get("target_entries") or {}
        return {
            "session": self._pd_flip_json_safe_timing(
                session.get("timing_debug") or {}
            ),
            "entries": [
                {
                    "rid": rid,
                    "phase": entry.get("phase"),
                    "source_queue": entry.get("source_queue")
                    or (entry.get("manifest") or {}).get("pd_flip_source_queue"),
                    "timing": self._pd_flip_json_safe_timing(
                        entry.get("timing_debug") or {}
                    ),
                }
                for rid, entry in entries.items()
            ],
        }

    @staticmethod
    def _pd_flip_migration_index_debug(session: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries = session.get("source_entries") or session.get("target_entries") or {}
        debug = []
        for rid, entry in entries.items():
            item = {
                "rid": rid,
                "phase": entry.get("phase"),
                "source_queue": entry.get("source_queue")
                or (entry.get("manifest") or {}).get("pd_flip_source_queue"),
                "source_index_shape": entry.get("source_index_shape"),
                "source_index_size": entry.get("source_index_size"),
                "target_index_shape": entry.get("target_index_shape"),
                "target_index_size": entry.get("target_index_size"),
            }
            delta = entry.get("delta") if isinstance(entry.get("delta"), dict) else {}
            if delta:
                item["delta_source_index_shape"] = delta.get("source_index_shape")
                item["delta_source_index_size"] = delta.get("source_index_size")
                item["delta_target_index_shape"] = delta.get("target_index_shape")
                item["delta_target_index_size"] = delta.get("target_index_size")
            if any(value is not None for key, value in item.items() if key != "rid"):
                debug.append(item)
        return debug

    def _pd_flip_migration_is_active(self) -> bool:
        return self._pd_flip_migration_status_dict()["enabled"]

    def _pd_flip_source_migration_blocks_idle(self) -> bool:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session or session.get("role") != "source":
            return False
        if session.get("state") in ("source_released", "source_aborted"):
            return False
        return bool(session.get("manifests") or session.get("source_entries"))

    def _pd_flip_migration_is_released(self) -> bool:
        status = self._pd_flip_migration_status_dict()
        return (
            status["state"] == "source_released"
            and status["pending_reqs"] == 0
            and status["failed_reqs"] == 0
        )

    def _pd_flip_target_held_reqs(self) -> List[Req]:
        session = getattr(self, "pd_flip_migration_session", None)
        if not session or session.get("role") != "target":
            return []

        held_reqs = []
        for entry in (session.get("target_entries") or {}).values():
            phase = entry.get("phase")
            if (
                phase not in ("transferring", "transferred_held", "ready_to_activate")
                or entry.get("request_released")
                or entry.get("request_adopted")
            ):
                continue
            decode_req = entry.get("decode_req")
            req = getattr(decode_req, "req", None)
            if req is not None and getattr(req, "req_pool_idx", None) is not None:
                held_reqs.append(req)
        return held_reqs

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
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
        )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            return

        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.use_ngram_embedding:
            return batch
        batch.ne_token_table = self.token_table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_input_len
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
        return batch

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
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
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (
                    PDFlipMigrationSourceStartReq,
                    self.start_pd_flip_migration_source,
                ),
                (
                    PDFlipMigrationTargetPrepareReq,
                    self.prepare_pd_flip_migration_target,
                ),
                (
                    PDFlipMigrationTargetCommitReq,
                    self.commit_pd_flip_migration_target,
                ),
                (
                    PDFlipMigrationTargetActivateReq,
                    self.activate_pd_flip_migration_target,
                ),
                (
                    PDFlipMigrationTargetAbortReq,
                    self.abort_pd_flip_migration_target,
                ),
                (PDFlipMigrationStatusReq, self.get_pd_flip_migration_status),
                (
                    PDFlipMigrationSourceFinishReq,
                    self.finish_pd_flip_migration_source,
                ),
                (
                    PDFlipMigrationSourceDeltaReq,
                    self.start_pd_flip_migration_source_delta,
                ),
                (
                    PDFlipMigrationTargetDeltaPrepareReq,
                    self.prepare_pd_flip_migration_target_delta,
                ),
                (PDFlipMigrationAbortReq, self.abort_pd_flip_migration),
                (PDRuntimeRoleSetReq, self.set_pd_runtime_role),
                (PDRuntimeRoleStatusReq, self.get_pd_runtime_role_status),
                (PDRuntimeRoleAdmissionReq, self.set_pd_runtime_admission),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadsReqInput, self.handle_get_loads_req),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
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

    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if self.running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
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
        }

        return result_dict

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        self._shutdown_requested = False
        self.active_pd_event_loop_role = None
        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            if self.pd_runtime_role_switch_enabled():
                self._run_pd_dispatch_loop()
            else:
                dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        # DFLASH fences its shared req_to_token writes with verify_done /
        # plan-stream deps, so the global WAR barrier only serializes plan
        # overlap. TODO: generalize this global-barrier enablement policy.
        self._war_barrier_enabled = (
            is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        ) and not self.spec_algorithm.is_dflash()
        with self.device_module.StreamContext(self.schedule_stream):
            if self.pd_runtime_role_switch_enabled():
                self._run_pd_dispatch_loop()
            else:
                dispatch_event_loop(self)

    def _run_pd_dispatch_loop(self) -> None:
        while not getattr(self, "_shutdown_requested", False):
            dispatch_event_loop(self)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # WAR barrier: this iter's schedule writes to shared GPU buffers wait for prev forward's reads.
            if self._war_barrier_enabled:
                self.schedule_stream.wait_stream(self.forward_stream)

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(self.last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.session_controller.maybe_reap(now)
        self.maybe_tick_pd_flip_state_machine()
        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.ipc_channels.recv_from_rpc is not None:
                        self.ipc_channels.recv_from_rpc.send_pyobj(output)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=self.dp_tp_cpu_group,
            get_forward_ct=lambda: self.forward_ct,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=self.tp_cpu_group,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            scheduler=self,
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if self.server_args.lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                self.server_args.max_loras_per_batch,
                self.server_args.lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(self)

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

    def init_request_receiver(self) -> None:
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.ipc_channels.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_forward_mode=lambda: (
                self.last_batch.forward_mode if self.last_batch is not None else None
            ),
            scripted_scheduler_hook=self.scripted_scheduler_hook,
        )

    def init_dp_attn_adapter(self) -> None:
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.weight_updater.offload_tags,
            ps=self.ps,
            server_args=self.server_args,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
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
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            disaggregation_mode=self.disaggregation_mode,
            page_size=self.page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            server_args=self.server_args,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
            get_pd_flip_held_reqs=self._pd_flip_target_held_reqs,
        )

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=self.server_args.kv_events_config,
            ps=self.ps,
            attn_tp_rank=self.ps.attn_tp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            attn_dp_rank=self.ps.attn_dp_rank,
            dp_rank=self.ps.dp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            get_stats=lambda: self.metrics_reporter.stats,
        )

    def init_load_inquirer(self) -> None:
        self.load_inquirer = SchedulerLoadInquirer(
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_disaggregation_mode=lambda: self.disaggregation_mode,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: self.disagg_prefill_bootstrap_queue,
            get_disagg_prefill_inflight_queue=lambda: self.disagg_prefill_inflight_queue,
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode_prealloc_queue,
            get_disagg_decode_transfer_queue=lambda: self.disagg_decode_transfer_queue,
            get_spec_total_num_accept_tokens=lambda: self.metrics_reporter.spec_total_num_accept_tokens,
            get_spec_total_num_forward_ct=lambda: self.metrics_reporter.spec_total_num_forward_ct,
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = SchedulerOutputStreamer(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=self.tree_cache,
            ps=self.ps,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: self.enable_hicache_storage,
            load_inquirer_get_loads=lambda req: self.load_inquirer.get_loads(req),
        )

    def init_batch_result_processor(self) -> None:
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            server_args=self.server_args,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                server_args=self.server_args, model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            abort_request=self.abort_request,
        )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        # Keep this bound consistent with PrefillAdder's admission budget:
        # ceil_page(input_len) + max_new_tokens + page_size must be strictly
        # smaller than max_total_num_tokens. Otherwise a request can be accepted
        # into the waiting queue but can never be scheduled, blocking the queue
        # and eventually making health checks fail.
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                (
                    req.sampling_params.max_new_tokens
                    if req.sampling_params.max_new_tokens is not None
                    else 1 << 30
                ),
                self.max_req_len - input_len - 1,
                self.max_total_num_tokens - paged_input_len - self.page_size - 1,
            ),
        )

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
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_processor_output(mm_inputs_dict)

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

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )

        if session_id is None:
            # Normal non-session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
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
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer

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
            )
            # TODO: set trace context
            if self.metrics_reporter.enable_metrics:
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

        if self.pd_flip_should_reject_new_work():
            self.reject_pd_flip_admission(req)
            return

        if self.spec_algorithm.is_dflash():
            error_msg = validate_dflash_request(req, self.enable_overlap)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            req.extend_image_inputs(image_inputs)
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
            self.server_args.allow_auto_truncate,
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

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            last_host_node = req.last_host_node
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                last_hash = last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:]

                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if not self._set_or_validate_priority(req):
            return
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
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
        )
        req.tokenizer = self.tokenizer

        if self.pd_flip_should_reject_new_work():
            self.reject_pd_flip_admission(req)
            return

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                # See companion call site above for the array.array wrap rationale.
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )

            req.extend_image_inputs(image_inputs)
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
            self.server_args.allow_auto_truncate,
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
        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
        device = self.device

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # Stash last token into relay; resolve_forward_inputs will gather.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )
        self.future_map.stash(batch.req_pool_indices, last_tokens)
        batch.input_ids = None

        if batch.return_logprob:
            batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # hybrid-SWA early-return) leaves fill_len == len(prefix_indices),
            # so there is nothing new to cache and stashing would be a no-op.
            if self.chunked_req.fill_len > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if self.running_batch.is_empty():
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge_batch(new_batch)
                self.running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            self.running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        ):
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()
            if self.running_batch.is_empty():
                self.running_batch.batch_is_full = False

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not self.server_args.speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            ret, need_sync=need_mlp_sync
        )

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)
            if self.enable_fpm:
                ret.fpm_start_time = self._fpm_batch_t0

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        res = min(res, self.req_to_token_pool.available_size())
        return res

    def _should_delay_dflash_prefill_for_batching(self, running_bs: int) -> bool:
        if not self.spec_algorithm.is_dflash():
            return False
        if running_bs <= 0 or self.chunked_req is not None:
            return False

        return should_delay_dflash_prefill_for_batching(
            running_bs=running_bs,
            num_allocatable_reqs=self.get_num_allocatable_reqs(running_bs),
            max_running_requests=self.max_running_requests,
            prefill_refill_target=self.dflash_prefill_refill_target,
        )

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
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
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if self._should_delay_dflash_prefill_for_batching(running_bs):
            return None

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in self.running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    self.running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if (
                    not added
                    and req.mamba_pool_idx is not None
                    and not getattr(req, "session", None)
                ):
                    self.tree_cache.req_to_token_pool.mamba_allocator.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )

        new_batch.contains_last_prefill_chunk = (
            self.chunked_req is None or len(can_run_list) != 1
        )

        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward.
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            self.running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_input_len
                    if self.chunked_req is not None
                    else 0
                ),
            ),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def _can_schedule_lora_req(
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

        if self.enable_lora_overlap_loading:
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

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Eagerly release lock_ref on completed write-through nodes so they
        # become evictable, improving batch scheduling headroom.
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            mamba_num_gained = (
                mamba_allocator.available_size() - old_mamba_available
                if mamba_allocator is not None
                else None
            )

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
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
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

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
        else:
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, batch: ScheduleBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.
        attr_snapshot = [
            getattr(batch, f.name, None) for f in dataclasses.fields(batch)
        ]
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        # List (not tuple) so that workers can register additional refs via
        # GenerationBatchResult.extra_keep_alive_refs after forward returns.
        self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
        """Make SB transactional across one forward (overlap and non-overlap).

        1. Snapshot SB fields so V2's mid-forward mutations (forward_mode /
           input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec
           only need sampling_info restored - V1 carries spec_info forward as
           next-iter draft input.
        2. Substitute sampling_info with a forward-only copy (orchestrator=None,
           shares the pre-accumulated penalty buffer) so V2's multiple init_new
           calls don't double-accumulate penalties.
        3. (overlap=True only) Pin (batch, snapshot) into batch_record_buf
           for 2 iters so GPU tensors in the snapshot survive the caching
           allocator past the forward stream. Must run AFTER the sampling_info
           swap so the forward-only copy gets pinned. The non-overlap (sync) path
           runs on a single stream and doesn't allocate batch_record_buf, so it
           passes overlap=False.
        """
        # 1. snapshot
        snapshot_v2_full = not batch.spec_algorithm.is_none()
        sched_snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if snapshot_v2_full
            else None
        )
        sched_sampling_info = batch.sampling_info

        # 2. sampling_info substitute
        if sched_sampling_info is not None:
            batch.sampling_info = sched_sampling_info.copy_for_forward()

        # 3. pin for 2-iter tensor lifetime (overlap path only)
        if overlap:
            self.record_batch_in_overlap(batch)

        try:
            yield
        finally:
            if snapshot_v2_full:
                for name, value in sched_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sched_sampling_info

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            if self.enable_overlap:
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                self.future_map.resolve_seq_lens_cpu(batch)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        fwd_kwargs = (
                            {
                                "on_publish": partial(
                                    self.future_map.publish, future_indices
                                )
                            }
                            if not batch.spec_algorithm.is_none()
                            else {}
                        )

                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        batch_result.copy_done = self.device_module.Event()
                        if batch_result.delay_sample_func is None:
                            stash_payload = (
                                batch_result.next_draft_input
                                if not batch.spec_algorithm.is_none()
                                else batch_result.next_token_ids
                            )
                            self.future_map.stash(future_indices, stash_payload)
                            batch_result.copy_to_cpu(
                                return_logprob=batch.return_logprob,
                                return_hidden_states=batch.return_hidden_states,
                            )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                batch.input_ids = None

                if not batch.spec_algorithm.is_none():
                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                if isinstance(batch_result.next_token_ids, torch.Tensor):
                    self.future_map.stash(
                        batch.req_pool_indices, batch_result.next_token_ids
                    )
                batch.input_ids = None
            elif not batch.spec_algorithm.is_none():
                # Non-overlap: drive the V2 worker synchronously (no
                # future_map relay / on_publish).
                resolve_forward_inputs(batch, self.future_map)
                with self._forward_isolation(batch, overlap=False):
                    batch_result = self.model_worker.forward_batch_generation(batch)
                # The isolation restore reverted the worker's in-forward SB edits;
                # re-apply what must carry to the next iter.
                batch.spec_info = batch_result.next_draft_input
                if batch_result.new_seq_lens is not None:
                    batch.seq_lens = batch_result.new_seq_lens
                    if batch.seq_lens_cpu is not None:
                        batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                batch.input_ids = None  # rebuilt next iter from draft_token
                self.update_cache_from_scheduler(batch, batch_result)
                # Sync D2H so the result processor can read CPU tensors.
                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
                if isinstance(batch_result.next_token_ids, torch.Tensor):
                    # Non-spec: relay via future_map, gathered next iter.
                    self.future_map.stash(
                        batch.req_pool_indices, batch_result.next_token_ids
                    )
                    batch.input_ids = None
                self.update_cache_from_scheduler(batch, batch_result)

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            if self.enable_overlap:
                self.record_batch_in_overlap(batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    resolve_forward_inputs(batch, self.future_map)
                    pooler_output = self.tp_worker.forward_batch_embedding(batch)
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                    )
                    ret.copy_to_cpu()
            else:
                resolve_forward_inputs(batch, self.future_map)
                pooler_output = self.tp_worker.forward_batch_embedding(batch)
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                )

        self._maybe_report_active_ranks()

        return ret

    def _maybe_report_active_ranks(self) -> None:
        if not (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            return
        # Get the tensors indicating rank activeness
        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        dp_active_ranks = tp_active_ranks.reshape(self.ps.dp_size, -1).prod(axis=1)
        self.ipc_channels.send_to_tokenizer.send_output(
            ActiveRanksOutput(status=dp_active_ranks.tolist())
        )

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            # Delay-sample is non-spec only; stash takes next_token_ids tensor.
            self.future_map.stash(
                batch_result.future_indices, batch_result.next_token_ids
            )
            batch_result.copy_to_cpu(
                return_logprob=self.cur_batch.return_logprob,
                return_hidden_states=self.cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        self.publish_load_snapshot(force=batch.forward_mode.is_extend())

        if batch.forward_mode.is_decode():
            self.batch_result_processor.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.batch_result_processor.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)

        self.metrics_reporter.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()
        self.metrics_reporter.update_device_timer()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
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
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        if not self.is_fully_idle():
            return

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp).
        if not self.enable_hisparse:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # reset device timer window so idle time isn't counted
        self.metrics_reporter.reset_device_timer_window()

        # Publish the idle state so /get_loads and DP balancing do not see stale load.
        self.publish_load_snapshot(force=True)

        # sleep until next event
        self.maybe_sleep_on_idle()

    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if not for_health_check:
            idle &= not self._pd_flip_source_migration_blocks_idle()

            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def _pp_microbatches_drained(self) -> bool:
        if self.ps.pp_size == 1:
            return True
        return all(x.is_empty() for x in self.running_mbs) and all(
            mb is None or mb.is_empty() for mb in self.mbs
        )

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
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
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
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

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.is_fully_idle():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.metrics_reporter.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_reporter.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(vars(get_global_server_args()))  # vars returns a ref to obj.__dict__
        ret["last_gen_throughput"] = self.metrics_reporter.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests
        ret["pd_flip"] = self.get_pd_flip_internal_state()

        if (
            not self.spec_algorithm.is_none()
            and self.metrics_reporter.spec_total_num_forward_ct > 0
        ):
            ret["avg_spec_accept_length"] = (
                self.metrics_reporter.spec_total_num_accept_tokens
                / self.metrics_reporter.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict

        # This field is not serializable.
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
                "pd_flip_prefill_slo_attainment",
                "pd_flip_decode_slo_attainment",
                "pd_flip_prefill_nodes",
                "pd_flip_decode_nodes",
                "pd_flip_slo_threshold",
                "pd_flip_window_seconds",
                "pd_flip_prepare_ack",
                "pd_flip_commit_ack",
                "pd_flip_abort",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.ps.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.ps.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if (
                not self.spec_algorithm.is_none()
                and self.metrics_reporter.spec_total_num_forward_ct > 0
            ):
                avg_spec_accept_length = (
                    self.metrics_reporter.spec_total_num_accept_tokens
                    / self.metrics_reporter.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.metrics_reporter.spec_total_num_accept_tokens = (
                self.metrics_reporter.spec_total_num_forward_ct
            ) = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

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

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                bootstrap_pending = req.pending_bootstrap
                maybe_release_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )
                if (
                    bootstrap_pending
                    and hasattr(req, "disagg_kv_sender")
                    and req.disagg_kv_sender is not None
                ):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.ipc_channels.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            if (
                not self.last_batch.is_empty()
                and self.disaggregation_mode != DisaggregationMode.PREFILL
            ):
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract" and not self.running_batch.is_empty():
            self.running_batch.filter_batch()
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_reporter.last_gen_throughput = 0.0
        if self.metrics_reporter.current_scheduler_metrics_enabled:
            self.metrics_reporter.metrics_collector.last_log_time = 0.0
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
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
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
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

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
        if self.ps.pp_rank == 0 and self.ps.tp_rank == 0 and self.ps.attn_cp_rank == 0:
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
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


def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)

    return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until shutdown)
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
            scheduler.metrics_reporter._shutdown_fpm()
