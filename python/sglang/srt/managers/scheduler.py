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
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import torch.distributed
import zmq
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext
from torch.distributed import barrier

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
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
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    SchedulerMetricsMixin,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_event_batch,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_batch,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_zmq_socket,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
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
        dp_rank: Optional[int],
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
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
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and tp_rank == 0
        )
        self.enable_trace = server_args.enable_trace
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None

        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init inter-process communication
        self.init_sockets(server_args, port_args)

        # Init pdmux context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        from sglang.srt.managers.tp_worker import TpModelWorker

        self.tp_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            server_args=server_args,
            nccl_port=port_args.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=dp_rank,
        )

        if server_args.speculative_draft_load_format is not None:
            server_args.load_format = server_args.speculative_draft_load_format
            logger.info(
                f"Using draft model load_format: '{server_args.speculative_draft_load_format}'"
            )

        # Draft workers are looked up via `SpeculativeAlgorithm` registry; new
        # algorithms should register their factory instead of patching this code.
        if self.spec_algorithm.name in {"EAGLE", "EAGLE3"}:
            draft_worker_kwargs["enable_overlap"] = self.enable_overlap
        self.draft_worker = self.spec_algorithm.create_draft_worker(
            **draft_worker_kwargs
        )

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
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # With DP attention enabled, the entry rank is attn_tp_rank==0;
        # otherwise the entry rank is TP group local rank 0.
        # For #11910, use the CPU communication group to broadcast VLM Python objects,
        # avoiding any coupling with CUDA streams/devices.
        if self.server_args.enable_dp_attention:
            self.cpu_group = self.attn_tp_cpu_group
            self.entry_rank = self.attn_tp_group.first_rank
            self.is_entry_rank = self.attn_tp_rank == 0
        else:
            self.cpu_group = self.tp_cpu_group
            self.entry_rank = self.tp_group.first_rank
            self.is_entry_rank = self.tp_group.rank_in_group == 0

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid = self.tp_worker.is_hybrid
        self.is_hybrid_gdn = self.tp_worker.model_runner.hybrid_gdn_config is not None

        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.sessions: Dict[str, Session] = {}
        self.default_stream: CudaStream = torch.get_device_module(
            self.device
        ).current_stream()
        if self.device == "cpu":
            self.default_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        # Enable preemption for priority scheduling.
        self.try_preemption = self.enable_priority_scheduling
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profiler()
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init disaggregation
        self.init_disaggregation()

        if self.enable_kv_cache_events:
            self.init_kv_events(server_args.kv_events_config)

        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Init overlap
        self.init_overlap()

        # Init mlp sync flag
        self.require_mlp_sync = require_mlp_sync(server_args)

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
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
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadReqInput, self.get_load),
            ]
        )

    def init_sockets(self, server_args: ServerArgs, port_args: PortArgs):
        context = zmq.Context(2)
        self.idle_sleeper = None

        class SenderWrapper:
            def __init__(self, socket: zmq.Socket):
                self.socket = socket

            def send_output(
                self,
                output: Union[BaseReq, BaseBatchReq],
                recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
            ):
                if self.socket is None:
                    return

                if (
                    isinstance(recv_obj, BaseReq)
                    and recv_obj.http_worker_ipc is not None
                    and output.http_worker_ipc is None
                ):
                    # handle communicator reqs for multi-http worker case
                    output.http_worker_ipc = recv_obj.http_worker_ipc

                self.socket.send_pyobj(output)

        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled():
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

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
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_memory_pool_and_cache(self):
        server_args = self.server_args

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            if self.is_hybrid:
                ChunkCacheClass = SWAChunkCache
            else:
                ChunkCacheClass = ChunkCache
            self.tree_cache = ChunkCacheClass(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        else:
            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(
                    disable=False,
                    use_hicache=self.enable_hierarchical_cache,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tp_cache_group=self.tp_cpu_group,
                    page_size=self.page_size,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    enable_metrics=self.enable_metrics,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                )
            elif self.enable_hierarchical_cache:
                self.tree_cache = HiRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tp_cache_group=(
                        self.attn_tp_cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_cpu_group
                    ),
                    page_size=self.page_size,
                    eviction_policy=server_args.radix_eviction_policy,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    hicache_io_backend=server_args.hicache_io_backend,
                    hicache_mem_layout=server_args.hicache_mem_layout,
                    enable_metrics=self.enable_metrics,
                    hicache_storage_backend=server_args.hicache_storage_backend,
                    hicache_storage_prefetch_policy=server_args.hicache_storage_prefetch_policy,
                    model_name=server_args.served_model_name,
                    storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
                    is_eagle=self.spec_algorithm.is_eagle(),
                )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid:
                self.tree_cache = SWARadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    sliding_window_size=self.sliding_window_size,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    is_eagle=self.spec_algorithm.is_eagle(),
                    enable_metrics=self.enable_metrics,
                )
            elif self.is_hybrid_gdn:
                self.tree_cache = MambaRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    enable_metrics=self.enable_metrics,
                )
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    enable_metrics=self.enable_metrics,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                    eviction_policy=server_args.radix_eviction_policy,
                )
            else:
                self.tree_cache = RadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    enable_metrics=self.enable_metrics,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                    eviction_policy=server_args.radix_eviction_policy,
                    is_eagle=self.spec_algorithm.is_eagle(),
                )

        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_decode_enable_offload_kvcache
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

        self.decode_mem_cache_buf_multiplier = (
            1
            if self.spec_algorithm.is_none()
            else (
                server_args.speculative_num_draft_tokens
                + (
                    (server_args.speculative_eagle_topk or 1)
                    * (server_args.speculative_num_steps or 1)
                )
            )
        )

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "100"))
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            draft_token_to_kv_pool = None
        elif self.spec_algorithm.is_eagle() and self.enable_overlap:
            draft_token_to_kv_pool = (
                self.draft_worker.draft_worker.draft_runner.token_to_kv_pool
            )
        else:
            draft_token_to_kv_pool = self.draft_worker.model_runner.token_to_kv_pool

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                hidden_states_dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
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
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                prefill_pp_size=self.server_args.disaggregation_prefill_pp,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                hidden_states_dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                decode_tp_size=self.server_args.disaggregation_decode_tp,
                decode_dp_size=self.server_args.disaggregation_decode_dp,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

    def init_overlap(self):
        self.future_map = None

        if not self.enable_overlap:
            return

        self.forward_stream: CudaStream = torch.get_device_module(self.device).Stream()
        self.forward_stream_ctx: CudaStreamContext = torch.get_device_module(
            self.device
        ).stream(self.forward_stream)
        self.copy_stream: CudaStream = torch.get_device_module(self.device).Stream()
        self.copy_stream_ctx: CudaStreamContext = torch.get_device_module(
            self.device
        ).stream(self.copy_stream)

        self.future_map = FutureMap(
            self.max_running_requests, self.device, self.spec_algorithm
        )
        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def init_moe_config(self):
        if hasattr(self.model_config.hf_config, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[Tuple[ScheduleBatch, GenerationBatchResult]] = deque()
        disable_consecutive_prefill_overlap = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
        )

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            disable_overlap_for_batch = (
                disable_consecutive_prefill_overlap
                and batch
                and batch.forward_mode.is_extend()
                and self.last_batch
                and self.last_batch.forward_mode.is_extend()
            )

            if disable_overlap_for_batch:
                pop_and_process()

            batch_result = None
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))

            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.launch_batch_sample_if_needed(batch_result)
            self.last_batch = batch

            if envs.SGLANG_ENABLE_RUNTIME_MEM_LEAK_CHECK.get():
                self._check_runtime_mem_leak()

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.device_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        if self.enable_trace:
            for req in recv_reqs:
                if isinstance(
                    req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    trace_set_proc_propagate_context(req.rid, req.trace_context)
                    trace_slice_start("", req.rid, anonymous=True)

        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            if is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None
                or not self.running_batch.is_empty()
                or len(self.offload_tags) > 0
            ):
                self.return_health_check_ct += 1
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if isinstance(output, RpcReqOutput):
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_output(output, recv_req)

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_dict(raw_mm_inputs) once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_dict

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
                and self.cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.cpu_group
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
        if self.is_entry_rank:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list, src=self.entry_rank, group=self.cpu_group
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list, src=self.entry_rank, group=self.cpu_group
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict: dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_dict(mm_inputs_dict)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

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
                custom_logit_processor=recv_req.custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                data_parallel_rank=recv_req.data_parallel_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"boostrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.set_finish_with_abort(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

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

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            # For prefill-only requests with logprob_start_len == -1, set logprob_start_len beyond input sequence
            # to skip input logprob computation entirely
            if req.is_prefill_only:
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # TODO: For text generation, evaluate setting logprob_start_len to len(req.origin_input_ids) as well
                req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if not req.is_prefill_only and req.logprob_start_len >= len(
            req.origin_input_ids
        ):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                        error_msg = f"Invalid grammar request with cache hit: {key=}"
                        req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
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
            req.init_next_round_input(self.tree_cache)
            if req.last_node.backuped:
                # only to initiate the prefetch if the last node is backuped
                # otherwise, the allocated GPU memory must be locked for integrity
                last_hash = req.last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]

                prefix_keys = (
                    req.last_node.get_prefix_hash_values(req.last_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    req.last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.wait_queue_entry_time = time.perf_counter()
            trace_slice_end(RequestStage.REQUEST_PROCESS, req.rid, auto_next_anon=True)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.prefill_bootstrap_queue_entry_time = time.perf_counter()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.decode_prealloc_queue_entry_time = time.perf_counter()
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
            self.send_to_tokenizer.send_output(abort_req, req)
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
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_output(
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
        return req_to_abort.rid == recv_req.rid

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            http_worker_ipc=recv_req.http_worker_ipc,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

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
        req.logprob_start_len = len(req.origin_input_ids) - 1
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

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self):
        is_radix_tree = isinstance(self.tree_cache, MambaRadixCache)
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_radix_tree else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_radix_tree else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()
        if self.chunked_req:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            # chunked request keeps its rid but will get a new req_pool_idx
            if self.tp_worker.model_runner.mambaish_config is not None:
                self.req_to_token_pool.free(
                    self.chunked_req.req_pool_idx, free_mamba_cache=False
                )
            else:
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention
        if need_mlp_sync:
            ret = self.prepare_mlp_sync_batch(ret)

        if ret:
            trace_event_batch("schedule", ret.reqs)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        if self.try_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked request has just been released.
        # In PP case, a chunked req can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked request to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and not self.chunked_req
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            lora_set = set([req.lora_id for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:

            if self.enable_lora and not self.tp_worker.can_run_lora_batch(
                lora_set
                | set([req.lora_id for req in adder.can_run_list])
                | set([req.lora_id])
            ):
                self.running_batch.batch_is_full = True
                break

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if not self.try_preemption:
                    break
                if not adder.preempt_to_schedule(req, self.server_args):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.add_latency(RequestStage.PREFILL_WAITING)

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.current_scheduler_metrics_enabled():
            self.log_prefill_stats(adder, can_run_list, running_bs, 0)

        for req in can_run_list:
            if req.time_stats.forward_entry_time == 0:
                # Avoid update chunked request many times
                req.time_stats.forward_entry_time = time.perf_counter()
                if self.enable_metrics:
                    self.metrics_collector.observe_queue_time(
                        req.time_stats.get_queueing_time(),
                    )

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
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
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

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            self.num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
                    AbortReq(abort_message=abort_reason.message, rid=req.rid), req
                )

            logger.info(
                "KV cache pool is full. Retract requests. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
            )

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Capture prefill start time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            current_time = time.perf_counter()
            for req in batch.reqs:
                req.time_stats.prefill_start_time_host = current_time

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # Run forward
        if self.is_generation:
            batch_or_worker_batch = batch

            if self.enable_overlap or self.spec_algorithm.is_none():
                # FIXME(lsyin): remove this if and finally unify the abstraction
                batch_or_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                # FIXME: remove this assert
                assert isinstance(batch_or_worker_batch, ModelWorkerBatch)
                model_worker_batch = batch_or_worker_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.default_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    batch_result = self.model_worker.forward_batch_generation(
                        model_worker_batch
                    )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = torch.get_device_module(
                        self.device
                    ).Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu()
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_v2_eagle:
                    # FIXME(lsyin): tmp code for eagle v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    #     # FIXME(lsyin): remove the allocate_lens in EagleDraftInput
                    #     allocate_lens=batch_result.next_draft_input.allocate_lens,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                batch_result = self.model_worker.forward_batch_generation(
                    batch_or_worker_batch
                )
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)

            # NOTE: future_indices_or_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = future_indices_or_next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob or self.spec_algorithm.is_eagle():
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            else:
                extend_input_len_per_req = None

            if batch.return_logprob:
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_logprob_start_len_per_req = None

            batch_result.extend_input_len_per_req = extend_input_len_per_req
            batch_result.extend_logprob_start_len_per_req = (
                extend_logprob_start_len_per_req
            )
            ret = batch_result
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(embeddings=embeddings)

        # Capture prefill end time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            current_time = time.perf_counter()
            for req in batch.reqs:
                req.time_stats.prefill_end_time_host = current_time

        return ret

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.default_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu()

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
            trace_slice_batch(RequestStage.DECODE_LOOP, batch.reqs)
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                if result.copy_done is not None:
                    result.copy_done.synchronize()

        self.maybe_send_health_check_signal()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_output(HealthCheckOutput())

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""

        num_ready_reqs = 0
        num_timeout_reqs = 0
        for req in self.grammar_queue:
            try:
                if req.finished():  # It is aborted by AbortReq
                    num_ready_reqs += 1
                    continue

                req.grammar = req.grammar.result(timeout=0.03)
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    error_msg = f"Invalid grammar request: {req.grammar_key=}"
                    req.set_finish_with_abort(error_msg)

                num_ready_reqs += 1
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # NOTE(lianmin): this timeout is the waiting time of the above line. It is
                # not the waiting time from it enters the grammar queue.
                if req.grammar_wait_ct > GRAMMAR_TIMEOUT / 0.03:
                    num_timeout_reqs = 1
                break

        if self.server_args.enable_dp_attention:
            tp_size = self.attn_tp_size
            tp_group = self.attn_tp_cpu_group
        else:
            tp_size = self.tp_size
            tp_group = self.tp_cpu_group

        if tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor([num_ready_reqs, num_timeout_reqs], dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
            num_ready_reqs_max, num_timeout_reqs_max = tensor.tolist()

            for i in range(num_ready_reqs, num_ready_reqs_max):
                req = self.grammar_queue[i]
                if req.finished():  # It is aborted by AbortReq
                    continue
                req.grammar = req.grammar.result()
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    error_msg = f"Invalid grammar request: {req.grammar_key=}"
                    req.set_finish_with_abort(error_msg)
        else:
            num_ready_reqs_max = num_ready_reqs
            num_timeout_reqs_max = num_timeout_reqs

        for i in range(num_ready_reqs, num_ready_reqs + num_timeout_reqs_max):
            req = self.grammar_queue[i]
            req.grammar.cancel()
            self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
            error_msg = f"Grammar preprocessing timed out for {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)

        num_ready_reqs = num_ready_reqs_max + num_timeout_reqs_max

        for req in self.grammar_queue[:num_ready_reqs]:
            self._add_request_to_queue(req)
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def _is_no_request(self):
        no_request = (
            len(self.waiting_queue) == 0
            and self.running_batch.is_empty()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            no_request &= (
                len(self.disagg_prefill_bootstrap_queue.queue) == 0
                and len(self.disagg_prefill_inflight_queue) == 0
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            no_request &= (
                len(self.disagg_decode_prealloc_queue.queue) == 0
                and len(self.disagg_decode_transfer_queue.queue) == 0
            )
        return no_request

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if self._is_no_request():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_accepted_tokens = 0
            self.spec_num_forward_ct = 0
            self.spec_total_num_accepted_tokens = 0
            self.spec_total_num_forward_ct = 0
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def get_load(self, recv_req: GetLoadReqInput = None) -> GetLoadReqOutput:
        # TODO(lsyin): use dynamically maintained num_waiting_tokens

        if self.is_hybrid:
            num_tokens_full = (
                self.full_tokens_per_layer
                - self.token_to_kv_pool_allocator.full_available_size()
                - self.tree_cache.full_evictable_size()
            )
            num_tokens_swa = (
                self.swa_tokens_per_layer
                - self.token_to_kv_pool_allocator.swa_available_size()
                - self.tree_cache.swa_evictable_size()
            )
            num_tokens = max(num_tokens_full, num_tokens_swa)
        elif self.is_hybrid_gdn:
            num_tokens = (
                self.max_total_num_tokens
                - self.token_to_kv_pool_allocator.available_size()
                - self.tree_cache.full_evictable_size()
            )
        else:
            num_tokens = (
                self.max_total_num_tokens
                - self.token_to_kv_pool_allocator.available_size()
                - self.tree_cache.evictable_size()
            )

        # Tokens in waiting queue, bootstrap queue, prealloc queue
        num_tokens += sum(len(req.origin_input_ids) for req in self.waiting_queue)
        num_waiting_reqs = len(self.waiting_queue)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            num_tokens += sum(
                len(req.origin_input_ids)
                for req in self.disagg_prefill_bootstrap_queue.queue
            )
            num_waiting_reqs += len(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            num_tokens += sum(
                len(req.req.origin_input_ids)
                for req in self.disagg_decode_prealloc_queue.queue
            )
            num_waiting_reqs += len(self.disagg_decode_prealloc_queue.queue)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=len(self.running_batch.reqs) + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_tokens=num_tokens,
        )

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = vars(get_global_server_args())
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
        }

        ret["memory_usage"]["graph"] = round(
            self.tp_worker.model_runner.graph_mem_usage, 2
        )

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )
        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

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
            ]
        )
        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break
        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

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
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)

            # For mamba radix cache
            if req.mamba_pool_idx is not None:
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            # Abort method 2: call `set_finish_with_abort`
            # The request will still run one prefill forward pass.
            # In this case, we change the input_ids to be only one token to make this prefill cheap.
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

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

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
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
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
            return OpenSessionReqOutput(session_id, True)

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]

    def get_print_prefix(self):
        prefix = ""
        if self.attn_dp_rank is not None:
            prefix += f" DP{self.attn_dp_rank}"
        if self.server_args.tp_size > 1:
            prefix += f" TP{self.tp_rank}"
        if self.pp_size > 1:
            prefix += f" PP{self.pp_rank}"
        return prefix

    def current_scheduler_metrics_enabled(self):
        return self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_output(recv_req, recv_req)
        return None


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = time.time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and time.time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = time.time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith("HEALTH_CHECK")


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the logger prefix
    prefix = ""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if (numa_node := server_args.numa_node) is not None:
        numa_bind_to_node(numa_node[gpu_id])

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
