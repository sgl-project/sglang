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
import warnings
from collections import defaultdict, deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import zmq
from torch.distributed import barrier

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
)
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HealthCheckOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    RpcReqInput,
    RpcReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_logger,
    crash_on_warnings,
    get_bool_env_var,
    get_zmq_socket,
    is_hpu,
    kill_itself_when_parent_died,
    pyspy_dump_schedulers,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

_is_hpu = is_hpu()

expert_distribution_recorder = ExpertDistributionRecorder()

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput
    next_token_ids: List[int]
    extend_input_len_per_req: List[int]
    extend_logprob_start_len_per_req: List[int]
    bid: int


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    bid: int


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.page_size = server_args.page_size

        # Distributed rank info
        self.dp_size = server_args.dp_size
        self.attn_tp_rank, self.attn_tp_size, self.dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init inter-process communication
        context = zmq.Context(2)
        if self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init tokenizer
        self.init_tokenizer()

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")
        if self.model_config.is_multimodal:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for multimodal models.")

        # Launch a tensor parallel worker
        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if self.device == "hpu":
            # scheduler will run on cpu if hpu is used
            self.device = "cpu"
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"chunked_prefill_size={server_args.chunked_prefill_size}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.num_prefill_tokens = 0
        self.last_decode_stats_tic = time.time()
        self.last_prefill_stats_tic = time.time()
        self.return_health_check_ct = 0
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU

        # Init session info
        self.sessions: Dict[str, Session] = {}

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
                server_args, self.tokenizer, self.model_config.vocab_size
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
        )
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        # Init profiler
        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[str] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profiler_target_forward_ct: Optional[int] = None

        # Init metrics stats
        self.init_metrics()

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (FlushCacheReq, self.flush_cache_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (ProfileReq, self.profile),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
            ]
        )

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()

    def init_tokenizer(self):
        server_args = self.server_args

        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
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
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )
        else:
            if self.enable_hierarchical_cache:
                self.tree_cache = HiRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tp_cache_group=self.tp_worker.get_tp_cpu_group(),
                    page_size=self.page_size,
                    hicache_ratio=server_args.hicache_ratio,
                )
            else:
                self.tree_cache = RadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                )

        self.decode_mem_cache_buf_multiplier = (
            1
            if self.spec_algorithm.is_none()
            else (
                server_args.speculative_num_draft_tokens
                + (
                    server_args.speculative_eagle_topk
                    * server_args.speculative_num_steps
                )
            )
        )

    def init_metrics(self):
        # The largest prefill length of a single request
        self._largest_prefill_len: int = 0
        # The largest context length (prefill + generation) of a single request
        self._largest_prefill_decode_len: int = 0
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.stats = SchedulerStats()
        if self.enable_metrics:
            engine_type = "unified"
            self.metrics_collector = SchedulerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    "engine_type": engine_type,
                },
            )

    def init_disaggregation(self):
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            aux_dtype = torch.int32
            # A list of metadata buffers. The shape is (b, metadata_size) where
            # b corresponds to a max running requests. The last shape * dtype.itemsize
            # should be larger than 64 bytes to work with RDMA, so we pad it.
            output_id_buffer = torch.zeros(
                (buffer_size, 16), dtype=aux_dtype, device="cpu"
            )
            metadata_buffers = [output_id_buffer]

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.tp_worker.get_attention_tp_cpu_group(),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=metadata_buffers,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=metadata_buffers,
                aux_dtype=aux_dtype,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.tp_worker.get_attention_tp_cpu_group(),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
            )
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            aux_dtype = torch.int32
            # A list of metadata buffers. The shape is (b, metadata_size) where
            # b corresponds to a max running requests. The last shape * dtype.itemsize
            # should be larger than 64 bytes to work with RDMA, so we pad it.
            output_id_buffer = torch.zeros(
                (buffer_size, 16), dtype=aux_dtype, device="cpu"
            )
            metadata_buffers = [output_id_buffer]

            self.disagg_prefill_pending_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=metadata_buffers,
                aux_dtype=aux_dtype,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.tp_worker.get_attention_tp_cpu_group(),
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_infight_queue: List[Req] = []

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
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None)

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self):
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_pending_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            if len(self.disagg_prefill_infight_queue) > 0:
                self.process_disagg_prefill_infight_queue()

            if batch is None and len(self.disagg_prefill_infight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(
                        batch.reqs, [False for _ in range(len(batch.reqs))]
                    )
                else:
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)

            if batch is None and (
                len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
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

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                attn_tp_rank_0 = self.dp_rank * self.attn_tp_size
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_rank,
                    self.attn_tp_cpu_group,
                    src=attn_tp_rank_0,
                )
            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs, self.tp_rank, self.tp_cpu_group
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            if is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None or not self.running_batch.is_empty()
            ):
                self.return_health_check_ct += 1
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if isinstance(output, RpcReqOutput):
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)

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

            # Handle custom logit processor passed to the request
            custom_logit_processor = recv_req.custom_logit_processor
            if (
                not self.server_args.enable_custom_logit_processor
                and custom_logit_processor is not None
            ):
                logger.warning(
                    "The SGLang server is not configured to enable custom logit processor."
                    "The custom logit processor passed in will be ignored."
                    "Please set --enable-custom-logits-processor to enable this feature."
                )
                custom_logit_processor = None

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_path=recv_req.lora_path,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_room=recv_req.bootstrap_room,
            )
            req.tokenizer = self.tokenizer

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.finished_reason = FINISH_ABORT(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self._add_request_to_queue(req)
                return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                error_msg = (
                    "Multimodal prompt is too long after expanding multimodal tokens. "
                    f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                )
                logger.error(error_msg)
                req.origin_input_ids = [0]
                req.multimodal_inputs = None
                req.sampling_params.max_new_tokens = 0
                req.finished_reason = FINISH_ABORT(
                    error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
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
            req.origin_input_ids = [0]
            req.sampling_params.max_new_tokens = 0
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len >= len(req.origin_input_ids):
            req.finished_reason = FINISH_ABORT(
                f"logprob_start_len, ({req.logprob_start_len}) is higher than the number of input tokens ({len(req.origin_input_ids)}). Request with a lower logprob_start_len.",
                HTTPStatus.BAD_REQUEST,
                "BadRequestError",
            )
            req.logprob_start_len = len(req.origin_input_ids) - 1
            self._add_request_to_queue(req)
            return

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            req.grammar = self.grammar_backend.get_cached_value(key)
            if not req.grammar:
                req.grammar = self.grammar_backend.get_future_value(key)
                add_to_grammar_queue = True

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def _add_request_to_queue(self, req: Req):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_pending_queue.add(req)

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req)

        else:
            req.queue_time_start = time.time()
            self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: List[Req], is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.extend(reqs)
        else:
            self.waiting_queue.extend(reqs)

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                error_msg = (
                    "Multimodal prompt is too long after expanding multimodal tokens. "
                    f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                )
                logger.error(error_msg)
                req.origin_input_ids = [0]
                req.multimodal_inputs = None
                req.sampling_params.max_new_tokens = 0
                req.finished_reason = FINISH_ABORT(
                    error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
                )
                req.queue_time_start = time.time()
                self.waiting_queue.append(req)
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

    def log_prefill_stats(
        self,
        adder: PrefillAdder,
        can_run_list: List[Req],
        running_bs: int,
    ):
        gap_latency = time.time() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.time()
        self.last_input_throughput = self.num_prefill_tokens / gap_latency
        self.num_prefill_tokens = 0

        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )
        self._largest_prefill_len = max(
            self._largest_prefill_len, adder.log_input_tokens
        )

        num_new_seq = len(can_run_list)
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"#running-req: {running_bs}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )
        logger.info(f)

        if self.enable_metrics:
            cache_hit_rate = adder.log_hit_tokens / (
                adder.log_input_tokens + adder.log_hit_tokens
            )
            self.stats.num_running_reqs = running_bs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(num_used / self.max_total_num_tokens, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            total_queue_latency = 0
            for req in can_run_list:
                total_queue_latency += req.queue_time_end - req.queue_time_start
            self.stats.avg_request_queue_latency = total_queue_latency / num_new_seq

            self.metrics_collector.log_stats(self.stats)

    def log_decode_stats(self):
        gap_latency = time.time() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.time()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = len(self.running_batch.reqs)
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        if self.spec_algorithm.is_none():
            msg = (
                f"Decode batch. "
                f"#running-req: {num_running_reqs}, "
                f"#token: {num_used}, "
                f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
                f"#queue-req: {len(self.waiting_queue)}, "
            )
            spec_accept_length = 0
        else:
            spec_accept_length = (
                self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct
            )
            self.cum_spec_accept_length += self.spec_num_total_accepted_tokens
            self.cum_spec_accept_count += self.spec_num_total_forward_ct
            self.spec_num_total_accepted_tokens = self.spec_num_total_forward_ct = 0
            msg = (
                f"Decode batch. "
                f"#running-req: {num_running_reqs}, "
                f"#token: {num_used}, "
                f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                f"accept len: {spec_accept_length:.2f}, "
                f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
                f"#queue-req: {len(self.waiting_queue)}, "
            )

        logger.info(msg)
        if self.enable_metrics:
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = num_used / self.max_total_num_tokens
            self.stats.cache_hit_rate = 0.0
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.spec_accept_length = spec_accept_length
            self.metrics_collector.log_stats(self.stats)

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )
        protected_size = self.tree_cache.protected_size()
        memory_leak = available_size != (
            self.max_total_num_tokens
            if not self.enable_hierarchical_cache
            else self.max_total_num_tokens - protected_size
        )
        if memory_leak:
            msg = (
                "token_to_kv_pool_allocator memory leak detected! "
                f"{available_size=}, {protected_size=}, {self.max_total_num_tokens=}\n"
                f"{self.token_to_kv_pool_allocator.available_size()=}\n"
                f"{self.tree_cache.evictable_size()=}\n"
            )
            warnings.warn(msg)
            if crash_on_warnings():
                raise ValueError(msg)

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            warnings.warn(msg)
            if crash_on_warnings():
                raise ValueError(msg)

        if (
            self.enable_metrics
            and self.attn_tp_rank == 0
            and time.time() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            num_used = self.max_total_num_tokens - (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = num_used / self.max_total_num_tokens
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.metrics_collector.log_stats(self.stats)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                # chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch()
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        new_batch = self.get_new_batch_prefill()
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
        if self.server_args.enable_dp_attention or self.server_args.enable_sp_layernorm:
            ret, _ = self.prepare_dp_attn_batch(ret)

        return ret

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if running_bs >= self.max_running_requests:
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            # check for completion of hierarchical cache activities to release memory
            self.tree_cache.writing_check()
            self.tree_cache.loading_check()

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.lora_paths:
            lora_set = set([req.lora_path for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if (
                self.lora_paths
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                self.running_batch.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.running_batch.batch_is_full = True
                break

            req.init_next_round_input(
                None if prefix_computed else self.tree_cache,
                self.enable_hierarchical_cache,
            )

            res = adder.add_one_req(
                req, self.chunked_req, self.enable_hierarchical_cache
            )
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (
                            self.running_batch is not None
                            and not self.running_batch.is_empty()
                        )
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
                req.queue_time_end = time.time()

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if self.enable_hierarchical_cache:
            self.tree_cache.ready_to_load_cache()

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
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
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self._extend_requests_to_queue(retracted_reqs)
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

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Check profiler
        if (
            self.profiler_target_forward_ct
            and self.profiler_target_forward_ct <= self.forward_ct
        ):
            self.stop_profile()

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none():
                model_worker_batch = batch.get_model_worker_batch()
                logits_output, next_token_ids = self.tp_worker.forward_batch_generation(
                    model_worker_batch
                )
                bid = model_worker_batch.bid
            else:
                (
                    logits_output,
                    next_token_ids,
                    bid,
                    num_accepted_tokens,
                ) = self.draft_worker.forward_batch_speculative_generation(batch)
                self.spec_num_total_accepted_tokens += (
                    num_accepted_tokens + batch.batch_size()
                )
                self.spec_num_total_forward_ct += batch.batch_size()
                self.num_generated_tokens += num_accepted_tokens
            batch.output_ids = next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_input_len_per_req = None
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=bid,
            )
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(
                embeddings=embeddings, bid=model_worker_batch.bid
            )
        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_batch_result(result.bid)
                if batch.next_batch_sampling_info:
                    batch.next_batch_sampling_info.update_regex_vocab_mask()
                    self.current_stream.synchronize()
                    batch.next_batch_sampling_info.sampling_info_done.set()
        elif batch.forward_mode.is_dummy_first():
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_pyobj(HealthCheckOutput())

    def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
        return self.prepare_dp_attn_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            tp_cpu_group=self.tp_cpu_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            spec_algorithm=self.spec_algorithm,
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
        )

    @staticmethod
    def prepare_dp_attn_batch_raw(
        local_batch: ScheduleBatch,
        dp_size,
        attn_tp_size: int,
        tp_cpu_group,
        get_idle_batch,
        disable_cuda_graph: bool,
        spec_algorithm,
        speculative_num_draft_tokens,
    ):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
            global_num_tokens_for_logprob = 0
        elif local_batch.forward_mode.is_decode():
            num_tokens = local_batch.batch_size()
            if not spec_algorithm.is_none() and spec_algorithm.is_eagle():
                num_tokens = num_tokens * speculative_num_draft_tokens
            global_num_tokens_for_logprob = num_tokens
        else:
            num_tokens = local_batch.extend_num_tokens
            global_num_tokens_for_logprob = sum(
                [
                    # We should have at least 1 token for sample in every case.
                    max(extend_len - logprob_start_len, 1)
                    for logprob_start_len, extend_len in zip(
                        local_batch.extend_logprob_start_lens, local_batch.extend_lens
                    )
                ]
            )

        if local_batch is None or local_batch.forward_mode.is_decode_or_idle():
            can_cuda_graph = 1
        else:
            can_cuda_graph = 0

        if not spec_algorithm.is_none():
            # TODO(sang): Support cuda graph when idle batch is there.
            if local_batch is None or local_batch.forward_mode.is_idle():
                can_cuda_graph = 0

        is_extend_in_batch = (
            local_batch.forward_mode.is_extend() if local_batch else False
        )
        local_info = torch.tensor(
            [
                num_tokens,
                can_cuda_graph,
                global_num_tokens_for_logprob,
                is_extend_in_batch,
            ],
            dtype=torch.int64,
        )
        global_info = torch.empty(
            (dp_size, attn_tp_size, 4),
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            global_info.flatten(),
            local_info,
            group=tp_cpu_group,
        )
        global_num_tokens = global_info[:, 0, 0].tolist()
        can_cuda_graph = min(global_info[:, 0, 1].tolist())
        global_num_tokens_for_logprob = global_info[:, 0, 2].tolist()
        is_extend_in_batch = global_info[:, 0, 3].tolist()

        if local_batch is None and max(global_num_tokens) > 0:
            local_batch = get_idle_batch()

        if local_batch is not None:
            local_batch.global_num_tokens = global_num_tokens
            local_batch.global_num_tokens_for_logprob = global_num_tokens_for_logprob

            # Check forward mode for cuda graph
            if not disable_cuda_graph:
                local_batch.can_run_dp_cuda_graph = can_cuda_graph

        return local_batch, any(is_extend_in_batch)

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        for req in self.grammar_queue:
            try:
                req.grammar = req.grammar.result(timeout=0.05)
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                break

        if self.server_args.enable_dp_attention:
            tp_size = self.attn_tp_size
            tp_group = self.attn_tp_cpu_group
        else:
            tp_size = self.tp_size
            tp_group = self.tp_cpu_group

        if tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
            num_ready_reqs_max = tensor.item()
            for i in range(num_ready_reqs, num_ready_reqs_max):
                self.grammar_queue[i].grammar = self.grammar_queue[i].grammar.result()
            num_ready_reqs = num_ready_reqs_max

        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.time()

        while True:
            current = time.time()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        # Print batch size and memory pool info to check whether there are de-sync issues.
        logger.error(
            f"{self.cur_batch.batch_size()=}, "
            f"{self.cur_batch.reqs=}, "
            f"{self.token_to_kv_pool_allocator.available_size()=}, "
            f"{self.tree_cache.evictable_size()=}, "
        )
        # Wait for some time so that the parent process can print the error.
        pyspy_dump_schedulers()
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)

    def flush_cache_wrapped(self, recv_req: FlushCacheReq):
        self.flush_cache()

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if len(self.waiting_queue) == 0 and self.running_batch.is_empty():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()

            if not self.spec_algorithm.is_none():
                self.draft_worker.model_runner.req_to_token_pool.clear()
                self.draft_worker.model_runner.token_to_kv_pool_allocator.clear()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cum_spec_accept_length = 0
            self.cum_spec_accept_count = 0
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

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
            ret["avg_spec_accept_length"] = (
                self.cum_spec_accept_length / self.cum_spec_accept_count
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict
        return GetInternalStateReqOutput(
            internal_state=ret,
        )

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
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
        if if_success:
            if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
                avg_spec_accept_length = (
                    self.cum_spec_accept_length / self.cum_spec_accept_count
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.cum_spec_accept_length = self.cum_spec_accept_count = 0
            for k, v in server_args_dict.items():
                global_server_args_dict[k] = v
            logger.info(f"Global server args updated! " f"{global_server_args_dict=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=global_server_args_dict,
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
            func(recv_req.parameters)
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def save_remote_model(self, params):
        url = params["url"]

        worker = self.tp_worker.worker

        worker.model_runner.save_remote_model(url)

    def save_sharded_model(self, params):
        worker = self.tp_worker.worker

        worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if req.rid.startswith(recv_req.rid):
                to_del.append(i)
                break

        # Sort in reverse order to avoid index issues when deleting
        for i in sorted(to_del, reverse=True):
            req = self.waiting_queue.pop(i)
            logger.debug(f"Abort queued request. {req.rid=}")
            return

        # Delete requests in the running batch
        for req in self.running_batch.reqs:
            if req.rid.startswith(recv_req.rid) and not req.finished():
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True
                return

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        success, message = self.tp_worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flash_cache_success = self.flush_cache()
                assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        self.memory_saver_adapter.check_validity(
            caller_name="release_memory_occupation"
        )
        self.stashed_model_static_state = _export_static_state(
            self.tp_worker.worker.model_runner.model
        )
        self.memory_saver_adapter.pause()
        self.flush_cache()
        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        self.memory_saver_adapter.check_validity(caller_name="resume_memory_occupation")
        self.memory_saver_adapter.resume()
        _import_static_state(
            self.tp_worker.worker.model_runner.model, self.stashed_model_static_state
        )
        del self.stashed_model_static_state
        return ResumeMemoryOccupationReqOutput()

    def profile(self, recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            return self.start_profile(
                recv_req.output_dir,
                recv_req.num_steps,
                recv_req.activities,
                recv_req.with_stack,
                recv_req.record_shapes,
            )
        else:
            return self.stop_profile()

    def start_profile(
        self,
        output_dir: Optional[str],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
    ) -> None:
        if self.profiler_activities:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = output_dir
        self.profiler_activities = activities
        logger.info(
            "Profiling starts. Traces will be saved to: %s",
            self.torch_profiler_output_dir,
        )

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        if _is_hpu:
            activity_map["HPU"] = torch.profiler.ProfilerActivity.HPU
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
            )
            self.torch_profiler.start()

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)

        if "CUDA_PROFILER" in activities:
            torch.cuda.cudart().cudaProfilerStart()

        if num_steps:
            self.profiler_target_forward_ct = self.forward_ct + num_steps
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None
            return ProfileReqOutput(success=True, message="Succeeded")

    def stop_profile(self) -> None:
        if self.profiler_activities is None:
            return

        logger.info("Stop profiling...")
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            self.torch_profiler.export_chrome_trace(
                os.path.join(
                    self.torch_profiler_output_dir,
                    str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
                )
            )

        if "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                str(time.time()) + f"-TP-{self.tp_rank}-memory" + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            torch.cuda.cudart().cudaProfilerStop()

        logger.info(
            "Profiling done. Traces are saved to: %s",
            self.torch_profiler_output_dir,
        )
        self.torch_profiler = None
        self.torch_profiler_output_dir = None
        self.profiler_activities = None

        if self.profiler_target_forward_ct:
            self.send_to_tokenizer.send_pyobj(
                ProfileReqOutput(success=True, message="Succeeded.")
            )

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        if recv_req == ExpertDistributionReq.START_RECORD:
            expert_distribution_recorder.start_record()
        elif recv_req == ExpertDistributionReq.STOP_RECORD:
            expert_distribution_recorder.stop_record()
        elif recv_req == ExpertDistributionReq.DUMP_RECORD:
            expert_distribution_recorder.dump_record()
        else:
            raise ValueError("Unrecognized ExpertDistributionReq value")
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


def is_health_check_generate_req(recv_req):
    return getattr(recv_req, "rid", "").startswith("HEALTH_CHECK")


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the prefix
    if dp_rank is None:
        prefix = f" TP{tp_rank}"
    else:
        prefix = f" DP{dp_rank} TP{tp_rank}"

    # Config the process
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode

        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            scheduler.event_loop_normal_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
