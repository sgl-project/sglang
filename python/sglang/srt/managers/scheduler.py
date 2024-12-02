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

import logging
import os
import signal
import threading
import time
import warnings
from collections import deque
from concurrent import futures
from types import SimpleNamespace
from typing import List, Optional

import psutil
import torch
import zmq

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    FlushCacheReq,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    ImageInputs,
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    configure_logger,
    crash_on_warnings,
    get_bool_env_var,
    get_zmq_socket,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode
test_retract = get_bool_env_var("SGLANG_TEST_RETRACT")


class Scheduler:
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
        self.disable_jump_forward = server_args.disable_jump_forward
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics

        # Session info
        self.sessions = {}

        # Init inter-process communication
        context = zmq.Context(2)

        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name
            )

            if server_args.skip_tokenizer_init:
                # Directly send to the tokenizer/api
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name
                )
            else:
                # Send to the detokenizer
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name
                )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init tokenizer
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
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        if self.model_config.is_multimodal:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for multimodal models.")

        if self.enable_overlap:
            self.disable_jump_forward = True

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
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init memory pool and cache
        self.req_to_token_pool, self.token_to_kv_pool = self.tp_worker.get_memory_pool()

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: Optional[ScheduleBatch] = None
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.time()
        self.stream_interval = server_args.stream_interval

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.being_chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            if server_args.grammar_backend == "outlines":
                from sglang.srt.constrained.outlines_backend import (
                    OutlinesGrammarBackend,
                )

                self.grammar_backend = OutlinesGrammarBackend(
                    self.tokenizer,
                    whitespace_pattern=server_args.constrained_json_whitespace_pattern,
                    allow_jump_forward=not server_args.disable_jump_forward,
                )
            elif server_args.grammar_backend == "xgrammar":
                from sglang.srt.constrained.xgrammar_backend import (
                    XGrammarGrammarBackend,
                )

                self.grammar_backend = XGrammarGrammarBackend(
                    self.tokenizer, vocab_size=self.model_config.vocab_size
                )
            else:
                raise ValueError(
                    f"Invalid grammar backend: {server_args.grammar_backend}"
                )
        else:
            self.grammar_backend = None

        # Init new token estimation
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

        # Tells whether the current running batch is full so that we can skip
        # the check of whether to prefill new requests.
        # This is an optimization to reduce the overhead of the prefill check.
        self.batch_is_full = False

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init profiler
        if os.getenv("SGLANG_TORCH_PROFILER_DIR", "") == "":
            self.profiler = None
        else:
            self.torch_profiler_trace_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                self.torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )

        # Init metrics stats
        self.stats = SchedulerStats()
        if self.enable_metrics:
            self.metrics_collector = SchedulerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    # TODO: Add lora name/path in the future,
                },
            )

    def watchdog_thread(self):
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.time()

        while True:
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if time.time() > self.watchdog_last_time + self.watchdog_timeout:
                        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = time.time()
            time.sleep(self.watchdog_timeout / 2)

        self.parent_process.send_signal(signal.SIGQUIT)

    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # A dummy first batch to start the pipeline for overlap scheduler.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None)

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
        elif local_batch.forward_mode.is_decode():
            num_tokens = local_batch.batch_size()
        else:
            num_tokens = local_batch.extend_num_tokens

        local_num_tokens = torch.tensor([num_tokens], dtype=torch.int64)
        global_num_tokens = torch.empty(self.tp_size, dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(
            global_num_tokens,
            local_num_tokens,
            group=self.tp_cpu_group,
        )

        if local_batch is None and global_num_tokens.max().item() > 0:
            local_batch = self.get_idle_batch()

        if local_batch is not None:
            local_batch.global_num_tokens = global_num_tokens.tolist()

            # Check forward mode for cuda graph
            if not self.server_args.disable_cuda_graph:
                forward_mode_state = torch.tensor(
                    (
                        1
                        if local_batch.forward_mode.is_decode()
                        or local_batch.forward_mode.is_idle()
                        else 0
                    ),
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    forward_mode_state,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_cpu_group,
                )
                local_batch.can_run_dp_cuda_graph = forward_mode_state.item() == 1

        return local_batch

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def recv_requests(self):
        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size != 1 and not self.server_args.enable_dp_attention:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                self.handle_generate_request(recv_req)
            elif isinstance(recv_req, TokenizedEmbeddingReqInput):
                self.handle_embedding_request(recv_req)
            elif isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req)
            elif isinstance(recv_req, UpdateWeightFromDiskReqInput):
                success, message = self.update_weights_from_disk(recv_req)
                self.send_to_tokenizer.send_pyobj(
                    UpdateWeightFromDiskReqOutput(success, message)
                )
            elif isinstance(recv_req, GetWeightsByNameReqInput):
                parameter = self.get_weights_by_name(recv_req)
                self.send_to_tokenizer.send_pyobj(GetWeightsByNameReqOutput(parameter))
            elif isinstance(recv_req, InitWeightsUpdateGroupReqInput):
                success, message = self.init_weights_update_group(recv_req)
                self.send_to_tokenizer.send_pyobj(
                    InitWeightsUpdateGroupReqOutput(success, message)
                )
            elif isinstance(recv_req, UpdateWeightsFromDistributedReqInput):
                success, message = self.update_weights_from_distributed(recv_req)
                self.send_to_tokenizer.send_pyobj(
                    UpdateWeightsFromDistributedReqOutput(success, message)
                )
            elif isinstance(recv_req, GetWeightsByNameReqInput):
                parameter = self.get_weights_by_name(recv_req)
                self.send_to_tokenizer.send_pyobj(GetWeightsByNameReqOutput(parameter))
            elif isinstance(recv_req, ProfileReq):
                if recv_req == ProfileReq.START_PROFILE:
                    self.start_profile()
                else:
                    self.stop_profile()
            elif isinstance(recv_req, OpenSessionReqInput):
                session_id = self.open_session(recv_req)
                self.send_to_tokenizer.send_pyobj(OpenSessionReqOutput(session_id))
            elif isinstance(recv_req, CloseSessionReqInput):
                self.close_session(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request
        if recv_req.session_id is None or recv_req.session_id not in self.sessions:

            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                lora_path=recv_req.lora_path,
                input_embeds=recv_req.input_embeds,
            )
            req.tokenizer = self.tokenizer

            if recv_req.session_id is not None:
                req.finished_reason = FINISH_ABORT(
                    f"Invalid request: session id {recv_req.session_id} does not exist"
                )
                self.waiting_queue.append(req)
                return
        else:
            # Create a new request from a previsou session
            session = self.sessions[recv_req.session_id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.waiting_queue.append(req)
                return

        # Handle image inputs
        if recv_req.image_inputs is not None:
            image_inputs = ImageInputs.from_dict(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                logger.error(
                    "Multimodal prompt is too long after expanding multimodal tokens. "
                    f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}. "
                )
                req.origin_input_ids = [0]
                req.image_inputs = None
                req.sampling_params.max_new_tokens = 0
                req.finished_reason = FINISH_ABORT(
                    "Multimodal prompt is too long. Check server logs for details."
                )
                self.waiting_queue.append(req)
                return

        # Copy more attributes
        req.return_logprob = recv_req.return_logprob
        req.top_logprobs_num = recv_req.top_logprobs_num
        req.stream = recv_req.stream
        req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len == -1:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(recv_req.input_ids) - 1

        # Truncate prompts that are too long
        if len(req.origin_input_ids) > self.max_req_input_len:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

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
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)

            req.grammar = self.grammar_backend.get_cached_value(key)
            if not req.grammar:
                req.grammar = self.grammar_backend.get_future_value(key)
                add_to_grammar_queue = True

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
            self.waiting_queue.append(req)

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

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        self.waiting_queue.append(req)

    def log_prefill_stats(self, adder, can_run_list, running_bs, has_being_chunked):
        if isinstance(self.tree_cache, RadixCache):
            self.tree_cache_metrics["total"] += (
                adder.log_input_tokens + adder.log_hit_tokens
            ) / 10**9
            self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
            tree_cache_hit_rate = (
                self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
            )
        else:
            tree_cache_hit_rate = 0.0

        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )

        logger.info(
            f"Prefill batch. "
            f"#new-seq: {len(can_run_list)}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"#running-req: {running_bs}, "
            f"#queue-req: {len(self.waiting_queue) + has_being_chunked}"
        )

        if self.enable_metrics:
            self.stats.num_running_reqs = running_bs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(num_used / self.max_total_num_tokens, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue) + has_being_chunked
            self.stats.cache_hit_rate = tree_cache_hit_rate
            self.metrics_collector.log_stats(self.stats)

    def log_decode_stats(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        gen_throughput = self.num_generated_tokens / (
            time.time() - self.last_decode_stats_tic
        )
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.time()
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        logger.info(
            f"Decode batch. "
            f"#running-req: {num_running_reqs}, "
            f"#token: {num_used}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        if self.enable_metrics:
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = num_used / self.max_total_num_tokens
            self.stats.gen_throughput = gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.metrics_collector.log_stats(self.stats)

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            msg = (
                "KV cache pool leak detected!"
                f"{available_size=}, {self.max_total_num_tokens=}\n"
            )
            warnings.warn(msg)
            if crash_on_warnings():
                raise ValueError(msg)

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            msg = (
                "Memory pool leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            warnings.warn(msg)
            if crash_on_warnings():
                raise ValueError(msg)

    def get_next_batch_to_run(self):
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.being_chunked_req:
                # Move the chunked request out of the batch
                self.last_batch.filter_batch(being_chunked_req=self.being_chunked_req)
                self.tree_cache.cache_unfinished_req(self.being_chunked_req)
                # being chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.being_chunked_req.req_pool_idx)
                self.batch_is_full = False

            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        # Run prefill first if possible
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch

        # Run decode
        if self.running_batch is None:
            return None
        self.running_batch = self.update_running_batch(self.running_batch)
        return self.running_batch

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.batch_is_full or len(self.waiting_queue) == 0
        ) and self.being_chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        if running_bs >= self.max_running_requests:
            self.batch_is_full = True
            return None

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.tree_cache,
            self.running_batch,
            self.new_token_ratio,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        has_being_chunked = self.being_chunked_req is not None
        if has_being_chunked:
            self.being_chunked_req.init_next_round_input()
            self.being_chunked_req = adder.add_being_chunked_req(self.being_chunked_req)

        if self.lora_paths:
            lora_set = (
                set([req.lora_path for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

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
                self.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.batch_is_full = True
                break

            req.init_next_round_input(None if prefix_computed else self.tree_cache)
            res = adder.add_one_req(req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.batch_is_full = True
                break

        # Update waiting queue
        can_run_list = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_being_chunked_req is not None:
            assert self.being_chunked_req is None
            self.being_chunked_req = adder.new_being_chunked_req

        if self.being_chunked_req:
            self.being_chunked_req.is_being_chunked += 1

        # Print stats
        if self.tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs, has_being_chunked)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
        )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and self.running_batch is not None
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        global test_retract

        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            self.batch_is_full = False
            return None

        # Check if decode out of memory
        if not batch.check_decode_mem() or (test_retract and batch.batch_size() > 10):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.waiting_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        # Check for jump-forward
        if not self.disable_jump_forward:
            jump_forward_reqs = batch.check_for_jump_forward(self.pad_input_ids_func)
            self.waiting_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                self.batch_is_full = False
                return None

        if batch.batch_size() < initial_bs:
            self.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def run_batch(self, batch: ScheduleBatch):
        """Run a batch."""
        self.forward_ct += 1

        if self.is_generation:
            model_worker_batch = batch.get_model_worker_batch()
            if batch.forward_mode.is_decode() or batch.extend_num_tokens != 0:
                logits_output, next_token_ids = self.tp_worker.forward_batch_generation(
                    model_worker_batch
                )
            elif batch.forward_mode.is_idle():
                model_worker_batch = batch.get_model_worker_batch()
                self.tp_worker.forward_batch_idle(model_worker_batch)
                return
            else:
                logits_output = None
                if self.skip_tokenizer_init:
                    next_token_ids = torch.full(
                        (batch.batch_size(),), self.tokenizer.eos_token_id
                    )
                else:
                    next_token_ids = torch.full((batch.batch_size(),), 0)
            batch.output_ids = next_token_ids
            ret = logits_output, next_token_ids, model_worker_batch.bid
        else:  # embedding or reward model
            assert batch.extend_num_tokens != 0
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = embeddings, model_worker_batch.bid
        return ret

    def process_batch_result(self, batch: ScheduleBatch, result):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
            if batch.is_empty():
                self.running_batch = None
        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_dummy_first():
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            torch.cuda.current_stream().synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def process_batch_result_prefill(self, batch: ScheduleBatch, result):

        if self.is_generation:
            logits_output, next_token_ids, bid = result

            if self.enable_overlap:
                logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            else:
                # Move next_token_ids and logprobs to cpu
                if batch.return_logprob:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs[
                            torch.arange(len(next_token_ids), device=self.device),
                            next_token_ids,
                        ].tolist()
                    )
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.tolist()
                    )
                    logits_output.normalized_prompt_logprobs = (
                        logits_output.normalized_prompt_logprobs.tolist()
                    )
                next_token_ids = next_token_ids.tolist()

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_being_chunked <= 0:
                    req.completion_tokens_wo_jump_forward += 1
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        self.tree_cache.cache_unfinished_req(req)

                    if req.return_logprob:
                        logprob_pt += self.add_logprob_return_values(
                            i, req, logprob_pt, next_token_ids, logits_output
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_being_chunked -= 1

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                torch.cuda.current_stream().synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_being_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_being_chunked -= 1

        self.stream_output(batch.reqs)

    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        logits_output, next_token_ids, bid = result
        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            next_token_logprobs = logits_output.next_token_logprobs
        else:
            # Move next_token_ids and logprobs to cpu
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs[
                    torch.arange(len(next_token_ids), device=self.device),
                    next_token_ids,
                ].tolist()
            next_token_ids = next_token_ids.tolist()

        self.token_to_kv_pool.free_group_begin()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one delayed token
                self.token_to_kv_pool.free(batch.out_cache_loc[i : i + 1])
                continue

            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob:
                req.output_token_logprobs.append(
                    (next_token_logprobs[i], next_token_id)
                )
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs.append(logits_output.output_top_logprobs[i])

            if req.grammar is not None:
                req.grammar.accept_token(next_token_id)

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            torch.cuda.current_stream().synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        self.stream_output(batch.reqs)

        self.token_to_kv_pool.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
        num_input_logprobs = req.extend_input_len - req.extend_logprob_start_len

        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            input_token_logprobs = output.input_token_logprobs[
                pt : pt + num_input_logprobs - 1 - req.last_update_decode_tokens
            ]
            input_token_ids = req.fill_ids[
                len(req.fill_ids)
                - num_input_logprobs
                + 1 : len(req.fill_ids)
                - req.last_update_decode_tokens
            ]

            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_ids = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_ids
            ]

            req.input_token_logprobs = list(zip(input_token_logprobs, input_token_ids))

            if (
                req.logprob_start_len == 0
            ):  # The first token does not have logprob, pad it.
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            # Some decode tokens are re-computed in an extend batch
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + num_input_logprobs
                            - 1
                            - req.last_update_decode_tokens : pt
                            + num_input_logprobs
                            - 1
                        ],
                        req.fill_ids[
                            len(req.fill_ids)
                            - req.last_update_decode_tokens : len(req.fill_ids)
                        ],
                    )
                )
            )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

        return num_input_logprobs

    def stream_output(self, reqs: List[Req]):
        """Stream the output to detokenizer."""
        output_rids = []
        output_meta_info: List[dict] = []
        output_finished_reason: List[BaseFinishReason] = []
        if self.is_generation:
            output_vids = []
            decoded_texts = []
            output_read_ids = []
            output_read_offsets = []
            output_ids = []
            output_skip_special_tokens = []
            output_spaces_between_special_tokens = []
            output_no_stop_trim = []
        else:  # embedding or reward model
            output_embeddings = []

        is_stream_iter = self.forward_ct_decode % self.stream_interval == 0

        for req in reqs:
            # TODO(lianmin): revisit this for overlap + retract + stream
            if req.finished() or (
                req.stream and (is_stream_iter or len(req.output_ids) == 1)
            ):
                output_rids.append(req.rid)
                output_finished_reason.append(req.finished_reason)
                if self.is_generation:
                    output_vids.append(req.vid)
                    decoded_texts.append(req.decoded_text)
                    read_ids, read_offset = req.init_incremental_detokenize()
                    output_read_ids.append(read_ids)
                    output_read_offsets.append(read_offset)
                    if self.skip_tokenizer_init:
                        output_ids.append(req.output_ids)
                    output_skip_special_tokens.append(
                        req.sampling_params.skip_special_tokens
                    )
                    output_spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )
                    output_no_stop_trim.append(req.sampling_params.no_stop_trim)

                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                        "completion_tokens": len(req.output_ids),
                        "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                        "cached_tokens": req.cached_tokens,
                        "finish_reason": (
                            req.finished_reason.to_json()
                            if req.finished_reason is not None
                            else None
                        ),
                    }
                    if req.return_logprob:
                        (
                            meta_info["input_token_logprobs"],
                            meta_info["output_token_logprobs"],
                            meta_info["input_top_logprobs"],
                            meta_info["output_top_logprobs"],
                            meta_info["normalized_prompt_logprob"],
                        ) = (
                            req.input_token_logprobs,
                            req.output_token_logprobs,
                            req.input_top_logprobs,
                            req.output_top_logprobs,
                            req.normalized_prompt_logprob,
                        )
                    output_meta_info.append(meta_info)
                else:  # embedding or reward model
                    output_embeddings.append(req.embedding)
                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                    }
                    output_meta_info.append(meta_info)

        # Send to detokenizer
        if output_rids:
            if self.is_generation:
                self.send_to_detokenizer.send_pyobj(
                    BatchTokenIDOut(
                        output_rids,
                        output_vids,
                        decoded_texts,
                        output_read_ids,
                        output_read_offsets,
                        output_ids,
                        output_skip_special_tokens,
                        output_spaces_between_special_tokens,
                        output_meta_info,
                        output_finished_reason,
                        output_no_stop_trim,
                    )
                )
            else:  # embedding or reward model
                self.send_to_detokenizer.send_pyobj(
                    BatchEmbeddingOut(
                        output_rids,
                        output_embeddings,
                        output_meta_info,
                        output_finished_reason,
                    )
                )

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        for req in self.grammar_queue:
            try:
                req.grammar = req.grammar.result(timeout=0.05)
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                break

        if self.tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group
            )
            num_ready_reqs_max = tensor.item()
            for i in range(num_ready_reqs, num_ready_reqs_max):
                self.grammar_queue[i].grammar = self.grammar_queue[i].grammar.result()
            num_ready_reqs = num_ready_reqs_max

        self.waiting_queue.extend(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]
            logger.debug(f"Abort queued request. {req.rid=}")
            return

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid and not req.finished():
                    logger.debug(f"Abort running request. {req.rid=}")
                    req.to_abort = True
                    break

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return parameter

    def start_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()
        self.profiler.export_chrome_trace(
            self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
        )
        logger.info("Profiler is done")

    def open_session(self, recv_req: OpenSessionReqInput) -> str:
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
        return session_id

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")

    suppress_other_loggers()
    parent_process = psutil.Process().parent()

    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {"status": "ready", "max_total_num_tokens": scheduler.max_total_num_tokens}
        )
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
