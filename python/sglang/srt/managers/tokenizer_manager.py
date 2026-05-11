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
"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import copy
import logging
import os
import pickle
import signal
import socket
import sys
import threading
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import Awaitable, Dict, List, Optional, Tuple, Union

import fastapi
import pybase64
import torch
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers import logprob_ops, request_tracing, spec_decoding_meta
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    HealthCheckOutput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.mm_utils import wrap_shm_features
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.managers.scheduler import is_health_check_generate_req
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager_components.multimodal_processor_owner import (
    MultimodalProcessor,
)
from sglang.srt.managers.tokenizer_manager_components.raw_tokenizer_wrapper import (
    RawTokenizerWrapper,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import (
    ReqState,
    init_req,
)
from sglang.srt.managers.tokenizer_manager_components.request_validator import (
    RequestValidator,
    RequestValidatorConfig,
)
from sglang.srt.managers.tokenizer_manager_components.score_request_handler import (
    ScoreRequestHandler,
    ScoreRequestHandlerConfig,
)
from sglang.srt.managers.tokenizer_manager_components.tokenized_request_builder import (
    TokenizedRequestBuilder,
    TokenizedRequestBuilderConfig,
)
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    convert_time_to_realtime,
    real_time,
    set_time_batch,
)
from sglang.srt.observability.request_metrics_exporter import (
    RequestMetricsExporterManager,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_tokenizer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_gc_warning,
    freeze_gc,
    get_bool_env_var,
    get_or_create_event_loop,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.request_logger import RequestLogger
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

logger = logging.getLogger(__name__)


class TokenizerManager(TokenizerControlMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.preferred_sampling_params = server_args.preferred_sampling_params
        self.crash_dump_folder = server_args.crash_dump_folder
        set_global_server_args_for_tokenizer(server_args)

        # Init model config
        self.init_model_config()

        # Initialize tokenizer and multimodal processor
        self.raw_tokenizer_wrapper = RawTokenizerWrapper()
        self.raw_tokenizer_wrapper.init_tokenizer_and_processor(
            server_args=self.server_args,
            model_config=self.model_config,
        )

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init running status
        self.init_running_status()

        # Init logging and dumping
        self.init_request_logging_and_dumping()

        # Init weight update
        self.init_weight_update()

        # Init LoRA status
        self.init_lora()

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()

        # Multimodal processor
        self.multimodal_processor = MultimodalProcessor.from_server_args(
            server_args=self.server_args,
            model_config=self.model_config,
            mm_processor=self.mm_processor,
        )

        # Tokenized request builder
        self.tokenized_request_builder = TokenizedRequestBuilder(
            tokenizer=self.tokenizer,
            config=TokenizedRequestBuilderConfig(
                vocab_size=self.model_config.vocab_size,
                preferred_sampling_params=self.preferred_sampling_params,
                sampling_params_class=SamplingParams,
                disaggregation_transfer_backend=self.server_args.disaggregation_transfer_backend,
            ),
        )

        # Request validator
        self.request_validator = RequestValidator(
            config=RequestValidatorConfig(
                context_len=self.context_len,
                num_reserved_tokens=self.num_reserved_tokens,
                is_generation=self.is_generation,
                validate_total_tokens=self.validate_total_tokens,
                allow_auto_truncate=self.server_args.allow_auto_truncate,
                enable_return_hidden_states=self.server_args.enable_return_hidden_states,
                enable_custom_logit_processor=self.server_args.enable_custom_logit_processor,
                limit_mm_data_per_request=self.server_args.limit_mm_data_per_request,
                is_matryoshka=self.model_config.is_matryoshka,
                matryoshka_dimensions=self.model_config.matryoshka_dimensions,
                hidden_size=self.model_config.hidden_size,
                model_path=self.model_config.model_path,
            ),
        )

        # Score request handler
        self.score_request_handler = ScoreRequestHandler(
            tokenizer=self.tokenizer,
            rid_to_state=self.rid_to_state,
            generate_request=self.generate_request,
            config=ScoreRequestHandlerConfig(
                is_generation=self.is_generation,
                enable_mis=self.server_args.enable_mis,
                model_config=self.model_config,
            ),
        )

        # Init request dispatcher
        self.init_request_dispatcher()

    def init_model_config(self):
        server_args = self.server_args
        model_config_class = getattr(self, "model_config_class", ModelConfig)

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = model_config_class.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        self.max_req_input_len = None  # Will be set later in engine.py
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.default_priority_value = server_args.default_priority_value
        speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        if speculative_algorithm.is_eagle():
            # In the current eagle implementation, we store the draft tokens in the output token slots,
            # so we need to reserve the space for the draft tokens.
            self.num_reserved_tokens = max(
                server_args.speculative_eagle_topk * server_args.speculative_num_steps,
                server_args.speculative_num_draft_tokens,
            )
        else:
            self.num_reserved_tokens = 0
        self.validate_total_tokens = True

    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if self.server_args.tokenizer_worker_num == 1:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
        else:
            from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper

            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )

            # Make sure that each request carries the tokenizer_ipc_name for response routing
            self.send_to_scheduler = SenderWrapper(port_args, send_to_scheduler)

    def init_running_status(self):
        # Request states
        self.rid_to_state: Dict[str, ReqState] = {}
        self.event_loop = None
        self.asyncio_tasks = set()

        # Health check
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = real_time()

        # Session
        self.session_futures = {}  # session_id -> asyncio event

        # Subprocess liveness watchdog — set by Engine or http_server after construction
        self._subprocess_watchdog = None

    def init_request_logging_and_dumping(self):
        # TODO: Refactor and organize the log export code.
        # Request logging
        self.request_logger = RequestLogger(
            log_requests=self.server_args.log_requests,
            log_requests_level=self.server_args.log_requests_level,
            log_requests_format=self.server_args.log_requests_format,
            log_requests_target=self.server_args.log_requests_target,
        )

        # Dumping
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_requests_exclude_meta_keys: List[str] = [
            "routed_experts",
            "hidden_states",
        ]
        self.dump_request_list: List[Tuple] = []
        self.crash_dump_request_list: deque[Tuple] = deque()
        self.crash_dump_performed = False  # Flag to ensure dump is only called once

        # Initialize performance metrics loggers with proper skip names
        _, obj_skip_names, out_skip_names = self.request_logger.metadata
        self.request_metrics_exporter_manager = RequestMetricsExporterManager(
            self.server_args, obj_skip_names, out_skip_names
        )

    def init_weight_update(self):
        # Initial weights status
        self.initial_weights_loaded = True
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False

        # Weight updates
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

    def init_lora(self):
        # LoRA
        # Initialize the `LoRARegistry` with initial LoRA adapter paths provided in `server_args`.
        # The registry dynamically updates as adapters are loaded / unloaded during runtime. It
        # serves as the source of truth for available adapters and maps user-friendly LoRA names
        # to internally used unique LoRA IDs.
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)
        # Lock to serialize LoRA update operations.
        # Please note that, unlike `model_update_lock`, this does not block inference, allowing
        # LoRA updates and inference to overlap.
        self.lora_update_lock = asyncio.Lock()
        # A cache for mapping the lora_name for LoRA adapters that have been loaded at any
        # point to their latest LoRARef objects, so that they can be
        # dynamically loaded if needed for inference
        self.lora_ref_cache: Dict[str, LoRARef] = {}
        if self.server_args.lora_paths is not None:
            for lora_ref in self.server_args.lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref

    def init_disaggregation(self):
        # PD Disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        start_disagg_service(self.server_args)
        # Single-source counter for auto-assigning fake bootstrap_room.
        self.fake_bootstrap_room_counter = 0

    def init_metric_collector_watchdog(self):
        # Metrics
        if self.enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                self.server_args.disaggregation_mode
            )

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if self.server_args.tokenizer_metrics_allowed_custom_labels:
                for label in self.server_args.tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            self.metrics_collector = TokenizerMetricsCollector(
                server_args=self.server_args,
                labels=labels,
                bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
                bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
                bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
            )

            start_cpu_monitor_thread("tokenizer")

        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",
            watchdog_timeout=self.server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),
        )

    def init_request_dispatcher(self):
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                (HealthCheckOutput, lambda x: None),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self.init_communicators(self.server_args)

        self.sampling_params_class = SamplingParams
        self.signal_handler_class = SignalHandler

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()

        # Normalize the request
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.warning(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        init_req(
            self.rid_to_state,
            obj=obj,
            request=request,
            enable_trace=self.server_args.enable_trace,
            disagg_mode=self.disaggregation_mode,
        )
        if self.server_args.language_only:
            TokenizerManager._handle_epd_disaggregation_encode_request(
                self.multimodal_processor, obj
            )
        if self.server_args.tokenizer_worker_num > 1:
            self._attach_multi_http_worker_info(obj)

        # Log the request
        self.request_logger.log_received_request(obj, self.tokenizer, request)

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            await self._validate_and_resolve_lora(obj)

            # Tokenize the request and send it to the scheduler
            if obj.is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(tokenized_obj)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, request):
                    yield response

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        token_type_ids = None
        is_cross_encoder_request = (
            isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
        )
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )

            # For audio-only requests (e.g., Whisper), text may be empty.
            # The multimodal processor will provide input_ids later.
            if not input_text and self.mm_processor and obj.contains_mm_input():
                # Use empty placeholder - multimodal processor will override
                input_ids = []
            else:
                input_ids, token_type_ids = (
                    await self.raw_tokenizer_wrapper._tokenize_texts(
                        input_text, is_cross_encoder_request
                    )
                )

        contains_mm_input = obj.contains_mm_input()
        is_mossvl = (
            "MossVLForConditionalGeneration"
            in self.model_config.hf_config.architectures
        )
        should_run_mm_processor = self.mm_processor is not None and (
            contains_mm_input or is_mossvl
        )

        if should_run_mm_processor:
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if obj.video_data is not None and not isinstance(obj.video_data, list):
                obj.video_data = [obj.video_data]
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            if contains_mm_input:
                self.request_validator._validate_mm_limits(obj)

            mm_inputs = None

            if (
                not self.server_args.language_only
                or self.server_args.encoder_transfer_backend
                in ["zmq_to_tokenizer", "mooncake"]
            ):
                if self.server_args.language_only:
                    mm_inputs = (
                        await self.multimodal_processor.mm_receiver.recv_mm_data(
                            request_obj=obj,
                            mm_processor=self.mm_processor,
                            prompt=(input_text or input_ids),
                            need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                        )
                    )
                if mm_inputs is None:
                    mm_inputs = await self.mm_processor.process_mm_data_async(
                        image_data=obj.image_data,
                        audio_data=obj.audio_data,
                        input_text=(input_text or input_ids),
                        request_obj=obj,
                        max_req_input_len=self.max_req_input_len,
                    )
            elif (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
                and not obj.need_wait_for_mm_inputs
            ):
                # In language_only mode with zmq_to_scheduler, if we didn't dispatch
                # to encoder (e.g., only one image), process locally like non-language_only mode
                mm_inputs = await self.mm_processor.process_mm_data_async(
                    image_data=obj.image_data,
                    audio_data=obj.audio_data,
                    input_text=(input_text or input_ids),
                    request_obj=obj,
                    max_req_input_len=self.max_req_input_len,
                )

            if mm_inputs and mm_inputs.input_ids is not None:
                input_ids = mm_inputs.input_ids
            if mm_inputs and mm_inputs.token_type_ids is not None:
                token_type_ids = mm_inputs.token_type_ids
                if not isinstance(token_type_ids, list):
                    token_type_ids = token_type_ids.flatten().tolist()
            if (
                envs.SGLANG_MM_PRECOMPUTE_HASH.get()
                and mm_inputs
                and mm_inputs.mm_items
            ):
                for item in mm_inputs.mm_items:
                    if isinstance(item, MultimodalDataItem):
                        item.set_pad_value()
        else:
            mm_inputs = None

        self.request_validator.validate_one(obj=obj, input_ids=input_ids)
        tokenized_obj = self.tokenized_request_builder.build(
            obj,
            input_text,
            input_ids,
            input_embeds,
            mm_inputs,
            token_type_ids,
        )
        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()
        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only."""
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

        # If batch does not have text nothing to tokenize
        # so lets construct the return object
        if not self._batch_has_text(batch_size, obj):
            # All requests already have input_ids, no need to tokenize
            return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

        self.request_validator.validate_batch_tokenization_constraints(
            batch_size=batch_size, obj=obj
        )

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Check if any request is a cross-encoder request
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests
        )

        # Batch tokenize all texts using unified method
        input_ids_list, token_type_ids_list = (
            await self.raw_tokenizer_wrapper._tokenize_texts(
                texts, is_cross_encoder_request
            )
        )

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self.request_validator.validate_one(obj=obj[i], input_ids=input_ids_list[i])
            token_type_ids = (
                token_type_ids_list[i] if token_type_ids_list is not None else None
            )
            tokenized_obj = self.tokenized_request_builder.build(
                req,
                req.text,
                input_ids_list[i],
                None,
                None,
                token_type_ids,
            )
            tokenized_obj.time_stats = self.rid_to_state[req.rid].time_stats
            self.rid_to_state[req.rid].time_stats.set_tokenize_finish_time()
            tokenized_objs.append(tokenized_obj)
        logger.debug(f"Completed batch processing for {batch_size} requests")
        return tokenized_objs

    def _batch_has_text(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """Check if any request in the batch contains text input."""
        for i in range(batch_size):
            if obj[i].text:
                return True
            elif self.is_generation and obj[i].contains_mm_input():
                return True

        return False

    def _should_use_batch_tokenization(self, batch_size, requests) -> bool:
        """Return True if we should run the tokenizer in batch mode.

        Current policy:
        - Respect explicit server flag `enable_tokenizer_batch_encode`.
        - Or, if no request has text or multimodal input (all use pre-tokenized input_ids or input_embeds), batch the requests without tokenization.
        - Batch tokenization does not support DP attention yet, and it will make everything goes to the first rank currently
        """
        return batch_size > 0 and (
            self.server_args.enable_tokenizer_batch_encode
            or (
                (not self.server_args.enable_dp_attention)
                and (not self._batch_has_text(batch_size, requests))
            )
        )

    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        tokenized_obj.time_stats.set_api_server_dispatch_time()
        tokenized_obj = wrap_shm_features(tokenized_obj)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()

    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

        set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
        self.send_to_scheduler.send_pyobj(batch_req)
        set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")

    def _coalesce_streaming_chunks(
        self,
        out_list: list,
        rid: str,
    ) -> dict:
        """Coalesce multiple incremental streaming chunks into one.

        Both text and output_ids are incremental deltas, so we concatenate them;
        all other fields (meta_info, etc.) are taken from the last chunk.
        """
        if len(out_list) >= 20:
            logger.warning(
                "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                "This may inflate P99 ITL for affected requests.",
                rid,
                len(out_list),
            )
        out = dict(out_list[-1])
        if "output_ids" in out:
            out["output_ids"] = [id for chunk in out_list for id in chunk["output_ids"]]
        if "text" in out:
            out["text"] = "".join(chunk["text"] for chunk in out_list)
        if "meta_info" in out:
            meta_info_list = [chunk["meta_info"] for chunk in out_list]
            meta_info = dict(meta_info_list[-1])
            for key in logprob_ops.INCREMENTAL_STREAMING_META_INFO_KEYS:
                if any(key in m for m in meta_info_list):
                    meta_info[key] = [
                        item for m in meta_info_list for item in m.get(key, [])
                    ]
            out["meta_info"] = meta_info
        return out

    async def _handle_abort_finish_reason(
        self,
        out: dict,
        state: ReqState,
        is_stream: bool,
    ) -> Optional[dict]:
        """Handle abort/error finish reasons from the scheduler.

        Returns the output dict if it should be yielded (stream abort), or None
        for normal flow. Raises ValueError or HTTPException for non-stream aborts.
        """
        finish_reason = out["meta_info"]["finish_reason"]

        if (
            finish_reason.get("type") == "abort"
            and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
        ):
            if not is_stream:
                raise ValueError(finish_reason["message"])
            return out

        if finish_reason.get("type") == "abort" and finish_reason.get(
            "status_code"
        ) in (
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ):
            # Delete the key to prevent resending abort request to the scheduler and
            # to ensure aborted request state is cleaned up.
            if state.obj.rid in self.rid_to_state:
                del self.rid_to_state[state.obj.rid]

            # Mark ongoing LoRA request as finished.
            if self.server_args.enable_lora and state.obj.lora_path:
                await self.lora_registry.release(state.obj.lora_id)
            if not is_stream:
                raise fastapi.HTTPException(
                    status_code=finish_reason["status_code"],
                    detail=finish_reason["message"],
                )
            return out

        return None

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]
        # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue

            # Drain all pending outputs atomically.
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()

            # With incremental streaming, each chunk is a delta — coalesce
            # multiple queued chunks to avoid dropping token ids.
            incremental_stream = (
                is_stream and self.server_args.incremental_streaming_output
            )
            if incremental_stream and len(out_list) > 1:
                out = self._coalesce_streaming_chunks(out_list, obj.rid)
            else:
                out = out_list[-1]

            # Resolve deferred text for non-incremental streaming.
            # _handle_batch_output sets "text": None on intermediate chunks
            # to avoid O(n) string rebuild per step (O(n^2) total).
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None
            ):
                out["text"] = state.get_text()

            if finished:
                # Record response sent time right before we log finished results and metrics.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                self.request_logger.log_finished_request(
                    obj,
                    out,
                    request=request,
                )

                if self.request_metrics_exporter_manager.exporter_enabled():
                    asyncio.create_task(
                        self.request_metrics_exporter_manager.write_record(obj, out)
                    )

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    abort_out = await self._handle_abort_finish_reason(
                        out, state, is_stream
                    )
                    if abort_out is not None:
                        yield abort_out
                        break

                yield out
                break

            if is_stream:
                # Record response sent time right before we send response.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                yield out
            else:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = await self._tokenize_one_request(tmp_obj)
                        self._send_one_request(tokenized_obj)
                        generators.append(self._wait_one_response(tmp_obj, request))
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                init_req(
                    self.rid_to_state,
                    obj=tmp_obj,
                    enable_trace=self.server_args.enable_trace,
                    disagg_mode=self.disaggregation_mode,
                )
                self._send_one_request(tokenized_obj)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    init_req(
                        self.rid_to_state,
                        obj=tmp_obj,
                        enable_trace=self.server_args.enable_trace,
                        disagg_mode=self.disaggregation_mode,
                    )
                    tokenized_obj.time_stats = self.rid_to_state[tmp_obj.rid].time_stats
                    self._send_one_request(tokenized_obj)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()
                del self.rid_to_state[objs[i].rid]

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self.send_to_scheduler.send_pyobj(obj)
            else:
                # we are using the model_update_lock to check if there is still on-going requests.
                while True:
                    # TODO: maybe make it async instead of fire-and-forget
                    self.abort_request(abort_all=True)
                    is_locked = await self.model_update_lock.is_locked()
                    if not is_locked:
                        break
                    await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = False
            await self.send_to_scheduler.send_pyobj(obj)
            self.is_pause_cond.notify_all()

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            success, message, num_paused_requests = (
                await self._wait_for_model_update_from_disk(obj)
            )

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message, num_paused_requests

    def _update_model_path_info(self, model_path: str, load_format: str):
        self.served_model_name = model_path
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.model_path = model_path

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            result = await self.model_update_result
            if result.success:
                self._update_model_path_info(obj.model_path, obj.load_format)
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self._update_model_path_info(obj.model_path, obj.load_format)
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    def configure_logging(self, obj: ConfigureLoggingReq):
        self.request_logger.configure(
            log_requests=obj.log_requests,
            log_requests_level=obj.log_requests_level,
            log_requests_format=obj.log_requests_format,
        )
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.dump_requests_exclude_meta_keys is not None:
            self.dump_requests_exclude_meta_keys = list(
                obj.dump_requests_exclude_meta_keys
            )
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        logging.info(f"Config logging: {obj=}")

    async def freeze_gc(self):
        """Send a freeze_gc message to the scheduler first, then freeze locally."""
        self.send_to_scheduler.send_pyobj(FreezeGCReq())
        freeze_gc("Tokenizer Manager")
        return None

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if self.event_loop is not None:
            return

        # Create and start the handle_loop task
        loop = get_or_create_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )
        self.event_loop = loop

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = self.signal_handler_class(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )

        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self._handle_batch_output(recv_obj)
            else:
                self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()
            self.soft_watchdog.feed()

    async def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        pending_notify: dict[str, ReqState] = {}
        batch_notify_size = self.server_args.batch_notify_size
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                # Known race: /health_generate pops its rid as soon as ANY message bumps last_receive_tstamp.
                if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                    continue
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
                "num_retractions": recv_obj.retraction_counts[i],
            }

            if self.enable_metrics:
                if recv_obj.time_stats is not None:
                    scheduler_time_stats = recv_obj.time_stats[i]
                    meta_info.update(scheduler_time_stats.convert_to_output_meta_info())

            if getattr(state.obj, "return_logprob", False):
                logprob_ops.absorb_recv(
                    meta_info,
                    state,
                    top_logprobs_num=state.obj.top_logprobs_num,
                    token_ids_logprob=state.obj.token_ids_logprob,
                    return_text_in_logprobs=state.obj.return_text_in_logprobs
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj=recv_obj,
                    recv_obj_index=i,
                    tokenizer=self.tokenizer,
                )

            if not isinstance(recv_obj, BatchEmbeddingOutput):
                meta_info.update(
                    {
                        "reasoning_tokens": recv_obj.reasoning_tokens[i],
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )
                # Add detailed cache breakdown if available
                if (
                    hasattr(recv_obj, "cached_tokens_details")
                    and recv_obj.cached_tokens_details
                ):
                    meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[
                        i
                    ]

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]
            if getattr(recv_obj, "routed_experts", None):
                val = recv_obj.routed_experts[i]
                if val is not None:
                    # BatchStrOutput is pre-encoded by the detokenizer;
                    # BatchTokenIDOutput (skip_tokenizer_init) bypasses it.
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["routed_experts"] = val
            if getattr(recv_obj, "indexer_topk", None):
                val = recv_obj.indexer_topk[i]
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["indexer_topk"] = val
            if getattr(recv_obj, "customized_info", None):
                for k, v in recv_obj.customized_info.items():
                    meta_info[k] = v[i]
            if getattr(recv_obj, "dp_ranks", None):
                meta_info["dp_rank"] = recv_obj.dp_ranks[i]

            state.finished = recv_obj.finished_reasons[i] is not None
            if isinstance(recv_obj, BatchStrOutput):
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream
                )
                delta_text = recv_obj.output_strs[i]
                delta_output_ids = recv_obj.output_ids[i]
                output_offset = state.last_output_offset
                state.append_text(delta_text)
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info, output_offset
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "text": delta_text,
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "text": state.get_text(),
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        out_dict = {
                            "text": None,
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
            elif isinstance(recv_obj, BatchTokenIDOutput):
                is_stream = getattr(state.obj, "stream", False)
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream
                )
                delta_output_ids = recv_obj.output_ids[i]
                output_offset = state.last_output_offset
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info, output_offset
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        out_dict = {
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
            else:
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }
                if (
                    recv_obj.pooled_hidden_states is not None
                    and recv_obj.pooled_hidden_states[i] is not None
                ):
                    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]

            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()

            if state.finished:
                if state.time_stats.trace_ctx.tracing_enable:
                    state.time_stats.trace_ctx.trace_set_root_attrs(
                        request_tracing.make_span_attrs(
                            state=state,
                            recv_obj=recv_obj,
                            i=i,
                            served_model_name=self.served_model_name,
                        )
                    )
                state.time_stats.set_finished_time()
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

                if self.server_args.speculative_algorithm:
                    spec_decoding_meta.fill_spec_decoding_meta(
                        meta_info,
                        recv_obj=recv_obj,
                        i=i,
                        speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                    )
                if self.enable_metrics:
                    scheduler_time_stats = (
                        recv_obj.time_stats[i]
                        if recv_obj.time_stats is not None
                        else None
                    )
                    completion_tokens = (
                        recv_obj.completion_tokens[i]
                        if not isinstance(recv_obj, BatchEmbeddingOutput)
                        else 0
                    )
                    meta_info.update(
                        state.time_stats.convert_to_output_meta_info(
                            scheduler_time_stats, completion_tokens
                        )
                    )

                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.server_args.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            if out_dict is not None:
                state.out_list.append(out_dict)
                pending_notify[rid] = state

                if len(pending_notify) >= batch_notify_size:
                    for s in pending_notify.values():
                        s.event.set()
                    pending_notify = {}
                    await asyncio.sleep(0)

            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

        # handle_loop awaits next recv immediately
        for s in pending_notify.values():
            s.event.set()

        # When skip_tokenizer_init is enabled, tokensizer_manager receives
        # BatchTokenIDOutput.
        if (
            self.server_args.dp_size > 1
            and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
            and recv_obj.load is not None
        ):
            load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
            self.send_to_scheduler.send_pyobj(load_update_req)

    def _request_has_grammar(self, obj: GenerateReqInput) -> bool:
        return (
            obj.sampling_params.get("json_schema", None)
            or obj.sampling_params.get("regex", None)
            or obj.sampling_params.get("ebnf", None)
            or obj.sampling_params.get("structural_tag", None)
        )

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOutput, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        custom_labels = getattr(state.obj, "custom_labels", None)
        labels = dict(self.metrics_collector.labels)
        if custom_labels:
            labels.update(custom_labels)
        if self.enable_priority_scheduling:
            priority = getattr(state.obj, "priority", None)
            if priority is not None:
                labels["priority"] = str(priority)
        if (
            not state.ttft_observed
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.ttft_observed = True
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels, state.time_stats.get_first_token_latency()
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    state.time_stats.get_interval(),
                    num_new_tokens,
                )
                state.time_stats.set_last_time()
                state.last_completion_tokens = completion_tokens

        if state.finished:
            # Get detailed cache breakdown if available
            cached_tokens_details = None
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                cached_tokens_details = recv_obj.cached_tokens_details[i]

            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.time_stats.get_e2e_latency(),
                self._request_has_grammar(state.obj),
                cached_tokens_details,
            )

    def dump_requests(self, state: ReqState, out_dict: dict):
        if self.dump_requests_exclude_meta_keys and isinstance(
            out_dict.get("meta_info"), dict
        ):
            exclude = self.dump_requests_exclude_meta_keys
            if any(k in out_dict["meta_info"] for k in exclude):
                filtered_meta = {
                    k: v for k, v in out_dict["meta_info"].items() if k not in exclude
                }
                out_dict = {**out_dict, "meta_info": filtered_meta}

        self.dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                convert_time_to_realtime(state.time_stats.finished_time),
            )
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            self._dump_data_to_file(
                data_list=self.dump_request_list,
                filename=filename,
                log_message=f"Dump {len(self.dump_request_list)} requests to {filename}",
            )
            self.dump_request_list = []

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = real_time()
        self.crash_dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                current_time,
            )
        )
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _dump_data_to_file(
        self, data_list: List[Tuple], filename: str, log_message: str
    ):
        logger.info(log_message)
        to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_list.copy(),
        }

        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                try:
                    pickle.dump(to_dump_with_server_args, f)
                except Exception as e:
                    # When the server is launched with --trust-remote-code,
                    # server_args sometimes fails to pickle. Retry without
                    # server_args so the request data still gets persisted.
                    logger.error(
                        f"Failed to pickle dump with server_args: {e!r}; "
                        "retrying without server_args"
                    )
                    f.seek(0)
                    f.truncate()
                    to_dump_with_server_args["server_args"] = None
                    pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def dump_requests_before_crash(
        self, hostname: str = os.getenv("HOSTNAME", socket.gethostname())
    ):
        if not self.crash_dump_folder:
            return

        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        else:
            self.crash_dump_performed = True

        logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")

        # Add finished requests from crash_dump_request_list
        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                state.time_stats.set_finished_time()
                unfinished_requests.append(
                    (
                        state.obj,
                        (
                            state.out_list[-1]
                            if state.out_list
                            else state.get_crash_dump_output()
                        ),
                        convert_time_to_realtime(state.time_stats.created_time),
                        convert_time_to_realtime(state.time_stats.finished_time),
                    )
                )
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return

        # Create a file
        filename = os.path.join(
            self.crash_dump_folder,
            hostname,
            f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl',
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Write the data to the file
        data_to_dump_with_server_args = {
            "server_args": self.server_args,  # Include server_args in the dump
            "requests": data_to_dump,
            "launch_command": " ".join(sys.argv),
        }
        with open(filename, "wb") as f:
            try:
                pickle.dump(data_to_dump_with_server_args, f)
            except Exception as e:
                # When the server is launched with --trust-remote-code,
                # server_args sometimes fails to pickle. Retry without
                # server_args so the request data still gets persisted.
                logger.error(
                    f"Failed to pickle dump with server_args: {e!r}; "
                    "retrying without server_args"
                )
                f.seek(0)
                f.truncate()
                data_to_dump_with_server_args["server_args"] = None
                pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
        )
        return filename

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            remaining_rids = list(self.rid_to_state.keys())

            if self.server_status == ServerStatus.UnHealthy:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.dump_requests_before_crash()
                self.force_exit_handler()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                self.dump_requests_before_crash()
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    def force_exit_handler(self):
        """Put some custom force exit logic here."""
        pass

    def _handle_abort_req(self, recv_obj: AbortReq):
        if is_health_check_generate_req(recv_obj):
            return
        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.time_stats.set_finished_time()

        abort_message = recv_obj.abort_message or "Abort in waiting queue"
        finish_reason = {
            "type": "abort",
            "message": abort_message,
        }
        if recv_obj.finished_reason:
            finish_reason = recv_obj.finished_reason
        meta_info = {
            "id": recv_obj.rid,
            "finish_reason": finish_reason,
            "weight_version": self.server_args.weight_version,
            "e2e_latency": state.time_stats.get_e2e_latency(),
        }
        is_stream = getattr(state.obj, "stream", False)
        if getattr(state.obj, "return_logprob", False):
            logprob_ops.fill_meta_info(
                meta_info,
                state,
                top_logprobs_num=state.obj.top_logprobs_num,
                token_ids_logprob=state.obj.token_ids_logprob,
                return_text_in_logprobs=state.obj.return_text_in_logprobs
                and not self.server_args.skip_tokenizer_init,
                tokenizer=self.tokenizer,
            )

        output_ids = state.output_ids
        meta_info["completion_tokens"] = len(output_ids)
        if is_stream:
            output_ids = [output_ids[-1]] if len(output_ids) > 0 else []
        out = {
            "text": state.get_text(),
            "output_ids": output_ids,
            "meta_info": meta_info,
        }
        state.out_list.append(out)
        state.event.set()

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.send_to_scheduler.send_pyobj(ranks)

    def _handle_open_session_req_output(self, recv_obj):
        future = self.session_futures.get(recv_obj.session_id)
        if future is None:
            logger.warning(
                "Open session response arrived after waiter cleanup: %s",
                recv_obj.session_id,
            )
            return
        if not future.done():
            future.set_result(recv_obj.session_id if recv_obj.success else None)

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are received
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)

    async def _validate_and_resolve_lora(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        if not obj.lora_path:
            return

        if not self.server_args.enable_lora:
            first_adapter = (
                obj.lora_path
                if isinstance(obj.lora_path, str)
                else next((a for a in obj.lora_path if a), None)
            )

            raise ValueError(
                f"LoRA adapter '{first_adapter}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

        await self._resolve_lora_path(obj)

    async def _resolve_lora_path(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        if isinstance(obj.lora_path, str):
            unique_lora_paths = set([obj.lora_path])
        else:
            unique_lora_paths = set(obj.lora_path)

        if (
            self.server_args.max_loaded_loras is not None
            and len(unique_lora_paths) > self.server_args.max_loaded_loras
        ):
            raise ValueError(
                f"Received request with {len(unique_lora_paths)} unique loras requested "
                f"but max loaded loras is {self.server_args.max_loaded_loras}"
            )

        # Reload all existing LoRA adapters that have been dynamically unloaded
        unregistered_loras = await self.lora_registry.get_unregistered_loras(
            unique_lora_paths
        )
        for lora_path in unregistered_loras:
            if lora_path is None:
                continue

            if lora_path not in self.lora_ref_cache:
                raise ValueError(
                    f"Got LoRA adapter that has never been loaded: {lora_path}\n"
                    f"All loaded adapters: {self.lora_ref_cache.keys()}."
                )

            logger.info(f"Reloading evicted adapter: {lora_path}")
            new_lora_ref = self.lora_ref_cache[lora_path]
            load_result = await self.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=new_lora_ref.lora_name,
                    lora_path=new_lora_ref.lora_path,
                    pinned=new_lora_ref.pinned,
                )
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message
            ):
                raise ValueError(
                    f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"
                )

        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
        # Propagate lora_id to any sub-objects already cached by __getitem__.
        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
            sub_obj.lora_id = (
                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id
            )

    @staticmethod
    def _should_dispatch_to_encoder(
        self: "MultimodalProcessor",
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> bool:
        """Check if the request should be dispatched to encoder for processing.

        Returns True if the request should be dispatched to encoder (multiple multimodal items),
        False if it should be processed locally (single multimodal item or no multimodal items).

        Args:
            obj: The request input object

        Returns:
            bool: True if should dispatch to encoder, False otherwise
        """
        if obj.batch_size > 1:
            logger.warning(
                "Batch request (batch_size=%d) is not supported in EPD disaggregation mode; skipping encoder dispatch.",
                obj.batch_size,
            )
            return False
        if not isinstance(obj, GenerateReqInput) or not obj.contains_mm_input():
            return False

        # Count image / video / audio items for dispatch threshold
        def _count_mm_items(data):
            return (
                len(data) if isinstance(data, list) else (1 if data is not None else 0)
            )

        total_mm_items = (
            _count_mm_items(getattr(obj, "image_data", None))
            + _count_mm_items(getattr(obj, "video_data", None))
            + _count_mm_items(getattr(obj, "audio_data", None))
        )
        return total_mm_items >= self.config.encoder_dispatch_min_items

    @staticmethod
    def _handle_epd_disaggregation_encode_request(
        self: "MultimodalProcessor",
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Handle EPD-disaggregation mode encoding request."""
        if isinstance(obj, GenerateReqInput) and obj.contains_mm_input():
            # dispatch to encoder by default
            should_dispatch = True
            if self.config.enable_adaptive_dispatch_to_encoder:
                should_dispatch = TokenizerManager._should_dispatch_to_encoder(
                    self, obj
                )

            # Set need_wait_for_mm_inputs flag based on whether we dispatch to encoder
            # This flag will be used in _tokenize_one_request to determine processing path
            if should_dispatch:
                obj.need_wait_for_mm_inputs = True
                if self.config.encoder_transfer_backend == "zmq_to_scheduler":
                    self.mm_receiver.send_encode_request(obj)
            else:
                obj.need_wait_for_mm_inputs = False

    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value."""
        if (
            self.enable_priority_scheduling
            and obj.priority is None
            and self.default_priority_value is not None
        ):
            obj.priority = self.default_priority_value

    # ---- raw_tokenizer_wrapper facade -----------------------------------
    # ``tokenizer`` / ``processor`` / ``mm_processor`` are the TokenizerManager
    # public read-API; storage is delegated to ``self.raw_tokenizer_wrapper``.
    # Read-only by design — writes go through ``self.raw_tokenizer_wrapper``
    # directly (the only writes happen inside
    # ``RawTokenizerWrapper.init_tokenizer_and_processor``).
    # ``async_dynamic_batch_tokenizer`` stays internal — access via
    # ``self.raw_tokenizer_wrapper.async_dynamic_batch_tokenizer`` directly.

    @property
    def tokenizer(self):
        return self.raw_tokenizer_wrapper.tokenizer

    @property
    def processor(self):
        return self.raw_tokenizer_wrapper.processor

    @property
    def mm_processor(self):
        return self.raw_tokenizer_wrapper.mm_processor


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error(
            f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed."
        )
        # Stop subprocess watchdog before killing processes to prevent false-positive
        # crash detection during normal shutdown
        if self.tokenizer_manager._subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog.stop()
        self.tokenizer_manager.dump_requests_before_crash()
        kill_process_tree(os.getpid())


# Note: request abort handling logic
# We should handle all of the following cases correctly.
#
# | entrypoint | is_streaming | status          | abort engine    | cancel asyncio task   | rid_to_state                |
# | ---------- | ------------ | --------------- | --------------- | --------------------- | --------------------------- |
# | http       | yes          | validation      | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | waiting queue   | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | running         | background task | fast api              | del in _handle_batch_output |
# | http       | no           | validation      | http exception  | http exception        | del in _handle_abort_req    |
# | http       | no           | waiting queue   | type 1          | type 1 exception      | del in _handle_abort_req    |
# | http       | no           | running         | type 3          | type 3 exception      | del in _handle_batch_output |
#
