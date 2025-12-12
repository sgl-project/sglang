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
import dataclasses
import logging
import os
import pickle
import signal
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

import fastapi
import orjson
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.async_mm_data_processor import AsyncMMDataProcessor
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchMultimodalOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    GetLoadReqInput,
    HealthCheckOutput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.mm_utils import TensorTransportMode
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.request_metrics_exporter import RequestMetricsExporterManager
from sglang.srt.managers.schedule_batch import RequestStage
from sglang.srt.managers.scheduler import is_health_check_generate_req
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_communicator_mixin import TokenizerCommunicatorMixin
from sglang.srt.managers.tokenizer_manager_multiitem_mixin import (
    TokenizerManagerMultiItemMixin,
)
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_tokenizer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.tracing.trace import (
    extract_trace_headers,
    trace_get_proc_propagate_context,
    trace_req_finish,
    trace_req_start,
    trace_set_remote_propagate_context,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import (
    configure_gc_warning,
    dataclass_to_string_truncated,
    freeze_gc,
    get_bool_env_var,
    get_or_create_event_loop,
    get_zmq_socket,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For metrics
    created_time: float
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # perf_counter equivalents for accurate time calculations
    finished_time_perf: float = 0.0
    first_token_time_perf: float = 0.0

    request_sent_to_scheduler_ts: float = 0.0
    response_sent_to_client_ts: float = 0.0

    # For streaming output
    last_output_offset: int = 0

    # For incremental state update.
    # TODO(lianmin): do not initialize some lists if not needed.
    text: str = ""
    output_ids: List[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)


class InputFormat(Enum):
    """Input format types for tokenization handling."""

    SINGLE_STRING = 1  # Regular single text like "Hello world"
    BATCH_STRINGS = 2  # Regular batch like ["Hello", "World"]
    CROSS_ENCODER_PAIRS = 3  # Cross-encoder pairs like [["query", "document"]]


class TokenizerManager(TokenizerCommunicatorMixin, TokenizerManagerMultiItemMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.log_requests
        self.log_requests_level = server_args.log_requests_level
        self.preferred_sampling_params = (
            orjson.loads(server_args.preferred_sampling_params)
            if server_args.preferred_sampling_params
            else None
        )
        self.crash_dump_folder = server_args.crash_dump_folder
        self.enable_trace = server_args.enable_trace

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        self.is_image_gen = self.model_config.is_image_gen
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        self.max_req_input_len = None  # Will be set later in engine.py
        speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.reserve_input_token_num = (
            0
            if speculative_algorithm.is_none()
            else server_args.speculative_num_draft_tokens
        )

        set_global_server_args_for_tokenizer(server_args)

        # Initialize tokenizer and processor
        if self.model_config.is_multimodal:
            import_processors("sglang.srt.multimodal.processors")
            if envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.value:
                import_processors(
                    envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.value, overwrite=True
                )
            _processor = _get_processor_wrapper(server_args)
            transport_mode = _determine_tensor_transport_mode(self.server_args)

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config, server_args, _processor, transport_mode
            )
            self.mm_data_processor = AsyncMMDataProcessor(
                self.mm_processor,
                max_concurrent_calls=self.server_args.mm_max_concurrent_calls,
                timeout_s=self.server_args.mm_per_request_timeout,
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self._initialize_multi_item_delimiter_text()
        else:
            self.mm_processor = self.processor = None

            if server_args.skip_tokenizer_init:
                self.tokenizer = None
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self._initialize_multi_item_delimiter_text()

        # Initialize async dynamic batch tokenizer if enabled (common for both multimodal and non-multimodal)
        if (
            server_args.enable_dynamic_batch_tokenizer
            and not server_args.skip_tokenizer_init
        ):
            self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
                self.tokenizer,
                max_batch_size=server_args.dynamic_batch_tokenizer_batch_size,
                batch_wait_timeout_s=server_args.dynamic_batch_tokenizer_batch_timeout,
            )
        else:
            self.async_dynamic_batch_tokenizer = None

        # Init inter-process communication
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

        # Request states
        self._chosen_loop = None
        self.rid_to_state: Dict[str, ReqState] = {}
        self.asyncio_tasks = set()

        # Health check
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = 0

        # Initial weights status
        self.initial_weights_loaded = True
        if server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False

        # Dumping
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: List[Tuple] = []
        self.log_request_metadata = self.get_log_request_metadata()
        self.crash_dump_request_list: deque[Tuple] = deque()
        self.crash_dump_performed = False  # Flag to ensure dump is only called once

        # Initialize performance metrics loggers with proper skip names
        _, obj_skip_names, out_skip_names = self.log_request_metadata
        self.request_metrics_exporter_manager = RequestMetricsExporterManager(
            self.server_args, obj_skip_names, out_skip_names
        )

        # Session
        self.session_futures = {}  # session_id -> asyncio event

        # Weight updates
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

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

        # Disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.bootstrap_server = start_disagg_service(self.server_args)

        # For load balancing
        self.current_load = 0
        self.current_load_lock = asyncio.Lock()

        # Metrics
        if self.enable_metrics:
            labels = {
                "model_name": self.server_args.served_model_name,
                # TODO: Add lora name/path in the future,
            }
            if server_args.tokenizer_metrics_allowed_custom_labels:
                for label in server_args.tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""
            self.metrics_collector = TokenizerMetricsCollector(
                server_args=server_args,
                labels=labels,
                bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
                bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
                bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
                collect_tokens_histogram=self.server_args.collect_tokens_histogram,
            )

        # Configure GC warning
        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)

        # Dispatcher and communicators
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOutput,
                        BatchEmbeddingOutput,
                        BatchTokenIDOutput,
                        BatchMultimodalOutput,
                    ),
                    self._handle_batch_output,
                ),
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                (HealthCheckOutput, lambda x: None),
            ]
        )
        self.init_communicators(server_args)

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = obj.received_time if obj.received_time else time.time()
        self.auto_create_handle_loop()
        obj.normalize_batch_and_arguments()

        if self.enable_trace:
            self._trace_request_start(obj, created_time, request)
        if self.server_args.tokenizer_worker_num > 1:
            self._attach_multi_http_worker_info(obj)
        if self.log_requests:
            self._log_received_request(obj)

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            if self.server_args.enable_lora and obj.lora_path:
                await self._resolve_lora_path(obj)

            if obj.is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                state = self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj, state, request):
                    yield response
            else:
                async for response in self._handle_batch_request(
                    obj, request, created_time
                ):
                    yield response

    def _detect_input_format(
        self, texts: Union[str, List[str]], is_cross_encoder: bool
    ) -> InputFormat:
        """Detect the format of input texts for proper tokenization handling.

        Returns:
            - InputFormat.SINGLE_STRING: Regular single text like "Hello world"
            - InputFormat.BATCH_STRINGS: Regular batch like ["Hello", "World"]
            - InputFormat.CROSS_ENCODER_PAIRS: Cross-encoder pairs like [["query", "document"]]
        """
        if isinstance(texts, str):
            return InputFormat.SINGLE_STRING

        if (
            is_cross_encoder
            and len(texts) > 0
            and isinstance(texts[0], list)
            and len(texts[0]) == 2
        ):
            return InputFormat.CROSS_ENCODER_PAIRS

        return InputFormat.BATCH_STRINGS

    def _prepare_tokenizer_input(
        self, texts: Union[str, List[str]], input_format: InputFormat
    ) -> Union[List[str], List[List[str]]]:
        """Prepare input for the tokenizer based on detected format."""
        if input_format == InputFormat.SINGLE_STRING:
            return [texts]  # Wrap single string for batch processing
        elif input_format == InputFormat.CROSS_ENCODER_PAIRS:
            return texts  # Already in correct format: [["query", "doc"]]
        else:  # BATCH_STRINGS
            return texts  # Already in correct format: ["text1", "text2"]

    def _extract_tokenizer_results(
        self,
        input_ids: List[List[int]],
        token_type_ids: Optional[List[List[int]]],
        input_format: InputFormat,
        original_batch_size: int,
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """Extract results from tokenizer output based on input format."""

        # For single inputs (string or single cross-encoder pair), extract first element
        if (
            input_format in [InputFormat.SINGLE_STRING, InputFormat.CROSS_ENCODER_PAIRS]
            and original_batch_size == 1
        ):
            single_input_ids = input_ids[0] if input_ids else []
            single_token_type_ids = token_type_ids[0] if token_type_ids else None
            return single_input_ids, single_token_type_ids

        # For true batches, return as-is
        return input_ids, token_type_ids

    async def _tokenize_texts(
        self, texts: Union[str, List[str]], is_cross_encoder: bool = False
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """
        Tokenize text(s) using the appropriate tokenizer strategy.

        This method handles multiple input formats and chooses between async dynamic
        batch tokenizer (for single texts only) and regular tokenizer.

        Args:
            texts: Text input in various formats:

                   Regular cases:
                   - Single string: "How are you?"
                   - Batch of strings: ["Hello", "World", "How are you?"]

                   Cross-encoder cases (sentence pairs for similarity/ranking):
                   - Single pair: [["query text", "document text"]]
                   - Multiple pairs: [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]

            is_cross_encoder: Whether to return token_type_ids for cross-encoder models.
                             Enables proper handling of sentence pairs with segment IDs.

        Returns:
            Single input cases:
                Tuple[List[int], Optional[List[int]]]: (input_ids, token_type_ids)
                Example: ([101, 2129, 102], [0, 0, 0]) for single text
                Example: ([101, 2129, 102, 4068, 102], [0, 0, 0, 1, 1]) for cross-encoder pair

            Batch input cases:
                Tuple[List[List[int]], Optional[List[List[int]]]]: (batch_input_ids, batch_token_type_ids)
                Example: ([[101, 2129, 102], [101, 4068, 102]], None) for regular batch

            Note: token_type_ids is None unless is_cross_encoder=True.
        """
        if not texts or self.tokenizer is None:
            raise ValueError("texts cannot be empty and tokenizer must be initialized")

        # Step 1: Detect input format and prepare for tokenization
        input_format = self._detect_input_format(texts, is_cross_encoder)
        tokenizer_input = self._prepare_tokenizer_input(texts, input_format)
        original_batch_size = len(texts) if not isinstance(texts, str) else 1

        # Step 2: Set up tokenizer arguments
        tokenizer_kwargs = (
            {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}
        )

        # Step 3: Choose tokenization strategy
        use_async_tokenizer = (
            self.async_dynamic_batch_tokenizer is not None
            and input_format == InputFormat.SINGLE_STRING
        )

        if use_async_tokenizer:
            logger.debug("Using async dynamic batch tokenizer for single text")
            result = await self.async_dynamic_batch_tokenizer.encode(
                tokenizer_input[0], **tokenizer_kwargs
            )
            # Convert to batch format for consistency
            input_ids = [result["input_ids"]]
            token_type_ids = (
                [result["token_type_ids"]]
                if is_cross_encoder and result.get("token_type_ids")
                else None
            )
        else:
            logger.debug(f"Using regular tokenizer for {len(tokenizer_input)} inputs")
            encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
            input_ids = encoded["input_ids"]
            token_type_ids = encoded.get("token_type_ids") if is_cross_encoder else None

        # Step 4: Extract results based on input format
        return self._extract_tokenizer_results(
            input_ids, token_type_ids, input_format, original_batch_size
        )

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

            input_ids, token_type_ids = await self._tokenize_texts(
                input_text, is_cross_encoder_request
            )

        if self.mm_processor and obj.contains_mm_input():
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            mm_inputs: Dict = await self.mm_data_processor.process(
                image_data=obj.image_data,
                audio_data=obj.audio_data,
                input_text_or_ids=(input_text or input_ids),
                request_obj=obj,
                max_req_input_len=self.max_req_input_len,
            )
            if mm_inputs and "input_ids" in mm_inputs:
                input_ids = mm_inputs["input_ids"]
        else:
            mm_inputs = None

        self._validate_one_request(obj, input_ids)
        trace_slice_end(RequestStage.TOKENIZE, obj.rid)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
        )

    def _validate_one_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""
        # FIXME: unify the length validation logic with the one in the scheduler.
        _max_req_len = self.context_len

        input_token_num = len(input_ids) if input_ids is not None else 0
        input_token_num += self.reserve_input_token_num
        if input_token_num >= self.context_len:
            if self.server_args.allow_auto_truncate:
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens). "
                    "Truncating the input."
                )
                del input_ids[_max_req_len:]
                input_token_num = len(input_ids)
            else:
                raise ValueError(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens)."
                )

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= _max_req_len
        ):
            if self.server_args.allow_auto_truncate:
                logger.warning(
                    f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                    f"exceeds the model's context length ({self.context_len} tokens). "
                    "Truncating max_new_tokens."
                )
                obj.sampling_params["max_new_tokens"] = max(
                    0, _max_req_len - input_token_num
                )
            else:
                total_tokens = max_new_tokens + input_token_num
                error_msg = (
                    f"Requested token count exceeds the model's maximum context length "
                    f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                    f"tokens: {input_token_num} tokens from the input messages and "
                    f"{max_new_tokens} tokens for the completion. Please reduce the number "
                    f"of tokens in the input messages or the completion to fit within the limit."
                )
                raise ValueError(error_msg)

        # Matryoshka embeddings validations
        if isinstance(obj, EmbeddingReqInput):
            self._validate_for_matryoshka_dim(obj)

        if isinstance(obj, GenerateReqInput):
            if (
                obj.return_hidden_states
                and not self.server_args.enable_return_hidden_states
            ):
                raise ValueError(
                    "The server is not configured to return the hidden states. "
                    "Please set `--enable-return-hidden-states` to enable this feature."
                )
            if (
                obj.custom_logit_processor
                and not self.server_args.enable_custom_logit_processor
            ):
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )

    def _validate_for_matryoshka_dim(self, obj: EmbeddingReqInput) -> None:
        """Validate the request for Matryoshka dim if it has the field set."""
        if obj.dimensions is None:
            return

        if not self.model_config.is_matryoshka:
            raise ValueError(
                f"Model '{self.model_config.model_path}' does not support matryoshka representation, "
                f"changing output dimensions will lead to poor results."
            )

        if obj.dimensions < 1:
            raise ValueError("Requested dimensions must be greater than 0")

        if (
            self.model_config.matryoshka_dimensions
            and obj.dimensions not in self.model_config.matryoshka_dimensions
        ):
            raise ValueError(
                f"Model '{self.model_config.model_path}' only supports {self.model_config.matryoshka_dimensions} matryoshka dimensions, "
                f"using other output dimensions will lead to poor results."
            )

        if obj.dimensions > self.model_config.hidden_size:
            raise ValueError(
                f"Provided dimensions are greater than max embedding dimension: {self.model_config.hidden_size}"
            )

    def _validate_input_ids_in_vocab(
        self, input_ids: List[int], vocab_size: int
    ) -> None:
        if any(id >= vocab_size for id in input_ids):
            raise ValueError(
                f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
            )

    def _get_sampling_params(self, sampling_kwargs: Dict) -> SamplingParams:
        return SamplingParams(**sampling_kwargs)

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        mm_inputs: Optional[Dict] = None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""
        # Parse sampling parameters
        # Note: if there are preferred sampling params, we use them if they are not
        # explicitly passed in sampling_params
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
        else:
            sampling_kwargs = obj.sampling_params
        sampling_params = self._get_sampling_params(sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

            tokenized_obj = TokenizedGenerateReqInput(
                input_text,
                input_ids,
                mm_inputs,
                sampling_params,
                obj.return_logprob,
                obj.logprob_start_len,
                obj.top_logprobs_num,
                obj.token_ids_logprob,
                obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                data_parallel_rank=obj.data_parallel_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                input_text,
                input_ids,
                mm_inputs,
                token_type_ids,
                sampling_params,
                rid=obj.rid,
                priority=obj.priority,
                dimensions=obj.dimensions,
                http_worker_ipc=obj.http_worker_ipc,
            )

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

        self._validate_batch_tokenization_constraints(batch_size, obj)

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Check if any request is a cross-encoder request
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests
        )

        # Batch tokenize all texts using unified method
        input_ids_list, token_type_ids_list = await self._tokenize_texts(
            texts, is_cross_encoder_request
        )

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self._validate_one_request(obj[i], input_ids_list[i])
            token_type_ids = (
                token_type_ids_list[i] if token_type_ids_list is not None else None
            )
            tokenized_objs.append(
                self._create_tokenized_object(
                    req, req.text, input_ids_list[i], None, None, token_type_ids
                )
            )
            trace_slice_end(RequestStage.TOKENIZE, req.rid)
        logger.debug(f"Completed batch processing for {batch_size} requests")
        return tokenized_objs

    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        """Validate constraints for batch tokenization processing."""
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

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
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        trace_slice_start(RequestStage.TOKENIZER_DISPATCH, obj.rid)
        tokenized_obj.trace_context = trace_get_proc_propagate_context(obj.rid)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
        state.request_sent_to_scheduler_ts = time.time()
        self.rid_to_state[obj.rid] = state
        trace_slice_end(
            RequestStage.TOKENIZER_DISPATCH, obj.rid, thread_finish_flag=True
        )
        return state

    def _send_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
        created_time: Optional[float] = None,
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

        self.send_to_scheduler.send_pyobj(batch_req)
        # Create states for each individual request in the batch
        for i, tokenized_obj in enumerate(tokenized_objs):
            tmp_obj = obj[i]
            state = ReqState(
                [], False, asyncio.Event(), tmp_obj, created_time=created_time
            )
            self.rid_to_state[tmp_obj.rid] = state

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        state: ReqState,
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
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

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                # For non-streaming cases, response has not been sent yet (`response_sent_to_client_ts` has not been set yet).
                # Record response sent time right before we log finished results and metrics.
                if not state.response_sent_to_client_ts:
                    state.response_sent_to_client_ts = time.time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.response_sent_to_client_ts
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    if self.model_config.is_multimodal_gen:
                        msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                    else:
                        msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                if self.request_metrics_exporter_manager.exporter_enabled():
                    # Asynchronously write metrics for this request using the exporter manager.
                    asyncio.create_task(
                        self.request_metrics_exporter_manager.write_record(obj, out)
                    )

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        if not obj.stream:
                            raise ValueError(finish_reason["message"])
                        else:
                            yield out
                            break

                    if finish_reason.get("type") == "abort" and finish_reason.get(
                        "status_code"
                    ) in (
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ):
                        # This is an abort request initiated by scheduler.
                        # Delete the key to prevent resending abort request to the scheduler and
                        # to ensure aborted request state is cleaned up.
                        if state.obj.rid in self.rid_to_state:
                            del self.rid_to_state[state.obj.rid]

                        # Mark ongoing LoRA request as finished.
                        if self.server_args.enable_lora and state.obj.lora_path:
                            await self.lora_registry.release(state.obj.lora_id)
                        if not obj.stream:
                            raise fastapi.HTTPException(
                                status_code=finish_reason["status_code"],
                                detail=finish_reason["message"],
                            )
                        else:
                            yield out
                            break
                yield out
                break

            state.event.clear()

            if obj.stream:
                # Record response sent time right before we send response.
                if not state.response_sent_to_client_ts:
                    state.response_sent_to_client_ts = time.time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.response_sent_to_client_ts
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
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                self._send_batch_request(obj, tokenized_objs, created_time)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    generators.append(
                        self._wait_one_response(
                            tmp_obj, self.rid_to_state[tmp_obj.rid], request
                        )
                    )
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
                        state = self._send_one_request(
                            tmp_obj, tokenized_obj, created_time
                        )
                        generators.append(
                            self._wait_one_response(tmp_obj, state, request)
                        )
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
                state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, state, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)

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

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            result = await self.model_update_result
            if result.success:
                self.served_model_name = obj.model_path
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        logging.info(f"Config logging: {obj=}")
        self.log_request_metadata = self.get_log_request_metadata()

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
        if self._chosen_loop is not None:
            current_loop = get_or_create_event_loop()
            assert (
                current_loop == self._chosen_loop
            ), f"Please ensure only one event loop is ever used with SGLang. Previous loop: {self._chosen_loop}, current loop: {current_loop}"
            return

        loop = get_or_create_event_loop()
        self._chosen_loop = loop
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )

        self.event_loop = loop

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.watch_load_thread))
        )

    def dump_requests_before_crash(self):
        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return

        if not self.crash_dump_folder:
            return

        logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")
        self.crash_dump_performed = True

        # Check if NFS directory is available
        # expected_nfs_dir = "/" + self.crash_dump_folder.lstrip("/").split("/")[0]
        # use_nfs_dir = os.path.isdir(expected_nfs_dir) and os.access(
        #     expected_nfs_dir, os.W_OK
        # )
        use_nfs_dir = False
        if not use_nfs_dir:
            logger.error(
                f"Expected NFS directory is not available or writable. Uploading to GCS."
            )

        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                unfinished_requests.append(
                    (
                        state.obj,
                        state.out_list[-1] if state.out_list else {},
                        state.created_time,
                        time.time(),
                    )
                )
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return

        object_name = f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
        filename = os.path.join(
            self.crash_dump_folder,
            os.getenv("HOSTNAME", None),
            object_name,
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Include server_args in the dump
        data_to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_to_dump,
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
        )

        def _upload_file_to_gcs(bucket_name, source_file_path, object_name):
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_filename(source_file_path, if_generation_match=0)
            logger.error(
                f"Successfully uploaded {source_file_path} to gs://{bucket_name}/{object_name}"
            )

        if not use_nfs_dir:
            _upload_file_to_gcs(
                "sglang_crash_dump",
                filename,
                os.getenv("HOSTNAME", None) + "/" + object_name,
            )

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
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
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

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = time.time()

    def _add_metric_if_present(
        self,
        recv_obj: Any,
        attr_name: str,
        meta_info: Dict[str, Any],
        index: int,
    ) -> None:
        """Add a metric to meta_info if it exists and is not None.

        Args:
            recv_obj: The received object that may contain the metric attribute
            attr_name: The name of the attribute to check
            meta_info: The dictionary to add the metric to
            index: The index to access the metric value in the attribute list
        """
        if (
            hasattr(recv_obj, attr_name)
            and getattr(recv_obj, attr_name)
            and getattr(recv_obj, attr_name)[index] is not None
        ):
            meta_info[attr_name] = getattr(recv_obj, attr_name)[index]

    def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchMultimodalOutput,
            BatchTokenIDOutput,
        ],
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
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
                "total_retractions": recv_obj.retraction_counts[i],
            }

            if self.enable_metrics:
                self._add_metric_if_present(recv_obj, "queue_time", meta_info, i)
                self._add_metric_if_present(
                    recv_obj, "prefill_launch_delay", meta_info, i
                )
                self._add_metric_if_present(
                    recv_obj, "prefill_launch_latency", meta_info, i
                )

            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOutput):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOutput):
                state.text += recv_obj.output_strs[i]
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "text": state.text,
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOutput):
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchMultimodalOutput):
                raise NotImplementedError("BatchMultimodalOut not implemented")
            else:
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.server_args.speculative_algorithm:
                    self._calculate_spec_decoding_metrics(meta_info, recv_obj, i)
                state.finished_time = time.time()
                state.finished_time_perf = time.perf_counter()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

                if self.enable_metrics:
                    self._calculate_timing_metrics(meta_info, state, recv_obj, i)

                trace_req_finish(rid, ts=int(state.finished_time * 1e9))

                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.server_args.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

    def add_logprob_to_meta_info(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
    ):
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            state.input_token_logprobs_val,
            state.input_token_logprobs_idx,
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            state.output_token_logprobs_val,
            state.output_token_logprobs_idx,
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_top_logprobs_val,
                state.input_top_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_top_logprobs_val,
                state.output_top_logprobs_idx,
                return_text_in_logprobs,
            )

        if token_ids_logprob is not None:
            meta_info["input_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_token_ids_logprobs_val,
                state.input_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_token_ids_logprobs"] = (
                self.detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val,
                    state.output_token_ids_logprobs_idx,
                    return_text_in_logprobs,
                )
            )

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOutput,
        recv_obj_index: int,
    ):
        if recv_obj.input_token_logprobs_val is None:
            return

        if len(recv_obj.input_token_logprobs_val) > 0:
            state.input_token_logprobs_val.extend(
                recv_obj.input_token_logprobs_val[recv_obj_index]
            )
            state.input_token_logprobs_idx.extend(
                recv_obj.input_token_logprobs_idx[recv_obj_index]
            )
        state.output_token_logprobs_val.extend(
            recv_obj.output_token_logprobs_val[recv_obj_index]
        )
        state.output_token_logprobs_idx.extend(
            recv_obj.output_token_logprobs_idx[recv_obj_index]
        )

        if top_logprobs_num > 0:
            if len(recv_obj.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(
                    recv_obj.input_top_logprobs_val[recv_obj_index]
                )
                state.input_top_logprobs_idx.extend(
                    recv_obj.input_top_logprobs_idx[recv_obj_index]
                )
            state.output_top_logprobs_val.extend(
                recv_obj.output_top_logprobs_val[recv_obj_index]
            )
            state.output_top_logprobs_idx.extend(
                recv_obj.output_top_logprobs_idx[recv_obj_index]
            )

        if token_ids_logprob is not None:
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )

        self.add_logprob_to_meta_info(
            meta_info,
            state,
            state.obj.top_logprobs_num,
            state.obj.token_ids_logprob,
            return_text_in_logprobs,
        )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret

    def _calculate_spec_decoding_metrics(
        self,
        meta_info: Dict[str, Any],
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchMultimodalOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> None:
        """Calculate speculative decoding metrics, such as acceptance rate and acceptance length metrics."""
        meta_info["spec_accept_rate"] = 0.0
        meta_info["spec_accept_length"] = 0
        meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

        # The draft tokens per speculative step (excluding the target-sampled token).
        num_guess_tokens = self.server_args.speculative_num_draft_tokens - 1

        if (
            recv_obj.spec_verify_ct[i] > 0
            and num_guess_tokens is not None
            and not isinstance(recv_obj, BatchEmbeddingOutput)
            and hasattr(recv_obj, "spec_accepted_tokens")
            # Checks that `spec_accepted_tokens[i]` will exist.
            and len(recv_obj.spec_accepted_tokens) > i
        ):
            total_draft_tokens = recv_obj.spec_verify_ct[i] * num_guess_tokens
            accepted_tokens = recv_obj.spec_accepted_tokens[i]

            # Calculate per-request acceptance rate and average acceptance length.
            if total_draft_tokens > 0:
                # Calculate acceptance rate: accepted / (steps * lookahead)
                meta_info["spec_accept_rate"] = accepted_tokens / total_draft_tokens
                meta_info["spec_accept_length"] = (
                    recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]
                )
                meta_info["spec_accept_token_num"] = accepted_tokens
                meta_info["spec_draft_token_num"] = total_draft_tokens
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

    def _calculate_timing_metrics(
        self,
        meta_info: Dict[str, Any],
        state: ReqState,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchMultimodalOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> None:
        """Calculate request-level timing metrics, such as inference time, decode throughput, and time per token."""
        # Request timing timestamps.
        if state.created_time > 0:
            meta_info["request_received_ts"] = state.created_time
        if state.request_sent_to_scheduler_ts > 0:
            meta_info["request_sent_to_scheduler_ts"] = (
                state.request_sent_to_scheduler_ts
            )
        # For embeddings, there's no separate prefill phase, so omit `prefill_finished_ts`.
        if (
            not isinstance(recv_obj, BatchEmbeddingOutput)
            and state.first_token_time > 0
        ):
            meta_info["prefill_finished_ts"] = state.first_token_time
        if state.response_sent_to_client_ts > 0:
            meta_info["response_sent_to_client_ts"] = state.response_sent_to_client_ts
        if state.finished_time > 0:
            meta_info["decode_finished_ts"] = state.finished_time

        # Inference time calculation.
        if (
            hasattr(recv_obj, "forward_entry_time")
            and recv_obj.forward_entry_time
            and recv_obj.forward_entry_time[i] is not None
            and state.finished_time_perf > 0.0
        ):
            inference_time = state.finished_time_perf - recv_obj.forward_entry_time[i]
            meta_info["inference_time"] = inference_time

        # Decode throughput, time per token calculation. Only calculated if TTFT is available.
        if (
            state.first_token_time_perf > 0.0
            and state.finished_time_perf > 0.0
            and not isinstance(recv_obj, BatchEmbeddingOutput)
            and recv_obj.completion_tokens[i] > 0
        ):
            decode_time = state.finished_time_perf - state.first_token_time_perf
            completion_tokens = recv_obj.completion_tokens[i]
            meta_info["decode_throughput"] = completion_tokens / decode_time

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOutput, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        custom_labels = getattr(state.obj, "custom_labels", None)
        labels = (
            {**self.metrics_collector.labels, **custom_labels}
            if custom_labels
            else self.metrics_collector.labels
        )
        if (
            state.first_token_time == 0.0
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.first_token_time = state.last_time = time.time()
            state.first_token_time_perf = time.perf_counter()
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels, state.first_token_time - state.created_time
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                new_time = time.time()
                interval = new_time - state.last_time
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    interval,
                    num_new_tokens,
                )
                state.last_time = new_time
                state.last_completion_tokens = completion_tokens

        if state.finished:
            has_grammar = (
                state.obj.sampling_params.get("json_schema", None)
                or state.obj.sampling_params.get("regex", None)
                or state.obj.sampling_params.get("ebnf", None)
                or state.obj.sampling_params.get("structural_tag", None)
            )

            retraction_count = (
                recv_obj.retraction_counts[i]
                if getattr(recv_obj, "retraction_counts", None)
                and i < len(recv_obj.retraction_counts)
                else 0
            )

            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.finished_time - state.created_time,
                has_grammar,
                retraction_count,
            )

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append(
            (state.obj, out_dict, state.created_time, time.time())
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
        current_time = time.time()
        self.crash_dump_request_list.append(
            (state.obj, out_dict, state.created_time, current_time)
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
                pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def _handle_abort_req(self, recv_obj: AbortReq):
        if is_health_check_generate_req(recv_obj):
            return
        state = self.rid_to_state[recv_obj.rid]
        state.finished = True

        abort_message = recv_obj.abort_message or "Abort in waiting queue"
        finish_reason = {
            "type": "abort",
            "message": abort_message,
        }
        if recv_obj.finished_reason:
            finish_reason = recv_obj.finished_reason
        meta_info = {"id": recv_obj.rid, "finish_reason": finish_reason}
        is_stream = getattr(state.obj, "stream", False)
        if getattr(state.obj, "return_logprob", False):
            self.add_logprob_to_meta_info(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs
                and not self.server_args.skip_tokenizer_init,
            )

        output_ids = state.output_ids
        meta_info["completion_tokens"] = len(output_ids)
        if is_stream:
            output_ids = [output_ids[-1]] if len(output_ids) > 0 else []
        out = {
            "text": state.text,
            "output_ids": output_ids,
            "meta_info": meta_info,
        }
        state.out_list.append(out)
        state.event.set()

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are received
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)

    def _extract_logprobs_for_tokens(
        self, logprobs_data: List, label_token_ids: List[int]
    ) -> Dict[int, float]:
        """
        Extract logprobs for specified token IDs from logprobs data.

        Args:
            logprobs_data: List of (logprob, token_id, text) tuples
            label_token_ids: Token IDs to extract logprobs for

        Returns:
            Dictionary mapping token_id to logprob
        """
        logprobs = {}
        if logprobs_data:
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
        return logprobs

    async def watch_load_thread(self):
        # Only for dp_controller when dp_size > 1
        if (
            self.server_args.dp_size == 1
            or self.server_args.load_balance_method == "round_robin"
        ):
            return

        while True:
            await asyncio.sleep(self.server_args.load_watch_interval)
            loads = await self.get_load_communicator(GetLoadReqInput())
            load_udpate_req = WatchLoadUpdateReq(loads=loads)
            self.send_to_scheduler.send_pyobj(load_udpate_req)

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

    def _log_received_request(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        max_length, skip_names, _ = self.log_request_metadata
        logger.info(
            f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
        )

        # FIXME: This is a temporary fix to get the text from the input ids.
        # We should remove this once we have a proper way.
        if (
            self.log_requests_level >= 2
            and obj.text is None
            and obj.input_ids is not None
            and self.tokenizer is not None
        ):
            decoded = self.tokenizer.decode(obj.input_ids, skip_special_tokens=False)
            obj.text = decoded

    def _trace_request_start(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        created_time: Optional[float] = None,
        request: Optional[fastapi.Request] = None,
    ):
        external_trace_header = None
        if request:
            if "trace_context" in request.headers:
                trace_set_remote_propagate_context(request.headers["trace_context"])
            else:
                external_trace_header = extract_trace_headers(request.headers)

        if obj.is_single:
            bootstrap_room = (
                obj.bootstrap_room if hasattr(obj, "bootstrap_room") else None
            )
            trace_req_start(
                obj.rid,
                bootstrap_room,
                ts=int(created_time * 1e9),
                role=self.server_args.disaggregation_mode,
                external_trace_header=external_trace_header,
            )
            trace_slice_start("", obj.rid, ts=int(created_time * 1e9), anonymous=True)
        else:
            for i in range(len(obj.rid)):
                bootstrap_room = (
                    obj.bootstrap_room[i]
                    if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                    else None
                )
                trace_req_start(
                    obj.rid[i],
                    bootstrap_room,
                    ts=int(created_time * 1e9),
                    role=self.server_args.disaggregation_mode,
                    external_trace_header=external_trace_header,
                )
                trace_slice_start(
                    "", obj.rid[i], ts=int(created_time * 1e9), anonymous=True
                )


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


def _get_processor_wrapper(server_args):
    try:
        processor = get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=not server_args.disable_fast_image_processor,
        )
    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
            )
            processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=True,
            )
        else:
            raise e
    return processor


def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


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
