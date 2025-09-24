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
import json
import logging
import math
import os
import pickle
import signal
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

import fastapi
import torch
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.aio_rwlock import RWLock
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    GetLoadReqInput,
    HealthCheckOutput,
    MultiTokenizerWrapper,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.mm_utils import TensorTransportMode
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.scheduler import is_health_check_generate_req
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_communicator_mixin import TokenizerCommunicatorMixin
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.tracing.trace import (
    trace_get_proc_propagate_context,
    trace_req_finish,
    trace_req_start,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import (
    configure_gc_warning,
    dataclass_to_string_truncated,
    freeze_gc,
    get_bool_env_var,
    get_origin_rid,
    get_zmq_socket,
    kill_process_tree,
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


class TokenizerManager(TokenizerCommunicatorMixin):
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
            json.loads(server_args.preferred_sampling_params)
            if server_args.preferred_sampling_params
            else None
        )
        self.crash_dump_folder = server_args.crash_dump_folder

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

        if self.model_config.is_multimodal:
            import_processors("sglang.srt.multimodal.processors")
            try:
                _processor = get_processor(
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
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=True,
                    )
                else:
                    raise e
            transport_mode = _determine_tensor_transport_mode(self.server_args)

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config, server_args, _processor, transport_mode
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        if self.server_args.tokenizer_worker_num > 1:
            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )
        else:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )

        # Request states
        self.no_create_loop = False
        self.rid_to_state: Dict[str, ReqState] = {}
        self.asyncio_tasks = set()

        # Health check
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = 0

        # Dumping
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: List[Tuple] = []
        self.log_request_metadata = self.get_log_request_metadata()
        self.crash_dump_request_list: deque[Tuple] = deque()
        self.crash_dump_performed = False  # Flag to ensure dump is only called once

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
            if server_args.tokenizer_metrics_allowed_customer_labels:
                for label in server_args.tokenizer_metrics_allowed_customer_labels:
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

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                        BatchMultimodalOut,
                    ),
                    self._handle_batch_output,
                ),
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (
                    FreezeGCReq,
                    lambda x: None,
                ),  # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                (HealthCheckOutput, lambda x: None),
            ]
        )

        self.init_communicators(server_args)

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()
        self.auto_create_handle_loop()
        obj.normalize_batch_and_arguments()

        if self.server_args.tokenizer_worker_num > 1:
            # Modify rid, add worker_id
            if isinstance(obj.rid, list):
                # If it's an array, add worker_id prefix to each element
                obj.rid = [f"{self.worker_id}_{rid}" for rid in obj.rid]
            else:
                # If it's a single value, add worker_id prefix
                obj.rid = f"{self.worker_id}_{obj.rid}"

        if obj.is_single:
            bootstrap_room = (
                obj.bootstrap_room if hasattr(obj, "bootstrap_room") else None
            )
            trace_req_start(obj.rid, bootstrap_room, ts=int(created_time * 1e9))
            trace_slice_start("", obj.rid, ts=int(created_time * 1e9), anonymous=True)
        else:
            for i in range(len(obj.rid)):
                bootstrap_room = (
                    obj.bootstrap_room[i]
                    if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                    else None
                )
                trace_req_start(obj.rid[i], bootstrap_room, ts=int(created_time * 1e9))
                trace_slice_start(
                    "", obj.rid[i], ts=int(created_time * 1e9), anonymous=True
                )

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
            )

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            if self.server_args.enable_lora and obj.lora_path:
                # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
                obj.lora_id = await self.lora_registry.acquire(obj.lora_path)

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
    ) -> str:
        """Detect the format of input texts for proper tokenization handling.

        Returns:
            - "single_string": Regular single text like "Hello world"
            - "batch_strings": Regular batch like ["Hello", "World"]
            - "cross_encoder_pairs": Cross-encoder pairs like [["query", "document"]]
        """
        if isinstance(texts, str):
            return "single_string"

        if (
            is_cross_encoder
            and len(texts) > 0
            and isinstance(texts[0], list)
            and len(texts[0]) == 2
        ):
            return "cross_encoder_pairs"

        return "batch_strings"

    def _prepare_tokenizer_input(
        self, texts: Union[str, List[str]], input_format: str
    ) -> Union[List[str], List[List[str]]]:
        """Prepare input for the tokenizer based on detected format."""
        if input_format == "single_string":
            return [texts]  # Wrap single string for batch processing
        elif input_format == "cross_encoder_pairs":
            return texts  # Already in correct format: [["query", "doc"]]
        else:  # batch_strings
            return texts  # Already in correct format: ["text1", "text2"]

    def _extract_tokenizer_results(
        self,
        input_ids: List[List[int]],
        token_type_ids: Optional[List[List[int]]],
        input_format: str,
        original_batch_size: int,
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """Extract results from tokenizer output based on input format."""

        # For single inputs (string or single cross-encoder pair), extract first element
        if (
            input_format in ["single_string", "cross_encoder_pairs"]
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
            and input_format == "single_string"
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
            if not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            mm_inputs: Dict = await self.mm_processor.process_mm_data_async(
                image_data=obj.image_data,
                audio_data=obj.audio_data,
                input_text=input_text or input_ids,
                request_obj=obj,
                max_req_input_len=self.max_req_input_len,
            )
            if mm_inputs and "input_ids" in mm_inputs:
                input_ids = mm_inputs["input_ids"]
        else:
            mm_inputs = None

        self._validate_one_request(obj, input_ids)
        trace_slice_end("tokenize", obj.rid)
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

    def _validate_input_ids_in_vocab(
        self, input_ids: List[int], vocab_size: int
    ) -> None:
        if any(id >= vocab_size for id in input_ids):
            raise ValueError(
                f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
            )

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
        sampling_params = SamplingParams(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                mm_inputs,
                sampling_params,
                obj.return_logprob,
                obj.logprob_start_len,
                obj.top_logprobs_num,
                obj.token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
                data_parallel_rank=obj.data_parallel_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                mm_inputs,
                token_type_ids,
                sampling_params,
                priority=obj.priority,
            )

        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only."""
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

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
            trace_slice_end("tokenize", req.rid)
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

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        trace_slice_start("dispatch", obj.rid)
        tokenized_obj.trace_context = trace_get_proc_propagate_context(obj.rid)
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
        self.rid_to_state[obj.rid] = state
        trace_slice_end("dispatch", obj.rid, thread_finish_flag=True)
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
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    if self.model_config.is_multimodal_gen:
                        msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                    else:
                        msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

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

                        raise fastapi.HTTPException(
                            status_code=finish_reason["status_code"],
                            detail=finish_reason["message"],
                        )
                yield out
                break

            state.event.clear()

            if obj.stream:
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
            if self.server_args.enable_tokenizer_batch_encode:
                # Validate batch tokenization constraints
                self._validate_batch_tokenization_constraints(batch_size, obj)

                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)

                # Send as a single batched request
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
        req = AbortReq(rid, abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            self.metrics_collector.observe_one_aborted_request()

    async def pause_generation(self):
        async with self.is_pause_cond:
            self.is_pause = True
            self.abort_request(abort_all=True)

    async def continue_generation(self):
        async with self.is_pause_cond:
            self.is_pause = False
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

        if True:  # Keep this redundant check to simplify some internal code sync
            # Hold the lock if it is not async. This means that weight sync
            # cannot run while requests are in progress.
            async with self.model_update_lock.writer_lock:
                return await self._wait_for_model_update_from_disk(obj)

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        if self.server_args.tokenizer_worker_num > 1:
            obj = MultiTokenizerWrapper(self.worker_id, obj)
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

    async def open_session(
        self, obj: OpenSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        if self.server_args.tokenizer_worker_num > 1:
            obj = MultiTokenizerWrapper(self.worker_id, obj)
        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    def get_log_request_metadata(self):
        max_length = None
        skip_names = None
        out_skip_names = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                        "sampling_params",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )
        return max_length, skip_names, out_skip_names

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
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = asyncio.get_event_loop()
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

            if self.server_status == ServerStatus.UnHealthy:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                self.dump_requests_before_crash()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                break

            logger.info(
                f"Gracefully exiting... remaining number of requests {remain_num_req}"
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

    def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOut, BatchEmbeddingOut, BatchMultimodalOut, BatchTokenIDOut
        ],
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            origin_rid = rid
            if self.server_args.tokenizer_worker_num > 1:
                origin_rid = get_origin_rid(rid)
            # Build meta_info and return value
            meta_info = {
                "id": origin_rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
            }

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

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                state.text += recv_obj.output_strs[i]
                if state.obj.stream:
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
            elif isinstance(recv_obj, BatchTokenIDOut):
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
            elif isinstance(recv_obj, BatchMultimodalOut):
                raise NotImplementedError("BatchMultimodalOut not implemented")
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.server_args.speculative_algorithm:
                    meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

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

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
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

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOut, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        customer_labels = getattr(state.obj, "customer_labels", None)
        labels = (
            {**self.metrics_collector.labels, **customer_labels}
            if customer_labels
            else self.metrics_collector.labels
        )
        if (
            state.first_token_time == 0.0
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.first_token_time = state.last_time = time.time()
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
            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.finished_time - state.created_time,
                has_grammar,
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

    def _handle_abort_req(self, recv_obj):
        if is_health_check_generate_req(recv_obj):
            return
        state = self.rid_to_state[recv_obj.rid]
        origin_rid = recv_obj.rid
        if self.server_args.tokenizer_worker_num > 1:
            origin_rid = get_origin_rid(origin_rid)
        state.finished = True
        if recv_obj.finished_reason:
            out = {
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": recv_obj.finished_reason,
                },
            }
        else:
            out = {
                "text": "",
                "meta_info": {
                    "id": origin_rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": "Abort before prefill",
                    },
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
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

    async def score_request(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        request: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        See Engine.score() for more details.
        """
        if label_token_ids is None:
            raise ValueError("label_token_ids must be provided")

        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
            for token_id in label_token_ids:
                if token_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"
                    )

        batch_request = GenerateReqInput(
            token_ids_logprob=label_token_ids,
            return_logprob=True,
            stream=False,
            sampling_params={"max_new_tokens": 0},
        )

        # Handle string or tokenized query/items
        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items
            if item_first:
                prompts = [f"{item}{query}" for item in items_list]
            else:
                prompts = [f"{query}{item}" for item in items_list]

            batch_request.text = prompts

        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            # Both query and items are token IDs
            if item_first:
                input_ids_list = [item + query for item in items]
            else:
                input_ids_list = [query + item for item in items]

            batch_request.input_ids = input_ids_list
        else:
            raise ValueError(
                "Invalid combination of query/items types for score_request."
            )

        results = await self.generate_request(batch_request, request).__anext__()
        scores = []

        for result in results:
            # Get logprobs for each token
            logprobs = {}

            # For scoring requests, we read from output_token_ids_logprobs since we want
            # the logprobs for specific tokens mentioned in the label_token_ids at
            # the next position after the last token in the prompt
            output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])

            # Check if output_logprobs is properly populated
            if (
                output_logprobs is None
                or not output_logprobs
                or len(output_logprobs) == 0
            ):
                raise RuntimeError(
                    f"output_logprobs is empty for request {result['meta_info'].get('id', '<unknown>')}. "
                    "This indicates token_ids_logprobs were not computed properly for the scoring request."
                )

            for logprob, token_id, _ in output_logprobs[0]:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob

            # Get scores in order of label_token_ids
            score_list = [
                logprobs.get(token_id, float("-inf")) for token_id in label_token_ids
            ]

            # Apply softmax to logprobs if needed
            if apply_softmax:
                score_list = torch.softmax(torch.tensor(score_list), dim=0).tolist()
            else:
                # Convert logprobs to probabilities if not using softmax
                score_list = [
                    math.exp(x) if x != float("-inf") else 0.0 for x in score_list
                ]

            scores.append(score_list)

        return scores

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


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"


def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


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
            "Received sigquit from a child process. It usually means the child failed."
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
