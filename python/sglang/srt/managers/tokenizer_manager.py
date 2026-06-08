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

from __future__ import annotations

import asyncio
import copy
import logging
import os
import signal
import sys
import threading
from contextlib import nullcontext
from enum import Enum
from http import HTTPStatus
from typing import Dict, Iterable, List, Optional, Union

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
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqOutput,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.mm_utils import wrap_shm_features
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager_components import (
    logprob_ops,
    request_tracing,
    spec_decoding_meta,
)
from sglang.srt.managers.tokenizer_manager_components.lora_controller import (
    LoraController,
)
from sglang.srt.managers.tokenizer_manager_components.multimodal_processor import (
    MultimodalProcessor,
)
from sglang.srt.managers.tokenizer_manager_components.raw_tokenizer_wrapper import (
    RawTokenizerWrapper,
)
from sglang.srt.managers.tokenizer_manager_components.request_log_manager import (
    RequestLogManager,
)
from sglang.srt.managers.tokenizer_manager_components.request_metrics_recorder import (
    RequestMetricsRecorder,
)
from sglang.srt.managers.tokenizer_manager_components.request_preparer import (
    RequestPreparer,
    RequestPreparerConfig,
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
from sglang.srt.managers.tokenizer_manager_components.session_controller import (
    SessionController,
)
from sglang.srt.managers.tokenizer_manager_components.tokenized_request_builder import (
    TokenizedRequestBuilder,
    TokenizedRequestBuilderConfig,
)
from sglang.srt.managers.tokenizer_manager_components.weight_updater_controller import (
    WeightUpdaterController,
)
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.srt.observability.req_time_stats import (
    real_time,
    set_time_batch,
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
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

logger = logging.getLogger(__name__)


class TokenizerManager(TokenizerControlMixin):
    """TokenizerManager is a process that tokenizes the text."""

    @property
    def serving_chat_class(self):
        """Return the serving chat class for OpenAI API.

        Override in subclass to provide custom serving behavior.
        """
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        return OpenAIServingChat

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

        # Initialize tokenizer and multimodalprocessor
        self.init_raw_tokenizer_wrapper()

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init running status
        self.init_running_status()

        # Init logging and dumping
        self.init_request_log_manager()

        # Init weight update
        self.init_weight_update()

        self.init_weight_updater_controller()

        # Init LoRA status
        self.init_lora_controller()

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()

        self.init_multimodal_processor()

        self.init_tokenized_request_builder()

        self.init_request_metrics_recorder()

        self.init_session_controller()

        self.init_request_validator()

        self.init_request_preparer()

        self.init_score_request_handler()

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
                server_args.max_speculative_num_draft_tokens,
            )
        else:
            self.num_reserved_tokens = 0
        self.validate_total_tokens = True

    def init_raw_tokenizer_wrapper(self):
        self.raw_tokenizer_wrapper = RawTokenizerWrapper()
        self.raw_tokenizer_wrapper.init_tokenizer_and_processor(
            server_args=self.server_args,
            model_config=self.model_config,
        )

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

        self.load_snapshot_reader = create_load_snapshot_reader(
            self.server_args,
            port_args,
            caller="TokenizerManager",
        )

    def init_running_status(self):
        # Request states
        self.rid_to_state: Dict[str, ReqState] = {}
        self.event_loop = None
        self.asyncio_tasks = set()

        # Health check
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = real_time()

        # Subprocess liveness watchdog — set by Engine or http_server after construction
        self._subprocess_watchdog = None

    def init_request_log_manager(self):
        self.request_log_manager = RequestLogManager.from_server_args(
            server_args=self.server_args,
        )

    def init_weight_update(self):
        # Weight updates
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

    def init_weight_updater_controller(self):
        self.weight_updater_controller = WeightUpdaterController(
            send_to_scheduler=self.send_to_scheduler,
            abort_request=self.abort_request,
            update_model_path_info=lambda model_path, load_format: self._update_model_path_info(
                model_path, load_format
            ),
            is_pause_getter=lambda: self.is_pause,
            is_pause_cond=self.is_pause_cond,
            model_update_lock=self.model_update_lock,
            server_args=self.server_args,
            auto_create_handle_loop=self.auto_create_handle_loop,
        )

    def init_lora_controller(self):
        self.lora_controller = LoraController(
            server_args=self.server_args,
            auto_create_handle_loop=self.auto_create_handle_loop,
        )

    def init_disaggregation(self):
        # PD Disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        # Keep a reference so the bootstrap server is not garbage-collected.
        self.bootstrap_server = start_disagg_service(self.server_args)

    def init_metric_collector_watchdog(self):
        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",
            watchdog_timeout=self.server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),
        )

    def init_multimodal_processor(self):
        self.multimodal_processor = MultimodalProcessor.from_server_args(
            server_args=self.server_args,
            model_config=self.model_config,
            mm_processor=self.mm_processor,
        )

    def init_tokenized_request_builder(self):
        self.tokenized_request_builder = TokenizedRequestBuilder(
            tokenizer=self.tokenizer,
            config=TokenizedRequestBuilderConfig(
                vocab_size=self.model_config.vocab_size,
                preferred_sampling_params=self.preferred_sampling_params,
                sampling_params_class=SamplingParams,
                disaggregation_transfer_backend=self.server_args.disaggregation_transfer_backend,
            ),
        )

    def init_request_metrics_recorder(self):
        self.request_metrics_recorder = RequestMetricsRecorder(
            server_args=self.server_args,
            enable_metrics=self.enable_metrics,
            enable_priority_scheduling=self.enable_priority_scheduling,
            get_disaggregation_mode=lambda: self.disaggregation_mode,
        )

    def init_session_controller(self):
        self.session_controller = SessionController(
            send_to_scheduler=self.send_to_scheduler,
            auto_create_handle_loop=self.auto_create_handle_loop,
            server_args=self.server_args,
        )

    def init_request_validator(self):
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

    def init_request_preparer(self):
        self.request_preparer = RequestPreparer(
            raw_tokenizer_wrapper=self.raw_tokenizer_wrapper,
            multimodal_processor=self.multimodal_processor,
            request_validator=self.request_validator,
            tokenized_request_builder=self.tokenized_request_builder,
            rid_to_state=self.rid_to_state,
            get_max_req_input_len=lambda: self.max_req_input_len,
            config=RequestPreparerConfig(
                skip_tokenizer_init=self.server_args.skip_tokenizer_init,
                enable_dp_attention=self.server_args.enable_dp_attention,
                enable_tokenizer_batch_encode=self.server_args.enable_tokenizer_batch_encode,
                is_generation=self.is_generation,
                disable_radix_cache=self.server_args.disable_radix_cache,
                is_multimodal=self.model_config.is_multimodal,
                architectures=self.model_config.hf_config.architectures,
                language_only=self.server_args.language_only,
                encoder_transfer_backend=self.server_args.encoder_transfer_backend,
            ),
        )

    def init_score_request_handler(self):
        self.score_request_handler = ScoreRequestHandler(
            tokenizer=self.tokenizer,
            rid_to_state=self.rid_to_state,
            generate_request=lambda obj, request=None: self.generate_request(
                obj, request
            ),
            config=ScoreRequestHandlerConfig(
                is_generation=self.is_generation,
                enable_mis=self.server_args.enable_mis,
                model_config=self.model_config,
            ),
        )

    def init_request_dispatcher(self):
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (AbortReq, self._handle_abort_req),
                (
                    OpenSessionReqOutput,
                    self.session_controller.handle_open_session_req_output,
                ),
                (
                    UpdateWeightFromDiskReqOutput,
                    self.weight_updater_controller.handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                (HealthCheckOutput, lambda x: None),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self.init_communicators(self.server_args)
        self.lora_controller.update_lora_adapter_communicator = (
            self.update_lora_adapter_communicator
        )
        self.weight_updater_controller.init_weights_update_group_communicator = (
            self.init_weights_update_group_communicator
        )
        self.weight_updater_controller.destroy_weights_update_group_communicator = (
            self.destroy_weights_update_group_communicator
        )
        self.weight_updater_controller.update_weights_from_distributed_communicator = (
            self.update_weights_from_distributed_communicator
        )
        self.weight_updater_controller.update_weights_from_tensor_communicator = (
            self.update_weights_from_tensor_communicator
        )
        self.weight_updater_controller.update_weights_from_ipc_communicator = (
            self.update_weights_from_ipc_communicator
        )
        self.weight_updater_controller.get_weights_by_name_communicator = (
            self.get_weights_by_name_communicator
        )
        self.weight_updater_controller.release_memory_occupation_communicator = (
            self.release_memory_occupation_communicator
        )
        self.weight_updater_controller.resume_memory_occupation_communicator = (
            self.resume_memory_occupation_communicator
        )
        self.weight_updater_controller.check_weights_communicator = (
            self.check_weights_communicator
        )

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
                logger.debug(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        if self.server_args.tokenizer_worker_num > 1:
            self._attach_multi_http_worker_info(obj)
        init_req(
            self.rid_to_state,
            obj=obj,
            request=request,
            enable_trace=self.server_args.enable_trace,
            disagg_mode=self.disaggregation_mode,
        )
        if self.server_args.language_only:
            self.multimodal_processor.maybe_dispatch_to_encoder(obj)

        # Log the request
        self.request_log_manager.request_logger.log_received_request(
            obj, self.tokenizer, request
        )

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        async with self.model_update_lock.reader_lock:
            await self.lora_controller._validate_and_resolve_lora(obj)

            # Tokenize the request and send it to the scheduler
            if obj.is_single:
                tokenized_obj = await self.request_preparer._tokenize_one_request(obj)
                state = self.rid_to_state[obj.rid]
                if obj.return_prompt_token_ids:
                    state.prompt_token_ids = list(tokenized_obj.input_ids)
                self._send_one_request(tokenized_obj)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, request):
                    yield response

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
        customized_info_keys: Optional[Iterable[str]] = None,
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
            incremental_streaming_keys = set(
                logprob_ops.INCREMENTAL_STREAMING_META_INFO_KEYS
            )
            if customized_info_keys is not None:
                incremental_streaming_keys.update(customized_info_keys)
            for key in incremental_streaming_keys:
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
                await self.lora_controller.lora_registry.release(state.obj.lora_id)
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
                out = self._coalesce_streaming_chunks(
                    out_list,
                    obj.rid,
                    state.customized_info_accumulated.keys(),
                )
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
                self.request_log_manager.request_logger.log_finished_request(
                    obj,
                    out,
                    request=request,
                )

                if (
                    self.request_log_manager.request_metrics_exporter_manager.exporter_enabled()
                ):
                    asyncio.create_task(
                        self.request_log_manager.request_metrics_exporter_manager.write_record(
                            obj, out
                        )
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
            if self.request_preparer._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = (
                    await self.request_preparer._batch_tokenize_and_process(
                        batch_size, obj
                    )
                )
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    state = self.rid_to_state[tmp_obj.rid]
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
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
                        tokenized_obj = (
                            await self.request_preparer._tokenize_one_request(tmp_obj)
                        )
                        state = self.rid_to_state[tmp_obj.rid]
                        if tmp_obj.return_prompt_token_ids:
                            state.prompt_token_ids = list(tokenized_obj.input_ids)
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
                *(self.request_preparer._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                # Ensure independent mm_items so wrap_shm_features won't mutate the original
                if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                    tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                    tokenized_obj.mm_inputs.mm_items = [
                        copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                    ]
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
                    # Ensure independent mm_items so wrap_shm_features won't mutate the original
                    if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                        tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                        tokenized_obj.mm_inputs.mm_items = [
                            copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                        ]
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    init_req(
                        self.rid_to_state,
                        obj=tmp_obj,
                        enable_trace=self.server_args.enable_trace,
                        disagg_mode=self.disaggregation_mode,
                    )
                    state = self.rid_to_state[tmp_obj.rid]
                    tokenized_obj.time_stats = state.time_stats
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
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
        # Empty rid would startswith-match every request on the scheduler.
        if not abort_all and not rid:
            logger.warning("Ignore abort_request with empty rid and abort_all=False")
            return
        if (
            not abort_all
            and self.server_args.tokenizer_worker_num == 1
            and rid not in self.rid_to_state
        ):
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.request_metrics_recorder.metrics_collector.observe_one_aborted_request(
                self.request_metrics_recorder.metrics_collector.labels
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

    def _update_model_path_info(self, model_path: str, load_format: str):
        self.served_model_name = model_path
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.model_path = model_path

    def configure_logging(self, obj: ConfigureLoggingReq):
        self.request_log_manager.request_logger.configure(
            log_requests=obj.log_requests,
            log_requests_level=obj.log_requests_level,
            log_requests_format=obj.log_requests_format,
        )
        if obj.dump_requests_folder is not None:
            self.request_log_manager.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.request_log_manager.dump_requests_threshold = (
                obj.dump_requests_threshold
            )
        if obj.dump_requests_exclude_meta_keys is not None:
            self.request_log_manager.dump_requests_exclude_meta_keys = list(
                obj.dump_requests_exclude_meta_keys
            )
        if obj.crash_dump_folder is not None:
            self.request_log_manager.crash_dump_folder = obj.crash_dump_folder
        if obj.log_level is not None:
            # setLevel() may raise exception if obj.log_level is illegal string.
            # Let the exception propagate to the caller.
            # Only legal requests will be sent to scheduler.
            logging.getLogger().setLevel(obj.log_level.upper())
            self.send_to_scheduler.send_pyobj(obj)
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

            # Surface scheduler load info on each response so clients can do
            # response-based flow control without polling /v1/loads. The
            # scheduler already piggy-backs the per-DP-rank load on
            # BatchStrOutput / BatchTokenIDOutput via the ``load`` field.
            load = getattr(recv_obj, "load", None)
            if load is not None:
                num_running_reqs = getattr(load, "num_running_reqs", None)
                num_waiting_reqs = getattr(load, "num_waiting_reqs", None)
                if num_running_reqs is not None:
                    meta_info["num_running_reqs"] = num_running_reqs
                if num_waiting_reqs is not None:
                    meta_info["num_waiting_reqs"] = num_waiting_reqs

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
                if recv_obj.customized_info is not None:
                    for k, v in recv_obj.customized_info.items():
                        if k not in state.customized_info_accumulated:
                            state.customized_info_accumulated[k] = []
                        state.customized_info_accumulated[k].extend(v[i])
                        meta_info[k] = state.customized_info_accumulated[k]

            if getattr(recv_obj, "output_hidden_states", None):
                hidden_states = recv_obj.output_hidden_states[i]
                if hidden_states is not None:
                    meta_info["hidden_states"] = hidden_states
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
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.append_text(delta_text)
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
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
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
            elif isinstance(recv_obj, BatchTokenIDOutput):
                is_stream = getattr(state.obj, "stream", False)
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream
                )
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
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
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
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
                            enable_trace=self.server_args.enable_trace,
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
                    asyncio.create_task(
                        self.lora_controller.lora_registry.release(state.obj.lora_id)
                    )

            if out_dict is not None:
                state.out_list.append(out_dict)
                pending_notify[rid] = state

                if len(pending_notify) >= batch_notify_size:
                    for s in pending_notify.values():
                        s.event.set()
                    pending_notify = {}
                    await asyncio.sleep(0)

            if self.enable_metrics and state.obj.log_metrics:
                self.request_metrics_recorder.collect_metrics(state, recv_obj, i)
            if (
                self.request_log_manager.dump_requests_folder
                and state.finished
                and state.obj.log_metrics
            ):
                self.request_log_manager.dump_requests(state, out_dict)
            if (
                self.request_log_manager.crash_dump_folder
                and state.finished
                and state.obj.log_metrics
            ):
                self.request_log_manager.record_request_for_crash_dump(state, out_dict)

        # handle_loop awaits next recv immediately
        for s in pending_notify.values():
            s.event.set()

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
                self.request_log_manager.dump_requests_before_crash(
                    rid_to_state=self.rid_to_state,
                )
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
        del self.rid_to_state[recv_obj.rid]

        state.out_list.append(out)
        state.event.set()

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.send_to_scheduler.send_pyobj(ranks)

    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value."""
        if (
            self.enable_priority_scheduling
            and obj.priority is None
            and self.default_priority_value is not None
        ):
            obj.priority = self.default_priority_value

    @property
    def tokenizer(self):
        return self.raw_tokenizer_wrapper.tokenizer

    @property
    def processor(self):
        return self.raw_tokenizer_wrapper.processor

    @property
    def mm_processor(self):
        return self.raw_tokenizer_wrapper.mm_processor

    @property
    def async_dynamic_batch_tokenizer(self):
        return self.raw_tokenizer_wrapper.async_dynamic_batch_tokenizer


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
            func.__self__.request_log_manager.dump_requests_before_crash(
                rid_to_state=func.__self__.rid_to_state
            )
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
        self.tokenizer_manager.request_log_manager.dump_requests_before_crash(
            rid_to_state=self.tokenizer_manager.rid_to_state
        )
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
