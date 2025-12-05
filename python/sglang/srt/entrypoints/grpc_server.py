"""
Standalone gRPC Server for SGLang - Fully separated from HTTP server.
Uses GrpcRequestManager for orchestration without tokenization.
"""

import asyncio
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from concurrent import futures
from typing import AsyncIterator, Dict, Optional

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection

import sglang
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.grpc import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from sglang.srt.grpc.grpc_request_manager import GrpcRequestManager
from sglang.srt.grpc.health_servicer import SGLangHealthServicer
from sglang.srt.grpc.scheduler_launcher import launch_scheduler_process_only
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams as SGLSamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


class SGLangSchedulerServicer(sglang_scheduler_pb2_grpc.SglangSchedulerServicer):
    """
    Standalone gRPC service implementation using GrpcRequestManager.
    Fully separated from HTTP server with its own process and no shared globals.
    """

    def __init__(
        self,
        request_manager: GrpcRequestManager,
        server_args: ServerArgs,
        model_info: Dict,
        scheduler_info: Dict,
        health_servicer: Optional[SGLangHealthServicer] = None,
    ):
        """Initialize the standalone gRPC service."""
        self.request_manager = request_manager
        self.server_args = server_args
        self.model_info = model_info
        self.scheduler_info = scheduler_info
        self.start_time = time.time()
        self.health_servicer = health_servicer

        # Start the request manager's event loop using auto_create_handle_loop
        self.request_manager.auto_create_handle_loop()

        logger.info("gRPC scheduler servicer initialized")

    async def Generate(
        self,
        request: sglang_scheduler_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[sglang_scheduler_pb2.GenerateResponse]:
        """Handle generation requests with streaming responses."""
        logger.info(f"Receive generation request: {request.request_id}")

        try:
            # Convert gRPC request to internal format
            tokenized_req = self._convert_generate_request(request)

            # Submit to request manager (automatically handles n>1)
            response_generator = self.request_manager.generate_request(
                obj=tokenized_req,
                request_id=request.request_id,
                grpc_context=context,
            )

            async for output in response_generator:
                # Handle batch responses (for n>1 non-streaming)
                if isinstance(output, list):
                    for batch_output in output:
                        if "error" in batch_output:
                            yield sglang_scheduler_pb2.GenerateResponse(
                                request_id=request.request_id,
                                error=sglang_scheduler_pb2.GenerateError(
                                    message=batch_output["error"],
                                    http_status_code=(
                                        "500" if "abort" not in batch_output else "499"
                                    ),
                                ),
                            )
                        else:
                            # All non-error batch outputs are final responses
                            yield self._create_completion_response(
                                request.request_id, batch_output
                            )
                else:
                    # Handle single response (for streaming or n=1 non-streaming)
                    if "error" in output:
                        yield sglang_scheduler_pb2.GenerateResponse(
                            request_id=request.request_id,
                            error=sglang_scheduler_pb2.GenerateError(
                                message=output["error"],
                                http_status_code=(
                                    "500" if "abort" not in output else "499"
                                ),
                            ),
                        )
                    elif output.get("finished", False):
                        yield self._create_completion_response(
                            request.request_id, output
                        )
                    else:
                        yield self._create_chunk_response(request.request_id, output)

        except Exception as e:
            logger.error(
                f"Generate failed for request {request.request_id}: {e}\n"
                f"{get_exception_traceback()}"
            )
            yield sglang_scheduler_pb2.GenerateResponse(
                request_id=request.request_id,
                error=sglang_scheduler_pb2.GenerateError(
                    message=str(e),
                    http_status_code="500",
                    details=get_exception_traceback(),
                ),
            )

    async def Embed(
        self,
        request: sglang_scheduler_pb2.EmbedRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.EmbedResponse:
        """Handle embedding requests."""
        logger.info(f"Receive embedding request: {request.request_id}")

        try:
            # Convert request
            tokenized_req = self._convert_embed_request(request)

            # Submit to request manager
            future = await self.request_manager.embedding_request(
                obj=tokenized_req,
                request_id=request.request_id,
            )

            # Wait for result
            result = await future

            # Create response
            return sglang_scheduler_pb2.EmbedResponse(
                request_id=request.request_id,
                complete=sglang_scheduler_pb2.EmbedComplete(
                    embedding=result["embedding"],
                    prompt_tokens=result.get("prompt_tokens", 0),
                    cached_tokens=0,
                    embedding_dim=len(result["embedding"]),
                ),
            )

        except Exception as e:
            logger.error(
                f"Embed failed for request {request.request_id}: {e}\n"
                f"{get_exception_traceback()}"
            )
            return sglang_scheduler_pb2.EmbedResponse(
                request_id=request.request_id,
                error=sglang_scheduler_pb2.EmbedError(
                    message=str(e),
                    code="INTERNAL_ERROR",
                    details=get_exception_traceback(),
                ),
            )

    async def HealthCheck(
        self,
        request: sglang_scheduler_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.HealthCheckResponse:
        """
        Check the health of the inference server by sending a special request to generate one token.
        Similar to HTTP server's /health endpoint.
        """
        rid = f"HEALTH_CHECK_{time.time()}"
        logger.info(f"Receive health check request: {rid}")

        if self.request_manager.gracefully_exit:
            logger.info(
                "Health check request received during shutdown. Returning unhealthy."
            )
            return sglang_scheduler_pb2.HealthCheckResponse(
                healthy=False, message="Server is shutting down"
            )

        # Create a special health check request
        sampling_params = SGLSamplingParams(max_new_tokens=1, temperature=0.0)
        sampling_params.normalize(tokenizer=None)

        # Create health check request
        is_generation = self.scheduler_info.get("is_generation", True)
        if is_generation:
            health_req = TokenizedGenerateReqInput(
                rid=rid,
                input_text="",
                input_ids=[0],
                sampling_params=sampling_params,
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                stream=False,
                mm_inputs=None,
                token_ids_logprob=None,
            )
            # Set disaggregation params if needed
            if self.server_args.disaggregation_mode != DisaggregationMode.NULL:
                health_req.bootstrap_host = FAKE_BOOTSTRAP_HOST
                health_req.bootstrap_room = 0
        else:
            health_req = TokenizedEmbeddingReqInput(
                rid=rid,
                input_text="",
                input_ids=[0],
            )

        # Submit health check request
        async def run_health_check():
            try:
                async for _ in self.request_manager.generate_request(
                    obj=health_req,
                    request_id=rid,
                ):
                    # Got at least one response, server is healthy
                    return True
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                return False
            return False

        task = asyncio.create_task(run_health_check())

        # Wait for response with timeout
        tic = time.time()
        while time.time() < tic + HEALTH_CHECK_TIMEOUT:
            await asyncio.sleep(1)
            # Check if we got a response from scheduler
            if self.request_manager.last_receive_tstamp > tic:
                task.cancel()
                # Clean up health check state
                self.request_manager._cleanup_request_state(rid)
                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=True, message="Health check passed"
                )

        # Timeout - server not responding
        task.cancel()
        self.request_manager._cleanup_request_state(rid)
        logger.warning(f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s")
        return sglang_scheduler_pb2.HealthCheckResponse(
            healthy=False, message=f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s"
        )

    async def Abort(
        self,
        request: sglang_scheduler_pb2.AbortRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.AbortResponse:
        """Abort an ongoing request."""
        logger.info(f"Receive abort request: {request.request_id}")

        try:
            success = await self.request_manager.abort_request(request.request_id)

            return sglang_scheduler_pb2.AbortResponse(
                success=success,
                message=f"Request {request.request_id} {'aborted' if success else 'not found'}",
            )
        except Exception as e:
            logger.error(
                f"Abort failed for request {request.request_id}: {e}\n"
                f"{get_exception_traceback()}"
            )
            return sglang_scheduler_pb2.AbortResponse(
                success=False,
                message=str(e),
            )

    async def GetModelInfo(
        self,
        _request: sglang_scheduler_pb2.GetModelInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetModelInfoResponse:
        """Get model information."""
        logger.debug("Receive model info request")

        is_generation = self.scheduler_info.get("is_generation")
        if is_generation is None:
            is_generation = not self.server_args.is_embedding

        return sglang_scheduler_pb2.GetModelInfoResponse(
            model_path=self.server_args.model_path,
            tokenizer_path=self.server_args.tokenizer_path or "",
            is_generation=is_generation,
            preferred_sampling_params=(
                self.server_args.preferred_sampling_params or ""
            ),
            weight_version=self.server_args.weight_version or "",
            served_model_name=self.server_args.served_model_name,
            max_context_length=self.model_info["max_context_length"],
            vocab_size=self.model_info["vocab_size"],
            supports_vision=self.model_info["supports_vision"],
            model_type=self.model_info.get("model_type") or "",
            architectures=self.model_info.get("architectures") or [],
            eos_token_ids=self.model_info["eos_token_ids"],
            pad_token_id=self.model_info["pad_token_id"],
            bos_token_id=self.model_info["bos_token_id"],
            max_req_input_len=self.model_info["max_req_input_len"],
        )

    async def GetServerInfo(
        self,
        _request: sglang_scheduler_pb2.GetServerInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetServerInfoResponse:
        """Get server information."""
        logger.debug("Receive server info request")

        server_args_dict = dataclasses.asdict(self.server_args)
        server_args_struct = Struct()

        def make_serializable(obj):
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple, set)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)

        serializable_args = make_serializable(server_args_dict)
        server_args_struct.update(serializable_args)

        # Convert scheduler_info to Struct
        scheduler_info_struct = Struct()
        scheduler_info_struct.update(self.scheduler_info)

        # Get runtime state from request manager
        manager_state = self.request_manager.get_server_info()

        # Calculate uptime
        uptime = time.time() - self.start_time

        # Create timestamp
        start_timestamp = Timestamp()
        start_timestamp.FromSeconds(int(self.start_time))

        return sglang_scheduler_pb2.GetServerInfoResponse(
            server_args=server_args_struct,
            scheduler_info=scheduler_info_struct,
            active_requests=manager_state["active_requests"],
            is_paused=manager_state["paused"],
            last_receive_timestamp=manager_state["last_receive_time"],
            uptime_seconds=uptime,
            sglang_version=sglang.__version__,
            server_type="grpc",
            start_time=start_timestamp,
        )

    # Helper methods for request/response conversion

    def _convert_generate_request(
        self, grpc_req: sglang_scheduler_pb2.GenerateRequest
    ) -> TokenizedGenerateReqInput:
        """Convert gRPC GenerateRequest to internal format."""

        # Extract tokenized input
        if not grpc_req.HasField("tokenized"):
            raise ValueError("Tokenized input must be provided")

        input_text = grpc_req.tokenized.original_text
        input_ids = list(grpc_req.tokenized.input_ids)

        # Convert sampling params
        sampling_params = self._convert_sampling_params(grpc_req.sampling_params)
        sampling_params.normalize(tokenizer=None)

        # Extract disaggregated params if present
        bootstrap_host = None
        bootstrap_port = None
        bootstrap_room = None
        if grpc_req.HasField("disaggregated_params"):
            # Don't use 'or None' as it treats 0 as falsy
            bootstrap_host = (
                grpc_req.disaggregated_params.bootstrap_host
                if grpc_req.disaggregated_params.bootstrap_host
                else None
            )
            bootstrap_port = (
                grpc_req.disaggregated_params.bootstrap_port
                if grpc_req.disaggregated_params.bootstrap_port
                else None
            )
            bootstrap_room = (
                grpc_req.disaggregated_params.bootstrap_room
            )  # Can be 0, don't use 'or None'

        # Create request
        return TokenizedGenerateReqInput(
            rid=grpc_req.request_id,
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=None,  # TODO: implement mm support
            sampling_params=sampling_params,
            return_logprob=grpc_req.return_logprob,
            logprob_start_len=(
                grpc_req.logprob_start_len
                if grpc_req.logprob_start_len is not None
                else -1
            ),
            top_logprobs_num=grpc_req.top_logprobs_num or 0,
            stream=grpc_req.stream or False,
            lora_id=grpc_req.lora_id if grpc_req.lora_id else None,
            token_ids_logprob=(
                list(grpc_req.token_ids_logprob) if grpc_req.token_ids_logprob else None
            ),
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
        )

    def _convert_embed_request(
        self, grpc_req: sglang_scheduler_pb2.EmbedRequest
    ) -> TokenizedEmbeddingReqInput:
        """Convert gRPC EmbedRequest to internal format."""

        # Extract tokenized input
        if not grpc_req.HasField("tokenized"):
            raise ValueError("Tokenized input must be provided")

        input_text = grpc_req.tokenized.original_text
        input_ids = list(grpc_req.tokenized.input_ids)

        return TokenizedEmbeddingReqInput(
            rid=grpc_req.request_id,
            input_text=input_text,
            input_ids=input_ids,
        )

    def _convert_sampling_params(
        self, grpc_params: sglang_scheduler_pb2.SamplingParams
    ) -> SGLSamplingParams:
        """Convert gRPC SamplingParams to internal format."""

        # Handle constraint types
        regex = None
        json_schema = None
        ebnf_grammar = None
        structural_tag = None

        if grpc_params.HasField("regex"):
            regex = grpc_params.regex
        elif grpc_params.HasField("json_schema"):
            json_schema = grpc_params.json_schema
        elif grpc_params.HasField("ebnf_grammar"):
            ebnf_grammar = grpc_params.ebnf_grammar
        elif grpc_params.HasField("structural_tag"):
            structural_tag = grpc_params.structural_tag

        # Handle optional parameters conversion
        custom_params = (
            MessageToDict(grpc_params.custom_params)
            if grpc_params.HasField("custom_params")
            else None
        )
        max_new_tokens = (
            grpc_params.max_new_tokens
            if grpc_params.HasField("max_new_tokens")
            else None
        )
        stream_interval = (
            grpc_params.stream_interval
            if grpc_params.HasField("stream_interval")
            else None
        )
        logit_bias = dict(grpc_params.logit_bias) if grpc_params.logit_bias else None
        stop = list(grpc_params.stop) if grpc_params.stop else None
        stop_token_ids = (
            list(grpc_params.stop_token_ids) if grpc_params.stop_token_ids else None
        )

        return SGLSamplingParams(
            temperature=grpc_params.temperature,
            top_p=grpc_params.top_p,
            top_k=grpc_params.top_k,
            min_p=grpc_params.min_p,
            frequency_penalty=grpc_params.frequency_penalty,
            presence_penalty=grpc_params.presence_penalty,
            repetition_penalty=grpc_params.repetition_penalty,
            max_new_tokens=max_new_tokens,
            min_new_tokens=grpc_params.min_new_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=grpc_params.skip_special_tokens,
            spaces_between_special_tokens=grpc_params.spaces_between_special_tokens,
            no_stop_trim=grpc_params.no_stop_trim,
            regex=regex,
            json_schema=json_schema,
            ebnf=ebnf_grammar,
            structural_tag=structural_tag,
            n=grpc_params.n,
            ignore_eos=grpc_params.ignore_eos,
            stream_interval=stream_interval,
            logit_bias=logit_bias,
            custom_params=custom_params,
        )

    def _convert_output_logprobs_to_proto(
        self, logprobs_data: Dict
    ) -> Optional[sglang_scheduler_pb2.OutputLogProbs]:
        """Convert output logprobs dict to proto (no None values, plain floats)."""
        if not logprobs_data:
            return None

        token_logprobs_val = logprobs_data.get("token_logprobs_val", [])
        token_logprobs_idx = logprobs_data.get("token_logprobs_idx", [])
        top_logprobs_val = logprobs_data.get("top_logprobs_val", [])
        top_logprobs_idx = logprobs_data.get("top_logprobs_idx", [])

        # Build TopLogProbs entries
        top_logprobs_proto = []
        if top_logprobs_val and top_logprobs_idx:
            for val_list, idx_list in zip(top_logprobs_val, top_logprobs_idx):
                top_logprobs_proto.append(
                    sglang_scheduler_pb2.TopLogProbs(
                        values=val_list,
                        token_ids=idx_list,
                    )
                )

        return sglang_scheduler_pb2.OutputLogProbs(
            token_logprobs=token_logprobs_val,  # Plain float array
            token_ids=token_logprobs_idx,
            top_logprobs=top_logprobs_proto,
        )

    def _convert_input_logprobs_to_proto(
        self, logprobs_data: Dict
    ) -> Optional[sglang_scheduler_pb2.InputLogProbs]:
        """Convert input logprobs dict to proto (first token is None, wrapped in InputTokenLogProb)."""
        if not logprobs_data:
            return None

        token_logprobs_val = logprobs_data.get("token_logprobs_val", [])
        token_logprobs_idx = logprobs_data.get("token_logprobs_idx", [])
        top_logprobs_val = logprobs_data.get("top_logprobs_val", [])
        top_logprobs_idx = logprobs_data.get("top_logprobs_idx", [])

        # Wrap values in InputTokenLogProb (None for first token, value for others)
        token_logprobs_wrapped = [
            (
                sglang_scheduler_pb2.InputTokenLogProb()
                if x is None
                else sglang_scheduler_pb2.InputTokenLogProb(value=x)
            )
            for x in token_logprobs_val
        ]

        # Build TopLogProbs entries
        top_logprobs_proto = []
        if top_logprobs_val and top_logprobs_idx:
            for val_list, idx_list in zip(top_logprobs_val, top_logprobs_idx):
                top_logprobs_proto.append(
                    sglang_scheduler_pb2.TopLogProbs(
                        values=val_list,
                        token_ids=idx_list,
                    )
                )

        return sglang_scheduler_pb2.InputLogProbs(
            token_logprobs=token_logprobs_wrapped,
            token_ids=token_logprobs_idx,
            top_logprobs=top_logprobs_proto,
        )

    def _create_chunk_response(
        self, request_id: str, output: Dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a streaming chunk response."""
        meta_info = output.get("meta_info", {})

        # Convert output logprobs if present
        output_logprobs_proto = self._convert_output_logprobs_to_proto(
            output.get("output_logprobs")
        )

        # Convert input logprobs if present (only in first chunk)
        input_logprobs_proto = self._convert_input_logprobs_to_proto(
            output.get("input_logprobs")
        )

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            chunk=sglang_scheduler_pb2.GenerateStreamChunk(
                token_ids=output.get("token_ids", []),
                prompt_tokens=meta_info.get("prompt_tokens", 0),
                completion_tokens=meta_info.get("completion_tokens", 0),
                cached_tokens=meta_info.get("cached_tokens", 0),
                output_logprobs=output_logprobs_proto,
                input_logprobs=input_logprobs_proto,
                index=output.get("index", 0),
            ),
        )

    def _create_completion_response(
        self, request_id: str, output: Dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a completion response."""

        # Extract meta info and finish reason details
        meta_info = output.get("meta_info", {})
        finish_reason_data = meta_info.get("finish_reason")

        # Determine finish reason, default is stop
        finish_reason = "stop"
        if finish_reason_data:
            if isinstance(finish_reason_data, dict):
                finish_reason_type = finish_reason_data.get("type")
            else:
                # Handle legacy string format
                finish_reason_type = finish_reason_data

            if finish_reason_type == "length":
                finish_reason = "length"
            elif finish_reason_type == "abort":
                finish_reason = "abort"

        # Extract matched_stop information
        matched_stop_kwargs = {}
        if isinstance(finish_reason_data, dict) and "matched" in finish_reason_data:
            matched = finish_reason_data["matched"]
            if isinstance(matched, int):
                matched_stop_kwargs["matched_token_id"] = matched
            elif isinstance(matched, str):
                matched_stop_kwargs["matched_stop_str"] = matched

        # Convert output logprobs if present
        output_logprobs_proto = self._convert_output_logprobs_to_proto(
            output.get("output_logprobs")
        )

        # Convert input logprobs if present
        input_logprobs_proto = self._convert_input_logprobs_to_proto(
            output.get("input_logprobs")
        )

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            complete=sglang_scheduler_pb2.GenerateComplete(
                output_ids=output.get("token_ids", []),
                finish_reason=finish_reason,
                prompt_tokens=meta_info.get("prompt_tokens", 0),
                completion_tokens=meta_info.get(
                    "completion_tokens", len(output.get("token_ids", []))
                ),
                cached_tokens=meta_info.get("cached_tokens", 0),
                output_logprobs=output_logprobs_proto,
                input_logprobs=input_logprobs_proto,
                index=output.get("index", 0),
                **matched_stop_kwargs,
            ),
        )

    async def shutdown(self):
        """Shutdown the service."""
        logger.info("Shutting down gRPC service")

        # Mark health service as NOT_SERVING before shutdown
        if self.health_servicer:
            self.health_servicer.set_not_serving()

        # Shutdown request manager (handles its own tasks)
        await self.request_manager.shutdown()


async def serve_grpc(
    server_args: ServerArgs,
    model_info: Optional[Dict] = None,
):
    """Start the standalone gRPC server with integrated scheduler."""

    # Start bootstrap server BEFORE launching scheduler processes (only in PREFILL mode)
    # This ensures the bootstrap server is ready when prefill schedulers try to register
    bootstrap_server = None
    if server_args.disaggregation_mode == "prefill":
        bootstrap_server = start_disagg_service(server_args)
        if bootstrap_server:
            logger.info(
                f"Bootstrap server started for disaggregation mode on {server_args.host}:{server_args.disaggregation_bootstrap_port}"
            )

    # Launch only the scheduler process(es) (no tokenizer/detokenizer needed for gRPC)
    logger.info("Launching scheduler process(es)...")
    scheduler_info, port_args, scheduler_procs = launch_scheduler_process_only(
        server_args=server_args,
    )

    # Load model config to get HF config info (same as TokenizerManager does)
    model_config = ModelConfig.from_server_args(server_args)

    # Update model info from scheduler info and model config
    if model_info is None:
        model_info = {
            "model_name": server_args.model_path,
            "max_context_length": scheduler_info.get(
                "max_total_num_tokens", server_args.context_length or 8192
            ),
            "vocab_size": scheduler_info.get("vocab_size", 128256),
            "supports_vision": scheduler_info.get("supports_vision", False),
            "model_type": getattr(model_config.hf_config, "model_type", None),
            "architectures": getattr(model_config.hf_config, "architectures", None),
            "max_req_input_len": scheduler_info.get("max_req_input_len", 8192),
            "eos_token_ids": scheduler_info.get("eos_token_ids", []),
            "pad_token_id": scheduler_info.get("pad_token_id", 0),
            "bos_token_id": scheduler_info.get("bos_token_id", 1),
        }

    # Create request manager with the correct port args
    # Note: We pass None for bootstrap_server since it's already started above
    request_manager = GrpcRequestManager(
        server_args=server_args,
        port_args=port_args,
        bootstrap_server=bootstrap_server,
    )

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    # Create standard health service (for Kubernetes probes)
    health_servicer = SGLangHealthServicer(
        request_manager=request_manager,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Add SGLang service
    servicer = SGLangSchedulerServicer(
        request_manager=request_manager,
        server_args=server_args,
        model_info=model_info,
        scheduler_info=scheduler_info,
        health_servicer=health_servicer,
    )
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)

    # Enable reflection
    SERVICE_NAMES = (
        sglang_scheduler_pb2.DESCRIPTOR.services_by_name["SglangScheduler"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start server
    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC server listening on {listen_addr}")

    # Start warmup in a separate thread
    warmup_thread = threading.Thread(
        target=_wait_and_warmup_grpc,
        args=(server_args, None, health_servicer),
    )
    warmup_thread.start()

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down gRPC server")

        # Shutdown request manager first - this closes ZMQ sockets and stops background tasks
        await servicer.shutdown()

        # Stop the gRPC server
        await server.stop(5.0)

        # Wait for warmup thread to finish
        if warmup_thread.is_alive():
            logger.info("Waiting for warmup thread to finish...")
            warmup_thread.join(timeout=5.0)

        # Terminate scheduler processes before exiting to avoid atexit hang
        # The scheduler processes have SIGINT ignored, so they won't get KeyboardInterrupt
        for i, proc in enumerate(scheduler_procs):
            if proc.is_alive():
                logger.info(f"Terminating scheduler process {i}...")
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():
                    logger.warning(
                        f"Scheduler process {i} did not terminate, killing..."
                    )
                    proc.kill()
                    proc.join(timeout=1.0)

        logger.info("All scheduler processes terminated")


def _execute_grpc_server_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection],
):
    """Execute warmup for gRPC server by checking health and sending test request."""
    try:
        # Connect to the gRPC server
        grpc_url = f"{server_args.host}:{server_args.port}"
        channel = grpc.insecure_channel(
            grpc_url,
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 256),
                ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ],
        )
        stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)

        # Wait until the server is launched (poll GetModelInfo)
        success = False
        last_error = None
        for _ in range(120):
            time.sleep(1)
            try:
                request = sglang_scheduler_pb2.GetModelInfoRequest()
                response = stub.GetModelInfo(request, timeout=5)
                success = True
                break
            except Exception as e:
                last_error = str(e)
                pass

        if not success:
            error_msg = f"gRPC server warmup failed: Could not connect to server after 120 seconds. Last error: {last_error}"
            logger.error(error_msg)
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(error_msg)
            channel.close()
            kill_process_tree(os.getpid())
            return False

        # Get model info to determine if it's generation or embedding
        is_generation = response.is_generation

        # Send a warmup request
        logger.info("Sending warmup request to gRPC server...")
        max_new_tokens = 8 if is_generation else 1

        if is_generation:
            warmup_request_kwargs = {
                "request_id": f"WARMUP_{time.time()}",
                "tokenized": sglang_scheduler_pb2.TokenizedInput(
                    input_ids=[
                        123,
                        456,
                        789,
                        234,
                        567,
                        890,
                        345,
                    ],  # Random-looking but safe token IDs
                    original_text="warmup request",
                ),
                "sampling_params": sglang_scheduler_pb2.SamplingParams(
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                ),
                "stream": False,
            }

            # Set disaggregation params if needed
            if server_args.disaggregation_mode != DisaggregationMode.NULL:
                warmup_request_kwargs["disaggregated_params"] = (
                    sglang_scheduler_pb2.DisaggregatedParams(
                        bootstrap_host=FAKE_BOOTSTRAP_HOST,
                        bootstrap_room=0,
                    )
                )

            warmup_request = sglang_scheduler_pb2.GenerateRequest(
                **warmup_request_kwargs
            )

            # Send the warmup request
            try:
                responses = list(stub.Generate(warmup_request, timeout=600))
                # Check if we got a valid response
                if responses and not responses[-1].HasField("error"):
                    logger.info("gRPC warmup request completed successfully")
                    success = True
                else:
                    error_msg = (
                        responses[-1].error.message if responses else "No response"
                    )
                    logger.warning(f"gRPC warmup request returned error: {error_msg}")
                    success = False
            except Exception as e:
                error_msg = f"gRPC warmup request failed: {e}"
                logger.error(error_msg)
                if pipe_finish_writer is not None:
                    pipe_finish_writer.send(error_msg)
                channel.close()
                kill_process_tree(os.getpid())
                return False
        else:
            # For embedding models
            warmup_request = sglang_scheduler_pb2.EmbedRequest(
                request_id=f"WARMUP_{time.time()}",
                tokenized=sglang_scheduler_pb2.TokenizedInput(
                    input_ids=[10, 11, 12],
                    original_text="test embedding",
                ),
            )

            try:
                response = stub.Embed(warmup_request, timeout=600)
                if not response.HasField("error"):
                    logger.info("gRPC warmup request completed successfully")
                    success = True
                else:
                    logger.warning(
                        f"gRPC warmup request returned error: {response.error.message}"
                    )
                    success = False
            except Exception as e:
                error_msg = f"gRPC warmup request failed: {e}"
                logger.error(error_msg)
                if pipe_finish_writer is not None:
                    pipe_finish_writer.send(error_msg)
                channel.close()
                kill_process_tree(os.getpid())
                return False

        channel.close()
        return success

    except Exception as e:
        error_msg = (
            f"gRPC warmup failed with exception: {e}\n{get_exception_traceback()}"
        )
        logger.error(error_msg)
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(error_msg)
        try:
            channel.close()
        except Exception:
            pass
        kill_process_tree(os.getpid())
        return False


def _wait_and_warmup_grpc(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection],
    health_servicer: Optional[SGLangHealthServicer] = None,
):
    """Wait for gRPC server to be ready and execute warmup."""
    if not server_args.skip_server_warmup:
        if not _execute_grpc_server_warmup(server_args, pipe_finish_writer):
            return
    else:
        logger.info("Skipping gRPC server warmup (skip_server_warmup=True)")

    # Mark health service as SERVING after warmup completes
    if health_servicer:
        health_servicer.set_serving()

    logger.info("The server is fired up and ready to roll!")

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")
