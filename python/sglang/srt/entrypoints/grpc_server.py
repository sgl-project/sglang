"""
Standalone gRPC Server for SGLang - Fully separated from HTTP server.
Uses GrpcRequestManager for orchestration without tokenization.
"""

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import signal
import time
from concurrent import futures
from typing import AsyncIterator, Dict, Optional, Tuple

import grpc
from grpc_reflection.v1alpha import reflection

from sglang.srt.entrypoints.grpc_request_manager import GrpcRequestManager
from sglang.srt.grpc import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.sampling.sampling_params import SamplingParams as SGLSamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import configure_logger, prepare_model_and_tokenizer
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


def _launch_scheduler_process_only(
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
) -> Tuple[Dict, PortArgs, list]:
    """
    Launch only the scheduler process(es) without tokenizer/detokenizer.
    Returns scheduler info, port args, and list of scheduler processes.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # Prepare model and tokenizer paths
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                proc = mp.Process(
                    target=run_scheduler_process,
                    args=(
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        None,
                        writer,
                        None,
                    ),
                )

                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    # TODO(CatherineSue): handle cases for multi-node

    # Wait for all scheduler processes to be ready
    scheduler_infos = []
    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise RuntimeError(f"Failed to initialize scheduler rank {i}")

        if data.get("status") != "ready":
            raise RuntimeError(
                f"Scheduler rank {i} initialization failed: {data.get('error', 'Unknown error')}"
            )
        scheduler_infos.append(data)

    logger.info(
        f"All {len(scheduler_procs)} scheduler process(es) initialized successfully"
    )

    # Return the first scheduler's info (they should all be the same)
    return scheduler_infos[0], port_args, scheduler_procs


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
    ):
        """Initialize the standalone gRPC service."""
        self.request_manager = request_manager
        self.server_args = server_args
        self.model_info = model_info
        self.start_time = time.time()

        # Start the request manager's event loop using auto_create_handle_loop
        self.request_manager.auto_create_handle_loop()

        logger.info("Standalone gRPC scheduler service initialized")

    async def Generate(
        self,
        request: sglang_scheduler_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[sglang_scheduler_pb2.GenerateResponse]:
        """Handle generation requests with streaming responses."""
        logger.info(f"Generation request: {request.request_id}")

        try:
            # Convert gRPC request to internal format
            tokenized_req = self._convert_generate_request(request)

            # Submit to request manager
            output_queue = await self.request_manager.generate_request(
                obj=tokenized_req,
                request_id=request.request_id,
                grpc_context=context,
            )

            # Stream outputs
            while True:
                try:
                    # Get output with timeout
                    output = await asyncio.wait_for(output_queue.get(), timeout=4)

                    # Check for errors
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
                        break

                    # Check if finished
                    if output.get("finished", False):
                        # Send completion
                        yield self._create_completion_response(
                            request.request_id, output
                        )
                        break
                    else:
                        # Send chunk
                        yield self._create_chunk_response(request.request_id, output)

                except asyncio.TimeoutError:
                    # Check if context is still active
                    if context.cancelled():
                        # Abort the request
                        await self.request_manager.abort_request(request.request_id)
                        break
                    continue

        except Exception as e:
            logger.error(f"Generate failed: {e}\n{get_exception_traceback()}")
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
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.EmbedResponse:
        """Handle embedding requests."""
        logger.info(f"Embedding request: {request.request_id}")

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
                    generation_time=time.time() - self.start_time,
                ),
            )

        except Exception as e:
            logger.error(f"Embed failed: {e}\n{get_exception_traceback()}")
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
        """Health check by generating from client input."""
        try:
            # Check if request manager is shutting down
            if self.request_manager.gracefully_exit:
                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=False, message="Server shutting down"
                )

            # Extract tokenized input from request
            if not request.HasField("tokenized"):
                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=False, message="Tokenized input required for health check"
                )

            input_text = request.tokenized.original_text
            input_ids = list(request.tokenized.input_ids)

            # Create health check request
            rid = f"HEALTH_CHECK_GRPC_{time.time()}"

            health_request = TokenizedGenerateReqInput(
                rid=rid,
                input_text=input_text,
                input_ids=input_ids,
                sampling_params=SGLSamplingParams(max_new_tokens=1, temperature=0.0),
                stream=False,
                mm_inputs=None,
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
            )

            logger.info(f"Sending health check request to request manager...")

            # Submit and wait for response
            output_queue = await self.request_manager.generate_request(
                health_request, request_id=rid
            )

            try:
                # Wait for response with configurable timeout
                response = await asyncio.wait_for(
                    output_queue.get(), timeout=HEALTH_CHECK_TIMEOUT
                )

                # Clean up
                if rid in self.request_manager.rid_to_state:
                    del self.request_manager.rid_to_state[rid]

                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=True, message="Health check passed"
                )

            except asyncio.TimeoutError:
                # Clean up on timeout
                if rid in self.request_manager.rid_to_state:
                    del self.request_manager.rid_to_state[rid]

                return sglang_scheduler_pb2.HealthCheckResponse(
                    healthy=False, message="Health check timeout"
                )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return sglang_scheduler_pb2.HealthCheckResponse(
                healthy=False, message=f"Health check error: {str(e)}"
            )

    async def Abort(
        self,
        request: sglang_scheduler_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.AbortResponse:
        """Abort an ongoing request."""
        logger.info(f"Aborting request: {request.request_id}")

        try:
            success = await self.request_manager.abort_request(request.request_id)

            return sglang_scheduler_pb2.AbortResponse(
                success=success,
                message=f"Request {request.request_id} {'aborted' if success else 'not found'}",
            )
        except Exception as e:
            logger.error(f"Abort failed: {e}")
            return sglang_scheduler_pb2.AbortResponse(
                success=False,
                message=str(e),
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

        # Create request
        return TokenizedGenerateReqInput(
            rid=grpc_req.request_id,
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=None,  # TODO: implement mm support
            sampling_params=sampling_params,
            return_logprob=grpc_req.return_logprob,
            logprob_start_len=grpc_req.logprob_start_len or -1,
            top_logprobs_num=grpc_req.top_logprobs_num or 0,
            stream=True,  # Always stream for gRPC
            lora_path=grpc_req.lora_id if grpc_req.lora_id else None,
            token_ids_logprob=(
                list(grpc_req.token_ids_logprob) if grpc_req.token_ids_logprob else None
            ),
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

        if grpc_params.HasField("regex"):
            regex = grpc_params.regex
        elif grpc_params.HasField("json_schema"):
            json_schema = grpc_params.json_schema
        elif grpc_params.HasField("ebnf_grammar"):
            ebnf_grammar = grpc_params.ebnf_grammar

        return SGLSamplingParams(
            temperature=grpc_params.temperature or 1.0,
            top_p=grpc_params.top_p or 1.0,
            top_k=grpc_params.top_k or -1,
            min_p=grpc_params.min_p or 0.0,
            frequency_penalty=grpc_params.frequency_penalty or 0.0,
            presence_penalty=grpc_params.presence_penalty or 0.0,
            repetition_penalty=grpc_params.repetition_penalty or 1.0,
            max_new_tokens=grpc_params.max_new_tokens or 128,
            min_new_tokens=grpc_params.min_new_tokens or 0,
            stop=list(grpc_params.stop) if grpc_params.stop else None,
            stop_token_ids=(
                list(grpc_params.stop_token_ids) if grpc_params.stop_token_ids else None
            ),
            skip_special_tokens=grpc_params.skip_special_tokens,
            spaces_between_special_tokens=grpc_params.spaces_between_special_tokens,
            regex=regex,
            json_schema=json_schema,
            ebnf=ebnf_grammar,
            n=grpc_params.n or 1,
            ignore_eos=grpc_params.ignore_eos,
        )

    def _create_chunk_response(
        self, request_id: str, output: Dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a streaming chunk response."""
        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            chunk=sglang_scheduler_pb2.GenerateStreamChunk(
                token_id=output["token_ids"][-1] if output.get("token_ids") else 0,
                text=output.get("text", ""),
                prompt_tokens=0,
                completion_tokens=len(output.get("token_ids", [])),
                cached_tokens=0,
                generation_time=time.time() - self.start_time,
                queue_time=0.0,
            ),
        )

    def _create_completion_response(
        self, request_id: str, output: Dict
    ) -> sglang_scheduler_pb2.GenerateResponse:
        """Create a completion response."""

        # Determine finish reason
        finish_reason = sglang_scheduler_pb2.GenerateComplete.STOP
        meta_info = output.get("meta_info", {})
        if meta_info.get("finish_reason") == "length":
            finish_reason = sglang_scheduler_pb2.GenerateComplete.LENGTH
        elif meta_info.get("finish_reason") == "eos_token":
            finish_reason = sglang_scheduler_pb2.GenerateComplete.EOS_TOKEN

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=request_id,
            complete=sglang_scheduler_pb2.GenerateComplete(
                output_ids=output.get("token_ids", []),
                output_text=output.get("text", ""),
                finish_reason=finish_reason,
            ),
        )

    async def shutdown(self):
        """Shutdown the service."""
        logger.info("Shutting down gRPC service")

        # Shutdown request manager (handles its own tasks)
        await self.request_manager.shutdown()


async def serve_grpc(
    server_args: ServerArgs,
    model_info: Optional[Dict] = None,
):
    """Start the standalone gRPC server with integrated scheduler."""

    # Launch only the scheduler process(es) (no tokenizer/detokenizer needed for gRPC)
    logger.info("Launching scheduler process(es)...")
    scheduler_info, port_args, scheduler_procs = _launch_scheduler_process_only(
        server_args=server_args,
    )

    # Update model info from scheduler info
    if model_info is None:
        model_info = {
            "model_name": server_args.model_path,
            "max_context_length": scheduler_info.get(
                "max_total_num_tokens", server_args.context_length or 8192
            ),
            "vocab_size": scheduler_info.get("vocab_size", 128256),
            "supports_vision": scheduler_info.get("supports_vision", False),
            "model_type": scheduler_info.get("model_type", "transformer"),
            "max_req_input_len": scheduler_info.get("max_req_input_len", 8192),
            "eos_token_ids": scheduler_info.get("eos_token_ids", []),
            "pad_token_id": scheduler_info.get("pad_token_id", 0),
            "bos_token_id": scheduler_info.get("bos_token_id", 1),
        }

    # Create request manager with the correct port args
    request_manager = GrpcRequestManager(
        server_args=server_args,
        port_args=port_args,
    )

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    # Add service
    servicer = SGLangSchedulerServicer(
        request_manager=request_manager,
        server_args=server_args,
        model_info=model_info,
    )
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)

    # Enable reflection
    SERVICE_NAMES = (
        sglang_scheduler_pb2.DESCRIPTOR.services_by_name["SglangScheduler"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start server
    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting standalone gRPC server on {listen_addr}")

    await server.start()

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
        await servicer.shutdown()
        await server.stop(5.0)

        # Terminate scheduler processes
        for i, proc in enumerate(scheduler_procs):
            if proc and proc.is_alive():
                logger.info(f"Terminating scheduler process {i}...")
                proc.terminate()
                proc.join(timeout=5.0)
                if proc.is_alive():
                    logger.warning(f"Force killing scheduler process {i}...")
                    proc.kill()
                    proc.join()


def main():
    """Main entry point for standalone gRPC server."""
    # Fix CUDA multiprocessing issues - must be called before any CUDA operations
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="SGLang Standalone gRPC Server")

    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=30000, help="gRPC server port")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--tokenizer-path", type=str, help="Tokenizer path")
    parser.add_argument("--context-length", type=int, help="Context length")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")

    # Runtime arguments
    parser.add_argument(
        "--max-running-requests", type=int, default=2048, help="Max concurrent requests"
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=1000000, help="Max total tokens"
    )
    parser.add_argument(
        "--max-prefill-tokens", type=int, default=16384, help="Max prefill tokens"
    )
    parser.add_argument(
        "--attention-backend", type=str, default="flashinfer", help="Attention backend"
    )
    parser.add_argument("--lora-paths", type=str, help="LoRA adapter paths")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Convert to ServerArgs with gRPC host and port
    server_args = ServerArgs(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or args.model_path,
        context_length=args.context_length,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        max_prefill_tokens=args.max_prefill_tokens,
        attention_backend=args.attention_backend,
        lora_paths=args.lora_paths.split(",") if args.lora_paths else None,
        log_level=args.log_level,
        # Override with gRPC server host and port
        host=args.host,
        port=args.port,
    )

    # Run server
    asyncio.run(
        serve_grpc(
            server_args=server_args,
        )
    )


if __name__ == "__main__":
    main()
