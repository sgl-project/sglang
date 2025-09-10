"""
gRPC Request Manager - Orchestrates request lifecycle without tokenization.
Mimics TokenizerManager's state management and ZMQ communication patterns.
"""

import asyncio
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union

import grpc
import zmq
import zmq.asyncio

from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    HealthCheckOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket, kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class GrpcSignalHandler:
    """Minimal signal handler for gRPC server - delegates real crash handling to scheduler."""

    def __init__(self, grpc_manager):
        self.grpc_manager = grpc_manager

    def sigterm_handler(self, signum=None, frame=None):
        """Handle SIGTERM by gracefully shutting down gRPC server."""
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Shutting down gRPC server..."
        )
        self.grpc_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        """Handle SIGQUIT from failed scheduler process."""
        logger.error(
            "Received SIGQUIT from scheduler process. Scheduler failed, shutting down gRPC server."
        )
        logger.info(
            "Note: Crash dumps are handled by the scheduler process, not the gRPC server."
        )
        # Just exit cleanly - the scheduler handles crash dumps
        kill_process_tree(os.getpid(), include_parent=True)


@dataclasses.dataclass
class GrpcReqState:
    """State tracking for a gRPC request."""

    # Request identification
    request_id: str
    grpc_context: Optional[grpc.aio.ServicerContext]

    # Communication
    out_queue: asyncio.Queue
    finished: bool
    event: asyncio.Event
    obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]

    # Metrics (same as TokenizerManager's ReqState)
    created_time: float
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # Streaming state
    last_output_offset: int = 0
    stream_finished: bool = False

    # Output accumulation
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

    # Session state
    session_id: Optional[str] = None
    is_session_request: bool = False


class GrpcRequestManager:
    """
    Manages gRPC request lifecycle, mimicking TokenizerManager's orchestration
    behaviors without tokenization.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        """Initialize the gRPC request manager."""
        self.server_args = server_args
        self.port_args = port_args

        # ZMQ Communication Setup (same pattern as TokenizerManager)
        context = zmq.asyncio.Context(2)

        # Socket for receiving outputs from scheduler
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, bind=True
        )

        # Socket for sending requests to scheduler
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, bind=True
        )

        # State Management (from TokenizerManager)
        self.rid_to_state: Dict[str, GrpcReqState] = {}
        self.asyncio_tasks: set = set()
        self.gracefully_exit = False
        self.no_create_loop = False
        self.event_loop = None

        # Pause/Resume Control
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

        # Metrics
        self.request_counter = 0
        self.request_counter_lock = asyncio.Lock()
        self.last_receive_tstamp = time.time()

        # Crash dump for debugging
        self.crash_dump_request_list = []
        self.crash_dump_performed = False

        logger.info(
            f"GrpcRequestManager initialized with ZMQ IPC: "
            f"recv={port_args.detokenizer_ipc_name}, "
            f"send={port_args.scheduler_input_ipc_name}"
        )

    async def generate_request(
        self,
        obj: TokenizedGenerateReqInput,
        request_id: Optional[str] = None,
        grpc_context: Optional[grpc.aio.ServicerContext] = None,
    ) -> asyncio.Queue:
        """
        Submit a generation request to the scheduler.
        Returns a queue for streaming outputs.
        """
        # Generate request ID if not provided
        if request_id is None:
            async with self.request_counter_lock:
                request_id = f"grpc-{self.request_counter}"
                self.request_counter += 1

        obj.rid = request_id

        # TODO: support log_request

        # Create request state
        state = GrpcReqState(
            request_id=request_id,
            grpc_context=grpc_context,
            out_queue=asyncio.Queue(),
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=time.time(),
        )

        # Track session if needed
        if hasattr(obj, "session_params") and obj.session_params:
            state.session_id = obj.session_params.session_id
            state.is_session_request = True

        # Register state
        self.rid_to_state[request_id] = state
        self.record_request_for_crash_dump(obj)

        # Send to scheduler via ZMQ
        try:
            await self._send_to_scheduler(obj)
        except Exception as e:
            # Clean up on failure
            del self.rid_to_state[request_id]
            raise RuntimeError(f"Failed to send request to scheduler: {e}")

        return state.out_queue

    async def embedding_request(
        self,
        obj: TokenizedEmbeddingReqInput,
        request_id: Optional[str] = None,
    ) -> asyncio.Future:
        """
        Submit an embedding request to the scheduler.
        Returns a future that will contain the embedding result.
        """
        # Generate request ID if not provided
        if request_id is None:
            async with self.request_counter_lock:
                request_id = f"grpc-embed-{self.request_counter}"
                self.request_counter += 1

        obj.rid = request_id

        # Create request state
        state = GrpcReqState(
            request_id=request_id,
            grpc_context=None,
            out_queue=asyncio.Queue(),
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=time.time(),
        )

        # Register state
        self.rid_to_state[request_id] = state

        # Create future for result
        future = asyncio.Future()

        # Send to scheduler
        try:
            await self._send_to_scheduler(obj)
        except Exception as e:
            del self.rid_to_state[request_id]
            future.set_exception(e)
            return future

        # Wait for result in background
        async def wait_for_result():
            try:
                # Wait for completion
                await state.event.wait()
                # Get result from queue
                result = await state.out_queue.get()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                # Clean up
                if request_id in self.rid_to_state:
                    del self.rid_to_state[request_id]

        asyncio.create_task(wait_for_result())
        return future

    async def abort_request(self, request_id: str) -> bool:
        """Abort a running request."""
        if request_id not in self.rid_to_state:
            return False

        # Send abort to scheduler
        abort_req = AbortReq(rid=request_id)
        try:
            await self._send_to_scheduler(abort_req)
        except Exception as e:
            logger.error(f"Failed to send abort request: {e}")
            return False

        # Mark as finished
        state = self.rid_to_state.get(request_id)
        if state:
            state.finished = True
            state.stream_finished = True
            state.event.set()

            # Send abort notification to output queue
            await state.out_queue.put({"error": "Request aborted", "abort": True})

        return True

    async def pause_generation(self):
        """Pause generation processing."""
        async with self.is_pause_cond:
            self.is_pause = True
            logger.info("Generation paused")

    async def resume_generation(self):
        """Resume generation processing."""
        async with self.is_pause_cond:
            self.is_pause = False
            self.is_pause_cond.notify_all()
            logger.info("Generation resumed")

    async def handle_loop(self):
        """
        Main event loop - processes outputs from scheduler.
        Mimics TokenizerManager's handle_loop.
        """
        while not self.gracefully_exit:
            try:
                # Receive from scheduler
                recv_obj = await self.recv_from_scheduler.recv_pyobj()
                self.last_receive_tstamp = time.time()

                # Check for pause
                async with self.is_pause_cond:
                    while self.is_pause:
                        await self.is_pause_cond.wait()

                # Handle different output types
                if isinstance(recv_obj, BatchTokenIDOut):
                    await self._handle_batch_output(recv_obj)
                elif isinstance(recv_obj, BatchEmbeddingOut):
                    await self._handle_embedding_output(recv_obj)
                elif isinstance(recv_obj, HealthCheckOutput):
                    await self._handle_health_check_output(recv_obj)
                else:
                    logger.warning(f"Unknown output type: {type(recv_obj)}")

            except zmq.error.Again:
                # Timeout, check if we should exit
                if self.gracefully_exit:
                    break
                continue
            except Exception as e:
                logger.error(f"Handle loop error: {e}\n{get_exception_traceback()}")
                if self.gracefully_exit:
                    break

    async def _handle_batch_output(self, batch_out: BatchTokenIDOut):
        """Handle batch generation output from scheduler."""
        # Process each request in the batch
        for i, rid in enumerate(batch_out.rids):
            if rid not in self.rid_to_state:
                continue

            state = self.rid_to_state[rid]

            # Update metrics
            now = time.time()
            if state.first_token_time == 0.0:
                state.first_token_time = now
            state.last_time = now

            # Extract output for this request
            output_data = {
                "request_id": rid,
                "text": batch_out.decoded_texts[i] if batch_out.decoded_texts else "",
                "token_ids": batch_out.output_ids[i] if batch_out.output_ids else [],
                "finished": batch_out.finished_reasons[i] is not None,
                "meta_info": {
                    "prompt_tokens": (
                        batch_out.prompt_tokens[i] if batch_out.prompt_tokens else 0
                    ),
                    "completion_tokens": (
                        batch_out.completion_tokens[i]
                        if batch_out.completion_tokens
                        else 0
                    ),
                    "finish_reason": (
                        str(batch_out.finished_reasons[i])
                        if batch_out.finished_reasons[i]
                        else None
                    ),
                },
            }

            # Add logprobs if available
            if batch_out.output_token_logprobs_val and i < len(
                batch_out.output_token_logprobs_val
            ):
                output_data["logprobs"] = {
                    "tokens": batch_out.output_token_logprobs_val[i],
                    "top_logprobs": (
                        batch_out.output_top_logprobs_val[i]
                        if batch_out.output_top_logprobs_val
                        and i < len(batch_out.output_top_logprobs_val)
                        else None
                    ),
                }

            # Update state
            if output_data["text"]:
                state.text += output_data["text"][state.last_output_offset :]
                state.last_output_offset = len(output_data["text"])

            if output_data["token_ids"]:
                state.output_ids.extend(output_data["token_ids"])

            # Send to output queue
            await state.out_queue.put(output_data)

            # Handle completion
            if output_data["finished"]:
                state.finished = True
                state.finished_time = now
                state.stream_finished = True
                state.event.set()

                # Remove from tracking after a delay
                async def cleanup():
                    await asyncio.sleep(5.0)
                    if rid in self.rid_to_state:
                        del self.rid_to_state[rid]

                asyncio.create_task(cleanup())

    async def _handle_embedding_output(self, batch_out: BatchEmbeddingOut):
        """Handle batch embedding output from scheduler."""
        for i, rid in enumerate(batch_out.rids):
            if rid not in self.rid_to_state:
                continue

            state = self.rid_to_state[rid]

            # Create result
            result = {
                "request_id": rid,
                "embedding": batch_out.embeddings[i],
                "prompt_tokens": (
                    batch_out.prompt_tokens[i] if batch_out.prompt_tokens else 0
                ),
                "finish_reason": (
                    batch_out.finish_reason[i] if batch_out.finish_reason else None
                ),
            }

            # Send result
            await state.out_queue.put(result)

            # Mark as finished
            state.finished = True
            state.finished_time = time.time()
            state.event.set()

    async def _handle_health_check_output(self, health_out: HealthCheckOutput):
        """Handle health check output from scheduler."""
        rid = health_out.rid

        if rid not in self.rid_to_state:
            logger.warning(f"Health check output for unknown request: {rid}")
            return

        state = self.rid_to_state[rid]

        # Create health check result
        result = {
            "request_id": rid,
            "healthy": True,  # If we got a response, scheduler is healthy
            "output_text": (
                health_out.output_str if hasattr(health_out, "output_str") else ""
            ),
            "finish_reason": (
                health_out.finish_reason
                if hasattr(health_out, "finish_reason")
                else "stop"
            ),
        }

        # Send result
        await state.out_queue.put(result)

        # Mark as finished
        state.finished = True
        state.finished_time = time.time()
        state.event.set()

    async def _send_to_scheduler(self, obj):
        """Send an object to the scheduler via ZMQ."""
        try:
            self.send_to_scheduler.send_pyobj(obj)
        except Exception as e:
            logger.error(f"Failed to send to scheduler: {e}")
            raise

    def record_request_for_crash_dump(self, obj):
        """Record request for potential crash dump."""
        if len(self.crash_dump_request_list) < 100:
            self.crash_dump_request_list.append(
                {
                    "time": time.time(),
                    "request_id": getattr(obj, "rid", "unknown"),
                    "type": type(obj).__name__,
                }
            )

    async def shutdown(self):
        """Gracefully shutdown the request manager."""
        logger.info("Shutting down GrpcRequestManager")
        self.gracefully_exit = True

        # Cancel all pending requests
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                await state.out_queue.put(
                    {"error": "Server shutting down", "shutdown": True}
                )
                state.finished = True
                state.event.set()

        # Wait for tasks to complete
        if self.asyncio_tasks:
            await asyncio.gather(*list(self.asyncio_tasks), return_exceptions=True)

        # Close ZMQ sockets
        self.recv_from_scheduler.close()
        self.send_to_scheduler.close()

        logger.info("GrpcRequestManager shutdown complete")

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information for health checks."""
        return {
            "active_requests": len(self.rid_to_state),
            "paused": self.is_pause,
            "last_receive_time": self.last_receive_tstamp,
        }

    def auto_create_handle_loop(self):
        """Automatically create and start the handle_loop task, matching TokenizerManager pattern."""
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = asyncio.get_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )

        self.event_loop = loop

        # We cannot add signal handler when the grpc manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = GrpcSignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )
        else:
            logger.warning(
                "Signal handler is not added because the grpc request manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "grpc request manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def sigterm_watchdog(self):
        """Watchdog to handle SIGTERM gracefully, matching TokenizerManager pattern."""
        while not self.gracefully_exit:
            await asyncio.sleep(1.0)


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"GrpcRequestManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, GrpcRequestManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)
