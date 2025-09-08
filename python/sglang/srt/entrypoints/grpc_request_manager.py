"""
gRPC Request Manager - Orchestrates request lifecycle without tokenization.
Mimics TokenizerManager's state management and ZMQ communication patterns.
"""

import asyncio
import dataclasses
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Union

import grpc
import zmq
import zmq.asyncio

from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    FlushCacheReq,
    OpenSessionReq,
    OpenSessionReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_exception_traceback, get_zmq_socket
from sglang.utils import get_ulimit_value

logger = logging.getLogger(__name__)


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
        port_args: Dict[str, int],
    ):
        """Initialize the gRPC request manager."""
        self.server_args = server_args
        self.port_args = port_args
        
        # ZMQ Communication Setup (same pattern as TokenizerManager)
        context = zmq.asyncio.Context(get_ulimit_value())
        
        # Socket for receiving outputs from scheduler
        self.recv_from_scheduler = get_zmq_socket(
            context,
            zmq.PULL,
            f"tcp://127.0.0.1:{port_args['scheduler_output_port']}",
            bind=False
        )
        
        # Socket for sending requests to scheduler
        self.send_to_scheduler = get_zmq_socket(
            context,
            zmq.PUSH,
            f"tcp://127.0.0.1:{port_args['scheduler_input_port']}",
            bind=False
        )
        
        # State Management (from TokenizerManager)
        self.rid_to_state: Dict[str, GrpcReqState] = {}
        self.asyncio_tasks: List[asyncio.Task] = []
        self.gracefully_exit = False
        
        # Session Management
        self.session_futures: Dict[str, asyncio.Future] = {}
        
        # Pause/Resume Control
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        
        # Model Update Synchronization
        self.model_update_lock = asyncio.Lock()
        self.model_update_result: Optional[UpdateWeightReqOutput] = None
        
        # LoRA Management
        self.lora_update_lock = asyncio.Lock()
        
        # Metrics
        self.request_counter = 0
        self.request_counter_lock = asyncio.Lock()
        self.last_receive_tstamp = time.time()
        
        # Crash dump for debugging
        self.crash_dump_request_list = []
        self.crash_dump_performed = False
        
        logger.info(
            f"GrpcRequestManager initialized with ZMQ ports: "
            f"recv={port_args['scheduler_output_port']}, "
            f"send={port_args['scheduler_input_port']}"
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
        if hasattr(obj, 'session_params') and obj.session_params:
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
            await state.out_queue.put({
                "error": "Request aborted",
                "abort": True
            })
        
        return True
    
    async def flush_cache(self) -> Dict[str, Any]:
        """Flush the KV cache."""
        flush_req = FlushCacheReq()
        await self._send_to_scheduler(flush_req)
        return {"success": True}
    
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
    
    async def open_session(
        self,
        session_id: str,
        capacity: int,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Open a new session."""
        req = OpenSessionReq(
            session_id=session_id,
            capacity=capacity,
        )
        
        # Create future for result
        future = asyncio.Future()
        self.session_futures[session_id] = future
        
        # Send request
        await self._send_to_scheduler(req)
        
        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            del self.session_futures[session_id]
            raise RuntimeError(f"Session open timeout for {session_id}")
    
    async def close_session(self, session_id: str) -> bool:
        """Close an existing session."""
        # Remove from session futures if present
        if session_id in self.session_futures:
            del self.session_futures[session_id]
        
        # TODO: Send close session request to scheduler
        return True
    
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
                elif isinstance(recv_obj, OpenSessionReqOutput):
                    await self._handle_session_output(recv_obj)
                elif isinstance(recv_obj, UpdateWeightReqOutput):
                    await self._handle_update_weights_output(recv_obj)
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
                "text": batch_out.output_strs[i] if batch_out.output_strs else "",
                "token_ids": batch_out.output_ids[i] if batch_out.output_ids else [],
                "finished": batch_out.finished_reason[i] is not None,
                "meta_info": batch_out.meta_info[i] if batch_out.meta_info else {},
            }
            
            # Add logprobs if available
            if batch_out.output_token_logprobs:
                output_data["logprobs"] = {
                    "tokens": batch_out.output_token_logprobs[i],
                    "top_logprobs": batch_out.output_top_logprobs[i]
                    if batch_out.output_top_logprobs else None
                }
            
            # Update state
            if output_data["text"]:
                state.text += output_data["text"][state.last_output_offset:]
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
                "prompt_tokens": batch_out.prompt_tokens[i] if batch_out.prompt_tokens else 0,
                "finish_reason": batch_out.finish_reason[i] if batch_out.finish_reason else None,
            }
            
            # Send result
            await state.out_queue.put(result)
            
            # Mark as finished
            state.finished = True
            state.finished_time = time.time()
            state.event.set()
    
    async def _handle_session_output(self, output: OpenSessionReqOutput):
        """Handle session open output."""
        session_id = output.session_id
        if session_id in self.session_futures:
            future = self.session_futures.pop(session_id)
            if output.success:
                future.set_result({
                    "session_id": session_id,
                    "success": True
                })
            else:
                future.set_exception(RuntimeError(f"Failed to open session: {output.error}"))
    
    async def _handle_update_weights_output(self, output: UpdateWeightReqOutput):
        """Handle model weights update output."""
        async with self.model_update_lock:
            self.model_update_result = output
    
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
            self.crash_dump_request_list.append({
                "time": time.time(),
                "request_id": getattr(obj, 'rid', 'unknown'),
                "type": type(obj).__name__,
            })
    
    async def shutdown(self):
        """Gracefully shutdown the request manager."""
        logger.info("Shutting down GrpcRequestManager")
        self.gracefully_exit = True
        
        # Cancel all pending requests
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                await state.out_queue.put({
                    "error": "Server shutting down",
                    "shutdown": True
                })
                state.finished = True
                state.event.set()
        
        # Wait for tasks to complete
        if self.asyncio_tasks:
            await asyncio.gather(*self.asyncio_tasks, return_exceptions=True)
        
        # Close ZMQ sockets
        self.recv_from_scheduler.close()
        self.send_to_scheduler.close()
        
        logger.info("GrpcRequestManager shutdown complete")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information for health checks."""
        return {
            "active_requests": len(self.rid_to_state),
            "paused": self.is_pause,
            "active_sessions": len(self.session_futures),
            "last_receive_time": self.last_receive_tstamp,
        }