from __future__ import annotations

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

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
"""

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import sys
import threading
from functools import partialmethod
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ContinueGenerationReqInput,
    PauseContinueBroadcast,
    PauseGenerationReqInput,
    TokenizerWorkerRegisterReq,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any):
        if ipc_name is None:
            # Some unhandled cases
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=False)
        self._mapping[ipc_name].send_pyobj(output)


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True
) -> Any:
    """Extract a field value from output by index, handling None and length checks.

    Args:
        output: The output object containing the field
        field_name: The name of the field to extract
        index: The index to access in the field list
        check_length: If True, check both field existence and length. If False, only check field existence.

    Returns:
        A list containing the field value at index, or None if not available.
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    if isinstance(field, dict):
        new_field = {}
        for k, v in field.items():
            if len(v) <= index:
                new_field[k] = None
            new_field[k] = v[index]
        return new_field

    if check_length:
        if len(field) <= index:
            return None

    return [field[index]]


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here."""
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_drafts=_extract_field_by_index(
                output, "spec_accepted_drafts", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),
            decode_ids=_extract_field_by_index(output, "decode_ids", i),
            read_offsets=_extract_field_by_index(output, "read_offsets", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            skip_special_tokens=_extract_field_by_index(
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            embeddings=_extract_field_by_index(output, "embeddings", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_accepted_drafts=_extract_field_by_index(
                output, "spec_accepted_drafts", i
            ),
            spec_acceptance_histogram=_extract_field_by_index(
                output, "spec_acceptance_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            output_strs=_extract_field_by_index(output, "output_strs", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(
                output, "indexer_topk", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is None:
                continue

            assert isinstance(
                recv_obj, BaseBatchReq
            ), "for multi-http-worker, recv_obj must be BaseBatchReq"

            # Send data using the corresponding socket
            for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):
                new_output = _handle_output_by_index(output, i)
                self.socket_mapping.send_output(ipc_name, new_output)


class MultiTokenizerRouter:
    """A router between tokenizer managers and the scheduler/detokenizer manager.

    Forward: tokenizer managers → router → scheduler.
    Backward: detokenizer manager → router → tokenizer managers.
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args
        context = zmq.asyncio.Context(3)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

        # Worker IPC names for pause/continue broadcasting
        self.all_worker_ipcs: set[str] = set()
        # Shared socket mapping (both coroutines run on self._loop, so safe)
        self.socket_mapping = SocketMapping()

    def _run_loop(self):
        self._loop.run_forever()

    async def router_worker_obj(self):
        """Forward path: workers → scheduler, with pause/continue broadcast."""
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()

            if isinstance(recv_obj, TokenizerWorkerRegisterReq):
                if recv_obj.worker_ipc_name not in self.all_worker_ipcs:
                    self.all_worker_ipcs.add(recv_obj.worker_ipc_name)
                    logger.info(
                        f"Router registered worker IPC: {recv_obj.worker_ipc_name} "
                        f"(total: {len(self.all_worker_ipcs)})"
                    )
                continue

            if isinstance(
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                # Broadcast to ALL workers so every worker's is_pause is set
                is_pause = isinstance(recv_obj, PauseGenerationReqInput)
                broadcast = PauseContinueBroadcast(is_pause=is_pause)
                for ipc_name in self.all_worker_ipcs:
                    self.socket_mapping.send_output(ipc_name, broadcast)
                # Forward to scheduler rank 0 (it broadcasts to all TP/PP/DP
                # ranks internally). Skip for abort mode which drains via polling.
                if not (
                    isinstance(recv_obj, PauseGenerationReqInput)
                    and recv_obj.mode == "abort"
                ):
                    await self.send_to_scheduler.send_pyobj(recv_obj)
                continue

            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        """Backward path: detokenizer → route results to correct worker."""
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        # prevent init prefill bootstrapserver again
        disaggregation_mode = server_args.disaggregation_mode
        server_args.disaggregation_mode = "null"
        super().__init__(server_args, port_args)

        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        # For PD disaggregtion
        self.server_args.disaggregation_mode = disaggregation_mode
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.disaggregation_transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        # Communicator
        self.register_multi_tokenizer_communicator = FanOutCommunicator(
            self.send_to_scheduler, 2
        )

        # Register this worker with the router for pause/continue broadcasting
        reg = TokenizerWorkerRegisterReq(worker_ipc_name=self.tokenizer_ipc_name)
        self.send_to_scheduler.send_pyobj(reg)

        # Future for awaiting pause/continue broadcast confirmation
        self._pause_continue_future: Optional[asyncio.Future] = None

        # Register PauseContinueBroadcast in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        from sglang.utils import TypeBasedDispatcher

        self._result_dispatcher += TypeBasedDispatcher(
            [(PauseContinueBroadcast, self._handle_pause_continue_broadcast)]
        )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        # Send to router which will broadcast to all workers
        # (router also handles forwarding to scheduler for non-abort modes)
        self.send_to_scheduler.send_pyobj(obj)
        await self._pause_continue_future

        if obj.mode == "abort":
            # Abort polling: only the originator checks its own lock state
            while True:
                self.abort_request(abort_all=True)
                is_locked = await self.model_update_lock.is_locked()
                if not is_locked:
                    break
                await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        self.send_to_scheduler.send_pyobj(obj)
        await self._pause_continue_future

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcast):
        """Called from handle_loop when a broadcast arrives from the router."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_pause_continue_broadcast(obj))

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcast):
        """Apply pause/continue state under the condition lock."""
        async with self.is_pause_cond:
            if obj.is_pause:
                self.is_pause = True
            else:
                self.is_pause = False
                self.is_pause_cond.notify_all()

        # Resolve the pending future if this worker initiated the pause/continue
        if self._pause_continue_future and not self._pause_continue_future.done():
            self._pause_continue_future.set_result(True)
            self._pause_continue_future = None

    def _attach_multi_http_worker_info(self, req: Union[BaseReq, BaseBatchReq]):

        if isinstance(req, BaseReq):
            req.http_worker_ipc = self.tokenizer_ipc_name
        elif isinstance(req, BaseBatchReq):
            req.http_worker_ipcs = [self.tokenizer_ipc_name] * len(req.rids)
        else:
            raise ValueError(f"Unknown req type: {type(req)}")


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID.

    Supports override via SGLANG_GRANIAN_PARENT_PID for workers whose
    multiprocessing parent PID differs from the shared-memory owner.
    """
    from sglang.srt.environ import envs

    override = envs.SGLANG_GRANIAN_PARENT_PID.get()
    if override is not None:
        return override
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args information to share memory for multi-tokenizer"""
    # get main process ID
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    return args_shm


def monkey_patch_uvicorn_multiprocessing(timeout: float = 10):
    """Monkey patch uvicorn multiprocessing is_alive timeout"""
    # from default 5s -> 10s
    try:
        from uvicorn.supervisors.multiprocess import Process

        Process.is_alive = partialmethod(Process.is_alive, timeout=timeout)

    except ImportError:
        logger.warning(
            "uvicorn.supervisors.multiprocess not found, skipping monkey patch"
        )


class SenderWrapper:
    def __init__(self, port_args: PortArgs, send_to_scheduler: zmq.Socket):
        self.port_args = port_args
        self.send_to_scheduler = send_to_scheduler

    def send_pyobj(self, obj):
        if isinstance(obj, BaseReq):
            obj.http_worker_ipc = self.port_args.tokenizer_ipc_name
        self.send_to_scheduler.send_pyobj(obj)
