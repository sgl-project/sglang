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
"""Mixin class and utils for multi-http-worker mode"""
import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import sys
import threading
from functools import partialmethod
from multiprocessing import shared_memory
from typing import Any, Dict

import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchMultimodalOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    MultiTokenizerRegisterReq,
    MultiTokenizerWrapper,
)
from sglang.srt.managers.tokenizer_communicator_mixin import _Communicator
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket, kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def register_ipc_mapping(
        self, recv_obj: MultiTokenizerRegisterReq, worker_id: str, is_tokenizer: bool
    ):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if worker_id in self._mapping:
            logger.warning(
                f"{type_str} already registered with worker {worker_id}, skipping..."
            )
            return
        logger.info(
            f"{type_str} not registered with worker {worker_id}, registering..."
        )
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, recv_obj.ipc_name, False)
        self._mapping[worker_id] = socket
        self._mapping[worker_id].send_pyobj(recv_obj)

    def send_output(self, worker_id: str, output: Any):
        if worker_id not in self._mapping:
            logger.error(
                f"worker ID {worker_id} not registered. Check if the server Process is alive"
            )
            return
        self._mapping[worker_id].send_pyobj(output)


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here."""
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            finished_reasons=(
                [output.finished_reasons[i]]
                if len(output.finished_reasons) > i
                else None
            ),
            decoded_texts=(
                [output.decoded_texts[i]] if len(output.decoded_texts) > i else None
            ),
            decode_ids=([output.decode_ids[i]] if len(output.decode_ids) > i else None),
            read_offsets=(
                [output.read_offsets[i]] if len(output.read_offsets) > i else None
            ),
            output_ids=(
                [output.output_ids[i]]
                if output.output_ids and len(output.output_ids) > i
                else None
            ),
            skip_special_tokens=(
                [output.skip_special_tokens[i]]
                if len(output.skip_special_tokens) > i
                else None
            ),
            spaces_between_special_tokens=(
                [output.spaces_between_special_tokens[i]]
                if len(output.spaces_between_special_tokens) > i
                else None
            ),
            no_stop_trim=(
                [output.no_stop_trim[i]] if len(output.no_stop_trim) > i else None
            ),
            prompt_tokens=(
                [output.prompt_tokens[i]] if len(output.prompt_tokens) > i else None
            ),
            completion_tokens=(
                [output.completion_tokens[i]]
                if len(output.completion_tokens) > i
                else None
            ),
            cached_tokens=(
                [output.cached_tokens[i]] if len(output.cached_tokens) > i else None
            ),
            spec_verify_ct=(
                [output.spec_verify_ct[i]] if len(output.spec_verify_ct) > i else None
            ),
            input_token_logprobs_val=(
                [output.input_token_logprobs_val[i]]
                if output.input_token_logprobs_val
                else None
            ),
            input_token_logprobs_idx=(
                [output.input_token_logprobs_idx[i]]
                if output.input_token_logprobs_idx
                else None
            ),
            output_token_logprobs_val=(
                [output.output_token_logprobs_val[i]]
                if output.output_token_logprobs_val
                else None
            ),
            output_token_logprobs_idx=(
                [output.output_token_logprobs_idx[i]]
                if output.output_token_logprobs_idx
                else None
            ),
            input_top_logprobs_val=(
                [output.input_top_logprobs_val[i]]
                if output.input_top_logprobs_val
                else None
            ),
            input_top_logprobs_idx=(
                [output.input_top_logprobs_idx[i]]
                if output.input_top_logprobs_idx
                else None
            ),
            output_top_logprobs_val=(
                [output.output_top_logprobs_val[i]]
                if output.output_top_logprobs_val
                else None
            ),
            output_top_logprobs_idx=(
                [output.output_top_logprobs_idx[i]]
                if output.output_top_logprobs_idx
                else None
            ),
            input_token_ids_logprobs_val=(
                [output.input_token_ids_logprobs_val[i]]
                if output.input_token_ids_logprobs_val
                else None
            ),
            input_token_ids_logprobs_idx=(
                [output.input_token_ids_logprobs_idx[i]]
                if output.input_token_ids_logprobs_idx
                else None
            ),
            output_token_ids_logprobs_val=(
                [output.output_token_ids_logprobs_val[i]]
                if output.output_token_ids_logprobs_val
                else None
            ),
            output_token_ids_logprobs_idx=(
                [output.output_token_ids_logprobs_idx[i]]
                if output.output_token_ids_logprobs_idx
                else None
            ),
            output_hidden_states=(
                [output.output_hidden_states[i]]
                if output.output_hidden_states
                else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=(
                [output.finished_reasons[i]]
                if len(output.finished_reasons) > i
                else None
            ),
            embeddings=([output.embeddings[i]] if len(output.embeddings) > i else None),
            prompt_tokens=(
                [output.prompt_tokens[i]] if len(output.prompt_tokens) > i else None
            ),
            cached_tokens=(
                [output.cached_tokens[i]] if len(output.cached_tokens) > i else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            finished_reasons=(
                [output.finished_reasons[i]]
                if len(output.finished_reasons) > i
                else None
            ),
            output_strs=(
                [output.output_strs[i]] if len(output.output_strs) > i else None
            ),
            output_ids=(
                [output.output_ids[i]]
                if output.output_ids and len(output.output_ids) > i
                else None
            ),
            prompt_tokens=(
                [output.prompt_tokens[i]] if len(output.prompt_tokens) > i else None
            ),
            completion_tokens=(
                [output.completion_tokens[i]]
                if len(output.completion_tokens) > i
                else None
            ),
            cached_tokens=(
                [output.cached_tokens[i]] if len(output.cached_tokens) > i else None
            ),
            spec_verify_ct=(
                [output.spec_verify_ct[i]] if len(output.spec_verify_ct) > i else None
            ),
            input_token_logprobs_val=(
                [output.input_token_logprobs_val[i]]
                if output.input_token_logprobs_val
                else None
            ),
            input_token_logprobs_idx=(
                [output.input_token_logprobs_idx[i]]
                if output.input_token_logprobs_idx
                else None
            ),
            output_token_logprobs_val=(
                [output.output_token_logprobs_val[i]]
                if output.output_token_logprobs_val
                else None
            ),
            output_token_logprobs_idx=(
                [output.output_token_logprobs_idx[i]]
                if output.output_token_logprobs_idx
                else None
            ),
            input_top_logprobs_val=(
                [output.input_top_logprobs_val[i]]
                if output.input_top_logprobs_val
                else None
            ),
            input_top_logprobs_idx=(
                [output.input_top_logprobs_idx[i]]
                if output.input_top_logprobs_idx
                else None
            ),
            output_top_logprobs_val=(
                [output.output_top_logprobs_val[i]]
                if output.output_top_logprobs_val
                else None
            ),
            output_top_logprobs_idx=(
                [output.output_top_logprobs_idx[i]]
                if output.output_top_logprobs_idx
                else None
            ),
            input_token_ids_logprobs_val=(
                [output.input_token_ids_logprobs_val[i]]
                if output.input_token_ids_logprobs_val
                else None
            ),
            input_token_ids_logprobs_idx=(
                [output.input_token_ids_logprobs_idx[i]]
                if output.input_token_ids_logprobs_idx
                else None
            ),
            output_token_ids_logprobs_val=(
                [output.output_token_ids_logprobs_val[i]]
                if output.output_token_ids_logprobs_val
                else None
            ),
            output_token_ids_logprobs_idx=(
                [output.output_token_ids_logprobs_idx[i]]
                if output.output_token_ids_logprobs_idx
                else None
            ),
            output_hidden_states=(
                [output.output_hidden_states[i]]
                if output.output_hidden_states
                else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchMultimodalOutput):
        new_output = BatchMultimodalOutput(
            rids=[output.rids[i]],
            finished_reasons=(
                [output.finished_reasons[i]]
                if len(output.finished_reasons) > i
                else None
            ),
            outputs=([output.outputs[i]] if len(output.outputs) > i else None),
            prompt_tokens=(
                [output.prompt_tokens[i]] if len(output.prompt_tokens) > i else None
            ),
            completion_tokens=(
                [output.completion_tokens[i]]
                if len(output.completion_tokens) > i
                else None
            ),
            cached_tokens=(
                [output.cached_tokens[i]] if len(output.cached_tokens) > i else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def get_worker_ids_from_req_rids(self, rids):
        if isinstance(rids, list):
            worker_ids = [int(rid.split("_")[0]) for rid in rids]
        elif isinstance(rids, str):
            worker_ids = [int(rids.split("_")[0])]
        else:
            worker_ids = []
        return worker_ids

    def maybe_clear_socket_mapping(self):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is None:
                continue
            # Extract worker_id from rid
            if isinstance(recv_obj.rids, list):
                worker_ids = self.get_worker_ids_from_req_rids(recv_obj.rids)
            else:
                raise RuntimeError(
                    f"for tokenizer_worker_num > 1, recv_obj.rids must be a list"
                )

            # Send data using the corresponding socket
            for i, worker_id in enumerate(worker_ids):
                if isinstance(recv_obj, MultiTokenizerRegisterReq):
                    self.socket_mapping.register_ipc_mapping(
                        recv_obj, worker_id, is_tokenizer=False
                    )
                else:
                    new_output = _handle_output_by_index(output, i)
                    self.socket_mapping.send_output(worker_id, new_output)


class MultiTokenizerRouter:
    """A router to receive requests from TokenizerWorker"""

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
        # Start handle_loop simultaneously
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

    def _run_loop(self):
        self._loop.run_forever()

    async def router_worker_obj(self):
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()
            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        # special reqs will recv from scheduler, need to route to right worker
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        """Distribute result to corresponding workers based on rid"""
        if isinstance(recv_obj, MultiTokenizerWrapper):
            worker_ids = [recv_obj.worker_id]
            recv_obj = recv_obj.obj
        else:
            worker_ids = self.get_worker_ids_from_req_rids(recv_obj.rids)

        if len(worker_ids) == 0:
            logger.error(f"Cannot find worker_id from rids {recv_obj.rids}")
            return

        # Distribute result to each worker
        for i, worker_id in enumerate(worker_ids):
            if isinstance(recv_obj, MultiTokenizerRegisterReq):
                self.socket_mapping.register_ipc_mapping(
                    recv_obj, worker_id, is_tokenizer=True
                )
            else:
                new_recv_obj = _handle_output_by_index(recv_obj, i)
                self.socket_mapping.send_output(worker_id, new_recv_obj)


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
        self.register_multi_tokenizer_communicator = _Communicator(
            self.send_to_scheduler, 2
        )
        self._result_dispatcher._mapping.append(
            (
                MultiTokenizerRegisterReq,
                self.register_multi_tokenizer_communicator.handle_recv,
            )
        )

    async def register_to_main_tokenizer_manager(self):
        """Register this worker to the main TokenizerManager"""
        # create a handle loop to receive messages from the main TokenizerManager
        self.auto_create_handle_loop()
        req = MultiTokenizerRegisterReq(rids=[f"{self.worker_id}_register"])
        req.ipc_name = self.tokenizer_ipc_name
        _Communicator.enable_multi_tokenizer = True
        await self.register_multi_tokenizer_communicator(req)


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
    """Get the main process ID"""
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
