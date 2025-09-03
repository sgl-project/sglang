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
"""MultiTokenizerMixin is a class that provides nesscary methods for MultiTokenizerManager and DetokenizerManager."""
import asyncio
import dataclasses
import json
import logging
import multiprocessing as multiprocessing
import os
import sys
import threading
from multiprocessing import shared_memory
from typing import Dict

import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
    MultiTokenizerRegisterReq,
    MultiTokenizerWarpper,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager, _Communicator
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket, kill_process_tree
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class MultiTokenizerMixin:
    """Mixin class for MultiTokenizerManager and DetokenizerManager"""

    def create_sockets_mapping(self):
        if not hasattr(self, "tokenizer_mapping"):
            self.tokenizer_mapping = {}
        # Create ZMQ context if needed
        if not hasattr(self, "_zmq_context"):
            self._zmq_context = zmq.Context()

    def init_tokenizer_mapping(
        self, recv_obj: MultiTokenizerRegisterReq, worker_id: str
    ):
        """init tokenizer mapping from register request"""
        ipc_name = recv_obj.ipc_name
        worker_id_int = int(worker_id)

        if worker_id_int not in self.tokenizer_mapping:
            socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
            self.tokenizer_mapping[worker_id_int] = socket
            self.tokenizer_mapping[worker_id_int].send_pyobj(recv_obj)
            return True
        else:
            return False

    def register_tokenizer_ipc(self, recv_obj, worker_id):
        if worker_id not in self.tokenizer_mapping:
            # register the worker if not already done
            if isinstance(recv_obj, MultiTokenizerRegisterReq):
                return self.init_tokenizer_mapping(recv_obj, worker_id)
            else:
                logger.error(
                    f"Worker {worker_id} not registered and not found in tokenizer mapping . "
                    "Please ensure the worker is registered correctly."
                )
        return False

    def _handle_output_by_index(self, output, i):
        """NOTE: A maintainable method is better here."""
        if isinstance(output, BatchTokenIDOut):
            new_output = BatchTokenIDOut(
                rids=[output.rids[i]],
                finished_reasons=(
                    [output.finished_reasons[i]]
                    if len(output.finished_reasons) > i
                    else None
                ),
                decoded_texts=(
                    [output.decoded_texts[i]] if len(output.decoded_texts) > i else None
                ),
                decode_ids=(
                    [output.decode_ids[i]] if len(output.decode_ids) > i else None
                ),
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
                    [output.spec_verify_ct[i]]
                    if len(output.spec_verify_ct) > i
                    else None
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
            )
        elif isinstance(output, BatchEmbeddingOut):
            new_output = BatchEmbeddingOut(
                rids=[output.rids[i]],
                finished_reasons=(
                    [output.finished_reasons[i]]
                    if len(output.finished_reasons) > i
                    else None
                ),
                embeddings=(
                    [output.embeddings[i]] if len(output.embeddings) > i else None
                ),
                prompt_tokens=(
                    [output.prompt_tokens[i]] if len(output.prompt_tokens) > i else None
                ),
                cached_tokens=(
                    [output.cached_tokens[i]] if len(output.cached_tokens) > i else None
                ),
            )
        elif isinstance(output, BatchStrOut):
            new_output = BatchStrOut(
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
                    [output.spec_verify_ct[i]]
                    if len(output.spec_verify_ct) > i
                    else None
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
            )
        elif isinstance(output, BatchMultimodalOut):
            new_output = BatchMultimodalOut(
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
            )
        else:
            new_output = output
        return new_output

    def get_worker_ids_from_req_rids(self, rids):
        if isinstance(rids, list):
            worker_ids = [int(rid.split("_")[0]) for rid in rids]
        elif isinstance(rids, str):
            worker_ids = [int(rids.split("_")[0])]
        else:
            worker_ids = []
        return worker_ids

    def multi_tokenizer_manager_event_loop(self):
        """The event loop that handles requests, for multi tokenizer manager mode only"""
        self.create_sockets_mapping()
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
                    if self.register_tokenizer_ipc(recv_obj, worker_id):
                        logger.info(
                            f"DetokenizerManager Created ZMQ socket for worker {worker_id}"
                        )
                    continue
                else:
                    if worker_id not in self.tokenizer_mapping:
                        logger.error(
                            f"Tokenizer Worker ID {worker_id} not registered. Check if the server Process {worker_id} is alive"
                        )
                        continue
                    new_output = self._handle_output_by_index(output, i)
                    self.tokenizer_mapping[worker_id].send_pyobj(new_output)

    def clear_tokenizer_mapping(self):
        if hasattr(self, "tokenizer_mapping"):
            for socket in self.tokenizer_mapping.values():
                try:
                    socket.close()
                except Exception as e:
                    logger.warning(f"Failed to close socket: {e}")
            self.tokenizer_mapping.clear()


class MultiTokenizerRouter(TokenizerManager, MultiTokenizerMixin):
    """A router to receive requests from MultiTokenizerManager"""

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
        self.init_disaggregation()

    def _run_loop(self):
        self._loop.run_forever()

    async def router_worker_obj(self):
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()
            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        # special reqs will recv from scheduler, need to route to right worker
        self.create_sockets_mapping()
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        """Distribute result to corresponding workers based on rid"""
        if isinstance(recv_obj, MultiTokenizerWarpper):
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
                if self.register_tokenizer_ipc(recv_obj, worker_id):
                    logger.info(
                        f"MultiTokenizerRouter Created ZMQ socket for worker {worker_id}"
                    )
                continue
            else:
                if worker_id not in self.tokenizer_mapping:
                    logger.error(
                        f"Tokenizer Worker ID {worker_id} not registered. Check if the server Process {worker_id} is alive"
                    )
                    continue
                new_recv_obj = self._handle_output_by_index(recv_obj, i)
                self.tokenizer_mapping[worker_id].send_pyobj(new_recv_obj)


class MultiTokenizerManager(TokenizerManager, MultiTokenizerMixin):
    """Multi Process Tokenizer Manager that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
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


def serialize_port_args(port_args: PortArgs) -> dict:
    """Serialize PortArgs into a shareable dictionary"""
    return {
        "tokenizer_ipc_name": port_args.tokenizer_ipc_name,
        "scheduler_input_ipc_name": port_args.scheduler_input_ipc_name,
        "detokenizer_ipc_name": port_args.detokenizer_ipc_name,
        "nccl_port": port_args.nccl_port,
        "rpc_ipc_name": port_args.rpc_ipc_name,
        "metrics_ipc_name": port_args.metrics_ipc_name,
        "tokenizer_worker_ipc_name": port_args.tokenizer_worker_ipc_name,
    }


def deserialize_data(port_args: dict, server_args: dict):
    """Deserialize data from shared dictionaries"""
    return PortArgs(**port_args), ServerArgs(**server_args)


def serialize_server_args(server_args: ServerArgs) -> dict:
    """Serialize ServerArgs into a shareable dictionary"""
    return dataclasses.asdict(server_args)


def serialize_scheduler_info(scheduler_info: Dict) -> dict:
    """Serialize scheduler_info into a shareable dictionary"""
    return scheduler_info


def deserialize_scheduler_info(data: dict) -> Dict:
    """Deserialize scheduler_info from a shared dictionary"""
    return data


def write_to_shared_memory(data: dict, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = json.dumps(data).encode("utf-8")
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


def read_from_shared_memory(name: str) -> dict:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = json.loads(bytes(shm.buf).decode("utf-8"))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def get_main_process_id() -> int:
    """Get the main process ID"""
    return multiprocessing.current_process()._parent_pid


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args information to share memory for multi-tokenizer"""
    # get main process ID
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")

    # Write port_args to shared memory
    port_args_shm = write_to_shared_memory(
        serialize_port_args(port_args), f"port_args_{current_pid}"
    )
    # Write server_args to shared memory
    server_args_shm = write_to_shared_memory(
        serialize_server_args(server_args), f"server_args_{current_pid}"
    )
    # Write scheduler_info to shared memory
    scheduler_info_shm = write_to_shared_memory(
        serialize_scheduler_info(scheduler_info), f"scheduler_info_{current_pid}"
    )

    port_args_shm.close()
    server_args_shm.close()
    scheduler_info_shm.close()

    return port_args_shm, server_args_shm, scheduler_info_shm
