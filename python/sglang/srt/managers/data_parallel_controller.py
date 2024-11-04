"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
import multiprocessing.connection
from enum import Enum, auto

import zmq

from sglang.srt.managers.io_struct import (
    ControllerInfo,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_parent_process,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

import random


# for pre radix scheduler
def _key_match(key0, key1):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def get_match_len(node, key, match_length: int) -> int:
    if len(key) == 0:
        return match_length

    if key[0] in node.children.keys():
        child = node.children[key[0]]
        prefix_len = _key_match(child.key, key)
        match_length += prefix_len
        if prefix_len < len(child.key):
            return match_length
        else:
            return get_match_len(child, key[prefix_len:], match_length)
    else:
        return match_length


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    RESOURCES_AWARE = auto()
    PRE_RADIX = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args, port_args) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.recv_from_tokenizer = get_zmq_socket(
            self.context, zmq.PULL, port_args.scheduler_input_ipc_name
        )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.RESOURCES_AWARE: self.resources_aware_scheduler,
            LoadBalanceMethod.PRE_RADIX: self.pre_radix_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # For resources aware
        self.dp_size = server_args.dp_size
        self.controller_info = ControllerInfo(server_args.dp_size)
        self.pre_available_kv_cache = []
        self.main_available_kv_cache = []

        self.pre_num_running_req = []
        self.main_num_running_req = []

        self.pre_num_waiting_req = []
        self.main_num_waiting_req = []

        # For pre_radix
        self.pre_raidx = server_args.load_balance_method == "pre_radix"

        # Start data parallel workers
        base_gpu_id = 0
        self.workers = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            send_to = self.launch_tensor_parallel_group(
                server_args, tmp_port_args, base_gpu_id, dp_rank, self.controller_info
            )

            self.workers.append(send_to)
            base_gpu_id += server_args.tp_size

        if self.pre_raidx:
            import threading

            self.newest_tree_cache = {}

            self.recv_tree_cache_lock = threading.Lock()
            self.recv_tree_cache_thread = threading.Thread(
                target=self.loop_for_recv_tree_cache
            )
        else:
            self.newest_tree_cache = None
            self.recv_tree_cache_thread = None

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        controller_info: ControllerInfo,
    ):
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = base_gpu_id + tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    dp_rank,
                    writer,
                    controller_info,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        send_to = get_zmq_socket(
            self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
        )

        # Wait for model to finish loading
        for i in range(len(scheduler_pipe_readers)):
            scheduler_pipe_readers[i].recv()

        return send_to

    def loop_for_recv_tree_cache(self):
        while True:
            self.recv_tree_cache()

    def recv_tree_cache(self):
        while True:
            recv_radix_cache = self.controller_info.radix_queue.get()
            if recv_radix_cache:
                # logger.info('[recv_tree_cache] receive new data')
                gpu_id = recv_radix_cache.gpu_id
                if (
                    gpu_id not in self.newest_tree_cache
                    or recv_radix_cache.time > self.newest_tree_cache[gpu_id].time
                ):
                    with self.recv_tree_cache_lock:
                        if gpu_id in self.newest_tree_cache:
                            del self.newest_tree_cache[gpu_id]
                        self.newest_tree_cache[gpu_id] = recv_radix_cache
                del recv_radix_cache

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    def update_memory_and_requests(self):
        available_mem = [k.value for k in self.controller_info.available_kv_cache]
        num_reqs_running = [k.value for k in self.controller_info.running_reqs]
        num_reqs_waiting = [k.value for k in self.controller_info.waiting_reqs]

        if not self.pre_available_kv_cache:
            self.pre_available_kv_cache = available_mem.copy()
        if not self.main_available_kv_cache:
            self.main_available_kv_cache = available_mem.copy()
        if self.pre_available_kv_cache != available_mem:
            self.pre_available_kv_cache = available_mem
            self.main_available_kv_cache = available_mem.copy()

        if not self.pre_num_running_req:
            self.pre_num_running_req = num_reqs_running.copy()
        if not self.main_num_running_req:
            self.main_num_running_req = num_reqs_running.copy()
        if self.pre_num_running_req != num_reqs_running:
            self.main_num_running_req = num_reqs_running
            self.pre_num_running_req = num_reqs_running.copy()

        if not self.pre_num_waiting_req:
            self.pre_num_waiting_req = num_reqs_waiting.copy()
        if not self.main_num_waiting_req:
            self.main_num_waiting_req = num_reqs_waiting.copy()
        if self.pre_num_waiting_req != num_reqs_waiting:
            self.main_num_waiting_req = num_reqs_waiting
            self.pre_num_waiting_req = num_reqs_waiting.copy()

    def allocate_gpu(self, req):
        all_waiting = min(self.main_num_waiting_req) > 0
        no_waiting = [1 if waiting == 0 else 0 for waiting in self.main_num_waiting_req]

        if all_waiting:
            ratio = [
                run / wait
                for run, wait in zip(
                    self.main_num_running_req, self.main_num_waiting_req
                )
            ]
            max_ratio = max(ratio)
            indices = [i for i, x in enumerate(ratio) if x == max_ratio]
            gpu_idx = random.choice(indices)
        else:
            filter_result = [
                a * b for a, b in zip(no_waiting, self.main_available_kv_cache)
            ]
            max_value = max(filter_result)
            max_indices = [
                index for index, value in enumerate(filter_result) if value == max_value
            ]
            gpu_idx = random.choice(max_indices)

        self.main_available_kv_cache[gpu_idx] -= len(req.input_ids)
        return gpu_idx

    def resources_aware_scheduler(self, req):
        self.update_memory_and_requests()
        gpu_idx = self.allocate_gpu(req)
        self.workers[gpu_idx].send_pyobj(req)

    def pre_radix_scheduler(self, req):
        prefix_lens = [0] * self.dp_size

        with self.recv_tree_cache_lock:
            for gpu_id, radix_cache in self.newest_tree_cache.items():
                pre_len = get_match_len(radix_cache.root_node, req.input_ids, 0)
                prefix_lens[gpu_id] = pre_len

            # NOTE: 100 is used to reduce the influence of random input
            # e.g. If the match nums is [1, 2, 0, 0, 0, 0], we think the scheduer method should be resources aware
            if max(prefix_lens) <= 100:
                self.resources_aware_scheduler(req)
            else:
                gpu_idx = prefix_lens.index(max(prefix_lens))
                self.workers[gpu_idx].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                # logger.info(f"[event_loop]{type(recv_req)}")
                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                else:
                    # Send other control messages to all workers
                    for worker in self.workers:
                        worker.send_pyobj(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    configure_logger(server_args)
    suppress_other_loggers()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send("ready")
        if controller.recv_tree_cache_thread:
            controller.recv_tree_cache_thread.start()
        controller.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
