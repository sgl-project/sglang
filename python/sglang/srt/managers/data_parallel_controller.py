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
"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
import threading
from enum import Enum, auto

import zmq

from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    bind_port,
    configure_logger,
    get_zmq_socket,
    kill_parent_process,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

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
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Start data parallel workers
        base_gpu_id = 0
        self.workers = [None] * server_args.dp_size

        threads = []
        sockets = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            if server_args.enable_dp_attention:
                # Data parallelism resues the tensor parallelism group,
                # so all dp ranks should use the same nccl port.
                tmp_port_args.nccl_port = port_args.nccl_port
            else:
                # This port is checked free in PortArgs.init_new.
                # We hold it first so that the next dp worker gets a different port
                sockets.append(bind_port(tmp_port_args.nccl_port))

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_worker_func,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank),
            )
            threads.append(thread)
            base_gpu_id += 1 if server_args.enable_dp_attention else server_args.tp_size

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def launch_worker_func(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        launch_func_ = (
            self.launch_tensor_parallel_process
            if server_args.enable_dp_attention
            else self.launch_tensor_parallel_group
        )
        self.workers[dp_rank] = launch_func_(
            server_args,
            port_args,
            base_gpu_id,
            dp_rank,
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
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
            gpu_id = server_args.base_gpu_id + base_gpu_id + tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, dp_rank, writer),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        send_to = get_zmq_socket(
            self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
        )

        # Wait for model to finish loading and get max token nums
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]

        return send_to

    def launch_tensor_parallel_process(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = base_gpu_id
        tp_rank = dp_rank
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank, dp_rank, writer),
        )
        proc.start()
        send_to = get_zmq_socket(
            self.context, zmq.PUSH, port_args.scheduler_input_ipc_name
        )

        scheduler_info = reader.recv()
        self.max_total_num_tokens = scheduler_info["max_total_num_tokens"]

        return send_to

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

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
        pipe_writer.send(
            {"status": "ready", "max_total_num_tokens": controller.max_total_num_tokens}
        )
        controller.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
