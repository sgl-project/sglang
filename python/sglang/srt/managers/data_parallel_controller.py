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

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from collections import deque
from enum import Enum, auto
from typing import List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    BlockReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import (
    bind_port,
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    MINIMUM_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self):
        # TODO: support minimum tokens method
        self.budget_queue = deque()

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget queue.
        Use num_reqs instead of num_waiting_reqs to balance decode running batch.
        """
        loads = load_update.loads
        self.budget_queue.clear()

        num_reqs = [load.num_reqs for load in loads]
        if not num_reqs:
            return

        max_num_reqs = max(num_reqs)
        if all(x == max_num_reqs for x in num_reqs):
            return

        while any(x != num_reqs[0] for x in num_reqs):
            min_load = min(num_reqs)
            min_indices = [i for i, x in enumerate(num_reqs) if x == min_load]
            second_min_load = min(x for x in num_reqs if x > min_load)
            self.budget_queue.extend(
                [loads[i].dp_rank for i in min_indices] * (second_min_load - min_load)
            )
            for idx in min_indices:
                num_reqs[idx] = second_min_load

    def dispatch(self):
        if self.budget_queue:
            return self.budget_queue.popleft()
        return None


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        # for dp balance
        self.global_balance_id = 0

        # Parse args
        self.max_total_num_tokens = None
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.MINIMUM_TOKENS: self.minimum_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget()

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.max_req_input_len = None

        self.init_dispatcher()

    def send_to_all_workers(self, obj):
        for worker in self.workers:
            worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching),
                (TokenizedEmbeddingReqInput, self.dispatching),
                (BlockReqInput, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # Determine the endpoint for inter-node communication
        if server_args.dist_init_addr is None:
            endpoint = f"tcp://127.0.0.1:{server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        else:
            endpoint = f"tcp://{server_args.dist_init_addr}"

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            rep_socket.close()

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 60 * 1000)  # 1 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 60 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        # Pre-allocate worker ports on node 0 to avoid conflicts
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                port_and_socket = get_zmq_socket(self.context, zmq.PUSH)
                worker_ports.append(port_and_socket[0])
                self.workers[dp_rank] = port_and_socket[1]
                logger.debug(f"Assigned port {port_and_socket[0]} to worker {dp_rank}")

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

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
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=run_scheduler_process,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess():
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.data_parallel_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
            self.workers[req.data_parallel_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        if self.server_args.disaggregation_mode == "null":
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
        else:
            assert (
                req.bootstrap_room is not None
            ), "req.bootstrap_room should not be None. Do not send requests directly to prefill or decode instances, but send to the router instead."
            self.workers[req.bootstrap_room % len(self.workers)].send_pyobj(req)

    def shortest_queue_scheduler(self, req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch()
        if target_worker is None:
            self.round_robin_scheduler(req)
        else:
            self.workers[target_worker].send_pyobj(req)

    def minimum_tokens_scheduler(self, req):
        if self.maybe_external_dp_rank_routing(req):
            return

        logger.warning(
            "The 'minimum_tokens' load balancing method is deprecated for now and will introduced later."
            "Fall back to 'round_robin_scheduler'"
        )
        self.round_robin_scheduler(req)

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
