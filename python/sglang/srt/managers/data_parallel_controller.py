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
import signal
import threading
import time
from enum import Enum, auto

import psutil
import setproctitle
import zmq

from fastapi import FastAPI, Request, HTTPException
import uvicorn

from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import bind_port, configure_logger, get_zmq_socket
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

class LoadTable:
    def __init__(self,dp_size):
        self._table = {
            rank:{
                "pending_requests":0,
                "nums_tokens":0,
                "last_updateTime":time.time(),
                "estimated_completed_time":0.0,
                "avg_processing_speed":5000, 
                "processing_history":[]
            } for rank in range(dp_size)
        }
        self._lock = threading.Lock()
        
    def update_processing_speed(self, rank, completed_tokens, time_taken):
        """update processing speed"""
        if time_taken <= 0:
            return
            
        with self._lock:
            current_speed = completed_tokens / time_taken
            
            alpha = 0.3  # smoothing factor
            old_speed = self._table[rank]["avg_processing_speed"]
            new_speed = alpha * current_speed + (1 - alpha) * old_speed
            
            self._table[rank]["avg_processing_speed"] = new_speed
            
    def _update_estimated_completion_time(self, rank):
        """def update estimated completion time"""
        with self._lock:
            rank_info = self._table[rank]
            pending_tokens = rank_info["nums_tokens"]
            processing_speed = rank_info["avg_processing_speed"]
            if processing_speed > 0:
                estimated_time = pending_tokens / processing_speed                
                self._table[rank]["estimated_completion_time"] = estimated_time
            else:
                rank_info["estimated_completion_time"] = float('inf')
            
    def update_add(self,rank,requests=0,tokens = 0):
        self._table[rank]["pending_requests"] += requests
        self._table[rank]["nums_tokens"] += tokens
        self._table[rank]["last_update"] = time.time()
        self._update_estimated_completion_time(rank)
        
        
    def update_sub(self, rank, requests=0, completed_tokens=0,processing_time = None):
        self._table[rank]["pending_requests"] = max(0, self._table[rank]["pending_requests"] - requests)
        self._table[rank]["nums_tokens"] = max(0, self._table[rank]["nums_tokens"] - completed_tokens)
        self._table[rank]["last_update"] = time.time()
        
        if processing_time is not None and processing_time > 0 and completed_tokens > 0:
            self.update_processing_speed(rank, completed_tokens, processing_time)
        self._update_estimated_completion_time(rank)

    def get_min_load_rank(self):
        """get minimal load rank"""
        with self._lock: 
            return min(
                self._table.items(), 
                key=lambda x: (       
                    x[1]["estimated_completed_time"], 
                    x[1]["nums_tokens"]  
                )
            )[0]
    

class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
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
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers = [None] * server_args.dp_size

        if server_args.enable_dp_attention:
            dp_port_args = self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
        else:
            dp_port_args = self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        # Only node rank 0 runs the real data parallel controller that dispatches the requests.
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    dp_port_args[dp_rank].scheduler_input_ipc_name,
                    True,
                )

        self.max_req_input_len = None
        
        if server_args.dp_size!=1 and server_args.load_balance_method=="shortest_queue":
            if server_args.node_rank == 0:
                self.load_table = LoadTable(server_args.dp_size)

                self.app = FastAPI()
                @self.app.post("/update_load")
                async def update_load(request:Request):
                    try:
                        data = await request.json()
                        rank = data["rank"]
                        completed_requests = data["completed_requests"]
                        completed_tokens = data["completed_tokens"]
                        taken_time = data["taken_time"]
                        self.load_table.update_sub(rank,requests=completed_requests,tokens=completed_tokens,processing_time = taken_time)
                        return {"status":"success"}
                    except Exception as e:
                        raise HTTPException(status_code=400,detail=str(e))
                if self.server_args.dist_init_addr != "null":   
                    self.master_ip = server_args.dist_init_addr.split(":")[0]
                    self.master_port = 8001
                    self.http_thread = threading.Thread(
                        target=self._run_http_server,
                        args=(self.master_port,),
                        daemon=True
                    )
                    self.http_thread.start()
                else:
                    raise NotImplementedError()
                

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            dp_port_args.append(tmp_port_args)

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
            base_gpu_id += server_args.tp_size * server_args.gpu_id_step

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

        return dp_port_args

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

    def launch_dp_attention_schedulers(self, server_args, port_args):
        self.launch_tensor_parallel_group(server_args, port_args, 0, None)
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            dp_port_args.append(PortArgs.init_new(server_args, dp_rank))
        return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
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
                    rank_port_args = PortArgs.init_new(server_args, dp_rank)
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
                proc = mp.Process(
                    target=run_scheduler_process,
                    args=(
                        server_args,
                        rank_port_args,
                        gpu_id,
                        tp_rank,
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

    def round_robin_scheduler(self, req: Req):
        if self.server_args.disaggregation_mode == "null":
            if req.data_parallel_rank is not None:
                logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
                self.workers[req.data_parallel_rank].send_pyobj(req)
            else:
                self.workers[self.round_robin_counter].send_pyobj(req)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
        else:
            if req.data_parallel_rank is not None:
                logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
                self.workers[req.data_parallel_rank].send_pyobj(req)
            else:
                self.workers[req.bootstrap_room % len(self.workers)].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        if input_requests.data_parallel_rank is not None:
            logger.debug(f"Direct routing to DP rank {input_requests.data_parallel_rank}")
            self.workers[input_requests.data_parallel_rank].send_pyobj(input_requests)
            return
        
        seq_len = len(input_requests.input_ids)
        selected_rank = self.load_table.get_min_load_rank()
        self.load_table.update_add(rank=selected_rank,requests=1,tokens=seq_len)
        self.workers[selected_rank].send_pyobj(input_requests)

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
                    # Send other control messages to first worker of tp group
                    for worker in self.workers[:: self.control_message_step]:
                        worker.send_pyobj(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
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
