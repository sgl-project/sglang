"""A controller that manages a group of tensor parallel workers."""

import multiprocessing
import logging
import os
import pickle

import torch
import torch.distributed as dist
import zmq
import zmq.asyncio

from sglang.srt.managers.controller.tp_worker import ModelTpServer
from sglang.srt.server_args import PortArgs, ServerArgs, ModelPortArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback


logger = logging.getLogger("srt.controller")


def run_tp_server(
    gpu_id: int,
    tp_rank: int,
    server_args: ServerArgs,
    model_port_args: ModelPortArgs,
    model_overide_args: dict,
):
    """Run a tp server."""
    try:
        model_server = ModelTpServer(
            gpu_id,
            tp_rank,
            server_args,
            model_port_args,
            model_overide_args,
        )
        tp_cpu_group = model_server.model_runner.tp_group.cpu_group

        while True:
            recv_reqs = broadcast_recv_input(None, tp_rank, tp_cpu_group)
            model_server.exposed_step(recv_reqs)
    except Exception:
        logger.error("Exception in run_tp_server:\n" + get_exception_traceback())
        raise


def launch_tp_servers(gpu_ids, tp_rank_range, server_args,
                      model_port_args, model_overide_args):
    """Launch multiple tp servers."""
    procs = []
    for i in tp_rank_range:
        proc = multiprocessing.Process(target=run_tp_server, args=(
            gpu_ids[i], i, server_args, model_port_args, model_overide_args
        ))
        proc.start()
        procs.append(proc)

    return procs


def broadcast_recv_input(data, rank, dist_group):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(list(serialized_data))
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.tolist())
        data = pickle.loads(serialized_data)
        return data


class ControllerSingle:
    """A controller that manages a group of tensor parallel workers."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs, model_overide_args: dict):
        # Parse args
        self.server_args = server_args

        # Init communication
        context = zmq.Context(2)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # Init model server
        tp_size_local = server_args.tp_size // server_args.nnodes
        gpu_ids = [i for _ in range(server_args.nnodes) for i in range(tp_size_local)]

        # Launch other tp ranks
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids, tp_rank_range, server_args,
                port_args.model_port_args[0], model_overide_args)

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.model_port_args[0],
            model_overide_args,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def loop_for_forward(self):
        while True:
            recv_reqs = self.recv_requests()

            if self.server_args.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break
        return recv_reqs


def start_controller_process(
    server_args: ServerArgs, port_args: PortArgs, pipe_writer, model_overide_args: dict
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = ControllerSingle(server_args, port_args, model_overide_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        for t in controller.tp_procs:
            os.kill(t.pid, 9)
        kill_parent_process()
