# SPDX-License-Identifier: Apache-2.0
"""Execute one stage group while serving peer-to-peer tensor transfers."""

import asyncio
import concurrent.futures
import pickle
import threading
import time

import torch
import torch.distributed as dist
import zmq
import zmq.asyncio
from torch.utils._pytree import tree_leaves, tree_map

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import (
    capture_request,
    fork_request,
    read_field,
    restore_request,
    write_field,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.plan import REQUEST_CONTEXT
from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import (
    TensorTransport,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    resolve_sampling_params_cls,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import TraceNullContext

logger = init_logger(__name__)


def run_partition(scheduler, req):
    """collect every rank's status before the root advertises a result"""
    error = None
    torch.cuda.reset_peak_memory_stats()
    try:
        result = scheduler.worker.pipeline.forward(req, scheduler.server_args)
        if isinstance(result, OutputBatch):
            if result.error:
                raise RuntimeError(result.error)
            req.extra["_deployment_result"] = result
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    peak = torch.cuda.max_memory_reserved() / (1024**2)
    if scheduler._is_multi_rank():
        group = get_world_group()
        statuses = [None] * len(group.ranks)
        dist.all_gather_object(statuses, (error, peak), group=group.cpu_group)
        error = next((item[0] for item in statuses if item[0]), None)
        peak = max(item[1] for item in statuses)
    req.extra["_deployment_peak_memory_mb"] = peak
    return error


def broadcast_request(scheduler, req):
    """broadcast one context within a TP/SP group, keeping rank-local tracing"""
    group = get_world_group()
    device = f"cuda:{scheduler.worker.local_rank}"
    if req is not None:
        context = capture_request(req)
        metadata = [
            (key, tuple(t.shape), t.dtype) for key, t in context["tensors"].items()
        ]
        metrics = req.metrics
        scheduler._broadcast_to_all_ranks((context["structure"], metadata, metrics))
    else:
        structure, metadata, metrics = scheduler._broadcast_to_all_ranks(None)
        context = {
            "structure": structure,
            "tensors": {
                key: torch.empty(shape, dtype=dtype, device=device)
                for key, shape, dtype in metadata
            },
        }
    # capture_request already supplies a flat tensor table; only metadata uses CPU RPC
    for key, tensor in context["tensors"].items():
        tensor = tensor.to(device)
        dist.broadcast(tensor, src=group.ranks[0], group=group.device_group)
        context["tensors"][key] = tensor
    restored = object.__new__(Req)
    vars(restored).update(
        metrics=metrics,
        trace_ctx=req.trace_ctx if req is not None else TraceNullContext(),
        modules={},
    )
    restore_request(restored, context, device)
    return restored


def _field_size(value):
    return sum(
        (leaf.numel() * leaf.element_size() + 511) // 512 * 512
        for leaf in tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    )


class DeploymentWorker:
    """serialize compute separately from transfers so downstream reads cannot block on it"""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.args = scheduler.server_args
        self.sampling_params_cls = resolve_sampling_params_cls(self.args)
        self.role = scheduler.worker.pipeline.deployment_plan.roles[
            self.args.stage_role
        ]
        self.device = torch.device(f"cuda:{scheduler.worker.local_rank}")
        self.results = {}
        # cancellation tombstones prevent late compute from reviving released results
        self.expired = {}
        self.lock = threading.Lock()
        self.compute = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.transfers = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.transport = None
        self.context = None

    def _execute(self, req):
        torch.cuda.set_device(self.device)
        if self.scheduler._is_multi_rank():
            self.scheduler._broadcast_to_all_ranks("deployment_compute")
            req = broadcast_request(self.scheduler, req)
        error = run_partition(self.scheduler, req)
        if error:
            raise RuntimeError(error)
        # staged results outlive compute and must not alias reusable output buffers
        values = {
            path: capture_request(req)
            if path == REQUEST_CONTEXT
            else read_field(req, path)
            for path in self.role.outputs
        }
        values = tree_map(
            lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
            values,
        )
        ready = torch.cuda.Event()
        ready.record()
        peak = max(
            req.extra["_deployment_peak_memory_mb"],
            torch.cuda.max_memory_reserved() / (1024**2),
        )
        return values, ready, req.metrics, peak

    async def _execute_message(self, message):
        ticket = message["ticket"]
        with self.lock:
            if len(self.results) >= self.args.disagg_max_slots_per_instance:
                raise RuntimeError("Stage worker result capacity exhausted")
        incoming = message["request"]
        # 1. lower raw/base sampling parameters once; preserve already model-specific ones
        params = None
        if isinstance(incoming, dict):
            params = self.sampling_params_cls()
            req = Req(sampling_params=params, **incoming)
            params._explicit_fields = set(incoming)
        else:
            req = fork_request(incoming)
            if type(req.sampling_params) is SamplingParams:
                params = self.sampling_params_cls()
                params._merge_with_user_params(req.sampling_params)
                params._explicit_fields = vars(req.sampling_params).get(
                    "_explicit_fields", set()
                )
        if params is not None:
            req.sampling_params = params
            params._adjust(self.args)
            params._validate()
            req.validate()
        # 2. restore upstream state only after constructing a valid local request
        for source in message["sources"]:
            values = await self.transport.receive(
                self.context, source, self.args.disagg_timeout
            )
            for path, value in values.items():
                if path == REQUEST_CONTEXT:
                    restore_request(req, value, self.device)
                else:
                    write_field(req, path, value)
        if "_deployment_output_fields" in req.extra:
            req.extra["_deployment_result"] = OutputBatch(
                **req.extra.pop("_deployment_output_fields")
            )
        loop = asyncio.get_running_loop()
        values, ready, metrics, peak = await loop.run_in_executor(
            self.compute, self._execute, req
        )
        # 3. publish owned tensor copies unless cancellation already retired this ticket
        with self.lock:
            if ticket in self.expired:
                raise RuntimeError("Request cancelled before stage completion")
            self.results[ticket] = (time.monotonic(), values, ready)
        return {
            "ticket": ticket,
            "sizes": {path: _field_size(value) for path, value in values.items()},
            "metrics": metrics,
            "peak_memory_mb": peak,
        }

    def _push(self, message):
        torch.cuda.set_device(self.device)
        with self.lock:
            _, values, ready = self.results[message["ticket"]]
            selected = {path: values[path] for path in message["fields"]}
        torch.cuda.current_stream().wait_event(ready)
        if message.get("cpu"):
            return {
                "values": tree_map(
                    lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, selected
                )
            }
        return self.transport.push(selected, message["target"])

    async def _handle(self, socket, envelope, message):
        try:
            method = message["method"]
            if method == "execute":
                reply = await self._execute_message(message)
            elif method == "push":
                reply = await asyncio.get_running_loop().run_in_executor(
                    self.transfers, self._push, message
                )
            elif method == "release":
                with self.lock:
                    self.results.pop(message["ticket"], None)
                    self.expired[message["ticket"]] = time.monotonic()
                reply = {}
            elif method == "ping":
                reply = {"role": self.role.name, "stages": self.role.stages}
            else:
                raise ValueError(f"Unknown deployment worker method: {method}")
        except Exception as exc:
            reply = {"error": str(exc)}
        await socket.send_multipart([*envelope, pickle.dumps(reply)])

    async def run(self):
        torch.cuda.set_device(self.device)
        self.transport = TensorTransport(
            self.device,
            self.args.disagg_p2p_hostname,
            self.args.disagg_transfer_pool_size,
            self.args.disagg_ib_device,
        )
        self.context = zmq.asyncio.Context()
        socket = self.context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.args.stage_endpoint)
        logger.info(
            "Stage role %s ready at %s", self.role.name, self.args.stage_endpoint
        )
        tasks = set()
        try:
            while self.scheduler._running:
                cutoff = time.monotonic() - self.args.disagg_timeout
                with self.lock:
                    self.results = {
                        key: value
                        for key, value in self.results.items()
                        if value[0] > cutoff
                    }
                    self.expired = {
                        key: value
                        for key, value in self.expired.items()
                        if value > cutoff
                    }
                try:
                    frames = await asyncio.wait_for(socket.recv_multipart(), 1)
                except TimeoutError:
                    continue
                task = asyncio.create_task(
                    self._handle(socket, frames[:-1], pickle.loads(frames[-1]))
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.compute.shutdown(wait=True)
            self.transfers.shutdown(wait=True)
            self.transport.close()
            socket.close()
            self.context.term()
