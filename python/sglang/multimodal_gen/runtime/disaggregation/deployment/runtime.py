# SPDX-License-Identifier: Apache-2.0
"""Stage-group workers and an asynchronous, tensor-free graph coordinator."""

import asyncio
import base64
import concurrent.futures
import pickle
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import zmq
import zmq.asyncio
from torch.utils._pytree import tree_flatten, tree_leaves, tree_map, tree_unflatten

from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import (
    fork_request,
    read_field,
    write_field,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import (
    TensorTransport,
    rpc,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


def run_partition(scheduler, req):
    error = None
    try:
        result = scheduler.worker.pipeline.forward(req, scheduler.server_args)
        if isinstance(result, OutputBatch):
            if result.error:
                raise RuntimeError(result.error)
            req.extra["_deployment_result"] = result
    except Exception as exc:
        error = str(exc)
    if scheduler._is_multi_rank():
        group = get_world_group()
        errors = [None] * len(group.ranks)
        dist.all_gather_object(errors, error, group=group.cpu_group)
        error = next((item for item in errors if item), None)
    return error


def broadcast_request(scheduler, req):
    group = get_world_group()
    if req is not None:
        leaves, spec = tree_flatten(vars(req))
        tensors = [(i, x) for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
        metadata = [(i, tuple(t.shape), t.dtype) for i, t in tensors]
        leaves = [None if isinstance(x, torch.Tensor) else x for x in leaves]
        scheduler._broadcast_to_all_ranks((leaves, spec, metadata))
    else:
        leaves, spec, metadata = scheduler._broadcast_to_all_ranks(None)
        tensors = [
            (
                i,
                torch.empty(
                    shape, dtype=dtype, device=f"cuda:{scheduler.worker.local_rank}"
                ),
            )
            for i, shape, dtype in metadata
        ]
    for index, tensor in tensors:
        if tensor.device.type != "cuda":
            tensor = tensor.to(f"cuda:{scheduler.worker.local_rank}")
        dist.broadcast(tensor, src=group.ranks[0], group=group.device_group)
        leaves[index] = tensor
    restored = object.__new__(Req)
    vars(restored).update(tree_unflatten(leaves, spec))
    return restored


def _field_size(value):
    return sum(
        (leaf.numel() * leaf.element_size() + 511) // 512 * 512
        for leaf in tree_leaves(value)
        if isinstance(leaf, torch.Tensor)
    )


class DeploymentWorker:
    """One placement group; computation is serialized, transfers remain serviceable."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.args = scheduler.server_args
        self.pipeline = scheduler.worker.pipeline
        self.plan = self.pipeline.deployment_plan
        self.role = self.plan.roles[self.args.stage_role]
        self.device = torch.device(f"cuda:{scheduler.worker.local_rank}")
        self.results = {}
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
        values = {path: read_field(req, path) for path in self.role.outputs}
        ready = torch.cuda.Event()
        ready.record()
        return values, ready, req.metrics

    async def _execute_message(self, message):
        ticket = message["ticket"]
        with self.lock:
            if len(self.results) >= self.args.disagg_max_slots_per_instance:
                raise RuntimeError("Stage worker result capacity exhausted")
        req = fork_request(message["request"])
        for source in message["sources"]:
            fields = source["fields"]
            values = await self.transport.receive(
                self.context,
                source["endpoint"],
                source["ticket"],
                fields,
                sum(source["sizes"][path] for path in fields),
                self.args.disagg_timeout,
            )
            for path, value in values.items():
                write_field(req, path, value)
        if "_deployment_output_fields" in req.extra:
            req.extra["_deployment_result"] = OutputBatch(
                **req.extra.pop("_deployment_output_fields")
            )
        loop = asyncio.get_running_loop()
        values, ready, metrics = await loop.run_in_executor(
            self.compute, self._execute, req
        )
        with self.lock:
            if ticket in self.expired:
                raise RuntimeError("Request cancelled before stage completion")
            self.results[ticket] = (time.monotonic(), values, ready)
        return {
            "ticket": ticket,
            "sizes": {path: _field_size(value) for path, value in values.items()},
            "metrics": metrics,
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


class DeploymentServer:
    """Route ready groups concurrently; tensors move directly between their workers."""

    def __init__(self, plan, timeout=600, capacity=8):
        self.plan = plan
        self.timeout = timeout
        self.capacity = capacity
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.thread = None
        self.context = None
        self.counters = defaultdict(int)
        self.slots = {}
        self.leases = {}

    async def execute(self, request, outputs=None, roles=None):
        if isinstance(request, Req) and request.image_path:
            request = fork_request(request)
            paths = (
                request.image_path
                if isinstance(request.image_path, (list, tuple))
                else [request.image_path]
            )
            request.image_path = [
                "data:application/octet-stream;base64,"
                + base64.b64encode(Path(path).read_bytes()).decode("ascii")
                if isinstance(path, str) and Path(path).is_file()
                else path
                for path in paths
            ]
        desired = (
            self.plan.outputs
            if outputs is None
            else {path: self.plan.outputs[path] for path in outputs}
        )
        selected = set(roles or desired.values())
        pending = list(selected)
        while pending:
            name = pending.pop()
            for dependency in self.plan.roles[name].dependencies - selected:
                selected.add(dependency)
                pending.append(dependency)
        if not selected <= self.plan.roles.keys():
            raise ValueError("Unknown requested role")
        for name in selected:
            if not self.plan.roles[name].dependencies <= selected:
                raise ValueError(f"Missing dependencies for role {name}")
        for path, name in desired.items():
            if name not in selected or path not in self.plan.roles[name].outputs:
                raise ValueError(
                    f"Output {path} is not exposed by selected role {name}"
                )
        job = uuid.uuid4().hex
        tasks, assignments = {}, {}

        async def run_role(name):
            role = self.plan.roles[name]
            for dependency in role.dependencies:
                await tasks[dependency]
            index = self.counters[name] % len(role.endpoints)
            self.counters[name] += 1
            endpoint = role.endpoints[index]
            ticket = f"{job}:{name}"
            assignments[name] = (endpoint, ticket)
            sources = []
            for dependency in role.dependencies:
                fields = [
                    path for path, source in role.inputs.items() if source == dependency
                ]
                if fields:
                    descriptor = await tasks[dependency]
                    sources.append(
                        {
                            **descriptor,
                            "endpoint": assignments[dependency][0],
                            "fields": fields,
                        }
                    )
            async with self.slots.setdefault(endpoint, asyncio.Semaphore(1)):
                return await rpc(
                    self.context,
                    endpoint,
                    {
                        "method": "execute",
                        "ticket": ticket,
                        "request": request,
                        "sources": sources,
                    },
                    self.timeout,
                )

        try:
            for name in self.plan.order:
                if name in selected:
                    tasks[name] = asyncio.create_task(run_role(name))
            async with asyncio.timeout(self.timeout):
                await asyncio.gather(*tasks.values())
            return {
                "sources": [
                    {
                        **tasks[name].result(),
                        "endpoint": assignments[name][0],
                        "fields": [
                            path for path, source in desired.items() if source == name
                        ],
                    }
                    for name in self.plan.order
                    if name in desired.values()
                ],
                "lease": assignments,
                "metrics": [task.result().get("metrics") for task in tasks.values()],
            }
        except BaseException:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            await self.release(assignments)
            raise

    async def release(self, lease):
        await asyncio.gather(
            *(
                rpc(
                    self.context,
                    endpoint,
                    {"method": "release", "ticket": ticket},
                    min(self.timeout, 5),
                )
                for endpoint, ticket in lease.values()
            ),
            return_exceptions=True,
        )

    async def _handle(self, socket, envelope, message, admission):
        response = None
        try:
            if isinstance(message, dict):
                if message["method"] == "ping":
                    response = {"status": "ok"}
                elif message["method"] == "encode":
                    await admission.acquire()
                    try:
                        response = await self.execute(
                            message["request"], message["outputs"]
                        )
                        lease_id = uuid.uuid4().hex
                        self.leases[lease_id] = (
                            response["lease"],
                            time.monotonic(),
                            admission,
                        )
                        response["lease"] = lease_id
                    except BaseException:
                        admission.release()
                        raise
                elif message["method"] == "release":
                    await self.release_lease(message["lease"])
                    response = {}
                else:
                    raise ValueError("Unknown deployment server method")
            else:
                requests = message if isinstance(message, list) else [message]
                if len(requests) != 1 or not isinstance(requests[0], Req):
                    raise ValueError(
                        "Deployment serving expects one generation request"
                    )
                async with admission:
                    outputs = [
                        path for path in self.plan.outputs if path.startswith("result.")
                    ]
                    if not outputs:
                        raise ValueError(
                            "This deployment returns encoder tensors; use EncodingClient"
                        )
                    started = time.monotonic()
                    result = await self.execute(requests[0], outputs)
                    try:
                        fields = {}
                        for source in result["sources"]:
                            reply = await rpc(
                                self.context,
                                source["endpoint"],
                                {
                                    "method": "push",
                                    "ticket": source["ticket"],
                                    "fields": source["fields"],
                                    "cpu": True,
                                },
                                self.timeout,
                            )
                            fields.update(reply["values"])
                        response = OutputBatch(
                            **{
                                path.removeprefix("result."): value
                                for path, value in fields.items()
                            }
                        )
                        metrics = RequestMetrics(requests[0].request_id)
                        for item in result["metrics"]:
                            if item is not None:
                                metrics.stages.update(item.stages)
                                metrics.denoising_stages.update(item.denoising_stages)
                                metrics.steps.extend(item.steps)
                        response.metrics = metrics
                        metrics.total_duration_ms = (time.monotonic() - started) * 1000
                    finally:
                        await self.release(result["lease"])
        except Exception as exc:
            response = (
                {"error": str(exc)}
                if isinstance(message, dict)
                else OutputBatch(error=str(exc))
            )
        await socket.send_multipart([*envelope, pickle.dumps(response)])

    async def serve(self, endpoint):
        self.context = zmq.asyncio.Context()
        socket = self.context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)
        admission = asyncio.Semaphore(self.capacity)
        tasks = set()
        self.ready.set()
        try:
            while not self.stopped.is_set():
                expired = [
                    key
                    for key, (_, started, _) in self.leases.items()
                    if time.monotonic() - started > self.timeout
                ]
                for key in expired:
                    await self.release_lease(key)
                try:
                    frames = await asyncio.wait_for(socket.recv_multipart(), 0.2)
                except TimeoutError:
                    continue
                task = asyncio.create_task(
                    self._handle(
                        socket, frames[:-1], pickle.loads(frames[-1]), admission
                    )
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                *(self.release_lease(key) for key in list(self.leases))
            )
            socket.close()
            self.context.term()

    async def release_lease(self, key):
        lease = self.leases.pop(key, None)
        if lease is not None:
            assignments, _, admission = lease
            try:
                await self.release(assignments)
            finally:
                admission.release()

    def start(self, endpoint):
        self.thread = threading.Thread(
            target=lambda: asyncio.run(self.serve(endpoint)), daemon=True
        )
        self.thread.start()
        if not self.ready.wait(30):
            raise RuntimeError("Deployment coordinator did not bind its endpoint")

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=self.timeout + 1)
