# SPDX-License-Identifier: Apache-2.0
"""Asynchronous dependency scheduling without routing tensors through the head."""

import asyncio
import base64
import pickle
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import fork_request
from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import rpc
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


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
                "peak_memory_mb": max(
                    (task.result().get("peak_memory_mb", 0) for task in tasks.values()),
                    default=0,
                ),
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
                        response.peak_memory_mb = result["peak_memory_mb"]
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
