# SPDX-License-Identifier: Apache-2.0
"""Asynchronous dependency scheduling without routing tensors through the head."""

import asyncio
import pickle
import threading
import time
import uuid

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import fork_request
from sglang.multimodal_gen.runtime.disaggregation.deployment.plan import REQUEST_CONTEXT
from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import rpc
from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    create_dispatch_policy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.utils.image_io import inline_local_image_paths
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics


class DeploymentServer:
    """schedule ready roles; retain worker results until the last consumer releases them"""

    def __init__(self, plan, timeout=600, capacity=8, dispatch_policy="round_robin"):
        if timeout <= 0 or capacity < 1:
            raise ValueError("Deployment timeout and capacity must be positive")
        self.plan = plan
        self.timeout = timeout
        self.capacity = capacity
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.thread = None
        self.context = None
        self.policies = {
            name: create_dispatch_policy(dispatch_policy, len(role.endpoints))
            for name, role in plan.roles.items()
        }
        self.free_slots = {
            name: [1] * len(role.endpoints) for name, role in plan.roles.items()
        }
        self.available = asyncio.Condition()
        self.admission = asyncio.Semaphore(capacity)
        # running jobs own tasks; completed client jobs retain admission through a lease
        self.leases = {}
        self.jobs = {}

    async def execute(self, request, outputs=None, role=None):
        """return result descriptors and their lease; the caller owns final release"""
        # 1. inline head-local uploads before dispatch; workers need no shared upload path
        if isinstance(request, Req) and request.image_path:
            request = fork_request(request)
            request.image_path = inline_local_image_paths(request.image_path)
        available = (
            self.plan.outputs
            if role is None
            else dict.fromkeys(self.plan.roles[role].outputs, role)
        )
        desired = (
            available
            if outputs is None
            else {path: available[path] for path in outputs}
        )
        if not desired or REQUEST_CONTEXT in desired:
            raise ValueError(
                "Encoding requires named outputs from explicit stage contracts"
            )
        # walk the validated DAG backwards so requested roles bring all their ancestors
        selected = set(desired.values())
        for name in reversed(self.plan.order):
            if name in selected:
                selected.update(self.plan.roles[name].dependencies)
        job = uuid.uuid4().hex
        tasks, assignments = {}, {}

        async def run_role(name):
            role = self.plan.roles[name]
            sources = []
            for dependency in role.dependencies:
                descriptor = await tasks[dependency]
                fields = [
                    path for path, source in role.inputs.items() if source == dependency
                ]
                if fields:
                    sources.append(
                        {
                            **descriptor,
                            "endpoint": assignments[dependency][0],
                            "fields": fields,
                        }
                    )
            # 2. reserve compute capacity, not result lifetime; leases outlive execution
            async with self.available:
                await self.available.wait_for(lambda: any(self.free_slots[name]))
                index = self.policies[name].select_with_capacity(self.free_slots[name])
                self.free_slots[name][index] -= 1
            endpoint = role.endpoints[index]
            ticket = f"{job}:{name}"
            assignments[name] = (endpoint, ticket)
            try:
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
            except Exception as exc:
                raise RuntimeError(f"Stage role {name} failed: {exc}") from exc
            finally:
                async with self.available:
                    self.free_slots[name][index] += 1
                    self.available.notify_all()

        try:
            # tasks are created topologically; independent branches can run concurrently
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
            # 3. cancellation is also a failure path: stop graph work before releasing buffers
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

    async def _handle(self, socket, envelope, message):
        try:
            if isinstance(message, dict):
                if message["method"] == "ping":
                    response = {"status": "ok"}
                elif message["method"] == "encode":
                    job_id = message["job_id"]
                    if job_id in self.jobs or job_id in self.leases:
                        raise ValueError("Duplicate encoding job ID")
                    self.jobs[job_id] = asyncio.current_task()
                    try:
                        await self.admission.acquire()
                        try:
                            response = await self.execute(
                                message["request"],
                                message["outputs"],
                                message.get("role"),
                            )
                            # returning descriptors does not release the client's admission permit
                            self.leases[job_id] = (
                                response["lease"],
                                time.monotonic(),
                            )
                            response["lease"] = job_id
                        except BaseException:
                            self.admission.release()
                            raise
                    finally:
                        self.jobs.pop(job_id, None)
                elif message["method"] == "release":
                    await self.release_lease(message["lease"])
                    response = {}
                elif message["method"] == "cancel":
                    job_id = message["job_id"]
                    task = self.jobs.pop(job_id, None)
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    await self.release_lease(job_id)
                    response = {}
                else:
                    raise ValueError("Unknown deployment server method")
            else:
                # HTTP consumes final results on the head; tensor clients pull from workers
                requests = message if isinstance(message, list) else [message]
                if len(requests) != 1 or not isinstance(requests[0], Req):
                    raise ValueError(
                        "Deployment serving expects one generation request"
                    )
                async with self.admission:
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
        socket.setsockopt(zmq.RCVHWM, self.capacity * 2)
        socket.bind(endpoint)
        tasks = set()
        work_tasks = set()
        self.ready.set()
        try:
            while not self.stopped.is_set():
                expired = [
                    key
                    for key, (_, started) in self.leases.items()
                    if time.monotonic() - started > self.timeout
                ]
                for key in expired:
                    await self.release_lease(key)
                try:
                    frames = await asyncio.wait_for(socket.recv_multipart(), 0.2)
                except TimeoutError:
                    continue
                message = pickle.loads(frames[-1])
                is_work = (
                    not isinstance(message, dict) or message.get("method") == "encode"
                )
                if is_work and len(work_tasks) >= self.capacity * 2:
                    error = "Deployment queue is full; reduce client concurrency"
                    reply = (
                        {"error": error}
                        if isinstance(message, dict)
                        else OutputBatch(error=error)
                    )
                    await socket.send_multipart([*frames[:-1], pickle.dumps(reply)])
                    continue
                task = asyncio.create_task(self._handle(socket, frames[:-1], message))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                if is_work:
                    work_tasks.add(task)
                    task.add_done_callback(work_tasks.discard)
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
        """release once, including cancellation/expiry racing with client cleanup"""
        lease = self.leases.pop(key, None)
        if lease is not None:
            assignments, _ = lease
            try:
                await self.release(assignments)
            finally:
                self.admission.release()

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
