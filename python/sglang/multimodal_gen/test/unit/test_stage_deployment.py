# SPDX-License-Identifier: Apache-2.0
"""Contracts, placement, and real coordinator fan-out/join without model weights."""

import asyncio
import pickle
import sys
from graphlib import CycleError
from types import SimpleNamespace

import pytest
import torch
import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment import client
from sglang.multimodal_gen.runtime.disaggregation.deployment.coordinator import (
    DeploymentServer,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import (
    pack_fields,
    unpack_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.plan import (
    Deployment,
    ExecutionPlan,
    RoleSpec,
    StageSpec,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


def specs():
    return (
        StageSpec(
            "text", outputs=("extra.text",), components=("text_encoder",), after=()
        ),
        StageSpec("image", outputs=("extra.image",), components=("vae",), after=()),
        StageSpec(
            "denoise",
            inputs=("extra.text", "extra.image"),
            outputs=("latents",),
            components=("transformer",),
            after=(),
        ),
        StageSpec(
            "decode",
            inputs=("latents",),
            outputs=("result.output",),
            components=("vae",),
        ),
    )


def test_contracts_compile_and_tensor_tree_round_trip():
    plan = ExecutionPlan.compile(
        Deployment(
            {
                stage.name: RoleSpec((stage.name,), (f"tcp://127.0.0.1:{19000 + i}",))
                for i, stage in enumerate(specs())
            }
        ),
        specs(),
    )
    assert not plan.roles["text"].dependencies
    assert not plan.roles["image"].dependencies
    assert plan.roles["denoise"].dependencies == {"text", "image"}
    assert plan.roles["denoise"].inputs == {
        "extra.text": "text",
        "extra.image": "image",
    }
    assert plan.roles["decode"].components == {"vae"}
    values = {
        "extra.text": (torch.arange(5), {"mask": torch.ones(5, dtype=torch.bool)}),
        "extra.image": None,
        "empty": torch.empty(0, 3),
    }
    tensors, scalar = pack_fields(values)
    restored = unpack_fields(tensors, scalar)
    torch.testing.assert_close(restored["extra.text"][0], values["extra.text"][0])
    assert restored["extra.image"] is None
    assert restored["empty"].shape == (0, 3)
    with pytest.raises(ValueError):
        unpack_fields({}, scalar)
    with pytest.raises(TypeError):
        pack_fields({"required": object()})


@pytest.mark.parametrize("deployed", [False, True])
def test_postprocess_can_precede_another_stage(deployed):
    stages = [object(), object()]
    req = Req(prompt="test")

    def execute(stage, current):
        if stage is stages[0]:
            return OutputBatch(output=torch.ones(3, 2, 2))
        current.output = current.output + 1
        return current

    executor = object.__new__(SyncExecutor)
    run = executor.stage_runner(
        stages, SimpleNamespace(stage_deployment="plan" if deployed else None), execute
    )
    current = run(stages[0], req)
    assert current is req
    current = run(stages[1], current)
    pipeline = SimpleNamespace(deployment_plan=object() if deployed else None)
    result = ComposedPipelineBase._finalize_stage_result(pipeline, current)
    if deployed:
        assert result is req
    else:
        assert isinstance(result, OutputBatch)
    torch.testing.assert_close(result.output, torch.full((3, 2, 2), 2.0))


@pytest.mark.parametrize(
    "roles",
    [
        {
            "a": RoleSpec(("text", "denoise"), ("tcp://localhost:1",)),
            "b": RoleSpec(("image", "decode"), ("tcp://localhost:2",)),
        },
        {"a": RoleSpec(("text", "unknown"), ("tcp://localhost:1",))},
        {
            "a": RoleSpec(("text",), ("tcp://localhost:1",)),
            "b": RoleSpec(("text",), ("tcp://localhost:2",)),
        },
    ],
)
def test_invalid_partition_rejected(roles):
    with pytest.raises((ValueError, CycleError)):
        ExecutionPlan.compile(Deployment(roles), specs())


def test_coordinator_parallel_join_and_cleanup():
    async def run():
        context = zmq.asyncio.Context()
        sockets, tasks, completed, live = [], [], set(), set()
        both_started = asyncio.Event()
        started = set()

        async def peer(name, socket):
            while True:
                message = pickle.loads(await socket.recv())
                method = message["method"]
                if method == "execute":
                    if name in ("text", "image"):
                        started.add(name)
                        if len(started) == 2:
                            both_started.set()
                        await asyncio.wait_for(both_started.wait(), 2)
                    if name == "denoise":
                        assert {"text", "image"} <= completed
                        assert {
                            item["ticket"].split(":")[-1] for item in message["sources"]
                        } == {"text", "image"}
                    completed.add(name)
                    live.add(message["ticket"])
                    reply = {
                        "ticket": message["ticket"],
                        "sizes": {path: 0 for path in plan.roles[name].outputs},
                    }
                else:
                    live.discard(message["ticket"])
                    reply = {}
                await socket.send(pickle.dumps(reply))

        roles = {}
        for stage in specs():
            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(socket)
            roles[stage.name] = RoleSpec((stage.name,), (f"tcp://127.0.0.1:{port}",))
        plan = ExecutionPlan.compile(Deployment(roles), specs())
        server = DeploymentServer(plan, timeout=5)
        server.context = context
        for name, socket in zip(roles, sockets):
            tasks.append(asyncio.create_task(peer(name, socket)))
        try:
            result = await server.execute(
                Req(image_path=["data:image/png;base64," + "A" * 1024])
            )
            assert completed == set(roles)
            assert result["sources"][0]["fields"] == ["result.output"]
            await server.release(result["lease"])
            assert not live
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for socket in sockets:
                socket.close()
            context.term()

    asyncio.run(run())


def test_client_failure_cancels_outstanding_rpc(monkeypatch):
    closed = []

    class NoGpuTransport:
        def __init__(self, *args):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(client, "TensorTransport", NoGpuTransport)

    async def run():
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        port = socket.bind_to_random_port("tcp://127.0.0.1")

        async def peer():
            first = True
            while True:
                frames = await socket.recv_multipart()
                message = pickle.loads(frames[-1])
                if message["method"] == "encode":
                    if not first:
                        continue
                    first = False
                    reply = {"error": "invalid encoder input"}
                else:
                    reply = {}
                await socket.send_multipart([*frames[:-1], pickle.dumps(reply)])

        task = asyncio.create_task(peer())
        try:
            with pytest.raises(RuntimeError, match="invalid encoder input"):
                async with client.EncodingClient(f"tcp://127.0.0.1:{port}") as encoder:
                    await asyncio.gather(encoder.encode({}), encoder.encode({}))
            assert closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            socket.close()
            context.term()

    asyncio.run(asyncio.wait_for(run(), 5))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x"]))
