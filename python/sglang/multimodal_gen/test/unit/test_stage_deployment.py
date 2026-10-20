# SPDX-License-Identifier: Apache-2.0
"""Contracts, placement, and real coordinator fan-out/join without model weights."""

import asyncio
import pickle
import sys
from graphlib import CycleError

import pytest
import torch
import zmq
import zmq.asyncio

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
from sglang.multimodal_gen.runtime.disaggregation.deployment.runtime import (
    DeploymentServer,
)


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
    }
    tensors, scalar = pack_fields(values)
    restored = unpack_fields(tensors, scalar)
    torch.testing.assert_close(restored["extra.text"][0], values["extra.text"][0])
    assert restored["extra.image"] is None
    with pytest.raises(TypeError):
        pack_fields({"required": object()})


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
            result = await server.execute(None)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x"]))
