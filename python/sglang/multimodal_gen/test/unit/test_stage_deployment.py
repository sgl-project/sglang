# SPDX-License-Identifier: Apache-2.0
"""Contracts, placement, and real coordinator fan-out/join without model weights."""

import asyncio
import json
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
    capture_request,
    pack_fields,
    restore_request,
    unpack_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.plan import (
    REQUEST_CONTEXT,
    Deployment,
    ExecutionPlan,
    RoleSpec,
    StageSpec,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import rpc
from sglang.multimodal_gen.runtime.disaggregation.deployment.worker import (
    DeploymentWorker,
    run_partition,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    TransferTensorBuffer,
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


def test_automatic_context_compiles_three_separate_roles():
    stages = (
        StageSpec("encode", components=("text_encoder", "tokenizer")),
        StageSpec("denoise", components=("transformer", "scheduler")),
        StageSpec("decode", components=("vae",)),
    )
    plan = ExecutionPlan.compile(
        Deployment(
            {
                name: RoleSpec((index,), (f"tcp://localhost:{19000 + index}",))
                for index, name in enumerate(("encoder", "denoiser", "decoder"))
            }
        ),
        stages,
    )
    assert plan.order == ("encoder", "denoiser", "decoder")
    assert plan.roles["denoiser"].inputs == {REQUEST_CONTEXT: "encoder"}
    assert plan.roles["decoder"].inputs == {REQUEST_CONTEXT: "denoiser"}
    assert plan.roles["encoder"].components == {"text_encoder", "tokenizer"}
    assert plan.outputs == {"result.output": "decoder"}
    assert REQUEST_CONTEXT not in plan.roles["decoder"].outputs


def test_request_context_preserves_nested_state_rng_and_aliases():
    req = Req(prompt="original", width=512, height=512)
    req.width = 640
    generator = torch.Generator().manual_seed(42)
    torch.randn(7, generator=generator)
    req.generator = [generator, generator]
    req.prompt_embeds = [torch.arange(6).reshape(2, 3)]
    req.extra["model_specific"] = {"mask": req.prompt_embeds[0], "sizes": (640, 512)}
    req.scheduler = SimpleNamespace(sigmas=torch.linspace(1, 0, 4), step_index=2)
    context = capture_request(req)
    # the control-plane structure contains tensor references, never tensor storage
    assert len(context["tensors"]) == 3
    tensors, scalar = pack_fields({REQUEST_CONTEXT: context})
    received = unpack_fields(tensors, scalar)[REQUEST_CONTEXT]
    restored = Req(prompt="receiver")
    metrics = restored.metrics
    restore_request(restored, received, "cpu")
    assert restored.width == 640
    assert restored.extra["model_specific"]["sizes"] == (640, 512)
    assert restored.extra["model_specific"]["mask"] is restored.prompt_embeds[0]
    assert restored.generator[0] is restored.generator[1]
    assert restored.metrics is metrics
    assert restored.scheduler.step_index == 2
    torch.testing.assert_close(restored.scheduler.sigmas, req.scheduler.sigmas)
    torch.testing.assert_close(
        torch.randn(16, generator=restored.generator[0]),
        torch.randn(16, generator=req.generator[0]),
        rtol=0,
        atol=0,
    )
    req.extra["module"] = torch.nn.Linear(2, 2)
    with pytest.raises(TypeError, match="Model modules"):
        capture_request(req)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor(3),
        torch.tensor(0.5),
        torch.tensor(True),
        torch.arange(6).reshape(2, 3).T,
    ],
)
def test_context_tensor_buffer_handles_scalars_and_views(tensor):
    buffer = object.__new__(TransferTensorBuffer)
    buffer._pool = torch.empty(64, dtype=torch.uint8)
    slot = SlotHandle(request_id="context", offset=0, size=64)
    assert buffer.write_tensor(slot, tensor) == tensor.numel() * tensor.element_size()
    restored = buffer.read_tensor(slot, list(tensor.shape), tensor.dtype)
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


@pytest.mark.parametrize("selectors", [(0, 1, 2, 3), ("text", 1, "denoise", 3)])
def test_stage_indices_resolve_to_registered_names(tmp_path, selectors):
    path = tmp_path / "deployment.json"
    path.write_text(
        json.dumps(
            {
                "roles": {
                    stage.name: {
                        "stages": [selector],
                        "endpoints": [f"tcp://localhost:{19000 + index}"],
                    }
                    for index, (stage, selector) in enumerate(zip(specs(), selectors))
                }
            }
        )
    )
    plan = ExecutionPlan.compile(Deployment.load(path), specs())
    assert [role.stages for role in plan.roles.values()] == [
        (stage.name,) for stage in specs()
    ]
    assert plan.roles["denoise"].dependencies == {"text", "image"}
    # indices refer to the full contract list, not the selected subgraph
    subset = ExecutionPlan.compile(
        Deployment(
            {"image": RoleSpec((1,), ("tcp://localhost:1",))},
            outputs=("extra.image",),
        ),
        specs(),
    )
    assert subset.roles["image"].stages == ("image",)
    grouped = ExecutionPlan.compile(
        Deployment({"worker": RoleSpec((3, 1, 0, 2), ("tcp://localhost:1",))}),
        specs(),
    )
    assert grouped.roles["worker"].stages == tuple(stage.name for stage in specs())


@pytest.mark.parametrize(
    "selectors", [(-1,), (4,), (True,), (1.0,), ("0",), (0, "text")]
)
def test_invalid_stage_indices_rejected(selectors):
    with pytest.raises(ValueError):
        ExecutionPlan.compile(
            Deployment({"worker": RoleSpec(selectors, ("tcp://localhost:1",))}),
            specs(),
        )


def test_worker_retains_reusable_model_outputs(monkeypatch):
    for method in ("set_device", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, method, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1024**2)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None)
    )
    buffer = torch.zeros(2)

    def forward(req, args):
        buffer.add_(1)
        req.latents = buffer
        return req

    worker = object.__new__(DeploymentWorker)
    worker.device = "cpu"
    worker.role = SimpleNamespace(outputs=("latents",))
    worker.scheduler = SimpleNamespace(
        server_args=None,
        worker=SimpleNamespace(pipeline=SimpleNamespace(forward=forward)),
        _is_multi_rank=lambda: False,
    )
    first, _, _, peak = worker._execute(Req(prompt="first"))
    second, *_ = worker._execute(Req(prompt="second"))
    torch.testing.assert_close(first["latents"], torch.ones(2))
    torch.testing.assert_close(second["latents"], torch.full((2,), 2.0))
    assert peak == 1


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


@pytest.mark.parametrize("colocate_writes", [False, True])
def test_exports_use_the_producing_roles_final_writer(colocate_writes):
    stages = (
        StageSpec("first", outputs=("extra.value",)),
        StageSpec("consume", inputs=("extra.value",), outputs=("result.output",)),
        StageSpec("last", outputs=("extra.value",)),
    )
    roles = {
        "producer": RoleSpec(
            ("first", "last") if colocate_writes else ("first",), ("tcp://localhost:1",)
        ),
        "consumer": RoleSpec(("consume",), ("tcp://localhost:2",)),
    }
    if colocate_writes:
        with pytest.raises(ValueError, match="overwritten version"):
            ExecutionPlan.compile(Deployment(roles), stages)
    else:
        roles["later"] = RoleSpec(("last",), ("tcp://localhost:3",))
        plan = ExecutionPlan.compile(Deployment(roles), stages)
        assert plan.roles["consumer"].inputs == {"extra.value": "producer"}


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
        server = DeploymentServer(plan, timeout=5, capacity=1)
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
            control = context.socket(zmq.ROUTER)
            sockets.append(control)
            port = control.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"

            async def control_peer():
                while True:
                    frames = await control.recv_multipart()
                    await server._handle(control, frames[:-1], pickle.loads(frames[-1]))

            tasks.append(asyncio.create_task(control_peer()))
            descriptor = await rpc(
                context,
                endpoint,
                {
                    "method": "encode",
                    "job_id": "leased",
                    "request": Req(prompt="leased"),
                    "outputs": None,
                },
                5,
            )
            assert server.admission.locked()
            for _ in range(2):
                await rpc(
                    context,
                    endpoint,
                    {"method": "release", "lease": descriptor["lease"]},
                    5,
                )
            assert not live and not server.admission.locked()
            await server.admission.acquire()
            assert server.admission.locked()
            server.admission.release()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for socket in sockets:
                socket.close()
            context.term()

    asyncio.run(run())


def test_empty_partition_error_is_not_success(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    def fail(*args):
        raise RuntimeError("")

    scheduler = SimpleNamespace(
        worker=SimpleNamespace(pipeline=SimpleNamespace(forward=fail)),
        server_args=None,
        _is_multi_rank=lambda: False,
    )
    assert run_partition(scheduler, SimpleNamespace(extra={})) == "RuntimeError"


@pytest.mark.parametrize("error", ["invalid encoder input", ""])
def test_client_failure_cancels_outstanding_rpc(monkeypatch, error):
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
                    reply = {"error": error}
                else:
                    reply = {}
                await socket.send_multipart([*frames[:-1], pickle.dumps(reply)])

        task = asyncio.create_task(peer())
        try:
            with pytest.raises(
                RuntimeError, match=error or "Remote deployment request failed"
            ):
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
