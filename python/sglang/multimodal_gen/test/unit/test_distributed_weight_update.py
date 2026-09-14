import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training import scheduler_post_training_mixin
from sglang.multimodal_gen.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClient,
)


def metadata(**overrides):
    body = dict(
        names=["weight"],
        dtypes=["float32"],
        shapes=[[2, 3]],
        group_name="update",
        target_modules=["transformer"],
    )
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "overrides",
    [
        {"names": []},
        {"dtypes": []},
        {"shapes": []},
        {"dtypes": ["invalid"]},
        {"shapes": [[-1, 3]]},
        {"target_modules": []},
        {"target_modules": ["a", "b"]},
        {"group_name": ""},
        {"names": ["a", "a"], "dtypes": ["float32"] * 2, "shapes": [[1], [1]]},
        {"weight_update_mode": "invalid"},
        {"lora_rank": 0},
    ],
)
def test_invalid_metadata(overrides):
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(
            metadata(**overrides), type=UpdateWeightsFromDistributedReqInput
        )


def test_metadata_preserves_tensor_order():
    body = metadata(
        names=["b", "a"], dtypes=["bfloat16", "float32"], shapes=[[3, 2], []]
    )
    req = msgspec.convert(body, type=UpdateWeightsFromDistributedReqInput)
    assert list(zip(req.names, req.dtypes, req.shapes, strict=True)) == [
        ("b", "bfloat16", [3, 2]),
        ("a", "float32", []),
    ]


def test_collective_request_reaches_dp_replicas_concurrently():
    async def run():
        client = AsyncSchedulerClient()
        client.context = object()
        client.request_logger = MagicMock()
        client.server_args = SimpleNamespace(scheduler_endpoints=["a", "b"])
        entered = []
        ready = asyncio.Event()

        async def forward(endpoint, batch, timeout):
            entered.append(endpoint)
            if len(entered) == 2:
                ready.set()
            await ready.wait()
            return OutputBatch(
                output={"success": endpoint == "a"},
                error="rank failure" if endpoint == "b" else None,
            )

        client._forward_one = forward
        req = InitWeightsUpdateGroupReqInput("127.0.0.1", 12345, 1, 3, "update")
        result = await asyncio.wait_for(client.forward(req), timeout=2)
        assert set(entered) == {"a", "b"}
        assert result.error == "rank failure"

    asyncio.run(run())


def test_sync_collective_request_reaches_dp_replicas_concurrently():
    client = SchedulerClient()
    client.request_logger = MagicMock()
    client.server_args = SimpleNamespace(scheduler_endpoints=["a", "b"])
    barrier = threading.Barrier(2, timeout=2)

    def forward(endpoint, batch, timeout):
        barrier.wait()
        return OutputBatch(output={"success": True})

    client._forward_one = forward
    req = InitWeightsUpdateGroupReqInput("127.0.0.1", 12345, 1, 3, "update")
    assert client.forward(req).output["success"]


def test_non_primary_worker_failure_reaches_http_result(monkeypatch):
    module = scheduler_post_training_mixin
    monkeypatch.setattr(
        module,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, cpu_group="world"),
    )

    def gather(results, result, group):
        results[:] = [result, (False, "load failed")]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    scheduler = module.SchedulerPostTrainingMixin()
    result = scheduler._weight_group_result((True, "loaded"))
    assert not result.output["success"]
    assert result.error == "Rank 1: load failed"
