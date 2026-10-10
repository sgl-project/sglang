import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    InitWeightsUpdateGroupReqInput,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training import scheduler_post_training_mixin
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient


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
        req = InitWeightsUpdateGroupReqInput(
            master_address="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=3,
            group_name="update",
        )
        result = await asyncio.wait_for(client.forward(req), timeout=2)
        assert set(entered) == {"a", "b"}
        assert result.error == "rank failure"

    asyncio.run(run())


@pytest.fixture
def two_rank_scheduler(monkeypatch):
    module = scheduler_post_training_mixin
    monkeypatch.setattr(
        module,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, cpu_group="world"),
    )
    gathered = []

    def gather(results, result, group):
        gathered.append(result)
        results[:] = [result, (False, "load failed")]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    return module.SchedulerPostTrainingMixin(), gathered


def test_non_primary_worker_failure_reaches_http_result(two_rank_scheduler):
    scheduler, _ = two_rank_scheduler
    result = scheduler._run_on_all_ranks(lambda req: (True, "loaded"), None)
    assert not result.output["success"]
    assert result.error == "Rank 1: load failed"


def test_raising_rank_still_joins_the_gather(two_rank_scheduler):
    scheduler, gathered = two_rank_scheduler

    def op(req):
        raise RuntimeError("out of memory")

    result = scheduler._run_on_all_ranks(op, None)
    assert gathered == [(False, "RuntimeError: out of memory")]
    assert result.error == "Rank 0: RuntimeError: out of memory"
