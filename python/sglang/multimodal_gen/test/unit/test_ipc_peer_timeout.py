# SPDX-License-Identifier: Apache-2.0
"""An expired Ulysses peer wait fails its request after the denoising stage and
marks the server unusable (the kernels: single_test_file/test_ipc_a2a_multi_gpu.py)."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor as parallel_executor_module
import sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor as executor_module
import sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor as sync_executor_module
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import (
    TIMEOUT_EXCHANGE,
    TIMEOUT_PIPELINE_INPUT,
    TIMEOUT_PIPELINE_OUTPUT,
    IpcPeerTimeoutError,
    describe_peer_timeouts,
)
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
    ParallelExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient

RANKS = list(range(8))


def test_message_names_ranks_peers_paths_and_switches() -> None:
    # ranks 0-2 and 4-6 waited for rank 3's input rows in the head-group
    # pipeline, rank 7 for its output rows; rank 3 itself saw every peer arrive
    codes = [TIMEOUT_PIPELINE_INPUT + 3] * 3 + [0] + [TIMEOUT_PIPELINE_INPUT + 3] * 3
    codes.append(TIMEOUT_PIPELINE_OUTPUT + 3)
    error = describe_peer_timeouts(codes, RANKS, 10.0)
    assert error.startswith("Ulysses IPC peer wait timed out after 10 s: ")
    assert (
        "ranks 0, 1, 2, 4, 5, 6 waited for rank 3 in the head-group pipeline (input "
        "rows); rank 7 waited for rank 3 in the head-group pipeline (output rows)."
        in error
    )
    assert "restart the server's process group" in error
    assert error.endswith(
        "turn the path off with SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS=0."
    )


def test_message_decodes_every_site() -> None:
    def single(code: int) -> str:
        return describe_peer_timeouts([code, 0], [4, 5], 0.25)

    # the plain exchange's spin flags both sides and does not record the peer
    error = single(TIMEOUT_EXCHANGE)
    assert "after 0.25 s: rank 4 waited for a peer in the copy-engine all-to-all." in (
        error
    )
    assert error.endswith("with SGLANG_DIFFUSION_IPC_A2A_MULTI=0.")
    assert "rank 4 waited for rank 5 in the head-group pipeline (input rows)" in (
        single(TIMEOUT_PIPELINE_INPUT + 1)
    )
    error = single(TIMEOUT_PIPELINE_OUTPUT + 1)
    assert "rank 4 waited for rank 5 in the head-group pipeline (output rows)" in error
    assert error.endswith("with SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS=0.")


class _Stage:
    def __init__(self, role: RoleType, parallelism=StageParallelismType.REPLICATED):
        self.role_affinity = role
        self._parallelism = parallelism

    def parallelism_type_for(self, batch) -> StageParallelismType:
        return self._parallelism


def _counting_executor(executor_cls):
    """An executor whose stages add one to the payload on the ranks that run them."""
    executor = executor_cls.__new__(executor_cls)
    executor._component_residency_request = lambda *args: nullcontext()
    executor._should_use_stage_nvtx = lambda *args: False
    executor._run_stage_with_executor_hooks = lambda stage, index, payload, *args: (
        payload + 1
    )
    return executor


def test_check_runs_after_denoising_stages_only(monkeypatch) -> None:
    check = Mock()
    monkeypatch.setattr(executor_module.IPC_A2A_MULTI, "check_timeout", check)
    monkeypatch.setattr(sync_executor_module, "raise_if_cancelled", lambda reqs: None)
    executor = _counting_executor(SyncExecutor)
    for role, checks in ((RoleType.ENCODER, 0), (RoleType.DENOISER, 1)):
        check.reset_mock()
        out = executor._run_profile_all_stages(
            [_Stage(role)], 0, SimpleNamespace(), None
        )
        assert out == 1 and check.call_count == checks


def test_ranks_that_skip_a_main_rank_stage_still_join_the_check(monkeypatch) -> None:
    # check_timeout is a group collective. A denoising stage can run on the main
    # rank alone (LTX-2's second stage under CFG parallel) while the transport
    # is up over every rank, so the ranks that skip the stage must reach it too.
    check = Mock()
    monkeypatch.setattr(executor_module.IPC_A2A_MULTI, "check_timeout", check)
    monkeypatch.setattr(
        parallel_executor_module, "raise_if_cancelled", lambda reqs: None
    )
    monkeypatch.setattr(parallel_executor_module, "get_cfg_group", lambda: None)
    monkeypatch.setattr(parallel_executor_module, "get_world_group", lambda: None)
    monkeypatch.setattr(parallel_executor_module.torch.distributed, "barrier", Mock())
    executor = _counting_executor(ParallelExecutor)
    stage = _Stage(RoleType.DENOISER, StageParallelismType.MAIN_RANK_ONLY)
    for rank, stages_run in ((0, 1), (1, 0)):
        check.reset_mock()
        monkeypatch.setattr(parallel_executor_module, "get_world_rank", lambda: rank)
        out = executor._execute_stages(
            [stage], 0, SimpleNamespace(enable_cfg_parallel=False), None
        )
        assert out == stages_run and check.call_count == 1


@pytest.mark.parametrize("reported_by", ["stage", "forward", "forward_after_streamed"])
def test_timed_out_request_fails_before_saving(
    monkeypatch, tmp_path, reported_by
) -> None:
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    worker._realtime_sessions = SimpleNamespace(attach=Mock())
    worker._release_warmup_pool = Mock()
    worker._record_output_peak_memory = Mock()
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    error = describe_peer_timeouts([TIMEOUT_EXCHANGE] * 2, [0, 1], 10.0)
    timeout = Mock(side_effect=IpcPeerTimeoutError(error))
    # a decode that encodes while it decodes writes the file inside the forward
    streamed = tmp_path / "r1.mp4"
    if reported_by == "stage":
        # the executor's check after a denoising stage raises inside the forward
        forward = timeout
    else:
        # no stage reported it, so the check after the forward does
        forward = OutputBatch
        if reported_by == "forward_after_streamed":
            streamed.write_bytes(b"incomplete")
            forward = partial(OutputBatch, output_file_paths=[str(streamed)])
        monkeypatch.setattr(executor_module.IPC_A2A_MULTI, "check_timeout", timeout)

    save = Mock()
    result = worker._execute_forward_common(
        SimpleNamespace(is_warmup=False, extra={}),
        forward_fn=forward,
        log_reqs=[],
        return_req=False,
        save_output_paths=save,
        error_context="request r1",
    )
    timeout.assert_called_once()
    save.assert_not_called()
    assert not streamed.exists()
    assert result.error == f"Error executing request r1: {error}"
    assert result.fatal_error == error


def test_fatal_reply_turns_health_unhealthy() -> None:
    error = describe_peer_timeouts([TIMEOUT_EXCHANGE] * 2, [0, 1], 10.0)
    client = AsyncSchedulerClient()
    client.context = object()
    client.server_args = SimpleNamespace(scheduler_endpoints=["ipc://replica-0"])
    client.request_logger = MagicMock()

    async def forward_one(endpoint, batch, timeout_ms):
        return OutputBatch(error=f"Error executing request r1: {error}", **reply)

    client._forward_one = forward_one
    reply = {}
    asyncio.run(client.forward(object()))
    assert client.failure_reason is None
    reply = {"fatal_error": error}
    result = asyncio.run(client.forward(object()))
    assert result.error.endswith(error) and client.failure_reason == error

    state = SimpleNamespace(
        server_warmup_done=asyncio.Event(),
        worker_processes=[],
        disagg_server=None,
        broker_task=None,
    )
    state.server_warmup_done.set()
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(http_server, "async_scheduler_client", client)
        response = asyncio.run(http_server.health(request))
    assert response.status_code == 503
    assert json.loads(response.body) == {"status": "unhealthy", "error": error}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
