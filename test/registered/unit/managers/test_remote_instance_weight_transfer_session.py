import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin


def _manager(runner, *, remote_weight_transfer_cpu_group=None):
    kwargs = {}
    if remote_weight_transfer_cpu_group is not None:
        kwargs["remote_weight_transfer_cpu_group"] = remote_weight_transfer_cpu_group
    return SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
        **kwargs,
    )


def _manifest(worker_id="source/dp0-pp0-ep0-tp0", lease_id="lease-0"):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "generation": 1,
        "lease_id": lease_id,
        "tensors": [{"worker_id": worker_id}],
    }


def test_remote_transfer_collective_does_not_block_scheduler_thread(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_begin(self, request):
        entered.set()
        assert release.wait(timeout=5)
        return BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=request.transfer_id,
            success=True,
            message="Success.",
            manifests=[_manifest()],
        )

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "begin_remote_instance_weight_transfer",
        blocking_begin,
    )
    manager = _manager(SimpleNamespace())
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        lease_timeout_sec=60,
    )

    assert manager.defer_begin_remote_instance_weight_transfer(request) is None
    assert entered.wait(timeout=1)
    assert manager.check_pending_remote_instance_weight_transfers() == []

    release.set()
    deadline = time.monotonic() + 1
    completed = []
    while not completed and time.monotonic() < deadline:
        completed = manager.check_pending_remote_instance_weight_transfers()
        time.sleep(0.01)

    assert len(completed) == 1
    output, completed_request = completed[0]
    assert output.success is True
    assert completed_request is request
    manager.close_remote_instance_weight_transfer_executor()


def test_begin_and_release_remote_transfer_snapshot(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)

    def all_gather_object(outputs, value, group):
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is True
    assert result.manifests == [manifest]
    assert released == []

    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )
    assert release.success is True
    assert released == ["lease-0"]


def test_begin_uses_dedicated_remote_transfer_cpu_group(monkeypatch) -> None:
    manifest = _manifest()
    remote_group = object()
    observed_groups = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner, remote_weight_transfer_cpu_group=remote_group)

    def get_world_size(group):
        observed_groups.append(("world_size", group))
        return 1

    def all_gather_object(outputs, value, group):
        observed_groups.append(("all_gather", group))
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.get_world_size", get_world_size)
    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is True
    assert observed_groups == [
        ("world_size", remote_group),
        ("all_gather", remote_group),
    ]


def test_begin_rolls_back_local_snapshot_when_another_rank_fails(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def all_gather_object(outputs, value, group):
        outputs[:] = [value, {"success": False, "message": "rank 1 failed"}]

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "rank 1 failed" in result.message
    assert released == ["lease-0"]


def test_begin_rolls_back_local_snapshot_when_collective_fails(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: (_ for _ in ()).throw(
            RuntimeError("collective failed")
        ),
    )

    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "collective failed" in result.message
    assert released == ["lease-0"]


def test_runtime_revision_commit_ignores_workers_without_manifest_support() -> None:
    SchedulerWeightUpdaterManager._commit_weight_runtime_revision(
        SimpleNamespace(model_runner=SimpleNamespace())
    )


def test_release_keeps_snapshot_lease_available_for_retry(monkeypatch) -> None:
    attempts = []

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)

    def all_gather_object(outputs, value, group):
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    def release(lease_id):
        attempts.append(lease_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary release failure")

    manager = _manager(SimpleNamespace(release_weight_runtime_manifest=release))
    manager.remote_weight_transfer_leases["transfer-1"] = "lease-0"
    request = ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")

    first = manager.release_remote_instance_weight_transfer(request)
    assert first.success is False
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    second = manager.release_remote_instance_weight_transfer(request)
    assert second.success is True
    assert attempts == ["lease-0", "lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def _tokenizer_manager(begin_results, release):
    events = []

    async def begin_communicator(request):
        events.append(("begin", request.transfer_id))
        return begin_results

    async def pause(request):
        events.append(("pause", request.mode))

    async def resume(request):
        events.append(("continue", request.torch_empty_cache))

    return SimpleNamespace(
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=True,
            model_path="Qwen/Qwen3.5-0.8B",
            revision="main",
        ),
        auto_create_handle_loop=lambda: None,
        is_pause=False,
        pause_generation=pause,
        continue_generation=resume,
        begin_remote_instance_weight_transfer_communicator=begin_communicator,
        release_remote_instance_weight_transfer_communicator=release,
        _remote_weight_transfer_events=events,
    )


def test_tokenizer_begin_pauses_only_snapshot_capture() -> None:
    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest()],
            )
        ],
        release,
    )

    asyncio.run(TokenizerControlMixin.begin_remote_instance_weight_transfer(manager))

    assert [event[0] for event in manager._remote_weight_transfer_events] == [
        "pause",
        "begin",
        "continue",
    ]
    assert manager._remote_weight_transfer_events[-1] == ("continue", False)


def test_tokenizer_begin_passes_ttl_to_scheduler_without_local_ownership() -> None:
    requests = []

    async def begin(request):
        requests.append(request)
        return [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest()],
            )
        ]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.begin_remote_instance_weight_transfer_communicator = begin

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager, lease_timeout_sec=60
        )
    )

    assert requests[0].lease_timeout_sec == 60
    assert result["lease_timeout_sec"] == 60
    assert not hasattr(manager, "_remote_weight_transfer_timeout_tasks")


def test_tokenizer_release_always_fans_out_without_local_session_state() -> None:
    requests = []

    async def release(request):
        requests.append(request)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    success, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager, "transfer-from-another-worker"
        )
    )

    assert success is True
    assert [request.transfer_id for request in requests] == [
        "transfer-from-another-worker"
    ]


def test_tokenizer_renew_fans_out_without_local_session_state() -> None:
    requests = []

    async def renew(request):
        requests.append(request)
        return [SimpleNamespace(success=True, message="Success.")]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.renew_remote_instance_weight_transfer_communicator = renew

    success, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            manager, "transfer-from-another-worker", lease_timeout_sec=60
        )
    )

    assert success is True
    assert requests == [
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-from-another-worker", lease_timeout_sec=60
        )
    ]


def test_tokenizer_begin_releases_successful_empty_manifest_response() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [SimpleNamespace(success=True, message="Success.", manifests=[])], release
    )

    with pytest.raises(RuntimeError, match="no runtime manifests"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert len(released) == 1


def test_tokenizer_begin_rejects_inconsistent_fanout_manifests() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest(worker_id="source/dp0-pp0-ep0-tp0")],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest(worker_id="source/dp1-pp0-ep0-tp0")],
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="inconsistent runtime manifests"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert len(released) == 1
