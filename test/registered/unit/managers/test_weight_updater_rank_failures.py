from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)


def _manager(worker):
    return SchedulerWeightUpdaterManager(
        tp_worker=worker,
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
    )


def _request(flush_cache=False):
    return SimpleNamespace(
        disable_draft_model=True,
        flush_cache=flush_cache,
        torch_empty_cache=False,
    )


@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.all_gather_object"
)
@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.get_world_size",
    return_value=2,
)
def test_update_weights_from_tensor_propagates_rank_failure(
    _world_size, all_gather_object
):
    worker = Mock()
    worker.update_weights_from_tensor.return_value = (True, "rank 0 ok")
    manager = _manager(worker)

    def gather(statuses, _local_status, group):
        statuses[:] = [(True, "rank 0 ok"), (False, "rank 1 failed")]

    all_gather_object.side_effect = gather

    result = manager.update_weights_from_tensor(_request())

    assert result.success is False
    assert result.message == "Weight update failed on TP rank 1: rank 1 failed"


@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.all_gather_object"
)
@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.get_world_size",
    return_value=2,
)
def test_update_weights_from_tensor_collects_local_exception(
    _world_size, all_gather_object
):
    worker = Mock()
    worker.update_weights_from_tensor.side_effect = RuntimeError("local boom")
    manager = _manager(worker)

    def gather(statuses, local_status, group):
        statuses[:] = [local_status, (True, "rank 1 ok")]

    all_gather_object.side_effect = gather

    result = manager.update_weights_from_tensor(_request())

    assert result.success is False
    assert "TP rank 0:" in result.message
    assert "RuntimeError: local boom" in result.message


@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.all_gather_object"
)
@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.get_world_size",
    return_value=2,
)
def test_update_weights_from_tensor_flushes_locally_before_aggregating_peer_failure(
    _world_size, all_gather_object
):
    worker = Mock()
    worker.update_weights_from_tensor.return_value = (True, "rank 0 ok")
    manager = _manager(worker)

    def gather(statuses, _local_status, group):
        manager.flush_cache.assert_called_once_with(empty_cache=False)
        statuses[:] = [(True, "rank 0 ok"), (False, "rank 1 failed")]

    all_gather_object.side_effect = gather

    result = manager.update_weights_from_tensor(_request(flush_cache=True))

    assert result.success is False
    assert "TP rank 1: rank 1 failed" in result.message


@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.all_gather_object"
)
@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.get_world_size",
    return_value=2,
)
def test_update_weights_from_tensor_propagates_flush_exception(
    _world_size, all_gather_object
):
    worker = Mock()
    worker.update_weights_from_tensor.return_value = (True, "rank 0 ok")
    manager = _manager(worker)
    manager.flush_cache.return_value = False

    def gather(statuses, local_status, group):
        statuses[:] = [local_status, (True, "rank 1 ok")]

    all_gather_object.side_effect = gather

    result = manager.update_weights_from_tensor(_request(flush_cache=True))

    assert result.success is False
    assert "TP rank 0:" in result.message
    assert "Cache flush failed after updating weights" in result.message


@patch(
    "sglang.srt.managers.scheduler_components.weight_updater.torch.distributed.get_world_size",
    return_value=1,
)
def test_update_weights_from_tensor_tp1_success(_world_size):
    worker = Mock()
    worker.update_weights_from_tensor.return_value = (True, "ok")
    manager = _manager(worker)

    result = manager.update_weights_from_tensor(_request(flush_cache=True))

    assert result.success is True
    assert result.message == "ok"
    manager.flush_cache.assert_called_once_with(empty_cache=False)
