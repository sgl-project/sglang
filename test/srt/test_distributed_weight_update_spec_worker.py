from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqInput
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import MultiLayerEagleWorkerV2


class _TestScheduler(SchedulerUpdateWeightsMixin):
    pass


def _distributed_req():
    return UpdateWeightsFromDistributedReqInput(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        flush_cache=False,
    )


def test_scheduler_routes_distributed_weight_update_to_spec_worker():
    scheduler = _TestScheduler()
    scheduler.enable_overlap = False
    scheduler.schedule_stream = Mock()
    scheduler.tp_cpu_group = object()
    scheduler.tp_worker = Mock()
    scheduler.draft_worker = Mock()
    scheduler.draft_worker.update_weights_from_distributed.return_value = (
        True,
        "updated",
    )

    with patch(
        "sglang.srt.managers.scheduler_update_weights_mixin.torch.distributed.barrier"
    ) as barrier:
        output = SchedulerUpdateWeightsMixin.update_weights_from_distributed(
            scheduler, _distributed_req()
        )

    assert output.success is True
    scheduler.draft_worker.update_weights_from_distributed.assert_called_once()
    scheduler.tp_worker.update_weights_from_distributed.assert_not_called()
    assert barrier.call_count == 2


def test_model_runner_loads_one_distributed_receive_into_multiple_runners():
    weights = [("model.layers.0.weight", object())]
    source_runner = SimpleNamespace(
        _model_update_group={"weight_update_group": object()},
        _receive_weights_from_distributed=Mock(return_value=weights),
    )
    draft_runner = SimpleNamespace(model=Mock())
    target_runner = SimpleNamespace(model=Mock())

    success, message = ModelRunner.update_weights_from_distributed_to_model_runners(
        source_runner,
        [draft_runner, target_runner],
        ["model.layers.0.weight"],
        ["float32"],
        [[1]],
        "weight_update_group",
    )

    assert success is True
    assert message == "Succeeded to update parameter online."
    source_runner._receive_weights_from_distributed.assert_called_once()
    draft_runner.model.load_weights.assert_called_once_with(weights)
    target_runner.model.load_weights.assert_called_once_with(weights)


def test_eagle_v2_distributed_update_loads_draft_and_target_from_target_group():
    req = _distributed_req()
    target_runner = Mock()
    target_runner.update_weights_from_distributed_to_model_runners.return_value = (
        True,
        "updated",
    )
    draft_runner = object()
    worker = object.__new__(EAGLEWorkerV2)
    worker._target_worker = SimpleNamespace(model_runner=target_runner)
    worker._draft_worker = SimpleNamespace(draft_runner=draft_runner)

    success, message = worker.update_weights_from_distributed(req)

    assert success is True
    assert message == "updated"
    target_runner.update_weights_from_distributed_to_model_runners.assert_called_once_with(
        [draft_runner, target_runner],
        req.names,
        req.dtypes,
        req.shapes,
        req.group_name,
        req.load_format,
    )


def test_multi_layer_eagle_v2_distributed_update_loads_all_drafts_and_target():
    req = _distributed_req()
    target_runner = Mock()
    target_runner.update_weights_from_distributed_to_model_runners.return_value = (
        True,
        "updated",
    )
    draft_runners = [object(), object()]
    worker = object.__new__(MultiLayerEagleWorkerV2)
    worker._target_worker = SimpleNamespace(model_runner=target_runner)
    worker._draft_worker = SimpleNamespace(draft_runner_list=draft_runners)

    success, message = worker.update_weights_from_distributed(req)

    assert success is True
    assert message == "updated"
    target_runner.update_weights_from_distributed_to_model_runners.assert_called_once_with(
        draft_runners + [target_runner],
        req.names,
        req.dtypes,
        req.shapes,
        req.group_name,
        req.load_format,
    )
