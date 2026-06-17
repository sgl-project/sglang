from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqInput
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.model_executor.model_runner import ModelRunner


class _TestScheduler(SchedulerUpdateWeightsMixin):
    pass


def _distributed_req(selector="both"):
    return UpdateWeightsFromDistributedReqInput(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        flush_cache=False,
        selector=selector,
    )


def _scheduler(tp_worker, draft_worker):
    scheduler = _TestScheduler()
    scheduler.enable_overlap = False
    scheduler.schedule_stream = Mock()
    scheduler.tp_cpu_group = object()
    scheduler.tp_worker = tp_worker
    scheduler.draft_worker = draft_worker
    return scheduler


def test_scheduler_distributed_update_fans_out_to_target_and_draft():
    # Default selector ("both"): the scheduler resolves the runner list via
    # get_model_runners; the target runner owns the update group, receives the
    # broadcast once, and loads it into both the target and the draft runner.
    target_runner = Mock()
    target_runner.update_weights_from_distributed_to_model_runners.return_value = (
        True,
        "updated",
    )
    draft_runner = object()
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=SimpleNamespace(
            iter_draft_runners=lambda: [("draft", draft_runner)]
        ),
    )

    with patch(
        "sglang.srt.managers.scheduler_update_weights_mixin.torch.distributed.barrier"
    ) as barrier:
        output = SchedulerUpdateWeightsMixin.update_weights_from_distributed(
            scheduler, _distributed_req()
        )

    assert output.success is True
    target_runner.update_weights_from_distributed_to_model_runners.assert_called_once_with(
        [target_runner, draft_runner],
        ["model.layers.0.weight"],
        ["float32"],
        [[1]],
        "weight_update_group",
        None,
    )
    assert barrier.call_count == 2


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    # selector="target" resolves to the target runner alone; the draft worker is
    # never enumerated.
    target_runner = Mock()
    target_runner.update_weights_from_distributed_to_model_runners.return_value = (
        True,
        "updated",
    )
    draft_worker = Mock()
    scheduler = _scheduler(
        tp_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=draft_worker,
    )

    with patch(
        "sglang.srt.managers.scheduler_update_weights_mixin.torch.distributed.barrier"
    ):
        output = SchedulerUpdateWeightsMixin.update_weights_from_distributed(
            scheduler, _distributed_req(selector="target")
        )

    assert output.success is True
    target_runner.update_weights_from_distributed_to_model_runners.assert_called_once_with(
        [target_runner],
        ["model.layers.0.weight"],
        ["float32"],
        [[1]],
        "weight_update_group",
        None,
    )
    draft_worker.iter_draft_runners.assert_not_called()


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
