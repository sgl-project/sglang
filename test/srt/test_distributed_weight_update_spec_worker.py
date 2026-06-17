from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqInput
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)


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


def test_scheduler_distributed_update_receives_once_on_target_loads_into_each():
    # Default selector ("both"): only the target (main model) owns the update group,
    # so it receives the broadcast once; that single weights object is then loaded
    # into every selected runner — receive once on the target, load into each.
    weights = object()
    target_runner = Mock()
    target_runner.receive_weights_from_distributed.return_value = weights
    draft_runner = Mock()
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
    target_runner.receive_weights_from_distributed.assert_called_once_with(
        ["model.layers.0.weight"],
        ["float32"],
        [[1]],
        "weight_update_group",
        None,
    )
    # The single received weights object is loaded into every selected runner.
    target_runner.load_weights.assert_called_once_with(weights)
    draft_runner.load_weights.assert_called_once_with(weights)
    assert barrier.call_count == 2


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    # selector="target": the target still receives once, but the draft worker is
    # never enumerated and no draft runner is loaded.
    weights = object()
    target_runner = Mock()
    target_runner.receive_weights_from_distributed.return_value = weights
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
    target_runner.receive_weights_from_distributed.assert_called_once()
    target_runner.load_weights.assert_called_once_with(weights)
    draft_worker.iter_draft_runners.assert_not_called()
