from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)


def _distributed_req(selector="all"):
    return UpdateWeightsFromDistributedReqInput(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        flush_cache=False,
        selector=selector,
    )


def _disk_req(flush_cache=True):
    return UpdateWeightFromDiskReqInput(model_path="/tmp/model", flush_cache=flush_cache)


def _manager(tp_worker, draft_worker):
    # metrics_collector defaults to None, so _observe_weight_load is a no-op and the
    # unused disagg/memory hooks below are never exercised by these tests.
    manager = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
        disaggregation_mode=None,
        get_disagg_decode_prealloc_queue=Mock(return_value=None),
        get_disagg_prefill_bootstrap_queue=Mock(return_value=None),
    )
    # update_weights_from_* assert an open begin_weight_update session.
    manager._weight_update_in_progress = True
    return manager


def test_scheduler_distributed_update_receives_once_on_target_loads_into_each():
    # Default selector ("all"): only the target (main model) owns the update group,
    # so it receives the broadcast once; that single weights object is then loaded
    # into every selected runner — receive once on the target, load into each.
    weights = object()
    target_runner = Mock()
    target_runner.receive_weights_from_distributed.return_value = weights
    draft_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            iter_runners=lambda: [("", target_runner)],
        ),
        draft_worker=SimpleNamespace(iter_runners=lambda: [("draft", draft_runner)]),
    )

    output = manager.update_weights_from_distributed(_distributed_req())

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


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    # selector="target": the target still receives once, but the draft worker is
    # never enumerated and no draft runner is loaded.
    weights = object()
    target_runner = Mock()
    target_runner.receive_weights_from_distributed.return_value = weights
    draft_worker = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            iter_runners=lambda: [("", target_runner)],
        ),
        draft_worker=draft_worker,
    )

    output = manager.update_weights_from_distributed(
        _distributed_req(selector="target")
    )

    assert output.success is True
    target_runner.receive_weights_from_distributed.assert_called_once()
    target_runner.load_weights.assert_called_once_with(weights)
    draft_worker.iter_runners.assert_not_called()


def _disk_manager(target_runner, draft_runner):
    return _manager(
        tp_worker=SimpleNamespace(iter_runners=lambda: [("", target_runner)]),
        draft_worker=SimpleNamespace(iter_runners=lambda: [("draft", draft_runner)]),
    )


def test_disk_update_fans_out_to_all_runners_and_flushes():
    # disk update reaches both the target and the draft runner via get_model_runners,
    # and the target's success drives the post-update cache flush.
    target_runner = Mock()
    target_runner.update_weights_from_disk.return_value = (True, "ok")
    draft_runner = Mock()
    draft_runner.update_weights_from_disk.return_value = (True, "ok")
    manager = _disk_manager(target_runner, draft_runner)

    output = manager.update_weights_from_disk(_disk_req(flush_cache=True))

    assert output.success is True
    # Both runners load from the same recv_req.model_path.
    target_runner.update_weights_from_disk.assert_called_once()
    draft_runner.update_weights_from_disk.assert_called_once()
    manager.flush_cache.assert_called_once()


def test_disk_update_flushes_on_target_success_even_when_draft_fails():
    # Pre-refactor behavior preserved: the target's KV cache is flushed off the
    # target's own success, independent of a later draft failure; the overall result
    # still reports the failure.
    target_runner = Mock()
    target_runner.update_weights_from_disk.return_value = (True, "ok")
    draft_runner = Mock()
    draft_runner.update_weights_from_disk.return_value = (False, "draft boom")
    manager = _disk_manager(target_runner, draft_runner)

    output = manager.update_weights_from_disk(_disk_req(flush_cache=True))

    assert output.success is False
    assert "draft boom" in output.message
    manager.flush_cache.assert_called_once()


def test_disk_update_target_failure_short_circuits_draft_and_flush():
    # The target runs first; its failure stops the fanout before the draft and skips
    # the flush (target never succeeded).
    target_runner = Mock()
    target_runner.update_weights_from_disk.return_value = (False, "target boom")
    draft_runner = Mock()
    manager = _disk_manager(target_runner, draft_runner)

    output = manager.update_weights_from_disk(_disk_req(flush_cache=True))

    assert output.success is False
    assert "target boom" in output.message
    draft_runner.update_weights_from_disk.assert_not_called()
    manager.flush_cache.assert_not_called()
