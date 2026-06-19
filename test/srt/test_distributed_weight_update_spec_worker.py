from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
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


def _session_manager(target_runner, draft_runner):
    return _manager(
        tp_worker=SimpleNamespace(iter_runners=lambda: [("", target_runner)]),
        draft_worker=SimpleNamespace(iter_runners=lambda: [("draft", draft_runner)]),
    )


def test_begin_weight_update_restores_target_and_draft():
    # The session begins on every runner (target + draft): the draft model is
    # restored to a loadable state identically to the target.
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._weight_update_in_progress = False

    with patch("torch.distributed.barrier"):
        output = manager.begin_weight_update(BeginWeightUpdateReqInput())

    assert output.success is True
    target_runner.begin_weight_update.assert_called_once_with()
    draft_runner.begin_weight_update.assert_called_once_with()
    assert manager._weight_update_in_progress is True
    assert manager._weight_update_loaded is False


def test_end_weight_update_runs_post_load_on_both_when_load_was_bypassed():
    # No load_weights happened this session (e.g. P2P/RDMA), so end runs
    # post_load_weights then quant finalize on BOTH target and draft.
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._weight_update_loaded = False

    with patch("torch.distributed.barrier"):
        output = manager.end_weight_update(EndWeightUpdateReqInput())

    assert output.success is True
    target_runner.end_weight_update.assert_called_once_with(run_post_load=True)
    draft_runner.end_weight_update.assert_called_once_with(run_post_load=True)
    assert manager._weight_update_in_progress is False


def test_end_weight_update_skips_post_load_on_both_when_weights_loaded():
    # A distributed/tensor load happened this session, so post_load is skipped on
    # both runners; only quant finalize runs.
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._weight_update_loaded = True

    with patch("torch.distributed.barrier"):
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.end_weight_update.assert_called_once_with(run_post_load=False)
    draft_runner.end_weight_update.assert_called_once_with(run_post_load=False)


def test_model_runner_begin_end_wire_to_loader_hooks():
    # ModelRunner.begin/end delegate to the loader: begin restores; end runs
    # post_load only when requested, always finalizes quant layout.
    import sglang.srt.model_executor.model_runner as mr

    runner = SimpleNamespace(model=object(), device="cpu")

    with patch.object(mr, "restore_weight") as restore:
        mr.ModelRunner.begin_weight_update(runner)
    restore.assert_called_once()

    with patch.object(mr, "post_load_weights") as post_load, patch.object(
        mr, "postprocess_weight"
    ) as postprocess:
        mr.ModelRunner.end_weight_update(runner, run_post_load=True)
    post_load.assert_called_once()
    postprocess.assert_called_once()

    with patch.object(mr, "post_load_weights") as post_load, patch.object(
        mr, "postprocess_weight"
    ) as postprocess:
        mr.ModelRunner.end_weight_update(runner, run_post_load=False)
    post_load.assert_not_called()
    postprocess.assert_called_once()


def test_begin_weight_update_selector_restores_only_selected_and_is_recorded():
    # begin(selector="draft") opens the session on the draft only; the target is
    # untouched, and the selector is recorded for end to reuse.
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._weight_update_in_progress = False

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))

    target_runner.begin_weight_update.assert_not_called()
    draft_runner.begin_weight_update.assert_called_once_with()
    assert manager._weight_update_selector == "draft"


def test_end_weight_update_reuses_session_selector_from_begin():
    # end has no selector of its own; it finalizes exactly the set begin opened.
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._weight_update_in_progress = False

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.end_weight_update.assert_not_called()
    draft_runner.end_weight_update.assert_called_once()


def test_begin_weight_update_rejects_reentry():
    # A second begin while a session is open would leave the first session's
    # restored runners unfinalized — reject it loudly.
    manager = _session_manager(Mock(), Mock())
    manager._weight_update_in_progress = True

    with patch("torch.distributed.barrier"):
        with pytest.raises(AssertionError, match="already open"):
            manager.begin_weight_update(BeginWeightUpdateReqInput())
