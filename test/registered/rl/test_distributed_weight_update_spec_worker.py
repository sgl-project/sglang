from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    InitWeightsUpdateGroupReqInput,
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
    # metrics_collector defaults to None, so _observe_weight_load is a no-op; the
    # reqs below set flush_cache=False so flush_cache is never called either.
    manager = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
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
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
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
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once_with(
        ["model.layers.0.weight"],
        ["float32"],
        [[1]],
        "weight_update_group",
        None,
    )
    # The single received weights object is loaded into every selected runner.
    target_runner.weight_updater.load_weights.assert_called_once_with(weights)
    draft_runner.weight_updater.load_weights.assert_called_once_with(weights)


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    # selector="target": the target still receives once, but the draft worker is
    # never enumerated and no draft runner is loaded.
    weights = object()
    target_runner = Mock()
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
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
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once()
    target_runner.weight_updater.load_weights.assert_called_once_with(weights)
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


def test_m2n_receive_forces_post_load_even_after_residual_broadcast():
    target_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            iter_runners=lambda: [("", target_runner)],
        ),
        draft_worker=None,
    )
    req = _distributed_req(selector="target")
    req.load_format = "nccl_m2n"

    output = manager.update_weights_from_distributed(req)

    assert output.success is True
    target_runner.receive_weights_from_m2n.assert_called_once_with("weight_update_group")
    assert manager._weight_update_requires_post_load is True

    # A later residual broadcast uses load_weights(), but must not erase the
    # model-level post-load requirement established by the direct M2N write.
    manager._weight_update_loaded = True
    with patch("torch.distributed.barrier"):
        manager.end_weight_update(EndWeightUpdateReqInput())
    target_runner.end_weight_update.assert_called_once_with(run_post_load=True)


def test_concurrent_m2n_waves_finalize_once_after_residual_updates():
    target_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            iter_runners=lambda: [("", target_runner)],
        ),
        draft_worker=None,
    )
    for groups in (["pp0", "pp1"], ["pp2", "pp3"]):
        req = _distributed_req(selector="target")
        req.load_format = "nccl_m2n"
        req.group_name = groups[0]
        req.m2n_group_names = groups
        assert manager.update_weights_from_distributed(req).success
        target_runner.end_weight_update.assert_not_called()
    assert target_runner.receive_weights_from_m2n_groups.call_args_list == [
        call(["pp0", "pp1"]),
        call(["pp2", "pp3"]),
    ]
    target_runner.receive_weights_from_m2n.assert_not_called()
    # The residual path still follows the entire bulk update and finalizes once.
    assert manager.update_weights_from_distributed(_distributed_req()).success
    with patch("torch.distributed.barrier"):
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    target_runner.end_weight_update.assert_called_once_with(run_post_load=True)


@pytest.mark.parametrize("groups", [None, ["pp0", "pp1"]])
def test_concurrent_m2n_groups_survive_scheduler_ipc(groups):
    import msgspec

    req = _distributed_req()
    req.m2n_group_names = groups
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=type(req))
    assert decoded.m2n_group_names == groups
    assert decoded.group_name == req.group_name
    assert decoded.flush_cache is False
    # Fields are positional on the wire; older single-group messages should
    # decode with the appended field's default, not shift existing fields.
    legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(req))[:-1]
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(legacy), type=type(req))
    assert decoded.m2n_group_names is None
    assert decoded.selector == req.selector


@pytest.mark.parametrize("groups", [[], ["wrong-first-group"]])
def test_concurrent_m2n_rejects_malformed_request_before_receive(groups):
    target_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(model_runner=target_runner), draft_worker=None
    )
    req = _distributed_req()
    req.load_format = "nccl_m2n"
    req.m2n_group_names = groups
    output = manager.update_weights_from_distributed(req)
    assert not output.success
    target_runner.receive_weights_from_m2n_groups.assert_not_called()


def test_concurrent_m2n_request_cannot_fall_through_to_broadcast():
    target_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(model_runner=target_runner), draft_worker=None
    )
    req = _distributed_req()
    req.m2n_group_names = [req.group_name, "pp1"]
    assert not manager.update_weights_from_distributed(req).success
    target_runner.receive_weights_from_distributed.assert_not_called()


def test_model_runner_resolves_complete_m2n_wave_before_receiving():
    from sglang.srt.model_executor.model_runner import ModelRunner

    receivers = {"pp0": Mock(), "pp1": Mock()}
    runner = SimpleNamespace(_m2n_receivers=receivers)
    with patch(
        "sglang.srt.weight_sync.nccl_m2n.NcclM2NReceiver.receive_many"
    ) as receive:
        for groups in ([], ["pp0", "pp0"], ["pp0", "missing"]):
            with pytest.raises((ValueError, RuntimeError)):
                ModelRunner.receive_weights_from_m2n_groups(runner, groups)
            receive.assert_not_called()
        ModelRunner.receive_weights_from_m2n_groups(runner, ["pp0", "pp1"])
        receive.assert_called_once_with([receivers["pp0"], receivers["pp1"]])


def test_m2n_group_initialization_rejects_a_draft_runner():
    tp_worker = Mock()
    tp_worker.init_weights_update_group.return_value = (True, "Success")
    tp_worker.destroy_weights_update_group.return_value = (True, "Success")
    manager = _manager(
        tp_worker=tp_worker,
        draft_worker=SimpleNamespace(iter_runners=lambda: [("draft", Mock())]),
    )
    req = InitWeightsUpdateGroupReqInput(
        master_address="127.0.0.1",
        master_port=1234,
        rank_offset=1,
        world_size=2,
        group_name="miles-m2n-test",
        backend="nccl",
        m2n_manifest={"schema_version": 1},
    )

    output = manager.init_weights_update_group(req)

    assert output.success is False
    tp_worker.destroy_weights_update_group.assert_called_once()


def test_destroying_an_absent_update_group_is_idempotent():
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = SimpleNamespace(
        _model_update_group={},
        _m2n_receivers={},
    )

    success, message = ModelRunner.destroy_weights_update_group(
        runner, "missing-legacy-group"
    )

    assert success is True
    assert "already absent" in message


def test_failed_update_group_destroy_remains_retryable():
    from sglang.srt.model_executor.model_runner import ModelRunner

    receiver = Mock()
    receiver.destroy.side_effect = [RuntimeError("injected destroy failure"), None]
    process_group = object()
    runner = SimpleNamespace(
        _model_update_group={"miles-m2n-old": process_group},
        _m2n_receivers={"miles-m2n-old": receiver},
    )

    with patch("torch.distributed.destroy_process_group") as destroy_group:
        success, message = ModelRunner.destroy_weights_update_group(
            runner, "miles-m2n-old"
        )

        assert success is False
        assert "injected destroy failure" in message
        assert runner._m2n_receivers["miles-m2n-old"] is receiver
        assert runner._model_update_group["miles-m2n-old"] is process_group

        success, _ = ModelRunner.destroy_weights_update_group(
            runner, "miles-m2n-old"
        )

    assert success is True
    assert receiver.destroy.call_count == 2
    destroy_group.assert_called_once_with(process_group)
    assert runner._m2n_receivers == {}
    assert runner._model_update_group == {}


def test_failed_process_group_destroy_remains_retryable():
    from sglang.srt.model_executor.model_runner import ModelRunner

    receiver = Mock()
    process_group = object()
    runner = SimpleNamespace(
        _model_update_group={"miles-m2n-old": process_group},
        _m2n_receivers={"miles-m2n-old": receiver},
    )

    with patch(
        "torch.distributed.destroy_process_group",
        side_effect=[RuntimeError("injected process-group failure"), None],
    ) as destroy_group:
        success, message = ModelRunner.destroy_weights_update_group(
            runner, "miles-m2n-old"
        )

        assert success is False
        assert "injected process-group failure" in message
        assert runner._model_update_group["miles-m2n-old"] is process_group
        assert runner._m2n_receivers["miles-m2n-old"] is receiver

        success, _ = ModelRunner.destroy_weights_update_group(
            runner, "miles-m2n-old"
        )

    assert success is True
    assert receiver.destroy.call_count == 2
    assert destroy_group.call_count == 2
    assert runner._model_update_group == {}
    assert runner._m2n_receivers == {}


def test_destroying_one_m2n_stage_retires_all_native_resources_before_any_group():
    from sglang.srt.model_executor.model_runner import ModelRunner

    events = []
    receivers = {}
    groups = {"residual": object()}
    for stage in range(2):
        name = f"miles-m2n-pp{stage}"
        receiver = Mock()
        receiver.stream.synchronize.side_effect = lambda stage=stage: events.append(
            f"sync:{stage}"
        )
        receiver.destroy.side_effect = lambda stage=stage: events.append(
            f"native:{stage}"
        )
        receivers[name] = receiver
        groups[name] = name
    runner = SimpleNamespace(_m2n_receivers=receivers, _model_update_group=groups)
    with patch(
        "torch.distributed.destroy_process_group",
        side_effect=lambda pg: events.append(f"pg:{pg}"),
    ):
        success, _ = ModelRunner.destroy_weights_update_group(runner, "miles-m2n-pp0")
        assert success
        assert ModelRunner.destroy_weights_update_group(runner, "miles-m2n-pp1")[0]
    assert events == [
        "sync:0",
        "sync:1",
        "native:0",
        "native:1",
        "pg:miles-m2n-pp0",
        "pg:miles-m2n-pp1",
    ]
    assert runner._m2n_receivers == {}
    assert list(runner._model_update_group) == ["residual"]


def test_partial_pp_group_teardown_can_resume_on_remaining_group():
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = SimpleNamespace(
        _m2n_receivers={"pp0": Mock(), "pp1": Mock()},
        _model_update_group={"pp0": "pg0", "pp1": "pg1"},
    )
    with patch(
        "torch.distributed.destroy_process_group",
        side_effect=[None, RuntimeError("PP1 destroy failed"), None],
    ) as destroy:
        success, message = ModelRunner.destroy_weights_update_group(runner, "pp0")
        assert not success
        assert "PP1 destroy failed" in message
        assert list(runner._m2n_receivers) == ["pp1"]
        assert ModelRunner.destroy_weights_update_group(runner, "pp1")[0]
    assert destroy.call_args_list == [call("pg0"), call("pg1"), call("pg1")]
    assert runner._m2n_receivers == {}
    assert runner._model_update_group == {}


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


def test_model_runner_retains_fp8_graph_storage_across_failure_and_reconnect():
    import torch

    import sglang.srt.model_executor.model_runner as mr

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        torch.zeros((2, 2), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    model.scale = torch.nn.Parameter(
        torch.zeros((2, 1), dtype=torch.int32), requires_grad=False
    )
    graph_weight, graph_scale = model.weight.detach(), model.scale.detach()
    manifests = [
        {"entries": [{"tensor_role": role, "destination": {"parameter": name}}]}
        for name, role in (("weight", "weight"), ("scale", "scale"))
    ]
    runner = SimpleNamespace(
        model=model,
        device="cpu",
        _m2n_receivers={
            str(pp): SimpleNamespace(manifest=manifest)
            for pp, manifest in enumerate(manifests)
        },
        _m2n_fp8_storage=None,
    )

    def restore_loadable(*args):
        model.weight.data = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
        model.scale.data = torch.ones((1, 1), dtype=torch.float32)

    with patch.object(mr, "restore_weight", side_effect=restore_loadable):
        mr.ModelRunner.begin_weight_update(runner)
        storage = runner._m2n_fp8_storage
        # Replacing/retiring every communicator must not retire the original
        # graph buffers, even if begin is called again during recovery.
        runner._m2n_receivers = {
            "replacement": SimpleNamespace(
                manifest={
                    "entries": [entry for m in manifests for entry in m["entries"]]
                }
            )
        }
        mr.ModelRunner.begin_weight_update(runner)
    assert runner._m2n_fp8_storage is storage

    with patch.object(
        mr, "postprocess_weight", side_effect=RuntimeError("pack failed")
    ):
        with pytest.raises(RuntimeError, match="pack failed"):
            mr.ModelRunner.end_weight_update(runner, run_post_load=False)
    assert runner._m2n_fp8_storage is storage

    def finalize(*args):
        model.weight = torch.nn.Parameter(
            torch.full((2, 2), 2, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        model.scale = torch.nn.Parameter(
            torch.full((2, 1), 3, dtype=torch.int32), requires_grad=False
        )
        model.scale.format_ue8m0 = True

    with patch.object(mr, "postprocess_weight", side_effect=finalize):
        mr.ModelRunner.end_weight_update(runner, run_post_load=False)

    assert runner._m2n_fp8_storage is None
    assert model.weight.data_ptr() == graph_weight.data_ptr()
    assert model.scale.data_ptr() == graph_scale.data_ptr()
    assert torch.all(graph_weight.float() == 2)
    assert torch.all(graph_scale == 3)
    assert model.scale.format_ue8m0 is True


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


@pytest.mark.parametrize("selector", ["all", "target", "draft"])
def test_begin_weight_update_restarts_with_the_same_selector(selector):
    target, draft = Mock(), Mock()
    manager = _session_manager(target, draft)
    manager._weight_update_in_progress = False

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector=selector))
        manager._weight_update_loaded = True
        manager._weight_update_requires_post_load = True
        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector=selector)
        )

        assert output.success is True
        assert manager._weight_update_selector == selector
        assert manager._weight_update_in_progress is True
        assert manager._weight_update_loaded is False
        assert manager._weight_update_requires_post_load is False
        manager.end_weight_update(EndWeightUpdateReqInput())

    for role, runner in (("target", target), ("draft", draft)):
        if selector in ("all", role):
            # Retry must not restore an already loadable runner a second time.
            runner.begin_weight_update.assert_called_once_with()
            runner.end_weight_update.assert_called_once_with(run_post_load=True)
        else:
            runner.begin_weight_update.assert_not_called()
            runner.end_weight_update.assert_not_called()


@pytest.mark.parametrize(
    "initial,restart",
    [
        (initial, restart)
        for initial in ("all", "target", "draft")
        for restart in ("all", "target", "draft")
        if initial != restart
    ],
)
def test_begin_weight_update_rejects_selector_changes_without_mutating_session(
    initial, restart
):
    target, draft = Mock(), Mock()
    manager = _session_manager(target, draft)
    manager._weight_update_in_progress = False

    with patch("torch.distributed.barrier") as barrier:
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector=initial))
        manager._weight_update_loaded = True
        manager._weight_update_requires_post_load = True
        barrier.reset_mock()
        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector=restart)
        )

        assert output.success is False
        assert "Cannot change the runner selector" in output.message
        assert manager._weight_update_selector == initial
        assert manager._weight_update_in_progress is True
        assert manager._weight_update_loaded is True
        assert manager._weight_update_requires_post_load is True
        barrier.assert_not_called()
        # The original transaction can still finalize exactly the runners
        # prepared by its first begin, including the all -> target regression.
        manager.end_weight_update(EndWeightUpdateReqInput())

    for role, runner in (("target", target), ("draft", draft)):
        if initial in ("all", role):
            runner.begin_weight_update.assert_called_once_with()
            runner.end_weight_update.assert_called_once_with(run_post_load=True)
        else:
            runner.begin_weight_update.assert_not_called()
            runner.end_weight_update.assert_not_called()
