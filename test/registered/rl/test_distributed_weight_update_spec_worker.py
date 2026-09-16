from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
    # metrics_collector defaults to None, so _observe_weight_load is a no-op; the
    # reqs below set flush_cache=False so flush_cache is never called either.
    manager = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
        scheduler=Mock(spec=["record_weight_version_change"]),
    )
    # update_weights_from_* assert an open begin_weight_update session.
    manager._weight_update_in_progress = True
    return manager


def test_scheduler_distributed_update_receives_once_on_target_loads_into_each():
    # Default selector ("all"): only the target (main model) owns the update group,
    # so it receives the broadcast once; that single weights object is then loaded
    # into every selected runner — receive once on the target, load into each.
    weights = [("model.layers.0.weight", object())]
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
    weights = [("model.layers.0.weight", object())]
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


def test_model_runner_begin_end_wire_to_loader_hooks():
    # ModelRunner.begin/end delegate to the loader: begin restores; end runs
    # post_load only when requested, always finalizes quant layout.
    import sglang.srt.model_executor.model_runner as mr

    runner = SimpleNamespace(
        model=object(), device="cpu", weight_updater=SimpleNamespace(_m2n_receivers={})
    )

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


@pytest.mark.parametrize("concurrent", [False, True])
def test_m2n_ipc_waves_and_residual_finalize_once(concurrent):
    import msgspec

    target = Mock()
    target.weight_updater.receive_weights_from_distributed.return_value = [
        ("residual.weight", object())
    ]
    manager = _manager(
        SimpleNamespace(model_runner=target, iter_runners=lambda: [("", target)]), None
    )
    for groups in (["pp0", "pp1"], ["pp2"]):
        req = _distributed_req(selector="target")
        req.load_format, req.group_name = "nccl_m2n", groups[0]
        req.m2n_group_names = groups if concurrent else None
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=type(req))
        assert decoded.m2n_group_names == req.m2n_group_names
        # Old positional IPC requests still decode with the appended default.
        legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(req))[:-1]
        assert (
            msgspec.msgpack.decode(
                msgspec.msgpack.encode(legacy), type=type(req)
            ).m2n_group_names
            is None
        )
        assert manager.update_weights_from_distributed(decoded).success
        target.end_weight_update.assert_not_called()
    if concurrent:
        assert target.weight_updater.receive_weights_from_m2n_groups.call_args_list == [
            call(["pp0", "pp1"]),
            call(["pp2"]),
        ]
    else:
        assert target.weight_updater.receive_weights_from_m2n.call_args_list == [
            call("pp0"),
            call("pp2"),
        ]
    manager._weight_update_sync_base = False
    assert not manager.update_weights_from_distributed(decoded).success
    manager._weight_update_sync_base = True
    assert manager.update_weights_from_distributed(_distributed_req()).success
    target.weight_updater.load_weights.assert_called_once()
    with patch("torch.distributed.barrier"):
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    target.end_weight_update.assert_called_once_with(run_post_load=True)


@pytest.mark.parametrize("failure", ["native", "process_group"])
def test_m2n_teardown_is_native_first_and_retryable(failure):
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        WeightUpdater,
    )

    events = []
    receivers = {name: Mock() for name in ("pp0", "pp1")}
    groups = {"pp0": "pg0", "pp1": "pg1", "residual": "residual"}
    runner = SimpleNamespace(
        _m2n_receivers=receivers.copy(), _model_update_group=groups.copy()
    )
    for name, receiver in receivers.items():
        receiver.stream.synchronize.side_effect = lambda name=name: events.append(
            ("sync", name)
        )
        receiver.destroy.side_effect = lambda name=name: events.append(("native", name))
    if failure == "native":
        receivers["pp1"].destroy.side_effect = RuntimeError("native failed")

    def destroy(pg):
        events.append(("pg", pg))
        if failure == "process_group" and pg == "pg1" and events.count(("pg", pg)) == 1:
            raise RuntimeError("process group failed")

    with patch("torch.distributed.destroy_process_group", side_effect=destroy):
        assert not WeightUpdater.destroy_weights_update_group(runner, "pp0")[0]
        assert events[:3] == [("sync", "pp0"), ("sync", "pp1"), ("native", "pp0")]
        if failure == "native":
            assert not any(kind == "pg" for kind, _ in events)
            assert runner._m2n_receivers == receivers
            receivers["pp1"].destroy.side_effect = lambda: events.append(
                ("native", "pp1")
            )
        else:
            assert events[3:5] == [("native", "pp1"), ("pg", "pg0")]
            assert list(runner._m2n_receivers) == ["pp1"]
        assert WeightUpdater.destroy_weights_update_group(runner, "pp1")[0]
        assert WeightUpdater.destroy_weights_update_group(runner, "pp0")[0]
    assert runner._m2n_receivers == {}
    assert runner._model_update_group == {"residual": "residual"}


@pytest.mark.parametrize(
    "initial,restart,sync_base",
    [
        ("target", "target", True),
        ("all", "target", True),
        ("draft", "all", True),
        ("target", "target", False),
    ],
)
def test_session_restart_keeps_original_selector_and_preparation(
    initial, restart, sync_base
):
    target, draft = Mock(), Mock()
    manager = _session_manager(target, draft)
    manager._weight_update_in_progress = False
    with patch("torch.distributed.barrier") as barrier:
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector=initial))
        manager._weight_update_loaded = manager._weight_update_requires_post_load = True
        barrier.reset_mock()
        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector=restart, sync_base=sync_base)
        )
        same_session = initial == restart and sync_base
        assert output.success is same_session
        assert manager._weight_update_selector == initial
        assert manager._weight_update_in_progress is True
        assert manager._weight_update_loaded is (not same_session)
        assert manager._weight_update_requires_post_load is (not same_session)
        if not same_session:
            barrier.assert_not_called()
        manager.end_weight_update(EndWeightUpdateReqInput())
    for role, runner in (("target", target), ("draft", draft)):
        assert runner.begin_weight_update.call_count == int(initial in ("all", role))
        assert runner.end_weight_update.call_count == int(initial in ("all", role))


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
        weight_updater=SimpleNamespace(
            _m2n_receivers={
                str(pp): SimpleNamespace(manifest=manifest)
                for pp, manifest in enumerate(manifests)
            }
        ),
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
        runner.weight_updater._m2n_receivers = {
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


@pytest.mark.parametrize("cache", ["ipc", "derived"])
def test_m2n_receive_respects_upstream_weight_cache_guards(cache):
    from sglang.srt.model_executor.model_runner_components import weight_updater as wu

    receiver = Mock()
    updater = object.__new__(wu.WeightUpdater)
    object.__setattr__(updater, "_m2n_receivers", {"pp0": receiver})
    object.__setattr__(
        updater,
        "get_model_runner",
        lambda: SimpleNamespace(
            server_args=SimpleNamespace(
                weight_cache_mode="attach" if cache == "ipc" else "off"
            )
        ),
    )
    with patch.object(
        wu, "_unsupported_derived_weight_cache_error", return_value="derived cache"
    ):
        with pytest.raises(RuntimeError, match="weight_cache|derived cache"):
            updater.receive_weights_from_m2n("pp0")
    receiver.receive.assert_not_called()
