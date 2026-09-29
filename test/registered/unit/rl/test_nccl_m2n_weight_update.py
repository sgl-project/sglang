from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    DestroyWeightsUpdateGroupReqInput,
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
        scheduler=Mock(spec=["record_weight_version_change"]),
    )
    # update_weights_from_* assert an open begin_weight_update session.
    manager._weight_update_in_progress = True
    return manager


def _session_manager(target_runner, draft_runner):
    target_runner.weight_updater._m2n_receivers = {}
    return _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner, iter_runners=lambda: [("", target_runner)]
        ),
        draft_worker=(
            SimpleNamespace(iter_runners=lambda: [("draft", draft_runner)])
            if draft_runner is not None
            else None
        ),
    )


def _init_group(manager, name="pp0", *, m2n=True, success=True):
    def initialize(req):
        if success and m2n:
            manager.tp_worker.model_runner.weight_updater._m2n_receivers[name] = Mock()
        return success, "initialized" if success else "failed"

    manager.tp_worker.init_weights_update_group = Mock(side_effect=initialize)
    return manager.init_weights_update_group(
        InitWeightsUpdateGroupReqInput(
            master_address="localhost",
            master_port=12345,
            rank_offset=1,
            world_size=2,
            group_name=name,
            m2n_manifest={"schema_version": 1} if m2n else None,
        )
    )


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


@pytest.mark.parametrize("cache", ["ipc", "derived"])
def test_m2n_receive_respects_upstream_weight_cache_guards(cache):
    from sglang.srt.model_executor.model_runner_components import weight_updater as wu

    receiver = Mock()
    updater = object.__new__(wu.WeightUpdater)
    object.__setattr__(updater, "_m2n_receivers", {"pp0": receiver})
    with (
        patch.object(
            wu,
            "get_model",
            return_value=SimpleNamespace(
                weight_cache_mode="attach" if cache == "ipc" else "off"
            ),
        ),
        patch.object(
            wu, "_unsupported_derived_weight_cache_error", return_value="derived cache"
        ),
    ):
        with pytest.raises(RuntimeError, match="weight_cache|derived cache"):
            updater.receive_weights_from_m2n("pp0")
    receiver.receive.assert_not_called()


@pytest.mark.parametrize(
    "initial,restart,sync_base,with_draft",
    [
        ("target", "target", True, True),
        ("all", "all", True, False),
        ("all", "target", True, False),
        ("target", "all", True, True),
        ("target", "target", False, True),
    ],
)
def test_m2n_session_restart_keeps_original_selector_and_preparation(
    initial, restart, sync_base, with_draft
):
    target, draft = Mock(), Mock()
    manager = _session_manager(target, draft if with_draft else None)
    manager._weight_update_in_progress = False
    assert _init_group(manager).success
    with patch("torch.distributed.barrier") as barrier:
        assert manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector=initial)
        ).success
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
        selected = initial in ("all", role) and (role == "target" or with_draft)
        assert runner.begin_weight_update.call_count == int(selected)
        assert runner.end_weight_update.call_count == int(selected)


def test_m2n_group_lifecycle_preserves_ordinary_session_reentry_guard():
    manager = _session_manager(Mock(), None)
    assert not _init_group(manager, success=False).success
    assert _init_group(manager, "residual", m2n=False).success
    assert not manager._m2n_update_groups
    with pytest.raises(AssertionError, match="already open"):
        manager.begin_weight_update(BeginWeightUpdateReqInput())

    assert _init_group(manager, "pp0").success
    assert _init_group(manager, "pp1").success
    receivers = manager.tp_worker.model_runner.weight_updater._m2n_receivers

    def destroy(req):
        if req.group_name == "pp0" and "pp0" in receivers:
            receivers.pop("pp0")
            return False, "failed while destroying pp1"
        if req.group_name == "pp1":
            receivers.clear()
        return True, "done"

    manager.tp_worker.destroy_weights_update_group = Mock(side_effect=destroy)
    assert manager.destroy_weights_update_group(
        DestroyWeightsUpdateGroupReqInput(group_name="residual")
    ).success
    assert manager._m2n_update_groups == {"pp0", "pp1"}
    req = DestroyWeightsUpdateGroupReqInput(group_name="pp0")
    assert not manager.destroy_weights_update_group(req).success
    assert manager._m2n_update_groups == {"pp1"}
    assert manager.destroy_weights_update_group(req).success
    assert manager._m2n_update_groups == {"pp1"}
    assert manager.destroy_weights_update_group(
        DestroyWeightsUpdateGroupReqInput(group_name="pp1")
    ).success
    assert not manager._m2n_update_groups
    with pytest.raises(AssertionError, match="already open"):
        manager.begin_weight_update(BeginWeightUpdateReqInput())


def test_m2n_with_draft_rejects_all_before_preparing_weights():
    target, draft = Mock(), Mock()
    manager = _session_manager(target, draft)
    manager._weight_update_in_progress = False
    assert _init_group(manager).success
    output = manager.begin_weight_update(BeginWeightUpdateReqInput())
    assert not output.success
    assert not manager._weight_update_in_progress
    target.begin_weight_update.assert_not_called()
    draft.begin_weight_update.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
