from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)


def _runner():
    runner = Mock()
    runner.weight_updater._m2n_receivers = {}
    runner.weight_updater.load_weights_from_distributed.return_value = (True, "Success")
    return runner


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
    # update_weights_from_* require an open begin_weight_update session.
    manager._session = _WeightUpdateSession(selector="all")
    return manager


def test_scheduler_distributed_update_receives_once_on_target_loads_into_each():
    # Default selector ("all"): only the target (main model) owns the update group,
    # so it receives the broadcast once; that single weights object is then loaded
    # into every selected runner — receive once on the target, load into each.
    weights = [("model.layers.0.weight", object())]
    target_runner = _runner()
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
    draft_runner = _runner()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=SimpleNamespace(
            weight_update_runners=lambda: [("draft", draft_runner)]
        ),
    )

    output = manager.update_weights_from_distributed(_distributed_req())

    assert output.success is True
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once_with(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        load_format=None,
    )
    # The single received weights object is loaded into every selected runner.
    target_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )
    draft_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    # selector="target": the target still receives once, but the draft worker is
    # never enumerated and no draft runner is loaded.
    weights = [("model.layers.0.weight", object())]
    target_runner = _runner()
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
    draft_worker = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=draft_worker,
    )

    output = manager.update_weights_from_distributed(
        _distributed_req(selector="target")
    )

    assert output.success is True
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once()
    target_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )
    draft_worker.weight_update_runners.assert_not_called()


def _session_manager(target_runner, draft_runner, *, m2n=False):
    target_runner.weight_updater._m2n_receivers = {"pp0": Mock()} if m2n else {}
    return _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=(
            SimpleNamespace(weight_update_runners=lambda: [("draft", draft_runner)])
            if draft_runner is not None
            else None
        ),
    )


def test_begin_weight_update_restores_target_and_draft():
    # The session begins on every runner (target + draft): the draft model is
    # restored to a loadable state identically to the target.
    target_runner = _runner()
    draft_runner = _runner()
    manager = _session_manager(target_runner, draft_runner)
    manager._session = None

    with patch("torch.distributed.barrier"):
        output = manager.begin_weight_update(BeginWeightUpdateReqInput())

    assert output.success is True
    target_runner.weight_updater.begin_weight_update.assert_called_once_with()
    draft_runner.weight_updater.begin_weight_update.assert_called_once_with()
    assert manager._session is not None
    assert manager._session.loaded_weights is False


def test_end_weight_update_runs_post_load_on_both_when_load_was_bypassed():
    # No load_weights happened this session (e.g. P2P/RDMA), so end runs
    # post_load_weights then quant finalize on BOTH target and draft.
    target_runner = _runner()
    draft_runner = _runner()
    manager = _session_manager(target_runner, draft_runner)
    manager._session = msgspec.structs.replace(manager._session, loaded_weights=False)

    with patch("torch.distributed.barrier"):
        output = manager.end_weight_update(EndWeightUpdateReqInput())

    assert output.success is True
    target_runner.weight_updater.end_weight_update.assert_called_once_with(
        run_post_load=True
    )
    draft_runner.weight_updater.end_weight_update.assert_called_once_with(
        run_post_load=True
    )
    assert manager._session is None


def test_end_weight_update_skips_post_load_on_both_when_weights_loaded():
    # A distributed/tensor load happened this session, so post_load is skipped on
    # both runners; only quant finalize runs.
    target_runner = _runner()
    draft_runner = _runner()
    manager = _session_manager(target_runner, draft_runner)
    manager._session = msgspec.structs.replace(manager._session, loaded_weights=True)

    with patch("torch.distributed.barrier"):
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.weight_updater.end_weight_update.assert_called_once_with(
        run_post_load=False
    )
    draft_runner.weight_updater.end_weight_update.assert_called_once_with(
        run_post_load=False
    )


def test_weight_updater_begin_end_wire_to_loader_hooks():
    from sglang.srt.model_executor.model_runner_components import weight_updater as wu

    model = object()
    updater = SimpleNamespace(get_model=lambda: model, device="cpu")
    device = torch.device("cpu")
    with patch.object(
        wu.DefaultModelLoader, "restore_weights_before_loading"
    ) as restore:
        wu.WeightUpdater.begin_weight_update(updater)
    restore.assert_called_once_with(model, device)

    for run_post_load in (True, False):
        with (
            patch.object(wu, "post_load_weights") as post_load,
            patch.object(wu.DefaultModelLoader, "postprocess_weights") as postprocess,
        ):
            wu.WeightUpdater.end_weight_update(updater, run_post_load=run_post_load)
        if run_post_load:
            post_load.assert_called_once_with(model)
        else:
            post_load.assert_not_called()
        postprocess.assert_called_once_with(model, device)


def test_begin_weight_update_selector_restores_only_selected_and_is_recorded():
    # begin(selector="draft") opens the session on the draft only; the target is
    # untouched, and the selector is recorded for end to reuse.
    target_runner = _runner()
    draft_runner = _runner()
    manager = _session_manager(target_runner, draft_runner)
    manager._session = None

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))

    target_runner.weight_updater.begin_weight_update.assert_not_called()
    draft_runner.weight_updater.begin_weight_update.assert_called_once_with()
    assert manager._session.selector == "draft"


def test_end_weight_update_reuses_session_selector_from_begin():
    # end has no selector of its own; it finalizes exactly the set begin opened.
    target_runner = _runner()
    draft_runner = _runner()
    manager = _session_manager(target_runner, draft_runner)
    manager._session = None

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.weight_updater.end_weight_update.assert_not_called()
    draft_runner.weight_updater.end_weight_update.assert_called_once()


def test_begin_weight_update_rejects_reentry():
    # A second begin while a session is open would leave the first session's
    # restored runners unfinalized — reject it loudly.
    manager = _session_manager(_runner(), _runner())
    manager._session = _WeightUpdateSession(selector="all")

    with patch("torch.distributed.barrier"):
        output = manager.begin_weight_update(BeginWeightUpdateReqInput())
    assert not output.success
    assert "already open" in output.message


@pytest.mark.parametrize("concurrent", [False, True])
def test_m2n_ipc_waves_and_residual_finalize_once(concurrent):
    target = _runner()
    target.weight_updater.receive_weights_from_distributed.return_value = [
        ("residual.weight", object())
    ]
    manager = _manager(
        SimpleNamespace(
            model_runner=target, weight_update_runners=lambda: [("target", target)]
        ),
        None,
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
        target.weight_updater.end_weight_update.assert_not_called()
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
    manager._session = msgspec.structs.replace(manager._session, sync_base=False)
    assert not manager.update_weights_from_distributed(decoded).success
    manager._session = msgspec.structs.replace(manager._session, sync_base=True)
    assert manager.update_weights_from_distributed(_distributed_req()).success
    target.weight_updater.load_weights_from_distributed.assert_called_once()
    with patch("torch.distributed.barrier"):
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    target.weight_updater.end_weight_update.assert_called_once_with(run_post_load=True)


@pytest.mark.parametrize("failure", ["native", "process_group"])
def test_m2n_teardown_preserves_resources_on_failure(failure):
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


@pytest.mark.parametrize("cache", ["ipc", "compensated_mhc", "model_owned"])
@pytest.mark.parametrize("concurrent", [False, True])
def test_m2n_receive_respects_upstream_weight_cache_guards(cache, concurrent):
    from sglang.srt.model_executor.model_runner_components import weight_updater as wu

    receiver = Mock()
    model = torch.nn.Module()
    model.child = torch.nn.Module()
    if cache == "compensated_mhc":
        model.child._hc_attn_tf32_parts = object()
    elif cache == "model_owned":
        model.child._derived_weight_cache_error = "model-owned derived cache"
    updater = object.__new__(wu.WeightUpdater)
    object.__setattr__(updater, "_m2n_receivers", {"pp0": receiver})
    object.__setattr__(updater, "get_model", lambda: model)
    with (
        patch.object(
            wu,
            "get_model",
            return_value=SimpleNamespace(
                weight_cache_mode="attach" if cache == "ipc" else "off"
            ),
        ),
        patch(
            "sglang.srt.weight_sync.nccl_m2n.NcclM2NReceiver.receive_many"
        ) as receive_many,
    ):
        with pytest.raises(
            RuntimeError, match="weight_cache|compensated mHC|derived cache"
        ):
            if concurrent:
                updater.receive_weights_from_m2n_groups(["pp0"])
            else:
                updater.receive_weights_from_m2n("pp0")
        receive_many.assert_not_called()
    receiver.receive.assert_not_called()


@pytest.mark.parametrize("m2n", [False, True])
@pytest.mark.parametrize(
    "selector,sync_base", [("target", True), ("all", True), ("target", False)]
)
def test_open_weight_update_session_rejects_reentry(m2n, selector, sync_base):
    target, draft = _runner(), _runner()
    manager = _session_manager(target, draft, m2n=m2n)
    manager._session = None
    with patch("torch.distributed.barrier") as barrier:
        assert manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector="target")
        ).success
        manager._session = msgspec.structs.replace(
            manager._session,
            loaded_weights=True,
            requires_post_load=True,
            pending_version="pending",
        )
        original_session = manager._session
        barrier.reset_mock()
        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector=selector, sync_base=sync_base)
        )
        assert not output.success
        assert "already open" in output.message
        assert manager._session is original_session
        barrier.assert_not_called()
    assert manager._session.selector == "target"
    assert manager._session.sync_base is True
    assert manager._session.loaded_weights is True
    assert manager._session.requires_post_load is True
    assert manager._session.pending_version == "pending"
    target.weight_updater.begin_weight_update.assert_called_once()
    draft.weight_updater.begin_weight_update.assert_not_called()


def test_failed_m2n_update_does_not_reopen_session():
    target = _runner()
    manager = _session_manager(target, None, m2n=True)
    manager._session = None
    with patch("torch.distributed.barrier"):
        assert manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector="target")
        ).success
    target.weight_updater.receive_weights_from_m2n.side_effect = RuntimeError(
        "transfer failed"
    )
    req = _distributed_req(selector="target")
    req.load_format = "nccl_m2n"
    output = manager.update_weights_from_distributed(req)
    assert not output.success
    assert "restart the affected rollout engines" in output.message
    assert manager._session is not None
    target.weight_updater.end_weight_update.assert_not_called()
    output = manager.begin_weight_update(BeginWeightUpdateReqInput(selector="target"))
    assert not output.success
    assert "already open" in output.message


def test_m2n_with_draft_rejects_all_before_preparing_weights():
    target, draft = _runner(), _runner()
    manager = _session_manager(target, draft, m2n=True)
    manager._session = None
    output = manager.begin_weight_update(BeginWeightUpdateReqInput())
    assert not output.success
    assert manager._session is None
    target.weight_updater.begin_weight_update.assert_not_called()
    draft.weight_updater.begin_weight_update.assert_not_called()


@pytest.mark.parametrize("concurrent", [False, True])
def test_m2n_target_session_leaves_draft_frozen(concurrent):
    target, draft = _runner(), _runner()
    manager = _session_manager(target, draft, m2n=True)
    manager._session = None
    req = _distributed_req(selector="target")
    req.load_format = "nccl_m2n"
    req.m2n_group_names = [req.group_name] if concurrent else None
    with patch("torch.distributed.barrier"):
        assert manager.begin_weight_update(
            BeginWeightUpdateReqInput(selector="target")
        ).success
        assert manager.update_weights_from_distributed(req).success
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    target.weight_updater.begin_weight_update.assert_called_once_with()
    target.weight_updater.end_weight_update.assert_called_once_with(run_post_load=True)
    assert draft.weight_updater.mock_calls == []


@pytest.mark.parametrize(
    "session_selector,selector",
    [("all", "target"), ("target", "all"), ("target", "draft")],
)
def test_m2n_rejects_incompatible_draft_selectors_before_receive(
    session_selector, selector
):
    target, draft = _runner(), _runner()
    manager = _session_manager(target, draft, m2n=True)
    manager._session = _WeightUpdateSession(selector=session_selector)
    req = _distributed_req(selector=selector)
    req.load_format = "nccl_m2n"
    output = manager.update_weights_from_distributed(req)
    assert not output.success
    target.weight_updater.receive_weights_from_m2n.assert_not_called()
    target.weight_updater.receive_weights_from_m2n_groups.assert_not_called()
    assert not manager._session.requires_post_load
    assert draft.weight_updater.mock_calls == []


@pytest.mark.parametrize("abort", [False, True])
def test_m2n_version_is_published_only_when_session_commits(abort):
    target = _runner()
    manager = _session_manager(target, None, m2n=True)
    req = _distributed_req(selector="target")
    req.load_format = "nccl_m2n"
    req.weight_version = "m2n-v2"
    assert manager.update_weights_from_distributed(req).success
    assert manager._session.pending_version == "m2n-v2"
    manager.scheduler.record_weight_version_change.assert_not_called()
    with patch("torch.distributed.barrier"):
        assert manager.end_weight_update(EndWeightUpdateReqInput(abort=abort)).success
    if abort:
        manager.scheduler.record_weight_version_change.assert_not_called()
    else:
        manager.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="m2n-v2"
        )
    assert manager._session is None


def test_m2n_post_load_requirement_does_not_leak_into_next_session():
    target = _runner()
    target.weight_updater.receive_weights_from_distributed.return_value = [
        ("residual.weight", object())
    ]
    manager = _session_manager(target, None, m2n=True)
    req = _distributed_req(selector="target")
    req.load_format = "nccl_m2n"
    assert manager.update_weights_from_distributed(req).success
    with patch("torch.distributed.barrier"):
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
        assert manager.begin_weight_update(BeginWeightUpdateReqInput()).success
        assert manager.update_weights_from_distributed(_distributed_req()).success
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    assert target.weight_updater.end_weight_update.call_args_list == [
        call(run_post_load=True),
        call(run_post_load=False),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
