import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.flydslep as flydslep
from sglang.srt.layers.moe.token_dispatcher.flydslep import (
    _FlyDSLCommStreamPool,
    _get_tbo_comm_stream,
    _recv_count_values,
    _resolve_eager_recv_cap,
    _resolve_tbo_child_cluster_rows,
    _resolve_tbo_geometry,
    _should_sync_recv_values,
    _validate_all_rank_recv_cap,
    _validate_stream_priority,
    prepare_tbo_eager_recv_cap_metadata,
)
from sglang.srt.utils.bounded_telemetry import (
    BoundedTelemetryLogger,
    format_telemetry_fields,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _resolve(
    *,
    tbo_enabled=True,
    shared_block_num=0,
    dispatch_block_num=0,
    combine_block_num=0,
):
    return _resolve_tbo_geometry(
        tbo_enabled=tbo_enabled,
        shared_block_num=shared_block_num,
        dispatch_block_num=dispatch_block_num,
        combine_block_num=combine_block_num,
    )


def test_shared_block_num_controls_both_phases_for_backward_compatibility():
    assert _resolve(shared_block_num=12) == (12, 4, 12, 4)


def test_independent_block_nums_take_precedence_per_phase():
    assert _resolve(
        shared_block_num=12, dispatch_block_num=8, combine_block_num=16
    ) == (8, 4, 16, 4)


@pytest.mark.parametrize(
    ("shared_block_num", "dispatch_block_num", "combine_block_num", "expected"),
    [
        (12, 8, 0, (8, 4, 12, 4)),
        (12, 0, 16, (12, 4, 16, 4)),
        (0, 8, 0, (8, 4, None, None)),
        (0, 0, 16, (None, None, 16, 4)),
    ],
)
def test_partial_override_falls_back_to_shared_or_tuning(
    shared_block_num, dispatch_block_num, combine_block_num, expected
):
    assert (
        _resolve(
            shared_block_num=shared_block_num,
            dispatch_block_num=dispatch_block_num,
            combine_block_num=combine_block_num,
        )
        == expected
    )


def test_zero_and_unset_values_leave_both_phases_to_tuning():
    assert _resolve() == (None, None, None, None)


def test_non_tbo_ignores_all_tbo_knobs():
    assert _resolve(
        tbo_enabled=False,
        shared_block_num=-1,
        dispatch_block_num=-2,
        combine_block_num=-3,
    ) == (None, None, None, None)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("SGLANG_FLYDSL_TBO_BLOCK_NUM", {"shared_block_num": -1}),
        (
            "SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM",
            {"dispatch_block_num": -1},
        ),
        (
            "SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM",
            {"combine_block_num": -1},
        ),
    ],
)
def test_negative_tbo_knobs_are_rejected(name, kwargs):
    with pytest.raises(ValueError, match=rf"^{name} must be non-negative; got -1$"):
        _resolve(**kwargs)


def test_comm_stream_priority_propagates_and_partitions_cache(monkeypatch):
    created_priorities = []

    def make_stream(*, priority):
        created_priorities.append(priority)
        return Mock(priority=priority)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(
        torch.cuda, "get_stream_priority_range", lambda: (0, -2), raising=False
    )
    monkeypatch.setattr(_FlyDSLCommStreamPool, "_streams", {})
    group = object()

    default_stream = _FlyDSLCommStreamPool.get(group)
    assert _FlyDSLCommStreamPool.get(group, priority=0) is default_stream
    high_priority_stream = _FlyDSLCommStreamPool.get(group, priority=-1)
    assert high_priority_stream is _FlyDSLCommStreamPool.get(group, priority=-1)
    assert high_priority_stream is not default_stream
    assert created_priorities == [0, -1]


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, "stream"),
        ("true", "stream"),
        ("false", None),
    ],
)
def test_comm_stream_control_default_on_off(monkeypatch, caplog, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("SGLANG_FLYDSL_TBO_USE_COMM_STREAM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_FLYDSL_TBO_USE_COMM_STREAM", env_value)
    priority_reader = Mock(return_value=-1)
    pool_get = Mock(return_value="stream")
    monkeypatch.setattr(flydslep, "get_int_env_var", priority_reader)
    monkeypatch.setattr(_FlyDSLCommStreamPool, "get", pool_get)
    group = object()

    with caplog.at_level("INFO", logger=flydslep.__name__):
        assert (
            _get_tbo_comm_stream(group, tbo_enabled=True, async_finish=True) == expected
        )

    expected_log = f"[FlyDSL TBO] dedicated comm stream enabled={expected is not None}"
    assert sum(record.message == expected_log for record in caplog.records) == 1
    if expected is None:
        priority_reader.assert_not_called()
        pool_get.assert_not_called()
    else:
        priority_reader.assert_called_once_with(
            "SGLANG_FLYDSL_TBO_COMM_STREAM_PRIORITY", 0
        )
        pool_get.assert_called_once_with(group, priority=-1)


def test_comm_stream_control_is_tbo_async_only(monkeypatch):
    bool_env_reader = Mock(return_value=True)
    priority_reader = Mock(return_value=0)
    pool_get = Mock(return_value="stream")
    monkeypatch.setattr(flydslep, "get_bool_env_var", bool_env_reader)
    monkeypatch.setattr(flydslep, "get_int_env_var", priority_reader)
    monkeypatch.setattr(_FlyDSLCommStreamPool, "get", pool_get)
    group = object()

    assert _get_tbo_comm_stream(group, tbo_enabled=False, async_finish=True) is None
    assert _get_tbo_comm_stream(group, tbo_enabled=True, async_finish=False) is None
    bool_env_reader.assert_not_called()
    priority_reader.assert_not_called()
    pool_get.assert_not_called()


def test_comm_stream_priority_env_not_read_when_stream_disabled(monkeypatch):
    priority_reader = Mock()
    pool_get = Mock()
    monkeypatch.setattr(flydslep, "get_bool_env_var", Mock(return_value=False))
    monkeypatch.setattr(flydslep, "get_int_env_var", priority_reader)
    monkeypatch.setattr(_FlyDSLCommStreamPool, "get", pool_get)

    assert _get_tbo_comm_stream(object(), tbo_enabled=True, async_finish=True) is None
    priority_reader.assert_not_called()
    pool_get.assert_not_called()


def test_comm_stream_priority_rejects_unsupported_value():
    _validate_stream_priority(-1, (0, -2))
    with pytest.raises(ValueError, match="outside torch's supported"):
        _validate_stream_priority(1, (0, -2))


@pytest.mark.parametrize(
    ("telemetry_enabled", "sync_values_enabled", "pending", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_sync_recv_values_gate(
    telemetry_enabled, sync_values_enabled, pending, expected
):
    assert (
        _should_sync_recv_values(
            telemetry_enabled=telemetry_enabled,
            sync_values_enabled=sync_values_enabled,
            pending=pending,
        )
        is expected
    )


def test_telemetry_keys_distinguish_dispatchers_with_same_child_id(monkeypatch):
    for module_name in ("aiter", "flydsl", "mori"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setattr(flydslep, "_use_aiter", False)
    monkeypatch.setattr(flydslep, "is_tbo_enabled", lambda: False)

    first = flydslep.FlyDSLEPDispatcher(object(), router_topk=1, instance_id=0)
    second = flydslep.FlyDSLEPDispatcher(object(), router_topk=1, instance_id=0)

    assert first._telemetry_dispatcher_id == id(first)
    assert second._telemetry_dispatcher_id == id(second)
    assert first._telemetry_dispatcher_id != second._telemetry_dispatcher_id
    assert flydslep._flydsl_telemetry._max_events == 256
    assert flydslep._flydsl_telemetry._rank_zero_only is True
    assert flydslep._flydsl_sync_values_telemetry._max_events == 128
    assert flydslep._flydsl_sync_values_telemetry._rank_zero_only is False

    telemetry = BoundedTelemetryLogger(
        Mock(), "[TEST_FLYDSL_TELEMETRY]", enabled=True, max_events=256
    )
    monkeypatch.setattr("sglang.srt.utils.bounded_telemetry.is_rank_zero", lambda: True)
    first_key = ("dispatch", first._telemetry_dispatcher_id)
    second_key = ("dispatch", second._telemetry_dispatcher_id)

    assert telemetry.log(first_key, "dispatch", child_id=first.instance_id)
    assert telemetry.log(second_key, "dispatch", child_id=second.instance_id)
    assert not telemetry.log(first_key, "dispatch", child_id=first.instance_id)


@pytest.mark.parametrize(("existing", "expected"), [(None, "1"), ("0", "0")])
def test_flydsl_dispatcher_defaults_no_fake_expert_with_explicit_rollback(
    monkeypatch, existing, expected
):
    for module_name in ("aiter", "flydsl", "mori"):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    monkeypatch.setattr(flydslep, "_use_aiter", False)
    monkeypatch.setattr(flydslep, "is_tbo_enabled", lambda: False)
    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    if existing is None:
        monkeypatch.delenv("AITER_FLYDSL_EP_NO_FAKE_EXPERT", raising=False)
    else:
        monkeypatch.setenv("AITER_FLYDSL_EP_NO_FAKE_EXPERT", existing)

    flydslep.FlyDSLEPDispatcher(object(), router_topk=6)

    assert os.environ["AITER_FLYDSL_EP_NO_FAKE_EXPERT"] == expected
    assert "AITER_CONFIG_FMOE" not in os.environ


def test_recv_count_values_and_diagnostic_formatting():
    out_idx = torch.tensor([[4, 5, -1], [4, 7, 1], [6, 6, 6]])
    counts, actual_total_recv = _recv_count_values(
        out_idx,
        torch.tensor([2]),
        local_expert_start=4,
        num_local_experts=3,
    )

    assert counts == [2, 1, 0]
    assert actual_total_recv == sum(counts) == 3
    assert format_telemetry_fields(
        event="dispatch_recv_values",
        global_rank=3,
        moe_ep_rank=1,
        dispatcher_id=1234,
        child_id=1,
        physical_recv_cap=8192,
        effective_recv_cap=4096,
        recv_counts_per_expert=counts,
        actual_total_recv=actual_total_recv,
        sync_values_diagnostic_perturbs_timing=True,
        sync_values_not_for_performance_benchmark_traces=True,
    ) == (
        "event=dispatch_recv_values global_rank=3 moe_ep_rank=1 "
        "dispatcher_id=1234 child_id=1 physical_recv_cap=8192 "
        "effective_recv_cap=4096 recv_counts_per_expert=[2,1,0] "
        "actual_total_recv=3 "
        "sync_values_diagnostic_perturbs_timing=true "
        "sync_values_not_for_performance_benchmark_traces=true"
    )


def test_child_cluster_rows_include_asymmetric_split_padding():
    # Rank-local parent rows are synchronized; child rows include independent
    # split alignment/padding on each rank.
    assert _resolve_tbo_child_cluster_rows(
        parent_global_num_tokens=[10, 14, 7],
        child_padded_rows_by_rank=[
            [4, 8],
            [0, 16],
            [7, 0],
        ],
    ) == (11, 24)


@pytest.mark.parametrize(
    ("cluster_rows", "physical_cap", "expected"),
    [
        (0, 128, 32),
        (0, 16, 16),
        (1, 128, 32),
        (32, 128, 32),
        (33, 128, 64),
        (70, 64, 64),
        (70, 128, 128),
        (129, 128, 128),
        # Dynamic variants remain power-of-two only. Conservatively keep the
        # existing physical path if a non-power-of-two clamp would be needed.
        (97, 96, None),
    ],
)
def test_eager_recv_cap_pow2_minimum_and_physical_clamp(
    cluster_rows, physical_cap, expected
):
    assert _resolve_eager_recv_cap(cluster_rows, physical_cap) == expected


@pytest.mark.parametrize(
    ("parent_rows", "child_rows"),
    [
        (None, [[1, 2]]),
        ([3], None),
        ([3, 4], [[1, 2]]),
        ([5], [[2, 2]]),  # child metadata would lose a parent row
        ([0], [[]]),
    ],
)
def test_child_cluster_rows_missing_or_inexact_metadata_disables(
    parent_rows, child_rows
):
    assert _resolve_tbo_child_cluster_rows(parent_rows, child_rows) is None


def test_all_rank_recv_cap_validator_accepts_agreement_and_rejects_failures():
    _validate_all_rank_recv_cap([64, 64, 64], [7, 20, 31])
    with pytest.raises(RuntimeError, match="mismatch across ranks"):
        _validate_all_rank_recv_cap([64, 32, 64], [7, 20, 5])
    with pytest.raises(RuntimeError, match="underbound"):
        _validate_all_rank_recv_cap([32, 32, 32], [7, 20, 31])


def test_prepare_tbo_eager_metadata_default_on_gathers_and_sets_both_children(
    monkeypatch,
):
    monkeypatch.delenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", raising=False)
    monkeypatch.delenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_VALIDATE", raising=False)
    children = [
        SimpleNamespace(tbo_padded_len=1),
        SimpleNamespace(tbo_padded_len=20),
    ]
    gathered_rows = ((1, 20), (9, 25), (10, 25))
    calls = []

    def fake_all_gather(outputs, local, *, group):
        calls.append((tuple(local.tolist()), group))
        for output, rows in zip(outputs, gathered_rows, strict=True):
            output.copy_(torch.tensor(rows))

    monkeypatch.setattr(flydslep.dist, "all_gather", fake_all_gather)
    group = SimpleNamespace(world_size=3, cpu_group="cpu-group")

    assert prepare_tbo_eager_recv_cap_metadata(
        parent_global_num_tokens=[20, 30, 30],
        children=children,
        group=group,
    )
    assert [child.flydsl_tbo_cluster_dispatch_rows for child in children] == [
        20,
        70,
    ]
    assert calls == [((1, 20), "cpu-group")]


def test_prepare_tbo_eager_diagnostic_detects_rank_cap_mismatch(monkeypatch):
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "true")
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_VALIDATE", "true")
    children = [
        SimpleNamespace(tbo_padded_len=4),
        SimpleNamespace(tbo_padded_len=4),
    ]
    calls = 0

    def fake_all_gather(outputs, local, *, group):
        nonlocal calls
        calls += 1
        values = ((4, 4), (4, 4)) if calls == 1 else ((32, 32, 4, 4), (64, 32, 4, 4))
        for output, rank_values in zip(outputs, values, strict=True):
            output.copy_(torch.tensor(rank_values))

    monkeypatch.setattr(flydslep.dist, "all_gather", fake_all_gather)

    with pytest.raises(RuntimeError, match="mismatch across ranks"):
        prepare_tbo_eager_recv_cap_metadata(
            parent_global_num_tokens=[8, 8],
            children=children,
            group=SimpleNamespace(world_size=2, cpu_group=object()),
        )
    assert calls == 2


def test_prepare_tbo_eager_metadata_missing_parent_disables_without_collective(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "true")
    all_gather = Mock()
    monkeypatch.setattr(flydslep.dist, "all_gather", all_gather)
    children = [SimpleNamespace(tbo_padded_len=3), SimpleNamespace(tbo_padded_len=5)]

    assert not prepare_tbo_eager_recv_cap_metadata(
        parent_global_num_tokens=None,
        children=children,
        group=SimpleNamespace(world_size=2, cpu_group=object()),
    )
    all_gather.assert_not_called()
    assert not hasattr(children[0], "flydsl_tbo_cluster_dispatch_rows")


def test_prepare_tbo_eager_metadata_explicit_off_skips_collective(monkeypatch):
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "0")
    all_gather = Mock()
    monkeypatch.setattr(flydslep.dist, "all_gather", all_gather)

    assert not prepare_tbo_eager_recv_cap_metadata(
        parent_global_num_tokens=[8, 8],
        children=[
            SimpleNamespace(tbo_padded_len=4),
            SimpleNamespace(tbo_padded_len=4),
        ],
        group=SimpleNamespace(world_size=2, cpu_group=object()),
    )
    all_gather.assert_not_called()


def test_eager_cap_metadata_does_not_change_legacy_decode_graph_path(
    monkeypatch,
):
    from sglang.srt.layers import dp_attention
    from sglang.srt.model_executor import runner

    dispatcher = object.__new__(flydslep.FlyDSLEPDispatcher)
    monkeypatch.delenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", raising=False)
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP", "true")
    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(dp_attention, "get_dp_global_num_tokens", lambda: [3, 9])

    # No eager child metadata: retain the pre-existing decode/CUDA-graph
    # global-capacity calculation (18 -> minimum bucket 32).
    assert dispatcher._resolve_dynamic_recv_cap(128, eager_cluster_rows=None) == 32
    # Eager TBO metadata is default-on and intentionally a separate path.
    assert dispatcher._resolve_dynamic_recv_cap(128, eager_cluster_rows=70) == 128
    # Explicit rollback retains the pre-existing decode/CUDA-graph fallback.
    monkeypatch.setenv("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "0")
    assert dispatcher._resolve_dynamic_recv_cap(128, eager_cluster_rows=70) == 32
