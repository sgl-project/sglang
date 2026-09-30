import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPDispatcher,
    MoriEPLLDispatchOutput,
    MoriEPNormalDispatchOutput,
    _MoriEPDispatcherImplBase,
    _MoriEPDispatcherImplLowLatency,
    _MoriEPDispatcherImplNormal,
    _MoriEPv2DispatcherImplNormal,
    _Stage,
    mori_recv_bound,
)
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_KINDS = ("epv1", "epv1_ll", "epv2")
_KERNEL_TYPES = SimpleNamespace(
    IntraNode="IntraNode",
    AsyncLL="AsyncLL",
    InterNodeV1="InterNodeV1",
    InterNodeV1LL="InterNodeV1LL",
)


@pytest.fixture
def make_dispatcher(monkeypatch):
    fake_mori = ModuleType("mori")
    fake_mori.ops = SimpleNamespace(EpDispatchCombineKernelType=_KERNEL_TYPES)
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    monkeypatch.setattr(adapter, "_should_record_expert_distribution", lambda: False)

    def make(
        kind,
        enabled=True,
        manual_cap=0,
        recv_rows=65536,
        local_rows=7,
        sender_rows=(7,) * 8,
        ep_size=8,
        rank=0,
        attn_tp_size=1,
        tbo=False,
        pow2=False,
    ):
        monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "1" if enabled else "0")
        monkeypatch.setenv("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", str(manual_cap))
        monkeypatch.setattr(adapter, "is_tbo_enabled", lambda: tbo)
        monkeypatch.setattr(
            adapter,
            "get_parallel",
            lambda: _recv_bound_parallel(
                ep_size=ep_size, rank=rank, attn_tp_size=attn_tp_size
            ),
        )
        cls = {
            "epv1": _MoriEPDispatcherImplNormal,
            "epv1_ll": _MoriEPDispatcherImplLowLatency,
            "epv2": _MoriEPv2DispatcherImplNormal,
        }[kind]
        dispatcher = cls.__new__(cls)
        _MoriEPDispatcherImplBase.__init__(
            dispatcher,
            group=None,
            router_topk=2,
            permute_fusion=False,
            num_experts=16,
            num_local_experts=2,
            hidden_size=4,
            params_dtype=torch.bfloat16,
            deepep_mode=DeepEPMode.NORMAL,
        )
        dispatcher._mori_op = SimpleNamespace(
            config=SimpleNamespace(
                kernel_type="AsyncLL" if kind == "epv1_ll" else "IntraNode"
            ),
            max_num_tokens_to_recv=Mock(return_value=recv_rows),
            cfg=SimpleNamespace(effective_max_recv=recv_rows, is_internode=False),
            backend_name="flydsl",
            dispatch_recv=Mock(),
        )
        dispatcher._num_tokens = local_rows
        dispatcher._dispatch_sender_rows = adapter.normalize_sender_rows(sender_rows)
        dispatcher._recv_cap_pow2_buckets = kind == "epv2" and pow2
        dispatcher._direct_output = False
        return dispatcher

    return make


def _recv_bound_parallel(ep_size=8, rank=0, attn_tp_size=1):
    return SimpleNamespace(
        moe_ep_size=ep_size,
        moe_ep_rank=rank,
        tp_size=ep_size,
        attn_dp_size=ep_size // attn_tp_size,
        attn_dp_rank=rank // attn_tp_size,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=rank % attn_tp_size,
        attn_cp_size=1,
        moe_tp_size=1,
        moe_dp_size=1,
        launch_world_rank=rank,
    )


@pytest.fixture
def mock_get_parallel(monkeypatch):
    getter = Mock(return_value=_recv_bound_parallel())
    monkeypatch.setattr(adapter, "get_parallel", getter)
    return getter


@pytest.mark.parametrize(
    "sender_rows,rank,expected,expected_pow2",
    [
        ([1] * 8, 0, 32, 32),
        ([7] * 8, 0, 64, 64),
        ([56] * 8, 3, 448, 512),
        ([448] * 8, 7, 3584, 4096),
        ([0, 1, 7, 33, 56, 128, 257, 448], 3, 960, 1024),
        ([0, 1, 7, 33, 56, 128, 257, 448], 0, 960, 1024),
        ([8192] * 8, 0, 65536, 65536),
        ([0] * 8, 0, 65536, 65536),
    ],
)
@pytest.mark.parametrize("pow2_buckets", [False, True])
def test_recv_bound_deduplicated_bound(
    make_dispatcher, sender_rows, rank, expected, expected_pow2, pow2_buckets
):
    dispatcher = make_dispatcher(
        "epv2",
        recv_rows=65536,
        local_rows=sender_rows[rank],
        sender_rows=sender_rows,
        rank=rank,
        pow2=pow2_buckets,
    )
    assert dispatcher._select_recv_cap() == (
        expected_pow2 if pow2_buckets else expected
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender_rows": None},
        {"sender_rows": [7] * 7},
        {"sender_rows": [7] * 7 + [-1]},
        {"local_rows": 8},
        {"tp_size": 16, "attn_tp_size": 2, "moe_tp_size": 2},
        {"attn_cp_size": 2, "attn_dp_size": 4},
        {"moe_ep_size": 1, "moe_tp_size": 8},
        {"moe_ep_size": 4, "moe_tp_size": 2},
        {"moe_ep_size": 4, "moe_dp_size": 2, "attn_cp_size": 2, "attn_dp_size": 4},
    ],
)
def test_recv_bound_falls_back_when_safety_is_unproved(mock_get_parallel, overrides):
    parallel = mock_get_parallel.return_value
    kwargs = {
        "local_rows": 7,
        "sender_rows": [7] * 8,
    }
    for name, value in overrides.items():
        if hasattr(parallel, name):
            setattr(parallel, name, value)
        else:
            kwargs[name] = value
    kwargs["sender_rows"] = adapter.normalize_sender_rows(kwargs["sender_rows"])
    recv_cap = mori_recv_bound(**kwargs)
    assert recv_cap == 0


@pytest.mark.parametrize(
    "attn_dp_size,group_rows,rank,expected",
    [
        # DP attention with attention TP 2: each DP group split over two ranks.
        (4, [16, 0, 33, 64], 5, 113),
        (4, [16, 0, 33, 64], 2, 113),
    ],
)
def test_recv_bound_sums_attention_tp_scattered_groups(
    mock_get_parallel, attn_dp_size, group_rows, rank, expected
):
    attn_tp_size = 8 // attn_dp_size
    attn_dp_rank, attn_tp_rank = divmod(rank, attn_tp_size)
    local_rows = (
        torch.arange(group_rows[attn_dp_rank])
        .tensor_split(attn_tp_size)[attn_tp_rank]
        .numel()
    )
    mock_get_parallel.return_value = _recv_bound_parallel(
        rank=rank, attn_tp_size=attn_tp_size
    )
    recv_cap = mori_recv_bound(
        local_rows=local_rows,
        sender_rows=tuple(group_rows),
    )
    assert recv_cap == expected


@pytest.mark.parametrize("stale_sender_rows", [None, [1152], [8]])
def test_recv_bound_without_dp_attention_ignores_stale_metadata(
    mock_get_parallel, stale_sender_rows
):
    """Without DP attention, DSpark draft forwards keep the target step's
    global_num_tokens; e.g. a stale 1152 against a real 1153-row draft batch must
    never bound a rank below the rows it can receive."""
    bounds = []
    for total_rows in [1153, 1160, 16384, 1151, 1152, *range(0, 80)]:
        for rank, chunk in enumerate(torch.arange(total_rows).tensor_split(8)):
            mock_get_parallel.return_value = _recv_bound_parallel(
                rank=rank, attn_tp_size=8
            )
            recv_cap = mori_recv_bound(
                local_rows=chunk.numel(),
                sender_rows=adapter.normalize_sender_rows(stale_sender_rows),
            )
            bounds.append((total_rows, rank, recv_cap))

    assert all(cap >= total for total, _, cap in bounds)
    for total_rows, _, recv_cap in bounds:
        if total_rows > 0:
            # Local-only slack is at most 2 * attn_tp_size - 1 rows before rounding.
            assert recv_cap < total_rows + 15


def test_recv_bound_returns_unrounded_rows(mock_get_parallel):
    recv_cap = mori_recv_bound(
        local_rows=7,
        sender_rows=(7,) * 8,
    )
    assert recv_cap == 56


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "sender_rows,rank,expected",
    [
        ([1] * 8, 0, 32),
        ([56] * 8, 3, 448),
        ([0, 1, 7, 33, 56, 128, 257, 448], 3, 960),
        ([0, 1, 7, 56], 0, 64),
        ([0, 56], 0, 64),
    ],
)
def test_dispatcher_bound_uses_sender_sum(
    make_dispatcher, kind, sender_rows, rank, expected
):
    dispatcher = make_dispatcher(
        kind,
        sender_rows=sender_rows,
        local_rows=sender_rows[rank],
        rank=rank,
        ep_size=len(sender_rows),
    )
    assert dispatcher._select_recv_cap() == expected


@pytest.mark.parametrize("kind", _KINDS)
def test_dispatcher_bound_ignores_stale_non_dpa_metadata(make_dispatcher, kind):
    dispatcher = make_dispatcher(
        kind, sender_rows=[1152], local_rows=7, rank=7, attn_tp_size=8
    )
    assert dispatcher._select_recv_cap() == 64


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"local_rows": 8}, 65536),
        ({"sender_rows": None}, 65536),
        ({"sender_rows": [7] * 7}, 65536),
        ({"tbo": True}, 65536),
        ({"recv_rows": 64}, 64),
        ({"recv_rows": 40}, 40),
        ({"sender_rows": [9000] * 8, "local_rows": 9000}, 65536),
        ({"enabled": False}, 0),
    ],
)
def test_dispatcher_bound_falls_back(make_dispatcher, kind, overrides, expected):
    dispatcher = make_dispatcher(kind, **overrides)
    assert dispatcher._select_recv_cap() == expected


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "enabled,tbo,verified,expected",
    [(False, False, True, 0), (True, True, True, 65536), (True, False, False, 65536)],
)
def test_dispatcher_checks_eligibility_before_computing_bound(
    make_dispatcher, monkeypatch, kind, enabled, tbo, verified, expected
):
    dispatcher = make_dispatcher(kind, enabled=enabled, tbo=tbo)
    dispatcher._is_recv_layout_verified = Mock(return_value=verified)
    calculate = Mock(side_effect=AssertionError("recv bound must not be calculated"))
    monkeypatch.setattr(adapter, "mori_recv_bound", calculate)
    assert dispatcher._select_recv_cap() == expected
    calculate.assert_not_called()


@pytest.mark.parametrize("kind", ["epv1", "epv1_ll"])
@pytest.mark.parametrize(
    "kernel_type,verified",
    [
        ("IntraNode", True),
        ("AsyncLL", True),
        ("InterNodeV1", False),
        ("InterNodeV1LL", False),
        (None, False),
    ],
)
def test_epv1_layout_check(make_dispatcher, kind, kernel_type, verified):
    dispatcher = make_dispatcher(kind)
    dispatcher.mori_op.config.kernel_type = kernel_type
    assert dispatcher._is_recv_layout_verified() is verified
    assert dispatcher._select_recv_cap() == (64 if verified else 65536)


@pytest.mark.parametrize(
    "backend,internode,verified",
    [
        ("flydsl", False, True),
        ("hip", False, True),
        ("unknown", False, False),
        ("flydsl", True, False),
        ("hip", True, False),
    ],
)
def test_epv2_layout_check(make_dispatcher, backend, internode, verified):
    dispatcher = make_dispatcher("epv2")
    dispatcher.mori_op.backend_name = backend
    dispatcher.mori_op.cfg.is_internode = internode
    assert dispatcher._is_recv_layout_verified() is verified
    assert dispatcher._select_recv_cap() == (64 if verified else 65536)


def test_epv2_missing_layout_metadata_keeps_full_view(make_dispatcher):
    dispatcher = make_dispatcher("epv2")
    del dispatcher.mori_op.backend_name
    del dispatcher.mori_op.cfg.is_internode
    assert dispatcher._select_recv_cap() == 65536


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("manual_cap,expected", [(1, 1), (33, 33), (100000, 65536)])
def test_manual_cap_overrides_automatic_bound(
    make_dispatcher, kind, enabled, manual_cap, expected
):
    dispatcher = make_dispatcher(
        kind,
        enabled=enabled,
        manual_cap=manual_cap,
        sender_rows=None,
        tbo=True,
        pow2=True,
    )
    dispatcher._is_recv_layout_verified = Mock(side_effect=AssertionError("unused"))
    assert dispatcher._select_recv_cap() == expected
    dispatcher._is_recv_layout_verified.assert_not_called()


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("manual_cap", [-1, 0])
def test_nonpositive_manual_cap_preserves_automatic_bound(
    make_dispatcher, kind, manual_cap
):
    dispatcher = make_dispatcher(kind, manual_cap=manual_cap)
    assert dispatcher._select_recv_cap() == 64


@pytest.mark.parametrize("kind", _KINDS)
def test_sender_snapshot_survives_dispatch_a_b_gap(make_dispatcher, monkeypatch, kind):
    import sglang.srt.layers.dp_attention as dp_attention

    sender_rows = [7] * 8
    read_metadata = Mock(side_effect=lambda: sender_rows)
    monkeypatch.setattr(dp_attention, "get_dp_global_num_tokens", read_metadata)
    dispatcher = make_dispatcher(kind)
    dispatcher._dispatch_core = Mock(return_value=(None,) * 5)
    topk = SimpleNamespace(
        topk_ids=torch.zeros((7, 2), dtype=torch.int32),
        topk_weights=torch.ones((7, 2)),
    )
    dispatcher.dispatch_a(torch.zeros((7, 4), dtype=torch.bfloat16), topk)
    sender_rows[:] = [8192] * 8
    assert dispatcher._select_recv_cap() == 64
    assert dispatcher._dispatch_sender_rows == (7,) * 8
    read_metadata.assert_called_once_with()


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "manual_cap,enabled", [(0, False), (0, True), (33, False), (33, True)]
)
@pytest.mark.parametrize("dynamic_api", [False, True])
def test_common_dispatch_b_selects_and_propagates_cap(
    make_dispatcher, monkeypatch, kind, manual_cap, enabled, dynamic_api
):
    dispatcher = make_dispatcher(
        kind, manual_cap=manual_cap, enabled=enabled, pow2=dynamic_api, recv_rows=2048
    )
    dispatcher._snapshot_sender_rows = Mock(return_value=(7,) * 8)
    dispatcher._select_recv_cap = Mock(wraps=dispatcher._select_recv_cap)
    hidden = torch.zeros((2048, 4), dtype=torch.bfloat16)
    weights = torch.ones((2048, 2))
    ids = torch.zeros((2048, 2), dtype=torch.int32)
    recv_count = torch.tensor([56])
    raw_output = (hidden, weights, None, ids, recv_count)
    if kind == "epv2":
        dispatcher.mori_op.dispatch = Mock(return_value=(*raw_output, object()))
        if dynamic_api:
            dispatcher.mori_op.prepare_recv_cap = Mock()
    else:
        dispatcher._dispatch_core = Mock(
            return_value=raw_output if kind == "epv1_ll" else (*raw_output, None)
        )

    outer = MoriEPDispatcher.__new__(MoriEPDispatcher)
    outer._stage = _Stage.INITIAL
    outer._get_impl = Mock(return_value=dispatcher)
    topk = SimpleNamespace(topk_ids=ids[:7], topk_weights=weights[:7])
    outer.dispatch_a(hidden[:7], topk)
    result = outer.dispatch_b()
    expected = manual_cap or (64 if enabled else 0)
    assert result.recv_cap == expected
    assert isinstance(
        result,
        MoriEPLLDispatchOutput if kind == "epv1_ll" else MoriEPNormalDispatchOutput,
    )
    dispatcher._select_recv_cap.assert_called_once_with()
    assert outer._stage == _Stage.AFTER_DISPATCH_B
    if kind == "epv2":
        kwargs = dispatcher.mori_op.dispatch.call_args.kwargs
        if dynamic_api:
            assert kwargs["recv_cap"] == (
                2048 if manual_cap or not enabled else expected
            )
            dispatcher.mori_op.prepare_recv_cap.assert_not_called()
        else:
            assert "recv_cap" not in kwargs
    else:
        dispatcher.mori_op.max_num_tokens_to_recv.assert_called_once_with()


@pytest.mark.parametrize("ep_size,expected", [(2, 128), (4, 224)])
def test_epv2_deduplicated_bound_for_smaller_ep_groups(
    make_dispatcher, ep_size, expected
):
    dispatcher = make_dispatcher(
        "epv2",
        recv_rows=ep_size * 8192,
        local_rows=56,
        sender_rows=(56,) * ep_size,
        ep_size=ep_size,
    )
    assert dispatcher._select_recv_cap() == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
