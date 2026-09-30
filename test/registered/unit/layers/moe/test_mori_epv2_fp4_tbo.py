import logging
import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

import sglang.srt.arg_groups.moe_hook as moe_hook
import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    CombineDtype,
    DispatchDtype,
    _get_epv2_launch_config,
    _MoriEPv1DispatcherImplBase,
    _MoriEPv2DispatcherImplNormal,
    _MoriEPv2LaunchConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_tbo_launch_config_defaults_are_phase_specific():
    assert _get_epv2_launch_config(
        tbo_enabled=True,
        dispatch_block_num=32,
        combine_block_num=48,
        dispatch_warp_num_per_block=4,
        combine_warp_num_per_block=4,
    ) == _MoriEPv2LaunchConfig(32, 4, 48, 4)


def test_non_tbo_launch_config_preserves_tuned_schedule():
    assert _get_epv2_launch_config(
        tbo_enabled=False,
        dispatch_block_num=-1,
        combine_block_num=-1,
        dispatch_warp_num_per_block=-1,
        combine_warp_num_per_block=-1,
    ) == _MoriEPv2LaunchConfig(None, None, None, None)


@pytest.mark.parametrize("field", range(4))
def test_tbo_launch_config_rejects_non_positive_values(field):
    values = [32, 48, 4, 4]
    values[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        _get_epv2_launch_config(
            tbo_enabled=True,
            dispatch_block_num=values[0],
            combine_block_num=values[1],
            dispatch_warp_num_per_block=values[2],
            combine_warp_num_per_block=values[3],
        )


def _dispatcher_for_quant_test():
    dispatcher = _MoriEPv2DispatcherImplNormal.__new__(_MoriEPv2DispatcherImplNormal)
    dispatcher.dispatch_dtype = DispatchDtype.bf16
    dispatcher.fp4_quant_func = object()
    dispatcher._mori_op = None
    dispatcher._initialize_op = Mock()
    return dispatcher


def test_quant_config_selects_fp4_asymmetric_transport(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert dispatcher.dispatch_dtype == DispatchDtype.fp4
    dispatcher._initialize_op.assert_called_once_with()


# Same auto dispatch dtype as EPv1 for each weight dtype.
@pytest.mark.parametrize(
    "weight_dtype,expected",
    [
        (torch.bfloat16, DispatchDtype.bf16),
        (torch.float8_e4m3fn, DispatchDtype.fp8),
        (torch.float8_e4m3fnuz, DispatchDtype.fp8),
    ],
)
def test_quant_config_default_dispatch_dtype(monkeypatch, weight_dtype, expected):
    monkeypatch.delenv("SGLANG_MORI_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": weight_dtype})
    assert dispatcher.dispatch_dtype == expected


def test_fp4_override_and_invalid_override(monkeypatch):
    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "fp4")
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == DispatchDtype.fp4

    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "invalid")
    with pytest.raises(ValueError, match="must be auto, bf16, fp8, fp4 or mxfp8"):
        dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})


@pytest.fixture
def combine_dispatcher(monkeypatch, request):
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "auto")
    monkeypatch.delenv("SGLANG_MORI_COMBINE_DTYPE", raising=False)
    monkeypatch.delenv("SGLANG_MORI_FP8_COMB", raising=False)
    monkeypatch.setattr(adapter, "logger", logging.getLogger(request.node.nodeid))
    monkeypatch.setattr(adapter, "_mori_fp8_direct_cast_saturates", lambda: True)
    return _dispatcher_for_quant_test()


@pytest.mark.parametrize("value", [None, "", "auto", "bf16", "BF16"])
def test_supported_combine_dtype_is_quiet(
    monkeypatch, caplog, combine_dispatcher, value
):
    if value is not None:
        monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", value)
    combine_dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert combine_dispatcher.combine_dtype == CombineDtype.bf16
    assert not caplog.messages


@pytest.mark.parametrize(
    "value,expected",
    [
        ("fp8", CombineDtype.fp8),
        ("FP8", CombineDtype.fp8),
        ("fp8_direct_cast", CombineDtype.fp8_direct_cast),
    ],
)
def test_quantized_combine_needs_bf16_dispatch(
    monkeypatch, caplog, combine_dispatcher, value, expected
):
    """MORI EPv2 rejects a quantized combine on a quantized-dispatch op."""
    monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", value)
    combine_dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert combine_dispatcher.combine_dtype == expected
    assert not caplog.messages

    for _ in range(2):
        combine_dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert combine_dispatcher.combine_dtype == CombineDtype.bf16
    assert combine_dispatcher.dispatch_dtype == DispatchDtype.fp4
    assert len(caplog.messages) == 1
    assert "requires bf16 dispatch" in caplog.messages[0]


def test_fp4_combine_warns_once_and_falls_back(monkeypatch, caplog, combine_dispatcher):
    monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", "fp4")
    for _ in range(2):
        combine_dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert combine_dispatcher.combine_dtype == CombineDtype.bf16
    assert len(caplog.messages) == 1
    assert "SGLANG_MORI_COMBINE_DTYPE=fp4" in caplog.messages[0]


@pytest.mark.parametrize("is_epv2", [True, False])
@pytest.mark.parametrize("saturates", [True, False])
def test_nan_direct_cast_falls_back_to_fp8_combine(
    monkeypatch, caplog, combine_dispatcher, is_epv2, saturates
):
    """A MORI whose fp8_direct_cast emits NaN past the fp8 max must not be used."""
    monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", "fp8_direct_cast")
    monkeypatch.setattr(adapter, "_mori_fp8_direct_cast_saturates", lambda: saturates)
    dispatcher = combine_dispatcher
    if not is_epv2:
        dispatcher = _MoriEPv1DispatcherImplBase.__new__(_MoriEPv1DispatcherImplBase)
    for _ in range(2):
        dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    if saturates:
        assert dispatcher.combine_dtype == CombineDtype.fp8_direct_cast
        assert not caplog.messages
    else:
        assert dispatcher.combine_dtype == CombineDtype.fp8
        assert len(caplog.messages) == 1
        assert "emits NaN" in caplog.messages[0]


def _dsv4_fp8_combine_cfg():
    # MORI cfg of DSv4-Pro TP8 EP8, bf16 dispatch, fp8_blockwise (scatter) combine.
    return SimpleNamespace(
        effective_max_recv=8 * 16384,
        max_num_inp_token_per_rank=16384,
        num_experts_per_token=7,
        world_size=8,
        hidden_dim=7168,
        token_nbytes=7168 * 2,
        combine_token_nbytes=7168 * 2,
        scale_dim=0,
        scale_type_size=0,
        is_scatter=True,
        wire_elem_size=1,
        fp8_blockwise=True,
        combine_scale_dim=7168 // 128,
    )


def test_epv2_vmm_budget_covers_scatter_combine_arena():
    """A quantized combine adds a scatter staging arena that outgrew a fixed 4 GiB window."""
    cfg = _dsv4_fp8_combine_cfg()
    # 4.413 GiB measured from MORI's flydsl SymmArena layout for this cfg.
    assert adapter._epv2_arena_bytes(cfg) >= int(4.413 * (1 << 30))
    with adapter.envs.SGLANG_MORI_EPV2_PER_RANK_VMM_GB.override(None):
        assert adapter._epv2_per_rank_vmm_gb(cfg) == 5
    with adapter.envs.SGLANG_MORI_EPV2_PER_RANK_VMM_GB.override(4):
        with pytest.raises(ValueError, match="SGLANG_MORI_EPV2_PER_RANK_VMM_GB=4"):
            adapter._epv2_per_rank_vmm_gb(cfg)


def test_invalid_combine_dtype_fails_before_initialization(
    monkeypatch, combine_dispatcher
):
    monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", "fp88")
    monkeypatch.setenv("SGLANG_MORI_FP8_COMB", "1")
    with pytest.raises(ValueError, match="SGLANG_MORI_COMBINE_DTYPE.*fp88"):
        combine_dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    combine_dispatcher._initialize_op.assert_not_called()


@pytest.mark.parametrize("legacy", ["0", "1"])
@pytest.mark.parametrize("current", [None, "", "auto", "bf16"])
def test_combine_dtype_takes_precedence_over_legacy_flag(
    monkeypatch, caplog, combine_dispatcher, legacy, current
):
    monkeypatch.setenv("SGLANG_MORI_FP8_COMB", legacy)
    if current is not None:
        monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", current)
    combine_dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    if current is None:
        assert combine_dispatcher.combine_dtype == (
            CombineDtype.fp8 if legacy == "1" else CombineDtype.bf16
        )
        assert len(caplog.messages) == 1
        assert "SGLANG_MORI_FP8_COMB is deprecated" in caplog.messages[0]
    else:
        assert combine_dispatcher.combine_dtype == CombineDtype.bf16
        assert not caplog.messages


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("comm_stream", [False, True])
def test_recv_capacity_api_compatibility(monkeypatch, dynamic, comm_stream):
    op = SimpleNamespace(
        cfg=SimpleNamespace(effective_max_recv=64),
        dispatch=Mock(return_value=(None, None, None, None, None, object())),
    )
    if dynamic:
        op.prepare_recv_cap = Mock()
    monkeypatch.setattr(adapter, "init_mori_epv2_op", Mock(return_value=op))
    monkeypatch.setattr(adapter, "get_int_env_var", lambda name, default: default)
    monkeypatch.setattr(torch, "cuda", MagicMock())
    dispatcher = Mock()
    _MoriEPv2DispatcherImplNormal._initialize_op(dispatcher)
    assert dispatcher._recv_cap_pow2_buckets is dynamic
    if dynamic:
        assert op.prepare_recv_cap.call_args_list == [call(32), call(64)]
    dispatcher.mori_op = op
    dispatcher._trim_recv = False
    dispatcher._direct_output = False
    dispatcher._recv_cap = 32
    dispatcher._manual_recv_cap = 0
    dispatcher._comm_stream = Mock() if comm_stream else None
    _MoriEPv2DispatcherImplNormal.dispatch_b(dispatcher, *((None,) * 5 + (Mock(),)))
    kwargs = {"return_routing": True}
    if dynamic:
        kwargs.update(recv_cap=32, clone_routing=False)
    op.dispatch.assert_called_once_with(None, None, None, None, **kwargs)


@pytest.fixture
def fake_mori_epv2_modules(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in ("mori", "mori.cco", "mori.ops", "mori.ops.dispatch_combine_v2")
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    modules["mori"].cco = modules["mori.cco"]
    modules["mori"].ops = modules["mori.ops"]
    modules["mori.ops"].dispatch_combine_v2 = modules["mori.ops.dispatch_combine_v2"]
    cco, epv2 = modules["mori.cco"], modules["mori.ops.dispatch_combine_v2"]
    cco.Communicator = Mock()
    epv2.EpDispatchCombineOp = Mock()
    return cco, epv2


@pytest.mark.parametrize("tbo", [False, True])
@pytest.mark.parametrize("setting,expected", [(None, True), ("0", False), ("1", True)])
def test_direct_output_setting_applies_to_tbo(
    monkeypatch, fake_mori_epv2_modules, tbo, setting, expected
):
    if setting is None:
        monkeypatch.delenv("SGLANG_MORI_EPV2_AITER_DIRECT_OUTPUT", raising=False)
    else:
        monkeypatch.setenv("SGLANG_MORI_EPV2_AITER_DIRECT_OUTPUT", setting)
    monkeypatch.setattr(adapter, "_use_aiter", False)
    monkeypatch.setattr(adapter, "is_tbo_enabled", lambda: tbo)
    monkeypatch.setattr(adapter, "_get_tbo_comm_stream", lambda *args, **kwargs: None)
    dispatcher = _MoriEPv2DispatcherImplNormal(
        group=Mock(),
        router_topk=2,
        permute_fusion=False,
        num_experts=4,
        num_local_experts=2,
        hidden_size=4,
        params_dtype=torch.bfloat16,
        deepep_mode=adapter.DeepEPMode.NORMAL,
        async_finish=True,
        instance_id=1,
    )
    assert dispatcher._direct_output is expected


def test_tbo_instances_have_independent_cached_combine_buffers(
    monkeypatch, fake_mori_epv2_modules
):
    cco, epv2 = fake_mori_epv2_modules
    cco.Communicator.get_unique_id.side_effect = object
    cco.Communicator.init.side_effect = lambda *args, **kwargs: SimpleNamespace(
        barrier=Mock()
    )
    epv2.EpDispatchCombineConfig = lambda **kwargs: SimpleNamespace(
        **kwargs, effective_max_recv=8, schedule="test"
    )

    def make_op(cfg, comm):
        arena = torch.empty((8, cfg.hidden_dim), dtype=torch.bfloat16)
        return SimpleNamespace(cfg=cfg, comm=comm, combine_in_view=lambda: arena)

    epv2.EpDispatchCombineOp.side_effect = make_op
    monkeypatch.setattr(
        adapter, "get_parallel", lambda: SimpleNamespace(moe_ep_size=2, moe_ep_rank=0)
    )
    monkeypatch.setattr(adapter, "_epv2_per_rank_vmm_gb", lambda cfg: 4)
    group = Mock()
    group.broadcast_object.side_effect = lambda obj, src: obj
    kwargs = dict(
        group=group,
        router_topk=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=4,
        params_dtype=torch.bfloat16,
        max_tokens_per_rank=4,
    )
    adapter.init_mori_epv2_op.cache_clear()
    adapter._init_cco_communicator.cache_clear()
    try:
        first = adapter.init_mori_epv2_op(**kwargs, instance_id=0)
        second = adapter.init_mori_epv2_op(**kwargs, instance_id=1)
        assert adapter.init_mori_epv2_op(**kwargs, instance_id=0) is first
        assert first is not second
        assert first.comm is not second.comm
        assert first.combine_in_view().data_ptr() != second.combine_in_view().data_ptr()
        assert epv2.EpDispatchCombineOp.call_count == 2
        assert cco.Communicator.init.call_count == 2
    finally:
        adapter.init_mori_epv2_op.cache_clear()
        adapter._init_cco_communicator.cache_clear()


@pytest.mark.parametrize("comm_stream", [False, True])
@pytest.mark.parametrize("direct_output", [False, True])
@pytest.mark.parametrize("recv_cap", [0, 3])
def test_tbo_direct_output_isolated_and_ordered_across_reuse(
    monkeypatch, comm_stream, direct_output, recv_cap
):
    trace = []
    compute, comm = Mock(name="compute"), Mock(name="comm")
    active_stream = [compute]
    for stream in (compute, comm):
        stream.wait_event.side_effect = lambda event, stream=stream: trace.append(
            ("wait", stream, event)
        )

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    def make_event(**kwargs):
        event = Mock(name="event")
        event.record.side_effect = lambda stream: trace.append(
            ("record", stream, event)
        )
        return event

    monkeypatch.setattr(
        torch,
        "cuda",
        SimpleNamespace(
            current_stream=lambda: active_stream[0],
            stream=use_stream,
            Event=make_event,
            is_current_stream_capturing=lambda: False,
        ),
    )
    children, arenas, routes = [], [], {}
    for child_id in range(2):
        arena = torch.full((8, 4), -1, dtype=torch.bfloat16)
        arenas.append(arena)
        raw = (
            torch.empty_like(arena),
            torch.ones((8, 2)),
            None,
            torch.zeros((8, 2), dtype=torch.int32),
            torch.tensor([2]),
        )

        def dispatch(*args, child_id=child_id, raw=raw, **kwargs):
            assert active_stream[0] is (comm if comm_stream else compute)
            trace.append(("dispatch", child_id))
            routes[child_id] = object()
            return (*raw, routes[child_id])

        def combine(hidden, *args, routing, child_id=child_id, arena=arena):
            assert active_stream[0] is (comm if comm_stream else compute)
            assert routing is routes[child_id]
            assert (hidden.data_ptr() == arena.data_ptr()) is direct_output
            trace.append(("combine", child_id))
            return hidden[:2].clone(), None

        child = _MoriEPv2DispatcherImplNormal.__new__(_MoriEPv2DispatcherImplNormal)
        child._mori_op = SimpleNamespace(
            dispatch=Mock(side_effect=dispatch),
            combine=Mock(side_effect=combine),
            combine_in_view=lambda arena=arena: arena,
        )
        child._direct_output = direct_output
        child._comm_stream = comm if comm_stream else None
        child.async_finish = True
        child.dispatch_dtype = DispatchDtype.bf16
        child._recv_cap = recv_cap
        children.append(child)

    assert arenas[0].data_ptr() != arenas[1].data_ptr()
    rows = recv_cap or 8
    for iteration in range(3):
        outputs = []
        for child_id, child in enumerate(children):
            ready = child._capture_event_if_async()
            start = len(trace)
            output = child.dispatch_b(None, None, None, None, torch.bfloat16, ready)
            if comm_stream:
                done = trace[start + 2][2]
                assert trace[start:] == [
                    ("wait", comm, ready),
                    ("dispatch", child_id),
                    ("record", comm, done),
                    ("wait", compute, done),
                ]
            outputs.append(output)
        intermediates = []
        for child_id, (child, output) in enumerate(zip(children, outputs)):
            if direct_output:
                assert output.expert_output.data_ptr() == arenas[child_id].data_ptr()
                expert_out = output.expert_output[:rows]
            else:
                assert output.expert_output is None
                expert_out = torch.empty((rows, 4), dtype=torch.bfloat16)
            expert_out.fill_(iteration * 10 + child_id + 1)
            intermediates.append(
                child.combine_a(expert_out, output.topk_ids, output.topk_weights)
            )
        # Consume in the opposite order with both children's expert outputs live.
        for child_id in (1, 0):
            start = len(trace)
            ready = intermediates[child_id][-1]
            result = children[child_id].combine_b(*intermediates[child_id])
            if comm_stream:
                done = trace[start + 2][2]
                assert trace[start:] == [
                    ("wait", comm, ready),
                    ("combine", child_id),
                    ("record", comm, done),
                    ("wait", compute, done),
                ]
            torch.testing.assert_close(
                result, torch.full_like(result, iteration * 10 + child_id + 1)
            )
        if direct_output and rows < 8:
            for arena in arenas:
                torch.testing.assert_close(
                    arena[rows:], torch.full_like(arena[rows:], -1)
                )


def test_direct_output_without_combine_view_falls_back():
    child = _MoriEPv2DispatcherImplNormal.__new__(_MoriEPv2DispatcherImplNormal)
    child._mori_op = SimpleNamespace(
        dispatch=Mock(return_value=(None, None, None, None, None, object()))
    )
    child._direct_output = True
    child._comm_stream = None
    child._recv_cap = 0
    output = child.dispatch_b(None, None, None, None, torch.bfloat16, None)
    assert output.expert_output is None


@pytest.mark.parametrize(
    "backend,expected",
    [("mori", True), ("deepep", False), ("none", True), ("flashinfer", True)],
)
def test_epv2_dp_graph_sync_respects_explicit_backend(monkeypatch, backend, expected):
    import sglang.srt.layers.moe.utils as moe_utils
    import sglang.srt.utils.common as common

    monkeypatch.setenv("SGLANG_MORI_EP_VERSION", "epv2")
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "0")
    monkeypatch.setattr(
        moe_utils, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend.MORI
    )
    monkeypatch.setattr(
        common,
        "get_parallel",
        lambda: SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            dp_size=8,
            tp_size=8,
            moe_dense_tp_size=1,
        ),
    )
    monkeypatch.setattr(
        common,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
    )

    assert common.require_mlp_tp_gather(moe_a2a_backend=backend) is expected
    assert common.require_mlp_tp_gather() is True


_MORI_VERSION_ENVS = (
    "SGLANG_MORI_EP_VERSION",
    "SGLANG_MORI_DISPATCH_DTYPE",
    "SGLANG_MORI_COMBINE_DTYPE",
    "SGLANG_MORI_FP8_COMB",
    "SGLANG_MORI_FP8_DISP",
    "SGLANG_MORI_FP4_DISP",
    "MORI_ENABLE_SDMA",
)


@pytest.fixture
def epv2_capable_env(monkeypatch):
    # setenv first so monkeypatch restores whatever a delenv removes.
    for name in _MORI_VERSION_ENVS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")
    monkeypatch.setattr(moe_hook, "_mori_epv2_installed", lambda: True)


@pytest.mark.parametrize(
    "explicit,reason,expected,env_after",
    [
        (None, None, "epv2", None),
        # The fallback must reach the env: worker processes re-read it.
        (None, "x", "epv1", "epv1"),
        # An explicit epv2 is honored, never silently downgraded.
        ("epv2", "x", "epv2", "epv2"),
    ],
)
def test_mori_ep_version_prefers_epv2_and_falls_back(
    monkeypatch, epv2_capable_env, explicit, reason, expected, env_after
):
    if explicit is not None:
        monkeypatch.setenv("SGLANG_MORI_EP_VERSION", explicit)
    monkeypatch.setattr(
        moe_hook, "_mori_epv2_unsupported_reason", lambda server_args: reason
    )
    assert moe_hook._resolve_mori_ep_version(None) == expected
    assert os.environ.get("SGLANG_MORI_EP_VERSION") == env_after


@pytest.mark.parametrize(
    "env,supported",
    [
        ({}, True),
        # Auto dispatch is fp4 for fp4 weights, where EPv2 drops the fp8 combine.
        ({"SGLANG_MORI_COMBINE_DTYPE": "fp8"}, False),
        ({"SGLANG_MORI_FP8_COMB": "1"}, False),
        (
            {"SGLANG_MORI_COMBINE_DTYPE": "fp8", "SGLANG_MORI_DISPATCH_DTYPE": "bf16"},
            True,
        ),
        ({"SGLANG_MORI_COMBINE_DTYPE": "fp4"}, False),
        ({"SGLANG_MORI_FP4_DISP": "0"}, False),
        ({"MORI_ENABLE_SDMA": "1"}, False),
        ({"SGLANG_USE_AITER": "0"}, False),
    ],
)
def test_mori_epv2_env_support(monkeypatch, epv2_capable_env, env, supported):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reason = moe_hook._mori_epv2_config_unsupported_reason(ep_size=8)
    assert (reason is None) == supported


def test_mori_epv2_topology_support(epv2_capable_env):
    assert moe_hook._mori_epv2_config_unsupported_reason(ep_size=8) is None
    assert moe_hook._mori_epv2_config_unsupported_reason(ep_size=16) is not None


def test_mori_epv2_unvalidated_model_only_warns(monkeypatch, caplog):
    monkeypatch.setattr(moe_hook, "logger", logging.getLogger("mori_epv2_model"))
    for architecture in ("DeepseekV4ForCausalLM", "DeepseekV3ForCausalLM"):
        moe_hook._warn_if_mori_epv2_unvalidated_model(
            hf_config=SimpleNamespace(architectures=[architecture])
        )
    assert len(caplog.messages) == 1
    assert "DeepseekV3ForCausalLM" in caplog.messages[0]
    assert (
        moe_hook._mori_epv2_dtype_unsupported_reason(model_dtype=torch.bfloat16) is None
    )
    assert moe_hook._mori_epv2_dtype_unsupported_reason(model_dtype=torch.float16)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
