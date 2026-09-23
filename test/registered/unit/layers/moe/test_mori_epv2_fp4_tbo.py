import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.moriep as adapter
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    CombineDtype,
    DispatchDtype,
    _get_epv2_launch_config,
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


def test_quant_config_defaults_to_bf16(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == DispatchDtype.bf16


def test_fp4_override_and_invalid_override(monkeypatch):
    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "fp4")
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == DispatchDtype.fp4

    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "invalid")
    with pytest.raises(ValueError, match="must be auto, bf16 or fp4 for EPv2"):
        dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})


@pytest.fixture
def combine_dispatcher(monkeypatch, request):
    monkeypatch.setenv("SGLANG_MORI_DISPATCH_DTYPE", "auto")
    monkeypatch.delenv("SGLANG_MORI_COMBINE_DTYPE", raising=False)
    monkeypatch.delenv("SGLANG_MORI_FP8_COMB", raising=False)
    monkeypatch.setattr(adapter, "logger", logging.getLogger(request.node.nodeid))
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


@pytest.mark.parametrize("value", ["fp8", "fp8_direct_cast", "fp4", "FP8"])
def test_unsupported_combine_dtype_warns_once_and_falls_back(
    monkeypatch, caplog, combine_dispatcher, value
):
    monkeypatch.setenv("SGLANG_MORI_COMBINE_DTYPE", value)
    for _ in range(2):
        combine_dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert combine_dispatcher.combine_dtype == CombineDtype.bf16
    assert combine_dispatcher.dispatch_dtype == DispatchDtype.fp4
    assert len(caplog.messages) == 1
    assert f"SGLANG_MORI_COMBINE_DTYPE={value.lower()}" in caplog.messages[0]
    assert "falling back to bf16" in caplog.messages[0]


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
    assert combine_dispatcher.combine_dtype == CombineDtype.bf16
    if current is None:
        assert len(caplog.messages) == 1
        assert "SGLANG_MORI_FP8_COMB is deprecated" in caplog.messages[0]
        assert "uses bf16 combine" in caplog.messages[0]
    else:
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
    if dynamic:
        assert op.prepare_recv_cap.call_args_list == [call(32), call(64)]
    dispatcher.mori_op = op
    dispatcher._trim_recv = False
    dispatcher._direct_output = False
    dispatcher._select_recv_cap.return_value = 32
    dispatcher._comm_stream = Mock() if comm_stream else None
    _MoriEPv2DispatcherImplNormal.dispatch_b(dispatcher, *((None,) * 5 + (Mock(),)))
    kwargs = {"return_routing": True}
    if dynamic:
        kwargs.update(recv_cap=32, clone_routing=False)
    op.dispatch.assert_called_once_with(None, None, None, None, **kwargs)


@pytest.mark.parametrize("rows,expected", [(0, 32), (35, 64), (448, 512), (8192, 8192)])
def test_optional_recv_bound_does_not_import_an_unavailable_dispatcher(rows, expected):
    dispatcher = SimpleNamespace(
        mori_op=SimpleNamespace(cfg=SimpleNamespace(effective_max_recv=65536)),
        _trim_recv=True,
    )
    assert _MoriEPv2DispatcherImplNormal._select_recv_cap(dispatcher, rows) == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
