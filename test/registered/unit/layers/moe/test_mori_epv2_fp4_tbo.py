from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

import sglang.srt.layers.moe.token_dispatcher.moriepv2 as adapter
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.moriepv2 import (
    MoriEPv2Dispatcher,
    _resolve_tbo_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_tbo_geometry_defaults_are_phase_specific():
    assert _resolve_tbo_geometry(
        tbo_enabled=True,
        dispatch_block_num=32,
        combine_block_num=48,
        dispatch_warp_num_per_block=4,
        combine_warp_num_per_block=4,
    ) == (32, 4, 48, 4)


def test_non_tbo_geometry_preserves_tuned_schedule():
    assert _resolve_tbo_geometry(
        tbo_enabled=False,
        dispatch_block_num=-1,
        combine_block_num=-1,
        dispatch_warp_num_per_block=-1,
        combine_warp_num_per_block=-1,
    ) == (None, None, None, None)


@pytest.mark.parametrize("field", range(4))
def test_tbo_geometry_rejects_non_positive_values(field):
    values = [32, 48, 4, 4]
    values[field] = 0
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_tbo_geometry(
            tbo_enabled=True,
            dispatch_block_num=values[0],
            combine_block_num=values[1],
            dispatch_warp_num_per_block=values[2],
            combine_warp_num_per_block=values[3],
        )


def _dispatcher_for_quant_test():
    dispatcher = MoriEPv2Dispatcher.__new__(MoriEPv2Dispatcher)
    BaseDispatcher.__init__(dispatcher)
    dispatcher.dispatch_dtype = torch.bfloat16
    dispatcher.fp4_quant_func = object()
    dispatcher._op = None
    dispatcher._initialize_op = Mock()
    return dispatcher


def test_quant_config_selects_fp4_asymmetric_transport(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})
    assert dispatcher.dispatch_dtype == torch.float4_e2m1fn_x2
    dispatcher._initialize_op.assert_called_once_with()


def test_quant_config_defaults_to_bf16(monkeypatch):
    monkeypatch.delenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", raising=False)
    dispatcher = _dispatcher_for_quant_test()
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == torch.bfloat16


def test_fp4_override_and_invalid_override(monkeypatch):
    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", "fp4")
    dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})
    assert dispatcher.dispatch_dtype == torch.float4_e2m1fn_x2

    dispatcher = _dispatcher_for_quant_test()
    monkeypatch.setenv("SGLANG_MORI_EPV2_DISPATCH_DTYPE", "invalid")
    with pytest.raises(ValueError, match="must be bf16 or fp4"):
        dispatcher.set_quant_config({"weight_dtype": torch.bfloat16})


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
    MoriEPv2Dispatcher._initialize_op(dispatcher)
    if dynamic:
        assert op.prepare_recv_cap.call_args_list == [call(32), call(64)]
    dispatcher.op = op
    dispatcher._trim_recv = False
    dispatcher._direct_output = False
    dispatcher._select_recv_cap.return_value = 32
    dispatcher._comm_stream = Mock() if comm_stream else None
    dispatcher._dispatch_intermediate_state = (None,) * 5 + (Mock(),)
    MoriEPv2Dispatcher.dispatch_b(dispatcher)
    kwargs = {"return_routing": True}
    if dynamic:
        kwargs.update(recv_cap=32, clone_routing=False)
    op.dispatch.assert_called_once_with(None, None, None, None, **kwargs)


@pytest.mark.parametrize("rows,expected", [(0, 32), (35, 64), (448, 512), (8192, 8192)])
def test_optional_recv_bound_does_not_import_an_unavailable_dispatcher(rows, expected):
    dispatcher = SimpleNamespace(
        op=SimpleNamespace(cfg=SimpleNamespace(effective_max_recv=65536))
    )
    assert MoriEPv2Dispatcher._select_recv_cap(dispatcher, rows) == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
