from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt import runtime_context
from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe.token_dispatcher.mscclpp.dispatcher import (
    MSCCLPPDispatcher,
    _MSCCLPPDispatcherImplLowLatency,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MSCCLPPEPLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


@pytest.fixture
def make_dispatcher(monkeypatch):
    group = Mock(spec=torch.distributed.ProcessGroup)
    resources = (Mock(), Mock(), torch.empty(1), 2)
    key = (group, 2, 1, 4, 1, 128, MSCCLPPEPLayout.EXPERT_MAJOR, False, None)
    monkeypatch.setattr(
        _MSCCLPPDispatcherImplLowLatency, "_shared_resources", {key: resources}
    )
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1),
    )
    monkeypatch.setattr(runtime_context, "max_prefill_buffer_tokens", lambda: 128)
    monkeypatch.setattr(
        _MSCCLPPDispatcherImplLowLatency,
        "_resolve_runtime_decode_capacity",
        classmethod(lambda cls, attn_tp_size: 0),
    )
    monkeypatch.setattr(
        _MSCCLPPDispatcherImplLowLatency,
        "_resolve_enable_direct_send",
        staticmethod(lambda output_layout: False),
    )

    def create():
        return MSCCLPPDispatcher(
            group=group,
            router_topk=1,
            num_experts=2,
            num_local_experts=1,
            hidden_size=4,
            params_dtype=torch.bfloat16,
        )

    return create


@pytest.mark.parametrize("global_num_tokens", [None, [0], [0, 0]])
def test_zero_active_capacity_uses_minimum_native_capacity(
    make_dispatcher, monkeypatch, global_num_tokens
) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    monkeypatch.setattr(
        dp_attention,
        "get_dp_global_num_tokens",
        lambda: global_num_tokens,
    )

    assert implementation._select_active_capacity(num_tokens=0) == 1

    dispatcher.close()


def test_empty_dispatch_passes_minimum_capacity_to_native(
    make_dispatcher, monkeypatch
) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    monkeypatch.setattr(
        dp_attention,
        "get_dp_global_num_tokens",
        lambda: [0, 0],
    )
    hidden_states = torch.empty((0, 4), dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.empty((0, 1), dtype=torch.float32),
        topk_ids=torch.empty((0, 1), dtype=torch.int64),
        router_logits=torch.empty((0, 2), dtype=torch.float32),
    )
    native_output = SimpleNamespace(
        tokens=torch.empty((1, 2, 4), dtype=torch.bfloat16),
        quant=None,
        layout=SimpleNamespace(num_tokens_per_expert=torch.zeros(1, dtype=torch.int32)),
    )
    implementation._moe_comm.dispatch.return_value = (native_output, object())

    dispatch_output = dispatcher.dispatch(hidden_states, topk_output)

    call = implementation._moe_comm.dispatch.call_args
    assert call.args[0] is hidden_states
    assert call.kwargs["runtime_max_tokens_per_rank"] == 1
    assert implementation._active_capacity == 1

    implementation._moe_comm.combine.return_value = hidden_states
    combined = dispatcher.combine(
        SimpleNamespace(
            hidden_states=dispatch_output.hidden_states,
            apply_router_weights=True,
        )
    )
    assert combined is hidden_states
    dispatcher.close()


def test_idle_dp_rank_uses_positive_global_capacity(
    make_dispatcher, monkeypatch
) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    monkeypatch.setattr(
        dp_attention,
        "get_dp_global_num_tokens",
        lambda: [0, 8],
    )

    assert implementation._select_active_capacity(num_tokens=0) == 8

    dispatcher.close()


def test_active_capacity_overflow_still_raises(make_dispatcher, monkeypatch) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    monkeypatch.setattr(
        dp_attention,
        "get_dp_global_num_tokens",
        lambda: [0, 129],
    )

    with pytest.raises(
        RuntimeError,
        match="active=129, allocated=128",
    ):
        implementation._select_active_capacity(num_tokens=0)

    dispatcher.close()


def test_close_releases_instance_ownership_and_is_idempotent(
    make_dispatcher, monkeypatch
) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    close = Mock(wraps=implementation.close)
    monkeypatch.setattr(implementation, "close", close)

    dispatcher.close()
    dispatcher.close()

    close.assert_called_once()
    assert dispatcher._dispatcher is None
    assert implementation._combine_handle is None
    assert implementation._active_capacity is None
    assert implementation._dispatch_output_buffer is None
    assert implementation._moe_comm is None
    assert implementation._ep_group is None


def test_close_preserves_resources_owned_by_another_dispatcher(make_dispatcher) -> None:
    first = make_dispatcher()
    second = make_dispatcher()
    implementation = second._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    group = implementation._ep_group
    communicator = implementation._moe_comm
    output_buffer = implementation._dispatch_output_buffer

    first.close()

    assert implementation._ep_group is group
    assert implementation._moe_comm is communicator
    assert implementation._dispatch_output_buffer is output_buffer
    assert len(_MSCCLPPDispatcherImplLowLatency._shared_resources) == 1
    group.close.assert_not_called()
    communicator.close.assert_not_called()

    second.close()
    MSCCLPPDispatcher.clear_shared_resources()
    assert not _MSCCLPPDispatcherImplLowLatency._shared_resources


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_closed_dispatcher_rejects_operations(make_dispatcher, operation) -> None:
    dispatcher = make_dispatcher()
    dispatcher.close()

    with pytest.raises(RuntimeError, match="MSCCLPPDispatcher is closed"):
        if operation == "dispatch":
            dispatcher.dispatch(torch.empty(0), Mock())
        else:
            dispatcher.combine(Mock())


def test_close_rejects_dispatch_waiting_for_combine(
    make_dispatcher, monkeypatch
) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    monkeypatch.setattr(implementation, "dispatch", Mock(return_value=Mock()))
    monkeypatch.setattr(implementation, "combine", Mock(return_value=torch.empty(0)))
    dispatcher.dispatch(torch.empty(0), Mock())

    with pytest.raises(RuntimeError, match="close called before combine"):
        dispatcher.close()
    assert dispatcher._dispatcher is implementation
    assert implementation._moe_comm is not None

    dispatcher.combine(Mock())
    dispatcher.close()
    assert dispatcher._dispatcher is None


def test_close_rejects_retained_native_dispatch_handle(make_dispatcher) -> None:
    dispatcher = make_dispatcher()
    implementation = dispatcher._dispatcher
    assert isinstance(implementation, _MSCCLPPDispatcherImplLowLatency)
    implementation._combine_handle = object()

    with pytest.raises(RuntimeError, match="close called before combine"):
        dispatcher.close()
    assert dispatcher._dispatcher is implementation
    assert implementation._moe_comm is not None
