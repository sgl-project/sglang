"""CPU regressions for NCCL EP initialization and native resource ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
    NcclEpBuffer,
    NcclEpDispatcher,
    _Stage,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("rank_major", [False, True])
def test_wrapped_model_initializes_each_dispatcher_once(monkeypatch, rank_major):
    dispatcher = NcclEpDispatcher.__new__(NcclEpDispatcher)
    dispatcher.layout = SimpleNamespace(is_rank_major=lambda: rank_major)
    dispatcher.init_comm_resources = Mock()
    dispatcher.init_handle_for_graph = Mock()
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in model.model.layers:
        layer.dispatcher = dispatcher
    monkeypatch.setattr(
        moe_utils, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend.NCCL_EP
    )
    ModelRunner.init_nccl_ep_comm_resources(SimpleNamespace(model=model))
    dispatcher.init_comm_resources.assert_called_once()
    assert dispatcher.init_handle_for_graph.call_count == int(not rank_major)


def test_destroy_closes_persistent_handles_before_group(monkeypatch):
    events = []
    dispatcher = NcclEpDispatcher.__new__(NcclEpDispatcher)
    dispatcher._stage = _Stage.INITIAL
    dispatcher._handle_persistent = True
    dispatcher._comm_initialized = True
    dispatcher.handle = SimpleNamespace(destroy=lambda: events.append("handle"))
    state = SimpleNamespace(
        group=SimpleNamespace(destroy=lambda: events.append("group")),
        dispatchers={dispatcher},
    )
    dispatcher.buffer = state
    monkeypatch.setattr(NcclEpBuffer, "_state", classmethod(lambda cls: state))
    NcclEpBuffer.destroy()
    NcclEpBuffer.destroy()
    assert events == ["handle", "group"]
    assert dispatcher.handle is None
    assert dispatcher.buffer is None
    assert not dispatcher._handle_persistent
    assert not dispatcher._comm_initialized


def test_destroy_does_not_release_group_with_active_transaction(monkeypatch):
    dispatcher = NcclEpDispatcher.__new__(NcclEpDispatcher)
    dispatcher._stage = _Stage.AFTER_DISPATCH_A
    group = Mock()
    state = SimpleNamespace(group=group, dispatchers={dispatcher})
    monkeypatch.setattr(NcclEpBuffer, "_state", classmethod(lambda cls: state))
    with pytest.raises(RuntimeError, match="active transaction"):
        NcclEpBuffer.destroy()
    group.destroy.assert_not_called()
