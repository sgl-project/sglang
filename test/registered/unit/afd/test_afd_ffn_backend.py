"""FFN plugin ownership and graph warmup lifecycle contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.config import AFDConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_backend_warmup_barrier_is_never_recorded_in_graph(monkeypatch):
    from sglang.srt.afd.pipeline import _finish_backend_warmup

    calls = []
    backend = SimpleNamespace(
        needs_warmup_barrier=True, finish_warmup=lambda: calls.append(1)
    )
    connector = SimpleNamespace(
        transport=SimpleNamespace(ffn_backend=backend), graph_strategy=SimpleNamespace()
    )
    for capturing, in_graph, count in [
        (False, False, 0),
        (True, False, 1),
        (True, True, 1),
    ]:
        connector.graph_strategy.capturing = capturing
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: in_graph)
        _finish_backend_warmup(connector)
        assert len(calls) == count
    connector.graph_strategy.capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    backend.needs_warmup_barrier = False
    _finish_backend_warmup(connector)
    assert calls == [1]


def test_backend_construction_failure_retains_partial_cleanup_ownership(monkeypatch):
    from sglang.srt.afd.expert_exchange import AFDDirectExchange
    from sglang.srt.afd.ffn_backend import build_ffn_backend

    transport = SimpleNamespace(ffn_backend=None)
    config = AFDConfig(lanes=4, attention_lanes=8, experimental_direct=True)
    comm = object()

    def fail_after_first_communicator(self, **kwargs):
        assert transport.ffn_backend is self
        self.communicators.append(("partial", comm))
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(AFDDirectExchange, "__init__", fail_after_first_communicator)
    with pytest.raises(RuntimeError, match="initialization failed"):
        build_ffn_backend(transport=transport, adapter=None, config=config, device=None)
    assert transport.ffn_backend.communicators == [("partial", comm)]
    transport.ffn_backend.close(seconds=1)

    # A disabled plugin neither creates backend resources nor needs an adapter.
    empty_transport = SimpleNamespace()
    assert (
        build_ffn_backend(
            transport=empty_transport, adapter=None, config=AFDConfig(), device=None
        )
        is None
    )
