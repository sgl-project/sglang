"""CPU tests of Kimi-K3's MLA eager break under prefill breakable CUDA graphs."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    context as bcg_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    context_manager as tc_context,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    get_prefill_raw_num_tokens,
    is_in_prefill_graph_scope,
    prefill_graph_scope,
)
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def prefill_graph_flags():
    """The state a prefill breakable-graph break runs under."""
    static_batch = SimpleNamespace(name="static")
    saved = (
        bcg_context._in_breakable_cuda_graph,
        tc_context._in_tc_piecewise_cuda_graph,
        tc_context._tc_piecewise_forward_context,
        capture_mode.is_capture_mode,
    )
    bcg_context._in_breakable_cuda_graph = True
    tc_context._tc_piecewise_forward_context = SimpleNamespace(
        forward_batch=static_batch
    )
    capture_mode.is_capture_mode = True
    try:
        with prefill_graph_scope(full_graph=False, raw_num_tokens=7):
            yield static_batch
    finally:
        (
            bcg_context._in_breakable_cuda_graph,
            tc_context._in_tc_piecewise_cuda_graph,
            tc_context._tc_piecewise_forward_context,
            capture_mode.is_capture_mode,
        ) = saved


def _assert_graph_state(static_batch):
    assert is_in_breakable_cuda_graph()
    assert get_is_capture_mode()
    assert get_tc_piecewise_forward_context().forward_batch is static_batch
    assert is_in_prefill_graph_scope()
    assert get_prefill_raw_num_tokens() == 7


def test_as_eager_prefill_suspends_and_restores(prefill_graph_flags):
    live = SimpleNamespace(name="live")
    with kimi_k3._as_eager_prefill(live):
        assert not is_in_breakable_cuda_graph()
        assert not get_is_capture_mode()
        assert get_tc_piecewise_forward_context() is None
        assert not is_in_prefill_graph_scope()
        assert get_prefill_raw_num_tokens() is None
        assert get_forward_batch() is live
    _assert_graph_state(prefill_graph_flags)


def test_as_eager_prefill_restores_on_error(prefill_graph_flags):
    with pytest.raises(RuntimeError):
        with kimi_k3._as_eager_prefill(SimpleNamespace()):
            raise RuntimeError("boom")
    _assert_graph_state(prefill_graph_flags)


def _fake_layer(seen, skips_capture_warmup=False):
    def inner(hidden_states, positions, forward_batch, zero_allocator):
        seen.append(
            dict(
                rows=hidden_states.shape[0],
                batch=forward_batch,
                graph=is_in_breakable_cuda_graph() or get_is_capture_mode(),
            )
        )
        return hidden_states * 2

    return SimpleNamespace(
        _run_self_attn_inner=inner,
        _mla_skips_capture_warmup=skips_capture_warmup,
    )


def test_mla_break_runs_live_rows_and_plans_once(prefill_graph_flags, monkeypatch):
    planned = []
    backend = SimpleNamespace(
        full_attn_backend=SimpleNamespace(
            init_forward_metadata=lambda fb: planned.append(
                (fb, is_in_breakable_cuda_graph())
            )
        )
    )
    monkeypatch.setattr(kimi_k3, "get_attn_backend", lambda: backend)
    live = SimpleNamespace(positions=torch.arange(3))
    prefill_graph_flags.live_forward_batch = live
    seen = []
    layer = _fake_layer(seen)
    hidden = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    for _ in range(2):  # two MLA layers of one replay
        out = torch.full((5, 4), 7.0)
        kimi_k3.KimiK3DecoderLayer._mla_self_attn_eager(layer, hidden, out)
        torch.testing.assert_close(out[:3], hidden[:3] * 2)
        assert torch.count_nonzero(out[3:]) == 0
    # The eager plan runs once per live batch, outside the graph flags.
    assert planned == [(live, False)]
    assert [s["rows"] for s in seen] == [3, 3]
    assert all(s["batch"] is live and not s["graph"] for s in seen)
    _assert_graph_state(prefill_graph_flags)


def test_mla_break_without_live_batch_uses_static_batch(
    prefill_graph_flags, monkeypatch
):
    # Capture warm-up: the dummy batch is the static batch and is unpadded.
    monkeypatch.setattr(
        kimi_k3,
        "get_attn_backend",
        lambda: SimpleNamespace(init_forward_metadata=lambda fb: None),
    )
    prefill_graph_flags.positions = torch.arange(4)
    seen = []
    hidden = torch.ones(4, 2)
    out = torch.zeros(4, 2)
    kimi_k3.KimiK3DecoderLayer._mla_self_attn_eager(_fake_layer(seen), hidden, out)
    assert seen[0]["batch"] is prefill_graph_flags and seen[0]["rows"] == 4
    torch.testing.assert_close(out, hidden * 2)


def test_dsa_dcp_break_skips_capture_warmup(prefill_graph_flags, monkeypatch):
    # K3.1 DSA under DCP: the capture batch has no DCP-sharded KV layout, so
    # the warm-up break only zeroes its output; a replay attends the live batch.
    monkeypatch.setattr(
        kimi_k3,
        "get_attn_backend",
        lambda: SimpleNamespace(init_forward_metadata=lambda fb: None),
    )
    prefill_graph_flags.positions = torch.arange(4)
    seen = []
    layer = _fake_layer(seen, skips_capture_warmup=True)
    hidden = torch.ones(4, 2)
    out = torch.full((4, 2), 3.0)
    kimi_k3.KimiK3DecoderLayer._mla_self_attn_eager(layer, hidden, out)
    assert seen == [] and torch.count_nonzero(out) == 0
    prefill_graph_flags.live_forward_batch = SimpleNamespace(positions=torch.arange(2))
    kimi_k3.KimiK3DecoderLayer._mla_self_attn_eager(layer, hidden, out)
    assert [s["rows"] for s in seen] == [2]
    torch.testing.assert_close(out[:2], hidden[:2] * 2)
    assert torch.count_nonzero(out[2:]) == 0
