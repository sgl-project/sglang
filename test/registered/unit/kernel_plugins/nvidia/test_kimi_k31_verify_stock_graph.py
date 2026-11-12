# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31 sparse verify graph replay above the megakernel's request capacity (K31MK_VERIFY_STOCK_GRAPH).

The graph runner pads a TARGET_VERIFY batch of B requests to the smallest captured size >= B. The megakernel covers
the request batches ``built.request_batches`` (1, 2, 4 on single-request builds; 1, 2, 4, 8 with K31MK_VERIFY_BATCHES), and
plan() returns None for any larger size, so a graph captured at a size above the largest capacity holds the stock
forward. Expected values below come from that rule, not from the code under test: without the knob only batches the
megakernel covers replay (larger ones run eager); with it every TARGET_VERIFY batch replays; other forward modes never
replay through this entry.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.kernel_plugins.nvidia import binding as B
from sglang.srt.kernel_plugins.nvidia import kimi_k31_verify_dsa_binding as D
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BATCHES = range(1, 17)


def _batch(b, verify=True):
    mode = SimpleNamespace(is_target_verify=lambda: verify)
    return SimpleNamespace(batch_size=b, forward_mode=mode, spec_info=SimpleNamespace(), seq_lens_cpu=None)


class _Built:
    tensor_binding_kind = 1

    def __init__(self, request_batches):
        self.request_batches = request_batches

    def supports(self, b, q, rows):
        return True  # plan()'s capacity gate is the only one under test


def _entry():
    return D.K31SparseVerifyHostBinding.__new__(D.K31SparseVerifyHostBinding)


@pytest.mark.parametrize("caps", [(1, 2, 4), (1, 2, 4, 8)], ids=["c1", "mkb8"])
@pytest.mark.parametrize("knob", [False, True], ids=["off", "on"])
def test_replay_policy(monkeypatch, caps, knob):
    monkeypatch.setattr(D, "STOCK_GRAPH_ABOVE_CAPACITY", knob)
    entry, built = _entry(), _Built(caps)
    got = {b: entry.can_replay_graph(built, _batch(b)) for b in BATCHES}
    want = {b: knob or b <= max(caps) for b in BATCHES}
    assert got == want, f"caps {caps} knob {knob}: replay {got}, expected {want}"
    assert not any(entry.can_replay_graph(built, _batch(b, verify=False)) for b in BATCHES), "non-verify batch replays"


@pytest.mark.parametrize("caps", [(1, 2, 4), (1, 2, 4, 8)], ids=["c1", "mkb8"])
def test_graphs_above_capacity_hold_the_stock_forward(monkeypatch, caps):
    """The knob is sound only while plan() gives no megakernel step at any captured size above the capacity
    (checked with the knob on, where it matters)."""
    monkeypatch.setattr(D, "STOCK_GRAPH_ABOVE_CAPACITY", True)
    entry, built = _entry(), _Built(caps)
    for b in BATCHES:
        step = entry.plan(built, SimpleNamespace(shape=(8 * b,)), _batch(b))
        assert (step is not None) == (b <= max(caps)), f"caps {caps}: plan at {b} requests gave {step!r}"


@pytest.mark.parametrize("knob", [False, True], ids=["off", "on"])
def test_runner_asks_the_entry(monkeypatch, knob):
    """megakernel_can_run_graph (the decode graph runner's gate) follows the entry's replay policy."""
    monkeypatch.setattr(D, "STOCK_GRAPH_ABOVE_CAPACITY", knob)
    entry = _entry()
    model = SimpleNamespace()
    model.__dict__.update(
        _megakernel=_Built((1, 2, 4)), _megakernel_prepare=(None, SimpleNamespace(tensor_entry=entry))
    )
    got = {b: B.megakernel_can_run_graph(model, _batch(b)) for b in BATCHES}
    assert got == {b: knob or b <= 4 for b in BATCHES}, f"knob {knob}: runner gate {got}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
