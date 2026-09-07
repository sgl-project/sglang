# SPDX-License-Identifier: Apache-2.0
"""TeaCache's compute/skip decisions must reach the request's decision_trace
when record_decision_trace is set, so the compile-trajectory gate can compare
eager vs. compiled decision traces (see runtime.utils.compile_trajectory_gate
.TrajectoryGate.require_decision_trace_match)."""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)


class _ToyTeaCacheModel(TeaCacheMixin):
    prefix = "wan"

    def __init__(self):
        self._init_teacache_state()


def _make_batch(record_decision_trace: bool) -> SimpleNamespace:
    return SimpleNamespace(
        record_decision_trace=record_decision_trace, decision_trace=[]
    )


def test_decision_recorded_when_requested():
    model = _ToyTeaCacheModel()
    batch = _make_batch(record_decision_trace=True)

    with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
        should_calc = model._compute_teacache_decision(
            modulated_inp=torch.randn(4),
            is_boundary_step=True,
            coefficients=[1.0],
            teacache_thresh=0.5,
        )

    assert batch.decision_trace == [should_calc]


def test_decision_not_recorded_when_not_requested():
    model = _ToyTeaCacheModel()
    batch = _make_batch(record_decision_trace=False)

    with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
        model._compute_teacache_decision(
            modulated_inp=torch.randn(4),
            is_boundary_step=True,
            coefficients=[1.0],
            teacache_thresh=0.5,
        )

    assert batch.decision_trace == []


def test_decision_trace_accumulates_across_steps():
    model = _ToyTeaCacheModel()
    batch = _make_batch(record_decision_trace=True)

    for step in range(3):
        with set_forward_context(
            current_timestep=step, attn_metadata=None, forward_batch=batch
        ):
            model._compute_teacache_decision(
                modulated_inp=torch.randn(4),
                is_boundary_step=(step == 0),
                coefficients=[1.0],
                teacache_thresh=0.5,
            )

    assert len(batch.decision_trace) == 3
    assert batch.decision_trace[0] is True  # boundary step always computes
