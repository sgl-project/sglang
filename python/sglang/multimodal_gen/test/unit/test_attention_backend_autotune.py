# SPDX-License-Identifier: Apache-2.0
"""The rules the autotuner has to obey: only switch on a clear, correct win.

Timing is stubbed here so the rules are what is under test, not the GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention import autotune
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

QUERY = torch.zeros(4, 4)
REFERENCE = torch.ones(2, 2)


class _Impl:
    def __init__(self, output=REFERENCE):
        self._output = output
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self._output


def _layer(incumbent):
    return SimpleNamespace(
        attn_impl=incumbent,
        backend=AttentionBackendEnum.TORCH_SDPA,
        head_size=128,
        dtype=torch.bfloat16,
        _attn_impl_ctor_kwargs={},
        _supported_attention_backends=set(),
    )


@pytest.fixture
def stub(monkeypatch):
    """Drive _choose with fixed candidates and fixed timings."""

    def install(candidates: list[tuple[str, _Impl, object]], timings: dict[int, float]):
        monkeypatch.setattr(autotune, "_candidates", lambda layer: candidates)
        monkeypatch.setattr(
            autotune, "_timed", lambda impl, args, kwargs: timings[id(impl)]
        )

    return install


def test_keeps_the_incumbent_without_a_clear_win(stub):
    incumbent, rival = _Impl(), _Impl()
    stub(
        [("rival", rival, AttentionBackendEnum.FA)],
        {id(incumbent): 10.0, id(rival): 9.9},  # 1%, under the margin
    )
    assert autotune._choose(_layer(incumbent), (QUERY,), {}) is None


def test_switches_when_a_candidate_wins_by_more_than_the_margin(stub):
    incumbent, rival = _Impl(), _Impl()
    stub(
        [("rival", rival, AttentionBackendEnum.FA)],
        {id(incumbent): 10.0, id(rival): 8.0},
    )
    chosen = autotune._choose(_layer(incumbent), (QUERY,), {})
    assert chosen is not None
    assert chosen[0] is rival
    assert chosen[1] is AttentionBackendEnum.FA


def test_a_faster_candidate_that_disagrees_is_rejected(stub):
    incumbent = _Impl()
    wrong = _Impl(output=REFERENCE * 5)
    stub(
        [("wrong", wrong, AttentionBackendEnum.FA)],
        {id(incumbent): 10.0, id(wrong): 1.0},
    )
    assert autotune._choose(_layer(incumbent), (QUERY,), {}) is None


def test_a_candidate_that_raises_is_skipped(stub, monkeypatch):
    incumbent = _Impl()
    broken = _Impl()

    def explode(*args, **kwargs):
        raise RuntimeError("unsupported here")

    broken.forward = explode
    stub([("broken", broken, AttentionBackendEnum.FA)], {id(incumbent): 10.0})
    assert autotune._choose(_layer(incumbent), (QUERY,), {}) is None


def test_small_calls_stay_on_the_default_and_leave_the_tuner_armed(monkeypatch):
    incumbent = _Impl()
    layer = _layer(incumbent)
    called = []
    monkeypatch.setattr(autotune, "_choose", lambda *a, **k: called.append(1))

    autotune.install(layer)
    small = torch.zeros(8, 8)
    assert small.numel() < autotune._MIN_TUNE_NUMEL
    incumbent.forward(small)

    assert called == [], "tuning must wait for a call worth measuring"
    assert layer.attn_impl is incumbent
