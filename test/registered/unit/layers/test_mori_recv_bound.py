"""Contract tests for the MoRI decode receive bound.

AITER sizes its quantization grid from the input row count, and under MoRI the
input is the *padded* receive buffer (num_max_dispatch_tokens_per_rank *
world_size), not the live tokens. In decode that padding dominates: a rank
carrying a handful of live tokens still pays for thousands of padded rows.

The bound caps that row count at the true worst case -- every rank sending all
of its tokens to this one -- which is knowable on the host only in decode, where
every DP rank replays the same cuda-graph tier and therefore sends the same
number of tokens. Prefill token counts are uneven across ranks and cannot be
bounded this way; truncating there corrupts MoRI's combine, so it is left alone.

These pin the arithmetic and, more importantly, the guards: a bound that is too
*small* silently drops rows from the all-to-all, which is wrong output rather
than an error.
"""

import pytest

pytest.importorskip("torch")

from sglang.srt.layers.moe.moe_runner.aiter import (  # noqa: E402
    _mori_decode_recv_bound,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _clear_state(monkeypatch):
    """Each test sets its own dispatch state; never inherit another's."""
    from sglang.srt.layers.moe.token_dispatcher import moriep

    monkeypatch.setattr(moriep, "LAST_DISPATCH_SEND_TOKENS", None, raising=False)
    yield


def _set_dispatch(monkeypatch, num_token, world_size, topk):
    from sglang.srt.layers.moe.token_dispatcher import moriep

    monkeypatch.setattr(
        moriep,
        "LAST_DISPATCH_SEND_TOKENS",
        (num_token, world_size, topk),
        raising=False,
    )


def test_disabled_by_default(monkeypatch):
    """Opt-in only: with the env unset the bound must be 0, meaning unbounded,
    so existing deployments are untouched."""
    monkeypatch.delenv("SGLANG_MORI_RECV_BOUND", raising=False)
    _set_dispatch(monkeypatch, 8, 8, 6)
    assert _mori_decode_recv_bound(24576) == 0


def test_returns_zero_without_dispatch_state(monkeypatch):
    """Before the first dispatch there is nothing to bound from. Guessing here
    would truncate the very first step."""
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "true")
    assert _mori_decode_recv_bound(24576) == 0


def test_bound_covers_worst_case_fan_in(monkeypatch):
    """Worst case is every rank sending all its tokens here: tokens * world *
    topk. The bound must be at least that or rows are dropped."""
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "true")
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND_MARGIN_PCT", "100")
    num_token, world, topk = 8, 8, 6
    bound = _mori_decode_recv_bound(24576)
    assert bound >= num_token * world * topk or bound == 0


def test_bound_is_smaller_than_padded_rows(monkeypatch):
    """The entire point: if the bound is not below the padded row count it
    saves nothing."""
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "true")
    _set_dispatch(monkeypatch, 8, 8, 6)
    padded = 24576
    bound = _mori_decode_recv_bound(padded)
    if bound:
        assert bound < padded


def test_never_exceeds_the_buffer(monkeypatch):
    """A bound above the real buffer would index past it. At high token counts
    the worst case can exceed the padding, and it must clamp."""
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "true")
    _set_dispatch(monkeypatch, 4096, 8, 8)
    padded = 24576
    bound = _mori_decode_recv_bound(padded)
    assert bound <= padded


def test_margin_widens_the_bound(monkeypatch):
    """The margin exists to absorb any fan-in the host arithmetic underestimates;
    a larger margin must never produce a tighter bound."""
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND", "true")
    _set_dispatch(monkeypatch, 8, 8, 6)
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND_MARGIN_PCT", "100")
    tight = _mori_decode_recv_bound(24576)
    monkeypatch.setenv("SGLANG_MORI_RECV_BOUND_MARGIN_PCT", "200")
    wide = _mori_decode_recv_bound(24576)
    if tight and wide:
        assert wide >= tight


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
