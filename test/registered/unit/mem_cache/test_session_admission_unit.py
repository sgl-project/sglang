"""Unit tests for streaming-session pin-budget admission + growth cap.

Both deadlock causes are covered deterministically with a fake cache (no GPU):
  1. concurrent burst  -> admission opens over-budget sessions SHARED
  2. single-session runaway growth -> pin-budget flips it SHARED at slot save

The invariant under test: KV pinned by ISOLATED sessions stays <= budget, and it
is restored only by tagging SHARED (never by releasing a live slot).
"""

from types import SimpleNamespace

from sglang.srt.session.session_controller import Residency, SessionController
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeSlot:
    def __init__(self, kv_allocated_len: int):
        self.kv_allocated_len = kv_allocated_len
        self.is_holding_kv = True


class _FakeAllocator:
    def __init__(self, size: int):
        self.size = size


class _FakeCache:
    """Minimal stand-in for StreamingSession: a slots dict + a sized allocator,
    and accepts the controller back-reference."""

    def __init__(self, pool_size: int):
        self.slots = {}
        self.token_to_kv_pool_allocator = _FakeAllocator(pool_size)


def _controller(pool_size: int, fraction: float) -> SessionController:
    ctrl = SessionController(_FakeCache(pool_size))
    ctrl.max_session_tokens = int(fraction * pool_size)  # deterministic, env-free
    return ctrl


def _open(ctrl: SessionController, sid: str, capacity: int = 64):
    return ctrl.open(
        SimpleNamespace(
            session_id=sid,
            capacity_of_str_len=capacity,
            streaming=True,
            timeout=300.0,
        )
    )


def test_admission_caps_concurrent_pinned_sessions():
    # budget 64; est = capacity//4 = 16 -> exactly 4 sessions fit, rest SHARED.
    ctrl = _controller(pool_size=128, fraction=0.5)
    for i in range(6):
        _open(ctrl, f"s{i}", capacity=64)

    residencies = [ctrl.sessions[f"s{i}"].residency for i in range(6)]
    assert residencies == [Residency.ISOLATED] * 4 + [Residency.SHARED] * 2
    assert ctrl.pinned_tokens() <= ctrl.max_session_tokens


def test_growth_cap_flips_runaway_session_to_shared():
    # A single pinned session grows across turns; when its slot crosses budget
    # the pin-budget check flips it to SHARED -- no release.
    ctrl = _controller(pool_size=200, fraction=0.5)  # budget 100
    _open(ctrl, "big", capacity=64)
    assert ctrl.sessions["big"].residency == Residency.ISOLATED

    # turn 1: slot at 40 (<= budget) -> stays pinned
    ctrl.tree_cache.slots["big"] = _FakeSlot(40)
    ctrl.on_slot_saved("big")
    assert ctrl.sessions["big"].residency == Residency.ISOLATED

    # turn 2: slot grew to 140 (> budget 100) -> flips SHARED
    ctrl.tree_cache.slots["big"].kv_allocated_len = 140
    ctrl.on_slot_saved("big")
    assert ctrl.sessions["big"].residency == Residency.SHARED
    # Once SHARED it is no longer counted against the pinned budget.
    assert ctrl.pinned_tokens() == 0


def test_growth_cap_keeps_one_session_when_others_idle():
    # Two pinned sessions; together they exceed budget after growth. The one that
    # just grew is flipped, leaving the pinned tier within budget.
    ctrl = _controller(pool_size=200, fraction=0.5)  # budget 100
    _open(ctrl, "a", capacity=64)
    _open(ctrl, "b", capacity=64)
    ctrl.tree_cache.slots["a"] = _FakeSlot(60)
    ctrl.tree_cache.slots["b"] = _FakeSlot(60)  # total 120 > 100
    ctrl.on_slot_saved("b")  # b just finished a turn
    assert ctrl.sessions["b"].residency == Residency.SHARED
    assert ctrl.sessions["a"].residency == Residency.ISOLATED
    assert ctrl.pinned_tokens() == 60  # only 'a' still pinned


def test_no_budget_is_legacy_unbounded():
    ctrl = SessionController(_FakeCache(pool_size=128))
    ctrl.max_session_tokens = None
    for i in range(10):
        _open(ctrl, f"s{i}", capacity=64)
    assert all(s.residency == Residency.ISOLATED for s in ctrl.sessions.values())
