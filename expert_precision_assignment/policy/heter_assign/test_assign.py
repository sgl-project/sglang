"""Pytest correctness tests for the ranking-policy assignment logic.

Verifies the two invariants that should hold regardless of model shape
or VRAM details:

1. K monotonicity in mc — as max_concurrency grows, KV reservation
   crowds out BF16 expert slots and budget_k shrinks, so the chosen
   heter set is non-expanding for all three policies.

2. Hessian preservation across mc:
     - hessian_importance: heter(mc_high) ⊆ heter(mc_low). Losing
       budget only drops the lowest-ranked Hessian experts.
     - hybrid: the top fo_cap by Hessian are always chosen whenever
       the budget admits at least fo_cap experts.
     - activation_frequency: included as a sanity check (its own
       ranking is also stable under K shrinkage).

Synthetic scores + a synthetic mc→budget_k map keep the tests fast,
deterministic, and free of HF / GPU dependencies. The vram-budget
math itself is covered by ``test_configs.py``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from assign_experts import _assign, _assign_hybrid  # noqa: E402


NUM_LAYERS = 8
NUM_EXPERTS = 64
TOTAL = NUM_LAYERS * NUM_EXPERTS  # 512


def _hessian_scores(seed: int = 42):
    """Deterministic synthetic Hessian-like scores in roughly [-0.2, 0.4]."""
    out = {}
    for L in range(NUM_LAYERS):
        for E in range(NUM_EXPERTS):
            out[(L, E)] = math.sin(seed + L * 17.3 + E * 5.7) * 0.3 + 0.1
    return out


def _token_counts(seed: int = 99):
    """Deterministic synthetic per-(L, E) routed-token counts in [0, 999]."""
    out = {}
    for L in range(NUM_LAYERS):
        for E in range(NUM_EXPERTS):
            out[(L, E)] = (seed * 31 + L * 113 + E * 7) % 1000
    return out


# Larger mc → bigger KV reserve → smaller bf16_budget → smaller K. The
# concrete ratios mimic the observed Qwen3-30B-A3B sweep on an 80 GB GPU
# (mc=8 admits everything, mc=128 admits ~half, mc=256 is infeasible)
# rescaled to TOTAL=512.
MC_TO_BUDGET_K = {
    8: TOTAL,
    16: int(TOTAL * 0.85),
    32: int(TOTAL * 0.65),
    64: int(TOTAL * 0.45),
    128: int(TOTAL * 0.25),
    256: 0,
}
MC_LIST = sorted(MC_TO_BUDGET_K.keys())


@pytest.fixture(scope="module")
def signals():
    return _hessian_scores(), _token_counts()


@pytest.fixture(scope="module")
def fo_cap(signals):
    """Synthetic |fo|-mean cap: experts with positive Hessian score."""
    hess, _ = signals
    return sum(1 for v in hess.values() if v > 0.0)


# ---------------------------------------------------------------------------
# 1. K decreases as mc grows
# ---------------------------------------------------------------------------

def test_hessian_importance_K_non_increasing_in_mc(signals, fo_cap):
    Ks = [min(MC_TO_BUDGET_K[mc], fo_cap) for mc in MC_LIST]
    assert Ks == sorted(Ks, reverse=True), f"K not monotone: {Ks}"


def test_activation_frequency_K_non_increasing_in_mc():
    Ks = [MC_TO_BUDGET_K[mc] for mc in MC_LIST]
    assert Ks == sorted(Ks, reverse=True), f"K not monotone: {Ks}"


def test_hybrid_K_non_increasing_in_mc(signals, fo_cap):
    Ks = []
    for mc in MC_LIST:
        budget_k = MC_TO_BUDGET_K[mc]
        k_hess = min(budget_k, fo_cap)
        k_fill = max(0, budget_k - k_hess)
        Ks.append(k_hess + k_fill)
    assert Ks == sorted(Ks, reverse=True), f"K not monotone: {Ks}"


# ---------------------------------------------------------------------------
# 2. Hessian-selected experts are preserved across mc
# ---------------------------------------------------------------------------

def test_hessian_importance_heter_set_shrinks_monotonically(signals, fo_cap):
    """heter(mc_high) ⊆ heter(mc_low) — only the lowest-ranked drop off."""
    hess, _ = signals
    sets = []
    for mc in MC_LIST:
        k = min(MC_TO_BUDGET_K[mc], fo_cap)
        heter, _ = _assign(hess, k)
        sets.append(set(heter))
    for i in range(1, len(sets)):
        assert sets[i].issubset(sets[i - 1]), (
            f"mc={MC_LIST[i]} heter set not a subset of mc={MC_LIST[i - 1]}"
        )


def test_activation_frequency_heter_set_shrinks_monotonically(signals):
    """Token-count ranking has the same nested-subset structure."""
    _, tokens = signals
    tc = {k: float(v) for k, v in tokens.items()}
    sets = []
    for mc in MC_LIST:
        heter, _ = _assign(tc, MC_TO_BUDGET_K[mc])
        sets.append(set(heter))
    for i in range(1, len(sets)):
        assert sets[i].issubset(sets[i - 1])


def test_hybrid_keeps_top_fo_cap_hessian_experts(signals, fo_cap):
    """Whenever budget_k ≥ fo_cap, the top fo_cap by Hessian are all in heter."""
    hess, tokens = signals
    top_hess = set(_assign(hess, fo_cap)[0])

    for mc in MC_LIST:
        budget_k = MC_TO_BUDGET_K[mc]
        k_hess = min(budget_k, fo_cap)
        k_fill = max(0, budget_k - k_hess)
        heter, _, _ = _assign_hybrid(hess, tokens, k_hess, k_fill)
        if k_hess == fo_cap:
            assert top_hess.issubset(set(heter)), (
                f"mc={mc}: top-fo_cap Hessian set not preserved in hybrid"
            )


def test_hybrid_hessian_segment_matches_pure_hessian(signals, fo_cap):
    """For the Hessian portion of hybrid, the chosen pairs == top-k_hess
    by Hessian — i.e., gap-fill never displaces a Hessian-ranked expert."""
    hess, tokens = signals
    for mc in MC_LIST:
        budget_k = MC_TO_BUDGET_K[mc]
        k_hess = min(budget_k, fo_cap)
        k_fill = max(0, budget_k - k_hess)
        heter, _, kind = _assign_hybrid(hess, tokens, k_hess, k_fill)
        hess_chosen = {p for p in heter if kind[p] == "hessian"}
        expected = set(_assign(hess, k_hess)[0])
        assert hess_chosen == expected, f"mc={mc}: hybrid hessian seg drift"


# ---------------------------------------------------------------------------
# Sanity invariants
# ---------------------------------------------------------------------------

def test_assign_partitions_pairs(signals):
    """heter and int4_only must be disjoint and cover every (L, E)."""
    hess, _ = signals
    heter, int4 = _assign(hess, K=200)
    assert len(heter) == 200
    assert len(int4) == TOTAL - 200
    assert set(heter).isdisjoint(set(int4))
    assert set(heter) | set(int4) == set(hess.keys())


def test_assign_hybrid_partitions_pairs(signals):
    hess, tokens = signals
    heter, int4, kind = _assign_hybrid(hess, tokens, k_hess=100, k_fill=80)
    assert len(heter) == 180
    assert len(int4) == TOTAL - 180
    assert set(heter).isdisjoint(set(int4))
    assert set(heter) | set(int4) == set(hess.keys())
    assert set(kind.keys()) == set(heter)
    assert sum(1 for v in kind.values() if v == "hessian") == 100
    assert sum(1 for v in kind.values() if v == "tokencount") == 80


def test_assign_hybrid_degenerate_no_fill(signals, fo_cap):
    """k_fill=0 → hybrid degenerates to pure top-k_hess by Hessian."""
    hess, tokens = signals
    heter_h, _, kind = _assign_hybrid(hess, tokens, k_hess=fo_cap, k_fill=0)
    heter_p, _ = _assign(hess, fo_cap)
    assert heter_h == heter_p
    assert all(v == "hessian" for v in kind.values())


def test_assign_hybrid_degenerate_no_hessian(signals):
    """k_hess=0 → hybrid degenerates to pure top-k_fill by token_count."""
    hess, tokens = signals
    tc = {k: float(v) for k, v in tokens.items()}
    heter_h, _, kind = _assign_hybrid(hess, tokens, k_hess=0, k_fill=120)
    heter_p, _ = _assign(tc, 120)
    assert heter_h == heter_p
    assert all(v == "tokencount" for v in kind.values())
