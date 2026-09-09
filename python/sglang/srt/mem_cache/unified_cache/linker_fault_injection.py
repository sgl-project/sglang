"""Fault injection for the external-cache linker load path.

Test-only, and a no-op unless ``SGLANG_TEST_LINKER_LOAD_FAILURE_PROB`` is set.
This mirrors ``_poll_with_failure_injection`` on the disaggregation path
(``srt/disaggregation/utils.py``), which is what
``test_disaggregation_basic.py::TestDisaggregationMooncakeFailure`` drives to
run gsm8k under injected KV-transfer failures.

Why inject here rather than stub the backend
--------------------------------------------
The failure is raised inside the same ``try`` a backend's own range-get failure
is raised in, so everything downstream of it runs exactly as it ships:
``LayerWiseLoadCounter.fail`` -> ``wait_until`` logging and returning instead of
raising -> the ``finally`` that publishes ``(rids, False)`` -> the MIN-reduce
across the attention group -> the scheduler abort -> the failed chain's detach
and free. The injected fault is the *input* to the code under test, not a
bypass of it. It stands in for the one thing a healthy backend cannot be asked
to do: return short.

Two lines are skipped versus a real short read -- the ``len(results) != ... or
not all(results)`` comparison that decides to raise. Those are covered by the
linker unit tests, and keeping that comparison backend-side is what lets this
hook stay free of any backend's result convention (Mooncake reports transferred
byte counts, UMBP reports per-key booleans).

Semantics
---------
The probability is *per load batch*, not per range get. A batch covers every
layer and every pool, so a per-call roll would make any usable probability fail
essentially every batch on a 61-layer model. One roll per batch keeps the knob
readable as "the fraction of load-backs that fail", independent of layer count,
model, and layer-group width.

A batch that loses its roll fails on its first range get, deterministically.
Spreading the failure over a random layer was tried and removed: a batch is as
narrow as one range get on some pool configurations, so any countdown long
enough to be interesting is longer than some batches, and the failure is then
silently lost -- which shows up as an injection rate well under the configured
one, i.e. a test that quietly stops testing. Which layer fails is also the less
interesting variable, since the request is mid-forward either way; what the
rank-scoping knob below varies is the axis that actually has no other
observable.
"""

from __future__ import annotations

import random
from typing import Callable

from sglang.srt.environ import envs

__all__ = [
    "InjectedLinkerLoadFailure",
    "arm_load_failure_injection",
]


class InjectedLinkerLoadFailure(RuntimeError):
    """Stands in for a backend range-get failure, under test only.

    Subclasses ``RuntimeError`` so every ``except`` on the load path treats it
    exactly as it treats the real short-read error.
    """


def _rank_is_selected(tp_rank: int) -> bool:
    """Whether this TP rank honours the injection.

    Unset (or ``all``) selects every rank. A comma-separated list selects only
    those, which is how the rank-local case is reached: a load failure is
    decided per rank, so the MIN-reduce across the attention group -- one rank
    failing has to abort the request on all of them -- has no observable at all
    unless the ranks can be made to disagree.
    """
    selected = envs.SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS.get()
    if selected is None or selected.strip().lower() in ("", "all"):
        return True
    return tp_rank in {int(part) for part in selected.split(",") if part.strip()}


def _never_fails(pool: str, where: str) -> None:
    """The disabled check. Prod takes this path on every range get."""


def arm_load_failure_injection(
    tp_rank: int,
    *,
    rng: random.Random | None = None,
) -> Callable[[str, str], None]:
    """Roll once for one load batch, and return that batch's per-range-get check.

    Call at the top of a layer-wise load batch; call the result before each
    range get, passing the pool name and a short description of where in the
    batch it is (``"layer=7"``, ``"layers=0..3"``). The check raises
    :class:`InjectedLinkerLoadFailure` on the batch's first range get, and is a
    plain no-op whenever injection is off -- which is always, outside tests.

    Because the roll happens here rather than per range get, the observed
    fraction of failed load batches equals ``prob`` regardless of how many
    layers, pools or chunks a batch turns out to span.
    """
    probability = envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.get()
    if probability <= 0 or not _rank_is_selected(tp_rank):
        return _never_fails

    chooser = rng if rng is not None else random
    if chooser.random() >= probability:
        return _never_fails

    def check(pool: str, where: str) -> None:
        raise InjectedLinkerLoadFailure(
            f"Injected external-linker load failure for pool={pool}, {where}, "
            f"tp_rank={tp_rank} "
            f"(SGLANG_TEST_LINKER_LOAD_FAILURE_PROB={probability})"
        )

    return check
