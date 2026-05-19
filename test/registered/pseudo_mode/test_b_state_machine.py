"""Type-b Hypothesis state machine for the pseudo-mode scheduler contract.

Drives a :class:`PseudoEngine` through randomly-ordered ``admit /
step / preempt / abort`` sequences and asserts a small set of
scheduler-contract invariants after every transition. The canary
machinery does the heavy lifting on the kernel side: this file's job
is just to generate a sequence space wide enough to exercise edge
cases the type-a scripted scenarios do not enumerate by hand.

Invariants checked here:

1. ``no_canary_violations`` — the primary witness. Any kernel-side
   slot / input mismatch surfaces here.
2. ``allocator_conservation`` — ``free + used == total`` on the KV
   pool. Catches accounting bugs where a release / retract path
   forgets to refund slots, or double-counts a refund.
3. ``active_reqs_consistent`` — every rid the scheduler currently
   reports as active was a rid we admitted and have not yet aborted
   or seen completed. Catches leaked reqs from internal data
   structures or rid collisions.

Deferred to v2 (documented gaps):

* ``block_table_in_held`` (spec §6.b 3rd invariant) requires reading
  per-req ``block_table`` against the allocator's ``held`` set. Both
  live inside the scheduler subprocess; the harness exposes only an
  aggregate ``free / used / total`` count. Add a ``_pseudo_block_tables``
  RPC + ``allocator_held_set`` accessor to enable this.
* ``positions_monotonic`` (spec §6.b 4th invariant) would assert per-req
  position counters strictly increase across decode steps. The current
  IPC reports ``output_len`` per active req but not the per-token
  positions actually fed into the model — the canary head kernel
  already catches the mismatch case end-to-end, so this stays a v2
  oracle-introspection feature, not a v1 state-machine invariant.

Hypothesis is not currently a sglang test dependency. The whole module
no-ops via ``unittest.skip`` if the import fails, so this test does not
gate CI on a new pip pin.
"""

from __future__ import annotations

import logging
import unittest
from test.registered.pseudo_mode._fake_prompt import fake_prompt
from test.registered.pseudo_mode._pseudo_engine import PseudoEngine, PseudoReqHandle
from test.registered.pseudo_mode._test_utils import (
    PSEUDO_MODE_MODEL,
    requires_cuda,
)

from sglang.test.ci.ci_register import register_cuda_ci
from typing import List, Set

logger = logging.getLogger(__name__)

try:
    from hypothesis import HealthCheck, settings
    from hypothesis import strategies as st
    from hypothesis.stateful import (
        RuleBasedStateMachine,
        invariant,
        precondition,
        rule,
    )

    _HYPOTHESIS_AVAILABLE = True
except ImportError as exc:
    logger.warning(
        "hypothesis is not installed; SchedulerContractMachine will be skipped: %s",
        exc,
    )
    _HYPOTHESIS_AVAILABLE = False

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


# Bound per-example operation count well below the Hypothesis default
# (~100) so each example finishes in seconds and CI stays under the
# stage-a wall-clock budget. ``max_examples`` is also lowered: the
# space we explore is small enough that 20 examples cover the
# interesting branches without trading too much shrinking quality.
_MAX_EXAMPLES: int = 20
_STATEFUL_STEPS: int = 30
_PROMPT_LEN_MIN: int = 8
_PROMPT_LEN_MAX: int = 64
_MAX_NEW_MIN: int = 1
_MAX_NEW_MAX: int = 8


@requires_cuda
@unittest.skipUnless(_HYPOTHESIS_AVAILABLE, "hypothesis is not installed")
class TestSchedulerContractMachine(unittest.TestCase):
    """Wraps the Hypothesis state machine into the unittest discovery path."""

    def test_scheduler_contract(self) -> None:
        """Run the state machine. Hypothesis handles example generation."""
        SchedulerContractMachine.TestCase.settings = settings(
            max_examples=_MAX_EXAMPLES,
            stateful_step_count=_STATEFUL_STEPS,
            deadline=None,
            suppress_health_check=[
                HealthCheck.too_slow,
                HealthCheck.data_too_large,
            ],
        )
        SchedulerContractMachine.TestCase().runTest()


if _HYPOTHESIS_AVAILABLE:

    class SchedulerContractMachine(RuleBasedStateMachine):
        """Random ``admit / step / preempt / abort`` driver + invariants.

        One engine is launched per Hypothesis example (i.e. per
        ``__init__``) and torn down in :meth:`teardown`. Reusing a single
        engine across examples is tempting for speed, but Hypothesis
        constructs a fresh instance per example so we honour that
        boundary — the only cost is the launch budget per example.
        """

        def __init__(self) -> None:
            super().__init__()
            self._engine: PseudoEngine = PseudoEngine.launch(
                model=PSEUDO_MODE_MODEL,
                num_hidden_layers=1,
                cuda_graph=False,
                radix_cache=False,
            )
            self._handles: List[PseudoReqHandle] = []
            self._admitted_rids: Set[str] = set()
            self._retired_rids: Set[str] = set()

        # --- rules -----------------------------------------------------

        @rule(
            prompt_len=st.integers(_PROMPT_LEN_MIN, _PROMPT_LEN_MAX),
            max_new=st.integers(_MAX_NEW_MIN, _MAX_NEW_MAX),
        )
        def admit(self, prompt_len: int, max_new: int) -> None:
            prompt = fake_prompt(prompt_len)
            handle = self._engine.admit(prompt=prompt, max_new_tokens=max_new)
            self._handles.append(handle)
            self._admitted_rids.add(handle.rid)

        @rule()
        def step(self) -> None:
            self._engine.step()
            self._prune_finished_handles()

        @precondition(lambda self: bool(self._handles))
        @rule(data=st.data())
        def preempt(self, data: st.DataObject) -> None:
            handle = data.draw(st.sampled_from(self._handles))
            self._engine.force_preempt(handle)
            # ``force_preempt`` drops the req from the running batch but
            # does not remove it from sglang's waiting queue — it can be
            # rescheduled, so the handle stays tracked.

        @precondition(lambda self: bool(self._handles))
        @rule(data=st.data())
        def abort(self, data: st.DataObject) -> None:
            handle = data.draw(st.sampled_from(self._handles))
            self._engine.abort(handle)
            self._retire_handle(handle)

        # --- invariants ------------------------------------------------

        @invariant()
        def no_canary_violations(self) -> None:
            self._engine.assert_no_canary_violations()

        @invariant()
        def allocator_conservation(self) -> None:
            stats = self._engine.allocator_stats()
            if stats.free + stats.used != stats.total:
                raise AssertionError(
                    f"Allocator conservation violated: "
                    f"free={stats.free} + used={stats.used} "
                    f"!= total={stats.total}"
                )

        @invariant()
        def active_reqs_consistent(self) -> None:
            for entry in self._engine.active_reqs():
                if entry.rid not in self._admitted_rids:
                    raise AssertionError(
                        f"Scheduler reports unknown active rid={entry.rid!r} "
                        f"(never admitted by this machine)"
                    )
                if entry.rid in self._retired_rids:
                    raise AssertionError(
                        f"Scheduler reports rid={entry.rid!r} as active "
                        f"after we aborted it"
                    )

        # NOTE: block_table_in_held + positions_monotonic invariants are
        # deferred to v2 (see module docstring); the harness IPC does
        # not yet expose the allocator's held-set or per-req positions.

        # --- lifecycle -------------------------------------------------

        def teardown(self) -> None:
            try:
                self._engine.shutdown()
            except Exception:
                logger.exception("PseudoEngine shutdown failed during teardown")

        # --- internals -------------------------------------------------

        def _prune_finished_handles(self) -> None:
            """Drop handles whose rid no longer appears in active_reqs.

            sglang removes finished reqs from the running + waiting sets
            once they hit max_new_tokens or EOS. We mirror that here so
            ``preempt`` / ``abort`` only sample live handles.
            """
            live_rids = {entry.rid for entry in self._engine.active_reqs()}
            still_active: List[PseudoReqHandle] = []
            for handle in self._handles:
                if handle.rid in live_rids:
                    still_active.append(handle)
                else:
                    self._retired_rids.add(handle.rid)
            self._handles = still_active

        def _retire_handle(self, handle: PseudoReqHandle) -> None:
            self._retired_rids.add(handle.rid)
            self._handles = [h for h in self._handles if h.rid != handle.rid]


if __name__ == "__main__":
    unittest.main(verbosity=3)
