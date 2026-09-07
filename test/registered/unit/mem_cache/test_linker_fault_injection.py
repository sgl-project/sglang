"""Unit tests for external-linker load-failure injection.

The injector exists so the load-failure contract can be tested against a
healthy backend, which cannot otherwise be asked to return short. These tests
pin the two properties an injected-fault test silently depends on:

* it is off unless asked, so nothing changes in production;
* when it is on, the observed failure rate is the configured one -- a fault
  injector that quietly stops injecting turns every downstream assertion
  vacuous, and passes while doing it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import random
import unittest

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    InjectedLinkerLoadFailure,
    arm_load_failure_injection,
)


def _run_batch(check, range_gets: int) -> bool:
    """Drive one load batch's worth of range gets. True if it failed."""
    try:
        for layer in range(range_gets):
            check("mha", f"layer={layer}")
        return False
    except InjectedLinkerLoadFailure:
        return True


def _failure_rate(probability, range_gets, trials=4000, seed=7):
    rng = random.Random(seed)
    with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(probability):
        failed = sum(
            _run_batch(arm_load_failure_injection(0, rng=rng), range_gets)
            for _ in range(trials)
        )
    return failed / trials


class TestLinkerFaultInjectionIsOffByDefault(CustomTestCase):
    def test_no_failure_without_the_env_var(self):
        check = arm_load_failure_injection(0)
        for layer in range(500):
            check("mha", f"layer={layer}")

    def test_zero_probability_is_the_same_as_unset(self):
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(0.0):
            self.assertFalse(_run_batch(arm_load_failure_injection(0), 500))


class TestLinkerFaultInjectionFires(CustomTestCase):
    def test_probability_one_fails_every_batch(self):
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
            for _ in range(50):
                self.assertTrue(_run_batch(arm_load_failure_injection(0), 61))

    def test_injected_error_is_a_runtime_error(self):
        """The load path's ``except`` clauses must not tell the two apart."""
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
            check = arm_load_failure_injection(0)
            with self.assertRaises(RuntimeError):
                check("mha", "layer=0")

    def test_message_names_the_pool_the_position_and_the_rank(self):
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
            check = arm_load_failure_injection(3)
            with self.assertRaises(InjectedLinkerLoadFailure) as caught:
                check("swa", "layers=0..3")
        message = str(caught.exception)
        self.assertIn("pool=swa", message)
        self.assertIn("layers=0..3", message)
        self.assertIn("tp_rank=3", message)

    def test_a_batch_fails_at_most_once(self):
        """The first range get raises, so the batch cannot fail twice."""
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
            check = arm_load_failure_injection(0)
            with self.assertRaises(InjectedLinkerLoadFailure):
                check("mha", "layer=0")
            with self.assertRaises(InjectedLinkerLoadFailure):
                check("mha", "layer=1")


class TestLinkerFaultInjectionRateIsPerBatch(CustomTestCase):
    """The knob has to mean "this fraction of load-backs fail".

    It is read once per batch rather than per range get. A per-range-get roll
    would scale with layer count, pool count and layer-group width, so the same
    number would mean something different on every model -- and on a 61-layer
    model any usable value would fail essentially every batch.
    """

    def test_rate_matches_the_setting(self):
        self.assertAlmostEqual(_failure_rate(0.25, range_gets=61), 0.25, delta=0.03)

    def test_rate_does_not_move_with_batch_width(self):
        # Width 1 is real: a pool config can put a single range get in a batch.
        rates = {width: _failure_rate(0.25, width) for width in (1, 2, 8, 61, 244)}
        self.assertLess(
            max(rates.values()) - min(rates.values()),
            0.02,
            f"injection rate varies with batch width: {rates}",
        )


class TestLinkerFaultInjectionRankScope(CustomTestCase):
    """Rank scoping is what gives the MIN-reduce an observable.

    A load failure is decided per rank. With every rank failing, a group-wide
    abort proves nothing -- each rank would have aborted on its own. Only an
    asymmetric case shows one rank's failure aborting the request on all of
    them.
    """

    def _fails_on(self, ranks_setting, tp_rank):
        with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
            with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS.override(ranks_setting):
                return _run_batch(arm_load_failure_injection(tp_rank), 61)

    def test_a_rank_list_selects_only_those_ranks(self):
        selected = {rank: self._fails_on("0,3", rank) for rank in (0, 1, 3, 5)}
        self.assertEqual(selected, {0: True, 1: False, 3: True, 5: False})

    def test_unset_and_all_select_every_rank(self):
        for setting in (None, "", "all", "ALL", " all "):
            with self.subTest(setting=setting):
                self.assertTrue(self._fails_on(setting, 7))


if __name__ == "__main__":
    unittest.main()
