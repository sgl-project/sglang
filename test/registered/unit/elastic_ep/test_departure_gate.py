"""Only the mlp_sync gather clears DRAIN: a timer exit would gate a survivor's batch
step while a held retiree still posts mlp_sync."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.elastic_ep.elastic_ep as eep
from sglang.srt.elastic_ep.elastic_ep import (
    departure_announce,
    departure_cleared,
    departure_observe,
    departure_pending,
    departure_reset,
)
from sglang.test.test_utils import CustomTestCase


class TestDepartureGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        departure_reset()
        self.addCleanup(departure_reset)

    def test_announce_stops_holding_the_cohort(self):
        self.assertTrue(departure_pending())
        departure_announce()
        departure_announce()
        self.assertFalse(departure_pending())

    def test_time_alone_never_clears(self):
        departure_announce()
        later = SimpleNamespace(monotonic=lambda: time.monotonic() + 3600.0)
        with mock.patch.object(eep, "time", later):
            self.assertFalse(departure_cleared())

    def test_the_gather_verdict_clears_and_reset_rearms(self):
        departure_announce()
        departure_observe(True)
        self.assertTrue(departure_cleared())
        departure_observe(False)
        self.assertFalse(departure_cleared())
        departure_reset()
        self.assertTrue(departure_pending())
        self.assertFalse(departure_cleared())


if __name__ == "__main__":
    unittest.main()
