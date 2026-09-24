"""Lifecycle tests for the shared pre-allocation bootstrap rendezvous."""

import itertools
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeferredBootstrap(unittest.TestCase):
    def setUp(self):
        self.table = DeferredBootstrap(timeout=10)
        self.owner = object()
        self.endpoint = ("decode", 1234)

    def test_either_arrival_order_and_no_completion_before_source(self):
        for order in (
            ("open", "register", "complete"),
            ("register", "open", "complete"),
            ("open", "complete", "register"),
        ):
            with self.subTest(order=order):
                table = DeferredBootstrap(10)
                result = None
                for action in order:
                    if action == "open":
                        state = table.open(1, self.owner)
                        self.assertFalse(state.failed)
                    elif action == "register":
                        result = table.register(1, self.endpoint)
                    else:
                        result = table.complete(1, self.owner)
                self.assertEqual(result, (self.endpoint, False))

    def test_cancel_at_every_point_prevents_later_ready(self):
        for order in itertools.permutations(("open", "register", "cancel")):
            with self.subTest(order=order):
                table = DeferredBootstrap(10)
                notifications = []
                for action in order:
                    if action == "open":
                        table.open(1, self.owner)
                    elif action == "register":
                        notifications.append(table.register(1, self.endpoint))
                    else:
                        notifications.append(table.fail(1))
                self.assertTrue(table.rooms[1].failed)
                self.assertIsNone(table.complete(1, self.owner))
                self.assertIn((self.endpoint, True), notifications)

    def test_duplicate_source_cannot_replace_owner(self):
        self.table.open(1, self.owner)
        other = object()
        self.assertIsNone(self.table.open(1, other))
        self.assertIsNone(self.table.complete(1, other))
        self.assertIsNone(self.table.close(1, other))
        self.assertIs(self.table.rooms[1].owner, self.owner)

    def test_endpoint_registration_is_idempotent_not_replaceable(self):
        self.table.open(1, self.owner)
        self.table.complete(1, self.owner)
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, False))
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, False))
        other = ("other", 9999)
        self.assertEqual(self.table.register(1, other), (other, True))
        self.assertEqual(self.table.rooms[1].endpoint, self.endpoint)

    def test_clear_keeps_failure_for_late_receiver(self):
        self.table.open(1, self.owner)
        self.table.close(1, self.owner)
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, True))
        self.assertIsNone(self.table.complete(1, self.owner))

    def test_expire_unmatched_and_terminal_rooms_not_active_sources(self):
        with patch(
            "sglang.srt.disaggregation.common.bootstrap.time.monotonic", return_value=0
        ):
            self.table.open(1, self.owner)
            self.table.register(2, self.endpoint)
            self.table.fail(3)
            self.table.open(4, object())
            owner = self.table.rooms[4].owner
            self.table.close(4, owner)
        with patch(
            "sglang.srt.disaggregation.common.bootstrap.time.monotonic", return_value=11
        ):
            self.table.register(5, self.endpoint)
        self.assertEqual(set(self.table.rooms), {1, 5})
        self.assertIs(self.table.rooms[1].owner, self.owner)

    def test_cancelled_source_owns_cleanup(self):
        self.table.fail(1)
        state = self.table.open(1, self.owner)
        self.assertTrue(state.failed)
        self.assertIs(state.owner, self.owner)
        self.table.close(1, self.owner)
        self.assertIsNone(state.owner)

    def test_abort_then_clear_does_not_send_duplicate_failures(self):
        self.table.open(1, self.owner)
        self.table.register(1, self.endpoint)
        self.assertEqual(self.table.fail(1), (self.endpoint, True))
        self.assertIsNone(self.table.fail(1))
        self.assertIsNone(self.table.close(1, self.owner))
        # A later endpoint must still learn that the source is gone.
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, True))

    def test_no_lock_escapes_any_operation(self):
        for operation in (
            lambda: self.table.open(1, self.owner),
            lambda: self.table.register(1, self.endpoint),
            lambda: self.table.complete(1, self.owner),
            lambda: self.table.fail(1),
            lambda: self.table.close(1, self.owner),
        ):
            operation()
            self.assertTrue(self.table.lock.acquire(blocking=False))
            self.table.lock.release()


if __name__ == "__main__":
    unittest.main()
