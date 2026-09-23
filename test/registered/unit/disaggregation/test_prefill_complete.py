"""Readiness ordering, failure and ownership tests without inference."""

import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from sglang.srt.disaggregation.prefill_complete import HEADER, PrefillCompleteManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillComplete(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.wire = deque()
        self.cancelled = []
        self.prefill = PrefillCompleteManager(
            send=lambda endpoint, frames: self.wire.append((endpoint, frames)),
            on_cancel=self.cancel_source,
        )
        self.decode = PrefillCompleteManager(
            send=lambda endpoint, frames: self.wire.append((endpoint, frames)),
            on_cancel=lambda room: self.fail("decode received source cancellation"),
        )
        self.endpoints = {("prefill", 1): self.prefill, ("decode", 2): self.decode}

    def cancel_source(self, room):
        self.cancelled.append(room)
        self.prefill.fail_source(room=room, reason="cancelled")

    def flush(self):
        while self.wire:
            endpoint, frames = self.wire.popleft()
            self.assertTrue(self.endpoints[endpoint].handle_message(frames))

    def receiver(self, room=1):
        return self.decode.add_destination(
            room=room, endpoint=("prefill", 1), timeout=5
        )

    def poll(self, receiver, room=1):
        return self.decode.poll(room=room, state=receiver, local_endpoint=("decode", 2))

    def test_subscribe_then_ready(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.assertEqual(self.poll(dest), (False, None))
        self.flush()
        self.assertTrue(dest.subscribed)
        self.assertEqual(self.poll(dest), (False, None))
        self.prefill.mark_ready(room=1, state=source)
        self.flush()
        self.assertEqual(self.poll(dest), (True, None))

    def test_ready_before_subscribe(self):
        source = self.prefill.add_source(room=1)
        self.prefill.mark_ready(room=1, state=source)
        self.assertFalse(self.wire)
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        self.assertEqual(self.poll(dest), (True, None))

    def test_subscription_before_source_does_not_own_a_room(self):
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        self.assertEqual(self.prefill._sources, {})
        self.assertFalse(dest.subscribed)
        source = self.prefill.add_source(room=1)
        self.prefill.mark_ready(room=1, state=source)
        with patch(
            "sglang.srt.disaggregation.prefill_complete.time.monotonic",
            return_value=dest.next_subscribe,
        ):
            self.poll(dest)
        self.flush()
        self.assertEqual(self.poll(dest), (True, None))

    def test_missing_source_retry_backs_off_and_stops_after_wait(self):
        clock = "sglang.srt.disaggregation.prefill_complete.time.monotonic"
        with patch(clock, return_value=0.0):
            dest = self.receiver()
        now = 0.0
        intervals = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.1, 0.1)
        for interval in intervals:
            with patch(clock, return_value=now):
                self.poll(dest)
            self.assertEqual(len(self.wire), 1)
            self.flush()
            self.assertAlmostEqual(dest.next_subscribe - now, interval)
            with patch(clock, return_value=now + interval / 2):
                self.poll(dest)
            self.assertFalse(self.wire)
            now = dest.next_subscribe
        self.prefill.add_source(room=1)
        with patch(clock, return_value=now):
            self.poll(dest)
        self.flush()
        self.assertTrue(dest.subscribed)
        with patch(clock, return_value=now + 1):
            self.poll(dest)
        self.assertFalse(self.wire)

    def test_duplicate_messages_are_idempotent(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        endpoint, subscribe = self.wire[0]
        self.wire.append((endpoint, subscribe))
        self.flush()
        for _ in range(3):
            self.prefill.mark_ready(room=1, state=source)
        self.flush()
        self.assertEqual(self.poll(dest), (True, None))
        self.assertEqual(len(self.prefill._sources), 1)

    def test_failure_wins_over_late_ready(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        self.prefill.mark_ready(room=1, state=source)
        self.prefill.fail_source(room=1, reason="prefill failed")
        # Deliver ERROR before the already-enqueued READY.
        self.wire.reverse()
        self.flush()
        self.assertEqual(self.poll(dest), (False, "prefill failed"))

    def test_failure_before_subscription(self):
        self.prefill.add_source(room=1)
        self.prefill.fail_source(room=1, reason="prefill failed")
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        self.assertEqual(self.poll(dest), (False, "prefill failed"))

    def test_deadline_even_after_subscription(self):
        self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        with patch(
            "sglang.srt.disaggregation.prefill_complete.time.monotonic",
            return_value=dest.deadline,
        ):
            ready, failure = self.poll(dest)
        self.assertFalse(ready)
        self.assertIn("Timed out", failure)

    def test_readiness_arriving_after_deadline_does_not_admit(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        self.prefill.mark_ready(room=1, state=source)
        self.flush()
        with patch(
            "sglang.srt.disaggregation.prefill_complete.time.monotonic",
            return_value=dest.deadline,
        ):
            ready, failure = self.poll(dest)
        self.assertFalse(ready)
        self.assertIn("Timed out", failure)

    def test_late_reply_cannot_ready_reused_room(self):
        source = self.prefill.add_source(room=1)
        old = self.receiver()
        self.poll(old)
        self.flush()
        self.prefill.mark_ready(room=1, state=source)
        self.decode.remove_destination(room=1, state=old)
        current = self.receiver()
        self.flush()
        self.assertFalse(current.ready)
        self.assertNotEqual(old.nonce, current.nonce)
        self.decode.remove_destination(room=1, state=old)
        self.assertIs(self.decode._destinations[1], current)

    def test_old_source_cannot_change_reused_room(self):
        old = self.prefill.add_source(room=1)
        self.prefill.remove_source(room=1, state=old)
        current = self.prefill.add_source(room=1)
        self.prefill.mark_ready(room=1, state=old)
        self.prefill.remove_source(room=1, state=old)
        self.assertFalse(current.ready)
        self.assertIs(self.prefill._sources[1], current)

    def test_late_subscriptions_do_not_recreate_cleared_source(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        subscribe = self.wire[0][1]
        self.prefill.remove_source(room=1, state=source)
        for _ in range(100):
            self.prefill.handle_message(subscribe)
        self.flush()
        self.assertEqual(self.prefill._sources, {})
        self.assertFalse(dest.ready)

    def test_second_destination_fails_without_replacing_owner(self):
        source = self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        subscribe = list(self.wire[0][1])
        self.flush()
        subscribe[3] = b"another-request"
        self.prefill.handle_message(subscribe)
        _, response = self.wire.popleft()
        self.assertEqual(response[1], b"ERROR")
        self.assertEqual(source.subscriber[1], dest.nonce)

    def test_cancel_requires_matching_live_subscription(self):
        self.prefill.add_source(room=1)
        dest = self.receiver()
        self.poll(dest)
        self.flush()
        cancel = [HEADER, b"CANCEL", b"1", b"old-nonce", b"", b""]
        self.prefill.handle_message(cancel)
        self.assertEqual(self.cancelled, [])
        cancel[3] = dest.nonce
        self.prefill.handle_message(cancel)
        self.flush()
        self.assertEqual(self.cancelled, [1])
        self.assertEqual(self.poll(dest), (False, "cancelled"))

    def test_malformed_and_unrelated_messages(self):
        for msg in ([HEADER], [HEADER, b"SUBSCRIBE", b"bad", b"n", b"ip", b"1"]):
            self.assertTrue(self.prefill.handle_message(msg))
        self.assertFalse(self.prefill.handle_message([b"ABORT", b"1"]))
        self.assertEqual(self.prefill._sources, {})

    def test_send_failure_is_a_request_failure(self):
        dest = self.receiver()
        with patch.object(self.decode, "_send", side_effect=OSError("disconnected")):
            ready, failure = self.poll(dest)
        self.assertFalse(ready)
        self.assertIn("disconnected", failure)

    def test_concurrent_subscribe_and_completion(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            for room in range(100):
                source = self.prefill.add_source(room=room)
                dest = self.receiver(room)
                subscribe = [
                    HEADER,
                    b"SUBSCRIBE",
                    str(room).encode(),
                    dest.nonce,
                    b"decode",
                    b"2",
                ]
                done = [
                    pool.submit(self.prefill.handle_message, subscribe),
                    pool.submit(self.prefill.mark_ready, room=room, state=source),
                ]
                for future in done:
                    future.result()
                self.flush()
                self.assertEqual(self.poll(dest, room), (True, None))
                self.prefill.remove_source(room=room, state=source)
                self.decode.remove_destination(room=room, state=dest)
        self.assertEqual(self.prefill._sources, {})
        self.assertEqual(self.decode._destinations, {})


if __name__ == "__main__":
    unittest.main()
