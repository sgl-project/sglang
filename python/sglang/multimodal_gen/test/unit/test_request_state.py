# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disagg request state machine."""

import unittest

from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)


class TestRequestState(unittest.TestCase):
    """Test RequestState enum and transitions."""

    def test_all_states_defined(self):
        expected = {
            "PENDING",
            "ENCODER_WAITING",
            "ENCODER_RUNNING",
            "ENCODER_DONE",
            "DENOISING_WAITING",
            "DENOISING_RUNNING",
            "DENOISING_DONE",
            "DECODER_WAITING",
            "DECODER_RUNNING",
            "DONE",
            "FAILED",
            "TIMED_OUT",
        }
        actual = {s.name for s in RequestState}
        self.assertEqual(actual, expected)


class TestRequestTracker(unittest.TestCase):
    """Test RequestTracker lifecycle management."""

    def test_submit_and_get(self):
        tracker = RequestTracker()
        record = tracker.submit("r1")
        self.assertEqual(record.request_id, "r1")
        self.assertEqual(record.state, RequestState.PENDING)
        self.assertFalse(record.is_terminal())

        got = tracker.get("r1")
        self.assertIs(got, record)

    def test_duplicate_submit_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        with self.assertRaises(ValueError):
            tracker.submit("r1")

    def test_full_lifecycle(self):
        tracker = RequestTracker()
        tracker.submit("r1")

        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_RUNNING)
        self.assertEqual(tracker.get("r1").encoder_instance, 0)

        tracker.transition("r1", RequestState.ENCODER_DONE)
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=1)
        tracker.transition("r1", RequestState.DENOISING_DONE)
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=2)
        tracker.transition("r1", RequestState.DONE)

        record = tracker.get("r1")
        self.assertEqual(record.state, RequestState.DONE)
        self.assertTrue(record.is_terminal())
        self.assertEqual(record.encoder_instance, 0)
        self.assertEqual(record.denoiser_instance, 1)
        self.assertEqual(record.decoder_instance, 2)

    def test_invalid_transition_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")

        # Cannot go from PENDING to DENOISING_RUNNING
        with self.assertRaises(ValueError):
            tracker.transition("r1", RequestState.DENOISING_RUNNING)

    def test_fail_from_any_active_state(self):
        for start_state in [
            RequestState.ENCODER_RUNNING,
            RequestState.ENCODER_DONE,
            RequestState.DENOISING_RUNNING,
        ]:
            tracker = RequestTracker()
            tracker.submit("r1")
            # Walk to start_state
            if start_state == RequestState.ENCODER_RUNNING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
            elif start_state == RequestState.ENCODER_DONE:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
            elif start_state == RequestState.DENOISING_RUNNING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_RUNNING)

            tracker.transition("r1", RequestState.FAILED, error="test error")
            record = tracker.get("r1")
            self.assertTrue(record.is_terminal())
            self.assertEqual(record.error, "test error")

    def test_timeout_from_active_state(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)
        tracker.transition("r1", RequestState.TIMED_OUT)
        self.assertTrue(tracker.get("r1").is_terminal())

    def test_timeout_from_terminal_raises(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)
        tracker.transition("r1", RequestState.FAILED)
        with self.assertRaises(ValueError):
            tracker.transition("r1", RequestState.TIMED_OUT)

    def test_unknown_request_raises(self):
        tracker = RequestTracker()
        with self.assertRaises(ValueError):
            tracker.transition("nonexistent", RequestState.ENCODER_RUNNING)

    def test_remove(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        record = tracker.remove("r1")
        self.assertIsNotNone(record)
        self.assertIsNone(tracker.get("r1"))
        self.assertIsNone(tracker.remove("r1"))  # Already removed

    def test_snapshot(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.submit("r2")
        tracker.transition("r1", RequestState.ENCODER_RUNNING)

        snap = tracker.snapshot()
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["active"], 2)
        self.assertIn("pending", snap["by_state"])
        self.assertIn("encoder_running", snap["by_state"])

    def test_elapsed(self):
        tracker = RequestTracker()
        record = tracker.submit("r1")
        self.assertGreater(record.elapsed_s(), 0.0)

    def test_waiting_states_lifecycle(self):
        """Test full lifecycle with WAITING states (capacity-aware dispatch)."""
        tracker = RequestTracker()
        tracker.submit("r1")

        # PENDING → ENCODER_WAITING (queued in TTA)
        tracker.transition("r1", RequestState.ENCODER_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_WAITING)

        # ENCODER_WAITING → ENCODER_RUNNING (slot available)
        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        self.assertEqual(tracker.get("r1").state, RequestState.ENCODER_RUNNING)

        tracker.transition("r1", RequestState.ENCODER_DONE)

        # ENCODER_DONE → DENOISING_WAITING (all denoisers full)
        tracker.transition("r1", RequestState.DENOISING_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.DENOISING_WAITING)

        # DENOISING_WAITING → DENOISING_RUNNING
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=1)

        tracker.transition("r1", RequestState.DENOISING_DONE)

        # DENOISING_DONE → DECODER_WAITING
        tracker.transition("r1", RequestState.DECODER_WAITING)
        self.assertEqual(tracker.get("r1").state, RequestState.DECODER_WAITING)

        # DECODER_WAITING → DECODER_RUNNING
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=0)

        tracker.transition("r1", RequestState.DONE)
        self.assertTrue(tracker.get("r1").is_terminal())

    def test_fail_from_waiting_states(self):
        """WAITING states can transition to FAILED."""
        for waiting_state in [
            RequestState.ENCODER_WAITING,
            RequestState.DENOISING_WAITING,
            RequestState.DECODER_WAITING,
        ]:
            tracker = RequestTracker()
            tracker.submit("r1")

            # Walk to the waiting state
            if waiting_state == RequestState.ENCODER_WAITING:
                tracker.transition("r1", RequestState.ENCODER_WAITING)
            elif waiting_state == RequestState.DENOISING_WAITING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_WAITING)
            elif waiting_state == RequestState.DECODER_WAITING:
                tracker.transition("r1", RequestState.ENCODER_RUNNING)
                tracker.transition("r1", RequestState.ENCODER_DONE)
                tracker.transition("r1", RequestState.DENOISING_RUNNING)
                tracker.transition("r1", RequestState.DENOISING_DONE)
                tracker.transition("r1", RequestState.DECODER_WAITING)

            tracker.transition("r1", RequestState.FAILED, error="timeout")
            self.assertTrue(tracker.get("r1").is_terminal())

    def test_skip_waiting_when_capacity_available(self):
        """When capacity is available, skip WAITING and go directly to RUNNING."""
        tracker = RequestTracker()
        tracker.submit("r1")

        # PENDING → ENCODER_RUNNING directly (skip ENCODER_WAITING)
        tracker.transition("r1", RequestState.ENCODER_RUNNING, encoder_instance=0)
        tracker.transition("r1", RequestState.ENCODER_DONE)

        # ENCODER_DONE → DENOISING_RUNNING directly
        tracker.transition("r1", RequestState.DENOISING_RUNNING, denoiser_instance=0)
        tracker.transition("r1", RequestState.DENOISING_DONE)

        # DENOISING_DONE → DECODER_RUNNING directly
        tracker.transition("r1", RequestState.DECODER_RUNNING, decoder_instance=0)
        tracker.transition("r1", RequestState.DONE)

        self.assertTrue(tracker.get("r1").is_terminal())

    def test_timeout_from_waiting_state(self):
        tracker = RequestTracker()
        tracker.submit("r1")
        tracker.transition("r1", RequestState.ENCODER_WAITING)
        tracker.transition("r1", RequestState.TIMED_OUT)
        self.assertTrue(tracker.get("r1").is_terminal())


if __name__ == "__main__":
    unittest.main()
