"""CPU-only tests for LoRA draining and adapter-slot admission."""

import unittest
from types import SimpleNamespace
from typing import Optional, cast
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.managers.schedule_batch import Req

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NOW = 1000.0
LORA_DRAIN_WAIT_THRESHOLD = 3.0
STARVING_WAIT = LORA_DRAIN_WAIT_THRESHOLD + 1.0


def make_req(
    lora_id: Optional[str],
    max_new_tokens: int = 10,
    waited_secs: float = 0.0,
    finished: bool = False,
) -> Req:
    """A stand-in for `Req` with only the fields the drainer reads."""
    req = SimpleNamespace(
        lora_id=lora_id,
        time_stats=SimpleNamespace(wait_queue_entry_time=NOW - waited_secs),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        output_ids=[],
        finished=lambda: finished,
        is_retracted=False,
    )
    return cast(Req, req)


def starving_req(lora_id: Optional[str]) -> Req:
    return make_req(lora_id, waited_secs=STARVING_WAIT)


class TestLoRADrainer(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch("time.monotonic", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_drainer(self, max_loras_per_batch: int) -> LoRADrainer:
        return LoRADrainer(max_loras_per_batch, LORA_DRAIN_WAIT_THRESHOLD)

    def assert_draining(self, drainer: LoRADrainer, lora_id: str):
        # Running requests in these tests need at most 100 tokens.
        self.assertFalse(drainer.can_schedule(make_req(lora_id, max_new_tokens=1000)))

    def assert_not_draining(self, drainer: LoRADrainer, lora_id: str):
        self.assertTrue(drainer.can_schedule(make_req(lora_id, max_new_tokens=1000)))

    def test_longest_waiting_adapter_drains_the_adapter_closest_to_done(self):
        drainer = self.make_drainer(max_loras_per_batch=2)
        waiting = [make_req("A", waited_secs=STARVING_WAIT + 1), starving_req("D")]
        running = [make_req("B", max_new_tokens=5), make_req("C", max_new_tokens=100)]
        drainer.update_draining_state(waiting, running)

        # 'A' has waited longest, and 'B' will leave the batch first.
        self.assertEqual(drainer.drained_adapter_to_drain["B"].starving_adapter, "A")
        self.assertEqual(drainer.drained_adapter_to_drain["C"].starving_adapter, "D")

    def test_drain_ends_when_the_drained_adapter_leaves_the_batch(self):
        drainer = self.make_drainer(max_loras_per_batch=1)
        waiting = [starving_req("A")]
        drainer.update_draining_state(waiting, [make_req("B")])
        self.assert_draining(drainer, "B")

        drainer.update_draining_state(waiting, [])
        self.assert_not_draining(drainer, "B")

    def test_drained_adapter_only_admits_requests_that_finish_with_it(self):
        drainer = self.make_drainer(max_loras_per_batch=1)
        running = [make_req("B", max_new_tokens=100)]
        drainer.update_draining_state([starving_req("A")], running)

        # 'B' has 100 tokens left.
        self.assertTrue(drainer.can_schedule(make_req("B", max_new_tokens=100)))
        self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=101)))

    def test_drained_adapter_admits_nothing_once_its_anchor_stops(self):
        for reason in ("finished", "retracted"):
            with self.subTest(reason=reason):
                drainer = self.make_drainer(max_loras_per_batch=1)
                waiting = [starving_req("A")]
                anchor = make_req("B", max_new_tokens=100)
                admitted = make_req("B", max_new_tokens=50)
                drainer.update_draining_state(waiting, [anchor, admitted])

                anchor.finished = lambda: reason == "finished"
                anchor.is_retracted = reason == "retracted"
                # Preemption can retract the anchor during the same admission pass.
                self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=1)))

                # A shorter request keeps B running after the anchor leaves.
                if anchor.is_retracted:
                    waiting.append(anchor)
                drainer.update_draining_state(waiting, [admitted])
                self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=1)))

    def test_drain_ends_once_the_starving_adapter_no_longer_needs_a_slot(self):
        """A drain whose starving adapter needs no slot must stop rejecting requests."""
        running_b = make_req("B", max_new_tokens=5)
        running_c = make_req("C", max_new_tokens=100)
        name_to_case = {
            # 'C' left the batch on its own and 'A' took its slot.
            "starving adapter got a slot": (
                [starving_req("A")],
                [make_req("A"), running_b],
            ),
            # 'A' has nothing waiting anymore, e.g. its requests were aborted.
            "starving adapter has nothing waiting": ([], [running_b, running_c]),
        }
        for name, (waiting, running) in name_to_case.items():
            with self.subTest(name):
                drainer = self.make_drainer(max_loras_per_batch=2)
                drainer.update_draining_state(
                    [starving_req("A")], [running_b, running_c]
                )
                self.assert_draining(drainer, "B")

                drainer.update_draining_state(waiting, running)
                self.assert_not_draining(drainer, "B")

    def test_adapter_in_the_batch_does_not_drain_others(self):
        """An adapter in the batch must not drain others for its own waiting requests."""
        drainer = self.make_drainer(max_loras_per_batch=2)
        # If 'A' could drain others, it would drain 'B': 'B' has fewer tokens left.
        running = [make_req("A", max_new_tokens=100), make_req("B")]
        drainer.update_draining_state([starving_req("A")], running)
        self.assert_not_draining(drainer, "B")

    def test_base_model_can_trigger_a_drain(self):
        """Starving base-model requests (lora_id None) must trigger a drain too."""
        drainer = self.make_drainer(max_loras_per_batch=1)
        drainer.update_draining_state([starving_req(None)], [make_req("B")])
        self.assert_draining(drainer, "B")

    def test_pinned_slots_only_drain_an_adapter_that_can_make_room(self):
        """Idle pins still occupy slots; finishing a pin cannot free its slot."""
        for pin_running in (False, True):
            with self.subTest(pin_running=pin_running):
                drainer = self.make_drainer(max_loras_per_batch=2)
                running = [make_req("A", max_new_tokens=100)]
                if pin_running:
                    running.append(make_req("P", max_new_tokens=5))
                drainer.update_draining_state(
                    [starving_req("B")], running, pinned_loras={"P"}
                )
                self.assert_draining(drainer, "A")
                self.assert_not_draining(drainer, "P")

    def test_finished_requests_do_not_hold_a_slot(self):
        """A finished request still in the batch must not keep its adapter's slot."""
        drainer = self.make_drainer(max_loras_per_batch=2)
        # If 'C' still held a slot, 'B' would be drained: it has fewer tokens left.
        running = [make_req("B"), make_req("C", max_new_tokens=100, finished=True)]
        drainer.update_draining_state([starving_req("A")], running)
        self.assert_not_draining(drainer, "B")


if __name__ == "__main__":
    unittest.main()
