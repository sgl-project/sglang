import unittest
from types import SimpleNamespace
from typing import Optional, cast
from unittest import mock

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
)

register_cuda_ci(est_time=59, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=100, suite="extra-a-test-1-gpu-small-amd")

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
    )
    return cast(Req, req)


def starving_req(lora_id: Optional[str]) -> Req:
    return make_req(lora_id, waited_secs=STARVING_WAIT)


class TestLoRADrainer(unittest.TestCase):
    def setUp(self):
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

    def test_drained_adapter_admits_nothing_once_its_anchor_finishes(self):
        """A drained adapter must stop admitting once its anchor stops early, at EOS."""
        drainer = self.make_drainer(max_loras_per_batch=1)
        waiting = [starving_req("A")]
        anchor = make_req("B", max_new_tokens=100)
        drainer.update_draining_state(waiting, [anchor])

        # 'B' stays in the batch for a request admitted during the drain.
        anchor.finished = lambda: True
        admitted = make_req("B", max_new_tokens=50)
        drainer.update_draining_state(waiting, [anchor, admitted])
        self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=1)))

    def test_drained_adapter_admits_nothing_once_its_anchor_is_retracted(self):
        drainer = self.make_drainer(max_loras_per_batch=1)
        waiting = [starving_req("A")]
        anchor = make_req("B", max_new_tokens=100)
        anchor.output_ids = [0]
        running = make_req("B", max_new_tokens=100)
        running.output_ids = [0] * 98
        drainer.update_draining_state(waiting, [anchor, running])
        self.assertTrue(drainer.can_schedule(make_req("B", max_new_tokens=3)))

        # Retraction requeues the unfinished anchor; its token budget stops shrinking.
        waiting.append(anchor)
        drainer.update_draining_state(waiting, [running])
        self.assertFalse(anchor.finished())
        self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=3)))
        self.assertFalse(drainer.can_schedule(make_req("B", max_new_tokens=1)))
        self.assertFalse(drainer.can_schedule(anchor))
        self.assertEqual(drainer.drained_adapter_to_drain["B"].starving_adapter, "A")

        drainer.update_draining_state(waiting, [])
        self.assert_not_draining(drainer, "B")
        self.assertTrue(drainer.can_schedule(waiting[0]))

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

    def test_finished_requests_do_not_hold_a_slot(self):
        """A finished request still in the batch must not keep its adapter's slot."""
        drainer = self.make_drainer(max_loras_per_batch=2)
        # If 'C' still held a slot, 'B' would be drained: it has fewer tokens left.
        running = [make_req("B"), make_req("C", max_new_tokens=100, finished=True)]
        drainer.update_draining_state([starving_req("A")], running)
        self.assert_not_draining(drainer, "B")

    def test_batch_splitting_with_drainer(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=LORA_DRAIN_WAIT_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
