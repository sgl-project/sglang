import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from sglang.srt.lora.lora_drainer import MAX_WAIT_TIME_SECS, LoRADrainer
from sglang.srt.managers.schedule_batch import Req

MOCK_START_TIME = 1000.0


def make_req(lora_id, wait_queue_entry_time, max_new_tokens, output_len=0):
    time_stats = SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time)
    sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
    req_ns = SimpleNamespace(
        lora_id=lora_id,
        time_stats=time_stats,
        sampling_params=sampling_params,
        output_ids=[0] * output_len,
    )
    return cast(Req, req_ns)


class TestLoRADrainer(unittest.TestCase):
    def test_update_draining_marks_adapter(self):
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(max_loras_per_batch=1)

            # Waiting request for adapter 'A' that has been waiting longer than threshold
            wait_entry = MOCK_START_TIME - (MAX_WAIT_TIME_SECS + 0.01)
            waiting_req = make_req("A", wait_entry, max_new_tokens=10)

            running_req = make_req("B", wait_entry, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[waiting_req],
                running_reqs=[running_req],
            )

            # Running adapter 'B' should be marked as draining for 'A'
            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")

            # Once running adapter 'B' finishes running, it should no longer be draining
            drainer.update_draining_state(waiting_queue=[waiting_req], running_reqs=[])
            self.assertIsNone(drainer.adapter_to_stats["B"].is_draining_for)

        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(max_loras_per_batch=2)

            # Two starving adapters should cause two running adapters to drain.
            wait_entryA = MOCK_START_TIME - (MAX_WAIT_TIME_SECS + 0.05)
            wait_entryD = MOCK_START_TIME - (MAX_WAIT_TIME_SECS + 0.01)
            starving_a = make_req("A", wait_entryA, max_new_tokens=10)
            starving_d = make_req("D", wait_entryD, max_new_tokens=10)

            # Running adapters B and C with different remaining tokens
            running_b = make_req("B", wait_entryA, max_new_tokens=5, output_len=0)
            running_c = make_req("C", wait_entryA, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[starving_a, starving_d],
                running_reqs=[running_b, running_c],
            )

            # B (smaller remaining tokens) should be drained for the most-starved adapter 'A'
            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")

            # C should be drained for the other starving adapter 'D'
            self.assertEqual(drainer.adapter_to_stats["C"].is_draining_for, "D")

    def test_can_schedule_respects_draining_tolerance(self):
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(max_loras_per_batch=1)

            wait_entry = MOCK_START_TIME - (MAX_WAIT_TIME_SECS + 0.01)
            starving_req = make_req("A", wait_entry, max_new_tokens=10)

            running_b = make_req("B", wait_entry, max_new_tokens=15, output_len=0)
            drainer.update_draining_state(
                waiting_queue=[starving_req],
                running_reqs=[running_b],
            )

            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")

            # max_new_tokens is less than running adapter B's max_new_tokens
            req_ok = make_req(
                lora_id="B", wait_queue_entry_time=0, max_new_tokens=10, output_len=0
            )
            self.assertTrue(drainer.can_schedule(req_ok))

            # max_new_tokens is more than running adapter B's max_new_tokens
            req_bad = make_req(
                lora_id="B", wait_queue_entry_time=0, max_new_tokens=20, output_len=0
            )
            self.assertFalse(drainer.can_schedule(req_bad))


if __name__ == "__main__":
    unittest.main()
