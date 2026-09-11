import threading
import time
import unittest
from concurrent.futures import Future

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ImmediateExecutor:
    def __init__(self):
        self.submit_count = 0

    def submit(self, fn, *args):
        self.submit_count += 1
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as e:
            future.set_exception(e)
        return future


class TestMooncakeCustomMemPoolBatch(CustomTestCase):
    def test_wait_for_transfer_rooms_wakes_on_terminal_status(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: KVPoll.Transferring, 8: KVPoll.Transferring}
        manager._request_status_lock = threading.RLock()
        manager._transfer_completion_condition = threading.Condition()

        def complete_rooms():
            time.sleep(0.01)
            manager.update_status(7, KVPoll.Success)
            manager.update_status(8, KVPoll.Success)

        thread = threading.Thread(target=complete_rooms)
        thread.start()
        try:
            self.assertTrue(manager.wait_for_transfer_rooms({7, 8}, 0.2))
        finally:
            thread.join()

    def test_wait_for_transfer_rooms_is_bounded(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: KVPoll.Transferring}
        manager._request_status_lock = threading.RLock()
        manager._transfer_completion_condition = threading.Condition()

        start_time = time.perf_counter()
        self.assertFalse(manager.wait_for_transfer_rooms({7}, 0.01))
        self.assertLess(time.perf_counter() - start_time, 0.1)

    def test_wait_for_transfer_rooms_treats_cleared_room_as_terminal(self):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {}
        manager._transfer_completion_condition = threading.Condition()

        self.assertTrue(manager.wait_for_transfer_rooms({7}, 0.01))


if __name__ == "__main__":
    unittest.main()
