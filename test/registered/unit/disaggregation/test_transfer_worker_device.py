"""KV-transfer worker threads run on their rank's device, not device 0."""

import threading
import unittest
from unittest.mock import MagicMock, call, patch

from sglang.srt.disaggregation.common.utils import bind_thread_to_device, run_on_device
from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBindThreadToDevice(CustomTestCase):
    def test_sets_the_platform_device(self):
        with (
            patch.object(
                current_platform, "get_device", return_value="dev3"
            ) as get_device,
            patch.object(current_platform, "set_device") as set_device,
        ):
            bind_thread_to_device(3)
        get_device.assert_called_once_with(3)
        set_device.assert_called_once_with("dev3")

    def test_ignores_platforms_without_set_device(self):
        with (
            patch.object(current_platform, "get_device", return_value="dev3"),
            patch.object(
                current_platform, "set_device", side_effect=NotImplementedError
            ),
        ):
            bind_thread_to_device(3)

    def test_run_on_device_binds_the_thread_before_the_worker_runs(self):
        order = MagicMock()
        worker = order.worker
        threads = {}
        order.bind.side_effect = lambda gpu_id: threads.setdefault(
            "bind", threading.get_ident()
        )
        worker.side_effect = lambda *args: threads.setdefault(
            "worker", threading.get_ident()
        )
        with patch(
            "sglang.srt.disaggregation.common.utils.bind_thread_to_device", order.bind
        ):
            t = threading.Thread(target=run_on_device, args=(3, worker, "q", 7))
            t.start()
            t.join()
        self.assertEqual(order.mock_calls, [call.bind(3), call.worker("q", 7)])
        self.assertEqual(threads["bind"], threads["worker"])
        self.assertNotEqual(threads["bind"], threading.get_ident())


if __name__ == "__main__":
    unittest.main()
