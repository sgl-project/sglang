import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-2-npu-a3", nightly=True)


class TestNPUEnableMetricsForAllScheduler(TestNPULoggingBase):
    """Testcase: Verify that the functionality of the --enable-metrics-for-all-scheduler parameter

    [Description]
        When the --enable-metrics-for-all-scheduler parameter is configured, all TP rank instances will independently
        record their own monitoring logs; when this parameter is not configured (default state), only the TP rank 0
        instance will record logs, and other TPrank instances will not generate log output, ensuring the log
        recording logic meets expectations.

    [Test Category] Parameter
    [Test Target] --enable-metrics-for-all-scheduler;
    """

    if_enable_metrics_for_all_scheduler = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--tp-size", 2])
        if cls.if_enable_metrics_for_all_scheduler:
            cls.other_args.extend(["--enable-metrics-for-all-scheduler"])
        cls.launch_server()

    def test_enable_metrics_for_all_scheduler(self):
        self.inference(times=2)
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        message_0 = (
            f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
            f',moe_ep_rank="0",pp_rank="0",tp_rank="0"}}'
        )
        message_1 = (
            f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
            f',moe_ep_rank="0",pp_rank="0",tp_rank="1"}}'
        )
        self.assertIn(message_0, response.text)
        if self.if_enable_metrics_for_all_scheduler:
            self.assertIn(message_1, response.text)
        else:
            self.assertNotIn(message_1, response.text)


class TestNPUDisableMetricsForAllScheduler(TestNPUEnableMetricsForAllScheduler):
    if_enable_metrics_for_all_scheduler = False


if __name__ == "__main__":
    unittest.main()
