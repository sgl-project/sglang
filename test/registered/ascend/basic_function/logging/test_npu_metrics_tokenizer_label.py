import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPUMetricsTokenizerLabel(TestNPULoggingBase):
    """Testcase: Verify the functionality of tokenizer-metrics custom label related parameters

    [Description]
        Parameters include: --tokenizer-metrics-custom-labels-header; --tokenizer-metrics-allowed-custom-labels
        Verifies the cooperative effect of the two parameters and their independent functions, ensuring that the transmission
        and verification of custom labels in tokenizer monitoring metrics take effect normally, as follows:
        1.  --tokenizer-metrics-custom-labels-header: Used to set the name of the HTTP request header, which is used
            by the client to pass custom labels to the server for tokenizer monitoring metrics statistics;
        2.  --tokenizer-metrics-allowed-custom-labels: Used to set the list of custom labels allowed to be received by
            the server; only the labels in this list will be recorded and counted by the tokenizer monitoring metrics,
            and labels not in the list will be filtered out.
        Overall, verify that after parameter configuration, the client can pass custom labels through the specified request
        header, the server only receives and includes the allowed labels in monitoring statistics, filters illegal labels.
        And verify monitoring grouped by custom tags.

    [Test Category] Parameter
    [Test Target] --tokenizer-metrics-custom-labels-header; --tokenizer-metrics-allowed-custom-labels;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics"])
        # HTTP header name for custom metrics labels (--tokenizer-metrics-custom-labels-header)
        cls.labels_header = "X-Metrics-Labels"
        # Allowed custom label name (--tokenizer-metrics-allowed-custom-labels)
        cls.my_label = "business_line"
        cls.other_args.extend(
            ["--tokenizer-metrics-custom-labels-header", cls.labels_header]
        )
        cls.other_args.extend(
            ["--tokenizer-metrics-allowed-custom-labels", cls.my_label]
        )
        cls.launch_server()

    def test_log_metrics_tokenizer_label(self):
        """Validate independent statistical aggregation of requests with custom labels via tokenizer metrics label parameters."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "Content-Type": "application/json",
                self.labels_header: f"{self.my_label}=customer_service",
                "text": f"just return me a long string, generate as much as possible.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1000,
                },
            },
        )
        self.assertEqual(response.status_code, 200)

        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        self.assertEqual(response.status_code, 200)
        metrics_content = response.text
        message = f"sglang:time_to_first_token_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)
        message = f"sglang:inter_token_latency_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)
        message = f"sglang:e2e_request_latency_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)


if __name__ == "__main__":
    unittest.main()
