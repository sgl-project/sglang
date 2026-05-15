import unittest

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPUMetricsDefaultBucketBoundary(TestNPULoggingBase):
    """Testcase: Verify the functionality of the metrics-related parameter group.

    [Description]
        The metrics-related parameter group includes: --enable-metrics; --collect-tokens-histogram;
        --bucket-inter-token-latency; --bucket-e2e-request-latency; --bucket-time-to-first-token;
        --prompt-tokens-buckets; --generation-tokens-buckets

        Verifies the cooperative effect of the parameter group and the independent function of each parameter, as follows:
        1.  --enable-metrics: Core switch parameter; when configured, the service enables monitoring function and
            supports obtaining various monitoring metrics through the metrics interface;
        2.  --collect-tokens-histogram: When configured, it enables the statistical function of token count-related
            metrics, including the statistics of prompt tokens and generation tokens;
        3.  --bucket-time-to-first-token, --bucket-inter-token-latency, --bucket-e2e-request-latency:
            Used to customize the statistical bucket boundaries of the corresponding latency-related metrics, which
            correspond to time-to-first-token latency, inter-token latency, and end-to-end request latency respectively;
        4.  --prompt-tokens-buckets, --generation-tokens-buckets:
            Used to customize the bucket boundaries of token count statistical metrics, which correspond to the
            statistical intervals of prompt tokens and generation tokens respectively.

        Overall, verify that after parameter configuration, the monitoring function is normally enabled, the metric
        statistics are accurate, the custom bucket boundaries take effect, and all configured monitoring information
        can be obtained normally through the metrics interface.

    [Test Category] Parameter
    [Test Target] --enable-metrics; --bucket-time-to-first-token; --bucket-inter-token-latency; --bucket-e2e-request-latency;
    --collect-tokens-histogram; --prompt-tokens-buckets; --generation-tokens-buckets;
    """

    @staticmethod
    def _verify_metrics_and_bucket_boundary(
        testcase,
        model,
        url,
        expected_time_to_first_token_bucket=None,
        expected_inter_token_latency_bucket=None,
        expected_e2e_request_latency_bucket=None,
        expected_prompt_tokens_bucket=None,
        expected_generation_tokens_bucket=None,
    ):
        """Validate that metrics buckets align with expected boundaries when --enable-metrics and bucket configuration parameters are set."""
        # Generate a sufficient number of tokens to monitor inter_token_latency_seconds_bucket
        response = requests.post(
            f"{url}/generate",
            json={
                "text": f"just return me a long string, generate as much as possible.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1000},
            },
        )
        testcase.assertEqual(response.status_code, 200)

        response = requests.get(f"{url}/metrics", timeout=10)
        testcase.assertEqual(response.status_code, 200)
        metrics_content = response.text
        if expected_time_to_first_token_bucket:
            for le in expected_time_to_first_token_bucket:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{model}"}}'
                testcase.assertIn(message, metrics_content)
        if expected_inter_token_latency_bucket:
            for le in expected_inter_token_latency_bucket:
                message = f'sglang:inter_token_latency_seconds_bucket{{le="{le}",model_name="{model}"}}'
                testcase.assertIn(message, metrics_content)
        if expected_e2e_request_latency_bucket:
            for le in expected_e2e_request_latency_bucket:
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{model}"}}'
                testcase.assertIn(message, metrics_content)
        if expected_prompt_tokens_bucket:
            for le in expected_prompt_tokens_bucket:
                message = f'sglang:prompt_tokens_histogram_bucket{{le="{le}",model_name="{model}"}}'
                testcase.assertIn(message, metrics_content)
        if expected_generation_tokens_bucket:
            for le in expected_generation_tokens_bucket:
                message = f'sglang:generation_tokens_histogram_bucket{{le="{le}",model_name="{model}"}}'
                testcase.assertIn(message, metrics_content)
        return metrics_content

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics", "--collect-tokens-histogram"])
        cls.set_default_bucket()
        cls.launch_server()

    @classmethod
    def set_default_bucket(cls):
        # Default bucket boundaries for time-to-first-token latency (seconds)
        # Used if --bucket-time-to-first-token is not explicitly configured
        cls.default_time_to_first_token_bucket = [
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
            "10.0",
            "20.0",
            "40.0",
            "60.0",
            "80.0",
            "100.0",
            "200.0",
            "400.0",
        ]
        # Default bucket boundaries for inter-token latency (seconds)
        # Used if --bucket-inter-token-latency is not explicitly configured
        cls.default_inter_token_latency_bucket = [
            "0.002",
            "0.004",
            "0.006",
            "0.008",
            "0.01",
            "0.015",
            "0.02",
            "0.025",
            "0.03",
            "0.035",
            "0.04",
            "0.06",
            "0.08",
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
        ]
        # Default bucket boundaries for end-to-end (E2E) request latency (seconds)
        # Used if --bucket-e2e-request-latency is not explicitly configured
        cls.default_e2e_request_latency_bucket = [
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
            "10.0",
            "20.0",
            "40.0",
            "60.0",
            "80.0",
            "100.0",
            "200.0",
            "400.0",
            "600.0",
            "1200.0",
            "1800.0",
            "2400.0",
        ]
        # Default bucket boundaries for prompt/generation token count histograms
        # Used if --prompt-tokens-buckets / --generation-tokens-bucket are not explicitly configured
        # Note: Prompt and generation token buckets use identical default boundaries
        cls.default_tokens_bucket = [
            "100.0",
            "300.0",
            "500.0",
            "700.0",
            "1000.0",
            "1500.0",
            "2000.0",
            "3000.0",
            "4000.0",
            "5000.0",
            "6000.0",
            "7000.0",
            "8000.0",
            "9000.0",
            "10000.0",
            "12000.0",
            "15000.0",
            "20000.0",
            "22000.0",
            "25000.0",
            "30000.0",
            "35000.0",
            "40000.0",
            "66000.0",
            "99000.0",
            "132000.0",
            "300000.0",
            "600000.0",
            "900000.0",
            "1.1e+06",
        ]

    def test_bucket_boundary(self):
        TestNPUMetricsDefaultBucketBoundary._verify_metrics_and_bucket_boundary(
            self,
            self.model,
            self.base_url,
            expected_time_to_first_token_bucket=self.default_time_to_first_token_bucket,
            expected_inter_token_latency_bucket=self.default_inter_token_latency_bucket,
            expected_e2e_request_latency_bucket=self.default_e2e_request_latency_bucket,
            expected_prompt_tokens_bucket=self.default_tokens_bucket,
            expected_generation_tokens_bucket=self.default_tokens_bucket,
        )


class TestNPUMetricsCustomBucketBoundary(TestNPULoggingBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--collect-tokens-histogram"])
        cls.set_custom_bucket()
        cls.other_args.extend(["--bucket-time-to-first-token", *cls.my_bucket])
        cls.other_args.extend(["--bucket-inter-token-latency", *cls.my_bucket])
        cls.other_args.extend(["--bucket-e2e-request-latency", *cls.my_bucket])
        cls.other_args.extend(
            ["--prompt-tokens-buckets", "custom", *cls.my_tokens_bucket]
        )
        cls.other_args.extend(
            ["--generation-tokens-buckets", "custom", *cls.my_tokens_bucket]
        )
        cls.launch_server()

    @classmethod
    def set_custom_bucket(cls):
        # Custom latency bucket boundaries (for testing non-default configurations)
        cls.my_bucket = ["0.1", "0.5", "1.0", "5.0", "10.0"]
        # Custom token count bucket boundaries (for testing custom configurations)
        cls.my_tokens_bucket = [
            "100.0",
            "1000.0",
            "10000.0",
            "100000.0",
            "300000.0",
            "600000.0",
            "900000.0",
        ]

    def test_bucket_boundary(self):
        TestNPUMetricsDefaultBucketBoundary._verify_metrics_and_bucket_boundary(
            self,
            self.model,
            self.base_url,
            expected_time_to_first_token_bucket=self.my_bucket,
            expected_inter_token_latency_bucket=self.my_bucket,
            expected_e2e_request_latency_bucket=self.my_bucket,
            expected_prompt_tokens_bucket=self.my_tokens_bucket,
            expected_generation_tokens_bucket=self.my_tokens_bucket,
        )


class TestNPUMetricsTSEBucketBoundary(TestNPULoggingBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--collect-tokens-histogram"])
        cls.set_tse_bucket()
        cls.other_args.extend(["--prompt-tokens-buckets", "tse", *cls.my_tse_set])
        cls.other_args.extend(["--generation-tokens-buckets", "tse", *cls.my_tse_set])
        cls.launch_server()

    @classmethod
    def set_tse_bucket(cls):
        # Two-Sided Exponential (TSE) Bucket Strategy Configuration
        # Format: [base_value, exponential_factor, number_of_steps]
        cls.my_tse_set = ["1000", "2", "8"]
        # Precomputed custom bucket boundaries using the TSE strategy
        cls.my_tse_bucket = [
            "984.0",
            "992.0",
            "996.0",
            "998.0",
            "1000.0",
            "1002.0",
            "1004.0",
            "1008.0",
            "1016.0",
        ]

    def test_bucket_boundary(self):
        TestNPUMetricsDefaultBucketBoundary._verify_metrics_and_bucket_boundary(
            self,
            self.model,
            self.base_url,
            expected_prompt_tokens_bucket=self.my_tse_bucket,
            expected_generation_tokens_bucket=self.my_tse_bucket,
        )


if __name__ == "__main__":
    unittest.main()
