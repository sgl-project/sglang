import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    is_in_ci,
    try_cached_model,
)

register_cuda_ci(est_time=120, suite="stage-c-test-8-gpu-h20")


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCache(PDDisaggregationServerBase):
    extra_decode_args = ["--disaggregation-decode-enable-radix-cache"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.launch_all()

    def _assert_process_healthy(self, name, process, url):
        self.assertIsNotNone(process, f"{name} process was not started")
        self.assertIsNone(
            process.poll(),
            f"{name} exited unexpectedly with code {process.returncode}",
        )
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()

    def test_decode_radix_cache_hits_and_workers_stay_alive(self):
        decode_info = requests.get(f"{self.decode_url}/server_info", timeout=10).json()
        self.assertFalse(
            decode_info.get("disable_radix_cache", True),
            "decode server did not enable radix cache",
        )

        result = run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=4,
            num_rounds=3,
            request_length=384,
            output_length=64,
            max_parallel=4,
        )
        self.assertGreater(
            result["overall"]["total_cached_tokens"],
            0,
            "expected decode radix cache to reuse at least some tokens",
        )

        # Give the schedulers a short idle window so any post-request leak/crash
        # paths have a chance to surface before the liveness checks below.
        time.sleep(5)

        self._assert_process_healthy("load balancer", self.process_lb, self.lb_url)
        self._assert_process_healthy("prefill", self.process_prefill, self.prefill_url)
        self._assert_process_healthy("decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
