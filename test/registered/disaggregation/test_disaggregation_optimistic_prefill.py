import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.disaggregation.prefill import should_force_retry
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=116, stage="base-b", runner_config="2-gpu-large")


FORCE_RETRY_PROB = 0.1


def rid_that_forces_retry(prefix: str) -> str:
    """Return a rid that the test retry sampler will select."""
    for _ in range(1000):
        rid = f"{prefix}{uuid.uuid4().hex}"
        req = SimpleNamespace(
            rid=rid,
            is_retracted=False,
            time_stats=SimpleNamespace(prefill_retry_count=0),
        )
        if should_force_retry(req):
            return rid
    raise RuntimeError("Failed to sample an optimistic prefill retry rid")


class OptimisticPrefillRetryCounterMixin:
    def _get_retry_counter(self) -> float:
        response = requests.get(f"{self.prefill_url}/metrics")
        response.raise_for_status()
        total = 0.0
        for family in text_string_to_metric_families(response.text):
            if family.name != "sglang:num_prefill_retries":
                continue
            for sample in family.samples:
                if sample.name == "sglang:num_prefill_retries_total":
                    total += sample.value
        return total

    def assert_retry_counter_increases(self, fn):
        before_retries = self._get_retry_counter()
        result = fn()
        after_retries = self._get_retry_counter()
        self.assertGreater(after_retries, before_retries)
        return result


class TestOptimisticPrefill(
    OptimisticPrefillRetryCounterMixin, PDDisaggregationServerBase
):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._force_retry_prob_was_set = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.is_set()
        )
        cls._force_retry_prob_value = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
        )
        envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(FORCE_RETRY_PROB)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-retries",
            "3",
            "--chunked-prefill-size",
            "128",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if getattr(cls, "_force_retry_prob_was_set", False):
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(
                    cls._force_retry_prob_value
                )
            else:
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.clear()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = self.assert_retry_counter_increases(lambda: run_eval(args))
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)
        time.sleep(1)  # trigger memory check

    def test_logprob(self):
        request_id = rid_that_forces_retry("logprob-retry-")
        prompt = f"{request_id}: " + "The capital of France is Paris. " * 900
        j = self.assert_retry_counter_increases(
            lambda: requests.post(
                self.lb_url + "/generate",
                json={
                    "rid": request_id,
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "return_logprob": True,
                    "return_input_logprob": True,
                    "logprob_start_len": 0,
                },
            ).json()
        )
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        self.assertGreater(j["meta_info"]["prompt_tokens"], 512)
        assert len(output_logprobs) == completion_tokens
        assert len(input_logprobs) > 0


class TestOptimisticPrefillFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # enable optimistic prefill retry sampling and disagg failure prob
        cls._force_retry_ctx = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.override(
                FORCE_RETRY_PROB
            )
        )
        cls._force_retry_ctx.__enter__()
        cls._disagg_failure_ctx = envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(
            FORCE_RETRY_PROB
        )
        cls._disagg_failure_ctx.__enter__()

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-retries",
            "3",
            "--chunked-prefill-size",
            "128",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
            "--load-format",
            "dummy",
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if getattr(cls, "_force_retry_ctx", None):
                cls._force_retry_ctx.__exit__(None, None, None)
            if getattr(cls, "_disagg_failure_ctx", None):
                cls._disagg_failure_ctx.__exit__(None, None, None)

    def test_survive_requests(self):
        # send many small requests to ensure the engine survives injected failures
        n = 100
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for i in range(n):
                rid = f"survive-{i}-{uuid.uuid4().hex}"
                futures.append(
                    executor.submit(
                        requests.post,
                        self.lb_url + "/generate",
                        json={
                            "rid": rid,
                            "text": "Hello world",
                            "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                        },
                        timeout=30,
                    )
                )
            for future in as_completed(futures):
                try:
                    _ = future.result()
                except Exception:
                    pass
        time.sleep(1)  # trigger memory check


if __name__ == "__main__":
    unittest.main()
