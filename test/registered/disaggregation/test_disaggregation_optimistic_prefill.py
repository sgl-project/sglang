import threading
import time
import unittest
import uuid
from types import SimpleNamespace

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")

# Qwen and nixl are for testing only and should not be committed
MODEL_NAME_FOR_TEST = "Qwen/Qwen2.5-14B-Instruct"


class NixlTransferBackendMixin(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]


PDDisaggregationServerBase = NixlTransferBackendMixin

SINGLE_REQUEST_DECODE_BLOCKER_COUNT = 16
SINGLE_REQUEST_BLOCKER_WARMUP_SECONDS = 0.2
SINGLE_REQUEST_BLOCKER_PROMPT_REPEATS = 800
SINGLE_REQUEST_BLOCKER_MAX_NEW_TOKENS = 512
DECODE_PRESSURE_ARGS = [
    "--max-total-tokens",
    "65536",
    "--num-reserved-decode-tokens",
    "1024",
    "--disable-radix-cache",
]


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

    def _start_decode_blockers(
        self, count: int = 0
    ) -> tuple[list[threading.Thread], list, list]:
        blocker_responses = []
        blocker_errors = []
        threads = []

        def _run_blocker():
            try:
                blocker_id = f"decode-pool-blocker-{uuid.uuid4().hex}"
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "rid": blocker_id,
                        "text": (
                            f"{blocker_id}: "
                            + "Count upward forever. "
                            * SINGLE_REQUEST_BLOCKER_PROMPT_REPEATS
                        ),
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": SINGLE_REQUEST_BLOCKER_MAX_NEW_TOKENS,
                        },
                    },
                    timeout=240,
                )
                blocker_responses.append(response)
            except BaseException as exc:
                blocker_errors.append(exc)

        for _ in range(count):
            thread = threading.Thread(target=_run_blocker)
            thread.start()
            threads.append(thread)

        return threads, blocker_responses, blocker_errors

    def assert_retry_counter_increases(
        self, fn, blocker_count: int = 0, blocker_warmup_seconds: float = 0.0
    ):
        before_retries = self._get_retry_counter()
        blocker_threads, blocker_responses, blocker_errors = (
            self._start_decode_blockers(count=blocker_count)
        )
        if blocker_count > 0 and blocker_warmup_seconds > 0:
            time.sleep(blocker_warmup_seconds)

        result = None
        error = None
        try:
            result = fn()
        except BaseException as exc:
            error = exc
        finally:
            for thread in blocker_threads:
                thread.join(timeout=240)

        for thread in blocker_threads:
            self.assertFalse(thread.is_alive())
        for response in blocker_responses:
            response.raise_for_status()
        if error is not None:
            raise error
        if blocker_errors:
            raise blocker_errors[0]

        after_retries = self._get_retry_counter()
        self.assertGreater(after_retries, before_retries)
        return result


class TestOptimisticPrefill(
    OptimisticPrefillRetryCounterMixin, PDDisaggregationServerBase
):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-retries",
            "3",
            "--chunked-prefill-size",
            "128",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        cls.extra_decode_args = DECODE_PRESSURE_ARGS
        cls.launch_all()

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

    def test_logprob(self):
        prompt = (
            f"logprob-retry-{uuid.uuid4().hex}: "
            + "The capital of France is Paris. " * 900
        )
        j = self.assert_retry_counter_increases(
            lambda: requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "return_logprob": True,
                    "return_input_logprob": True,
                    "logprob_start_len": 0,
                },
            ).json(),
            blocker_count=SINGLE_REQUEST_DECODE_BLOCKER_COUNT,
            blocker_warmup_seconds=SINGLE_REQUEST_BLOCKER_WARMUP_SECONDS,
        )
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        assert len(output_logprobs) == completion_tokens
        assert len(input_logprobs) > 0

    def test_long_prompt_chunked(self):
        request_id = f"retry-probe-{uuid.uuid4().hex}"
        unique_prefix = f"{request_id}: "
        long_prompt = (
            unique_prefix
            + "Tell me a very detailed story about "
            + "the history of optimistic prefill scheduling " * 320
        )

        j = self.assert_retry_counter_increases(
            lambda: requests.post(
                self.lb_url + "/generate",
                json={
                    "rid": request_id,
                    "text": long_prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            ).json(),
            blocker_count=SINGLE_REQUEST_DECODE_BLOCKER_COUNT,
            blocker_warmup_seconds=SINGLE_REQUEST_BLOCKER_WARMUP_SECONDS,
        )
        self.assertIn("text", j)
        self.assertGreater(len(j["text"]), 0)
        self.assertGreater(j["meta_info"]["prompt_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
