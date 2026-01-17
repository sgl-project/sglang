import os
from concurrent.futures import ThreadPoolExecutor

import openai
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

# 
# register_cuda_ci(est_time=120, suite="stage-b-test-large-2-gpu")
register_cuda_ci(est_time=120, suite="stage-a-test-1")

class TestDisaggregationMetadataCorruptionE2E(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--max-running-requests",
            "2",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=os.environ.copy(),
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--max-running-requests",
            "2",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        decode_env = os.environ.copy()
        decode_env["SGLANG_PD_TEST_INJECT_METADATA_CORRUPTION"] = "1"
        decode_env["SGLANG_PD_TEST_INJECT_METADATA_RESTORE_DELAY_MS"] = "20"
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=decode_env,
        )

    def test_first_token_consistency_under_metadata_race(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        token_a = tokenizer.encode(" world", add_special_tokens=False)[0]
        token_b = tokenizer.encode(" friend", add_special_tokens=False)[0]
        expected_a = tokenizer.decode(
            [token_a], clean_up_tokenization_spaces=False
        )
        expected_b = tokenizer.decode(
            [token_b], clean_up_tokenization_spaces=False
        )

        def _run(prompt: str, token_id: int):
            res = client.completions.create(
                model="dummy",
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logit_bias={str(token_id): 100},
            ).model_dump()
            return res["choices"][0]["text"]

        num_requests_per_prompt = 20
        requests = (
            [("Hello", token_a, expected_a)] * num_requests_per_prompt
            + [("Goodbye", token_b, expected_b)] * num_requests_per_prompt
        )

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(_run, prompt, token_id)
                for prompt, token_id, _expected in requests
            ]

        results = [future.result() for future in futures]
        for result, (_prompt, _token_id, expected) in zip(results, requests):
            self.assertEqual(
                result,
                expected,
                f"Unexpected first token: got={result!r}, expected={expected!r}",
            )
