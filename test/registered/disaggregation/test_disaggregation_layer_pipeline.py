"""E2E smoke test for layer-pipeline disaggregated KV transfer."""

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=240, stage="base-b", runner_config="2-gpu-large")


_LAYER_PIPELINE_ARGS = [
    "--enable-disagg-layer-pipeline",
    "--disagg-layer-group-size",
    "4",
    "--disagg-layer-pipeline-min-prefill-len",
    "0",
]


class TestDisaggregationLayerPipeline(PDDisaggregationServerBase):
    extra_prefill_args = list(_LAYER_PIPELINE_ARGS)
    extra_decode_args = list(_LAYER_PIPELINE_ARGS)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_generate_logprob_shape(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is ",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=60,
        )
        response.raise_for_status()

        j = response.json()
        self.assertTrue(j["text"])

        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        self.assertEqual(
            len(output_logprobs),
            completion_tokens,
        )
        self.assertGreater(len(input_logprobs), 0)
